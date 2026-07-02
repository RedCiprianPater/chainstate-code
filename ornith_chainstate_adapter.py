#!/usr/bin/env python3
"""
Ornith-1.0 × CHAINSTATE Integration Adapter
Main bridge between the Ornith coding agent and the CHAINSTATE symbolic blockchain.

Modes:
  LOCAL  — torch+transformers installed → loads ORNITH_MODEL in-process
  API    — no torch, or model load fails → proxies to ORNITH_BASE_URL
           (any OpenAI-compatible endpoint: vLLM, TGI, Together, etc.)

Endpoints:
  POST /v1/generate      code + 65,536-d symbolic embedding
  POST /v1/query         cognitive transaction → CHAINSTATE worker
  POST /v1/consensus     swarm consensus participation
  POST /v1/asm/compile   NWO-ASM IR generation
  POST /v1/embed         symbolic embedding only
  GET  /status           adapter + worker health
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Optional heavy deps — adapter still works API-only without them ──
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ── Configuration ────────────────────────────────────────────────────
ORNITH_MODEL      = os.getenv("ORNITH_MODEL", "deepreinforce-ai/Ornith-1.0-9B")
ORNITH_BASE_URL   = os.getenv("ORNITH_BASE_URL", "http://localhost:8000/v1")
ORNITH_API_KEY    = os.getenv("ORNITH_API_KEY", "EMPTY")
CHAINSTATE_WORKER = os.getenv("CHAINSTATE_WORKER", "https://chainstate-worker.ciprianpater.workers.dev")

# CHAINSTATE canonical constants (Base mainnet 8453)
CHAIN_ID   = 8453
STATE_TOKEN = "0x9533DF992fd4bCAbB8d8462572449fc45F727d8a"
USDC_TOKEN  = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
SPLITTER    = "0x93a7962f75475b7e3Fbb62d3A23194f8833b1BE4"
TREASURY    = "0x2E964e1c0e3Fa2C0dfD484B2E6D2189dfCF20958"

# USE subspace layout — sums to exactly 2^16 = 65,536
SUBSPACES = [
    ("math", 0,     4096),
    ("sci",  4096,  8192),
    ("lang", 12288, 16384),
    ("occ",  28672, 4096),
    ("emo",  32768, 16384),
    ("ctrl", 49152, 16384),
]
USE_DIMS = 65536

app = FastAPI(title="Ornith-CHAINSTATE Adapter", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class SymbolicTransaction:
    """A CHAINSTATE cognitive transaction."""
    query: str
    symbolic_embedding: List[float]
    code_solution: Optional[str] = None
    scaffold: Optional[str] = None
    consensus_depth: int = 3
    swarm_size: int = 20


class OrnithChainstateBridge:
    """Main bridge class."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.client = httpx.AsyncClient(timeout=120.0)
        if _TORCH_OK:
            self._load_model()
        else:
            print("ℹ torch/transformers not installed — API mode only")

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ORNITH_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(
                ORNITH_MODEL, torch_dtype="auto", device_map="auto",
            )
            print(f"✓ Loaded Ornith model locally: {ORNITH_MODEL}")
        except Exception as e:
            print(f"⚠ Local model load failed (falling back to API mode): {e}")
            self.model = None

    async def generate_code(self, prompt: str, temperature: float = 0.6) -> Dict[str, str]:
        """Generate code using Ornith — local if loaded, else API."""
        if self.model is not None:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=20,
                )
            output_ids = generated[0][inputs.input_ids.shape[1]:]
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            reasoning, answer = self._parse_thinking(content)
            return {"reasoning": reasoning, "code": answer, "mode": "local"}

        # API mode
        response = await self.client.post(
            f"{ORNITH_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {ORNITH_API_KEY}"},
            json={
                "model": "Ornith-1.0",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 2048,
            },
        )
        data = response.json()
        message = data["choices"][0]["message"]
        return {
            "reasoning": message.get("reasoning_content", ""),
            "code": message.get("content", ""),
            "mode": "api",
        }

    @staticmethod
    def _parse_thinking(text: str) -> tuple:
        """Parse <think> blocks from Ornith output."""
        if "</think>" in text:
            reasoning, answer = text.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "").strip()
            answer = answer.strip()
        else:
            reasoning, answer = "", text.strip()
        return reasoning, answer

    @staticmethod
    def generate_symbolic_embedding(text: str) -> List[float]:
        """Generate the 65,536-d Universal Semiotic Embedding.

        Simplified deterministic scorer — in production, swap for the full
        CHAINSTATE embedding model. Subspace activation is driven by actual
        codepoint membership so the dominant_subspace matches worker routing.
        """
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        embedding = np.zeros(USE_DIMS, dtype=np.float32)

        pools = {
            "math": "∫∂∇∆∑∏∀∃∈∉∪∩⊂⊃∞≈≠≤≥√→⇒↔",
            "sci":  "⚛🔬🧬⚗ℏΦΨΩ☢☣",
            "occ":  "☉☽☿♀♂♃♄♅♆♇⚹☤☥☦☪☯✡☮",
            "ctrl": "⇒⇐⇑⇓⇔⇗⇘⇙⇖↻",
        }
        starts = {name: (start, dims) for name, start, dims in SUBSPACES}

        for name, pool in pools.items():
            if any(c in text for c in pool):
                start, dims = starts[name]
                embedding[start:start+dims] = rng.standard_normal(dims) * 0.5 + 0.5

        # lang — any letter content activates a deterministic slice
        if any(c.isalpha() for c in text):
            start, dims = starts["lang"]
            slice_len = min(1024, dims)
            offset = abs(hash(text)) % (dims - slice_len)
            embedding[start+offset:start+offset+slice_len] = rng.standard_normal(slice_len) * 0.4

        # emo — non-ASCII beyond the pools above
        if any(ord(c) > 0x1F000 for c in text):
            start, dims = starts["emo"]
            embedding[start:start+dims] = rng.standard_normal(dims) * 0.3

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    async def submit_to_chainstate(self, tx: SymbolicTransaction) -> Dict:
        """Submit transaction to the CHAINSTATE worker."""
        response = await self.client.post(
            f"{CHAINSTATE_WORKER}/query",
            json={
                "query": tx.query,
                "swarmSize": tx.swarm_size,
                "consensusDepth": tx.consensus_depth,
                "cache": True,
            },
        )
        return response.json()

    @staticmethod
    async def compile_to_asm(code: str, target: str = "gpu") -> Dict:
        """Compile code to NWO-ASM Process-Matrix IR."""
        asm_ir = f"""; NWO-ASM IR — generated by Ornith-CHAINSTATE adapter
; Target: {target}
; Timestamp: {time.time():.0f}

.PROCESS matrix_compute
  LOAD code_segment
  TRANSFORM symbolic_embedding
  TARGET {target}
  EXECUTE substrate
  CONSENSUS k=20 depth=3
  RECEIPT emit
  STORE result_segment
.END

.CODE
{code}
.END
"""
        return {
            "ir": asm_ir,
            "target": target,
            "estimated_cycles": max(1, len(code.split("\n"))) * 100,
        }


bridge = OrnithChainstateBridge()

# ── API Endpoints ────────────────────────────────────────────────────

@app.post("/v1/generate")
async def generate_endpoint(request: Dict):
    """Generate code + symbolic embedding."""
    prompt = request.get("prompt", "")
    temperature = float(request.get("temperature", 0.6))
    result = await bridge.generate_code(prompt, temperature)
    embedding = bridge.generate_symbolic_embedding(result["code"])
    return {
        "reasoning": result["reasoning"],
        "code": result["code"],
        "symbolic_embedding": embedding,
        "model": ORNITH_MODEL,
        "mode": result.get("mode", "api"),
    }


@app.post("/v1/embed")
async def embed_endpoint(request: Dict):
    """Symbolic embedding only — no code generation."""
    text = request.get("text", "")
    return {
        "symbolic_embedding": bridge.generate_symbolic_embedding(text),
        "dims": USE_DIMS,
        "subspaces": [{"name": n, "start": s, "dims": d} for n, s, d in SUBSPACES],
    }


@app.post("/v1/query")
async def query_endpoint(request: Dict):
    """Query CHAINSTATE with Ornith-enhanced context."""
    query = request.get("query", "")
    swarm_size = int(request.get("swarmSize", 20))
    consensus_depth = int(request.get("consensusDepth", 3))

    result = await bridge.generate_code(query)
    tx = SymbolicTransaction(
        query=query,
        symbolic_embedding=bridge.generate_symbolic_embedding(result["code"]),
        code_solution=result["code"],
        scaffold=result["reasoning"],
        consensus_depth=consensus_depth,
        swarm_size=swarm_size,
    )
    chainstate_result = await bridge.submit_to_chainstate(tx)
    return {"ornith_result": result, "chainstate_result": chainstate_result}


@app.post("/v1/consensus")
async def consensus_endpoint(request: Dict):
    """Submit an Ornith state to swarm consensus."""
    state = request.get("state", [])
    reputation = float(request.get("reputation", 1.0))
    response = await bridge.client.post(
        f"{CHAINSTATE_WORKER}/consensus",
        json={"state": state, "reputation": reputation},
    )
    return response.json()


@app.post("/v1/asm/compile")
async def asm_compile_endpoint(request: Dict):
    """Compile code to NWO-ASM IR."""
    code = request.get("code", "")
    target = request.get("target", "gpu")
    return await bridge.compile_to_asm(code, target)


@app.get("/status")
async def status_endpoint():
    """Adapter + upstream worker health."""
    worker = None
    try:
        r = await bridge.client.get(f"{CHAINSTATE_WORKER}/status", timeout=10.0)
        worker = r.json()
    except Exception as e:
        worker = {"error": str(e)[:120]}
    return {
        "status": "healthy",
        "adapter_version": "1.1.0",
        "ornith_model": ORNITH_MODEL,
        "mode": "local" if bridge.model is not None else "api",
        "torch_available": _TORCH_OK,
        "chainstate_worker": CHAINSTATE_WORKER,
        "chain_id": CHAIN_ID,
        "contracts": {
            "state": STATE_TOKEN,
            "usdc": USDC_TOKEN,
            "splitter": SPLITTER,
            "treasury": TREASURY,
        },
        "worker_status": worker,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
