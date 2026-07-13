#!/usr/bin/env python3
"""
Ornith-1.0 × CHAINSTATE Integration Adapter · v1.2.0

Extends v1.1.0 with ASI-Evolve integration endpoints:
  POST /v1/evolve/start    start an ASI-Evolve run against chainstate_evolve/
  POST /v1/evolve/stop     halt the current run
  GET  /v1/evolve/status   current round, best score, IDLE/RUNNING, PID
  GET  /v1/evolve/history  every trial with score, motivation, lesson
  GET  /v1/evolve/best     the top-scoring NWO-ASM program found so far

Owner-controlled: start/stop require X-Owner-Token header matching
$EVOLVE_OWNER_TOKEN. Read endpoints are public (LIVE dashboard).

Modes (Ornith side, unchanged):
  LOCAL  — torch+transformers installed → loads ORNITH_MODEL in-process
  API    — no torch, or model load fails → proxies to ORNITH_BASE_URL
"""

import os
import json
import time
import signal
import subprocess
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# ── Optional heavy deps — adapter defaults to API-only mode ──────────
# On free-tier hosts (Render 512MB), importing torch alone eats 200-400MB
# and loading a 9B model needs ~18GB. So we ONLY try to import torch when
# the operator explicitly opts in via ORNITH_MODE=local. Default is API mode:
# every code-gen request proxies to ORNITH_BASE_URL (HF Router, vLLM, etc).
_ORNITH_MODE = os.getenv("ORNITH_MODE", "api").strip().lower()
if _ORNITH_MODE == "local":
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _TORCH_OK = True
    except ImportError:
        _TORCH_OK = False
        print("⚠ ORNITH_MODE=local but torch/transformers not installed — falling back to API mode")
else:
    _TORCH_OK = False

# ── Configuration ────────────────────────────────────────────────────
ORNITH_MODEL      = os.getenv("ORNITH_MODEL", "deepreinforce-ai/Ornith-1.0-9B")
ORNITH_BASE_URL   = os.getenv("ORNITH_BASE_URL", "http://localhost:8000/v1")
ORNITH_API_KEY    = os.getenv("ORNITH_API_KEY", "EMPTY")
CHAINSTATE_WORKER = os.getenv("CHAINSTATE_WORKER", "https://chainstate-worker.ciprianpater.workers.dev")

# ASI-Evolve integration
EVOLVE_ROOT         = Path(os.getenv("EVOLVE_ROOT", "./chainstate_evolve")).resolve()
EVOLVE_DB_PATH      = EVOLVE_ROOT / "database" / "experiments.sqlite"
EVOLVE_PIDFILE      = EVOLVE_ROOT / "evolve.pid"
EVOLVE_LOG          = EVOLVE_ROOT / "evolve.log"
EVOLVE_OWNER_TOKEN  = os.getenv("EVOLVE_OWNER_TOKEN", "")  # required for start/stop
EVOLVE_STEPS_MAX    = int(os.getenv("EVOLVE_STEPS_MAX", "50"))  # hard cap per run

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

# Ecosystem owner — stamped in every response
ECOSYSTEM_OWNER = "Ciprian Florin Pater"

app = FastAPI(title="Ornith-CHAINSTATE Adapter", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-Owner-Token"],
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
        if "</think>" in text:
            reasoning, answer = text.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "").strip()
            answer = answer.strip()
        else:
            reasoning, answer = "", text.strip()
        return reasoning, answer

    @staticmethod
    def generate_symbolic_embedding(text: str) -> List[float]:
        """Generate the 65,536-d Universal Semiotic Embedding."""
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

        if any(c.isalpha() for c in text):
            start, dims = starts["lang"]
            slice_len = min(1024, dims)
            offset = abs(hash(text)) % (dims - slice_len)
            embedding[start+offset:start+offset+slice_len] = rng.standard_normal(slice_len) * 0.4

        if any(ord(c) > 0x1F000 for c in text):
            start, dims = starts["emo"]
            embedding[start:start+dims] = rng.standard_normal(dims) * 0.3

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    async def submit_to_chainstate(self, tx: SymbolicTransaction) -> Dict:
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
        asm_ir = f"""; NWO-ASM IR — generated by Ornith-CHAINSTATE adapter
; Target: {target}
; Owner: {ECOSYSTEM_OWNER}
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

# ══════════════════════════════════════════════════════════════════
# ASI-Evolve integration helpers
# ══════════════════════════════════════════════════════════════════

def _require_owner(token: Optional[str]):
    """Verify the caller holds the owner token before mutating evolve state."""
    if not EVOLVE_OWNER_TOKEN:
        raise HTTPException(500, "EVOLVE_OWNER_TOKEN not configured on the server")
    if token != EVOLVE_OWNER_TOKEN:
        raise HTTPException(403, "invalid owner token")


def _evolve_is_running() -> Optional[int]:
    """Return PID if the ASI-Evolve loop is running, else None."""
    if not EVOLVE_PIDFILE.exists():
        return None
    try:
        pid = int(EVOLVE_PIDFILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        try:
            EVOLVE_PIDFILE.unlink()
        except FileNotFoundError:
            pass
        return None


def _read_evolve_db(query: str, params: tuple = ()) -> List[Dict]:
    """Read from ASI-Evolve's SQLite experiment database (read-only)."""
    if not EVOLVE_DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{EVOLVE_DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]
        conn.close()
        return rows
    except sqlite3.Error as e:
        return [{"error": str(e)[:200]}]


# ══════════════════════════════════════════════════════════════════
# Existing endpoints (v1.1.0) — unchanged
# ══════════════════════════════════════════════════════════════════

@app.post("/v1/generate")
async def generate_endpoint(request: Dict):
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
    text = request.get("text", "")
    return {
        "symbolic_embedding": bridge.generate_symbolic_embedding(text),
        "dims": USE_DIMS,
        "subspaces": [{"name": n, "start": s, "dims": d} for n, s, d in SUBSPACES],
    }


@app.post("/v1/query")
async def query_endpoint(request: Dict):
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
    state = request.get("state", [])
    reputation = float(request.get("reputation", 1.0))
    response = await bridge.client.post(
        f"{CHAINSTATE_WORKER}/consensus",
        json={"state": state, "reputation": reputation},
    )
    return response.json()


@app.post("/v1/asm/compile")
async def asm_compile_endpoint(request: Dict):
    code = request.get("code", "")
    target = request.get("target", "gpu")
    return await bridge.compile_to_asm(code, target)


# ══════════════════════════════════════════════════════════════════
# NEW · v1.2.0 · ASI-Evolve endpoints
# ══════════════════════════════════════════════════════════════════

@app.post("/v1/evolve/start")
async def evolve_start(
    request: Dict,
    x_owner_token: Optional[str] = Header(None),
):
    """Start an ASI-Evolve run against chainstate_evolve/.

    Owner-authenticated. Steps hard-capped at EVOLVE_STEPS_MAX.
    Returns immediately with the PID; use /v1/evolve/status to poll.
    """
    _require_owner(x_owner_token)

    existing = _evolve_is_running()
    if existing:
        raise HTTPException(409, f"evolve loop already running (pid {existing})")

    requested = int(request.get("steps", 20))
    steps = max(1, min(requested, EVOLVE_STEPS_MAX))
    sample_n = max(1, min(int(request.get("sample_n", 3)), 10))

    if not EVOLVE_ROOT.exists():
        raise HTTPException(500, f"EVOLVE_ROOT does not exist: {EVOLVE_ROOT}")

    eval_script = EVOLVE_ROOT / "eval.sh"
    if not eval_script.exists():
        raise HTTPException(500, f"missing eval.sh at {eval_script}")

    log_fp = open(EVOLVE_LOG, "a")
    log_fp.write(f"\n═══ RUN START · {time.strftime('%Y-%m-%d %H:%M:%S')} · owner: {ECOSYSTEM_OWNER} ═══\n")
    log_fp.flush()

    proc = subprocess.Popen(
        [
            "python", "main.py",
            "--experiment", "chainstate_evolve",
            "--steps", str(steps),
            "--sample-n", str(sample_n),
            "--eval-script", str(eval_script),
        ],
        cwd=os.getenv("ASI_EVOLVE_ROOT", "./ASI-Evolve"),
        stdout=log_fp, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    EVOLVE_PIDFILE.write_text(str(proc.pid))
    return {
        "started": True,
        "pid": proc.pid,
        "steps": steps,
        "sample_n": sample_n,
        "steps_cap": EVOLVE_STEPS_MAX,
        "owner": ECOSYSTEM_OWNER,
        "note": "Hard step cap enforced. Stop anytime via POST /v1/evolve/stop.",
    }


@app.post("/v1/evolve/stop")
async def evolve_stop(x_owner_token: Optional[str] = Header(None)):
    """Halt the current ASI-Evolve run (SIGTERM to the process group)."""
    _require_owner(x_owner_token)
    pid = _evolve_is_running()
    if not pid:
        return {"stopped": False, "reason": "no evolve loop running"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(0.5)
        if _evolve_is_running():
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError) as e:
        return {"stopped": False, "reason": str(e)}
    try:
        EVOLVE_PIDFILE.unlink()
    except FileNotFoundError:
        pass
    return {"stopped": True, "pid": pid, "owner": ECOSYSTEM_OWNER}


@app.get("/v1/evolve/status")
async def evolve_status():
    """Public read-only status. Never mutates state."""
    pid = _evolve_is_running()
    state = "RUNNING" if pid else "IDLE"

    # Round + best-score summary from the experiment database
    rounds = _read_evolve_db(
        "SELECT COUNT(*) AS n, MAX(score) AS best_score FROM nodes"
    )
    latest = _read_evolve_db(
        "SELECT id, score, motivation, lesson, created_at "
        "FROM nodes ORDER BY id DESC LIMIT 1"
    )

    round_count = 0
    best_score = None
    if rounds and "n" in rounds[0]:
        round_count = rounds[0].get("n") or 0
        best_score = rounds[0].get("best_score")

    return {
        "state": state,
        "pid": pid,
        "rounds_completed": round_count,
        "best_score_so_far": best_score,
        "latest_trial": latest[0] if latest else None,
        "steps_cap_per_run": EVOLVE_STEPS_MAX,
        "owner": ECOSYSTEM_OWNER,
        "experiment": "chainstate_evolve",
        "objective": "maximize consensus confidence · minimize gas · minimize rounds-to-converge",
        "evolve_root": str(EVOLVE_ROOT),
        "db_present": EVOLVE_DB_PATH.exists(),
    }


@app.get("/v1/evolve/history")
async def evolve_history(limit: int = 50):
    """Every trial the loop has run, newest first."""
    limit = max(1, min(limit, 200))
    rows = _read_evolve_db(
        "SELECT id, parent_id, score, motivation, lesson, created_at "
        "FROM nodes ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return {
        "count": len(rows),
        "trials": rows,
        "owner": ECOSYSTEM_OWNER,
    }


@app.get("/v1/evolve/best")
async def evolve_best():
    """The top-scoring NWO-ASM program discovered so far."""
    rows = _read_evolve_db(
        "SELECT id, score, code, motivation, lesson, created_at "
        "FROM nodes ORDER BY score DESC NULLS LAST LIMIT 1"
    )
    if not rows:
        return {"present": False, "note": "no trials completed yet"}
    return {"present": True, "best": rows[0], "owner": ECOSYSTEM_OWNER}


# ══════════════════════════════════════════════════════════════════

@app.post("/v1/substrate/gpu")
async def substrate_gpu_endpoint(request: Dict):
    """Real numpy work invoked by the main worker when a query specifies TARGET gpu.

    Takes the query and the dominant subspace, projects both to the USE,
    computes cosine similarity plus a dense matrix eigenvalue metric on the
    projection. Returns real timing and metric values that get attached to the
    consensus receipt as `gpu_metrics`.

    This is what makes TARGET gpu materially different from TARGET edge in the
    receipt: edge queries never call this endpoint, so their gas_used is lower
    but their receipt lacks the `gpu_metrics` block. The ASI-Evolve loop can
    learn to prefer or avoid gpu based on the score trade-off.
    """
    import time
    t0 = time.perf_counter()

    query = str(request.get("query", ""))
    dominant = str(request.get("dominant_subspace", "math"))

    if not query:
        return {"error": "query required", "owner": ECOSYSTEM_OWNER}

    # Real numpy computation: project query to USE, compute self-similarity
    embedding = np.array(bridge.generate_symbolic_embedding(query), dtype=np.float32)

    # Slice out the dominant subspace's block for a dense inner-product study
    subspace_ranges = {n: (s, s + d) for n, s, d in SUBSPACES}
    if dominant not in subspace_ranges:
        dominant = "math"
    start, end = subspace_ranges[dominant]
    block = embedding[start:end]

    # Real work: 512-dim inner-product matrix over a random projection of the block
    # (this is CPU-bound but bounded — completes in ~10-30 ms on Render's 1 CPU)
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    proj = rng.standard_normal((min(512, len(block)), 128)).astype(np.float32)
    projected = block[:proj.shape[0]] @ proj
    gram = projected.reshape(-1, 1) @ projected.reshape(1, -1)

    # Metrics: block norm, projected norm, trace of gram, top eigenvalue estimate
    block_norm = float(np.linalg.norm(block))
    proj_norm = float(np.linalg.norm(projected))
    trace = float(np.trace(gram))
    # Power-iteration estimate of top eigenvalue (5 iterations, well-bounded)
    v = rng.standard_normal(gram.shape[0]).astype(np.float32)
    for _ in range(5):
        v = gram @ v
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
    top_eig = float((gram @ v) @ v)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "substrate": "gpu",
        "dominant_subspace": dominant,
        "block_dim": int(end - start),
        "block_norm": block_norm,
        "projected_norm": proj_norm,
        "gram_trace": trace,
        "top_eigenvalue": top_eig,
        "elapsed_ms": elapsed_ms,
        "adapter_version": "1.3.0",
        "owner": ECOSYSTEM_OWNER,
    }


@app.get("/status")
async def status_endpoint():
    """Adapter + upstream worker health, plus evolve state summary."""
    worker = None
    try:
        r = await bridge.client.get(f"{CHAINSTATE_WORKER}/status", timeout=10.0)
        worker = r.json()
    except Exception as e:
        worker = {"error": str(e)[:120]}

    evolve_pid = _evolve_is_running()

    return {
        "status": "healthy",
        "adapter_version": "1.3.0",
        "ornith_model": ORNITH_MODEL,
        "mode": "local" if bridge.model is not None else "api",
        "torch_available": _TORCH_OK,
        "chainstate_worker": CHAINSTATE_WORKER,
        "chain_id": CHAIN_ID,
        "owner": ECOSYSTEM_OWNER,
        "contracts": {
            "state": STATE_TOKEN,
            "usdc": USDC_TOKEN,
            "splitter": SPLITTER,
            "treasury": TREASURY,
        },
        "worker_status": worker,
        "evolve": {
            "state": "RUNNING" if evolve_pid else "IDLE",
            "pid": evolve_pid,
            "root": str(EVOLVE_ROOT),
            "db_present": EVOLVE_DB_PATH.exists(),
            "steps_cap_per_run": EVOLVE_STEPS_MAX,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
