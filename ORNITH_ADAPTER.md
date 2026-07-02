# Ornith-1.0 × CHAINSTATE Integration Adapter

FastAPI bridge that connects the Ornith-1.0 coding agent to the CHAINSTATE symbolic-weight blockchain.

**Adapter version:** 1.1.0
**Frontend:** [CHAINSTATE CODE Space](https://cpater-ornith-chainstate.static.hf.space)
**Chain:** Base mainnet 8453

## Overview

**Ornith-1.0** — self-improving RL coding agent (9B / 35B / 397B MoE) that generates both solutions and the scaffolds that train them. Repo: [deepreinforce-ai/Ornith-1](https://github.com/deepreinforce-ai/Ornith-1).

**CHAINSTATE** — symbolic-weight blockchain where transactions ARE cognitive queries. Weights are universal symbols in a 65,536-dimensional space across six structured subspaces (math / sci / lang / occ / emo / ctrl).

**This adapter** — makes every Ornith generation a symbolic transaction: embedded to 65,536-d, submitted to reputation-weighted swarm consensus at the CHAINSTATE worker, compiled to NWO-ASM Process-Matrix IR for substrate dispatch.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Ornith-1.0    │────▶│  CHAINSTATE      │────▶│  NWO-ASM        │
│  Coding Agent   │     │  Symbolic Chain  │     │  Process-Matrix │
│  (9B/35B/397B)  │◀────│  (PoCW Consensus)│◀────│  IR Dispatch    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  Local / API    │     │  65,536-d USE    │
│  (OpenAI-compat)│     │  (6 Subspaces)   │
└─────────────────┘     └──────────────────┘
```

## Modes

The adapter auto-selects at boot:

| Mode | When | Behavior |
|------|------|----------|
| **LOCAL** | `torch` + `transformers` installed AND `ORNITH_MODEL` loadable | Loads the model in-process; runs inference locally |
| **API** | No torch, or local load fails | Proxies to `ORNITH_BASE_URL` (any OpenAI-compatible endpoint: vLLM, TGI, Together, Fireworks) |

Both modes expose identical endpoints. `GET /status` reports which mode is active.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements-ornith.txt
```

For API-only mode (no local inference), you can skip torch:

```bash
pip install fastapi uvicorn httpx pydantic numpy
```

### 2. Configure environment

```bash
export ORNITH_MODEL="deepreinforce-ai/Ornith-1.0-35B"      # required for LOCAL mode
export ORNITH_BASE_URL="http://localhost:8000/v1"           # used in API mode
export ORNITH_API_KEY="EMPTY"                                # bearer token if needed
export CHAINSTATE_WORKER="https://chainstate-worker.ciprianpater.workers.dev"
```

### 3. Run the adapter

```bash
python ornith_chainstate_adapter.py
# or with reload:
uvicorn ornith_chainstate_adapter:app --reload --port 8080
```

### 4. Connect from the frontend

Open [CHAINSTATE CODE](https://cpater-ornith-chainstate.static.hf.space) → Terminal → type:

```
adapter https://your-adapter-host.onrender.com
```

Every SIM label flips to LIVE.

## File layout

| Path | Purpose |
|------|---------|
| `ornith_chainstate_adapter.py` | FastAPI adapter (this bridge) |
| `requirements-ornith.txt` | Python dependencies |
| `package.json` | npm scripts (dev convenience) |
| `examples/cognitive_transaction.py` | Example: code generation → symbolic transaction → receipt |
| `examples/swarm_inference.py` | Example: multiple Ornith nodes → reputation-weighted consensus |
| `ORNITH_ADAPTER.md` | This file |
| `.github/workflows/deploy.yml` | HF Space auto-deploy on push |

## API endpoints

### `POST /v1/generate` — code + symbolic embedding

Ornith generates code; adapter computes the 65,536-d embedding of the result.

**Request:**
```json
{
  "prompt": "Write a Python function for SHA3-256 hashing",
  "temperature": 0.6
}
```

**Response:**
```json
{
  "reasoning": "…extracted from <think> block…",
  "code": "import hashlib\n…",
  "symbolic_embedding": [/* 65,536 floats */],
  "model": "deepreinforce-ai/Ornith-1.0-35B",
  "mode": "local"
}
```

### `POST /v1/embed` — symbolic embedding only

No code generation; just the 65,536-d vector plus the subspace layout.

**Request:**
```json
{ "text": "∫∂x → ?" }
```

**Response:**
```json
{
  "symbolic_embedding": [/* 65,536 floats */],
  "dims": 65536,
  "subspaces": [
    {"name": "math", "start": 0,     "dims": 4096},
    {"name": "sci",  "start": 4096,  "dims": 8192},
    {"name": "lang", "start": 12288, "dims": 16384},
    {"name": "occ",  "start": 28672, "dims": 4096},
    {"name": "emo",  "start": 32768, "dims": 16384},
    {"name": "ctrl", "start": 49152, "dims": 16384}
  ]
}
```

### `POST /v1/query` — cognitive transaction

Full pipeline: Ornith generates code, adapter embeds it, submits to CHAINSTATE.

**Request:**
```json
{
  "query": "Optimize database query for O(n log n) → O(1)",
  "swarmSize": 20,
  "consensusDepth": 3
}
```

**Response:**
```json
{
  "ornith_result": { "code": "…", "reasoning": "…", "mode": "local" },
  "chainstate_result": {
    "qHash": "…",
    "dominant_subspace": "lang",
    "top_symbols": ["…"],
    "confidence": 0.856,
    "participatingNodes": 20,
    "consensusDepth": 3,
    "gasUsed": "0.001930",
    "cache": "MISS",
    "timestamp": "…"
  }
}
```

### `POST /v1/consensus` — swarm consensus participation

Passthrough to the worker's `/consensus` endpoint. Submit a 65,536-d state + reputation to participate in a log-pool round.

```json
{
  "state": [/* 65,536 floats */],
  "reputation": 0.95
}
```

### `POST /v1/asm/compile` — NWO-ASM IR

Compiles the given code to a `.PROCESS` block targeting the requested substrate.

**Request:**
```json
{
  "code": "def solve(): ...",
  "target": "gpu"
}
```

**Response:**
```json
{
  "ir": ".PROCESS matrix_compute\n  LOAD code_segment\n  ...\n.END",
  "target": "gpu",
  "estimated_cycles": 400
}
```

Valid targets: `gpu`, `qpu`, `npu`, `edge`.

### `GET /status` — adapter + upstream worker health

Returns adapter version, mode, model, and pings the upstream CHAINSTATE worker.

```json
{
  "status": "healthy",
  "adapter_version": "1.1.0",
  "ornith_model": "deepreinforce-ai/Ornith-1.0-35B",
  "mode": "local",
  "torch_available": true,
  "chainstate_worker": "https://chainstate-worker.ciprianpater.workers.dev",
  "chain_id": 8453,
  "contracts": {
    "state":    "0x9533DF992fd4bCAbB8d8462572449fc45F727d8a",
    "usdc":     "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "splitter": "0x93a7962f75475b7e3Fbb62d3A23194f8833b1BE4",
    "treasury": "0x2E964e1c0e3Fa2C0dfD484B2E6D2189dfCF20958"
  },
  "worker_status": { /* passthrough of worker /status */ }
}
```

## Integration points

### 1. Symbolic transaction generation

Ornith converts a code-generation request into a CHAINSTATE cognitive transaction. The generated code is embedded into 65,536-d and submitted to the worker for reputation-weighted log-pool consensus.

### 2. Consensus node participation

Multiple Ornith instances (different sizes / reputations) can act as swarm nodes. Each submits a symbolic state; the log-pool converges at cosine ≥ 0.95 in 3–7 rounds. See `examples/swarm_inference.py`.

### 3. NWO-ASM IR generation

Ornith-generated code compiles to Process-Matrix IR (`.PROCESS` blocks) for dispatch to GPU / QPU / NPU / edge substrates. The Builder view in the frontend produces the same IR from drag-and-drop bricks.

## Pricing (settlement in USDC on Base 8453)

| Operation | Cost |
|-----------|------|
| Code generation (Ornith-9B) | $0.00050 |
| Code generation (Ornith-35B) | $0.00190 |
| Symbolic embedding | $0.00012 |
| Consensus participation | $0.00040 |
| NWO-ASM compilation | $0.00040 |

Economic split: 35% swarm node operators · 35% treasury · 30% platform · 15% affiliate kickback where applicable. Splitter contract: `0x93a7962f75475b7e3Fbb62d3A23194f8833b1BE4`.

## Deploying to Render

1. Push this repo to GitHub
2. New Render Web Service → connect the repo
3. **Build command:** `pip install -r requirements-ornith.txt`
4. **Start command:** `uvicorn ornith_chainstate_adapter:app --host 0.0.0.0 --port $PORT`
5. Set env vars: `ORNITH_BASE_URL`, `ORNITH_API_KEY`, `CHAINSTATE_WORKER`, `ORNITH_MODEL`
6. Copy the Render URL, paste into the frontend terminal: `adapter <url>`

For LOCAL mode on Render you'll need a GPU instance and enough disk for the model weights. For API mode, any starter dyno works.

## Honest status labels

| Component | Status | Notes |
|-----------|--------|-------|
| CHAINSTATE Worker (`/query`, `/status`) | **LIVE** | Cloudflare, KV-backed, receipts genuine |
| 65,536-d symbolic embedding | **LIVE** | Deterministic subspace-aware scorer in this adapter |
| PoCW swarm consensus | **HYBRID** | Consensus math + receipt LIVE; participating nodes currently emulated by worker until real operators register via `POST /beacon` |
| Ornith generation | **HYBRID** | LIVE in local or API mode; SIM template used by frontend when no adapter URL is configured |
| NWO-ASM IR emission | **LIVE** | Deterministic client-side compilation |
| Wallet memory (frontend) | **LIVE** | Client-side localStorage keyed to address |
| On-chain receipt anchoring | **PLANNED** | Every 100 receipts → IPFS merkle root (roadmap) |

## License

MIT — same as CHAINSTATE, Ornith-1.0, and the rest of the NWO ecosystem.
