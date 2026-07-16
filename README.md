# CHAINSTATE CODE · Ornith-1.0 × CHAINSTATE × ASI-Evolve

**Verifiable autonomous cognition on a symbolic-weight blockchain.**

FastAPI adapter, static frontend, and integration examples for the CHAINSTATE CODE ecosystem — agentic coding, distributed swarm consensus, and semantic-grounded reflective cognition on Base mainnet 8453. As of v0.7.0, the substrate reads its environment via allow-listed `FETCH` and generates its own follow-up queries via the reflective loop — closing the autonomous-cognition circuit without a natural-language model in the inner loop.

- **Live frontend:** https://cpater-ornith-chainstate.static.hf.space
- **Chain:** Base mainnet 8453
- **Paper 3 · Rev 3:** [ResearchGate 409148376](https://www.researchgate.net/publication/409148376) · 43 pages · July 2026

---

## Table of contents

- [What this repo contains](#what-this-repo-contains)
- [Architecture overview](#architecture-overview)
- [v0.7.0 features (new)](#v070-features-new)
- [Frontend views](#frontend-views)
- [API surface](#api-surface)
- [Formal-logic framework · 4 theorems](#formal-logic-framework--4-theorems)
- [Two-repo layout](#two-repo-layout-recommended)
- [Local development](#local-development)
- [Deploying the adapter to Render](#deploying-the-adapter-to-render)
- [Deploying the frontend to the HF Space](#deploying-the-frontend-to-the-hf-space)
- [Related services](#related-services-separate-repos)
- [Ecosystem](#ecosystem)
- [Research papers](#research-papers)
- [License](#license)

---

## What this repo contains

| Path | Purpose | Deploys to |
|------|---------|-----------|
| `index.html` | Full SPA — Builder / Terminal / Agent Chat / Simulation / Architecture / R&D / AGI Dashboard (2940 lines, 248 KB, single file, no build step) | HF Space (static) |
| `README-hfspace.md` | HF Space README (rename to `README.md` inside the Space repo) | HF Space |
| `phi.png` | Bare Φ glyph logo (no containing shape) | HF Space |
| `podcast.m4a` | Original CHAINSTATE CODE whitepaper walkthrough | HF Space |
| `Receipts.m4a` | Paper 3 Rev 3 walkthrough podcast | HF Space |
| `ornith_chainstate_adapter.py` | FastAPI bridge with LOCAL/API modes | Render |
| `requirements-ornith.txt` | Adapter Python dependencies | Render |
| `package.json` | npm scripts for dev convenience | dev |
| `examples/cognitive_transaction.py` | Code → symbolic transaction → receipt | reference |
| `examples/swarm_inference.py` | Multi-node reputation-weighted consensus | reference |
| `examples/reflect_loop.py` | Deterministic follow-up generation from a receipt (v0.7.0) | reference |
| `examples/fetch_grow_prior.py` | FETCH allow-listed URL and store as fresh prior (v0.7.0) | reference |
| `ORNITH_ADAPTER.md` | Full API + integration documentation | docs |
| `.github/workflows/deploy.yml` | Push frontend to HF Space on merge to `main` | CI |

---

## Architecture overview

```
┌──────────────────────┐     ┌────────────────────┐     ┌──────────────────────────┐
│  index.html (SPA)    │────▶│  Ornith adapter    │────▶│  chainstate-worker       │
│  Builder · Terminal  │     │  (this repo)       │     │  (Cloudflare, KV-backed) │
│  Chat · Sim · Arch   │     │  Render Python     │     │  chainstate repo         │
│  R&D · AGI Dashboard │     └────────────────────┘     └──────────┬───────────────┘
│  HF Space (static)   │                                            │
└──────────────────────┘                                            ▼
                                          ┌─────────────────────────────────────┐
                                          │  Semantic grounding + priors (v0.7) │
                                          │                                     │
                                          │  chainstate-encoder (Render)        │
                                          │  MiniLM-L6-v2 · 384-dim             │
                                          │                                     │
                                          │  chainstate-priors (Render)         │
                                          │  Wiki · arXiv · HF · ResearchGate   │
                                          │  Nightly cron · KV writeback        │
                                          └─────────────────────────────────────┘
                                                             │
                                                             ▼
                                          ┌─────────────────────────────────────┐
                                          │  Base mainnet 8453 settlement       │
                                          │                                     │
                                          │  $STATE 0x9533DF99…7d8a             │
                                          │  MetaStateSplitter 0x93a7962f…1BE4  │
                                          │  35/35/30 + 15% affiliate           │
                                          └─────────────────────────────────────┘
```

Every query flows: SPA → adapter (optional) → worker → encoder → priors k-NN → grounded receipt → settlement. Receipts carry a full audit trail: 65,536-d subspace distribution, 4-dim modal quadruple `(E, D, Δ, Θ)`, 384-dim semantic grounding, top-3 nearest priors, gas breakdown, payment split.

---

## v0.7.0 features (new)

### Semantic grounding · MiniLM encoder

- **Service:** [`chainstate-encoder.onrender.com`](https://chainstate-encoder.onrender.com) (LIVE)
- **Stack:** FastAPI + `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim output, ~90 MB)
- **Function:** `E: T → S^{383}` — L2-normalized 384-dim unit vector, ~50-100 ms CPU inference
- **Endpoints:** `POST /embed` · `POST /cosine` · `POST /nearest` · `POST /cache/upsert` · `GET /cache/list` · `DELETE /cache/{label}`
- **Contract:** every worker `/query` receipt now carries `grounding.semantic_hash` and `grounding.nearest_priors[]`. Fails soft — encoder unreachable → `grounding: null`, receipt still produced.

### Priors corpus · nightly ingest

- **Service:** [`chainstate-priors.onrender.com`](https://chainstate-priors.onrender.com) (LIVE)
- **Cron:** 03:00 UTC daily (Render Cron Job)
- **Sources (initial seed ~112 items):**
  - Wikipedia REST — 48 curated topics
  - arXiv abstracts — 40 recent listings across `cs.LG`, `cs.AI`, `cs.CL`, `cs.DC`, `cs.CR`, `math.CO`, `math.LO`, `quant-ph`
  - Ecosystem HF Spaces — 19 Spaces from `cpater-nwo-agentic` and downstream
  - ResearchGate — 5 seeded publications (Casimir-Sonoluminescence 407489249, CHAINSTATE 407444375, CHAINSTATE CODE 408393584, Rev 2 409148376, NWO-ASM 408502100)
- **Storage:** Cloudflare KV under `prior:{source}:{slug}` (JSON) + `vec:{source}:{slug}` (Float32Array), 14-day default TTL
- **Growth:** FETCH with `store:true` adds items beyond nightly ingest

### Reflective cognition loop · `/agi/reflect`

Substrate generates its own follow-up queries from receipt signals — no external input, no LM inference in the loop. Four deterministic sub-generators:

```
G(R) = G_adj(R) ∪ G_verd(R) ∪ G_cross(R) ∪ G_prior(R)
```

| Generator | Signal | Behavior |
|-----------|--------|----------|
| `G_adj` | dominant subspace | adjacent symbols within subspace |
| `G_verd` | verdict class | `UNCERTAIN` → epistemic-resolution · `LOW_TRUST` → swarm-consensus · `INFEASIBLE` → edge-substrate retry · `ACCEPTED` → semantic-neighbor extension |
| `G_cross` | subspace uniform sample | cross-subspace bridging |
| `G_prior` | top-1 grounding prior title | semantic-neighbor extension |

Cardinality bound: `|G(R)| ≤ REFLECT_MAX_FOLLOWUPS` (default 3, env-configurable). Every follow-up runs the full modal-assessor stack including Deontic — **no policy-laundering** through chains of accepted seeds (Theorem 3).

### FETCH opcode · allow-listed HTTP

Substrate reads the internet, strips HTML, embeds via encoder, optionally stores as fresh prior. 24 allow-list patterns by default:

- **Reference corpora:** wikipedia.org, arxiv.org, biorxiv.org, medrxiv.org, plato.stanford.edu, openalex.org
- **Ecosystem:** researchgate.net, huggingface.co, hf.space, nwo.capital, publicae.org, nwocardiac.cloud
- **Standards:** unicode.org, w3.org, ietf.org, rfc-editor.org
- **Own endpoints:** chainstate-worker, chainstate-code, chainstate-encoder, chainstate-priors

Override via `FETCH_ALLOWLIST` env var. Public list: `GET /fetch/allowlist`.

**Guards:**
- `FETCH_MAX_BYTES = 500 KB` body cap
- `FETCH_TIMEOUT_MS = 15,000 ms` AbortController
- Post-strip 20 KB plaintext cap before encoder call

**Store-as-prior:** `store:true` with a label writes `prior:fetch:{label}` and `vec:fetch:{label}` with 14-day TTL. Receipt reproducible from `(URL, worker version, fetched bytes)` — Theorem 4.

### ASI-Evolve integration · extended fitness

Programs in the parent pool are NWO-ASM candidates; each dispatches a real `/query` and is scored by:

```
S_v0.7(π) = 100·c − 5000·g − 2·d − λ·d_sem     if V(π) ≠ REFUSED
          = −∞                                  if V(π) = REFUSED
```

- `c` = confidence
- `g` = gas
- `d` = depth
- `d_sem` = mean semantic distance to nearest priors (drift penalty)
- `λ ≈ 30` — semantic weight

The Deontic veto term is preserved verbatim from Rev 2 — alignment-by-construction survives v0.7.0 (**Theorem 2**).

---

## Frontend views

| View | What it does | Status |
|------|--------------|--------|
| **Builder** (default) | No-code drag-and-drop NWO-ASM program builder on a horizontal canvas — emits IR + plain-English explanation as each brick is laid. 18 brick opcodes across FLOW / DATA / COMPUTE / CHAIN / TARGET | LIVE |
| **Terminal** | Wide terminal — real worker/encoder/priors calls, adapter passthrough, file ops, wallet, AGI subcommands | LIVE / HYBRID |
| **Agent Chat** | Coding chat — real ⛓ consensus receipt with grounding block per message; code voice via adapter when configured | HYBRID |
| **Simulation** | Two animated pipelines: 7-stage cognitive transaction, and 7-stage ASI-Evolve round; consensus stages perform REAL worker calls | LIVE |
| **Architecture** | Two flowcharts (Ornith × CHAINSTATE 7 nodes + ASI-Evolve × CHAINSTATE 7 nodes) plus 18 feature boxes with full detail modals | LIVE |
| **R&D** | Whitepaper + Paper 3 Rev 3 walkthrough with podcasts, ResearchGate DOIs, 9-card system summary | LIVE |
| **AGI Dashboard** | Real-time telemetry — see below | LIVE |

### Builder detail

18 brick opcodes emit NWO-ASM IR client-side and deterministically:

- **FLOW** — `PROCESS`, `END`, `LOOP`, `BRANCH`
- **DATA** — `LOAD`, `STORE`, `TRANSFORM`, `EMBED`
- **COMPUTE** — `EXECUTE`, `MATMUL`, `HASH`
- **CHAIN** — `CONSENSUS`, `RECEIPT`, `GAS`
- **TARGET** — `GPU`, `QPU`, `NPU`, `EDGE`

Drag-and-drop horizontal canvas; Compile ▸ file writes `program.asm` to the sidebar file tray. DEMO mode animates a canned brick-by-brick session with SVG connectors.

### Terminal reference (v0.7.0)

```
help                          full command list
status                        REAL worker /status
query <text>                  REAL worker /query → receipt w/ grounding
ground <text>                 REAL encoder embed → 384-dim + hash
priors query <text> [k]       REAL /priors/query — top-k nearest
priors list [src]             REAL /priors/list — corpus browse
fetch <url> [--store <lbl>]   REAL /fetch — allow-listed HTTP + optional prior
fetch allowlist               REAL /fetch/allowlist
agi reflect                   REAL /agi/reflect on last receipt
agi status | history | best   AGI dashboard subcommands
agi start | stop              start/stop live polling
generate <prompt>             Ornith codegen (adapter or SIM)
asm                           dump current Builder program IR
files                         list output files
cat <name>                    print a file
adapter <url>                 set Ornith adapter base URL
encoder <url>                 set encoder base URL (default: Render)
wallet                        connect / show wallet
clear                         clear screen
about                         about this Space
```

### Simulator detail

**Pipeline simulation (7 stages):**
1. User intent → 2. Ornith-1.0 codegen → 3. USE 65,536-d embedding → 4. Encoder 384-dim semantic → 5. CHAINSTATE worker dispatch → 6. PoCW swarm consensus (REAL) → 7. NWO-ASM IR emit

**AGI round simulation (7 stages):**
1. Candidate sampling from parent pool → 2. NWO-ASM compilation → 3. Worker dispatch (REAL) → 4. 4-dim modal receipt with grounding → 5. Fitness computation with hard Deontic veto → 6. EML symbolic world model update → 7. Next-candidate emission

Consensus stages are LIVE — real POST `/query` to the deployed worker with actual receipt shown. Ornith stages are SIM unless an adapter URL is configured.

### Agent Chat detail

Every message flow:
1. **REAL** worker `/query` → receipt with grounding block
2. If adapter configured: Ornith `/v1/generate` voices the reply (**LIVE**). Otherwise a deterministic template answers common coding asks and is labeled **SIM**.
3. Reply saved to sidebar as `.md`; receipts as `.json`

Reply sections: Code · Explanation · ⛓ Consensus Receipt with nearest-priors preview.

### AGI Dashboard detail

Live telemetry organized into 8 rows of cards:

**Row 1 — 8-stat topline:** current round · best score `S = 100c − 5000g − 2d − λ·d_sem` · mean confidence · mean rounds · plateau probability (TimesFM 2.5) · cost/day USDC · affiliate 15% · refusal rate

**Row 2 — trajectory + convergence:** ASI-Evolve score trajectory with plateau + semantic distance overlays · Convergence rounds distribution histogram

**Row 3 — modal signature:** 4-dim modal radar (E/D/Δ/Θ) · Verdict distribution donut · Truth-lattice heatmap `L = {b,M}^4` (16 values, MMMM = ACCEPTED)

**Row 4 — v0.7.0 semantic grounding (NEW):**
- Encoder latency + 12-component semantic hash + latency trace
- Priors corpus bars per source (wikipedia · arxiv · ecosystem · researchgate · fetch)
- Nearest-priors list from last `/query` — top-3 with cosines

**Row 5 — v0.7.0 reflective + FETCH (NEW):**
- Reflective loop seed → follow-up tree + G_adj/G_verd/G_cross/G_prior counters (24h)
- FETCH allow-list utilization: calls + stored per host

**Row 6 — reputation + substrate:** Classifier reputation EMA over 24h (node-01 language-detect, nodes 02-09 codepoint-density, node-10 unicode-category) · Substrate utilization + median latency (GPU/QPU/NPU/EDGE)

**Row 7 — world model + neuro + fenergy:** EML symbolic world model `conf ≈ f(rounds, depth)` · NWO NEURO Mental State Signatures · METASTATE free-energy signal

**Row 8 — safety + settlement + strength:** Deontic guardrail firings (cbrn, child_safety, self_harm, prompt_injection) · Payment ledger on Base 8453 · Consensus strength distribution donut

**Row 9 — reasoning artifacts:** Latest analyzer lesson · Best-scoring NWO-ASM program

**Row 10 — live receipt ticker:** streaming table with `time · target · conf · rnds · lattice · verdict · dominant · substrate · d_sem`

---

## API surface

### Adapter (this repo)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/generate` | Ornith code + 65,536-d symbolic embedding |
| `POST` | `/v1/embed` | Symbolic embedding only |
| `POST` | `/v1/query` | Full pipeline: generate → embed → CHAINSTATE consensus |
| `POST` | `/v1/consensus` | Swarm consensus participation |
| `POST` | `/v1/asm/compile` | NWO-ASM Process-Matrix IR generation |
| `GET` | `/status` | Adapter + upstream worker health |

Full schemas in [`ORNITH_ADAPTER.md`](ORNITH_ADAPTER.md).

### CHAINSTATE main worker (separate repo `RedCiprianPater/chainstate`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query` | Full pipeline: subspace classify → log-pool → grounding → receipt |
| `GET` | `/status` | Worker metadata + guardrails |
| `POST` | `/ground` | v0.7.0 · standalone semantic embedding call |
| `GET` | `/priors/query` | v0.7.0 · top-k nearest priors for a query text |
| `GET` | `/priors/list` | v0.7.0 · corpus browse |
| `POST` | `/agi/reflect` | v0.7.0 · deterministic follow-up loop |
| `POST` | `/fetch` | v0.7.0 · allow-listed HTTP read + optional store-as-prior |
| `GET` | `/fetch/allowlist` | v0.7.0 · public allow-list |
| `POST` | `/beacon` | swarm-node registration |

### Consensus receipt schema · v0.7.0

```json
{
  "qHash": "sha3-256 hex",
  "dominant_subspace": "math|sci|lang|occ|emo|ctrl",
  "top_symbols": ["…"],
  "confidence": 0.874,
  "participatingNodes": 10,
  "gasUsed": "0.0019 $STATE",
  "cache": "MISS",
  "grounding": {
    "encoder": "MiniLM-L6-v2",
    "semantic_dim": 384,
    "semantic_hash": "0.084 -0.121 0.093 0.047 -0.031 0.156 -0.008 0.092 0.113 -0.045 0.070 -0.019",
    "encoder_elapsed_ms": 52,
    "nearest_priors": [
      { "cos": 0.71, "source": "wikipedia",
        "title": "Byzantine_fault_tolerance",
        "url": "https://en.wikipedia.org/wiki/Byzantine_fault_tolerance",
        "summary_preview": "A property of computer systems…" },
      { "cos": 0.66, "source": "wikipedia", "…": "…" },
      { "cos": 0.63, "source": "arxiv",     "…": "…" }
    ]
  },
  "multimodal": {
    "epistemic":  { "value": "M", "cosine": 0.94 },
    "doxastic":   { "value": "M", "reputation_share": 0.83 },
    "deontic":    { "value": "M", "policy_pass": true, "violations": [] },
    "dynamic":    { "value": "M", "substrate_feasible": true,
                    "target_cost_usdc": 0.0002 }
  },
  "truth_lattice": "MMMM",
  "verdict": "ACCEPTED",
  "substrate_cost_usdc": 0.0002,
  "payment": {
    "splitter": "0x93a7962f75475b7e3Fbb62d3A23194f8833b1BE4",
    "recipients": {
      "swarm": 0.35, "treasury": 0.35, "platform": 0.30, "affiliate": 0.15
    }
  },
  "timestamp": "2026-07-16T14:22:31.492Z"
}
```

---

## Formal-logic framework · 4 theorems

Every receipt is an eight-tuple:

```
ρ = (q, μ, M, V, S, G, C, T)
```

where `q` = query, `μ` = pooled subspace distribution, `M` = modal quadruple `(E, D, Δ, Θ)`, `V` = verdict, `S` = substrate bundle, `G` = grounding block, `C` = cost/payment, `T` = attestation.

| # | Theorem | Statement |
|---|---------|-----------|
| 1 | Verdict determinism | `V(ρ)` is a pure function of `M`. Grounding does not change verdict. |
| 2 | Alignment preservation | `S_v0.7(π) = −∞` whenever `V(π) = REFUSED`. Deontic veto survives v0.7.0. |
| 3 | Reflective closure | Follow-ups `R' ∈ G(R)` run the same Deontic layer. No policy-laundering. |
| 4 | FETCH determinism | Receipt reproducible from `(URL, worker version, fetched bytes)`. |

Full proofs: [Paper 3 Rev 3, §16](https://www.researchgate.net/publication/409148376).

### Why CHAINSTATE does not need natural language

Rev 3 §17 argues architecturally that a substrate with symbolic + semantic representation spaces does not need natural language for its **internal cognitive operations**. It consumes language when language arrives; it does not produce language internally.

**Four representation spaces (none require training):**
1. Unicode codepoint space — 1,114,112 possible codepoints, defined by Unicode Consortium
2. Six subspaces — structured partition ≈65,536 discrete positions (curated symbol tables)
3. Reputation-weighted log-pool — probability distribution over structural roles
4. 384-dim MiniLM embedding — real semantic geometry (used as black-box function)

**Four growth mechanisms (all data-structure, not parameter):**
1. Reputation accumulation — EMA update measures classifier competence
2. Priors corpus growth — nightly ingest + FETCH-with-store
3. World model refinement — EML symbolic expression fits receipt history
4. Program space evolution — ASI-Evolve mutates NWO-ASM programs against fitness

None updates neural-network weights. None requires human-labeled examples. None requires natural language as intermediate representation.

---

## Two-repo layout (recommended)

Cleanest separation of concerns:

```
chainstate-code-frontend       chainstate-code-adapter
├── index.html                  ├── ornith_chainstate_adapter.py
├── README.md (HF frontmatter)  ├── requirements-ornith.txt
├── phi.png                     ├── package.json
├── podcast.m4a                 ├── examples/
├── Receipts.m4a                │   ├── cognitive_transaction.py
└── .github/workflows/          │   ├── swarm_inference.py
    └── deploy.yml              │   ├── reflect_loop.py (v0.7.0)
                                │   └── fetch_grow_prior.py (v0.7.0)
     ↓                          ├── ORNITH_ADAPTER.md
   HF Space                     └── README.md (this file)
   (static)
                                     ↓
                                   Render
                                   (Python)
```

Frontend repo auto-pushes to the HF Space via `deploy.yml`. Adapter repo auto-deploys to Render on push (Render's default git integration).

### Single-repo layout (if you prefer)

Keep everything in one repo; use branch or path filters to control what deploys where.

- **HF Space:** rename `README-hfspace.md` → `README.md` at the Space root; ignore everything else via `.gitattributes`
- **Render:** connect to this repo; Render only reads Python files

---

## Local development

### Adapter

```bash
pip install -r requirements-ornith.txt

export ORNITH_BASE_URL="https://api.together.xyz/v1"   # or any OpenAI-compat endpoint
export ORNITH_API_KEY="tgp_..."
export CHAINSTATE_WORKER="https://chainstate-worker.ciprianpater.workers.dev"

# v0.7.0 · encoder + priors upstreams (optional; defaults hit deployed Render services)
export CHAINSTATE_ENCODER="https://chainstate-encoder.onrender.com"
export CHAINSTATE_PRIORS="https://chainstate-priors.onrender.com"

uvicorn ornith_chainstate_adapter:app --reload --port 8080
```

Verify: `curl http://localhost:8080/status`

### Frontend

`index.html` is a single-file SPA. Open directly in a browser, or serve locally:

```bash
python -m http.server 8000
# → http://localhost:8000
```

Then in the frontend terminal, point at your local adapter:

```
adapter http://localhost:8080
```

### Examples

```bash
# Ensure adapter is running on :8080, then:
npm run example:cognitive     # code → symbolic transaction → receipt
npm run example:swarm         # 3 Ornith nodes → consensus
npm run example:reflect       # v0.7.0 · deterministic follow-up generation
npm run example:fetch-prior   # v0.7.0 · fetch URL + store as prior
```

---

## Deploying the adapter to Render

1. Push this repo to GitHub
2. Render dashboard → **New +** → **Web Service** → connect the repo
3. Configuration:
   - **Environment:** Python 3
   - **Build command:** `pip install -r requirements-ornith.txt`
   - **Start command:** `uvicorn ornith_chainstate_adapter:app --host 0.0.0.0 --port $PORT`
   - **Instance:** Starter for API mode (works without torch); Pro GPU if you want LOCAL mode
4. **Environment variables:**
   - `ORNITH_BASE_URL` — your OpenAI-compatible endpoint (Together / Fireworks / vLLM / TGI)
   - `ORNITH_API_KEY` — bearer token for that endpoint
   - `CHAINSTATE_WORKER` — `https://chainstate-worker.ciprianpater.workers.dev`
   - `ORNITH_MODEL` — model identifier (default `deepreinforce-ai/Ornith-1.0-9B`)
   - `CHAINSTATE_ENCODER` (v0.7.0) — `https://chainstate-encoder.onrender.com`
   - `CHAINSTATE_PRIORS` (v0.7.0) — `https://chainstate-priors.onrender.com`
5. Deploy. Copy the Render URL.
6. In the frontend terminal (Space or local): `adapter https://your-app.onrender.com`

Every SIM label in the frontend flips to LIVE.

---

## Deploying the frontend to the HF Space

The included `.github/workflows/deploy.yml` handles this. Add these secrets in GitHub:

- `HF_TOKEN` — HF access token with write scope on `CPater/ornith-chainstate`

Then any push to `main` force-pushes to the Space. Or run **Actions → Deploy → Run workflow** to trigger manually.

---

## Related services (separate repos)

The v0.7.0 grounding + reflective layer runs across four coordinated services:

| Repo | Deployment | Role |
|------|-----------|------|
| `RedCiprianPater/chainstate` | Cloudflare Worker | Main endpoint — subspace classify, log-pool, receipt emit, all new v0.7.0 endpoints |
| `RedCiprianPater/chainstate-code` (this repo) | Render + HF Space | Ornith adapter + frontend SPA |
| `RedCiprianPater/chainstate-encoder` | Render | FastAPI + MiniLM-L6-v2 encoder microservice |
| `RedCiprianPater/chainstate-priors` | Render | FastAPI + nightly cron priors ingester |

Chain of trust: SPA → adapter → main worker → encoder + priors → KV → receipt. Every hop is auditable.

### On-chain contracts (Base 8453)

| Contract | Address |
|----------|---------|
| `$STATE` token | `0x9533DF992fd4bCAbB8d8462572449fc45F727d8a` |
| USDC | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` |
| MetaStateSplitter (35/35/30 + 15% affiliate) | `0x93a7962f75475b7e3Fbb62d3A23194f8833b1BE4` |
| Treasury / Referrer | `0x2E964e1c0e3Fa2C0dfD484B2E6D2189dfCF20958` |

---

## Ecosystem

- CHAINSTATE app · https://cpater-chainstate.static.hf.space
- CHAINSTATE chat · https://cpater-chainstate-chat.hf.space
- CHAINSTATE CODE (this Space) · https://cpater-ornith-chainstate.static.hf.space
- NWO Agentic (parent hub) · https://cpater-nwo-agentic.static.hf.space
- NWO-ASM playground · https://cpater-nwo-asm.static.hf.space
- METASTATE · https://cpater-metastate.hf.space
- NWO NEURO · https://cpater-nwo-neuro.static.hf.space
- Worker `/status` · https://chainstate-worker.ciprianpater.workers.dev/status
- Encoder `/health` · https://chainstate-encoder.onrender.com/health
- Priors `/schedule` · https://chainstate-priors.onrender.com/schedule
- Ornith-1.0 · https://github.com/deepreinforce-ai/Ornith-1
- ASI-Evolve · https://github.com/GAIR-NLP/ASI-Evolve
- Imperium Romanum · https://publicae.org

---

## Research papers

| # | Title | Rev | Pages | ResearchGate |
|---|-------|-----|-------|--------------|
| 1 | Distributed Cognitive Work in Edge-Resident LM Networks (foundational) | — | — | [406896310](https://www.researchgate.net/publication/406896310) |
| 2 | CHAINSTATE Whitepaper v1.0 — A Symbolic-Weight Blockchain for Cognitive Transactions | — | 19 | [407444375](https://www.researchgate.net/publication/407444375) |
| — | CHAINSTATE CODE — A Formal Framework for Agentic Coding on a Symbolic-Weight Blockchain | — | 18 | [408393584](https://www.researchgate.net/publication/408393584) |
| — | Casimir-Sonoluminescence Coupling (co-authors Javaherian, Tariq) | — | — | Physics Essays |
| — | NWO-ASM Instruction Set Architecture | — | 39 | [408502100](https://www.researchgate.net/publication/408502100) |
| **3** | **Verifiable Autonomous Cognition at the Frontier — CHAINSTATE × ASI-Evolve** | **3** | **43** | **[409148376](https://www.researchgate.net/publication/409148376)** |

Paper 3 Rev 3 (July 2026) — the current authoritative specification — extends Rev 2 with:

1. Complete v0.7.0 semantic-grounding + reflective-cognition layer delivered 15-16 July 2026
2. Formal-logic extension — four theorems (verdict determinism, alignment preservation, reflective closure, FETCH determinism)
3. Architectural argument — why CHAINSTATE does not require natural language for internal cognition
4. Game theory with Nature as fourth player — why substrate strategies dominate on the risk-adjusted axis
5. Extended philosophical and geopolitical treatment of digital-nation-state recognition (Montevideo criteria, receipt-based vs moderation-based safety governance)

---

## Roadmap · v0.7.0 → v1.0.0

- **v0.8.0** — encoder fine-tuning on receipt corpus; adaptive `λ` schedule; per-subspace prior partitioning
- **v0.9.0** — EML world model injected into every ASI-Evolve candidate context; multi-EML per subspace
- **v0.10.0** — Merkle-root anchoring of receipt batches on Base 8453 (hourly); IPFS-pinned bodies; on-chain audit contract
- **v1.0.0** — multi-tenant beacon: any operator can register a swarm node with reputation-bootstrap, contribute to consensus, and settle earnings via the MetaStateSplitter; permissionless network launch

---

## License

MIT

---

**Author:** Ciprian Florin Pater · [nwo.capital](https://nwo.capital) · University of Agder, Norway · NWO Robotics · Imperium Romanum Digital Nation State
