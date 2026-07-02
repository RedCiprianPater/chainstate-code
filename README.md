# chainstate-code · Ornith-1.0 × CHAINSTATE

FastAPI adapter and static frontend for the CHAINSTATE CODE Space — agentic coding on a symbolic-weight blockchain.

**Live frontend:** https://cpater-ornith-chainstate.static.hf.space
**Chain:** Base mainnet 8453

## What this repo contains

| Path | Purpose | Deploys to |
|------|---------|-----------|
| `index.html` | Full SPA — Builder / Terminal / Agent Chat / Simulation / Architecture | HF Space (static) |
| `README-hfspace.md` | HF Space README (rename to `README.md` inside the Space repo) | HF Space |
| `ornith_chainstate_adapter.py` | FastAPI bridge with LOCAL/API modes | Render |
| `requirements-ornith.txt` | Adapter Python dependencies | Render |
| `package.json` | npm scripts for dev convenience | dev |
| `examples/cognitive_transaction.py` | Code → symbolic transaction → receipt | reference |
| `examples/swarm_inference.py` | Multi-node reputation-weighted consensus | reference |
| `ORNITH_ADAPTER.md` | Full API + integration documentation | docs |
| `.github/workflows/deploy.yml` | Push frontend to HF Space on merge to main | CI |

## Two-repo layout (recommended)

Cleanest separation of concerns:

```
chainstate-code-frontend       chainstate-code-adapter
├── index.html                  ├── ornith_chainstate_adapter.py
├── README.md (HF frontmatter)  ├── requirements-ornith.txt
├── phi.png                     ├── package.json
└── .github/workflows/          ├── examples/
    └── deploy.yml              │   ├── cognitive_transaction.py
                                │   └── swarm_inference.py
     ↓                          ├── ORNITH_ADAPTER.md
   HF Space                     └── README.md (this file)
   (static)
                                     ↓
                                   Render
                                   (Python)
```

**Frontend repo** auto-pushes to the HF Space via `deploy.yml`.
**Adapter repo** auto-deploys to Render on push (Render's default git integration).

## Single-repo layout (if you prefer)

Keep everything in one repo; use branch or path filters to control what deploys where.

- HF Space: rename `README-hfspace.md` → `README.md` at the Space root; ignore everything else via `.gitattributes`
- Render: connect to this repo; Render only reads Python files

## Local development

### Adapter

```bash
pip install -r requirements-ornith.txt

export ORNITH_BASE_URL="https://api.together.xyz/v1"   # or any OpenAI-compat endpoint
export ORNITH_API_KEY="tgp_..."
export CHAINSTATE_WORKER="https://chainstate-worker.ciprianpater.workers.dev"

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
npm run example:cognitive   # code → symbolic transaction → receipt
npm run example:swarm        # 3 Ornith nodes → consensus
```

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
5. Deploy. Copy the Render URL.
6. In the frontend terminal (Space or local): `adapter https://your-app.onrender.com`

Every SIM label in the frontend flips to LIVE.

## Deploying the frontend to the HF Space

The included `.github/workflows/deploy.yml` handles this. Add these secrets in GitHub:

- `HF_TOKEN` — HF access token with write scope on `CPater/ornith-chainstate`

Then any push to `main` force-pushes to the Space. Or run **Actions → Deploy → Run workflow** to trigger manually.

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/generate` | Ornith code + 65,536-d symbolic embedding |
| POST | `/v1/embed` | Symbolic embedding only |
| POST | `/v1/query` | Full pipeline: generate → embed → CHAINSTATE consensus |
| POST | `/v1/consensus` | Swarm consensus participation |
| POST | `/v1/asm/compile` | NWO-ASM Process-Matrix IR generation |
| GET | `/status` | Adapter + upstream worker health |

Full schemas in `ORNITH_ADAPTER.md`.

## Ecosystem

- CHAINSTATE app · https://cpater-chainstate.static.hf.space
- CHAINSTATE chat · https://cpater-chainstate-chat.hf.space
- NWO-ASM playground · https://cpater-nwo-asm.static.hf.space
- Worker `/status` · https://chainstate-worker.ciprianpater.workers.dev/status
- Ornith-1.0 · https://github.com/deepreinforce-ai/Ornith-1

## License

MIT
