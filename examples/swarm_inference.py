#!/usr/bin/env python3
"""
Example: Ornith as CHAINSTATE Swarm Consensus Nodes

Demonstrates reputation-weighted Bayesian log-pooling with multiple Ornith
instances contributing symbolic states for the same query.

  1) N Ornith nodes each generate a solution + 65,536-d symbolic state
  2) States are pooled: p_cons(x) ∝ ∏ p_i(x)^w_i,  w_i = r_i / Σ r_j
  3) The node whose state has the highest cosine similarity to the
     consensus wins; its code is the accepted output

Prereq: adapter running (default http://localhost:8080)
  uvicorn ornith_chainstate_adapter:app --port 8080
"""

import asyncio
import os
import sys
import httpx
import numpy as np

ADAPTER_URL = os.getenv("ADAPTER_URL", "http://localhost:8080")


async def swarm_consensus_example():
    async with httpx.AsyncClient(timeout=120.0) as client:

        # ── Adapter health ───────────────────────────────────────
        print(f"→ checking adapter at {ADAPTER_URL}")
        try:
            status = (await client.get(f"{ADAPTER_URL}/status")).json()
        except Exception as e:
            print(f"✗ adapter unreachable: {e}")
            sys.exit(1)
        print(f"  mode={status.get('mode')}  model={status.get('ornith_model')}")

        # ── Simulate N Ornith nodes ──────────────────────────────
        nodes = [
            {"name": "Ornith-9B-A",  "reputation": 0.95, "temperature": 0.55},
            {"name": "Ornith-35B-B", "reputation": 0.98, "temperature": 0.60},
            {"name": "Ornith-9B-C",  "reputation": 0.92, "temperature": 0.70},
        ]
        query = "Optimize database query for O(n log n) → O(1)"

        print(f"\n🌐 Swarm consensus")
        print(f"   query        {query!r}")
        print(f"   nodes        {len(nodes)}")
        print(f"   total_rep    {sum(n['reputation'] for n in nodes):.2f}")

        # ── Each node generates a state ──────────────────────────
        results = []
        for node in nodes:
            print(f"\n🦅 {node['name']}  (rep={node['reputation']}, T={node['temperature']})")
            r = await client.post(
                f"{ADAPTER_URL}/v1/generate",
                json={
                    "prompt": f"{query}\nProvide solution with brief mathematical justification.",
                    "temperature": node["temperature"],
                },
            )
            r.raise_for_status()
            gen = r.json()
            state = np.asarray(gen["symbolic_embedding"], dtype=np.float32)
            results.append({
                "node": node["name"],
                "reputation": node["reputation"],
                "state": state,
                "code": gen["code"],
            })
            print(f"   state dim={state.shape[0]}  ‖v‖={np.linalg.norm(state):.4f}")

        # ── Reputation-weighted log-pool ─────────────────────────
        total_rep = sum(r["reputation"] for r in results)
        weights = [r["reputation"] / total_rep for r in results]
        print(f"\n∑ Reputation-weighted log-pool")
        print(f"   weights  {[round(w,3) for w in weights]}")

        log_pooled = np.zeros(65536, dtype=np.float32)
        for r, w in zip(results, weights):
            log_pooled += np.log(np.abs(r["state"]) + 1e-10) * w
        consensus = np.exp(log_pooled - np.max(log_pooled))
        consensus = consensus / consensus.sum()

        # ── Submit consensus to worker ───────────────────────────
        print(f"\n⛓ Submitting consensus (POST /v1/consensus)…")
        try:
            r = await client.post(
                f"{ADAPTER_URL}/v1/consensus",
                json={"state": consensus.tolist(), "reputation": total_rep},
            )
            cs = r.json()
        except Exception as e:
            cs = {"error": str(e)}

        print(f"   status  {cs.get('status', cs.get('error', 'unknown'))}")
        if cs.get("rounds") is not None:
            print(f"   rounds  {cs['rounds']}")
        if cs.get("cosine") is not None:
            print(f"   cosine  {cs['cosine']:.4f}")

        # ── Cosine-similarity to consensus → winner ─────────────
        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        ranked = sorted(results, key=lambda r: cos(r["state"], consensus), reverse=True)

        print(f"\n🏆 Consensus ranking (cosine similarity to pooled state)")
        for i, r in enumerate(ranked, 1):
            marker = "★" if i == 1 else " "
            print(f"   {marker} {i}. {r['node']:<16} cos={cos(r['state'], consensus):.4f}  rep={r['reputation']}")

        winner = ranked[0]
        print(f"\n📝 Winning solution — {winner['node']}:")
        for line in winner["code"].splitlines()[:14]:
            print(f"   {line}")
        if len(winner["code"].splitlines()) > 14:
            print("   …")

        print(f"\n─── consensus complete ───")


if __name__ == "__main__":
    asyncio.run(swarm_consensus_example())
