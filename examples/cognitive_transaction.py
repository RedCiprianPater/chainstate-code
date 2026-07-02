#!/usr/bin/env python3
"""
Example: Cognitive Transaction — Code Generation as a CHAINSTATE Transaction

Demonstrates the full Ornith × CHAINSTATE pipeline:
  1) Ornith generates code + reasoning + 65,536-d symbolic embedding
  2) The generation is submitted to CHAINSTATE as a symbolic transaction,
     producing a real ⛓ consensus receipt
  3) The code is compiled to NWO-ASM Process-Matrix IR

Prereq: adapter running (default http://localhost:8080)
  uvicorn ornith_chainstate_adapter:app --port 8080
"""

import asyncio
import os
import sys
import httpx

ADAPTER_URL = os.getenv("ADAPTER_URL", "http://localhost:8080")


async def cognitive_transaction_example():
    async with httpx.AsyncClient(timeout=120.0) as client:

        # ── Step 1: adapter health ────────────────────────────────
        print(f"→ checking adapter at {ADAPTER_URL}")
        try:
            status = (await client.get(f"{ADAPTER_URL}/status")).json()
        except Exception as e:
            print(f"✗ adapter unreachable: {e}")
            sys.exit(1)
        print(f"  mode={status.get('mode')}  model={status.get('ornith_model')}")
        worker = (status.get("worker_status") or {})
        if worker.get("error"):
            print(f"  ⚠ upstream worker: {worker['error']}")
        else:
            print(f"  ⛓ worker LIVE  version={worker.get('worker_version','?')}")

        # ── Step 2: Ornith generation ─────────────────────────────
        prompt = "Write a Python function is_prime(n) that returns True if n is prime. Include docstring and type hints."
        print(f"\n🦅 Ornith generating for prompt:")
        print(f"   {prompt!r}")

        r = await client.post(
            f"{ADAPTER_URL}/v1/generate",
            json={"prompt": prompt, "temperature": 0.6},
        )
        r.raise_for_status()
        gen = r.json()

        print(f"\n📝 Code ({gen.get('mode','?')} mode):")
        print("   " + "\n   ".join(gen["code"].splitlines()[:12]) + ("\n   …" if len(gen["code"].splitlines()) > 12 else ""))
        if gen.get("reasoning"):
            print(f"\n🧠 Reasoning excerpt:")
            print(f"   {gen['reasoning'][:220]}{'…' if len(gen['reasoning']) > 220 else ''}")

        emb = gen["symbolic_embedding"]
        print(f"\n🔢 Symbolic embedding: dim={len(emb)}  ‖v‖≈{sum(x*x for x in emb)**0.5:.4f}")

        # ── Step 3: submit as cognitive transaction ───────────────
        print(f"\n⛓ Submitting to CHAINSTATE (POST /v1/query)…")
        r = await client.post(
            f"{ADAPTER_URL}/v1/query",
            json={
                "query": "Generate prime-checking function ∫∂n ∈ ℙ → bool",
                "swarmSize": 20,
                "consensusDepth": 3,
            },
        )
        r.raise_for_status()
        tx = r.json()
        cs = tx.get("chainstate_result", {})

        print(f"\n✅ Consensus Receipt:")
        print(f"   qHash              {cs.get('qHash','—')}")
        print(f"   dominant_subspace  {cs.get('dominant_subspace','—')}")
        print(f"   top_symbols        {cs.get('top_symbols','—')}")
        print(f"   confidence         {cs.get('confidence','—')}")
        print(f"   participatingNodes {cs.get('participatingNodes','—')}")
        print(f"   gasUsed            {cs.get('gasUsed','—')} $STATE")
        print(f"   cache              {cs.get('cache','—')}")
        print(f"   timestamp          {cs.get('timestamp','—')}")

        # ── Step 4: compile to NWO-ASM IR ────────────────────────
        print(f"\n⚙️ Compiling to NWO-ASM IR (target=gpu)…")
        r = await client.post(
            f"{ADAPTER_URL}/v1/asm/compile",
            json={"code": gen["code"], "target": "gpu"},
        )
        r.raise_for_status()
        asm = r.json()

        print(f"\n🔧 IR:")
        for line in asm["ir"].splitlines():
            print(f"   {line}")
        print(f"\n   target            {asm['target']}")
        print(f"   estimated_cycles  {asm['estimated_cycles']}")

        print(f"\n─── pipeline complete ───")


if __name__ == "__main__":
    asyncio.run(cognitive_transaction_example())
