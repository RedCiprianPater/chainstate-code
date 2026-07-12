"""
Seed the ASI-Evolve Cognition Store with CHAINSTATE domain knowledge.

Run once before starting the evolve loop:
    python init_cognition.py

Owner: Ciprian Florin Pater
"""

import os
import sys

# ASI-Evolve import — assumes ASI-Evolve is installed in the parent env
try:
    from cognition.store import CognitionStore
except ImportError:
    print("✗ Cannot import cognition.store — is ASI-Evolve installed?", file=sys.stderr)
    print("  Try: cd ../ASI-Evolve && pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(HERE, "cognition_data")

store = CognitionStore(storage_dir=STORAGE_DIR)

store.add([
    {
        "title": "CHAINSTATE Subspace Layout",
        "content": (
            "The Universal Semiotic Embedding partitions a 65,536-dimensional space "
            "into six named subspaces: math (4,096 dims, 0-4,095), sci (8,192 dims, "
            "4,096-12,287), lang (16,384 dims, 12,288-28,671), occ (4,096 dims, "
            "28,672-32,767), emo (16,384 dims, 32,768-49,151), ctrl (16,384 dims, "
            "49,152-65,535). Dimensions sum to exactly 2^16. Classification is by "
            "codepoint-count majority; ties are broken in favor of the larger-dim "
            "subspace to preserve the entropy budget."
        ),
    },
    {
        "title": "Log-Pool Convergence Kernel",
        "content": (
            "Consensus is a reputation-weighted Bayesian log-pool: "
            "p_cons(x) proportional to product of p_i(x)^w_i, where "
            "w_i = r_i / sum(r_j). Equivalently in log-space: "
            "log p_cons(x) = sum(w_i * log p_i(x)) - log Z. "
            "Convergence criterion: cosine(p_cons_t, p_cons_(t-1)) >= 0.95. "
            "Typical convergence: 3-7 rounds. This is Eq. 4-6 of the CHAINSTATE "
            "CODE whitepaper."
        ),
    },
    {
        "title": "Same-Subspace Density Prior",
        "content": (
            "Empirically, queries whose symbols cluster in one dominant subspace "
            "converge in fewer log-pool rounds than mixed-subspace queries. "
            "Rationale: reputation weights are subspace-neutral, but node states "
            "for one-subspace queries have narrower probability mass, so their "
            "geometric mean concentrates faster. Practical rule: bias the seed "
            "payload toward a single subspace when confidence is the primary "
            "objective; mix subspaces only when the task requires it."
        ),
    },
    {
        "title": "Substrate-Subspace Affinity",
        "content": (
            "Observed affinity between dominant subspace and preferred TARGET: "
            "math and sci often route well to gpu (dense tensor ops on structured "
            "symbol pools); occ and emo often route well to edge (small pools, "
            "low latency benefit from proximity). ctrl is substrate-neutral. "
            "lang is context-dependent — code-heavy lang queries prefer gpu, "
            "natural-language lang queries can go to edge. qpu is currently a "
            "planned target; specifying it does not fail the receipt but does "
            "not accelerate convergence in this deployment."
        ),
    },
    {
        "title": "KV Cache Behavior",
        "content": (
            "The CHAINSTATE worker caches receipts by qHash = SHA3-256(query) "
            "with a 5-minute TTL. Cache HIT reduces gas from ~0.0019 $STATE to "
            "~0.00012 $STATE (16x reduction). During evolutionary search this "
            "matters: identical payloads reused across rounds will score with "
            "artificially low gas. The evaluator randomizes symbol order between "
            "trials to force cache MISS and get honest gas measurements."
        ),
    },
    {
        "title": "Swarm Size vs Depth Trade-off",
        "content": (
            "Larger swarmSize (k) increases confidence variance-of-mean at O(1/sqrt(k)) "
            "but adds linear gas. Higher consensusDepth adds log-pool rounds "
            "which sharpen convergence but also add gas linearly. Rule of thumb: "
            "if a query converges at depth=1 with k=20, larger k rarely helps. "
            "If a query fails to converge at depth=3, doubling k helps more than "
            "doubling depth (breaks reputation-weighting deadlocks)."
        ),
    },
    {
        "title": "NWO-ASM Opcode Constraints",
        "content": (
            "Valid opcodes: FLOW (PROCESS, END, LOOP, BRANCH), DATA (LOAD, STORE, "
            "TRANSFORM, EMBED), COMPUTE (EXECUTE, MATMUL, HASH), CHAIN (CONSENSUS, "
            "RECEIPT, GAS), TARGET (gpu, qpu, npu, edge). Every valid program "
            "opens with .PROCESS and closes with .END. Programs with mismatched "
            "block delimiters are rejected at compile time. LOOP with n > 8 is "
            "rejected as unrolling budget. BRANCH requires a labelled destination."
        ),
    },
    {
        "title": "Symbol Pools by Subspace",
        "content": (
            "math pool includes integral, partial, nabla, sum, product, element-of, "
            "for-all, exists, infinity, approximately-equal, less-or-equal, "
            "greater-or-equal, and logic arrows. "
            "sci pool includes reduced Planck, atom, biohazard, DNA, microscope, "
            "test-tube. "
            "occ pool includes sun, moon, mercury, venus, mars, jupiter, saturn "
            "planetary symbols; zodiac; alchemical sigils. "
            "emo pool draws from Unicode 15.1 emoji including brain, robot, "
            "dragon, sparkle, fire, water. "
            "ctrl pool includes double-arrows in all directions, refresh cycles. "
            "lang pool spans Greek, Cyrillic, CJK, Arabic, Hebrew, Devanagari, Korean."
        ),
    },
    {
        "title": "Objective Restated",
        "content": (
            "score = (confidence * 100) - (gas_used * 5000) - (consensus_rounds * 2). "
            "Maximize this. High-confidence low-gas fast-converging programs win. "
            "The search space is combinatorial across: payload symbol choice, "
            "payload symbol ordering, swarm size in [10,100], consensus depth in "
            "[1,7], TARGET substrate in {gpu, qpu, npu, edge}, optional LOOP/BRANCH "
            "insertion. Every dimension has honest empirical priors above."
        ),
    },
    {
        "title": "Ownership and Governance",
        "content": (
            "The CHAINSTATE ecosystem is owned by Ciprian Florin Pater. This "
            "search runs under the owner's authority, with the objective set by "
            "the owner. Every trial the loop produces is written to the "
            "experiment database and published read-only to a public dashboard. "
            "The loop is interruptible at any time via POST /v1/evolve/stop. "
            "Runs are hard-capped at EVOLVE_STEPS_MAX rounds per invocation."
        ),
    },
])

print(f"✓ Seeded {STORAGE_DIR} with 10 CHAINSTATE cognition items")
