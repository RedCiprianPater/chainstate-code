# CHAINSTATE NWO-ASM Program Evolution

**Owner:** Ciprian Florin Pater · **Ecosystem:** CHAINSTATE · **Chain:** Base mainnet 8453

## Task

Write a Python function `build_program(seed_symbols: list[str]) -> str` that returns a valid NWO-ASM program as a string. The program will be submitted to the CHAINSTATE Cloudflare worker, and the returned consensus receipt will score the candidate.

## Interface

```python
def build_program(seed_symbols: list[str]) -> str:
    """
    seed_symbols: a shuffled subset of the 64k CHAINSTATE symbol space,
                  drawn from the six subspaces (math, sci, lang, occ, emo, ctrl).
                  You do not need to use all of them.
    returns:      a valid NWO-ASM program as a single string, opening with
                  .PROCESS <name>, containing at least one CONSENSUS opcode
                  and one RECEIPT opcode, closing with .END.
    """
```

## What "valid" means

Every candidate MUST:

1. Start with `.PROCESS <name>` and end with `.END`.
2. Contain exactly one `CONSENSUS k=<n> depth=<d>` opcode where n ∈ [10, 100], d ∈ [1, 7].
3. Contain exactly one `RECEIPT emit` opcode after CONSENSUS.
4. Contain at least one `TARGET <substrate>` opcode where substrate ∈ {gpu, qpu, npu, edge}.
5. Encode symbols from the seed set as a query payload — the evaluator will extract them and submit as the `query` field on `POST /query`.

## Scoring — what "better" means

Every candidate is submitted through the real CHAINSTATE worker via `POST /query`. The receipt returned drives the score:

```
score = (confidence · 100) - (gas_used · 5000) - (consensus_rounds · 2)
```

Interpretation:
- **Higher confidence** raises the score (the swarm agreed the program is coherent).
- **Higher gas** lowers the score (efficiency matters).
- **More rounds to converge** lowers the score (fast convergence matters).

Candidates that fail validation return score 0. Candidates that raise an exception during evaluation return score 0.

## Constraints

- **No network calls in the candidate program itself.** The candidate is a program builder, not a network client. The evaluator handles all worker communication.
- **No filesystem writes** in the candidate.
- **Runtime cap:** each candidate must complete in under 5 seconds of CPU time (evaluator enforces via SIGALRM).
- **Program length:** ≤ 500 characters of NWO-ASM.

## Baseline

The initial program is a straightforward FCFS-style linear program: LOAD → TRANSFORM → EMBED → TARGET gpu → EXECUTE → CONSENSUS k=20 depth=3 → RECEIPT emit → STORE → END. It typically scores in the 50–70 range on cache-miss submissions.

## What to explore

The Cognition Store is seeded with the CHAINSTATE subspace layout (Section 4.1 of the whitepaper), the log-pool convergence kernel (Eq. 4), and the observed empirical property that queries whose symbols cluster in a single subspace tend to converge faster than mixed-subspace queries. Beyond that, the search is open. Some directions likely to help:

- Choosing swarm size and consensus depth adaptively based on symbol density in each subspace.
- Grouping high-affinity symbol pairs (co-occurring across the sample pool) at the head of the query.
- Selecting TARGET substrate based on the dominant subspace (occ/emo often route well to edge; math/sci to gpu).
- Using LOOP or BRANCH bricks to iterate a small refinement before committing to CONSENSUS.

These are hypotheses. The Analyzer decides which ones held up round-to-round.

## Objective, restated

**Maximize CHAINSTATE consensus confidence while minimizing gas and time-to-convergence, using only valid NWO-ASM opcodes, over programs whose payloads are drawn from the 64k-symbol space.**

## Provenance

Every trial writes to `database/experiments.sqlite`. The read-only public dashboard at `cpater-ornith-chainstate.static.hf.space` (Evolve view) polls that database. Owner-controlled; step-limited; interruptible.
