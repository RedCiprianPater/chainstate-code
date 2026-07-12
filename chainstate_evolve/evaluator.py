"""
CHAINSTATE evaluator for ASI-Evolve.

Called once per trial: load the candidate's build_program function, generate
a randomized seed payload from the 64k symbol pool, submit through the real
CHAINSTATE worker, and compute the score.

Owner: Ciprian Florin Pater
"""

import importlib.util
import json
import os
import random
import signal
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

CHAINSTATE_WORKER = os.getenv(
    "CHAINSTATE_WORKER",
    "https://chainstate-worker.ciprianpater.workers.dev",
)
CANDIDATE_TIMEOUT_S = 5   # per-candidate build_program CPU budget
WORKER_TIMEOUT_S    = 45  # network round-trip budget

# Trimmed but representative symbol pools from the six subspaces.
# ASI-Evolve's seed_symbols argument is drawn from these.
SYMBOL_POOLS = {
    "math": list("∫∂∇∆∑∏∀∃∈∉∪∩⊂⊃∞∝≈≠≤≥√→⇒↔"),
    "sci":  list("ℏΦΨΩ⚛☢☣") + ["🧬", "🔬", "⚗"],
    "lang": list("ΑΒΓΔαβγАБВ道心学智一二三"),
    "occ":  list("☉☽☿♀♂♃♄☯☤☥☦☪✡☮"),
    "emo":  ["🧠", "🤔", "💎", "✨", "🔥", "🌟", "⚡"],
    "ctrl": list("⇒⇐⇑⇓⇔↻⇄⇆↔"),
}


class CandidateTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise CandidateTimeout("candidate build_program exceeded CPU budget")


def load_candidate(path):
    """Import the candidate program file and return its build_program function."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_program"):
        raise AttributeError("candidate must define build_program(seed_symbols)")
    return mod.build_program


def seed_symbols(n=16, trial_seed=None):
    """Draw a shuffled subset of n symbols from across the pools."""
    rng = random.Random(trial_seed if trial_seed is not None else time.time_ns())
    pool = []
    for pool_symbols in SYMBOL_POOLS.values():
        pool.extend(pool_symbols)
    rng.shuffle(pool)
    return pool[:n]


def validate_program(program):
    """Return (True, None) if valid; (False, reason) otherwise."""
    if not isinstance(program, str):
        return False, "build_program did not return a string"
    if len(program) > 500:
        return False, f"program too long ({len(program)} > 500 chars)"
    if not program.strip().startswith(".PROCESS"):
        return False, "missing .PROCESS header"
    if ".END" not in program:
        return False, "missing .END terminator"
    if "CONSENSUS" not in program:
        return False, "missing CONSENSUS opcode"
    if "RECEIPT emit" not in program and "RECEIPT" not in program:
        return False, "missing RECEIPT opcode"
    if "TARGET" not in program:
        return False, "missing TARGET opcode"
    return True, None


def extract_payload(program):
    """Pull the LOAD payload='...' or payload=... symbols out of the program."""
    import re
    m = re.search(r"LOAD\s+payload\s*=\s*['\"]([^'\"]*)['\"]", program)
    if m:
        return m.group(1)
    m2 = re.search(r"LOAD\s+([^\n]+)", program)
    return m2.group(1).strip() if m2 else ""


def submit_to_chainstate(query, swarm_size, consensus_depth):
    """POST to the real CHAINSTATE worker; return the parsed receipt or None."""
    payload = json.dumps({
        "query": query,
        "swarmSize": swarm_size,
        "consensusDepth": consensus_depth,
        "cache": False,  # honest gas measurement
    }).encode("utf-8")
    req = Request(
        f"{CHAINSTATE_WORKER}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=WORKER_TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        return {"error": str(e)[:200]}


def extract_params_from_program(program):
    """Parse k and depth from the CONSENSUS line (with safe defaults)."""
    import re
    k = 20
    depth = 3
    m = re.search(r"CONSENSUS\s+k\s*=\s*(\d+)", program)
    if m:
        k = max(10, min(100, int(m.group(1))))
    m = re.search(r"depth\s*=\s*(\d+)", program)
    if m:
        depth = max(1, min(7, int(m.group(1))))
    return k, depth


def score_receipt(receipt, k_used, depth_used):
    """score = confidence*100 - gas*5000 - depth*2. Rejects receipts with errors."""
    if not receipt or "error" in receipt:
        return 0.0, {"reason": "no_receipt", "detail": receipt}
    try:
        confidence = float(receipt.get("confidence", 0.0))
        gas = float(receipt.get("gasUsed", 1.0))
        depth = int(receipt.get("consensusDepth", depth_used))
    except (TypeError, ValueError):
        return 0.0, {"reason": "malformed_receipt", "detail": receipt}
    score = (confidence * 100.0) - (gas * 5000.0) - (depth * 2.0)
    return score, {
        "confidence": confidence,
        "gas": gas,
        "depth_reported": depth,
        "dominant_subspace": receipt.get("dominant_subspace"),
        "top_symbols": receipt.get("top_symbols"),
        "participating_nodes": receipt.get("participatingNodes"),
        "cache": receipt.get("cache", "MISS"),
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"score": 0.0, "metrics": {"reason": "no_candidate_path"}}))
        return
    candidate_path = sys.argv[1]

    try:
        build = load_candidate(candidate_path)
    except Exception as e:
        print(json.dumps({"score": 0.0, "metrics": {"reason": "load_error", "detail": str(e)[:200]}}))
        return

    trial_seed = os.getpid() ^ time.time_ns()
    symbols = seed_symbols(16, trial_seed=trial_seed)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CANDIDATE_TIMEOUT_S)
    try:
        program = build(symbols)
    except CandidateTimeout:
        print(json.dumps({"score": 0.0, "metrics": {"reason": "candidate_timeout"}}))
        return
    except Exception as e:
        print(json.dumps({"score": 0.0, "metrics": {"reason": "build_exception", "detail": str(e)[:200]}}))
        return
    finally:
        signal.alarm(0)

    ok, reason = validate_program(program)
    if not ok:
        print(json.dumps({"score": 0.0, "metrics": {"reason": "invalid", "detail": reason}}))
        return

    query = extract_payload(program)
    if not query:
        # Fall back to the symbols themselves as the payload
        query = "".join(symbols[:12])
    k, depth = extract_params_from_program(program)

    receipt = submit_to_chainstate(query, k, depth)
    score, metrics = score_receipt(receipt, k, depth)
    metrics["k"] = k
    metrics["depth_requested"] = depth
    metrics["program_length"] = len(program)

    print(json.dumps({"score": round(score, 3), "metrics": metrics}))


if __name__ == "__main__":
    main()
