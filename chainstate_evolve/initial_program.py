"""
CHAINSTATE NWO-ASM baseline builder.

Owner: Ciprian Florin Pater
Ecosystem: CHAINSTATE · Base mainnet 8453

This is the initial program ASI-Evolve starts from. It is deliberately simple:
a linear NWO-ASM program with default consensus parameters. Expect it to score
in the 50-70 range on cache-miss submissions. Evolutionary rounds should
discover swarm-size, consensus-depth, and target-substrate adaptations that
substantially improve on this.
"""


def build_program(seed_symbols):
    """Return a valid NWO-ASM program that encodes the seed symbols as the query payload.

    Args:
        seed_symbols: list[str] — a shuffled subset of the 64k CHAINSTATE
                      symbol space, drawn across the six subspaces.

    Returns:
        str — a valid NWO-ASM program.
    """
    # Trim to a manageable payload — the worker classifies by density anyway
    payload_symbols = seed_symbols[:12]
    payload = "".join(payload_symbols)

    program = (
        f".PROCESS baseline_linear\n"
        f"  LOAD payload={payload!r}\n"
        f"  TRANSFORM symbolic_embedding\n"
        f"  EMBED use65536\n"
        f"  TARGET gpu\n"
        f"  EXECUTE substrate\n"
        f"  CONSENSUS k=20 depth=3\n"
        f"  RECEIPT emit\n"
        f"  STORE result_segment\n"
        f".END\n"
    )
    return program
