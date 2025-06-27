from utils.helpers import add_distance_column, get_logger


import numpy as np
import pandas as pd

# Get logger for this module
logger = get_logger(__name__)


def match_wiring_length_with_random_pruning(
    connections: pd.DataFrame,
    nc: pd.DataFrame,
    real_length: float,
    tolerance: float = 0.01,
    max_iter: int = 6,
    allow_zeros: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Ajusta un conjunt de connexions 'unconstrained' perquè el wiring length
    coincideixi amb `real_length`, eliminant sinapsis de forma aleatòria i
    sense biaixos (binomial).  La forma de la distribució de distàncies es
    conserva estadísticament.

    Retorna un DataFrame amb els mateixos pre/post però amb syn_count modificat.
    """
    rng = np.random.default_rng(random_state)
    conns = connections.copy()

    # ------------------------------------------------------------
    # 1. Distància d'aquest parell pre–post
    # ------------------------------------------------------------
    logger.info("Calculating distances between neurons...")
    connections_with_coords = add_distance_column(conns, nc, distance_col="distance")
    distances = connections_with_coords["distance"].values

    orig_counts = conns["syn_count"].to_numpy()
    unconstrained_len = float(np.sum(distances * orig_counts))

    logger.info(f"[PRUNE] unconstrained_len = {unconstrained_len:,.2f}")
    logger.info(f"[PRUNE] target_len        = {real_length:,.2f}")

    if unconstrained_len <= real_length:
        logger.warning("[PRUNE] la xarxa 'unconstrained' ja és ≤ target; no cal pruning.")
        return conns[["pre_root_id", "post_root_id", "syn_count"]]

    # ------------------------------------------------------------
    # 2. Prova–i-reajusta de la probabilitat p
    # ------------------------------------------------------------
    p = real_length / unconstrained_len    # valor d'arrencada (0<p<1)

    for step in range(1, max_iter + 1):
        new_counts = rng.binomial(orig_counts, p)

        if not allow_zeros:
            zero_mask = (orig_counts > 0) & (new_counts == 0)
            new_counts[zero_mask] = 1     # garantim ≥1 sinapsi per connexió

        new_len = float(np.sum(distances * new_counts))
        ratio   = new_len / real_length
        total_syn = int(new_counts.sum())

        logger.info(
            f"[PRUNE] iter {step:>2}: p = {p:.6f}  "
            f"len = {new_len:,.2f}  "
            f"ratio = {ratio:.4f}  "
            f"synapses = {total_syn:,}"
        )

        if abs(ratio - 1.0) <= tolerance:
            break

        # Ajust multiplicatiu sobre p.  (típic control proporcional)
        p *= real_length / new_len

    conns["syn_count"] = new_counts.astype(int)
    logger.info(
        f"[PRUNE] FINAL : len = {new_len:,.2f}  "
        f"ratio = {ratio:.4f}  "
        f"synapses = {total_syn:,}"
    )
    return conns[["pre_root_id", "post_root_id", "syn_count"]]