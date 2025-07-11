import numpy as np
import pandas as pd

from utils.helpers import get_logger
from utils.randomizers.numba_helpers import euclidean_rows

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
    *,
    silent: bool = False,
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

    prev_disabled = logger.disabled
    logger.disabled = silent or prev_disabled

    # ------------------------------------------------------------
    # NumPy distance vector (no pandas merge)
    # ------------------------------------------------------------

    pre_ids  = conns["pre_root_id"].to_numpy(dtype=np.int64)
    post_ids = conns["post_root_id"].to_numpy(dtype=np.int64)
    orig_counts = conns["syn_count"].to_numpy(dtype=np.int32)

    roots = nc["root_id"].to_numpy(dtype=np.int64)
    coords = nc[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32)

    order = np.argsort(roots)
    roots_s = roots[order]
    coords_s = coords[order]

    idx_pre  = np.searchsorted(roots_s, pre_ids)
    idx_post = np.searchsorted(roots_s, post_ids)

    pre_xyz  = coords_s[idx_pre]
    post_xyz = coords_s[idx_post]

    distances = euclidean_rows(pre_xyz, post_xyz)

    if not silent:
        logger.info("Distances computed (NumPy)")

    unconstrained_len = float(np.sum(distances * orig_counts))

    if not silent:
        logger.info(
            f"[PRUNE] unconstrained_len = {unconstrained_len:,.2f}\n"
            f"[PRUNE] target_len        = {real_length:,.2f}"
        )

    if unconstrained_len <= real_length:
        if not silent:
            logger.warning("[PRUNE] La xarxa 'unconstrained' ja és ≤ target; no cal pruning.")
        return conns[["pre_root_id", "post_root_id", "syn_count"]]

    # ------------------------------------------------------------
    # 2. Single global binomial thinning (vectorised) -------------
    # ------------------------------------------------------------

    p_keep = real_length / unconstrained_len  # 0 < p_keep < 1
    new_counts = rng.binomial(orig_counts, p_keep)

    if not allow_zeros:
        zero_mask = (orig_counts > 0) & (new_counts == 0)
        new_counts[zero_mask] = 1

    new_len = float(np.sum(distances * new_counts))
    ratio   = new_len / real_length

    if not silent:
        logger.info(
            f"[PRUNE] one-shot draw : len = {new_len:,.2f}  "
            f"ratio = {ratio:.4f}  "
            f"synapses = {int(new_counts.sum()):,}"
        )

    # ------------------------------------------------------------
    # 3. Lightweight stochastic correction (≤2 passes) ------------
    # ------------------------------------------------------------

    passes = 0
    while abs(ratio - 1.0) > tolerance and passes < 2:
        diff        = new_len - real_length
        passes     += 1

        if diff > 0:  # Too long → randomly drop extra synapses
            q_drop = min(1.0, diff / new_len)  # expected reduction ≈ diff
            to_drop = rng.binomial(new_counts, q_drop)
            new_counts -= to_drop

            if not allow_zeros:
                zero_mask = (orig_counts > 0) & (new_counts == 0)
                new_counts[zero_mask] = 1

        else:  # Too short → randomly restore some previously removed synapses
            missing      = orig_counts - new_counts
            capacity_len = float(np.sum(distances * missing))

            if capacity_len == 0:
                break  # Cannot add back anything

            q_add = min(1.0, (-diff) / capacity_len)
            to_add = rng.binomial(missing, q_add)
            new_counts += to_add

        # Recompute
        new_len = float(np.sum(distances * new_counts))
        ratio   = new_len / real_length

        if not silent:
            logger.info(
                f"[PRUNE] correction {passes}: len = {new_len:,.2f}  ratio = {ratio:.4f}"
            )

    conns["syn_count"] = new_counts.astype(int)

    if not silent:
        logger.info(
            f"[PRUNE] FINAL : len = {new_len:,.2f}  ratio = {ratio:.4f}  "
            f"synapses = {int(new_counts.sum()):,}  passes = {passes}"
        )

    result = conns[["pre_root_id", "post_root_id", "syn_count"]]

    logger.disabled = prev_disabled
    return result