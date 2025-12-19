import os
import re
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit

from paths import PROJECT_ROOT


def parse_counts(path: str):
    """Extreu n_yellow i n_blue del nom: .../img_<ny>_<nb>_... ."""
    m = re.search(r'img_(\d+)_(\d+)_', path)
    if m:
        return int(m.group(1)), int(m.group(2))
    return np.nan, np.nan

def parse_equalized(path: str):
    """Marca True si el path conté 'equaliz' (equalized/equalization...)."""
    return "equaliz" in path.lower()

def halberda_p(r, w):
    """Funció psicomètrica del paper (en funció del rati r>=1)."""
    return norm.cdf((r - 1.0) / (w * np.sqrt(1.0 + r**2)))

def fit_weber(group_df, w0=0.25, lower=1e-4, upper=1.0):
    r = group_df["r"].to_numpy(dtype=float)
    y = group_df["p_hat"].to_numpy(dtype=float)
    n = group_df["n_trials"].to_numpy(dtype=float)

    # Desviació estàndard binomial de p̂: sqrt(p(1-p)/n) (clipejada per estabilitat)
    eps = 1e-6
    sigma = np.sqrt(np.clip(y * (1 - y) / n, eps, None))

    # Ajust no lineal ponderat
    popt, pcov = curve_fit(
        halberda_p, r, y,
        p0=[w0], bounds=(lower, upper),
        sigma=sigma, absolute_sigma=True,
        maxfev=20000
    )
    w = float(popt[0])
    # SE i IC95% (aprox.; si estàs a la frontera, millor bootstrap)
    se = float(np.sqrt(pcov[0, 0])) if pcov.size else np.nan
    ci_low = max(lower, w - 1.96 * se) if np.isfinite(se) else np.nan
    ci_high = min(upper, w + 1.96 * se) if np.isfinite(se) else np.nan

    out = {
        "w": w, "w_se": se,
        "w_ci_low": ci_low, "w_ci_high": ci_high,
        "n_bins": group_df["r"].nunique(),
        "n_trials_total": int(group_df["n_trials"].sum())
    }
    return pd.Series(out)

def compute_weber_ratio(data_file_name: str):
    file_path = os.path.join(PROJECT_ROOT, "data", data_file_name)
    df = pd.read_csv(file_path)
    df = df.rename(columns=lambda s: s.strip().lower().replace(" ", "_"))
    df["n_yellow"], df["n_blue"] = zip(*df["image"].map(parse_counts))
    df["equalized"] = df["image"].map(parse_equalized)
    # rati r>=1
    larger = np.maximum(df["n_yellow"], df["n_blue"])
    smaller = np.minimum(df["n_yellow"], df["n_blue"])
    df["r"] = larger / smaller
    df = df[np.isfinite(df["r"]) & (df["r"] > 1)]

    # --- 2) Agregació per rati (cada r és un bin) ---
    values = (
        df.groupby(["equalized", "r"], as_index=False)
        .agg(n_trials=("is_correct", "size"),
            k=("is_correct", "sum"))
    )
    values["p_hat"] = values["k"] / values["n_trials"]

    # Resultats per condició equalized
    res_by_eq = values.groupby("equalized").apply(fit_weber).reset_index()

    # (Opcional) Resultat global col·lapsant equalized
    res_global = fit_weber(values.assign(equalized="ALL").groupby("equalized").get_group("ALL")).to_frame().T
    res_global["equalized"] = "ALL"
    res_global = res_global[["equalized", "w", "w_se", "w_ci_low", "w_ci_high", "n_bins", "n_trials_total"]]

    # Taula final
    return pd.concat([res_by_eq, res_global], ignore_index=True)
