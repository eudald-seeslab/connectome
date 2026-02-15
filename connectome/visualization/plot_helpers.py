import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from paths import PROJECT_ROOT
from connectome.visualization.plot_config import (
    apply_plot_style,
    RANDOMIZATION_NAMES,
    RANDOMIZATION_COLORS,
    get_randomization_colors,
)


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


# ---- Research-specific plot functions (comparison across randomizations) ----

ORDER = ["Biological", "Binned", "Neuron binned", "Unconstrained", "Connection-pruned"]
NAME_ALIAS = {
    "biological": "biological",
    "binned": "random_binned",
    "random_binned": "random_binned",
    "neuron_binned": "neuron_binned",
    "unconstrained": "unconstrained",
    "connection_pruned": "connection_pruned",
}


def _prepare_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse filename, compute Weber ratio, equalized flag, and return tidy df."""
    df = df.copy()

    df["yellow"] = df["Image"].apply(lambda x: os.path.basename(x).split("_")[1])
    df["blue"]   = df["Image"].apply(lambda x: os.path.basename(x).split("_")[2])

    def _weber(row):
        try:
            y = int(row["yellow"])
            b = int(row["blue"])
            lo, hi = sorted((y, b))
            return np.inf if lo == 0 else hi / lo
        except Exception:
            return np.nan

    df["weber_ratio"] = df.apply(_weber, axis=1)
    df = df[df["weber_ratio"] >= 1.33]
    df["equalized"] = df["Image"].str.lower().str.contains("equalized")
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy and standard error by Weber ratio (only equalized)."""
    tidy = df[df["equalized"] == True].copy()

    agg = (
        tidy.groupby("weber_ratio")["Is correct"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .dropna(subset=["weber_ratio"])
    )
    agg["mean"] *= 100
    agg["std"]  *= 100
    agg["se"]   = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["weber_ratio"] = agg["weber_ratio"].round(3)

    return agg.sort_values("weber_ratio")


def plot_weber_by_randomization(
    folder: str,
    files_to_keys: dict[str, str] | None = None,
    *,
    min_weber: float | None = None,
    save_path: str | None = None,
    ax=None,
    show_legend=True,
):
    """Compare classification accuracy vs. Weber ratio for biological
    connectome and multiple randomizations (surface-equalized only)."""
    if files_to_keys is None:
        files_to_keys = {
            "biological_results.csv":        "biological",
            "binned_results.csv":            "random_binned",
            "neuron_binned_results.csv":     "neuron_binned",
            "unconstrained_results.csv":     "unconstrained",
            "connection_pruned_results.csv": "connection_pruned",
            "random_pruned_results.csv":     "random_pruned",
        }

    curves = []
    for filename, key in files_to_keys.items():
        candidates = [
            os.path.join(folder, filename),
            *glob.glob(os.path.join(folder, filename))
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
        if csv_path is None:
            continue

        df = pd.read_csv(csv_path)
        df = _prepare_results_df(df)
        agg = _aggregate(df)
        if min_weber is not None:
            agg = agg[agg["weber_ratio"] >= float(min_weber)]
        if agg.empty:
            continue

        display = RANDOMIZATION_NAMES.get(key, key)
        color = RANDOMIZATION_COLORS.get(key, "black")
        curves.append((display, color, agg))

    apply_plot_style()

    if ax is None:
        width_inches = 183 / 25.4
        height_inches = width_inches * 0.75
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)
    else:
        fig = ax.get_figure()

    for label, color, data in curves:
        ax.errorbar(
            data["weber_ratio"],
            data["mean"],
            yerr=data["se"],
            label=label,
            color=color,
            marker="o",
            linewidth=2,
            capsize=3,
            capthick=1,
        )

    ax.set_xlabel("Weber Ratio", fontsize=16)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.set_ylim(40, 105)
    if show_legend:
        ax.legend(frameon=False, loc="lower right", fontsize=14)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_ans_accuracies_by_randomization(all_results: pd.DataFrame, ax=None, save=False):
    """Plot Weber fraction by randomization type."""
    apply_plot_style()

    df = all_results.copy()
    df["key"] = df["dataframe"].map(lambda k: NAME_ALIAS.get(k, k))
    df["label"] = df["key"].map(RANDOMIZATION_NAMES)
    df["color"] = df["label"].map(get_randomization_colors)

    df = df[df["label"].isin(ORDER)]
    df["label"] = pd.Categorical(df["label"], categories=ORDER, ordered=True)
    df = df.sort_values("label")

    x = np.arange(len(df))
    y = df["w"].to_numpy(float)
    yerr = df["w_se"].to_numpy(float)
    colors = df["color"].to_list()
    labels = df["label"].to_list()

    if ax is None:
        width_inches = 183 / 25.4
        height_inches = width_inches * 0.75
        fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)
    else:
        fig = ax.get_figure()

    for i, (yi, sei, ci) in enumerate(zip(y, yerr, colors)):
        ax.errorbar(i, yi, yerr=sei, fmt="o", color=ci, capsize=4, elinewidth=2)

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks(x, labels, rotation=35, ha="right")

    pad = 5 * (np.nanmax(yerr) if len(yerr) else 0.0)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    ax.set_ylabel("Weber fraction")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save:
        plots_dir = os.path.join("../..", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(os.path.join(plots_dir, "ans_accuracy.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(plots_dir, "ans_accuracy.pdf"), dpi=300, bbox_inches="tight")

    return ax
