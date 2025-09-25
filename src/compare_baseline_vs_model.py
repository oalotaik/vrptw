import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DAY_ORDER_BASE = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu"]
DAY_ORDER_MODEL = ["SAT", "SUN", "MON", "TUE", "WED", "THU"]

METRICS = [
    "num_routes",
    "total_visits",
    "avg_route_length_min",
    "std_route_length_min",
    "avg_stops_per_route",
    "std_stops_per_route",
    "avg_overtime_hours",
    "std_overtime_hours",
    "pct_routes_with_overtime",
]


def load_per_day(path: Path, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "day" not in df.columns:
        raise ValueError(f"{kind} per-day file missing 'day' column: {path}")
    df["day"] = df["day"].astype(str)
    df["day_norm"] = df["day"].str.upper()
    df["day_plot"] = df["day_norm"].str.title()
    df = df[df["day_norm"].isin(DAY_ORDER_MODEL)].copy()
    df["order"] = df["day_norm"].apply(lambda x: DAY_ORDER_MODEL.index(x))
    df = df.sort_values("order").reset_index(drop=True)
    return df


def find_files(
    baseline_dir: Path,
    model_dir: Path,
    baseline_csv: Path = None,
    model_csv: Path = None,
):
    if baseline_csv is None:
        baseline_csv = baseline_dir / "baseline_routes_per_day_metrics.csv"
    if model_csv is None:
        model_csv = model_dir / "per_day_metrics.csv"
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline per-day metrics not found: {baseline_csv}")
    if not model_csv.exists():
        raise FileNotFoundError(f"Model per-day metrics not found: {model_csv}")
    return baseline_csv, model_csv


def comparison_table(baseline_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    b = baseline_df.copy()
    m = model_df.copy()

    keep = ["day_norm", "day_plot"] + METRICS
    missing_b = [c for c in METRICS if c not in b.columns]
    missing_m = [c for c in METRICS if c not in m.columns]
    if missing_b:
        raise ValueError(f"Baseline is missing columns: {missing_b}")
    if missing_m:
        raise ValueError(f"Model is missing columns: {missing_m}")

    b = b[keep]
    m = m[["day_norm", "day_plot"] + METRICS]

    b = b.rename(columns={c: f"{c}_baseline" for c in METRICS})
    m = m.rename(columns={c: f"{c}_model" for c in METRICS})

    merged = pd.merge(b, m, on=["day_norm", "day_plot"], how="inner")

    for c in METRICS:
        merged[f"{c}_delta"] = merged[f"{c}_model"] - merged[f"{c}_baseline"]
        denom = merged[f"{c}_baseline"].replace(0, np.nan)
        merged[f"{c}_pct_change"] = (merged[f"{c}_delta"] / denom) * 100.0
        merged[f"{c}_pct_change"] = merged[f"{c}_pct_change"].fillna(0.0)

    return merged


def save_tables(comp: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "comparison_per_day.csv", index=False)

    rows = []
    for c in METRICS:
        rows.append(
            {
                "metric": c,
                "mean_baseline": comp[f"{c}_baseline"].mean(),
                "mean_model": comp[f"{c}_model"].mean(),
                "mean_delta": comp[f"{c}_delta"].mean(),
                "mean_pct_change": comp[f"{c}_pct_change"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "summary_deltas.csv", index=False)


def make_line_plots(comp: pd.DataFrame, out_dir: Path):
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    x = comp["day_plot"].tolist()

    for c in METRICS:
        yb = comp[f"{c}_baseline"].values
        ym = comp[f"{c}_model"].values
        plt.figure()
        _set_pub_style()
        plt.plot(x, yb, marker="o", label="baseline")
        plt.plot(x, ym, marker="o", label="model")
        plt.title(f"{c} by day")
        plt.xlabel("Day")
        plt.ylabel(c)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"line_{c}.png")
        plt.savefig(plot_dir / f"line_{c}.svg")
        plt.close()

    for c in METRICS:
        yd = comp[f"{c}_delta"].values
        plt.figure()
        _set_pub_style()
        plt.plot(x, yd, marker="o")
        plt.title(f"{c} (model - baseline) by day")
        plt.xlabel("Day")
        plt.ylabel(f"{c} delta")
        plt.tight_layout()
        plt.savefig(plot_dir / f"line_{c}_delta.png")
        plt.savefig(plot_dir / f"line_{c}_delta.svg")
        plt.close()


def _set_pub_style():
    """Matplotlib settings for publication-quality, single-figure charts.
    No explicit colors; use defaults. One chart per figure.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (6, 4),  # nice aspect ratio for papers
            "figure.dpi": 150,
            "savefig.dpi": 300,  # hi-res PNG output
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
        }
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare baseline vs model per-day metrics and produce line plots."
    )
    ap.add_argument(
        "--baseline_dir",
        default="results/baseline_routes_analysis",
        help="Folder containing baseline_routes_per_day_metrics.csv",
    )
    ap.add_argument(
        "--model_dir",
        default=None,
        help="Folder containing per_day_metrics.csv (e.g., results/<run>/analysis). If omitted, will look for the most recent results/*/analysis.",
    )
    ap.add_argument(
        "--baseline_csv",
        default=None,
        help="Explicit path to baseline per-day CSV (overrides --baseline_dir)",
    )
    ap.add_argument(
        "--model_csv",
        default=None,
        help="Explicit path to model per-day CSV (overrides --model_dir)",
    )
    ap.add_argument(
        "--out_dir",
        default="results/comparison",
        help="Where to save comparison CSVs and plots",
    )
    args = ap.parse_args()

    # Resolve model CSV
    model_csv = Path(args.model_csv) if args.model_csv else None
    if model_csv is None:
        if args.model_dir:
            model_csv = Path(args.model_dir) / "per_day_metrics.csv"
        else:
            results_root = Path("results")
            candidates = []
            if results_root.exists():
                for d in results_root.iterdir():
                    p = d / "analysis" / "per_day_metrics.csv"
                    if p.exists():
                        candidates.append(p)
            if not candidates:
                raise FileNotFoundError(
                    "Could not find model per_day_metrics.csv. Provide --model_dir or --model_csv."
                )
            model_csv = max(candidates, key=lambda p: p.stat().st_mtime)

    baseline_csv = (
        Path(args.baseline_csv)
        if args.baseline_csv
        else Path(args.baseline_dir) / "baseline_routes_per_day_metrics.csv"
    )

    out_dir = Path(args.out_dir)

    bdf = load_per_day(baseline_csv, "baseline")
    mdf = load_per_day(model_csv, "model")

    comp = comparison_table(bdf, mdf)
    save_tables(comp, out_dir)
    make_line_plots(comp, out_dir)

    print(f"[OK] Wrote comparison tables and plots to: {out_dir}")


if __name__ == "__main__":
    main()
