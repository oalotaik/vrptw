import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SHIFT_MIN = 540
DAYS = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu"]  # column names in baseline CSV


# def load_baseline(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     required = {
#         "Branch",
#         "CLIENT",
#         "Sat",
#         "Sun",
#         "Mon",
#         "Tue",
#         "Wed",
#         "Thu",
#         "Route",
#         "Class",
#         "time",
#     }
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"baseline file missing columns: {sorted(missing)}")
#     # normalize types
#     for d in DAYS:
#         df[d] = df[d].astype(str)
#     df["time"] = df["time"].astype(float)
#     # Keep only needed columns
#     return df


### New load_baseline starts here ###
def load_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    required = {
        "Branch",
        "CLIENT",
        "Sat",
        "Sun",
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Route",
        "Class",
        "time",
    }
    missing = required - set(df.columns)
    # Try case-insensitive rescue for day names if needed
    if missing:
        day_targets = {
            "sat": "Sat",
            "sun": "Sun",
            "mon": "Mon",
            "tue": "Tue",
            "wed": "Wed",
            "thu": "Thu",
        }
        for c in list(df.columns):
            key = c.strip().lower()
            if key in day_targets and day_targets[key] not in df.columns:
                df.rename(columns={c: day_targets[key]}, inplace=True)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"baseline file missing columns: {sorted(missing)}")

    # Coerce day flags to {0,1} integers safely
    for d in ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu"]:
        df[d] = (
            pd.to_numeric(df[d], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(lower=0, upper=1)
        )

    # Coerce service time to numeric minutes
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0)

    # Clean up text fields to avoid grouping surprises
    for col in ["Branch", "CLIENT", "Route", "Class"]:
        df[col] = df[col].astype(str).str.strip()

    return df


### New load_baseline ends here ###


def per_day_route_table(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Build a per-route table for a specific day using predefined Route IDs.
    One stop per Branch if that Branch has any client to be served that day in that Route.
    service_min (per Branch) = sum of 'time' over clients with day flag = 1.
    Then per-route aggregation:
      - stops_count = #branches in route with positive service
      - service_sum = sum(service_min over branches in the route)
      - route_length_min = service_sum + avg_travel_min * (stops_count - 1)  (0 to/from depot)
    Returns rows at (Route, Branch) level first; route-level metrics computed later.
    """
    tmp = df.copy()
    tmp["service_min"] = tmp["time"] * tmp[day]
    # aggregate to Branch within Route (sum over clients of same Branch & Route)
    by_branch = tmp.groupby(["Route", "Branch"], as_index=False)["service_min"].sum()
    # keep only positive service
    by_branch = by_branch[by_branch["service_min"] > 0].reset_index(drop=True)
    return by_branch


def build_per_route_metrics(
    by_branch: pd.DataFrame, day: str, avg_travel_min: float
) -> pd.DataFrame:
    """
    Aggregate (Route, Branch)-level to Route-level metrics for the day.
    """
    if by_branch.empty:
        return pd.DataFrame(
            columns=[
                "day",
                "route_id",
                "route_length_min",
                "stops_count",
                "overtime_min",
                "overtime_hours",
                "utilization",
            ]
        )
    agg = by_branch.groupby("Route", as_index=False).agg(
        stops_count=("Branch", "nunique"), service_sum=("service_min", "sum")
    )
    agg["route_length_min"] = agg["service_sum"] + np.maximum(
        agg["stops_count"] - 1, 0
    ) * float(avg_travel_min)
    agg["overtime_min"] = np.maximum(agg["route_length_min"] - SHIFT_MIN, 0.0)
    agg["overtime_hours"] = agg["overtime_min"] / 60.0
    agg["utilization"] = np.minimum(agg["route_length_min"] / SHIFT_MIN, 1.0)
    # tidy
    out = agg[
        [
            "Route",
            "stops_count",
            "route_length_min",
            "overtime_min",
            "overtime_hours",
            "utilization",
        ]
    ].copy()
    out.insert(0, "day", day)
    out = out.rename(columns={"Route": "route_id"})
    # enforce types
    out["stops_count"] = out["stops_count"].astype(int)
    return out


def agg_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": float(series.mean()) if not series.empty else 0.0,
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()) if not series.empty else 0.0,
        "p50": float(series.median()) if not series.empty else 0.0,
        "p90": float(series.quantile(0.90)) if not series.empty else 0.0,
        "max": float(series.max()) if not series.empty else 0.0,
    }


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0 or np.allclose(x, 0):
        return 0.0
    x.sort()
    n = x.size
    cum = np.cumsum(x, dtype=float)
    g = (n + 1 - 2 * (cum.sum() / cum[-1])) / n
    return float(g)


def analyze(
    baseline_csv: Path, avg_travel_min: float, out_dir: Path, make_plots: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_baseline(baseline_csv)

    per_route_all = []
    per_day_rows = []

    for d in DAYS:
        by_branch = per_day_route_table(df, d)
        per_route = build_per_route_metrics(by_branch, d, avg_travel_min=avg_travel_min)
        per_route_all.append(per_route)

        # day-level metrics
        stats_len = agg_stats(per_route["route_length_min"])
        stats_stops = agg_stats(per_route["stops_count"])
        stats_ot_h = agg_stats(per_route["overtime_hours"])
        per_day_rows.append(
            {
                "day": d,
                "num_routes": int(per_route.shape[0]),
                "total_visits": int(
                    by_branch["Branch"].nunique() if not by_branch.empty else 0
                ),
                "avg_route_length_min": stats_len["mean"],
                "std_route_length_min": stats_len["std"],
                "avg_stops_per_route": stats_stops["mean"],
                "std_stops_per_route": stats_stops["std"],
                "avg_overtime_hours": stats_ot_h["mean"],
                "std_overtime_hours": stats_ot_h["std"],
                "pct_routes_with_overtime": float(
                    (per_route["overtime_min"] > 0).mean() * 100.0
                )
                if not per_route.empty
                else 0.0,
                "max_overtime_hours": float(
                    per_route["overtime_hours"].max() if not per_route.empty else 0.0
                ),
                "p90_route_length_min": stats_len["p90"],
            }
        )

    per_route_all = (
        pd.concat(per_route_all, ignore_index=True)
        if per_route_all
        else pd.DataFrame(columns=["day"])
    )
    per_day = pd.DataFrame(per_day_rows)

    overall = {
        "avg_route_length_min": float(
            per_route_all["route_length_min"].mean() if not per_route_all.empty else 0.0
        ),
        "std_route_length_min": float(
            per_route_all["route_length_min"].std(ddof=1)
            if per_route_all.shape[0] > 1
            else 0.0
        ),
        "avg_stops_per_route": float(
            per_route_all["stops_count"].mean() if not per_route_all.empty else 0.0
        ),
        "std_stops_per_route": float(
            per_route_all["stops_count"].std(ddof=1)
            if per_route_all.shape[0] > 1
            else 0.0
        ),
        "mean_total_visits_per_day": float(
            per_day["total_visits"].mean() if not per_day.empty else 0.0
        ),
        "std_total_visits_per_day": float(
            per_day["total_visits"].std(ddof=1) if per_day.shape[0] > 1 else 0.0
        ),
        "mean_routes_per_day": float(
            per_day["num_routes"].mean() if not per_day.empty else 0.0
        ),
        "std_routes_per_day": float(
            per_day["num_routes"].std(ddof=1) if per_day.shape[0] > 1 else 0.0
        ),
        "avg_overtime_hours": float(
            per_route_all["overtime_hours"].mean() if not per_route_all.empty else 0.0
        ),
        "std_overtime_hours": float(
            per_route_all["overtime_hours"].std(ddof=1)
            if per_route_all.shape[0] > 1
            else 0.0
        ),
        # extras
        "pct_routes_with_overtime": float(
            (per_route_all["overtime_min"] > 0).mean() * 100.0
        )
        if not per_route_all.empty
        else 0.0,
        "max_overtime_hours": float(
            per_route_all["overtime_hours"].max() if not per_route_all.empty else 0.0
        ),
        "p90_route_length_min": float(
            per_route_all["route_length_min"].quantile(0.90)
            if not per_route_all.empty
            else 0.0
        ),
        "gini_route_length": gini(per_route_all["route_length_min"].to_numpy())
        if not per_route_all.empty
        else 0.0,
        "gini_stops": gini(per_route_all["stops_count"].to_numpy())
        if not per_route_all.empty
        else 0.0,
        "pct_routes_util_over_0_9": float(
            (per_route_all["utilization"] >= 0.9).mean() * 100.0
        )
        if not per_route_all.empty
        else 0.0,
    }
    overall_df = pd.DataFrame([overall])

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    per_route_all.to_csv(out_dir / "baseline_routes_per_route_metrics.csv", index=False)
    per_day.to_csv(out_dir / "baseline_routes_per_day_metrics.csv", index=False)
    overall_df.to_csv(out_dir / "baseline_routes_overall_metrics.csv", index=False)

    # Console summary
    print(f"[OK] Baseline (using predefined Route IDs) analysis saved under: {out_dir}")
    print("\nPer-day summary (key fields):")
    sel = [
        "day",
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
    print(per_day[sel].to_string(index=False))
    print("\nOverall summary:")
    print(overall_df.to_string(index=False))

    # Optional plots
    if make_plots and not per_route_all.empty:
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Histograms
        for col, fname, title, xlabel in [
            (
                "route_length_min",
                "hist_route_length_overall.png",
                "Route Length (Overall)",
                "Route length (min)",
            ),
            (
                "stops_count",
                "hist_stops_overall.png",
                "Stops per Route (Overall)",
                "Stops per route",
            ),
            (
                "overtime_hours",
                "hist_overtime_overall.png",
                "Overtime Hours (Overall)",
                "Overtime (hours)",
            ),
            (
                "utilization",
                "hist_utilization_overall.png",
                "Route Utilization (Overall)",
                "Utilization (0-1)",
            ),
        ]:
            series = per_route_all[col].dropna().values
            plt.figure()
            plt.hist(series, bins=20)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(plot_dir / fname)
            plt.close()

        # Avg Overtime Hours by Day (line)
        plt.figure()
        _set_pub_style()
        plt.plot(per_day["day"], per_day["avg_overtime_hours"], marker="o")
        plt.title("Avg Overtime Hours by Day (Baseline, Predefined Routes)")
        plt.xlabel("Day")
        plt.ylabel("Avg overtime (hours)")
        plt.tight_layout()
        plt.savefig(plot_dir / "line_avg_overtime_by_day.png")
        plt.savefig(plot_dir / "line_avg_overtime_by_day.svg")
        plt.close()

        # Number of Routes by Day (line)
        plt.figure()
        _set_pub_style()
        plt.plot(per_day["day"], per_day["num_routes"], marker="o")
        plt.title("Number of Routes by Day (Baseline, Predefined Routes)")
        plt.xlabel("Day")
        plt.ylabel("Routes")
        plt.tight_layout()
        plt.savefig(plot_dir / "line_routes_by_day.png")
        plt.savefig(plot_dir / "line_routes_by_day.svg")
        plt.close()

        # Total Visits by Day (line)
        plt.figure()
        _set_pub_style()
        plt.plot(per_day["day"], per_day["total_visits"], marker="o")
        plt.title("Total Visits by Day (Baseline, Predefined Routes)")
        plt.xlabel("Day")
        plt.ylabel("Visits")
        plt.tight_layout()
        plt.savefig(plot_dir / "line_total_visits_by_day.png")
        plt.savefig(plot_dir / "line_total_visits_by_day.svg")
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
        description="Analyze baseline schedule using predefined Route IDs in baseline_schedule_with_time.csv."
    )
    ap.add_argument(
        "--baseline_csv", required=True, help="Path to baseline_schedule_with_time.csv"
    )
    ap.add_argument(
        "--avg_travel_min",
        type=float,
        default=12.0,
        help="Assumed average travel time between branches (minutes). Default: 12",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output folder (default: <baseline_csv_dir>/analysis_baseline_routes)",
    )
    ap.add_argument("--make_plots", action="store_true", help="Generate PNG charts")
    args = ap.parse_args()

    baseline_csv = Path(args.baseline_csv)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else baseline_csv.parent / "analysis_baseline_routes"
    )

    analyze(
        baseline_csv,
        avg_travel_min=args.avg_travel_min,
        out_dir=out_dir,
        make_plots=args.make_plots,
    )


if __name__ == "__main__":
    main()
