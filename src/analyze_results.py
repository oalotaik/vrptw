import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SHIFT_MIN = 540  # 9 hours

DAYS = ["SAT", "SUN", "MON", "TUE", "WED", "THU"]


def find_latest_results_dir(results_root: Path) -> Path:
    """Pick the most recent subfolder under results/ by modification time."""
    subdirs = [p for p in results_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found under {results_root}")
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    return latest


def load_day_file(dirpath: Path, day: str) -> pd.DataFrame:
    f = dirpath / f"routes_{day}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")
    df = pd.read_csv(f)
    expected_cols = {"day", "vehicle", "seq", "location", "start_time_min"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{f} missing columns: {sorted(missing)}")
    # Ensure types
    df["vehicle"] = df["vehicle"].astype(int)
    df["seq"] = df["seq"].astype(int)
    df["start_time_min"] = df["start_time_min"].astype(int)
    return df


def per_route_metrics(day_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-route metrics for a day's dataframe.
    A route == one vehicle's path for that day.
    - route_length_min: end time at final DEPOT row (or max time as fallback)
    - stops_count: count of non-DEPOT locations
    - overtime_min/hours
    - utilization = min(route_length/SHIFT_MIN, 1.0)
    """
    # end time: last depot or max start_time_min as fallback
    end_times = (
        day_df[day_df["location"] == "DEPOT"]
        .groupby("vehicle", as_index=False)["start_time_min"]
        .max()
        .rename(columns={"start_time_min": "route_length_min"})
    )
    # fallback if no DEPOT rows (shouldn't happen): use max per vehicle
    if end_times.empty or len(end_times) != day_df["vehicle"].nunique():
        max_times = day_df.groupby("vehicle", as_index=False)["start_time_min"].max()
        max_times = max_times.rename(columns={"start_time_min": "route_length_min"})
        # prefer depot timing when present
        end_times = (
            max_times
            if end_times.empty
            else pd.concat([end_times, max_times])
            .groupby("vehicle", as_index=False)["route_length_min"]
            .max()
        )

    # stops: count of rows excluding DEPOT
    stops = (
        day_df[day_df["location"] != "DEPOT"]
        .groupby("vehicle", as_index=False)
        .size()
        .rename(columns={"size": "stops_count"})
    )

    routes = pd.merge(end_times, stops, on="vehicle", how="left").fillna(
        {"stops_count": 0}
    )
    routes["route_length_min"] = routes["route_length_min"].astype(float)
    routes["overtime_min"] = np.maximum(routes["route_length_min"] - SHIFT_MIN, 0.0)
    routes["overtime_hours"] = routes["overtime_min"] / 60.0
    routes["utilization"] = np.minimum(routes["route_length_min"] / SHIFT_MIN, 1.0)
    return routes


def agg_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": float(series.mean()) if not series.empty else 0.0,
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()) if not series.empty else 0.0,
        "p50": float(series.median()) if not series.empty else 0.0,
        "p90": float(series.quantile(0.90)) if not series.empty else 0.0,
        "max": float(series.max()) if not series.empty else 0.0,
    }


def analyze_day(
    day: str, df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    routes = per_route_metrics(df)
    stats = {
        "route_length_min": agg_stats(routes["route_length_min"]),
        "stops_count": agg_stats(routes["stops_count"]),
        "overtime_min": agg_stats(routes["overtime_min"]),
        "overtime_hours": agg_stats(routes["overtime_hours"]),
        "utilization": agg_stats(routes["utilization"]),
    }
    # key day-level outputs requested
    per_day_summary = {
        "day": day,
        "num_routes": int(routes.shape[0]),
        "total_visits": int((df["location"] != "DEPOT").sum()),
        "avg_route_length_min": stats["route_length_min"]["mean"],
        "std_route_length_min": stats["route_length_min"]["std"],
        "avg_stops_per_route": stats["stops_count"]["mean"],
        "std_stops_per_route": stats["stops_count"]["std"],
        "avg_overtime_hours": stats["overtime_hours"]["mean"],
        "std_overtime_hours": stats["overtime_hours"]["std"],
        "pct_routes_with_overtime": float((routes["overtime_min"] > 0).mean() * 100.0),
        "max_overtime_hours": stats["overtime_hours"]["max"],
        "p90_route_length_min": stats["route_length_min"]["p90"],
    }
    return routes, per_day_summary


def analyze_week(dirpath: Path, make_plots_flag: bool = False) -> None:
    # Collect per-route for all days and per-day summaries
    per_route_all = []
    per_day_rows = []
    for day in DAYS:
        df = load_day_file(dirpath, day)
        routes, day_row = analyze_day(day, df)
        routes.insert(0, "day", day)
        per_route_all.append(routes)
        per_day_rows.append(day_row)

    per_route_all = pd.concat(per_route_all, ignore_index=True)
    per_day = pd.DataFrame(per_day_rows)

    # Overall (across 6 days) aggregates
    overall = {
        # 1) average route length + variation (std)
        "avg_route_length_min": per_route_all["route_length_min"].mean(),
        "std_route_length_min": per_route_all["route_length_min"].std(ddof=1),
        # 2) avg #stops per route + variation
        "avg_stops_per_route": per_route_all["stops_count"].mean(),
        "std_stops_per_route": per_route_all["stops_count"].std(ddof=1),
        # 3) total visits per day + variation (std across days)
        "mean_total_visits_per_day": per_day["total_visits"].mean(),
        "std_total_visits_per_day": per_day["total_visits"].std(ddof=1),
        # 4) number of routes (per day only) -> we still report mean/std across days for convenience
        "mean_routes_per_day": per_day["num_routes"].mean(),
        "std_routes_per_day": per_day["num_routes"].std(ddof=1),
        # 5) average overtime hours + variation (across routes)
        "avg_overtime_hours": per_route_all["overtime_hours"].mean(),
        "std_overtime_hours": per_route_all["overtime_hours"].std(ddof=1),
        # Extra useful KPIs
        "pct_routes_with_overtime": float(
            (per_route_all["overtime_min"] > 0).mean() * 100.0
        ),
        "max_overtime_hours": per_route_all["overtime_hours"].max(),
        "p90_route_length_min": per_route_all["route_length_min"].quantile(0.90),
        # Balance indicators
        "gini_route_length": gini(per_route_all["route_length_min"].to_numpy()),
        "gini_stops": gini(per_route_all["stops_count"].to_numpy()),
    }
    overall_df = pd.DataFrame([overall])

    # Save next to dirpath as analysis files
    out_dir = dirpath / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_route_all.to_csv(out_dir / "per_route_metrics.csv", index=False)
    per_day.to_csv(out_dir / "per_day_metrics.csv", index=False)
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)

    # Optional plots
    if make_plots_flag:
        make_plots(out_dir, per_route_all, per_day)

    # Also print a concise console summary
    print(f"[OK] Analysis saved under: {out_dir}")
    print("\nPer-day summary (key fields):")
    print(
        per_day[
            [
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
        ].to_string(index=False)
    )
    print("\nOverall summary:")
    print(overall_df.to_string(index=False))


def gini(x: np.ndarray) -> float:
    """Gini coefficient (0=perfect equality, 1=max inequality)."""
    x = x.astype(float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x, dtype=float)
    g = (n + 1 - 2 * (cum.sum() / cum[-1])) / n
    return float(g)


def make_plots(
    out_dir: Path, per_route_all: pd.DataFrame, per_day: pd.DataFrame
) -> None:
    """Create basic charts and save PNGs into out_dir.
    Charts (each one figure):
      - Histogram: route_length_min (overall)
      - Histogram: stops_count (overall)
      - Histogram: overtime_hours (overall)
      - Histogram: utilization (overall)
      - Bar: avg overtime hours by day
      - Bar: num routes by day
      - Bar: total visits by day
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overall histograms
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
        plt.savefig(out_dir / fname)
        plt.close()

    # Per-day bars
    # Avg overtime hours by day
    plt.figure()
    plt.bar(per_day["day"], per_day["avg_overtime_hours"])
    plt.title("Avg Overtime Hours by Day")
    plt.xlabel("Day")
    plt.ylabel("Avg overtime (hours)")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_avg_overtime_by_day.png")
    plt.close()

    # Num routes by day
    plt.figure()
    plt.bar(per_day["day"], per_day["num_routes"])
    plt.title("Number of Routes by Day")
    plt.xlabel("Day")
    plt.ylabel("Routes")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_routes_by_day.png")
    plt.close()

    # Total visits by day
    plt.figure()
    plt.bar(per_day["day"], per_day["total_visits"])
    plt.title("Total Visits by Day")
    plt.xlabel("Day")
    plt.ylabel("Visits")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_total_visits_by_day.png")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Analyze VRPTW routing outputs for six days."
    )
    ap.add_argument(
        "--results_root",
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Path to the 'results' folder (sibling of src). Defaults to ../results from this script.",
    )
    ap.add_argument(
        "--dir",
        default=None,
        help="Optional: specific results subfolder name under --results_root. If omitted, pick the most recent.",
    )
    ap.add_argument(
        "--make_plots",
        action="store_true",
        help="Generate PNG charts in the analysis folder",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    if args.dir:
        dirpath = results_root / args.dir
        if not dirpath.exists():
            raise FileNotFoundError(f"Specified folder not found: {dirpath}")
    else:
        dirpath = find_latest_results_dir(results_root)

    analyze_week(dirpath, make_plots_flag=args.make_plots)


if __name__ == "__main__":
    main()
