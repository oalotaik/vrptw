import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict

# Expect this to live next to vrptw_daily_solver.py inside src/
import vrptw_daily_solver as daily

DAYS = ["SAT", "SUN", "MON", "TUE", "WED", "THU"]


def render_weekly_bar(done: int, total: int, label: str = "weekly"):
    """Render/refresh the weekly bar on its own line (above the daily line)."""
    import shutil, sys

    cols = shutil.get_terminal_size((80, 20)).columns
    bar_width = max(10, min(50, cols - 40))
    pct = 0.0 if total == 0 else (done / total)
    filled = int(bar_width * pct)
    bar = "#" * filled + "." * (bar_width - filled)
    # Move cursor up one line, clear it, print weekly, then return to daily line
    sys.stdout.write("\x1b[F")  # cursor up one line
    sys.stdout.write("\x1b[2K")  # clear entire line
    sys.stdout.write(f"{label} [{bar}] {int(pct * 100):3d}%  {done}/{total} days\n")
    sys.stdout.flush()


def save_outputs_weekly(result: Dict, out_dir: Path, out_prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{out_prefix}.json"
    csv_path = out_dir / f"{out_prefix}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    rows = []
    for r in result.get("routes", []):
        v = r["vehicle"]
        for k, s in enumerate(r["stops"]):
            rows.append(
                {
                    "day": result.get("day"),
                    "vehicle": v,
                    "seq": k,
                    "location": s["node"],
                    "start_time_min": s["start_time"],
                }
            )
    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(json_path), str(csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--schedule",
        default="data\\processed\\schedule.csv",
        help="Path to weekly schedule CSV",
    )
    ap.add_argument(
        "--time_matrix",
        default="data\\processed\\time_matrix.csv",
        help="Path to 61x61 time matrix CSV",
    )
    ap.add_argument(
        "--time_windows",
        default=None,
        help="Optional time windows CSV (location,start,end in minutes)",
    )
    ap.add_argument(
        "--max_vehicles",
        type=int,
        default=10,
        help="Upper bound per day (default: #stops that day)",
    )
    ap.add_argument(
        "--vehicle_fixed_cost",
        type=int,
        default=10**6,
        help="Vehicle activation penalty",
    )
    ap.add_argument(
        "--time_limit", type=int, default=30, help="Per-day solver time limit (seconds)"
    )
    ap.add_argument(
        "--out_dir_name",
        default=None,
        help="Subfolder name under results/ (default: weekly_run_YYYYmmdd_HHMMSS)",
    )
    ap.add_argument(
        "--show_day_bars",
        action="store_true",
        default=True,
        help="Show per-day countdown bars (default: on)",
    )
    args = ap.parse_args()

    schedule_df = daily.load_schedule(args.schedule)
    matrix_df = daily.load_time_matrix(args.time_matrix)

    schedule_names = set(schedule_df["location"])
    matrix_names = set(matrix_df.index.tolist())
    missing = schedule_names - matrix_names
    if missing:
        raise ValueError(
            f"These schedule locations are missing from time_matrix: {sorted(missing)[:10]} ..."
        )

    tw_map = daily.load_time_windows(args.time_windows, matrix_df.index.tolist())

    base_results = Path(__file__).resolve().parent.parent / "results"
    if args.out_dir_name:
        sub = args.out_dir_name
    else:
        import datetime as _dt

        sub = "weekly_run_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_results / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize two-line layout: print placeholder weekly bar + blank daily line
    total_days = len(DAYS)
    sys.stdout.write("\n")  # daily line placeholder
    sys.stdout.flush()
    # First weekly render
    render_weekly_bar(0, total_days, label="weekly solve")

    summary: List[Dict] = []
    done_days = 0

    for day in DAYS:
        data = daily.build_day_instance(
            day=day,
            schedule_df=schedule_df,
            time_matrix_df=matrix_df,
            time_windows_map=tw_map,
            max_vehicles=args.max_vehicles,
            vehicle_fixed_cost=args.vehicle_fixed_cost,
        )

        # Per-day countdown bar printed on the bottom line only
        bar_day = None
        if args.show_day_bars:
            bar_day = daily.TimeProgressBar(
                args.time_limit, label=f"{day} solve", newline_on_stop=False
            ).start()
        try:
            result = daily.solve_vrptw(data, time_limit_s=args.time_limit)
        finally:
            if bar_day:
                bar_day.stop()

        save_outputs_weekly(result, out_dir=out_dir, out_prefix=f"routes_{day}")

        summary.append(
            {
                "feasible": bool(result.get("feasible", False)),
                "day": day,
                "used_vehicles": int(result.get("used_vehicles", 0)),
            }
        )

        # Update weekly (top line) once per day, leave daily line for the next day
        done_days += 1
        render_weekly_bar(done_days, total_days, label="weekly solve")

    # After finishing, print a final newline so following prints don't overwrite bars
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Write summary files
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feasible", "day", "used_vehicles"])
        w.writeheader()
        w.writerows(summary)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Minimal console lines
    for row in summary:
        print(
            f"feasible={row['feasible']} day={row['day']} used_vehicles={row['used_vehicles']}"
        )
    # print(f"[OK] Weekly outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
