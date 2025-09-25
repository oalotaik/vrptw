import argparse
from pathlib import Path
import pandas as pd


def normalize_series(s: pd.Series) -> pd.Series:
    """Trim whitespace and unify case for robust joins."""
    return s.astype(str).str.strip()


def main():
    ap = argparse.ArgumentParser(
        description="Add 'time' column to baseline schedule using ser_time lookup."
    )
    ap.add_argument(
        "--schedule",
        required=True,
        default="../data/processed/baseline_schedule.csv",
        help="Path to schedule CSV with columns: Branch, CLIENT, Sat..Thu, Route, Class",
    )
    ap.add_argument(
        "--ser_time",
        required=True,
        default="../data/processed/ser_time.csv",
        help="Path to ser_time CSV with columns: Class, <client1>, <client2>, ...",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <schedule_basename>_with_time.csv next to input)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, fail on any unmatched (Class, CLIENT) pair; otherwise warn.",
    )
    args = ap.parse_args()

    schedule_path = Path(args.schedule)
    ser_time_path = Path(args.ser_time)

    # Load inputs
    sched = pd.read_csv(schedule_path)
    svc = pd.read_csv(ser_time_path)

    # Basic validations
    required_sched_cols = {
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
    }
    missing_sched = required_sched_cols - set(sched.columns)
    if missing_sched:
        raise ValueError(f"schedule is missing columns: {sorted(missing_sched)}")

    if "Class" not in svc.columns:
        raise ValueError(
            "ser_time must have a 'Class' column plus one column per client."
        )

    # Normalize keys
    sched["CLIENT_norm"] = normalize_series(sched["CLIENT"])
    sched["Class_norm"] = normalize_series(sched["Class"])
    svc["Class_norm"] = normalize_series(svc["Class"])

    # Convert ser_time from wide (Class, client1, client2, ...) to long (Class_norm, CLIENT_norm, time)
    svc_long = svc.melt(
        id_vars=["Class", "Class_norm"], var_name="CLIENT", value_name="time"
    )
    svc_long["CLIENT_norm"] = normalize_series(svc_long["CLIENT"])

    # Keep only the normalized lookup columns and time
    lookup = svc_long[["Class_norm", "CLIENT_norm", "time"]].copy()

    # Merge onto schedule
    merged = sched.merge(lookup, on=["Class_norm", "CLIENT_norm"], how="left")

    # Report unmatched
    unmatched = merged[merged["time"].isna()][["Branch", "CLIENT", "Class"]].copy()
    if not unmatched.empty:
        msg = f"WARNING: {len(unmatched)} rows had no matching time in ser_time (by Class & CLIENT)."
        if args.strict:
            # Save debug file then raise
            dbg = schedule_path.with_name(schedule_path.stem + "_unmatched.csv")
            unmatched.to_csv(dbg, index=False)
            raise RuntimeError(msg + f" Details saved to: {dbg}")
        else:
            print(msg + " Sample:")
            print(unmatched.head(10).to_string(index=False))

    # Finalize: drop helper cols, keep original columns + new 'time'
    merged = merged.drop(columns=["CLIENT_norm", "Class_norm"])

    # Write output
    out_path = (
        Path(args.out)
        if args.out
        else schedule_path.with_name(schedule_path.stem + "_with_time.csv")
    )
    merged.to_csv(out_path, index=False)
    print(f"[OK] Wrote enriched schedule: {out_path}")
    if not unmatched.empty and not args.strict:
        dbg = out_path.with_name(
            out_path.stem.replace("_with_time", "") + "_unmatched.csv"
        )
        unmatched.to_csv(dbg, index=False)
        print(f"[INFO] Unmatched rows saved to: {dbg}")


if __name__ == "__main__":
    main()
