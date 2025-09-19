import argparse
import json
import sys
import time
import threading
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

DAYS = ["SAT", "SUN", "MON", "TUE", "WED", "THU"]


# -------- Utilities to load data --------
def load_schedule(schedule_path: str) -> pd.DataFrame:
    df = pd.read_csv(schedule_path)
    assert "location" in df.columns, "schedule.csv must have a 'location' column"
    for d in DAYS:
        assert d in df.columns, f"schedule.csv missing day column: {d}"
    return df


def load_time_matrix(time_matrix_path: str) -> pd.DataFrame:
    tm = pd.read_csv(time_matrix_path, index_col=0)
    assert tm.index.tolist() == tm.columns.tolist(), (
        "time_matrix.csv must be square with same ordered location labels on index and columns"
    )
    return tm


def load_time_windows(
    path: Optional[str], locations: List[str]
) -> Dict[str, Tuple[int, int]]:
    if path is None or not Path(path).exists():
        nine_h = 9 * 60
        return {loc: (0, nine_h) for loc in locations}
    tw = pd.read_csv(path)
    assert {"location", "start", "end"}.issubset(tw.columns), (
        "time_windows.csv must have columns: location,start,end"
    )
    m = dict(zip(tw["location"], zip(tw["start"], tw["end"])))
    nine_h = 9 * 60
    return {loc: m.get(loc, (0, nine_h)) for loc in locations}


def build_day_instance(
    day: str,
    schedule_df: pd.DataFrame,
    time_matrix_df: pd.DataFrame,
    time_windows_map: Dict[str, Tuple[int, int]],
    max_vehicles: Optional[int] = None,
    vehicle_fixed_cost: int = 10**6,
):
    assert day in DAYS, f"day must be one of {DAYS}"
    todays = schedule_df.loc[schedule_df[day] > 0, ["location", day]].copy()
    todays.rename(columns={day: "service"}, inplace=True)
    locations_today = todays["location"].tolist()
    if len(locations_today) == 0:
        raise ValueError(f"No locations scheduled for {day}")
    tm_sub = time_matrix_df.loc[locations_today, locations_today].copy()
    N = len(locations_today)
    data = {
        "locations": locations_today,
        "num_customers": N,
        "service_times": todays["service"].astype(int).tolist(),
        "time_windows": [time_windows_map[loc] for loc in locations_today],
        "time_matrix": tm_sub.values.astype(int).tolist(),
        "shift_minutes": 12 * 60,
        "vehicle_fixed_cost": vehicle_fixed_cost,
        "max_vehicles": (max_vehicles if max_vehicles is not None else N),
        "day": day,
    }
    return data


# -------- Solver --------
def solve_vrptw(data: dict, time_limit_s: int = 30) -> dict:
    N = data["num_customers"]
    depot = N
    manager = pywrapcp.RoutingIndexManager(N + 1, data["max_vehicles"], depot)
    routing = pywrapcp.RoutingModel(manager)

    locs = data["locations"]
    service_times = data["service_times"]
    time_matrix = data["time_matrix"]
    time_windows = data["time_windows"]
    shift = data["shift_minutes"]

    def travel_time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        if i == depot or j == depot:
            return 0
        return int(time_matrix[i][j])

    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        travel = 0 if (i == depot or j == depot) else time_matrix[i][j]
        service = 0 if i == depot else service_times[i]
        return int(travel + service)

    travel_eval = routing.RegisterTransitCallback(travel_time_cb)
    time_eval = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_eval)

    for v in range(data["max_vehicles"]):
        routing.SetFixedCostOfVehicle(int(data["vehicle_fixed_cost"]), v)

    routing.AddDimension(
        time_eval,
        shift,
        shift,
        True,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    # Encourage compact routes (secondary objective; keep small vs vehicle fixed cost):
    # time_dim.SetGlobalSpanCostCoefficient(100)

    for i in range(N):
        idx = manager.NodeToIndex(i)
        start, end = time_windows[i]
        time_dim.CumulVar(idx).SetRange(int(start), int(end))

    depot_idx = manager.NodeToIndex(depot)
    time_dim.CumulVar(depot_idx).SetRange(0, 0)

    for v in range(data["max_vehicles"]):
        end = routing.End(v)
        time_dim.CumulVar(end).SetMax(shift)

    params = pywrapcp.DefaultRoutingSearchParameters()

    # params.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    # )

    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {
            "feasible": False,
            "day": data["day"],
            "used_vehicles": 0,
            "message": "No feasible solution found.",
        }

    routes = []
    used_vehicles = 0
    total_travel = 0
    for v in range(data["max_vehicles"]):
        index = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
        used_vehicles += 1
        r = {"vehicle": v, "stops": []}
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            t = solution.Value(time_dim.CumulVar(index))
            r["stops"].append(
                {"node": "DEPOT" if node == depot else locs[node], "start_time": int(t)}
            )
            nxt = solution.Value(routing.NextVar(index))
            total_travel += routing.GetArcCostForVehicle(index, nxt, v)
            index = nxt
        node = manager.IndexToNode(index)
        t = solution.Value(time_dim.CumulVar(index))
        r["stops"].append(
            {"node": "DEPOT" if node == depot else locs[node], "start_time": int(t)}
        )
        routes.append(r)

    return {
        "feasible": True,
        "day": data["day"],
        "used_vehicles": used_vehicles,
        "total_travel_time": int(total_travel),
        "routes": routes,
    }


# -------- Output saving into results/ (sibling of src/) --------
def save_outputs(result: dict, out_prefix: str):
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{out_prefix}.json"
    csv_path = results_dir / f"{out_prefix}.csv"

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
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(json_path), str(csv_path)


# -------- Console progress (time countdown) --------
class TimeProgressBar:
    def __init__(
        self,
        total_seconds: int,
        label: str = "solve",
        update_interval: float = 0.2,
        newline_on_stop: bool = True,
    ):
        self.total = max(int(total_seconds), 0)
        self.label = label
        self.update_interval = update_interval
        self._stop = threading.Event()
        self._thread = None
        self._newline_on_stop = bool(newline_on_stop)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # Clear the line
        sys.stdout.write(
            "" + " " * (shutil.get_terminal_size((80, 20)).columns - 1) + ""
        )
        sys.stdout.flush()
        if self._newline_on_stop:
            sys.stdout.write("")
            sys.stdout.flush()

    def _run(self):
        start = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - start
            if self.total > 0:
                if elapsed > self.total:
                    elapsed = self.total
                pct = elapsed / self.total
                remaining = self.total - elapsed
            else:
                pct = 1.0
                remaining = 0
            cols = shutil.get_terminal_size((80, 20)).columns
            bar_width = max(10, min(50, cols - 40))
            filled = int(bar_width * pct)
            bar = "#" * filled + "." * (bar_width - filled)
            mm = int(remaining // 60)
            ss = int(remaining % 60)
            sys.stdout.write(
                f"\r{self.label} [{bar}] {int(pct * 100):3d}%  {mm:02d}:{ss:02d} left"
            )
            sys.stdout.flush()
            if self.total and elapsed >= self.total:
                break
            time.sleep(self.update_interval)


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--day", required=True, choices=DAYS, help="Day to solve (SAT..THU)"
    )
    ap.add_argument("--schedule", default="data\\processed\\schedule.csv")
    ap.add_argument("--time_matrix", default="data\\processed\\time_matrix.csv")
    ap.add_argument("--time_windows", default=None)
    ap.add_argument("--max_vehicles", type=int, default=10)
    ap.add_argument("--vehicle_fixed_cost", type=int, default=10**6)
    ap.add_argument("--time_limit", type=int, default=30)
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    schedule_df = load_schedule(args.schedule)
    time_matrix_df = load_time_matrix(args.time_matrix)
    schedule_names = set(schedule_df["location"])
    matrix_names = set(time_matrix_df.index.tolist())
    missing_in_matrix = schedule_names - matrix_names
    if missing_in_matrix:
        raise ValueError(
            f"These schedule locations are missing from time_matrix: {sorted(missing_in_matrix)[:10]} ..."
        )

    tw_map = load_time_windows(args.time_windows, time_matrix_df.index.tolist())

    data = build_day_instance(
        day=args.day,
        schedule_df=schedule_df,
        time_matrix_df=time_matrix_df,
        time_windows_map=tw_map,
        max_vehicles=args.max_vehicles,
        vehicle_fixed_cost=args.vehicle_fixed_cost,
    )

    # Countdown bar during solve
    bar = TimeProgressBar(args.time_limit, label=f"{args.day} solve").start()
    try:
        result = solve_vrptw(data, time_limit_s=args.time_limit)
    finally:
        bar.stop()

    if args.out_prefix is None:
        out_prefix = f"routes_{args.day}"
    else:
        out_prefix = args.out_prefix
    save_outputs(result, out_prefix)

    # Minimal final print
    print(
        f"feasible={result['feasible']} day={result['day']} used_vehicles={result.get('used_vehicles', 0)}"
    )


if __name__ == "__main__":
    main()
