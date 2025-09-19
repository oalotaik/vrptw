import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

DAYS = ["SAT", "SUN", "MON", "TUE", "WED", "THU"]


def load_schedule(schedule_path: str) -> pd.DataFrame:
    df = pd.read_csv(schedule_path)
    assert "location" in df.columns, "schedule.csv must have a 'location' column"
    for d in DAYS:
        assert d in df.columns, f"schedule.csv missing day column: {d}"
    return df


def load_time_matrix(time_matrix_path: str) -> pd.DataFrame:
    tm = pd.read_csv(time_matrix_path, index_col=0)
    # Expect square matrix with identical index/columns = location names
    assert tm.index.tolist() == tm.columns.tolist(), (
        "time_matrix.csv must be square with same ordered location labels on index and columns"
    )
    return tm


def load_time_windows(
    path: Optional[str], locations: List[str]
) -> Dict[str, Tuple[int, int]]:
    if path is None or not Path(path).exists():
        # Wide windows by default: [0, 9h]
        nine_h = 9 * 60
        return {loc: (0, nine_h) for loc in locations}
    tw = pd.read_csv(path)
    assert {"location", "start", "end"}.issubset(tw.columns), (
        "time_windows.csv must have columns: location,start,end"
    )
    m = dict(zip(tw["location"], zip(tw["start"], tw["end"])))
    nine_h = 9 * 60
    # Any missing => default wide
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
    # Locations visited today (service time > 0)
    todays = schedule_df.loc[schedule_df[day] > 0, ["location", day]].copy()
    todays.rename(columns={day: "service"}, inplace=True)
    locations_today = todays["location"].tolist()

    if len(locations_today) == 0:
        raise ValueError(f"No locations scheduled for {day}")

    # Subset the full time matrix to today's locations (preserve order)
    tm_sub = time_matrix_df.loc[locations_today, locations_today].copy()

    # Build data dict for OR-Tools
    N = len(locations_today)
    data = {
        "locations": locations_today,
        "num_customers": N,
        "service_times": todays["service"].astype(int).tolist(),
        "time_windows": [
            time_windows_map[loc] for loc in locations_today
        ],  # list aligned to locations_today
        "time_matrix": tm_sub.values.astype(int).tolist(),
        "shift_minutes": 9 * 60,
        "vehicle_fixed_cost": vehicle_fixed_cost,
        "max_vehicles": (
            max_vehicles if max_vehicles is not None else N
        ),  # upper bound: each stop could be its own route
        "day": day,
    }
    return data


def solve_vrptw(data: dict, time_limit_s: int = 60) -> dict:
    N = data["num_customers"]
    depot = N  # virtual depot after customers
    manager = pywrapcp.RoutingIndexManager(N + 1, data["max_vehicles"], depot)
    routing = pywrapcp.RoutingModel(manager)

    # Index mapping for readability
    locs = data["locations"]
    service_times = data["service_times"]
    time_matrix = data["time_matrix"]
    time_windows = data["time_windows"]
    shift = data["shift_minutes"]

    # Transit callbacks
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

    # Vehicle fixed cost to minimize #vehicles
    for v in range(data["max_vehicles"]):
        routing.SetFixedCostOfVehicle(int(data["vehicle_fixed_cost"]), v)

    # Time dimension
    routing.AddDimension(
        time_eval,
        shift,  # allow waiting up to shift
        shift,  # latest time
        True,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Apply time windows to customer nodes
    for i in range(N):
        idx = manager.NodeToIndex(i)
        start, end = time_windows[i]
        time_dim.CumulVar(idx).SetRange(int(start), int(end))

    # Depot starts at 0
    depot_idx = manager.NodeToIndex(depot)
    time_dim.CumulVar(depot_idx).SetRange(0, 0)

    # End times capped by shift
    for v in range(data["max_vehicles"]):
        end = routing.End(v)
        time_dim.CumulVar(end).SetMax(shift)

    # Search params
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {"feasible": False, "message": "No feasible solution found."}

    # Extract routes
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
                {
                    "node": "DEPOT" if node == depot else locs[node],
                    "start_time": int(t),
                }
            )
            nxt = solution.Value(routing.NextVar(index))
            total_travel += routing.GetArcCostForVehicle(index, nxt, v)
            index = nxt
        # end node
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


def save_outputs(result: dict, out_prefix: str):
    # JSON
    json_path = f"{out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # CSV (one row per stop)
    rows = []
    for r in result.get("routes", []):
        v = r["vehicle"]
        for k, s in enumerate(r["stops"]):
            rows.append(
                {
                    "day": result["day"],
                    "vehicle": v,
                    "seq": k,
                    "location": s["node"],
                    "start_time_min": s["start_time"],
                }
            )
    import pandas as pd

    pd.DataFrame(rows).to_csv(f"{out_prefix}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--day", required=True, choices=DAYS, help="Day to solve (SAT..THU)"
    )
    ap.add_argument("--schedule", default="../data/processed/schedule.csv")
    ap.add_argument("--time_matrix", default="../data/processed/time_matrix.csv")
    ap.add_argument(
        "--time_windows",
        default=None,
        help="Optional CSV with columns: location,start,end (minutes)",
    )
    ap.add_argument(
        "--max_vehicles",
        type=int,
        default=15,
        help="Upper bound on vehicles (default: #stops)",
    )
    ap.add_argument("--vehicle_fixed_cost", type=int, default=10**6)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument(
        "--out_prefix", default=None, help="Prefix for outputs; default: routes_{DAY}"
    )
    args = ap.parse_args()

    schedule_df = load_schedule(args.schedule)
    time_matrix_df = load_time_matrix(args.time_matrix)

    # Validate names alignment
    schedule_names = set(schedule_df["location"])
    matrix_names = set(time_matrix_df.index.tolist())
    missing_in_matrix = schedule_names - matrix_names
    if missing_in_matrix:
        raise ValueError(
            f"These schedule locations are missing from time_matrix: {sorted(missing_in_matrix)[:10]} ..."
        )

    # Build time windows map (global, per location)
    # Note: if you have day-specific windows, you can provide separate files or extend loader logic.
    tw_map = load_time_windows(args.time_windows, time_matrix_df.index.tolist())

    data = build_day_instance(
        day=args.day,
        schedule_df=schedule_df,
        time_matrix_df=time_matrix_df,
        time_windows_map=tw_map,
        max_vehicles=args.max_vehicles,
        vehicle_fixed_cost=args.vehicle_fixed_cost,
    )
    result = solve_vrptw(data, time_limit_s=args.time_limit)

    if args.out_prefix is None:
        out_prefix = f"routes_{args.day}"
    else:
        out_prefix = args.out_prefix
    save_outputs(result, out_prefix)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
