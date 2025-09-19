import pandas as pd

data = pd.read_csv("../data/raw/min_clients_service_time_schedule.csv")
data = data.groupby(["location"])[["SAT", "SUN", "MON", "TUE", "WED", "THU"]].sum()
data.to_csv("../data/processed/schedule.csv", index=True)
