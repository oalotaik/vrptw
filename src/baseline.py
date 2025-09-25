"""
this script is for processing baseline schedule to use it as baseline
for comparing the resutls of the proposed model.
"""

import pandas as pd


loc_classes = pd.read_excel(
    "../data/raw/all-data-raw.xlsx", sheet_name="Location", index_col="branch"
)
loc_classes.drop(columns=["long", "lat"], inplace=True)

schedule = pd.read_excel("../data/raw/all-data-raw.xlsx", sheet_name="Schedule")
schedule = schedule.merge(loc_classes, left_on="Branch", right_index=True)
schedule.drop(
    columns=["Region", "City", "Retailer", "Weekly Visit", "Supervisor"], inplace=True
)

ser_time = pd.read_excel("../data/raw/all-data-raw.xlsx", sheet_name="Time")
schedule.to_csv("../data/processed/baseline_schedule.csv", index=False)
ser_time.to_csv("../data/processed/ser_time.csv", index=False)
