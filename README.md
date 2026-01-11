# VRPTW: Weekly Schedule with Time-Delay Constraints

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Optimization](https://img.shields.io/badge/Optimization-VRPTW-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Overview

This repository contains a Python-based modeling approach for a **unique variant of the Vehicle Routing Problem with Time Windows (VRPTW)**. Unlike standard VRPTW formulations, this model addresses complex scheduling requirements over a **weekly horizon**, incorporating service times, selective time windows, and critical time-delay constraints.

The core objective is to optimize routing while strictly enforcing **time gaps between consecutive visits**, making it highly applicable to scenarios like recurring maintenance, healthcare logistics, or security patrolling where "visit spacing" is mandatory.

## 🚀 Key Features

* **Weekly Scheduling Horizon:** Optimization is performed over a multi-day (7-day) period rather than a single day.
* **Time-Delay Constraints:** Enforces a minimum and/or maximum time gap between consecutive visits to the same node or specific node pairs.
* **Selective Time Windows:** Handles nodes that have specific availability windows (e.g., business hours) versus those available 24/7.
* **Service Time Integration:** accurately accounts for the duration of tasks performed at each stop.


## Results

To be updated

## Citation
If this project was useful for your research, please cite:
```bash
@article{Alotaik2025project,
  title={Project Title},
  author={Alotaik, O.},
  year={2025}
}
```


