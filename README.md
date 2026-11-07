# House Edge — Team SSD (TAMU Datathon 2026)

Next-spin payout probability forecasts (0×, 1×, 2×, 5×, 20×) for slot machines from gameplay logs.

- `runs/05-eps/entry/` — the submitted main-round entry (`predict.py --data <public> --out <csv>`, standard library only, deterministic).
- `src/` — data exploration and model development scripts. `src/make_entry.py` packages an entry.
- `experiments.md` — experiment log with validation points per condition.

Validation: 58.16 / 60 technical points (starter `pooled_recent` baseline: 30.83).

The organizer data kit is not included; place it in `kit/` to run the scripts.
