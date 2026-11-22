# Step 1 results

| Model | Photos (failed) | Surface recall | Family accuracy | Same-material agreement | Photos complete | Extra boxes | Latency mean / max | Cost per photo |
|---|---|---|---|---|---|---|---|---|
| `google/gemini-2.5-flash` | 11 (1) | 41% | 72% | 75% | 2/11 | 9 | 17.3s / 84.4s | $0.0023 |
| `qwen/qwen3-vl-235b-a22b-instruct` | 11 (0) | 80% | 91% | 83% | 7/11 | 18 | 12.4s / 51.8s | $0.0014 |

Targets from PLAN.md: photos complete >= 9/11, family accuracy >= 85%, cost <= $0.005 per photo, latency <= 10 s.
