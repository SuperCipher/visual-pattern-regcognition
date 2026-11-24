# visual-pattern-recognition

A self-serve web app (planned) where a client uploads an interior photo. The app finds every
finished surface (walls, ceilings, columns, islands, counter fronts), groups surfaces that use the
same material, and suggests the closest products from the catalog.

Status: **step 1 of 3, model test.** See [PLAN.md](PLAN.md) for the full plan.

## Step 1: which vision model finds surfaces best?

`experiments/` runs sample photos through vision-language models on OpenRouter and scores them
against hand labels.

| File | What it does |
|---|---|
| `experiments/prompts/surface_list.md` | Prompt: list surfaces as JSON (box, type, family, pattern, colour, material group) |
| `experiments/labels.json` | Hand labels for the 11 sample photos (ground truth) |
| `experiments/run_vlm.py` | Calls each model for each photo, saves JSON + an overlay PNG |
| `experiments/evaluate.py` | Scores the outputs and writes `experiments/results.md` |
| `experiments/test_scripts.py` | Offline tests for the two scripts |

### Run it

```sh
nix develop                      # python, uv and ruff (or install uv yourself)
mkdir -p experiments/samples     # sample photos are third-party images, kept out of git
cp /path/to/samples/*.png experiments/samples/   # named 01.png ... 11.png to match labels.json

export OPENROUTER_API_KEY=...    # https://openrouter.ai/keys, never commit it
uv run experiments/run_vlm.py --list-models     # check the model ids still exist
uv run experiments/run_vlm.py                   # default: Qwen3-VL and Gemini Flash
uv run experiments/evaluate.py                  # writes experiments/results.md
```

Useful flags: `--models a,b` to choose models, `--dry-run` to build requests without calling the
API, `--max-side 1024` to send smaller images (cheaper).

### Tests and lint

```sh
uv run --with pillow python -m unittest experiments/test_scripts.py
uvx ruff check experiments/ && uvx ruff format --check experiments/
```

### Metrics

- **Surface recall:** share of labelled surfaces that a predicted box overlaps (IoU ≥ 0.3).
- **Family accuracy:** of those, share with an acceptable material family.
- **Same-material agreement:** whether pairs of surfaces are correctly grouped as the same or
  different material (the dedupe step).
- **Photos complete:** photos where every labelled material was found.
- **Extra boxes:** predictions that overlap nothing labelled (may be valid, or noise).
- **Latency and cost** per photo, from OpenRouter's usage data.
