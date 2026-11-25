# visual-pattern-recognition

A self-serve web app (planned) where a client uploads an interior photo. The app finds every
finished surface (walls, ceilings, columns, islands, counter fronts), groups surfaces that use the
same material, and suggests the closest products from the catalog.

Status: **step 2 of 3, catalog + matching.** Step 1 chose Qwen3-VL for finding surfaces. See [PLAN.md](PLAN.md).

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

## Step 2: catalog + matching

```sh
uv run catalog/scrape_site.py                    # products + portfolio photos from blackbone.co.th
uv run --extra ml catalog/build_index.py         # embed every swatch -> catalog/index/*.npz
export OPENROUTER_API_KEY=...
uv run --extra ml experiments/match_projects.py  # match portfolio photos, writes experiments/results_step2.md
```

| Path | What it does |
|---|---|
| `catalog/scrape_site.py` | Products (decor code, name, category, swatch) and portfolio projects from the site's public APIs |
| `catalog/build_index.py` | Swatch embeddings + main colour per decor, one index per embedding model |
| `matcher/` | Colour (CIELAB, lighting correction), embedders (colour histogram, DINOv2, CLIP) and ranking |
| `experiments/match_projects.py` | Checks the decors each portfolio project names show up in the top 5 for its photos |
| `experiments/export_review.py` | Bundles a matching run with thumbnails into `experiments/out/review_data.json` for the visual review page |

Ranking: the vision model's material family picks the decor category (wood → woodgrain,
stone → stone, paint/plaster → solid colour, ...). Decors are scored by image similarity
(swatch embedded whole and as four quadrants, averaged) minus a colour-distance penalty, which
dominates for solid colours. In `auto` mode the category is a hard filter for families the
vision model labels reliably (wood, metal, paint) and only a head start for plaster, which it
also uses for glossy marble. `auto-hub` also lowers "hub" decors that sit close to almost any
wall crop (CSLS-style correction), so a few decors stop crowding every top 5.

Tests: `uv run --extra ml python -m unittest matcher/test_matcher.py catalog/test_scrape_site.py experiments/test_scripts.py`
