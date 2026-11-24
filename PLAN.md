# PLAN: Interior surface → product matcher

Status: **step 1 in progress** (scripts and labels done; model runs need an OpenRouter key)

## Goal
A self-serve web app where a client uploads an interior photo. The app finds every finished
surface (walls, islands, counter fronts, columns, ceilings), groups surfaces that use
the same material, and suggests the closest products from a 2,000–5,000 item catalog.

## Constraints
- At most ~1,000 recognitions a week. Keep running cost near zero.
- Prefer a static site on a CDN, with no servers to run.

## Assumptions (confirmed by Pat)
- The catalog covers several material families, for example plaster textures, wood veneer, stone
  and marble, tiles and mosaic, and wallpaper. It isn't only the plaster sheet.
- Client photos look like `samples/interiors/`: well lit, design-magazine style, with 2–4
  distinct materials per photo.
- Results are "closest look from our range", ranked, never "no match".
- Islands, columns and ceilings count as surfaces.

## Architecture
```
Browser (static site on Cloudflare Pages)
  ├─ upload photo, resize to ≤1280px
  ├─ POST → Cloudflare Worker (free tier; only hides the API key)
  │        └─ VLM (Qwen3-VL or Gemini Flash): list surfaces →
  │           [{box, material_family, pattern, colour, description}]
  ├─ in browser: crop each box → image embedding (SigLIP/CLIP via transformers.js)
  ├─ dedupe: merge surfaces with the same family + close embedding
  └─ match: filter catalog.json by family → cosine rank → top 5
catalog.json on the CDN: id, name, family, colour, image URL, embedding (int8)
```
- Why a VLM for detection: these photos mix wood, stone, tile and wallpaper on walls, islands
  and columns. Fixed "wall" segmentation models (ADE20K) would miss many of them. A VLM can name
  the material family, which narrows the search.
- Why no vector DB: 5,000 × 512-dim int8 is about 2.5 MB. Brute-force search in the browser is
  instant.
- Estimated cost: about 4,300 calls a month at roughly $0.001–0.004 each ≈ $5–20 a month. The
  Worker and Pages are free tier.

## Steps
1. **Test on the sample photos**
   - Script: run the 11 sample photos through 2 VLMs (Qwen3-VL, Gemini Flash), with a prompt that
     returns JSON surface lists.
   - Hand-label the surfaces in the 11 photos to use as ground truth.
   - Compare detection quality, family accuracy, latency and cost per image. Pick one.
   - Output: `experiments/` folder plus a short results table.
2. **Catalog + matching**
   - Script: product images + CSV → `catalog.json` with embeddings.
   - Crop → embed → dedupe → rank, checked on the sample photos (is the right product in the top 5?).
3. **Web app**
   - Upload → highlighted surfaces → top 5 products per material, linking to product pages.
   - Deploy to Cloudflare Pages + Worker. Add a README.

## Files (step 1)
- `README.md`: project overview (new)
- `experiments/prompts/surface_list.md`: VLM prompt (new)
- `experiments/run_vlm.py`: runs the sample photos through each model and saves JSON (new)
- `experiments/labels.json`: hand labels (new)
- `experiments/evaluate.py`: scores outputs against labels (new)
- `experiments/test_scripts.py`: offline tests (new)
- `experiments/results.md`: comparison (new)
- `flake.nix`, `ruff.toml`: dev shell and lint config (new)

Commit point: end of step 1, on a branch, with a draft PR.

## Verification
- Every surface the labels call a distinct material is found in at least 9 of the 11 photos.
- Material family is correct for ≥85% of detected surfaces.
- Cost per image ≤ $0.005. Response time ≤ 10 s.

## Risks
- VLM boxes are rough, so crops may include furniture. Fallback: refine with SAM in the browser.
- Lighting shifts colour. Match on texture first and treat colour as a soft filter.
- Low-contrast plaster textures (orange peel, spray sand) aren't visible in whole-room shots.
  Offer an optional close-up upload.
- A Worker is still a tiny backend. A fully static setup would mean in-browser models only, which
  are weaker at naming materials.

## Needs from Pat
- Product data: a CSV or list (id, name, family, colour) plus one image per product, or a sample
  of ~50 for step 2.
- An API key for the chosen VLM before step 1 runs (OpenRouter or Alibaba DashScope for Qwen,
  Google AI Studio for Gemini).
