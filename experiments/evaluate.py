# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Score model outputs in experiments/out/ against the hand labels and write results.md.

Usage (from the repo root):
    uv run experiments/evaluate.py
    uv run experiments/evaluate.py --out experiments/out --report experiments/results.md

Metrics, per model:
- surface recall   share of labelled surfaces that some predicted box overlaps (IoU >= 0.3)
- family accuracy  of those matched surfaces, share whose predicted family is acceptable
- group agreement  for pairs of matched surfaces in one photo, share where "same material?"
                   agrees with the labels (this is the dedupe step)
- photos complete  photos where every labelled material group was found at least once
- extra boxes      predicted boxes that overlap no labelled surface (may be valid, or noise)
- latency / cost   mean and worst seconds, mean USD per photo
- box order check  also scores with x/y swapped; if that scores higher the model probably
                   returned [y1, x1, y2, x2] and the prompt or parser needs a fix
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATCH_IOU = 0.3
EXTRA_IOU = 0.1

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Label:
    box: Box
    families: tuple[str, ...]
    group: str


@dataclass(frozen=True)
class Pred:
    box: Box
    family: str
    group: str


@dataclass
class ModelScore:
    model: str
    photos: int
    failures: int
    recall: float
    family_acc: float
    group_agree: float
    photos_complete: int
    extra_boxes: int
    latency_mean: float
    latency_max: float
    cost_mean: float | None
    swapped_recall: float


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def swap_xy(b: Box) -> Box:
    return (b[1], b[0], b[3], b[2])


def best_match(label: Label, preds: list[Pred]) -> Pred | None:
    """The predicted surface overlapping this label most, if it clears MATCH_IOU."""
    scored = [(iou(label.box, p.box), p) for p in preds]
    score, pred = max(scored, key=lambda t: t[0], default=(0.0, None))
    return pred if score >= MATCH_IOU else None


def score_photo(labels: list[Label], preds: list[Pred]) -> dict:
    matches = [(lab, best_match(lab, preds)) for lab in labels]
    found = [(lab, p) for lab, p in matches if p is not None]
    pairs = list(combinations(found, 2))
    return {
        "labels": len(labels),
        "found": len(found),
        "family_ok": sum(p.family in lab.families for lab, p in found),
        "pairs": len(pairs),
        "pairs_ok": sum((a.group == b.group) == (pa.group == pb.group) for (a, pa), (b, pb) in pairs),
        "complete": {lab.group for lab in labels} <= {lab.group for lab, _ in found},
        "extra": sum(all(iou(lab.box, p.box) < EXTRA_IOU for lab in labels) for p in preds),
    }


def load_labels(path: Path) -> dict[str, list[Label]]:
    raw = json.loads(path.read_text())
    return {
        img: [Label(tuple(s["box"]), tuple(s["family"]), s["group"]) for s in surfaces]
        for img, surfaces in raw.items()
        if not img.startswith("_")
    }


def load_run(path: Path) -> tuple[dict, list[Pred]]:
    run = json.loads(path.read_text())
    preds = [Pred(tuple(s["box"]), s["family"], s["material_group"]) for s in run.get("surfaces", [])]
    return run, preds


def ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


def score_model(model_dir: Path, labels: dict[str, list[Label]]) -> ModelScore | None:
    runs = sorted(model_dir.glob("*.json"))
    if not runs:
        return None
    totals = {"labels": 0, "found": 0, "family_ok": 0, "pairs": 0, "pairs_ok": 0, "complete": 0, "extra": 0}
    swapped_found = 0
    latencies, costs, failures, model = [], [], 0, model_dir.name
    for path in runs:
        run, preds = load_run(path)
        model = run.get("model", model)
        img_labels = labels.get(f"{path.stem}.png", [])
        latencies.append(run.get("latency_s", 0.0))
        if run.get("cost_usd") is not None:
            costs.append(run["cost_usd"])
        if not run.get("ok"):
            failures += 1
        photo = score_photo(img_labels, preds)
        for k in totals:
            totals[k] += int(photo[k])
        swapped = [Pred(swap_xy(p.box), p.family, p.group) for p in preds]
        swapped_found += score_photo(img_labels, swapped)["found"]
    return ModelScore(
        model=model,
        photos=len(runs),
        failures=failures,
        recall=ratio(totals["found"], totals["labels"]),
        family_acc=ratio(totals["family_ok"], totals["found"]),
        group_agree=ratio(totals["pairs_ok"], totals["pairs"]),
        photos_complete=totals["complete"],
        extra_boxes=totals["extra"],
        latency_mean=statistics.mean(latencies),
        latency_max=max(latencies),
        cost_mean=statistics.mean(costs) if costs else None,
        swapped_recall=ratio(swapped_found, totals["labels"]),
    )


def render(scores: list[ModelScore]) -> str:
    head = (
        "| Model | Photos (failed) | Surface recall | Family accuracy | Same-material agreement "
        "| Photos complete | Extra boxes | Latency mean / max | Cost per photo |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| `{s.model}` | {s.photos} ({s.failures}) | {s.recall:.0%} | {s.family_acc:.0%} | {s.group_agree:.0%} "
        f"| {s.photos_complete}/{s.photos} | {s.extra_boxes} | {s.latency_mean:.1f}s / {s.latency_max:.1f}s "
        f"| {'n/a' if s.cost_mean is None else f'${s.cost_mean:.4f}'} |\n"
        for s in scores
    )
    warnings = "".join(
        f"\n> `{s.model}`: recall is higher with x/y swapped ({s.swapped_recall:.0%} vs {s.recall:.0%}). "
        "It probably returns [y1, x1, y2, x2]; fix the prompt or parser before trusting its scores.\n"
        for s in scores
        if s.swapped_recall > s.recall + 0.1
    )
    targets = (
        "\nTargets from PLAN.md: photos complete >= 9/11, family accuracy >= 85%, "
        "cost <= $0.005 per photo, latency <= 10 s.\n"
    )
    return "# Step 1 results\n\n" + head + rows + warnings + targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=HERE / "labels.json")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--report", type=Path, default=HERE / "results.md")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    scores = [s for d in sorted(args.out.iterdir()) if d.is_dir() if (s := score_model(d, labels))]
    if not scores:
        raise SystemExit(f"No model outputs in {args.out}. Run run_vlm.py first.")
    report = render(scores)
    args.report.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
