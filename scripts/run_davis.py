"""The DAVIS 2017 val run: SAM2-S hands over to SAM2-L at 25/50/75%.

    python scripts/run_davis.py --davis /data/DAVIS \
        --checkpoints /path/to/sam2/checkpoints \
        --stats configs/stats --out runs/davis_s2l

Order of operations, and none of it is optional:

    scripts/dump_sam2_state.py     confirm the state layout
    tests/test_state_roundtrip.py  confirm export/import is lossless
    --methods direct_copy --limit 5
                                   if a verbatim copy already lands near Warm
                                   Target there is no representation gap inside
                                   SAM 2 -- learn that before the full grid
    --methods warm_target          Warm Target is the denominator of Retention;
                                   if it is wrong, every number is wrong by the
                                   same factor and nothing will look odd
    the whole thing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoeval.baselines import BASELINES, layer1_names
from hoeval.baselines_latent import register_layer2
from hoeval.datasets import davis2017
from hoeval.models.registry import build
from hoeval.protocol import build_manifest, load_manifest
from hoeval.results import save_records, summarize, to_frame
from hoeval.runner import evaluate_dataset

REPORT_COLS = ["method", "n", "jf_gtvis", "jf_at_5_gtvis", "retention", "target_frames"]


def resolve_methods(spec: str, layer2: list[str]) -> list[str]:
    out: list[str] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token == "all":
            out += layer1_names() + layer2
        elif token in BASELINES:
            out.append(token)
        else:
            raise SystemExit("unknown method %r; have %s" % (token, sorted(BASELINES)))
    return list(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", required=True, help="unpacked DAVIS 2017 trainval root")
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--source", default="sam2_small")
    ap.add_argument("--target", default="sam2_large")
    ap.add_argument("--manifest", default="configs/davis_val_switches.json")
    ap.add_argument("--stats", default=None,
                    help="directory of <model>.json from fit_norm_stats.py; "
                         "without it norm_matched is skipped, not stubbed")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = davis2017(a.davis, "val")
    man_path = Path(a.manifest)
    if man_path.exists():
        manifest = load_manifest(man_path)
        print("manifest %s  digest %s  points %d"
              % (man_path, manifest.digest, len(manifest.points)))
    else:
        manifest = build_manifest(ds.name, ds.video_lengths())
        manifest.save(man_path)
        print("manifest built and frozen at %s (digest %s)"
              % (man_path, manifest.digest))

    src = build(a.source, a.checkpoints, device=a.device)
    tgt = build(a.target, a.checkpoints, device=a.device)
    fingerprints = {src.model_id: src.fingerprint, tgt.model_id: tgt.fingerprint}
    print("source %-12s %s" % (src.model_id, src.fingerprint))
    print("target %-12s %s" % (tgt.model_id, tgt.fingerprint))
    print("memory window: %s" % src.reads)

    books = {}
    if a.stats:
        for m in (src.model_id, tgt.model_id):
            p = Path(a.stats) / ("%s.json" % m)
            if p.exists():
                books[m] = p
            else:
                print("no statistics at %s -- norm_matched will be skipped" % p)
    layer2 = register_layer2(
        source_model=src.model_id, target_model=tgt.model_id,
        source_stats=books.get(src.model_id), target_stats=books.get(tgt.model_id),
    )
    methods = resolve_methods(a.methods, layer2)
    print("methods: %s" % methods)

    t0 = time.perf_counter()

    def progress(video, n_rows):
        print("  %-28s %6d rows  %7.1fs" % (video, n_rows, time.perf_counter() - t0),
              flush=True)

    records = evaluate_dataset(
        dataset=ds, manifest=manifest, source=src, target=tgt, methods=methods,
        limit=a.limit, cache_dir=a.cache_dir, fingerprints=fingerprints,
        on_video=progress,
    )
    summary = summarize(to_frame(records))
    print(summary.sort_values("jf_gtvis", ascending=False)[REPORT_COLS]
          .to_string(index=False, float_format=lambda v: "%.3f" % v))

    save_records(records, out_dir / "records.csv")
    (out_dir / "run.json").write_text(json.dumps({
        "dataset": ds.name,
        "davis": a.davis,
        "manifest": str(man_path),
        "manifest_digest": manifest.digest,
        "models": {
            m.model_id: {"checkpoint": m.checkpoint, "fingerprint": m.fingerprint,
                         "config": m.config, "reads": m.reads}
            for m in (src, tgt)
        },
        "methods": methods,
        "stats": a.stats,
        "rows": len(records),
    }, indent=2), encoding="utf-8")
    print("\n%d rows -> %s" % (len(records), out_dir / "records.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
