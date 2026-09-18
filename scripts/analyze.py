"""Turn a records file into the results tables.

    python scripts/analyze.py --records runs/davis_s2l/records.csv --out runs/davis_s2l

    1. main table        method x {GT-visible J&F, J&F@5, Retention, target_frames}
    2. by switch point   Retention per method at 25 / 50 / 75%

Rules this file enforces so the reader does not have to trust a summary:

  * Retention is a ratio of means, never a mean of per-row ratios.
  * A NaN `@5` means the window held no visible ground truth.  It is dropped,
    never filled with 0.
  * Retention above 100 is printed as-is.
  * Rows from different manifest digests are refused outright.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from hoeval.results import load_records, summarize

MAIN = ["method", "n", "jf_gtvis", "jf_at_5_gtvis", "retention", "target_frames"]


def _md(df: pd.DataFrame, index: bool = False) -> str:
    """Markdown table without pulling in `tabulate`."""
    d = df.reset_index() if index else df
    cells = [[str(c) for c in d.columns]]
    for _, row in d.iterrows():
        cells.append([("%.2f" % v) if isinstance(v, float) else str(v) for v in row])
    widths = [max(len(r[i]) for r in cells) for i in range(len(cells[0]))]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cells[0], widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
            for r in cells[1:]]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_records(a.records)

    digests = sorted(df["manifest_digest"].unique())
    if len(digests) > 1:
        raise SystemExit(
            "records span %d manifests (%s). They were produced against "
            "different switch points and cannot be compared." % (len(digests), digests)
        )

    lines = ["# DAVIS 2017 val -- handoff baselines", "",
             "manifest `%s`, %d rows, %d videos, %d objects, %d methods"
             % (digests[0], len(df), df["video"].nunique(),
                df[["video", "obj_id"]].drop_duplicates().shape[0],
                df["method"].nunique()), ""]

    table = summarize(df).sort_values("jf_gtvis", ascending=False)[MAIN]
    table.to_csv(out / "table_main.csv", index=False)
    print("\n== main table ==")
    print(table.to_string(index=False, float_format=lambda v: "%.2f" % v))
    lines += ["## Main table", "", _md(table), ""]

    if df["fraction"].nunique() > 1:
        piv = (summarize(df, by=("fraction", "method"))
               .pivot_table(index="method", columns="fraction", values="retention")
               .round(2))
        piv.to_csv(out / "table_by_fraction.csv")
        print("\n== retention by switch point ==")
        print(piv.to_string(float_format=lambda v: "%.2f" % v))
        lines += ["## Retention (%) by switch point", "", _md(piv, index=True), ""]

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote %s" % (out / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
