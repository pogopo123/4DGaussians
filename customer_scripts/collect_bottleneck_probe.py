#!/usr/bin/env python
"""Gom PSNR train/test cua 3 run bottleneck probe thanh 1 bang."""
import re, sys, os

TAGS = ["bn_baseline", "bn_fatmlp", "bn_hires"]
LABEL = {"bn_baseline": "baseline", "bn_fatmlp": "MLP x4", "bn_hires": "HexPlane hi-res"}
PAT = re.compile(r"\[ITER (\d+)\] Evaluating (train|test): L1 ([\d.eE+-]+) PSNR ([\d.eE+-]+)")


def parse(tag):
    """-> {(stage, iter, split): psnr}. Coarse chay 3000 iter roi fine reset ve 0,
    nen lan thu 2 gap 1 iteration da xuat hien = da sang stage fine."""
    path = f"output/multipleview/{tag}/run.log"
    if not os.path.exists(path):
        return {}, None
    txt = open(path, errors="ignore").read()
    out, seen, stage = {}, set(), "coarse"
    for it, split, _l1, ps in PAT.findall(txt):
        it = int(it)
        if (it, split) in seen:
            stage = "fine"
        seen.add((it, split))
        out[(stage, it, split)] = float(ps)
    pts = re.findall(r"point=(\d+)", txt)
    return out, (int(pts[-1]) if pts else None)


def main():
    data = {t: parse(t) for t in TAGS}
    iters = sorted({k[1] for t in TAGS for k in data[t][0] if k[0] == "fine"})
    for split in ("train", "test"):
        print(f"\n=== PSNR {split} (fine stage) ===")
        print(f"{'iter':>7} " + "".join(f"{LABEL[t]:>18}" for t in TAGS) + f"{'Δ fat':>10}{'Δ hires':>10}")
        for it in iters:
            row, vals = f"{it:>7} ", []
            for t in TAGS:
                v = data[t][0].get(("fine", it, split))
                vals.append(v)
                row += f"{v:>18.3f}" if v is not None else f"{'-':>18}"
            b = vals[0]
            for v in vals[1:]:
                row += f"{v - b:>+10.3f}" if (v is not None and b is not None) else f"{'-':>10}"
            print(row)
    print("\n=== so Gaussian cuoi cung ===")
    for t in TAGS:
        print(f"{LABEL[t]:>18}: {data[t][1]}")


if __name__ == "__main__":
    sys.exit(main())
