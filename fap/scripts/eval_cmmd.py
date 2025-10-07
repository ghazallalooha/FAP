import argparse
from fap.eval.evaluate_cmmd import evaluate_clean
ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--fix", type=int, default=3)
args = ap.parse_args()
evaluate_clean(args.data_root, args.batch, args.fix, args.ckpt)
