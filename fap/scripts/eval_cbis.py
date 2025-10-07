import argparse, os
from fap.eval.evaluate_cbis import evaluate_clean
ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--labels_test", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--fix", type=int, default=3)
args = ap.parse_args()
evaluate_clean(args.images, args.labels_test, args.batch, args.fix, args.ckpt)
