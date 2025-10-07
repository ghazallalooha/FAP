import argparse
from fap.eval.evaluate_cmmd import certify_randomized_smoothing
ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--fix", type=int, default=3)
ap.add_argument("--sigma", type=float, default=0.12)
ap.add_argument("--N", type=int, default=10000)
args = ap.parse_args()
certify_randomized_smoothing(args.data_root, 1, args.fix, args.ckpt, args.sigma, args.N)
