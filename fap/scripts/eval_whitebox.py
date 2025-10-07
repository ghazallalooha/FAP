import argparse
from fap.eval.eval_whitebox import robust_accuracy_pgd
ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--eps", type=float, default=0.004)
ap.add_argument("--alpha", type=float, default=0.001)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--fix", type=int, default=3)
args = ap.parse_args()
robust_accuracy_pgd(args.data_root, args.ckpt, args.eps, args.alpha, args.steps, args.fix)
