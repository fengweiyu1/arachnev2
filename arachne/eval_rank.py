"""
Ranking-based localisation evaluator.

Given localisation outputs (loc.*.all_cost.pkl) and ground-truth faulty weights,
reports whether the true fault is ranked before/after/tied with non-faulty weights.

Usage (FashionMNIST example):
python3 -m arachne.eval_rank \
  --fid-file final_data/data/rq1_fault_ids/fm.target.fault_ids.csv \
  --model-template "data/models/rq1_faulty_mdl/fashion_mnist/1/fmnist_simple_seed{fault_id}.h5" \
  --results sbfl:results/rq1_sbfl_all/sbfl/sbfl localiser:results/rq1_localiser_all/localiser/bl \
  --output results/rq1_rank_compare_fm.csv

The script is intentionally generic: supply more --results entries to compare
additional localisation methods.
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from arachne.utils.model_util import (
    convert_cifar_to_channels_last,
    convert_gtsrb_to_channels_last,
)


@dataclass
class RankResult:
    method: str
    seed: int
    fault_id: int
    layer: int
    w_idx: Tuple[int, ...]
    rank: int
    total: int
    before: int
    tied: int
    after: int
    score: str
    found: bool


def parse_args():
    parser = argparse.ArgumentParser(description="Ranking-based localisation comparison.")
    parser.add_argument(
        "--fid-file",
        required=True,
        help="CSV mapping seed->fault_id (e.g., final_data/data/rq1_fault_ids/fm.target.fault_ids.csv)",
    )
    parser.add_argument(
        "--model-template",
        required=True,
        help="Path template to faulty model, must contain '{fault_id}' (e.g., data/models/.../fmnist_simple_seed{fault_id}.h5)",
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="List of method:result_dir pairs, e.g., sbfl:results/rq1_sbfl_all/sbfl/sbfl localiser:results/rq1_localiser_all/localiser/bl",
    )
    parser.add_argument(
        "--output",
        default="results/rank_compare.csv",
        help="Output CSV path for the summary.",
    )
    parser.add_argument(
        "--hitks",
        default="1,5,10,50,100",
        help="Comma-separated list of K for hit@K metrics.",
    )
    return parser.parse_args()


def load_weight_shapes(model_path: str) -> Dict[int, Tuple[int, ...]]:
    """Return mapping layer_index -> shape of first weight tensor."""
    mdl = load_model(model_path, compile=False)
    shapes = {}
    for idx, layer in enumerate(mdl.layers):
        ws = layer.get_weights()
        if ws:
            shapes[idx] = tuple(ws[0].shape)
    return shapes


def flatten_idx(idx: Iterable[int], shape: Tuple[int, ...]) -> int:
    idx_tuple = tuple(int(x) for x in idx)
    return int(np.ravel_multi_index(idx_tuple, shape))


def canonical_key(key: List, shapes: Dict[int, Tuple[int, ...]]) -> Tuple[int, int]:
    """
    Convert localisation key [layer, weight] to (layer, flat_idx).
    weight may already be flat or a multi-d index.
    """
    layer = int(key[0])
    w = key[1]
    if isinstance(w, (list, tuple, np.ndarray)):
        if np.ndim(w) == 0:  # scalar stored in array
            w = int(np.array(w))
        elif len(np.array(w).shape) == 0:
            w = int(w)
        else:
            w = flatten_idx(w, shapes[layer])
    else:
        w = int(w)
    return layer, w


def normalise_score(score) -> str:
    """Turn various score types into a comparable string for grouping."""
    if isinstance(score, (float, int, np.floating, np.integer)):
        return f"{float(score):.12g}"
    if isinstance(score, np.ndarray):
        if score.size == 0:
            return ""
        return "|".join(f"{float(x):.12g}" for x in score.flatten())
    if isinstance(score, (list, tuple)):
        return "|".join(normalise_score(x) for x in score)
    return str(score)


def evaluate_method(
    method: str,
    result_dir: str,
    fid_df: pd.DataFrame,
    model_template: str,
) -> List[RankResult]:
    results: List[RankResult] = []
    for _, row in fid_df.iterrows():
        seed = int(row["idx"])
        fault_id = int(row["id"])

        model_path = model_template.format(fault_id=fault_id)
        if not os.path.exists(model_path):
            # Try converting channels_first models to channels_last on the fly.
            if "cifar" in model_path and "rq1_faulty_mdl_cifar_cl" in model_path:
                src = model_path.replace("rq1_faulty_mdl_cifar_cl", "rq1_faulty_mdl")
                if os.path.exists(src):
                    print(f"[INFO] Converting CIFAR model for seed {seed} to channels_last")
                    convert_cifar_to_channels_last(src, model_path)
            elif "gtsrb" in model_path and "rq1_faulty_mdl_gtsrb_cl" in model_path:
                # The original GTSRB models live under data/models/rq1_faulty_mdl/GTSRB/1/...
                fname = os.path.basename(model_path)
                src = os.path.join("data", "models", "rq1_faulty_mdl", "GTSRB", "1", fname)
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    print(f"[INFO] Converting GTSRB model for seed {seed} to channels_last")
                    convert_gtsrb_to_channels_last(src, model_path)

        if not os.path.exists(model_path):
            print(f"[WARN] Model not found for seed {seed}: {model_path}")
            continue
        shapes = load_weight_shapes(model_path)

        # ground truth weights
        gt_path = os.path.join(os.path.dirname(model_path), f"faulty_nws.{fault_id}.pkl")
        if not os.path.exists(gt_path):
            # Try original channels_first location.
            gt_path_orig = os.path.join("data", "models", "rq1_faulty_mdl", "GTSRB", "1", f"faulty_nws.{fault_id}.pkl")
            if os.path.exists(gt_path_orig):
                gt_path = gt_path_orig
            else:
                print(f"[WARN] GT file not found for seed {seed}: {gt_path}")
                continue
        gt_df = pd.read_pickle(gt_path)

        # locate all_cost file
        glob_candidates = [
            f"loc.{seed}.1.qexec_qres_sbfl.all_cost.pkl",
            f"loc.{seed}.1.qexec_bres_sbfl.all_cost.pkl",
            f"loc.{seed}.1.bexec_qres_guider.all_cost.pkl",
            f"loc.{seed}.1.bexec_bres_sbfl.all_cost.pkl",
            f"loc.{seed}.1.all_cost.pkl",
        ]
        all_cost_path = None
        for name in glob_candidates:
            candidate = os.path.join(result_dir, name)
            if os.path.exists(candidate):
                all_cost_path = candidate
                break
        if all_cost_path is None:
            print(f"[WARN] all_cost not found for seed {seed} in {result_dir}")
            continue

        with open(all_cost_path, "rb") as f:
            all_cost = pickle.load(f)

        # build canonical ranking list
        ranking = []
        for entry in all_cost:
            key, score = entry
            ranking.append((canonical_key(key, shapes), score))

        # ensure order is as provided (assumed sorted). If needed, could sort here.
        scores_norm = [normalise_score(s) for _, s in ranking]

        for _, gt_row in gt_df.iterrows():
            layer = int(gt_row["layer"])
            w_idx = tuple(gt_row["w_idx"])
            flat_gt = canonical_key([layer, w_idx], shapes)

            found_idx = [i for i, (k, _) in enumerate(ranking) if k == flat_gt]
            if not found_idx:
                results.append(
                    RankResult(
                        method,
                        seed,
                        fault_id,
                        layer,
                        w_idx,
                        rank=-1,
                        total=len(ranking),
                        before=len(ranking),
                        tied=0,
                        after=0,
                        score="",
                        found=False,
                    )
                )
                continue

            rank = found_idx[0]
            fault_score_norm = scores_norm[rank]
            # tie group
            tie_indices = [i for i, s in enumerate(scores_norm) if s == fault_score_norm]
            tied = len(tie_indices)
            before = min(tie_indices)
            after = len(ranking) - max(tie_indices) - 1

            results.append(
                RankResult(
                    method,
                    seed,
                    fault_id,
                    layer,
                    w_idx,
                    rank=rank,
                    total=len(ranking),
                    before=before,
                    tied=tied,
                    after=after,
                    score=fault_score_norm,
                    found=True,
                )
            )
    return results


def main():
    args = parse_args()

    fid_df = pd.read_csv(args.fid_file)
    method_dirs = dict(item.split(":", 1) for item in args.results)
    hitks = [int(k) for k in args.hitks.split(",") if k.strip()]

    all_results: List[RankResult] = []
    for method, res_dir in method_dirs.items():
        res = evaluate_method(method, res_dir, fid_df, args.model_template)
        all_results.extend(res)

    if not all_results:
        print("No results generated.")
        return

    out_df = pd.DataFrame([r.__dict__ for r in all_results])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # add hit@K columns
    for k in hitks:
        out_df[f"hit@{k}"] = (out_df["rank"] >= 0) & (out_df["rank"] < k)

    out_df.to_csv(args.output, index=False)

    # quick aggregate: mean rank per method
    rank_summary = (
        out_df[out_df["found"]]
        .groupby("method")["rank"]
        .agg(["mean", "median", "min", "max"])
    )
    hit_summary = out_df.groupby("method")[[f"hit@{k}" for k in hitks]].mean()
    print("Saved detailed report to", args.output)
    print("Summary (rank):")
    print(rank_summary)
    print("Summary (hit@K):")
    print(hit_summary)


if __name__ == "__main__":
    main()
