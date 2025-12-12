import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


def read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def parse_interactions(interactions: Dict[str, Any]) -> Tuple[Dict[str, Any], int, Optional[int]]:
    ev = interactions.get("Evaluation", {})
    code_data = {
        "used_clarification": ev.get("used_clarification", False),
        "num_clarifications": ev.get("num_clarifications", 0),
        "used_move": ev.get("used_move", False),
        "num_moves": ev.get("num_moves", 0),
        "used_undo": ev.get("used_undo", False),
        "num_undos": ev.get("num_undos", 0),
        "used_remove": ev.get("used_remove", False),
        "num_removes": ev.get("num_removes", 0),
        "used_clear": ev.get("used_clear", False),
        "num_clears": ev.get("num_clears", 0),
        "use_dspy_reconst": ev.get("use_dspy_reconst", ev.get("use_dspy", "Not defined")),
        "use_dspy_history": ev.get("use_dspy_history", "Not defined"),
        "used_retry_reconst": ev.get("used_retry_reconst", False),
    }
    num_turns = interactions.get("Complete turns", 0)
    boardinfo = ev.get("boardinfo", {})
    total_shapes = boardinfo.get("total_shapes") if isinstance(boardinfo, dict) else None
    return code_data, num_turns, total_shapes


def parse_scores(scores: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    es = scores.get("episode scores", {})
    main = es.get("Main Score")
    reconst = es.get("reconstruction_status")
    return main, reconst


def process_episode(episode_path: str) -> Dict[str, Any]:
    stats = {
        "accuracy_vals": [],
        "failed_episodes": [],
        "aborted_episodes": [],
        "code_stats": {},
        "dialog_turns": 0,
        "total_shapes": None,
        "outcome": "unknown",
    }

    interactions_path = os.path.join(episode_path, "interactions.json")
    scores_path = os.path.join(episode_path, "scores.json")

    if os.path.exists(interactions_path):
        interactions = read_json_file(interactions_path) or {}
        code_data, num_turns, total_shapes = parse_interactions(interactions)
        stats["code_stats"] = code_data
        stats["dialog_turns"] = num_turns
        stats["total_shapes"] = total_shapes

    if os.path.exists(scores_path):
        scores = read_json_file(scores_path) or {}
        main_score, reconst_score = parse_scores(scores)

        if isinstance(main_score, float) and np.isnan(main_score):
            stats["accuracy_vals"].append(0)
            stats["aborted_episodes"].append(os.path.basename(episode_path))
            stats["outcome"] = "aborted"
        elif main_score == 0:
            stats["failed_episodes"].append(os.path.basename(episode_path))
            stats["outcome"] = "failed"
            stats["accuracy_vals"].append(0)
        else:
            stats["accuracy_vals"].append(main_score)
            stats["outcome"] = "success"
    return stats


def compute_scores(base_dir: str, verbose: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue

        for game in os.listdir(model_path):
            game_path = os.path.join(model_path, game)
            if not os.path.isdir(game_path):
                continue
            results.setdefault(game, {})
            results[game].setdefault(model, {})

            for exp in os.listdir(game_path):
                exp_path = os.path.join(game_path, exp)
                if not os.path.isdir(exp_path):
                    continue
                results[game][model].setdefault(exp, {})

                accuracy_data: List[float] = []
                failed_episodes: List[str] = []
                aborted_episodes: List[str] = []
                code_stats: Dict[str, Any] = {}
                # dialog_stats_by_shape maps shape_key -> list of (episode, turns)
                dialog_stats_by_shape: Dict[str, List[Tuple[str, int]]] = {}
                shape_stats: Dict[str, Dict[str, int]] = {}
                shape_code_stats: Dict[str, Dict[str, Any]] = {}
                failed_episodes_by_shape: Dict[str, List[str]] = {}

                episodes = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                num_episodes = len(episodes)

                for episode in episodes:
                    episode_path = os.path.join(exp_path, episode)
                    ep_stats = process_episode(episode_path)
                    # determine shape key early so it's available to all aggregations
                    total_shapes = ep_stats.get("total_shapes")
                    shape_key_str = str(total_shapes) if total_shapes is not None else "unknown"

                    accuracy_data.extend(ep_stats.get("accuracy_vals", []))
                    failed_episodes.extend(ep_stats.get("failed_episodes", []))                   
                    aborted_episodes.extend(ep_stats.get("aborted_episodes", []))

                    if ep_stats.get("code_stats"):
                        code_stats[episode] = ep_stats["code_stats"]
                        # update per-shape code stats
                        code_data = ep_stats["code_stats"]
                        # ensure shape_code_stats entry
                        if shape_key_str not in shape_code_stats:
                            shape_code_stats[shape_key_str] = {
                                "used_clarification_count": 0,
                                "num_clarifications": 0,
                                "ep_clarifications": [],
                                "ep_remove": [],
                                "ep_move": [],
                                "ep_clear": [],
                                "ep_undo": [],
                                "ep_reconst_retry": [],
                            }
                        scs = shape_code_stats[shape_key_str]
                        if code_data.get("used_clarification"):
                            scs["used_clarification_count"] += 1
                            scs["ep_clarifications"].append(episode)
                        scs["num_clarifications"] += int(code_data.get("num_clarifications", 0))
                        if code_data.get("used_remove"):
                            scs["ep_remove"].append(episode)
                        if code_data.get("used_move"):
                            scs["ep_move"].append(episode)
                        if code_data.get("used_clear"):
                            scs["ep_clear"].append(episode)
                        if code_data.get("used_undo"):
                            scs["ep_undo"].append(episode)
                        if code_data.get("used_retry_reconst"):
                            scs["ep_reconst_retry"].append(episode)
                    if ep_stats.get("dialog_turns") is not None:
                        # Group dialog turns by total_shapes: store list of (episode, turns)
                        dialog_stats_by_shape.setdefault(shape_key_str, []).append(
                            (episode, int(ep_stats["dialog_turns"]))
                        )
                    # aggregate by total_shapes
                    # use shape_key_str defined above
                    if shape_key_str not in shape_stats:
                        shape_stats[shape_key_str] = {
                            "total": 0,
                            "reconst_success": 0,
                            "reconst_failure": 0,
                            "aborted": 0,
                        }
                    outcome = ep_stats.get("outcome", "unknown")
                    if outcome == "success":
                        shape_stats[shape_key_str]["reconst_success"] += 1
                    elif outcome == "failed":
                        shape_stats[shape_key_str]["reconst_failure"] += 1
                        # record failed episode under its shape key
                        failed_episodes_by_shape.setdefault(shape_key_str, []).append(episode)
                    elif outcome == "aborted":
                        shape_stats[shape_key_str]["aborted"] += 1
                    shape_stats[shape_key_str]["total"] += 1

                total_aborted = len(aborted_episodes)
                total_failed = len(failed_episodes)
                total_episodes = num_episodes
                total_run_episodes = total_episodes - total_aborted
                total_reconst_episodes = total_run_episodes - total_failed

                reconst_accuracy = round(total_reconst_episodes / total_episodes, 2) if total_episodes > 0 else 0

                # legacy flat lists (kept for backward compatibility)
                ep_clarifications_flat = [ep for ep, d in code_stats.items() if d.get("used_clarification")]
                ep_remove_flat = [ep for ep, d in code_stats.items() if d.get("used_remove")]
                ep_move_flat = [ep for ep, d in code_stats.items() if d.get("used_move")]
                ep_clear_flat = [ep for ep, d in code_stats.items() if d.get("used_clear")]
                ep_undo_flat = [ep for ep, d in code_stats.items() if d.get("used_undo")]
                ep_reconst_retry_flat = [ep for ep, d in code_stats.items() if d.get("used_retry_reconst")]

                # grouped by shape: create dictionaries mapping shape_key -> lists/counts
                used_clarification_by_shape: Dict[str, bool] = {}
                num_clarifications_by_shape: Dict[str, int] = {}
                ep_clarifications_by_shape: Dict[str, List[str]] = {}
                ep_remove_by_shape: Dict[str, List[str]] = {}
                ep_move_by_shape: Dict[str, List[str]] = {}
                ep_clear_by_shape: Dict[str, List[str]] = {}
                ep_undo_by_shape: Dict[str, List[str]] = {}
                ep_reconst_retry_by_shape: Dict[str, List[str]] = {}

                for sk, scs in shape_code_stats.items():
                    used_clarification_by_shape[sk] = scs.get("used_clarification_count", 0) > 0
                    num_clarifications_by_shape[sk] = scs.get("num_clarifications", 0)
                    ep_clarifications_by_shape[sk] = scs.get("ep_clarifications", [])
                    ep_remove_by_shape[sk] = scs.get("ep_remove", [])
                    ep_move_by_shape[sk] = scs.get("ep_move", [])
                    ep_clear_by_shape[sk] = scs.get("ep_clear", [])
                    ep_undo_by_shape[sk] = scs.get("ep_undo", [])
                    ep_reconst_retry_by_shape[sk] = scs.get("ep_reconst_retry", [])

                # expose grouped lists under the main ep_* names (shape-keyed)
                ep_clarifications = ep_clarifications_by_shape
                ep_remove = ep_remove_by_shape
                ep_move = ep_move_by_shape
                ep_clear = ep_clear_by_shape
                ep_undo = ep_undo_by_shape
                ep_reconst_retry = ep_reconst_retry_by_shape


                # compute percentages per shape
                for k, v in shape_stats.items():
                    total = v.get("total", 0)
                    reconst_succ = v.get("reconst_success", 0)
                    reconst_fail = v.get("reconst_failure", 0)
                    aborted = v.get("aborted", 0)

                    v["reconst_success_pct"] = round((reconst_succ / total), 2) if total > 0 else 0.0
                    v["reconst_failure_pct"] = round((reconst_fail / total), 2) if total > 0 else 0.0
                    v["aborted_pct"] = round((aborted / total), 2) if total > 0 else 0.0
                    reconst_episodes = total - aborted - reconst_fail

                # Build dialog_stats summary grouped by shape: list of (episode, turns) plus avg/max/min
                dialog_stats: Dict[str, Any] = {}
                for sk, ep_list in dialog_stats_by_shape.items():
                    turns = [t for (_, t) in ep_list]
                    avg_turns = round((sum(turns) / len(turns)), 2) if turns else 0
                    dialog_stats[sk] = {
                        "episodes": ep_list,
                        "avg_turns": np.median(turns) if turns else 0, #avg_turns
                        "max_turns": max(turns) if turns else 0,
                        "min_turns": min(turns) if turns else 0,
                    }

                save_data = {
                    "num_episodes": num_episodes,
                    "accuracy": round((float(np.sum(accuracy_data)) / num_episodes), 2) if num_episodes > 0 else 0,
                    "num_failed_episodes": total_failed,
                    "num_aborted_episodes": total_aborted,
                    "failed_episodes": failed_episodes,
                    "aborted_episodes": aborted_episodes,
                    "reconst_accuracy": reconst_accuracy,
                    "used_clarification": any(d.get("used_clarification") for d in code_stats.values()),
                    "num_clarifications": sum(d.get("num_clarifications", 0) for d in code_stats.values()),
                    # ep_* now grouped by shape (shape_key -> list of episodes)
                    "ep_clarifications": ep_clarifications,
                    "ep_remove": ep_remove,
                    "ep_move": ep_move,
                    "ep_clear": ep_clear,
                    "ep_undo": ep_undo,
                    "ep_reconst_retry": ep_reconst_retry,
                    "failed_episodes_by_shape": failed_episodes_by_shape,
                    # legacy flat lists retained under _flat names
                    "ep_clarifications_flat": ep_clarifications_flat,
                    "ep_remove_flat": ep_remove_flat,
                    "ep_move_flat": ep_move_flat,
                    "ep_clear_flat": ep_clear_flat,
                    "ep_undo_flat": ep_undo_flat,
                    "ep_reconst_retry_flat": ep_reconst_retry_flat,
                    "dialog_stats": dialog_stats,
                    "shape_stats": shape_stats,
                    "shape_code_stats": shape_code_stats,
                    # grouped-by-shape summaries for quick access
                    "used_clarification_by_shape": used_clarification_by_shape,
                    "num_clarifications_by_shape": num_clarifications_by_shape,
                    "ep_clarifications_by_shape": ep_clarifications_by_shape,
                    "ep_remove_by_shape": ep_remove_by_shape,
                    "ep_move_by_shape": ep_move_by_shape,
                    "ep_clear_by_shape": ep_clear_by_shape,
                    "ep_undo_by_shape": ep_undo_by_shape,
                    "ep_reconst_retry_by_shape": ep_reconst_retry_by_shape,
                }

                if verbose:
                    print(f"Game: {game}, Model: {model}, Experiment: {exp}")
                    print(f"Total Episodes:{total_episodes}, total_reconst_episodes: {total_reconst_episodes}, Aborted: {total_aborted}, Failed: {total_failed}")
                    print(f"reconst_accuracy: {reconst_accuracy}")

                results[game][model][exp] = save_data

    outpath = os.path.join(base_dir, "overall_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute overall scores from experiment directories")
    parser.add_argument("base_dir", nargs="?", default="/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/collaborate/r5", help="Base directory containing model results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()

    compute_scores(args.base_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()