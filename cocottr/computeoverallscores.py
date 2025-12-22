import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from utils.prepareasciirep import PrepareASCIIRep


def read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
    
def getshapesinfo(occupied_cells: dict) -> Dict[str, int]:
    """Get shape information from occupied_cells dictionary.
    
    Args:
        occupied_cells: Dictionary mapping cell coordinates to lists of elements in those cells.
    Returns:
        A dictionary with shape types as keys and their counts as values.
    """
    print(occupied_cells)

    prepascii = PrepareASCIIRep()
    shapes_list = []
    for key, value in occupied_cells.items():
        shape_info = prepascii._prepare_shape_color_dict(value, True)
        shapes_list.append(shape_info)

    print(shapes_list)



def computelevels(gt_code: str, boardsize: dict) -> Optional[int]:
    """Compute the number of levels/layers in a grid based on gtcode.
    
    Args:
        gt_code: The ground truth code function as a string
        
    Returns:
        Maximum number of layers across all occupied cells, or None if computation fails
    """
    try:
        prepascii = PrepareASCIIRep()
        occupied_cells = prepascii.get_occupied_cells(gt_code, boardsize)
        if not occupied_cells:
            return None
        max_layers = max(len(elements) for elements in occupied_cells.values())
        return max_layers
    except Exception:
        return None
    

def computelevels_genresponse(occupied_cells: dict) -> Optional[int]:
    """Compute the number of levels/layers in a grid based on gtcode.
    
    Args:
        gt_code: The ground truth code function as a string
        
    Returns:
        Maximum number of layers across all occupied cells, or None if computation fails
    """
    max_layers = max(len(elements) for elements in occupied_cells.values())
    prepascii = PrepareASCIIRep()
    shapes, colors = prepascii.getshapescolorsfromoppcells(occupied_cells)

    return max_layers, shapes, colors




def _get_levels_executed_from_interactions(interactions_path: str) -> Optional[int]:
    """Return levels executed so far using computelevels_genresponse() on occupied_cells.

    Reads Evaluation->genresponse list, picks last valid non-empty occupied_cells;
    falls back by traversing backward. Returns None if unavailable.
    """
    try:
        with open(interactions_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, None, None

    eval_block = data.get("Evaluation", {})
    genresponse = eval_block.get("genresponse", [])
    if not isinstance(genresponse, list) or not genresponse:
        return None, None, None

    occupied_cells = None
    for item in reversed(genresponse):
        if not isinstance(item, dict):
            continue
        oc = item.get("occupied_cells")
        if isinstance(oc, dict) and len(oc) > 0:
            occupied_cells = oc
            break

    if occupied_cells is None:
        return None, None, None

    try:
        return computelevels_genresponse(occupied_cells)
    except Exception:
        return None, None, None

def parse_interactions(interactions: Dict[str, Any]) -> Tuple[Dict[str, Any], int, Optional[int], Optional[int], Optional[int], Optional[str]]:
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
    play_turns = interactions.get("Played turns")
    n_turns = interactions.get("n_turns")
    gtcode = boardinfo.get("gtcode")
    boardsize = boardinfo.get("size")
    combo_name = boardinfo.get("combo_name")
    shapes_list = boardinfo.get("shapes")
    colors_list = boardinfo.get("colors")
    return code_data, num_turns, total_shapes, play_turns, n_turns, gtcode, boardsize, combo_name, shapes_list, colors_list


def parse_scores(scores: Dict[str, Any]) -> Tuple[Optional[float], Optional[bool]]:
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
        "play_turns": None,
        "n_turns": None,
        "reconstruction_status": None,
        "gtcode": None,
        "num_levels": None,
    }

    interactions_path = os.path.join(episode_path, "interactions.json")
    scores_path = os.path.join(episode_path, "scores.json")

    if os.path.exists(interactions_path):
        interactions = read_json_file(interactions_path) or {}
        code_data, num_turns, total_shapes, play_turns, n_turns, gtcode, boardsize, combo_name, shapes_list_gt, colors_list_gt = parse_interactions(interactions)
        stats["code_stats"] = code_data
        stats["dialog_turns"] = num_turns
        stats["total_shapes"] = total_shapes
        stats["play_turns"] = play_turns
        stats["n_turns"] = n_turns
        stats["gtcode"] = gtcode
        stats["combo_name"] = combo_name
        stats["shapes_list_gt"] = shapes_list_gt
        stats["colors_list_gt"] = colors_list_gt
        # Compute number of levels
        if gtcode:
            num_levels = computelevels(gtcode, boardsize)
            stats["num_levels"] = num_levels if num_levels is not None else "unknown"
        else:
            stats["num_levels"] = "unknown"
        # Get reconstruction_status from Evaluation section
        ev = interactions.get("Evaluation", {})
        stats["reconstruction_status"] = ev.get("reconstruction_status")

    if os.path.exists(scores_path):
        scores = read_json_file(scores_path) or {}
        main_score, reconst_status_from_scores = parse_scores(scores)
        # Use reconstruction_status from interactions.json if not already set
        if stats["reconstruction_status"] is None:
            stats["reconstruction_status"] = reconst_status_from_scores

        if isinstance(main_score, float) and np.isnan(main_score):
            stats["accuracy_vals"].append(0)
            stats["aborted_episodes"].append(os.path.basename(episode_path))
            stats["outcome"] = "aborted"
        elif main_score == 0:
            stats["failed_episodes"].append(os.path.basename(episode_path))
            stats["outcome"] = "failed"
            stats["accuracy_vals"].append(0)
            
            # Determine if episode is conclusive_wrong or non_conclusive (only for failed episodes)
            reconst_status = stats["reconstruction_status"]
            play_turns = stats["play_turns"]
            n_turns = stats["n_turns"]
            
            if reconst_status is False:
                if play_turns is not None and n_turns is not None:
                    if play_turns == n_turns:
                        stats["outcome"] = "non_conclusive"
                    elif play_turns < n_turns:
                        stats["outcome"] = "conclusive_wrong"
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
                conclusive_wrong_episodes: List[str] = []
                non_conclusive_episodes: List[str] = []
                code_stats: Dict[str, Any] = {}
                # dialog_stats_by_shape maps shape_key -> list of (episode, turns)
                dialog_stats_by_shape: Dict[str, List[Tuple[str, int]]] = {}
                shape_stats: Dict[str, Dict[str, int]] = {}
                shape_code_stats: Dict[str, Dict[str, Any]] = {}
                # Track level-wise stats for each shape: {shape_key: {level_key: {total, success, failure}}}
                shape_level_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
                # failed_episodes_by_shape now has structure: {shape_key: {level: [episodes]}}
                failed_episodes_by_shape: Dict[str, Dict[str, List[str]]] = {}
                conclusive_wrong_episodes_by_shape: Dict[str, List[str]] = {}
                non_conclusive_episodes_by_shape: Dict[str, List[str]] = {}

                episodes = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                num_episodes = len(episodes)

                for episode in episodes:
                    episode_path = os.path.join(exp_path, episode)
                    ep_stats = process_episode(episode_path)
                    # determine shape key early so it's available to all aggregations
                    total_shapes = ep_stats.get("total_shapes")
                    shape_key_str = str(total_shapes) if total_shapes is not None else "unknown"
                    # get level information
                    num_levels = ep_stats.get("num_levels")
                    level_key_str = f"Level-{num_levels}" if num_levels is not None else "Level-unknown"

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
                            "conclusive_wrong": 0,
                            "non_conclusive": 0,
                        }
                    
                    # Initialize level stats for this shape if needed
                    if shape_key_str not in shape_level_stats:
                        shape_level_stats[shape_key_str] = {}
                    if level_key_str not in shape_level_stats[shape_key_str]:
                        shape_level_stats[shape_key_str][level_key_str] = {
                            "total": 0,
                            "success": 0,
                            "failure": 0,
                        }
                    
                    outcome = ep_stats.get("outcome", "unknown")
                    # Update level stats
                    shape_level_stats[shape_key_str][level_key_str]["total"] += 1
                    # Pre-compute levels executed info from interactions.json
                    levels_exec_val, shapes_gen, colors_gen = _get_levels_executed_from_interactions(os.path.join(episode_path, "interactions.json"))
                    shapes_gen_count = len(shapes_gen) if shapes_gen else 0
                    shapes_gt_count = len(ep_stats.get("shapes_list_gt", [])) if ep_stats.get("shapes_list_gt") else 0
                    shapes_gen_pct = round((shapes_gen_count / shapes_gt_count), 2) if shapes_gt_count > 0 else 0.0
                    ep_info = {
                        "episode": episode,
                        "levels_executed": levels_exec_val if levels_exec_val is not None else "unknown",
                        "gt_shapes": ep_stats.get("shapes_list_gt"),
                        "gt_colors": ep_stats.get("colors_list_gt"),
                        "gen_shapes": shapes_gen,
                        "gen_colors": colors_gen,
                        "gen_shapes_count": shapes_gen_count,
                        "gen_shapes_pct_of_gt": shapes_gen_pct,
                    }
                    
                    if outcome == "success":
                        shape_stats[shape_key_str]["reconst_success"] += 1
                        shape_level_stats[shape_key_str][level_key_str]["success"] += 1
                    elif outcome == "failed":
                        shape_stats[shape_key_str]["reconst_failure"] += 1
                        shape_level_stats[shape_key_str][level_key_str]["failure"] += 1
                        # record failed episode under its shape key and level
                        if shape_key_str not in failed_episodes_by_shape:
                            failed_episodes_by_shape[shape_key_str] = {}
                        failed_episodes_by_shape[shape_key_str].setdefault(level_key_str, []).append(ep_info)
                    elif outcome == "aborted":
                        shape_stats[shape_key_str]["aborted"] += 1
                        # Aborted doesn't count in level success/failure
                    elif outcome == "conclusive_wrong":
                        shape_stats[shape_key_str]["conclusive_wrong"] += 1
                        shape_stats[shape_key_str]["reconst_failure"] += 1
                        shape_level_stats[shape_key_str][level_key_str]["failure"] += 1
                        conclusive_wrong_episodes.append(episode)
                        conclusive_wrong_episodes_by_shape.setdefault(shape_key_str, []).append(episode)
                        # conclusive_wrong is also a failed episode - categorize by level
                        if shape_key_str not in failed_episodes_by_shape:
                            failed_episodes_by_shape[shape_key_str] = {}
                        failed_episodes_by_shape[shape_key_str].setdefault(level_key_str, []).append(ep_info)
                    elif outcome == "non_conclusive":
                        shape_stats[shape_key_str]["non_conclusive"] += 1
                        shape_stats[shape_key_str]["reconst_failure"] += 1
                        shape_level_stats[shape_key_str][level_key_str]["failure"] += 1
                        non_conclusive_episodes.append(episode)
                        non_conclusive_episodes_by_shape.setdefault(shape_key_str, []).append(episode)
                        # non_conclusive is also a failed episode - categorize by level
                        if shape_key_str not in failed_episodes_by_shape:
                            failed_episodes_by_shape[shape_key_str] = {}
                        failed_episodes_by_shape[shape_key_str].setdefault(level_key_str, []).append(ep_info)
                    shape_stats[shape_key_str]["total"] += 1

                # Compute level summary for each shape
                failed_episodes_by_shape_summary: Dict[str, Dict[str, int]] = {}
                for shape_key, levels_dict in failed_episodes_by_shape.items():
                    failed_episodes_by_shape_summary[shape_key] = {
                        level: len(episodes_list) for level, episodes_list in levels_dict.items()
                    }

                # Enhance failed_episodes_by_shape with per-level executed summary (counts and percentages)
                for shape_key, levels_dict in list(failed_episodes_by_shape.items()):
                    for level_key, episodes_list in list(levels_dict.items()):
                        # episodes_list contains dicts {episode, levels_executed, gen_shapes_pct_of_gt, ...}
                        total_eps = len(episodes_list)
                        exec_counts: Dict[str, int] = {}
                        gen_shape_pcts: List[float] = []
                        for ep in episodes_list:
                            val = ep.get("levels_executed", "unknown")
                            key = str(val)
                            exec_counts[key] = exec_counts.get(key, 0) + 1
                            pct = ep.get("gen_shapes_pct_of_gt")
                            if pct is not None:
                                gen_shape_pcts.append(pct)
                        # Build summary with keys formatted as Level-{}
                        summary: Dict[str, Dict[str, Any]] = {}
                        for raw_key, cnt in exec_counts.items():
                            level_key_fmt = f"Level-{raw_key}"
                            pct = round((cnt / total_eps), 2) if total_eps > 0 else 0.0
                            summary[level_key_fmt] = {
                                "episodes": cnt,
                                "episodes_pct": pct,
                            }
                        # Add gen_shapes_pct summary if available
                        if gen_shape_pcts:
                            avg_gen_pct = round((sum(gen_shape_pcts) / len(gen_shape_pcts)), 2)
                            median_gen_pct = round(float(np.median(gen_shape_pcts)), 2)
                            min_gen_pct = round(min(gen_shape_pcts), 2)
                            max_gen_pct = round(max(gen_shape_pcts), 2)
                        else:
                            avg_gen_pct = 0.0
                            median_gen_pct = 0.0
                            min_gen_pct = 0.0
                            max_gen_pct = 0.0
                        # Replace list with richer dict holding episodes and summary
                        failed_episodes_by_shape[shape_key][level_key] = {
                            "episodes": episodes_list,
                            "levels_executed_summary": summary,
                            "gen_shapes_pct_summary": {
                                "avg": avg_gen_pct,
                                "median": median_gen_pct,
                                "min": min_gen_pct,
                                "max": max_gen_pct,
                            },
                        }

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
                    conclusive_wrong = v.get("conclusive_wrong", 0)
                    non_conclusive = v.get("non_conclusive", 0)

                    v["reconst_success_pct"] = round((reconst_succ / total), 2) if total > 0 else 0.0
                    v["reconst_failure_pct"] = round((reconst_fail / total), 2) if total > 0 else 0.0
                    v["aborted_pct"] = round((aborted / total), 2) if total > 0 else 0.0
                    v["conclusive_wrong_pct"] = round((conclusive_wrong / total), 2) if total > 0 else 0.0
                    v["non_conclusive_pct"] = round((non_conclusive / total), 2) if total > 0 else 0.0
                    reconst_episodes = total - aborted - reconst_fail
                    
                    # Add level-wise statistics to shape_stats
                    v["level_stats"] = {}
                    if k in shape_level_stats:
                        for level_key, level_data in shape_level_stats[k].items():
                            level_total = level_data.get("total", 0)
                            level_success = level_data.get("success", 0)
                            level_failure = level_data.get("failure", 0)
                            
                            # Get gen_shapes_pct_summary and counts from failed_episodes_by_shape if available
                            gen_shapes_pct_summary = {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
                            gen_shapes_count_summary = {"total": 0, "avg": 0.0, "median": 0.0, "min": 0, "max": 0}
                            
                            if k in failed_episodes_by_shape and level_key in failed_episodes_by_shape[k]:
                                level_failed_data = failed_episodes_by_shape[k][level_key]
                                if "gen_shapes_pct_summary" in level_failed_data:
                                    gen_shapes_pct_summary = level_failed_data["gen_shapes_pct_summary"]
                                
                                # Extract counts from episodes
                                episodes_list = level_failed_data.get("episodes", [])
                                if episodes_list:
                                    gen_counts = [ep.get("gen_shapes_count", 0) for ep in episodes_list if isinstance(ep, dict)]
                                    if gen_counts:
                                        gen_shapes_count_summary = {
                                            "total": sum(gen_counts),
                                            "avg": round((sum(gen_counts) / len(gen_counts)), 2),
                                            "median": round(float(np.median(gen_counts)), 2),
                                            "min": min(gen_counts),
                                            "max": max(gen_counts),
                                        }
                            
                            v["level_stats"][level_key] = {
                                "episodes": level_total,
                                "episodes_pct_of_total": round((level_total / total), 2) if total > 0 else 0.0,
                                "success": level_success,
                                "failure": level_failure,
                                "success_pct": round((level_success / level_total), 2) if level_total > 0 else 0.0,
                                "failure_pct": round((level_failure / level_total), 2) if level_total > 0 else 0.0,
                                "gen_shapes_pct_summary": gen_shapes_pct_summary,
                                "gen_shapes_count_summary": gen_shapes_count_summary,
                            }

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
                    "num_conclusive_wrong_episodes": len(conclusive_wrong_episodes),
                    "num_non_conclusive_episodes": len(non_conclusive_episodes),
                    "failed_episodes": failed_episodes,
                    "aborted_episodes": aborted_episodes,
                    "conclusive_wrong_episodes": conclusive_wrong_episodes,
                    "non_conclusive_episodes": non_conclusive_episodes,
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
                    "failed_episodes_by_shape_summary": failed_episodes_by_shape_summary,
                    "conclusive_wrong_episodes_by_shape": conclusive_wrong_episodes_by_shape,
                    "non_conclusive_episodes_by_shape": non_conclusive_episodes_by_shape,
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
                    print(f"Conclusive Wrong: {len(conclusive_wrong_episodes)}, Non-Conclusive: {len(non_conclusive_episodes)}")
                    print(f"reconst_accuracy: {reconst_accuracy}")

                results[game][model][exp] = save_data

    outpath = os.path.join(base_dir, "overall_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute overall scores from experiment directories")
    parser.add_argument("base_dir", nargs="?", default="/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/cocottr/r2", help="Base directory containing model results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()

    compute_scores(args.base_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()