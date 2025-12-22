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

def parse_scores(scores: Dict[str, Any]) -> Tuple[Optional[float], Optional[bool]]:
    es = scores.get("episode scores", {})
    main = es.get("Main Score")
    reconst = es.get("reconstruction_status")
    return main, reconst


def process_episode(episode_path: str) -> Dict[str, Any]:
    
    stats = {"reconstruction_status": [],
             "reuse_status": [],
             "repeat_status": [],
             "overall_success": [],
             "overall_loss": [],
             "overall_abort": []
             }    
    
    interactions_path = os.path.join(episode_path, "interactions.json")
    scores_path = os.path.join(episode_path, "scores.json")



    if os.path.exists(interactions_path):
        interactions = read_json_file(interactions_path) or {}
        ev = interactions.get("Evaluation", {})        
        reconst_status = 1 if ev.get("reconstruction_status") == True else 0
        reuse_status = 1 if ev.get("reuse_status") == True else 0
        repeat_status = 1 if ev.get("repeat_status") == True else 0
        stats["reconstruction_status"].append(reconst_status)
        stats["reuse_status"].append(reuse_status)
        stats["repeat_status"].append(repeat_status)
        if ev.get("Aborted") == True:
            stats["overall_abort"].append(1)
        elif ev.get("Lose") == True:
            stats["overall_loss"].append(1)
        elif ev.get("Success") == True:
            stats["overall_success"].append(1)
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

                episodes = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                num_episodes = len(episodes)

                for episode in episodes:
                    episode_path = os.path.join(exp_path, episode)
                    ep_stats = process_episode(episode_path)
                    accuracy_data.extend(ep_stats.get("overall_success", []))
                    failed_episodes.extend(ep_stats.get("overall_loss", []))                   
                    aborted_episodes.extend(ep_stats.get("overall_abort", []))

                total_aborted = len(aborted_episodes)
                total_failed = len(failed_episodes)
                total_episodes = num_episodes
                aborted_percent = round((total_aborted / total_episodes),2) if total_episodes > 0 else 0.0
                failed_percent = round((total_failed / total_episodes),2) if total_episodes > 0 else 0.0
                success_percent = round(1.0 - aborted_percent - failed_percent,2)

                print(f"Number of Episodes: {total_episodes}, Success: {success_percent}, Aborted: {aborted_percent}, Failed: {failed_percent}")


def main():
    parser = argparse.ArgumentParser(description="Compute overall scores from experiment directories")
    parser.add_argument("base_dir", nargs="?", default="/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/cocottr/r1", help="Base directory containing model results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()

    compute_scores(args.base_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()