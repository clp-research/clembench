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
    
    stats = {"reuse_status": [],
             "overall_success": [],
             "overall_loss": [],
             "overall_abort": [],
             "num_reuse_turns": 0,
             "from_scratch": 0,
             }    
    
    interactions_path = os.path.join(episode_path, "interactions.json")
    scores_path = os.path.join(episode_path, "scores.json")



    if os.path.exists(interactions_path):
        interactions = read_json_file(interactions_path) or {}
        ev = interactions.get("Evaluation", {})        
        reuse_status = 1 if ev.get("reuse_success") == True else 0
        stats["reuse_status"].append(reuse_status)
        if interactions.get("Aborted") == True:
            stats["overall_abort"].append(1)
        elif interactions.get("Lose") == True:
            stats["overall_loss"].append(1)
        elif interactions.get("Success") == True:
            stats["overall_success"].append(1)

        stats["num_reuse_turns"] = ev.get("play_turns_reuse")
        #stats["from_scratch"] = int(ev.get("used_oracle_code_as_skill_not_available"))

        if ev["use_oracle_code"] or ev["used_oracle_code_as_skill_not_available"]:
            stats["from_scratch"]  = 1
        else:
            stats["from_scratch"]  = 0
            

    return stats


def compute_scores(base_dir: str, verbose: bool = True, use_gpt_skills=False) -> Dict[str, Any]:

    with open("/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/skillreuse/resources/data/en/learnedskills_r3_2_clp-chat.json", "r") as f:
        clp_skills_data = json.load(f)

    clp_skill_len = 0#len(clp_skills_data)

    with open("/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/skillreuse/resources/data/en/learnedskills_r3_2_gptcodex.json", "r") as f:
        gpt_skills_data = json.load(f)

    gpt_skill_len = 0#len(gpt_skills_data)

    if use_gpt_skills:
        skill_len = gpt_skill_len
    else:
        skill_len = clp_skill_len


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
                reuse_success_episodes: List[int] = []
                reuse_turns: List[int] = []
                overall_success: int = 0
                from_scratch_episodes: List[int] = []

                episodes = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                num_episodes = len(episodes)

                for episode in episodes:
                    episode_path = os.path.join(exp_path, episode)
                    ep_stats = process_episode(episode_path)
                    #if ep_stats["overall_success"]:
                    #   overall_success += 1
                    accuracy_data.extend(ep_stats.get("overall_success", []))
                    failed_episodes.extend(ep_stats.get("overall_loss", []))                   
                    aborted_episodes.extend(ep_stats.get("overall_abort", []))
                    reuse_success_episodes.extend(ep_stats.get("reuse_status", []))

                    if ep_stats["reuse_status"] and ep_stats["reuse_status"][0]==1:
                        reuse_turns.append(ep_stats.get("num_reuse_turns"))

                        if ep_stats["from_scratch"] == 1:
                            from_scratch_episodes.append(1)

                total_aborted = len(aborted_episodes)
                total_failed = len(failed_episodes)
                total_episodes = num_episodes
                aborted_percent = round((total_aborted / total_episodes),2) if total_episodes > 0 else 0.0
                failed_percent = round((total_failed / total_episodes),2) if total_episodes > 0 else 0.0
                success_percent = round((sum(accuracy_data)/total_episodes),2)

                total_reuse_success = sum(reuse_success_episodes)
                reuse_rate = round((total_reuse_success / total_episodes),2) if total_episodes > 0 else 0.0

                total_scratch_success = len(from_scratch_episodes)
                print(total_scratch_success, total_reuse_success)
                reuse_api_success_eps = total_reuse_success - total_scratch_success
                reuse_api_success_rate = round((reuse_api_success_eps/total_episodes), 2)

                total_scratch_eps = (165-skill_len)*3
                print(total_scratch_eps, skill_len)
                reuse_scratch_success_rate = round((total_scratch_success/total_scratch_eps), 2)


                print(f"Number of Episodes: {total_episodes}, Success: {success_percent}, Aborted: {aborted_percent}, Failed: {failed_percent}")
                print(f"Reuse Success Rate: {reuse_rate}")
                print(f"Reuse API Success Rate: {reuse_api_success_rate}, Scratch Success Rate: {reuse_scratch_success_rate}")

                min_reuse_turns, median_reuse_turns, max_reuse_turns = np.min(reuse_turns), np.median(reuse_turns), np.max(reuse_turns)
                print(f"Min Reuse turns: {min_reuse_turns}, Median Reuse turns: {median_reuse_turns}, Max Reuse turns: {max_reuse_turns}")



def main():
    parser = argparse.ArgumentParser(description="Compute overall scores from experiment directories")
    parser.add_argument("base_dir", nargs="?", default="r1", help="Base directory containing model results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()

    compute_scores(args.base_dir, verbose=not args.quiet, use_gpt_skills=True)


if __name__ == "__main__":
    main()