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
             "optim_status": [],
             "reuse_status": [],
             "repeat_status": [],
             "ttr_status": [],
             "overall_success": [],
             "overall_loss": [],
             "overall_abort": [],
             "num_reconst_turns": 0,
             "num_reuse_turns": 0,
             "num_repeat_turns": 0,
             "num_total_turns": 0,                                       
             }    
    
    interactions_path = os.path.join(episode_path, "interactions.json")
    scores_path = os.path.join(episode_path, "scores.json")



    if os.path.exists(interactions_path):
        interactions = read_json_file(interactions_path) or {}
        ev = interactions.get("Evaluation", {})        
        reconst_status = 1 if ev.get("reconstruction_status") == True else 0
        optimization_status = 1 if ev.get("optim_success") == True else 0
        reuse_status = 1 if ev.get("reuse_success") == True else 0
        repeat_status = 1 if ev.get("repeat_success") == True else 0
        ttr_status = 1 if ev.get("ttr_success") == True else 0
        stats["reconstruction_status"].append(reconst_status)
        stats["optim_status"].append(optimization_status)        
        stats["reuse_status"].append(reuse_status)
        stats["repeat_status"].append(repeat_status)
        stats["ttr_status"].append(ttr_status)        
        if ev.get("Aborted") == True:
            stats["overall_abort"].append(1)
        elif ev.get("Lose") == True:
            stats["overall_loss"].append(1)
        elif ev.get("ttr_success") == True:
            stats["overall_success"].append(1)

        stats["num_reconst_turns"] = ev.get("play_turns_reconst")
        stats["num_optim_turns"] = ev.get("play_turns_optim")        
        stats["num_reuse_turns"] = ev.get("play_turns_reuse")
        stats["num_repeat_turns"] = ev.get("play_turns_repeat") 
        stats["num_total_turns"] = ev.get("play_turns_total") 

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
                reconst_success_episodes: List[int] = []
                optim_success_episodes: List[int] = []
                reuse_success_episodes: List[int] = []
                repeat_success_episodes: List[int] = []
                ttr_success_episodes: List[int] = []
                repeat_failed_episodes: List[str] = []
                reconst_turns: List[int] = []
                optim_turns: List[int] = []                
                reuse_turns:  List[int] = []
                repeat_turns:  List[int] = []
                overall_turns:  List[int] = []

                episodes = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                num_episodes = len(episodes)

                for episode in episodes:
                    episode_path = os.path.join(exp_path, episode)
                    ep_stats = process_episode(episode_path)
                    accuracy_data.extend(ep_stats.get("overall_success", []))
                    failed_episodes.extend(ep_stats.get("overall_loss", []))                   
                    aborted_episodes.extend(ep_stats.get("overall_abort", []))
                    reconst_success_episodes.extend(ep_stats.get("reconstruction_status", []))
                    optim_success_episodes.extend(ep_stats.get("optim_status", []))
                    reuse_success_episodes.extend(ep_stats.get("reuse_status", []))
                    repeat_success_episodes.extend(ep_stats.get("repeat_status", []))
                    ttr_success_episodes.extend(ep_stats.get("ttr_status", []))

                    if ep_stats["reconstruction_status"] and ep_stats["reconstruction_status"][0]==1:
                        reconst_turns.append(ep_stats.get("num_reconst_turns"))
                    if ep_stats["optim_status"] and ep_stats["optim_status"][0]==1:
                        optim_turns.append(ep_stats.get("num_optim_turns"))                        
                    if ep_stats["reuse_status"] and ep_stats["reuse_status"][0]==1:
                        reuse_turns.append(ep_stats.get("num_reuse_turns"))
                    if ep_stats["repeat_status"] and ep_stats["repeat_status"][0]==1:
                            repeat_turns.append(ep_stats.get("num_repeat_turns"))
                    
                    if ep_stats["reconstruction_status"] and ep_stats["reconstruction_status"][0]==1 and ep_stats["optim_status"] and ep_stats["optim_status"][0]==1 and ep_stats["reuse_status"] and ep_stats["reuse_status"][0] == 1 and ep_stats["repeat_status"] and ep_stats["repeat_status"][0] == 0:
                            repeat_failed_episodes.append(episode)
                    overall_turns.append(ep_stats.get("num_total_turns"))                                                            

                total_aborted = len(aborted_episodes)
                total_failed = len(failed_episodes)
                total_episodes = num_episodes
                aborted_percent = round((total_aborted / total_episodes),2) if total_episodes > 0 else 0.0
                failed_percent = round((total_failed / total_episodes),2) if total_episodes > 0 else 0.0
                success_percent = round(1.0 - aborted_percent - failed_percent,2)

                total_reconst_success = sum(reconst_success_episodes)
                reconst_rate = round((total_reconst_success / total_episodes),2) if total_episodes > 0 else 0.0

                total_optim_success = sum(optim_success_episodes)
                optim_rate = round((total_optim_success / total_episodes),2) if total_episodes > 0 else 0.0

                total_reuse_success = sum(reuse_success_episodes)
                reuse_rate = round((total_reuse_success / total_episodes),2) if total_episodes > 0 else 0.0
                total_repeat_success = sum(repeat_success_episodes)
                repeat_rate = round((total_repeat_success / total_episodes),2) if total_episodes > 0 else 0.0
                total_ttr_success = sum(ttr_success_episodes)
                ttr_rate = round((total_ttr_success / total_episodes),2) if total_episodes > 0 else 0.0


                print(f"Number of Episodes: {total_episodes}, Success: {success_percent}, Aborted: {aborted_percent}, Failed: {failed_percent}")
                print(f"Reconstruction Success Rate: {reconst_rate}, Optim Success Rate: {optim_rate}, Reuse Success Rate: {reuse_rate}, Repeat Success Rate: {repeat_rate} TTR Rate: {ttr_rate}")

                min_reconst_turns, median_reconst_turns, max_reconst_turns = np.min(reconst_turns), np.median(reconst_turns), np.max(reconst_turns)
                print(f"Min Reconst turns: {min_reconst_turns}, Median Reconst turns: {median_reconst_turns}, Max Reconst turns: {max_reconst_turns}")


                if optim_turns:
                    min_optim_turns, median_optim_turns, max_optim_turns = np.min(optim_turns), np.median(optim_turns), np.max(optim_turns)
                    print(f"Min Optim turns: {min_optim_turns}, Median Optim turns: {median_optim_turns}, Max Optim turns: {max_optim_turns}")


                if reuse_turns:
                    min_reuse_turns, median_reuse_turns, max_reuse_turns = np.min(reuse_turns), np.median(reuse_turns), np.max(reuse_turns)
                    print(f"Min Reuse turns: {min_reuse_turns}, Median Reuse turns: {median_reuse_turns}, Max Reuse turns: {max_reuse_turns}")


                if repeat_turns:
                    min_repeat_turns, median_repeat_turns, max_repeat_turns = np.min(repeat_turns), np.median(repeat_turns), np.max(repeat_turns)
                    print(f"Min Repeat turns: {min_repeat_turns}, Median Repeat turns: {median_repeat_turns}, Max Repeat turns: {max_repeat_turns}")                

                #with open("repeat_failed_rp4-2.json", "w") as f:
                #    json.dump(repeat_failed_episodes, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Compute overall scores from experiment directories")
    parser.add_argument("base_dir", nargs="?", default="/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/cocottr_optimizer/rponlyreconst_2", help="Base directory containing model results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()

    compute_scores(args.base_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()