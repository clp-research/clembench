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


def isinstanceinfoavailable(episode_path):
    interactions_path = os.path.join(episode_path, "interactions.json")
    if os.path.exists(interactions_path):
        return True
    return False

def _prepare_instance_info(episode_path):
    interaction_data = {}
    interactions_path = os.path.join(episode_path, "interactions.json")
    if os.path.exists(interactions_path):
        interactions = read_json_file(interactions_path) or {}
        interaction_data["overall_success"] = interactions["Success"]
        interaction_data["overall_abort"] = interactions["Aborted"]
        interaction_data["overall_loss"] = interactions["Lose"]                

        ev = interactions.get("Evaluation", {})
        boardinfo = ev["boardinfo"]["simple_reuse"]
        interaction_data["shapes"] = boardinfo["shapes"]
        interaction_data["num_shapes"] = len(interaction_data["shapes"])
        interaction_data["comboname"] = boardinfo["combo_name"]
        interaction_data["colors"] = boardinfo["colors"]
        interaction_data["reuse_gt_code_board"] = {"function": boardinfo["code"]["single_turn"]["function"],
                        "usage": boardinfo["code"]["single_turn"]["usage"],}
        interaction_data["reuse_gt_code_validation"] = ev["used_gtcode_for_validation"]
        interaction_data["reuseinput_code"] = ev["reuse_input_data"]
        #interaction_data["reuseplay_turns"] = ev["play_turns"]
        interaction_data["reuseplay_turns"] = len(ev["genresponse"]["reuse"])

        interaction_data["useskills"] = ev["use_skills"]
        interaction_data["skillsfilename"] = ev["existing_skills_filename"]
        interaction_data["skillscode"] = ev["skills_code"]
        interaction_data["use_oracle_code"] = ev["use_oracle_code"]
        interaction_data["used_oracle_code_as_skill_not_available"] = ev["used_oracle_code_as_skill_not_available"]
        interaction_data["reuse_success"] = ev["reuse_success"]
        interaction_data["genresponse"] = ev["genresponse"]
        interaction_data["turncode"] = interaction_data["genresponse"]["reuse"]
        interaction_data["geninstructions"] = {}
        interaction_data["gencode"] = {}
        interaction_data["gencfq"] = {}
        interaction_data["genothers"] = {}
        for index, turn in enumerate(interaction_data["turncode"]):
            if "instruction" not in turn:
                if interaction_data["overall_abort"]:
                    continue

                print(f"Episode {episode_path} is missing instructions in genresponse\n{turn}")
                input()
            interaction_data["geninstructions"][index+1] = turn["instruction"]

            if "response" not in turn:
                if interaction_data["overall_abort"]:
                    continue
                print(f"Episode {episode_path} is missing response in genresponse\n{turn}")
                input()

            resp_status = turn["response"]["status"]
            if resp_status == "code":
                interaction_data["gencode"][index+1] = turn["response"]["details"]
            elif resp_status == "clarification":
                interaction_data["gencfq"][index+1] = turn["response"]["details"]
            else:
                interaction_data["genothers"][index+1] = {turn["response"]["status"]: turn["response"]["details"]}

        interaction_data["reuse_response"] = ev.get("reuse_genresponse", {})
        reuse_response = interaction_data["reuse_response"] 
        interaction_data["used_clarification"] = reuse_response["used_clarification"]
        interaction_data["num_clarification"] = reuse_response["num_clarifications"]        
        interaction_data["used_remove"] = reuse_response["used_remove"]
        interaction_data["num_removes"] = reuse_response["num_removes"]
        interaction_data["used_move"] = reuse_response["used_move"]
        interaction_data["num_move"] = reuse_response["num_moves"]
        interaction_data["used_undo"] = reuse_response["used_undo"]
        interaction_data["num_undo"] = reuse_response["num_undos"]
        interaction_data["used_clear"] = reuse_response["used_clear"]
        interaction_data["num_clear"] = reuse_response["num_clears"]
        return interaction_data
    else:
        return None


def process_episode(episode_path: str, statsdict) -> Dict[str, Any]:
    
    interavail = isinstanceinfoavailable(episode_path)
    if not interavail:
        return

    interdata = _prepare_instance_info(episode_path)
    if not interdata:
        return

    episode_num = episode_path.split("/")[-1]


    if interdata["overall_success"] != interdata["reuse_success"]:
        print(f"Difference between overall success and reuse success flag for episode: {episode_num}")
        input()

    play_turns = interdata["reuseplay_turns"]

    if len(interdata["gencode"]) > play_turns:
        print(f"Code Generation turns are more than play turns: {episode_num}")
        print(f'play_turns: {play_turns}, gencodelength: {len(interdata["gencode"])}')
        input()


    if interdata["overall_loss"]:
        statsdict["faileps"].append({episode_num:play_turns})
        statsdict["shape_stats"][interdata["num_shapes"]]["faileps"] += 1

    elif interdata["overall_success"]:
        statsdict["successeps"].append({episode_num:play_turns})
        statsdict["shape_stats"][interdata["num_shapes"]]["successeps"] += 1

    elif interdata["overall_abort"]:
        statsdict["aborteps"].append({episode_num:play_turns})
        statsdict["shape_stats"][interdata["num_shapes"]]["aborteps"] += 1


    if interdata["gencfq"] and interdata["used_clarification"] == False:
        print(f"Difference between cfq flag and genresponse for episode: {episode_num}")
        input()

    if interdata["skillscode"] is None and interdata["use_skills"]:
        print(f"Difference between skillcode data and useskills flag for episode: {episode_num}")
        input()

    if interdata["used_clarification"]:
        statsdict["clarification_eps"].append({interdata["num_shapes"]:{interdata["comboname"]:{episode_num:play_turns, "success": interdata["overall_success"], "abort": interdata["overall_success"]}}})

    if interdata["used_remove"] or interdata["used_move"] or interdata["used_clear"]:
        statsdict["correction_eps"].append({interdata["num_shapes"]:{interdata["comboname"]:{episode_num:play_turns, "success": interdata["overall_success"], "abort": interdata["overall_success"]}}})

    if interdata["used_undo"]:
        statsdict["undo_eps"].append({episode_num:play_turns})

    statsdict["num_turns"].append({episode_num:play_turns})

    if interdata["used_oracle_code_as_skill_not_available"]:
        statsdict["skill_unavail_eps"].append({interdata["num_shapes"]:{interdata["comboname"]:{episode_num:play_turns}}})

    if interdata["use_oracle_code"]:
        statsdict["skill_oracle_eps"].append({episode_num:play_turns})


    statsdict["gencode"].append({episode_num:interdata["gencode"]})
    statsdict["geninstructions"].append({episode_num:interdata["geninstructions"]})

    statsdict["num_turns_code_gen"].append({episode_num:{len(interdata["gencode"]): interdata["gencode"]}})
    statsdict["num_skills"] = len(interdata["skillscode"])

    if interdata["skillsfilename"] not in statsdict["skillsfilename"]:
        statsdict["skillsfilename"][interdata["skillsfilename"]] = 0
    statsdict["skillsfilename"][interdata["skillsfilename"]] += 1


    if interdata["comboname"] not in interdata["skillscode"]:
        statsdict["skillnotavailcombos"].append(interdata["comboname"])
        statsdict["skillnotavailepscnt"]+=1
        if interdata["overall_success"]:
            statsdict["successepsfromscratch"].append({episode_num:play_turns})
        elif interdata["overall_abort"]:
            statsdict["abortepsfromscratch"].append({episode_num:play_turns})
        elif interdata["overall_loss"]:
            statsdict["failepsfromscratch"].append({episode_num:play_turns})

    elif interdata["comboname"] in interdata["skillscode"]:
        statsdict["skillavailcombos"].append(interdata["comboname"])
        statsdict["skillavailepscnt"]+=1
        if interdata["overall_success"]:
            statsdict["successepsskill"].append({episode_num:play_turns})
        elif interdata["overall_abort"]:
            statsdict["abortepsskill"].append({episode_num:play_turns})
        elif interdata["overall_loss"]:
            statsdict["failepsskill"].append({episode_num:play_turns})


def _process_correction_data(expstats):
    if expstats is None:
        return

    correpisodes = expstats["correction_eps"]
    num_success = 0
    num_abort = 0
    num_failure = 0
    corrdata = {}
    for data in correpisodes:
        shapeval = list(data.keys())[0]
        if shapeval not in corrdata:
            corrdata[shapeval] = {}
        combodata = data[shapeval]
        comboname = list(combodata.keys())[0]
        if comboname not in corrdata[shapeval]:
            corrdata[shapeval][comboname] = {}
        if combodata[comboname]["success"]:
            num_success+=1
        elif combodata[comboname]["abort"]:
            num_abort+=1
        else:
            num_failure+=1
        combodata[comboname].pop("success")
        combodata[comboname].pop("abort")
        corrdata[shapeval][comboname].update(combodata[comboname])  

    corr_counts_per_shape = {}
    corr_turns_per_shape = {}
    corr_eps_per_shape = {}
    corr_combos_per_shape = {}
    for shapeval, data in corrdata.items():
        corr_counts_per_shape[shapeval] = 0
        corr_turns_per_shape[shapeval] = []
        corr_combos_per_shape[shapeval] = []
        corr_eps_per_shape[shapeval] = []
        for comboname, epdata in data.items():
            corr_combos_per_shape[shapeval].append(comboname)
            corr_counts_per_shape[shapeval] += len(data[comboname])
            corr_turns_per_shape[shapeval].append(list(epdata.values())[0])
            episode_name = list(epdata.keys())[0]
            corr_eps_per_shape[shapeval].append(episode_name)


    corrdetails = {"num_corr_eps": len(correpisodes), "num_success": num_success, "num_abort": num_abort, "num_failure": num_failure,
                   "details": {"corr_counts_per_shape": corr_counts_per_shape, "corr_turns_per_shape": corr_turns_per_shape, "corr_eps_per_shape": corr_eps_per_shape, "corr_combos_per_shape": corr_combos_per_shape}}

    if len(correpisodes):
        corrdetails["success"] = round((num_success/len(correpisodes)),3)
        corrdetails["abort"] = round((num_abort/len(correpisodes)),3)
        corrdetails["failure"] = round((num_failure/len(correpisodes)),3)
    else:
        corrdetails["success"] = 0
        corrdetails["abort"] = 0
        corrdetails["failure"] = 0


    
    return corrdetails


def _process_cfq_data(expstats):
    if expstats is None:
        return

    cfqepisodes = expstats["clarification_eps"]
    num_success = 0
    num_abort = 0
    num_failure = 0
    cfqdata = {}
    cfq_success_per_shape = {}    
    for data in cfqepisodes:
        shapeval = list(data.keys())[0]
        if shapeval not in cfqdata:
            cfqdata[shapeval] = {}
        if shapeval not in cfq_success_per_shape:
            cfq_success_per_shape[shapeval] = {"successep": 0, "abortep": 0, "failureep": 0}
        combodata = data[shapeval]
        comboname = list(combodata.keys())[0]
        if comboname not in cfqdata[shapeval]:
            cfqdata[shapeval][comboname] = {}
        if combodata[comboname]["success"]:
            num_success+=1
            cfq_success_per_shape[shapeval]["successep"] += 1
        elif combodata[comboname]["abort"]:
            num_abort+=1
            cfq_success_per_shape[shapeval]["abortep"] += 1
        else:
            num_failure+=1
            cfq_success_per_shape[shapeval]["failureep"] += 1
        combodata[comboname].pop("success")
        combodata[comboname].pop("abort")
        cfqdata[shapeval][comboname].update(combodata[comboname])  

    for shapeval in cfq_success_per_shape:
        toteps = cfq_success_per_shape[shapeval]["successep"] + cfq_success_per_shape[shapeval]["abortep"] + cfq_success_per_shape[shapeval]["failureep"]

        if toteps:
            success = round((cfq_success_per_shape[shapeval]["successep"]/toteps),3)
            abort = round((cfq_success_per_shape[shapeval]["abortep"]/toteps),3)
            failure = round((cfq_success_per_shape[shapeval]["failureep"]/toteps),3)
        else:
            success = 0
            abort = 0
            failure = 0
        cfq_success_per_shape[shapeval]["success"] = success
        cfq_success_per_shape[shapeval]["abort"] = abort
        cfq_success_per_shape[shapeval]["failure"] = failure


    cfq_counts_per_shape = {}
    cfq_turns_per_shape = {}
    cfq_eps_per_shape = {}
    cfq_combos_per_shape = {}
    for shapeval, data in cfqdata.items():
        cfq_counts_per_shape[shapeval] = 0
        cfq_turns_per_shape[shapeval] = []
        cfq_combos_per_shape[shapeval] = []
        cfq_eps_per_shape[shapeval] = []
        for comboname, epdata in data.items():
            cfq_combos_per_shape[shapeval].append(comboname)
            cfq_counts_per_shape[shapeval] += len(data[comboname])
            cfq_turns_per_shape[shapeval].append(list(epdata.values())[0])
            episode_name = list(epdata.keys())[0]
            cfq_eps_per_shape[shapeval].append(episode_name)


    cfqdetails = {"num_cfq_eps": len(cfqepisodes), "num_success": num_success, "num_abort": num_abort, "num_failure": num_failure,
                  "details": {"cfq_counts_per_shape": cfq_counts_per_shape, "cfq_success_per_shape": cfq_success_per_shape,
                   "cfq_turns_per_shape": cfq_turns_per_shape, "cfq_eps_per_shape": cfq_eps_per_shape, "cfq_combos_per_shape": cfq_combos_per_shape}}

    if len(cfqepisodes):
        cfqdetails["success"] = round((num_success/len(cfqepisodes)),3)
        cfqdetails["abort"] = round((num_abort/len(cfqepisodes)),3)
        cfqdetails["failure"] = round((num_failure/len(cfqepisodes)),3)
    else:
        cfqdetails["success"] = 0
        cfqdetails["abort"] = 0
        cfqdetails["failure"] = 0


    
    return cfqdetails

def _process_unavailskill_episodes(expstats):
    if expstats is None:
        return

    skillunavaileps = expstats["skill_unavail_eps"]

    shapedata = {}
    for shapecnt in skillunavaileps:
        shapeval = list(shapecnt.keys())[0]
        if shapeval not in shapedata:
            shapedata[shapeval] = {}
        combodata = shapecnt[shapeval]
        comboname = list(combodata.keys())[0]
        if comboname not in shapedata[shapeval]:
            shapedata[shapeval][comboname] = {}
        shapedata[shapeval][comboname].update(combodata[comboname])

    totalmissedskills = 0
    combocounts = {}
    combonamecounts = {}
    for shapeval, data in shapedata.items():
        totalmissedskills += len(list(data.keys()))
        combocounts[shapeval] = len(data)
        combonamecounts[shapeval] = {}
        for comboname, epdata in data.items():
            if comboname not in combonamecounts[shapeval]:
                combonamecounts[shapeval][comboname] = 0
            combonamecounts[shapeval][comboname] += len(epdata)

    print(f"Missed skills for {totalmissedskills} combos")
    skillunavailcombodata = {"num_combos":totalmissedskills, "combo_counts_per_shape": combocounts,
                             "combonames_counts": combonamecounts }
    return skillunavailcombodata

def _process_skillusage(expstats):
    if expstats is None:
        return

    totaleps = expstats["num_episodes"]
    totalskills = expstats["num_skills"]
    totaleps_skill_sum = len(expstats["successepsskill"]) + len(expstats["abortepsskill"]) + len(expstats["failepsskill"])
    totaleps_skill = expstats["skillavailepscnt"]
    if totaleps_skill != totaleps_skill_sum:
        print(f"Diff between eps skill totaleps_skill: {totaleps_skill}, totaleps_skill_sum: {totaleps_skill_sum}")
        input()
    totaleps_nonskill_sum = len(expstats["successepsfromscratch"]) + len(expstats["abortepsfromscratch"]) + len(expstats["failepsfromscratch"])
    totaleps_nonskill = expstats["skillnotavailepscnt"]
    if totaleps_nonskill != totaleps_nonskill_sum:
        print(f"Diff between eps skill totaleps_nonskill: {totaleps_nonskill}, totaleps_nonskill_sum: {totaleps_nonskill_sum}")
        input()


    eps_skill_avail = {"num_episodes": totaleps_skill, 
                         "num_success": len(expstats["successepsskill"]),
                         "num_abort": len(expstats["abortepsskill"]),
                         "num_failure": len(expstats["failepsskill"]),
                         "success":round((len(expstats["successepsskill"])/totaleps_skill), 3),
                         "overall_success": round((len(expstats["successepsskill"])/totaleps), 3),                         
                         "abort":round((len(expstats["abortepsskill"])/totaleps_skill), 3),
                         "failure":round((len(expstats["failepsskill"])/totaleps_skill), 3),}

    eps_skill_notavail = {"num_episodes": totaleps_nonskill, 
                         "num_success": len(expstats["successepsfromscratch"]),
                         "num_abort": len(expstats["abortepsfromscratch"]),
                         "num_failure": len(expstats["failepsfromscratch"]),    
                         "success":round((len(expstats["successepsfromscratch"])/totaleps_nonskill), 3),
                         "overall_success": round((len(expstats["successepsfromscratch"])/totaleps), 3),
                         "abort":round((len(expstats["abortepsfromscratch"])/totaleps_nonskill), 3),
                         "failure":round((len(expstats["failepsfromscratch"])/totaleps_nonskill), 3),}
    eps_skill_details = {"num_skill_avail_eps": totaleps_skill, "num_skill_nonavail_eps": totaleps_nonskill,
                         "per_skill_avail_eps": round((totaleps_skill/totaleps),3),
                          "per_skill_nonavail_eps": round((totaleps_nonskill/totaleps),3),
                         "skillavaildata": eps_skill_avail,
                         "skillnonavaildata": eps_skill_notavail }

    print(f"TotalSkillAvailEps: {totaleps_skill}, ActualAvailSkills_Eps: ({totalskills}, {totalskills*3}), SkillNonAvailEps: {totaleps_nonskill}")
    print(f'eps_skill_avail[success,abort,failure]: {eps_skill_avail["success"]}, {eps_skill_avail["abort"]}, {eps_skill_avail["failure"]}')
    print(f'eps_skill_notavail[success,abort,failure]: {eps_skill_notavail["success"]}, {eps_skill_notavail["abort"]}, {eps_skill_notavail["failure"]}')
    print(f'eps_skill_avail[overall]: {eps_skill_avail["overall_success"]}, ps_skill_notavail[overall]: {eps_skill_notavail["overall_success"]}')

    skillunavailcombodata = _process_unavailskill_episodes(expstats)
    eps_skill_notavail["details"] = skillunavailcombodata

    return eps_skill_details

def _process_overall(expstats):
    if expstats is None:
        return

    totaleps = expstats["num_episodes"]
    totalsuccess = len(expstats["successeps"])
    totalabort = len(expstats["aborteps"])
    totalfailure = len(expstats["faileps"])
    totalskills = expstats["num_skills"]

    overall_stats = {"num_episodes": totaleps, "success_eps": totalsuccess, "abort_eps": totalabort, "fail_eps": totalfailure,
                     "successrate": round((totalsuccess/totaleps),3), "abortrate": round((totalabort/totaleps),3),
                     "failurerate": round((totalfailure/totaleps),3), "num_skills": totalskills,
                     }

    print(f"Total Episodes: {totaleps}, Success: {totalsuccess}, Abort: {totalabort}, Failure: {totalfailure}")
    print(f'Success: {overall_stats["successrate"]}, Abort: {overall_stats["abortrate"]}, Failure: {overall_stats["failurerate"]}')
    
    return overall_stats


def checkdiffbwskillstest(episode_path, combonamedict):
    interavail = isinstanceinfoavailable(episode_path)
    if not interavail:
        return    

    interaction_data = {}
    interactions_path = os.path.join(episode_path, "interactions.json")
    if os.path.exists(interactions_path):
        interactions = read_json_file(interactions_path) or {}
        skillcode = interactions["Evaluation"]["skills_code"]
        comboname = interactions["Evaluation"]["boardinfo"]["simple_reuse"]["combo_name"]
        if comboname in skillcode:
            combonamedict["skillavail"].add(comboname)
        else:
            if comboname == "wbvbvns":
                print(episode_path)
                input()
            combonamedict["skillnotavail"].add(comboname)
    else:
        pass


def _process_combo_diff(expstats, combonamestats):
    if expstats is None or combonamestats is None:
        return

    skillfilenames = list(expstats["skillsfilename"].keys())
    if len(skillfilenames) > 1:
        print(f"More than one skillname file in the instances! {skillfilenames}")

    skilldata_base = read_json_file(f"resources/data/en/{skillfilenames[0]}")
    numskill_base = len(skilldata_base)
    print(f"Base skills available from the file: {skillfilenames[0]} -> {numskill_base}")
    numskill_instances = len(combonamestats["skillavail"])
    print(f"Skills available from the interactions file: -> {numskill_instances}")

    if set(expstats["skillavailcombos"]) != combonamestats["skillavail"]:
        print(f'Difference in notavailcombos: {set(expstas["skillavailcombos"]) - combonamestats["skillavail"]}')
    else:
        print("No difference in avail skills")

    if set(expstats["skillnotavailcombos"]) != combonamestats["skillnotavail"]:
        print(f'Difference in notavailcombos: {set(expstas["skillnotavailcombos"]) - combonamestats["skillnotavail"]}')        
    else:
        print("No difference in not avail skills")


    skill_inbase_notincode = []
    for skill in skilldata_base:
        #if skill not in combonamestats["skillavail"] and skill not in combonamestats["skillnotavail"]:
        if skill not in expstats["skillavailcombos"] and skill not in expstats["skillnotavailcombos"]:
            skill_inbase_notincode.append(skill)

    #The difference would be because for this particular comboname, all the episodes would be aborted and hence it is not in expstats dict
    print(skill_inbase_notincode)



def compute_scores(base_dir: str, verbose: bool = True, use_gpt_skills=False) -> Dict[str, Any]:

    with open("/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/skillreuse/resources/data/en/learnedskills_clp_lat.json", "r") as f:
        clp_skills_data = json.load(f)

    clp_skill_len = 0#len(clp_skills_data)

    with open("/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/skillreuse/resources/data/en/learnedskills_gpt_lat.json", "r") as f:
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
                cfq_episodes = {"present": [], "not_present": []}

                episodes = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                num_episodes = len(episodes)
                expstats = {"successeps": [], "aborteps": [], "faileps": [], "clarification_eps": [],
                            "correction_eps": [], "undo_eps":[], "num_turns": [], "num_skills": 0, "skill_unavail_eps": [],
                            "skill_oracle_eps": [], "successepsfromscratch": [], "abortepsfromscratch": [],
                            "failepsfromscratch": [], "successepsskill": [], "abortepsskill": [], "failepsskill": [],
                            "gencode": [], "geninstructions": [], "num_turns_code_gen": [], "num_episodes": num_episodes,
                            "skillavailepscnt": 0, "skillnotavailepscnt": 0, "skillsfilename": {}, "skillavailcombos": [],
                            "skillnotavailcombos": [],
                            "shape_stats":{2: {"successeps": 0, "aborteps": 0, "faileps": 0},
                                           3: {"successeps": 0, "aborteps": 0, "faileps": 0},
                                           4: {"successeps": 0, "aborteps": 0, "faileps": 0},
                                           5: {"successeps": 0, "aborteps": 0, "faileps": 0}},}
                combonamestats = {"skillavail": set(), "skillnotavail": set()}

                for episode in episodes:
                    episode_path = os.path.join(exp_path, episode)
                    process_episode(episode_path, expstats)
                    #checkdiffbwskillstest(episode_path, combonamestats)
                overall_stats = _process_overall(expstats)
                eps_skill_details = _process_skillusage(expstats)
                cfq_details = _process_cfq_data(expstats)
                corr_details = _process_correction_data(expstats)
                #_process_combo_diff(expstats, combonamestats)
                results[game][model][exp] = {"overall_stats": overall_stats, "skill_data": eps_skill_details,
                                             "clarifications": cfq_details, "corrections": corr_details,}

    with open(f"{base_dir}/overallstats.json", 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Compute overall scores from experiment directories")
    parser.add_argument("base_dir", nargs="?", default="/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/skillreuse/rp1_gptskills_clp", help="Base directory containing model results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()

    compute_scores(args.base_dir, verbose=not args.quiet, use_gpt_skills=True)


if __name__ == "__main__":
    main()                    