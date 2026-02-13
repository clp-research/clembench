import os
import numpy as np
import timeit
from typing import Dict, Any, List, Optional, Tuple
import json
import ast
import dis
import argparse
from tqdm import tqdm
from utils.coco import init_board, put, clear


class PutCallCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0

    def visit_Call(self, node: ast.Call):
        # Matches `put(...)` (not obj.put or module.put)
        if isinstance(node.func, ast.Name) and node.func.id == "put":
            self.count += 1
        self.generic_visit(node)


class ASTComputation:
    def __init__(self):
        self.pcc = PutCallCounter()

    def get_top_level_function(self, tree: ast.AST):
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        return funcs[0] if funcs else None    
    
    def contains_function(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        return self.get_top_level_function(tree) is not None    
    

    def count_put_calls(self, tree: ast.AST) -> int:
        self.pcc.count = 0
        self.pcc.visit(tree)
        return self.pcc.count  

    def function_has_loop(self, func: ast.FunctionDef) -> bool:
        return any(isinstance(node, ast.For) for node in ast.walk(func))      


class ComputeOptimalness:

    def __init__(self):
        self.astcomp = ASTComputation()


    def read_json_file(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None        

    def is_optimized_function(self, input_code: str, generated_code: str) -> bool:

        try:
            input_tree = ast.parse(input_code)
            output_tree = ast.parse(generated_code)
        except SyntaxError:
            return False, None  # generated code isn't even valid Python

        input_puts = self.astcomp.count_put_calls(input_tree)

        details = {"function_name": None, "has_loop": None, "func_put_calls": None, "input_put_calls": input_puts}

        func = self.astcomp.get_top_level_function(output_tree)
        if func is None:
            details["function_name"] = None
        else:
            details["function_name"] = func.name


        # Check it has a loop
        if not self.astcomp.function_has_loop(func):
            #return False, details
            details["has_loop"] = False
        else:
            details["has_loop"] = True

        # Count put calls inside the function only
        func_puts = self.astcomp.count_put_calls(output_tree)
        details["func_put_calls"] = func_puts
        details["input_put_calls"] = input_puts

        # Heuristic: must have fewer put calls than input and at least 1
        if not (1 <= func_puts < input_puts):
            details["reduced_put_calls"] = False
        else:
            details["reduced_put_calls"] = True

        if details is None:
            print("Something is wrong in computiong optimization")
            input()

        if details["function_name"] is not None and details["has_loop"] and details["reduced_put_calls"]:
            return True, details
        else:
            return False, details




    def executecode(self, code, func_usage):
        board = init_board(8,8)
        code_exec = f"{code}\n{func_usage}"

        ns = {"board": board, "put": put, "clear": clear}

        try:
            exec(code_exec, ns)
            func_obj = next(v for v in ns.values() if callable(v))
        except Exception as e:
            #print(f"Error executing code: {e}\n{code_exec}")
            #input()
            return None, None

        return board, func_obj


    def time_snippet(self, snippet_runner, repeat=5, number=1000):
        times = timeit.repeat(snippet_runner, repeat=repeat, number=number)
        return min(times) / number  # best per-iteration time
    
    def computetime(self, baseline_src: str, candidate_src: str, func_usage: str, variant: str) -> float:
        # Execute once to obtain callable objects (if possible)
        try:
            #if variant == "single_turn_sc":
            #    _, base_func_obj = self.executecode(baseline_src, func_usage)
            #elif variant in ["multi_turn", "reconstruct-multi_turn"]:
            #    _, base_func_obj = self.executecode(baseline_src, "")
            _, base_func_obj = self.executecode(baseline_src, func_usage)
        except Exception:
            base_func_obj = None

        try:
            _, candidate_func_obj = self.executecode(candidate_src, func_usage)
        except Exception:
            candidate_func_obj = None

        func_obj = {"baseline": base_func_obj, "candidate": candidate_func_obj}

        # Measure timings by repeatedly running the executecode (we ignore return values)
        def run_baseline():
            try:
                self.executecode(baseline_src, "")
            except Exception:
                return None

        def run_candidate():
            try:
                self.executecode(candidate_src, func_usage)
            except Exception:
                return None

        baseline_time = self.time_snippet(run_baseline)
        candidate_time = self.time_snippet(run_candidate)

        time_obj = {"baseline": baseline_time, "candidate": candidate_time}

        if baseline_time is None or candidate_time is None or candidate_time == 0:
            speedup = float('nan')
        else:
            speedup = round(baseline_time / candidate_time, 3)

        return speedup, func_obj, time_obj
    
    def parse_scores(self, scores: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        es = scores.get("episode scores", {})
        main_score = es.get("Main Score")
        reconst = "NA"#es.get("reconstruction_status")
        return main_score, reconst

    def parse_interactions(self, interactions: Dict[str, Any]) -> Tuple[Dict[str, Any], int, Optional[int]]:
        ev = interactions.get("Evaluation", {})
        code_data = None
        num_turns = interactions.get("Complete turns", 0)
        boardinfo = ev.get("boardinfo", {})
        variant = boardinfo.get("variant") if isinstance(boardinfo, dict) else None
        total_shapes = boardinfo.get("total_shapes") if isinstance(boardinfo, dict) else None
        optimized_function = ev.get("optimized_function", None)
        optim_func_signature = ev.get("optimized_function_signature", None)
        func_usage = ev.get("func_usage", None)
        print("variant:", variant)
        if variant == "single_turn_sc":
            input_code_func = boardinfo["code"]["single_turn_sc"]["function"]
        elif variant == "multi_turn":
            input_code_func = "\n".join(boardinfo["code"]["multi_turn"]["output"])
        elif variant == "reconstruct-multi_turn":
            input_code_func = "\n".join(boardinfo["code"]["multi_turn"]["output"])            
        else:
            #print(f"Unknown variant in parse_interactions: {variant}")
            #input()
            inst_code_pairs = ev["optim_input_data"]["inst_code_pairs"]
            code_pairs = [pair["code_snippet"] for pair in inst_code_pairs if "code_snippet" in pair]
            code_pairs = "\n".join(code_pairs)
            print(code_pairs)
            input_code_func = code_pairs#boardinfo["simple"]["code"]["single_turn"]["function"]
        interaction_code = {"input_code_func": input_code_func, "generated_code_func": optimized_function,
                            "func_usage": func_usage,
                           "generated_code_signature": optim_func_signature}
        return code_data, num_turns, total_shapes, interaction_code, variant


    def bytecode_len(self, fn):
        return len(list(dis.Bytecode(fn)))


    def process_episode(self, episode_path: str) -> Dict[str, Any]:
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
            interactions = self.read_json_file(interactions_path) or {}
            code_data, num_turns, total_shapes, interaction_code, variant = self.parse_interactions(interactions)

            # basic interaction-derived stats
            stats["total_shapes"] = total_shapes
            stats["dialog_turns"] = num_turns

            # seed code / generated code info
            stats["code_stats"] = interaction_code or {}

            # analyze optimization properties
            input_code_func = interaction_code.get("input_code_func", "")
            generated_code_func = interaction_code.get("generated_code_func", "")

            try:
                is_optimized_func, details = self.is_optimized_function(input_code_func, generated_code_func)
            except Exception:
                is_optimized_func, details = False, {"has_loop": None, "func_put_calls": None, "input_put_calls": None, "function_name": None, "reduced_put_calls": None}

            #print(f"Episode: {episode_path}, is_optimized_func: {is_optimized_func}, details: {details}")
            #input()

            stats["code_stats"]["is_optimized_func"] = is_optimized_func
            stats["code_stats"]["func_details"] = {
                "has_loop": details.get("has_loop", None),
                "func_put_calls": details.get("func_put_calls", None),
                "input_put_calls": details.get("input_put_calls", None),
                "function_name": details.get("function_name", None),
                "reduced_put_calls": details.get("reduced_put_calls", None),
            }

            # include signature / usage metadata if available
            if interaction_code:
                stats["code_stats"]["generated_code_signature"] = interaction_code.get("generated_code_signature")
                stats["code_stats"]["func_usage"] = interaction_code.get("func_usage")

            # measure runtime speedup and bytecode lengths (guarded)
            time_taken, func_obj, time_obj = self.computetime(
                interaction_code.get("input_code_func", ""),
                interaction_code.get("generated_code_func", ""),
                interaction_code.get("func_usage", ""),
                variant
            )
            stats["code_stats"]["time_speedup"] = time_taken
            stats["code_stats"]["is_speed_optimized"] = time_taken > 1.0 if isinstance(time_taken, float) and not np.isnan(time_taken) else None
            stats["code_stats"]["time_obj"] = time_obj
            # bytecode lengths may not be available if execution failed
            input_code_len = None
            generated_code_len = None
            try:
                if func_obj and func_obj.get("baseline"):
                    input_code_len = self.bytecode_len(func_obj["baseline"])
            except Exception:
                input_code_len = None
            try:
                if func_obj and func_obj.get("candidate"):
                    generated_code_len = self.bytecode_len(func_obj["candidate"])
            except Exception:
                generated_code_len = None

            stats["code_stats"]["code_lenth"] = {}
            stats["code_stats"]["code_lenth"]["input_code_bytecode_len"] = input_code_len
            stats["code_stats"]["code_lenth"]["generated_code_bytecode_len"] = generated_code_len           
            if input_code_len is not None and generated_code_len is not None:
                stats["code_stats"]["code_lenth"]["bytecode_len_reduction"] = (input_code_len - generated_code_len)                 
                if generated_code_len < input_code_len:
                    stats["code_stats"]["is_optimized_len"] = "reduced"
                else:
                    stats["code_stats"]["is_optimized_len"] = "not_reduced"
            else:
                stats["code_stats"]["code_lenth"]["bytecode_len_reduction"] = None
                stats["code_stats"]["is_optimized_len"] = None

        if os.path.exists(scores_path):
            scores = self.read_json_file(scores_path) or {}
            main_score, reconst_score = self.parse_scores(scores)
        else:
            main_score, reconst_score = np.nan, None

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




    def run(self, base_dir):
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

                    episodes = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
                    num_episodes = len(episodes)

                    # Aggregation containers per experiment
                    accuracy_data: List[float] = []
                    failed_episodes: List[str] = []
                    aborted_episodes: List[str] = []
                    dialog_turns_list: List[int] = []
                    optimized_flags: List[bool] = []
                    time_speedups: List[float] = []
                    bytecode_reductions: List[int] = []
                    optimized_len_flags: List[bool] = []
                    # per-episode storage
                    episodes_data: Dict[str, Any] = {}
                    successful_episodes = 0
                    
                    # Optimization categorization
                    fully_optimized_episodes: List[str] = []
                    partially_optimized_speed: List[str] = []
                    partially_optimized_length: List[str] = []
                    partially_optimized_func: List[str] = []
                    partially_optimized_func_speed: List[str] = []
                    partially_optimized_func_length: List[str] = []
                    partially_optimized_speed_length: List[str] = []
                    not_optimized_episodes: List[str] = []
                    
                    # Categorization by number of shapes
                    shapes_categorization: Dict[int, Dict[str, Any]] = {}

                    for episode in tqdm(episodes, desc=f"{game}/{model}/{exp}"):
                        episode_path = os.path.join(exp_path, episode)
                        ep_stats = self.process_episode(episode_path)

                        # accuracy / outcome
                        accuracy_data.extend(ep_stats.get("accuracy_vals", []))
                        failed_episodes.extend(ep_stats.get("failed_episodes", []))
                        aborted_episodes.extend(ep_stats.get("aborted_episodes", []))

                        # dialog turns
                        dt = ep_stats.get("dialog_turns")
                        if dt is not None:
                            try:
                                dialog_turns_list.append(int(dt))
                            except Exception:
                                pass

                        # code/optimization stats
                        cs = ep_stats.get("code_stats") or {}
                        cs["outcome"] = ep_stats.get("outcome")
                        cs["episode"] = episode

                        # store per-episode code_stats for later (all episodes)
                        episodes_data[episode] = cs

                        success = ep_stats.get("outcome") == "success"

                        # Categorize by number of shapes (track success and failure)
                        total_shapes_val = ep_stats.get("total_shapes")
                        if total_shapes_val is not None:
                            if total_shapes_val not in shapes_categorization:
                                shapes_categorization[total_shapes_val] = {
                                    "fully_optimized": [],
                                    "speed_only": [],
                                    "length_only": [],
                                    "func_only": [],
                                    "func_and_speed": [],
                                    "func_and_length": [],
                                    "speed_and_length": [],
                                    "not_optimized": [],
                                    "failed": [],
                                    "aborted": [],
                                }
                            if not success:
                                if ep_stats.get("outcome") == "failed":
                                    shapes_categorization[total_shapes_val]["failed"].append(episode)
                                elif ep_stats.get("outcome") == "aborted":
                                    shapes_categorization[total_shapes_val]["aborted"].append(episode)

                        # Skip optimization tallies for failed/aborted episodes
                        if not success:
                            continue

                        successful_episodes += 1

                        is_opt = cs.get("is_optimized_func")
                        if isinstance(is_opt, bool):
                            optimized_flags.append(is_opt)

                        ts = cs.get("time_speedup")
                        try:
                            if ts is not None and not (isinstance(ts, float) and np.isnan(ts)):
                                time_speedups.append(float(ts))
                        except Exception:
                            pass

                        cl = cs.get("code_lenth", {}) or {}
                        red = cl.get("bytecode_len_reduction")
                        if isinstance(red, (int, float)):
                            try:
                                bytecode_reductions.append(int(red))
                            except Exception:
                                pass

                        is_opt_len = cs.get("is_optimized_len")
                        if isinstance(is_opt_len, bool):
                            optimized_len_flags.append(is_opt_len)
                        
                        # Categorize episode optimization
                        is_func_opt = cs.get("is_optimized_func", False) is True
                        is_speed_opt = cs.get("is_speed_optimized", False) is True
                        is_len_opt = cs.get("is_optimized_len") == "reduced"
                        
                        # Count how many criteria are met
                        criteria_met = sum([is_func_opt, is_speed_opt, is_len_opt])
                        
                        if criteria_met == 3:
                            # Fully optimized: all three criteria met
                            fully_optimized_episodes.append(episode)
                        elif criteria_met == 2:
                            # Partially optimized: two criteria met
                            if is_func_opt and is_speed_opt:
                                partially_optimized_func_speed.append(episode)
                            elif is_func_opt and is_len_opt:
                                partially_optimized_func_length.append(episode)
                            elif is_speed_opt and is_len_opt:
                                partially_optimized_speed_length.append(episode)
                        elif criteria_met == 1:
                            # Partially optimized: only one criterion met
                            if is_func_opt:
                                partially_optimized_func.append(episode)
                            elif is_speed_opt:
                                partially_optimized_speed.append(episode)
                            elif is_len_opt:
                                partially_optimized_length.append(episode)
                        else:
                            # Not optimized: no criteria met
                            not_optimized_episodes.append(episode)
                        
                        # Categorize by number of shapes (successful only)
                        total_shapes_val = ep_stats.get("total_shapes")
                        if total_shapes_val is not None:
                            if total_shapes_val not in shapes_categorization:
                                shapes_categorization[total_shapes_val] = {
                                    "fully_optimized": [],
                                    "speed_only": [],
                                    "length_only": [],
                                    "func_only": [],
                                    "func_and_speed": [],
                                    "func_and_length": [],
                                    "speed_and_length": [],
                                    "not_optimized": [],
                                    "failed": [],
                                    "aborted": [],
                                }
                            
                            if criteria_met == 3:
                                shapes_categorization[total_shapes_val]["fully_optimized"].append(episode)
                            elif criteria_met == 2:
                                if is_func_opt and is_speed_opt:
                                    shapes_categorization[total_shapes_val]["func_and_speed"].append(episode)
                                elif is_func_opt and is_len_opt:
                                    shapes_categorization[total_shapes_val]["func_and_length"].append(episode)
                                elif is_speed_opt and is_len_opt:
                                    shapes_categorization[total_shapes_val]["speed_and_length"].append(episode)
                            elif criteria_met == 1:
                                if is_func_opt:
                                    shapes_categorization[total_shapes_val]["func_only"].append(episode)
                                elif is_speed_opt:
                                    shapes_categorization[total_shapes_val]["speed_only"].append(episode)
                                elif is_len_opt:
                                    shapes_categorization[total_shapes_val]["length_only"].append(episode)
                            else:
                                shapes_categorization[total_shapes_val]["not_optimized"].append(episode)
                    # finalize aggregations for this experiment (single pass values)
                    total_episodes = num_episodes
                    num_failed = len(failed_episodes)
                    num_aborted = len(aborted_episodes)

                    num_optimized = sum(1 for v in optimized_flags if v)
                    pct_optimized = round((num_optimized / successful_episodes), 2) if successful_episodes > 0 else 0.0

                    num_reduced_len = sum(1 for v in bytecode_reductions if v and int(v) > 0)
                    pct_reduced_len = round((num_reduced_len / successful_episodes), 2) if successful_episodes > 0 else 0.0

                    num_faster = sum(1 for v in time_speedups if v is not None and not (isinstance(v, float) and np.isnan(v)) and float(v) > 1.0)
                    pct_faster = round((num_faster / successful_episodes), 2) if successful_episodes > 0 else 0.0

                    avg_dialog_turns = round(float(np.mean(dialog_turns_list)), 2) if dialog_turns_list else None

                    # Calculate optimization category counts
                    num_fully_optimized = len(fully_optimized_episodes)
                    num_partially_optimized = (len(partially_optimized_speed) + len(partially_optimized_length) + 
                                               len(partially_optimized_func) + len(partially_optimized_func_speed) + 
                                               len(partially_optimized_func_length) + len(partially_optimized_speed_length))
                    num_not_optimized = len(not_optimized_episodes)
                    
                    pct_fully_optimized = round((num_fully_optimized / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    pct_partially_optimized = round((num_partially_optimized / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    pct_not_optimized = round((num_not_optimized / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    
                    score_fully_optimized = round((num_fully_optimized / successful_episodes) * 100, 2) if successful_episodes > 0 else 0.0
                    score_partially_optimized = round((num_partially_optimized / successful_episodes) * 100, 2) if successful_episodes > 0 else 0.0
                    score_not_optimized = round((num_not_optimized / successful_episodes) * 100, 2) if successful_episodes > 0 else 0.0

                    # Per-category percentages (overall, successful episodes only)
                    cat_speed_only_pct = round((len(partially_optimized_speed) / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    cat_length_only_pct = round((len(partially_optimized_length) / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    cat_func_only_pct = round((len(partially_optimized_func) / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    cat_func_speed_pct = round((len(partially_optimized_func_speed) / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    cat_func_length_pct = round((len(partially_optimized_func_length) / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    cat_speed_length_pct = round((len(partially_optimized_speed_length) / successful_episodes), 2) if successful_episodes > 0 else 0.0
                    
                    # Process shapes categorization
                    by_shapes: Dict[str, Any] = {}
                    for num_shapes in sorted(shapes_categorization.keys()):
                        shape_data = shapes_categorization[num_shapes]
                        num_fully = len(shape_data["fully_optimized"])
                        num_speed_only = len(shape_data["speed_only"])
                        num_length_only = len(shape_data["length_only"])
                        num_func_only = len(shape_data["func_only"])
                        num_func_speed = len(shape_data["func_and_speed"])
                        num_func_length = len(shape_data["func_and_length"])
                        num_speed_length = len(shape_data["speed_and_length"])
                        num_not = len(shape_data["not_optimized"])
                        num_failed_shape = len(shape_data["failed"])
                        num_aborted_shape = len(shape_data["aborted"])
                        num_partially = num_speed_only + num_length_only + num_func_only + num_func_speed + num_func_length + num_speed_length
                        total_success_shape = num_fully + num_partially + num_not
                        total_shape_all = total_success_shape + num_failed_shape + num_aborted_shape

                        # Per-category percentages within this shape bucket (successful only)
                        shape_cat_speed_only_pct = round((num_speed_only / total_success_shape), 2) if total_success_shape > 0 else 0.0
                        shape_cat_length_only_pct = round((num_length_only / total_success_shape), 2) if total_success_shape > 0 else 0.0
                        shape_cat_func_only_pct = round((num_func_only / total_success_shape), 2) if total_success_shape > 0 else 0.0
                        shape_cat_func_speed_pct = round((num_func_speed / total_success_shape), 2) if total_success_shape > 0 else 0.0
                        shape_cat_func_length_pct = round((num_func_length / total_success_shape), 2) if total_success_shape > 0 else 0.0
                        shape_cat_speed_length_pct = round((num_speed_length / total_success_shape), 2) if total_success_shape > 0 else 0.0
                        
                        by_shapes[str(num_shapes)] = {
                            "total_episodes": total_shape_all,
                            "total_successful_episodes": total_success_shape,
                            "fully_optimized": {
                                "count": num_fully,
                                "percentage": round((num_fully / total_success_shape), 2) if total_success_shape > 0 else 0.0,
                                "score": round((num_fully / total_success_shape) * 100, 2) if total_success_shape > 0 else 0.0,
                                "episodes": shape_data["fully_optimized"]
                            },
                            "partially_optimized": {
                                "count": num_partially,
                                "percentage": round((num_partially / total_success_shape), 2) if total_success_shape > 0 else 0.0,
                                "score": round((num_partially / total_success_shape) * 100, 2) if total_success_shape > 0 else 0.0,
                                "by_category": {
                                    "speed_only": {
                                        "count": num_speed_only,
                                        "percentage": shape_cat_speed_only_pct,
                                        "episodes": shape_data["speed_only"]
                                    },
                                    "length_only": {
                                        "count": num_length_only,
                                        "percentage": shape_cat_length_only_pct,
                                        "episodes": shape_data["length_only"]
                                    },
                                    "func_only": {
                                        "count": num_func_only,
                                        "percentage": shape_cat_func_only_pct,
                                        "episodes": shape_data["func_only"]
                                    },
                                    "func_and_speed": {
                                        "count": num_func_speed,
                                        "percentage": shape_cat_func_speed_pct,
                                        "episodes": shape_data["func_and_speed"]
                                    },
                                    "func_and_length": {
                                        "count": num_func_length,
                                        "percentage": shape_cat_func_length_pct,
                                        "episodes": shape_data["func_and_length"]
                                    },
                                    "speed_and_length": {
                                        "count": num_speed_length,
                                        "percentage": shape_cat_speed_length_pct,
                                        "episodes": shape_data["speed_and_length"]
                                    }
                                }
                            },
                            "not_optimized": {
                                "count": num_not,
                                "percentage": round((num_not / total_success_shape), 2) if total_success_shape > 0 else 0.0,
                                "score": round((num_not / total_success_shape) * 100, 2) if total_success_shape > 0 else 0.0,
                                "episodes": shape_data["not_optimized"]
                            },
                            "failed": {
                                "count": num_failed_shape,
                                "percentage": round((num_failed_shape / total_shape_all), 2) if total_shape_all > 0 else 0.0,
                                "score": round((num_failed_shape / total_shape_all) * 100, 2) if total_shape_all > 0 else 0.0,
                                "episodes": shape_data["failed"]
                            },
                            "aborted": {
                                "count": num_aborted_shape,
                                "percentage": round((num_aborted_shape / total_shape_all), 2) if total_shape_all > 0 else 0.0,
                                "score": round((num_aborted_shape / total_shape_all) * 100, 2) if total_shape_all > 0 else 0.0,
                                "episodes": shape_data["aborted"]
                            }
                        }

                    save_data = {
                        "num_episodes": total_episodes,
                        "num_successful_episodes": successful_episodes,
                        "num_optimized_functions": num_optimized,
                        "pct_optimized_functions": pct_optimized,
                        "num_reduced_length": num_reduced_len,
                        "pct_reduced_length": pct_reduced_len,
                        "num_faster_functions": num_faster,
                        "pct_faster_functions": pct_faster,
                        "accuracy": round((float(np.sum(accuracy_data)) / total_episodes), 2) if total_episodes > 0 else 0,
                        "num_failed_episodes": num_failed,
                        "num_aborted_episodes": num_aborted,
                        "failed_episodes": failed_episodes,
                        "aborted_episodes": aborted_episodes,
                        "avg_dialog_turns": avg_dialog_turns,
                        # Optimization categorization
                        "fully_optimized": {
                            "count": num_fully_optimized,
                            "percentage": pct_fully_optimized,
                            "score": score_fully_optimized,
                            "episodes": fully_optimized_episodes
                        },
                        "partially_optimized": {
                            "count": num_partially_optimized,
                            "percentage": pct_partially_optimized,
                            "score": score_partially_optimized,
                            "by_category": {
                                "speed_only": {
                                    "count": len(partially_optimized_speed),
                                    "percentage": cat_speed_only_pct,
                                    "episodes": partially_optimized_speed
                                },
                                "length_only": {
                                    "count": len(partially_optimized_length),
                                    "percentage": cat_length_only_pct,
                                    "episodes": partially_optimized_length
                                },
                                "func_only": {
                                    "count": len(partially_optimized_func),
                                    "percentage": cat_func_only_pct,
                                    "episodes": partially_optimized_func
                                },
                                "func_and_speed": {
                                    "count": len(partially_optimized_func_speed),
                                    "percentage": cat_func_speed_pct,
                                    "episodes": partially_optimized_func_speed
                                },
                                "func_and_length": {
                                    "count": len(partially_optimized_func_length),
                                    "percentage": cat_func_length_pct,
                                    "episodes": partially_optimized_func_length
                                },
                                "speed_and_length": {
                                    "count": len(partially_optimized_speed_length),
                                    "percentage": cat_speed_length_pct,
                                    "episodes": partially_optimized_speed_length
                                }
                            }
                        },
                        "not_optimized": {
                            "count": num_not_optimized,
                            "percentage": pct_not_optimized,
                            "score": score_not_optimized,
                            "episodes": not_optimized_episodes
                        },
                        # Categorization by number of shapes
                        "by_num_shapes": by_shapes,
                        # Per-episode code_stats dictionary
                        "episodes": episodes_data,
                    }

                    results[game][model][exp] = save_data

        outpath = os.path.join(base_dir, "overall_results_optimized.json")
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)

        return results                    
    
def main():
    parser = argparse.ArgumentParser(description="Compute overall scores from experiment directories")
    parser.add_argument("base_dir", nargs="?", default="/home/admin/Desktop/codebase/cocobots/testimageccbts_local/clemnew/clembench/cocottr_optimizer/rp2", help="Base directory containing model results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose printing")
    args = parser.parse_args()

    cscores = ComputeOptimalness()
    cscores.run(args.base_dir)


if __name__ == "__main__":
    main()    