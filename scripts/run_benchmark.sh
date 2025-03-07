#!/bin/bash
# Usage: scripts/run_benchmark.sh

source prepare_path.sh
mkdir -p logs

version="v1.6_control"
games=(
"taboo"
"referencegame"
"imagegame"
#"wordle"
#"wordle_withclue"
#"wordle_withcritic"
"privateshared"
"codenames"
"matchit_ascii"
#"guess_what"
#"textmapworld"
)
models=(
"llama-70b"
"mock"
#"claude-3-opus-20240229"
#"gpt-4-turbo-2024-04-09"
#"command-r-plus"
#"Llama-3-8b-chat-hf"
#"Llama-3-70b-chat-hf"
)

echo
echo "==================================================="
echo "RUNNING: Benchmark Run Version ${version}"
echo "==================================================="
echo


for game in "${games[@]}"; do
  for model in "${models[@]}"; do
    echo "Testing ${model} on ${game}"
    # currently, instances.json is the default file for the current benchmark version
    # older files are being renamed retrospectively, but the framework also allows
    # to input a specific instances file, so we could also use the version number here like this: -i instances_"${version}".json
    { time python3 clemcore/cli.py run -g "${game}" -m "${model}" -r "results/${version}"; } 2>&1 | tee logs/runtime."${game}"."${model}".log
    { time python3 clemcore/cli.py transcribe -g "${game}" -r "results/${version}"; } 2>&1 | tee logs/runtime.transcribe."${game}".log
    { time python3 clemcore/cli.py score -g "${game}" -r "results/${version}"; } 2>&1 | tee logs/runtime.score."${game}".log
  done
done
echo "Evaluating results"
{ time python3 evaluation/bencheval.py -p "results/${version}"; }

echo "==================================================="
echo "FINISHED: Benchmark Run Version ${version}"
echo "==================================================="