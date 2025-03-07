#!/bin/bash
# Usage: scripts/run_multiling.sh

source prepare_path.sh
mkdir -p logs

version="v1.6"
games=(
"referencegame"
#"taboo"
#"imagegame"
#"wordle"
#"wordle_withclue"
#"wordle_withcritic"
#"privateshared"
#"codenames"
#"matchit_ascii"
#"guess_what"
#"textmapworld"
)

languages=("ar" "de" "en" "es" "it" "ja" "pt" "ru" "te" "tk" "tr" "zh")

models=(
"llama-70b" # = llama-3-70b-instruct
#"mock"
#"claude-3-opus-20240229"
#"gpt-4-turbo-2024-04-09"
#"aya-23-35B"
#"Llama-3-SauerkrautLM-70b-Instruct"
#"Meta-Llama-3.1-70B-Instruct"
#"Mixtral-8x22B-Instruct-v0.1"
#"Qwen1.5-72B-Chat"
)

echo
echo "==================================================="
echo "RUNNING: Benchmark Run Version ${version}"
echo "==================================================="
echo

for lang in "${languages[@]}"; do
  for game in "${games[@]}"; do
    for model in "${models[@]}"; do
      echo "Running ${model} on ${game}_${lang}"
      { time python3 clemcore/cli.py run -g "${game}" -m "${model}" -i "instances_${version}_${lang}.json" -r "results/${version}/${lang}"; } 2>&1 | tee "logs/run.${game}.${lang}.${model}.log"
    done
    echo "Transcribing ${game} in ${lang}"
    { time python3 clemcore/cli.py transcribe -g "${game}" -r "results/${version}/${lang}"; } 2>&1 | tee "logs/transcribe.${game}.${lang}.log"
    echo "Scoring ${game} in ${lang}"
    { time python3 clemcore/cli.py score -g "${game}" -r "results/${version}/${lang}"; } 2>&1 | tee "logs/score.${game}.${lang}.log"
  done
  echo "Evaluating all models across all games in ${lang}"
{ time python3 evaluation/bencheval.py -p "results/${version}/${lang}"; }
done

echo "==================================================="
echo "FINISHED: Benchmark Run Version ${version}"
echo "==================================================="
