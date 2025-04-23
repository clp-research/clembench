#!/bin/bash
if [ $# -ne 1 ]; then
    echo "Usage: $0 <basic|hf|vllm>"
    exit 2
fi

MODE=$1
if [[ "$MODE" != "basic" && "$MODE" != "hf" && "$MODE" != "vllm" ]]; then
    echo "Error: Invalid argument. Please use one of: basic, hf, vllm"
    exit 2
fi

if [[ "$(basename "$PWD")" == "clembench" ]]; then
  echo "Error: Script should be called from the repositories parent directory, but is called from inside 'clembench'."
  exit 2
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
VERSION_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
VERSION_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
if [[ $VERSION_MAJOR -lt 3 || ($VERSION_MAJOR -eq 3 && $VERSION_MINOR -lt 10) ]]; then
    echo "Error: Python version must be at least 3.10 but is $PYTHON_VERSION"
    exit 1
fi

VENV_NAME="venv"
if [[ "$MODE" == "hf" ]]; then
    VENV_NAME="venv_hf"
fi;
if [[ "$MODE" == "vllm" ]]; then
    VENV_NAME="venv_vllm"
fi;

echo "Mode selected: $MODE"
echo "Virtual environment name: $VENV_NAME"

python3 -m venv "$VENV_NAME"
source "$VENV_NAME"/bin/activate

pip3 install -r clembench/requirements.txt

if [[ "$MODE" == "hf" ]]; then
    pip3 install "clemcore[huggingface]"
fi
if [[ "$MODE" == "vllm" ]]; then
    pip3 install "clemcore[vllm]"
fi

mv clembench/key.json.template key.json
echo "Please add the keys to the key.json manually."
