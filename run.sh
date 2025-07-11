#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <game_name> [model:mock]"
    exit 1
fi

game_name="$1"
model_name="${2:-mock}"

clem run -g "${game_name}" -m "${model_name}"
clem score -g "${game_name}"
clem transcribe -g "${game_name}"
