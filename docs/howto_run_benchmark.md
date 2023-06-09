# Running the benchmark

## Setting up

### Install dependencies

```
pip install -r requirements.txt
```

### API Key

Create a file `key.json` in the project root and past in your api key (and organisation optionally).

```
{
  "openai": {
            "organisation": "<value>", 
            "api_key": "<value>"
            },
  "anthropic": {
            "api_key": "<value>"
            },
  "alephalpha": {
            "api_key": "<value>"
            }
}
```

Note: You can look up your api key for OpenAI at https://platform.openai.com/account/api-keys and for Anthoropic
at https://console.anthropic.com/account/keys, AlephAlpha can be found
here: https://docs.aleph-alpha.com/docs/introduction/luminous/

### Available models

Currently available values are:

- `"gpt-4"`
- `"gpt-3.5-turbo"`
- `"text-davinci-003"`
- `"claude-v1.3"`
- `"claude-v1.3-100k"`
- `"luminous-supreme-control"`
- `"luminous-supreme"`
- `"luminous-extended"`
- `"luminous-base"`
- `"google/flan-t5-xxl"`

Models can be added in `clemgame/api.py`.

## Running the benchmark

Go into the project root and prepare path to run from cmdline

```
source prepare_path.sh
```

Then run the cli script

```
python3 scripts/cli.py --help
```

or run game masters individually

```
python3 clembench/games/privateshared/master.py
```

## Running the evaluation

All details from running the benchmarked are logged in the respective game directories,
with the format described in ```logdoc.md```.

We provide an evaluation script at `evaluation/basiceval.py` that produces a number of tables and visualizations for the benchmark. New models (their name abbreviation), metrics (their range) and game/model (their order) must be added manually to the constants in ```evaluation/evalutils.py```.
