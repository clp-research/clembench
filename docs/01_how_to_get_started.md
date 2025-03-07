## Instructions on how to run existing games:

* clone [clembench repo](https://github.com/AnneBeyer/clembench.git) and checkout branch `game_registry` [TODO: merge into clp-research/clembench `main` once everything is tested]

* clone [clemgames repo](https://github.com/clp-research/clemgames.git) into sibling directory to clembench

* change into the clembench directory

	* run prepare_path.sh to add clembench to your python path (NOTE: If you are using a different shell than bash, or if for any other reason `echo $PYTHONPATH` still does not contain the current path, call `export PYTHONPATH=.:$PYTHONPATH` directly in your shell instead)
		
	* create key.json either by adapting key.json.template or running setup.sh (also see setup.sh to create a python environment as necessary) [TODO: this is redundant, which version do we want to keep? setup.sh potentially overwrites existing key.json files...]

	* to evaluate a model, add the required keys to key.json for either connecting to the model served by our group (see pinned post in mattermost channel CLEM-Club), or using any model api by creating an account (e.g., groq, openAI, ...), or downloading and running any (supported) hf model locally (this can be done by running the code on our server (requires A100 access ([TODO] link to the other internal doc) and running `export HUGGINGFACE_HUB_CACHE=/data/>USERNAME>/huggingface_cache`). See `clembench/clemcore/backends/model_registry.json` for all supported models (and instruction in docs on how to add new ones)

	* install requirements (if not already done by setup.sh)(`.txt` for general use, `_hf.txt` for using huggingface models locally, `_llamacpp....txt` to run models via llamacpp?) [TODO]

	* if necessary, install game specific requirements (see respective game folders in clemgames [TODO: currently still part of general requirements]) [TODO: do we want something like a setup.sh for installing several/all game requirements as well?]	

	* to run(/score/transcribe) a game, call `clemcore/cli.py run` (e.g., `clemcore/cli.py run -g taboo -m mock`, see file (or -h) for more example calls)
		* see `clemcore/clemgames/game_registry.json` for all supported games
		* see `clemcore/backends/model_registry.json` for all supported models
		* instances are expected to live in game directory under `in/instances.json` if not specified otherwise
		* results will be stored in `./results` if not specified otherwise
		* detailed logs will be written to `./clembench.log`

	* after running `clemcore/cli.py score -g GAME` use `evaluation/bencheval.py` for creating evaluation table (collects results from `./results` by default, but can be overwritten to any location), which calculates `clemscore, %Played and Success` for all games and models found in the results directory and creates an overview in `./results/results.[csv|html]`
	* see the ´clembench/scripts´ folder for example scripts on how to run the whole pipeline


## Instructions on how to add new games:

* follow the steps above, but instead of cloning the clemgames repository, create your own game repository (for compatibility, also create it in a sibling directory to clembench)

* open clembench in your IDE

* open (attach) your game repository in your IDE

* make game repo know about clembench and vice versa (for now required to facilitate development in IDE; clemcore might become pip-installable at some point) (in PyCharm: tick boxes in Settings-Project-Project Dependencies, in VSCode ?[TODO])

* if developing a new game or running one from a different location, add an entry for the game in clembench/clemgames/game_registry_custom.json (give the path either relative to the clembench directory, e.g., `../clemgames/taboo/` or as an absolute path)

* create instances (see existing instancegenerator.py files in the game directories for examples) in YOURGAME/in, storing required resources in YOURGAME/resources and create the game master in master.py in YOURGAME. (TODO: update and link to how_to_develop_games, add script to create (and describe) game structure (provided by YP))

* to develop the game structure, it can be helpful to first define custom responses in players that always answer according to the formal rules and can be run using `-m mock` [TODO: add details here?]

* for evaluation, use `evaluation/bencheval.py` as a starting point and potentially extend it for game specific evaluation (other examples to be added to evaluation/)

* to add your final game to the official collection, create a PR in the clemgames repo

* for any required changes regarding the clemcore framework, open an issue/PR in the clembench repo

## Instructions on how to update games to the new framework version:

+ if you started developing your game before December 2024, and if it was not added to the official games yet, you need to update your game to the new framework version as described [here](howto_update_to_v2.md)
* some games in `clemgames` still need to be updated (this is work in progress)
