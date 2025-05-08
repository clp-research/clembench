## Collections of games to be run with the [clemcore framework](https://github.com/clp-research/clemcore)

## Installation (Python 3.10+)

Install the required dependencies to run all games: 

`pip install -r clembench/requirements.txt`

This will also install the `clem` CLI tool.

### Versioning

The clembench release versions tags are structured like `major.minor.patch` where

- `major` indicates the compatibility with a major clemcore version e.g. `2.x.x` is only compatible with clemcore versions `2.x.x`
- `minor` indicates changes to the games in the benchmark that don't affect compatibility with clemcore e.g. refactorings, additions or removals of games
- `patch` indicates smaller adjustments or hotfixes to the games that don't affect compatibility with clemcore 

The following image visualizes the established dependencies and version history:

<p style="text-align: center">
  <a href="versions.png">
    <img src="versions.png" alt="Thumbnail" width="500"/>
  </a>
</p>


### API Keys

To use APIs (OpenAI, Anthropic, Google, Mistral, etc.), create a `key.json` file that includes the required fields for each backend. The template file (key.json.template) is provided.

Copy the file into `/Users/<YOUR_USER_NAME>/.clemcore/` for MacOS or `` for Linux.

### Models, Backends, Games

After the installation you will have access to the `clem` CLI tool. The main functions are:

```
(myclem) clem list games               # list the games available for a run
(myclem) clem list backends            # list the backends available for a run
(myclem) clem list models              # list the models available for a run
(myclem) clem run -g <game> -m <model> # runs the game benchmark; also scores
(myclem) clem transcribe               # translates interactions into html files
(myclem) clem score                    # computes individual performance measures
(myclem) clem eval                     # computes overall performances measures; requires scores
```

To add new custom models, populate the `model_registry.json` file with the required fields  (template is provided as *model_registry.json.template*).

To run your custom game, populate the `game_registry.json` file with the required fields and directory path (template is provided as *game_registry.json.template*).


We welcome you to contribute to or extend the benchmark with your own games and models. 
Please open a pull request in the respective repository. 
You can find more information on how to use the benchmark in the links below.

However, the following documentation needs still to be checked for up-to-dateness.

- [How to run the benchmark and evaluation locally](docs/howto_run_benchmark.md)
- [How to run the benchmark, update leaderboard workflow](docs/howto_benchmark_workflow.md)
- [How to add a new model](docs/howto_add_models.md)
- [How to add and run your own game](docs/howto_add_games.md)
- [How to integrate with Slurk](docs/howto_slurk.md)
