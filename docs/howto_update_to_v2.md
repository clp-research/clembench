# How to update your Game for v2.0

The current update contains a refactoring of the clembench core code in order to separate the games from the framework. 

Why was this neccesary? 
- As more and more games are being developed, they blow up the repository and the requirements, motivating the move to separate them from the core framework.
- Now we do not need to load all games anymore at the start. A `game_registry` now contains the `game_path` and the `game_name` of all possible games, but only the game passed to `cli.py` is actually loaded.

IMPORTANT: In order to avoid loosing any game code, please copy (or move) your game code from `clembench/games/YOUR_GAME` to a separate location (ideally create a new github repository) before updating your clembench fork. If you want to keep your commit history, [here](https://docs.github.com/en/get-started/using-git/splitting-a-subfolder-out-into-a-new-repository) is a guide on how to extract a subfolder into a new repository. 

## Separation of games and clemcore
Games can (and should) now live outside the clembench repository. The game path (and further information on a specific game version) are now stored in a `game_registry.json` (similar to the model_registry introduced in the last update). It (currently) lives under `clembench/clemcore/clemgame/game_registry.json`. 
To add games that are not (yet) addd to the official collection, create an entry of the same form in `clembench/clemcore/clemgame/game_registry_custom.json`
(similar to adding custom model versions in the model_registry_custom.json)
```json
[
  {
    "game_name": "taboo",
    "game_path": "../clemgames/taboo", # relative to clembench or absolute 
    "description": "Taboo game between two agents where one has to describe a word for the other to guess.", # copied from GameBenchmark get_description() in master.py
    "main_game": "taboo", # relevant for games with different versions, otherwise same as game_name,
    "instances": "instances", # OPTIONAL; if this key does not exist, instances.json will be used, if it exists, the instances file with the name given here will be used 
    "players": "two", # [one|multi]
    "image": "none", # [one|multi]
    "benchmark": ["0.9", "1.0", "1.5", "2.0"], # list of benchmark versions this game is part of; can be empty
    "languages": ["en"], # use ISO- codes for available languages
  }
]
```
This has implications on the `GameResourceLocator`, which now distinguishes two locations: the `game_path` for game specific resources, needed for creating and loading the instances, and the results path, which is still created the usual way in the clembench directory (by default), namely by creating the sub-directories `results/<dialogue_partners>/<game_name>`

The InstanceGenerator only needs access to the game resources, which is now defined by the location of `instancegenerator.py` itself
```python
# instancegenerator.py
+ import os

- GAME_NAME = "taboo"

  class TabooGameInstanceGenerator(GameInstanceGenerator):

      def __init__(self):
-         super().__init__(GAME_NAME)
+         super().__init__(os.path.dirname(os.path.abspath(__file__)))

```
You also need to make sure that all paths to resource files (like prompt templates) 
are given relative to your game directory, so the GameResourceLocator 
(from which the InstanceGenerator inherits) will find them.

In master.py, `GAME_NAME` is also removed as a variable and `self.game_name` and `self.game_path` are instantiated from `clemcore/clemgame/game_registry.json` instead (allowing future adaptions like different versions of games pointing to the same game code but defining variations in terms of parameters). 
The method `applies_to(cls, game_name: str)` is no longer needed.


The GameScorer doesn't need access to the game resources but only requires `self.game_name` to configure the results path.

In the GameBenchmark class, the game description and the number of players are now also extracted from the GameSpec (via the `game_registry`).
```python
# master.py

+ import os # only required if main() is defined as below

- GAME_NAME = "taboo"

  class Taboo(DialogueGameMaster):
    
-     def __init__(self, experiment: Dict, player_models: List[Model]):
-         super().__init__(GAME_NAME, experiment, player_models)
+     def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
+         super().__init__(game_name, game_path, experiment, player_models)

        ...

-     @classmethod
-     def applies_to(cls, game_name: str) -> bool:
-         return game_name == GAME_NAME
  
        ...
        
  class TabooScorer(GameScorer):
    
-     def __init__(self, experiment: Dict, game_instance: Dict):
-         super().__init__(GAME_NAME, experiment, game_instance)
+     def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
+         super().__init__(game_name, experiment, game_instance)

    
  class TabooGameBenchmark(GameBenchmark):

-      def __init__(self):
-          super().__init__(GAME_NAME)

+     def __init__(self, game_spec: GameSpec): # see next section on how to import GameSpec
+          super().__init__(game_spec)

-     def get_description(self):
-         return "Taboo game between two agents where one has to describe a word for the other to guess."

-     def is_single_player(self):
-         return False

      def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
-         return Taboo(experiment, player_models)
+         return Taboo(self.game_name, self.game_path, experiment, player_models)

      def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
-         return TabooScorer(experiment, game_instance)
+         return TabooScorer(self.game_name, experiment, game_instance)     

  # if main is defined
  def main(): 
      # select one experiment and instance
+     game_path = os.path.dirname(os.path.abspath(__file__))
+     experiments = file_utils.load_json("in/instances.json", game_path)
-     experiments = file_utils.load_json("in/instances.json", "taboo")
    
```
### Games using images (or other resources provided in the instances file)
If your game uses images or other external resources, make sure that the paths in the instances file are relative to the game directory and that the GameMaster takes care of combining them with the actual game directory (using `self.game_path`) after reading them from the instance file.
See [cloudgame](https://github.com/clp-research/clemgames/tree/main/cloudgame) for a minimal example. 

## Restructuring of clembench core code

During the implementation of the changes above, the main code was also restructured and transformed into an actual python module, which is now named `clemcore` and contains the sub-modules `backends`, `clemgame`, `utils` and the entry-point `cli.py`. (The module documentation can be found at `clembench/docs/html/index.html`)

This means that all imports have to be adapted to the new structure as follows (the example uses master.py, if your game consists of more game files, adapt them accordingly)


```python
# instancegenerator.py

from clemcore.clemgame import GameInstanceGenerator

```

```python
# master.py

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameBenchmark, Player, DialogueGameMaster, GameScorer

from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
# or 
import clemcore.clemgame.metrics as ms

from clemcore.utils import file_utils, string_utils

# if your game requires game-specific imports (like, for example, utils.py), change them to be relative to your game folder  
- from games.YOURGAME.utils import ...
+ from utils import ...

```

## Adaption of logging

The configuration of the logging module (set in `clemcore/__init__.py`) 
are actually passed on automatically and don't need to be imported over 
and over again, so get_logger was removed and has to be replaced by 
the logging call below

```python
- from clemgame import get_logger
- logger = get_logger(__name__)

+ import logging
+ logger = logging.getLogger(__name__)
```

## Game specific requirements
If your game requires specific packages, create a `requirements.txt` in your games folder. Game-specific requirements should no longer be part of the general clembench requirements.
