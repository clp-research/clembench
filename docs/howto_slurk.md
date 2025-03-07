# The Lightweight Dialogue Game framework

## How to integrate with Slurk

### Multiplayer games between a human player and a LM using slurk

In order to have a human player interact with a language model via the slurk
chat interface, the game master is also a slurk bot that connects to a running
slurk server. The game loop is then controlled via incoming slurk events:

```python
class ChatMaster(GameMaster, APIChatBot):
    def __init__(self, experiment, model_name, token, user, task, host, port):
        GameMaster.__init__(self, GAME_NAME, experiment, model_name)
        APIChatBot.__init__(self, token, user, int(task), host, port)
        self.sio.on("text_message", self.incoming_message())
```

In order to start the game loop you can either directly call the ChatMaster object's `run()` method or its `play()` method (that then calls run()).

The core logic is driven by `text_message` events that are handled by the `incoming_message()` method. The bot checks that the message came from the relevant user, sets the Player message to the incoming string and creates the new dialog object to pass on to the LM just like before.

In the example, the slurk bot also implements general game logic. Before the interaction properly starts, the human player has to type the `/ready` command and when the dialog ends, the room is closed.

#### Running the GameMaster as a slurk bot

The repository contains a setup that lets human players enter an entry room that dynamically creates individual rooms for each player (ConciergeBot).

In order to run the whole setup locally, you need:

- docker
- the [slurk repository](https://github.com/clp-research/slurk)
- the [slurk-bots repository](https://github.com/clp-research/slurk-bots) (containing the ConciergeBot)
- this repository (101_clembench)

All three repositories should be located at the same level:
```commandline
.
├── 101_clembench
├── slurk
└── slurk-bots
```

Navigate into the slurk-bots repository and run
````commandline
../101_clembench/scripts/slurk.sh
````

This script will set up a local slurk server, build and run both bots, and generate an access token to log into slurk at `localhost:5000`, like so (you can choose any name):
```commandline
TOKEN for logging into slurk interface at localhost:5000
64211e7b-8e90-4cd8-a07a-d24130e0c81d
```

In the mock version, this will just repeat "I don't know anything about that." for the LLM. To connect to a model, change "mock" to a model name in games/chatgame/master.py's main(). Running slurk.sh will then let you interact with the LM via a local slurk server.
