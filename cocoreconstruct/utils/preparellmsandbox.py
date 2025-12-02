import numpy as np
import json

from utils.coco import init_board
from llm_sandbox import SandboxSession, SandboxBackend

import logging

logger = logging.getLogger(__name__)


class PrepareLLMSandBox:
    def __init__(self, config: dict):
        # Create ONE session for the game
        self.session = SandboxSession(
            lang=config["sandbox_lang"],
            backend=SandboxBackend.DOCKER,
            image=config["sandbox_docker_image"],
            docker_network="isolated",
            skip_environment_setup=config["skip_docker_env_setup"],   # only if your image supports it
            verbose=False,
        )
        # Manually enter context so we can keep it open
        self.session.__enter__()
        logger.info("Initialized LLM Sandbox session.")

    def run_code(self, response: str, board: np.ndarray, max_rows: int = 8, max_cols: int = 8) -> dict:
        """Run a single turn's code and return stdout, stderr, exit code."""
        logger.info(f"Preparing to run code in LLM Sandbox session. Response: {response}, max_rows: {max_rows}, max_cols: {max_cols}")
        if board is not None:
            logger.info("Using provided board state for code execution.")
            current_board = board
        else:
            logger.info("Initializing new board for code execution.")
            current_board = init_board(max_rows, max_cols)


        code_stats = {"move": 0, "remove": 0, "clear": 0, "undo": 0}

        if "removeshape" in response:
            response = response.replace("removeshape", "self.removeshape")
            code_stats["remove"] += 1
        elif "undo" in response:
            response = response.replace("undo", "self.undo")
            code_stats["undo"] += 1
        else:
            if "move(" in response:
                code_stats["move"] += 1
            if "clear(" in response:
                code_stats["clear"] += 1

        logger.info(f"Code to execute: {response}, {type(response)}")


        code_exec = f"""
import numpy as np
import json
from coco import(
        init_board,
        plot_board,
        put,
        move,
        remove,
        clear,
        SameShapeStackingError,
        SameShapeAtAlternateLevels,
        NotOnTopOfScrewError,
        DepthMismatchError,
)


error = None
board=np.array({current_board.tolist()})
try:
    {response}
except Exception as e:
    error = str(e)    
np.save("board.npy", board)
with open("result.json", "w") as f:
    data = {{"error": error}}
    json.dump(data, f)
"""
        retdata = {"code_stats": code_stats}
        logger.info("Running code in LLM Sandbox session.")
        result = self.session.run(code_exec)


        if not result.exit_code:
            # Copy files out
            self.session.copy_from_runtime("/sandbox/board.npy", "board.npy")
            self.session.copy_from_runtime("/sandbox/result.json", "result.json")        
            board = np.load("board.npy")
            with open("result.json", "r") as f:
                codeerror = json.load(f)
                retdata["error"] = codeerror.get("error")

        else:
            board = None
            retdata["error"]= "Sandbox execution failed"

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "board": board,
            "error": retdata.get("error"),
            "code_stats": retdata.get("code_stats"),
        }

    def close(self):
        """Close the sandbox when the game ends."""
        logger.info("Closing LLM Sandbox session.")
        self.session.__exit__(None, None, None)        