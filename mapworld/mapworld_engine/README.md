# MapWorld Engine

**MapWorld** is a lightweight game engine for evaluating **multimodal conversational LLMs (cLLMs)** in a room-to-room navigation setting.  
It provides a controlled environment where an agent must navigate through a connected map of rooms (derived from graph structures) and reach a target, using multimodal cues.  

The environment is inspired by:  
- [ADE20K dataset](https://ade20k.csail.mit.edu/) — for realistic room images and categories.  
- [Sempix MapWorld](https://github.com/clp-research/sempix/tree/master/03_Tasks/MapWorld) — for graph-based map generation.  
- [Gymnasium](https://gymnasium.farama.org/) — for standardized RL-style environment interaction.  

---

## Features
- Create **different graph topologies** (grid, path, cycle, tree, star, ladder, etc.).
- Assign **room types and categories** from ADE20K.
- Introduce **ambiguity** (e.g., multiple bedrooms) to test model disambiguation.
- Render rooms with **realistic images** or as graph layouts.  
- Built on **Gymnasium API**, enabling easy integration with RL agents and LLM-powered agents.

---

## Notebook Demo
For a step-by-step walkthrough of how to:  
- Generate maps,  
- Assign categories and images,  
- Instantiate and interact with the environment,  
- Render episodes,  

see the provided Jupyter notebook:   
[`howto_setup_mapworld.ipynb`](./howto_setup_mapworld.ipynb)

---