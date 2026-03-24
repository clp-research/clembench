# Clean Up

Implemented by [Ruilin Yang](https://github.com/RuilinYang-beta) and [Karl Osswald](https://github.com/ansovald/)

Clean Up is a dialogue-based cooperative game for evaluating large language models (LLMs) in goal-directed strategic negotiation. The game tests **spatial reasoning, negotiation, and strategy development** by requiring two players to align object configurations across individually visible grids.

## Game Overview

Clean Up is a turn-based dialogue game for two players focused on cooperative strategy development and object rearrangement. Both players are presented with identical 7x7 ASCII-like grids, with a number of randomly distributed objects in the form of capital letters placed on them. Placement is different for each player and each player can only see their own grid. The goal is to rearrange the objects in such away that their respective positions match across both grids. Scoring is based on the Euclidean distance of the identical objects on both grids and the number of penalties accumulated through rule violations.

### Difficulty Levels and Instances 
All grids have the same size (7 × 7), and contain obstacles in form of horizontal and vertical lines, branches, crossings, and corners. Difficulty is controlled along two dimensions: number of empty cells and number of objects. Easy has 34 empty cells, medium 29, and hard 24:

```
 1234567     1234567     1234567
╔═══════╗   ╔═╤═╤═╤═╗   ╔══╤════╗
║◌◌◌◌◌◌◌║ 1 ║◌│◌│◌│◌║ 1 ║◌◌│◌◌◌◌║ 1
║◌◌◌◌◌◌◌║ 2 ║◌└─┼─┴─╢ 2 ╟──┘◌◌┌─╢ 2
║◌◌◌◌◌┌─╢ 3 ║◌◌◌│◌◌◌║ 3 ║◌◌◌◌◌│◌║ 3
║◌◌◌┌─┤◌║ 4 ╟──┬┴───╢ 4 ╟─┐◌┌─┼─╢ 4
║◌◌◌│◌├─╢ 5 ║◌◌│◌◌◌◌║ 5 ║◌├─┤◌│◌║ 5
╟───┼─┘◌║ 6 ║◌◌│◌◌◌◌║ 6 ╟─┤◌├─┤◌║ 6
║◌◌◌│◌◌◌║ 7 ║◌◌│◌◌◌◌║ 7 ║◌│◌│◌│◌║ 7
╚═══╧═══╝   ╚══╧════╝   ╚═╧═╧═╧═╝
  easy       medium       hard
```

For each difficulty level, there are instances with 3, 5, and 7 objects.

### Game Mechanics
The game is turn based, and the maximum number of rounds is fixed to $4 \times n_{obj}$ (where $n_{obj}$ is the object count). Players can collectively accumulate $2 \times n_{obj} + 2$ penalties.

Initially, P1 (Player 1) is provided with a game description including round and penalty limit and their respective grid, following which they have to send a message to P2 (Player 2). P2 receives the same prompt and P1's message. In each turn, a player can either send a message to their counterpart or move an object on their grid.

Each following turn starts with a message from the Game Master consisting of three parts: (1) feedback on the last turn, i.e., either a notification that the message has been passed on or the updated grid if they moved an object; (2) information on the game state, i.e., current and maximum penalties and rounds, and (3) either the other player's message or a notification that they moved an object. 

If a player does not follow the format, tries to move an object to a non-empty space or outside the grid bounds, or tries to move an object that doesn't exist, they receive a penalty and are re-prompted with information on the nature of their mistake, and have to send a new command.

The game ends if (1) both players agree to end it, (2) round limit is exceeded, or (3) penalty limit is exceeded.

### Evaluation Metrics
A game counts as failure and is not scored if the penalty limit is exceeded. The Quality Score $\mathbf{QS}$ is calculated as follows:
$$\mathbf{QS} = \mathbf{DS} \cdot \mathbf{PS} \cdot 100; \ \mathbf{QS} \in [0, 100]$$

#### Distance Score
$\mathbf{DS} \in [0,1]$ is calculated from three components, each representing the sum of Euclidean distances for all identical objects on both grids: the Initial Distance Sum $\mathbf{I}$ at the start of the game, the Final Distance Sum $\mathbf{F}$ at the end of the game, and the Expected Distance Sum $\mathbf{E}$ that approximates the distances for randomly placed objects. 

The expected distance $\mathbb{E}$ for two independent variables $i$ and $j$ on one discrete dimension $\{1, 2, \dots, w\}$ is calculated by summing up the absolute values of all differences of all possible combinations of $i$ and $j$, and dividing it by the square of the maximum value $w$:

$$\begin{align*}
    \mathbb{E}[|i - j|] &= \frac{1}{w^2} \sum_{i=1}^{w} \sum_{j=1}^{w} |i - j| \\
    &= \frac{1}{w^2} \cdot \frac{(w-1)(w+1)}{3} \\
    &= \frac{w^2 - 1}{3w}
\end{align*}$$

For two dimensions, we plug this sum into the formula for the Euclidean distance, with $w$ being the grid width and $h$ the height, and calculate the expected distance of two objects $o_1$ and $o_2$ as follows:
$$\begin{align*}
    \mathbb{E}[|o_1, o_2|] = \sqrt{\left(\frac{w^2 - 1}{3 \times w}\right)^2 + \left(\frac{h^2 - 1}{3 \times h}\right)^2}
\end{align*}$$
For a $7 \times 7$ grid, this evaluates to:
$$\begin{align*}
    \mathbb{E}[|o_1, o_2|] &= \sqrt{2\left(\frac{7^2 - 1}{3 \times 7}\right)^2} \\
    &= \frac{16}{7} \sqrt{2} \\
    &\approx 3.232\dots 
\end{align*}$$

The expected distance sum $\mathbf{E}$ is calculated by multiplying $\mathbb{E}$ with the number of objects. Note that this does not take objects already placed on the grid into account, but it is a sufficiently close approximation.

The Expected Distance Score $ES$ and Distance Reduction Score $RS$, quantifying how close the players came to the goal of perfect alignment of all objects, are calculated as follows:
$$\begin{align*}
ES &= \max \left\{0,1 - \frac{\mathbf{F}}{\mathbf{E}} \right\} \\
RS &= \max \left\{0,1 - \frac{\mathbf{F}}{\mathbf{I}} \right\}
\end{align*}$$
$\mathbf{DS}$ is then either $0$ if final placement is worse than random, or calculated as the mean of both scores:
$$\begin{align*}
\mathbf{DS} &= \begin{cases}
\frac{ES+RS}{2} & if \quad ES > 0 \\
0 & otherwise
\end{cases}
\end{align*}$$

#### Penalty Score
$\mathbf{PS} \in [0.5,1]$ is calculated from the penalty count $P$ normalized against max. penalties $P_m$ as follows:
$$\mathbf{PS} = \frac{P_m}{P-2P_m}+1.5$$
We chose the hyperbolic function because it is lenient for low $P$ and harsher for $P$ close to $P_m$, and the interval of $[0.5,1]$ to have a clear offset between aborted games and successful games with high penalty count.

# Reference

Clean Up is featured in the following paper:

Hakimov, S., Bernard R, Leiber T, Osswald K, Richert K, Yang R, Bernardi R, Schlangen D. (2026). [The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models.](https://arxiv.org/abs/2510.08098) *The 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL).*