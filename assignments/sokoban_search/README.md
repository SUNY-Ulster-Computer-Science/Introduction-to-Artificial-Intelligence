# Assignment: Sokoban Agents

In this assignment you build **search-based agents** that solve [Sokoban](https://en.wikipedia.org/wiki/Sokoban) puzzles,
then compare how different search strategies behave on the same problems.

## Learning goals

By the end you will be able to:

1. Write an agent that maps percepts to actions and plans by searching.
2. Implement depth-first and A* search by adapting a provided breadth-first search.
3. Compare the algorithms on solved / plan length / nodes expanded, and explain the differences.

## Setup

Follow **Getting Started** in the [root README](../../README.md) to clone the repository and create and activate a virtual
environment. `pytest` is installed along with the other dependencies.

Run every command below **from the repository root** to avoid import errors. With your virtual environment activated:

```
python -m assignments.sokoban_search.harness          # run experiments
python -m pytest assignments/sokoban_search/tests     # run the tests
```

If you use uv, you can skip activation and prefix each command with `uv run` instead:

```
uv run python -m assignments.sokoban_search.harness
uv run python -m pytest assignments/sokoban_search/tests
```

## The game

Push every box (`$`) onto a goal (`.`). You (`@`) can walk up, down, left and right, and can push **one** box at a time.
Boxes cannot be pulled, and cannot be pushed into a wall or into another box.

| Symbol | Meaning | Symbol | Meaning          |
|--------|---------|--------|------------------|
| `#`    | wall    | `*`    | box on a goal    |
| `.`    | goal    | `+`    | player on a goal |
| `$`    | box     | `@`    | player           |

Actions are the members of the `Action` enum: `Action.UP`, `Action.DOWN`, `Action.LEFT`, `Action.RIGHT`. Every action,
whether you walk or push, costs 1, so the best plan is the one with the fewest actions.

## What is provided

Read these files before you start. **Do not modify them.**

| File              | Contents                                                                                            |
|-------------------|-----------------------------------------------------------------------------------------------------|
| `game/sokoban.py` | `Action`, `Position`, `Objects`, `Percept`, and the `Sokoban` class. Read the module docstring.     |
| `game/search.py`  | `Node`, `SearchResult`, `reconstruct_plan`, and the **shared search signature and counting rules**. |
| `bfs.py`          | A fully worked breadth-first search. **This is your template.**                                     |
| `harness.py`      | Runs algorithms on levels and prints a results table.                                               |
| `levels/`         | Six levels, `L1` to `L6`.                                                                           |
| `tests/`          | The tests you should make pass.                                                                     |

### Two roles for `Sokoban`

The `Sokoban` class plays two roles:

- **The real environment.** The harness/test owns it and advances it with `update(action)`. Your agent never sees it.
  The agent only receives a `Percept`: the walls, the goals, and the current `Objects` (the player and the boxes).
- **The agent's model of the world.** Your agent builds its own private copy with `Sokoban.from_percept(percept)`. That
  copy is *static*: calling `update` on it raises an error. You use it to imagine the consequences of actions without
  changing anything real.

### The search interface

Every search algorithm in this assignment has the same signature:

```python
def some_search(game: Sokoban, node_limit: int) -> SearchResult
```

Inside a search, use only these methods on `game`:

| Method                       | Returns                                                                         |
|------------------------------|---------------------------------------------------------------------------------|
| `game.objects()`             | The starting state (an `Objects`: player position and the set of box positions) |
| `game.legal_actions(state)`  | The legal actions in `state`, always in the order UP, DOWN, LEFT, RIGHT         |
| `game.result(state, action)` | The new state after a legal action                                              |
| `game.is_goal(state)`        | `True` when every box is on a goal                                              |

States are hashable, so they can go straight into a `set` or `dict`. A `SearchResult` holds the `plan` (a list of
`Action`, or `None` if no plan was found) and `nodes_expanded`.

**Counting convention.** Every algorithm follows the same rules so results are comparable. Read the docstring in
`game/search.py` carefully; in short:

- A node is *expanded* when it is removed from the frontier, goal-tested, and (if it is not a goal) its children are
  generated.
- `nodes_expanded` counts every expanded node, including the final goal node.
- The goal test happens when a node is removed from the frontier, not when it is generated.
- If `nodes_expanded` reaches `node_limit` before a solution is found, stop and return `SearchResult(None, nodes_expanded)`.

## What you implement

| Part | File        | Task                                      |
|------|-------------|-------------------------------------------|
| 1    | `agents.py` | `SearchAgent._plan` and `SearchAgent.act` |
| 2    | `dfs.py`    | `dfs_search`                              |
| 3    | `astar.py`  | `astar_search`                            |

In each of these files, there are TODO comments and exceptions which you will replace with your implementation.

## Harness Program

A harness program has been provided to allow for easier debugging and visualization if your agent's behavior.

```
python -m assignments.sokoban_search.harness                          # everything
python -m assignments.sokoban_search.harness --algos bfs astar        # some algorithms
python -m assignments.sokoban_search.harness --levels L3 L5           # some levels
python -m assignments.sokoban_search.harness --display                # show game states
```

For each algorithm and level the harness reports whether the puzzle was solved, the plan length, the nodes expanded,
and the wall-clock time.

## Testing

```
python -m pytest assignments/sokoban_search/tests                       # everything
python -m pytest assignments/sokoban_search/tests/test_dfs.py           # one file
```

The tests replay your plans in the real environment, so any *correct* plan passes.

## Levels

| Level | Description                                                         |
|-------|---------------------------------------------------------------------|
| L1    | Tiny sanity check                                                   |
| L2    | One box in an open room                                             |
| L3    | Two boxes in a larger room                                          |
| L4    | Three boxes                                                         |
| L5    | Two boxes with the goals on the far side of a wall                  |
| L6    | Something is different about this one. What does each algorithm do? |
