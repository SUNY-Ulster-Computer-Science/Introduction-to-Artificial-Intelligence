"""Experiment harness.

For each (algorithm, level) pair it creates a fresh real environment and a
SearchAgent, runs one episode, and reports:
    solved       are all boxes on goals?
    plan length  number of actions executed
    expanded     nodes expanded by the search
    seconds      wall-clock time"""

import argparse
import time
from pathlib import Path

from .agents import SearchAgent
from .astar import astar_search
from .bfs import bfs_search
from .dfs import dfs_search
from .game.sokoban import Sokoban, parse_level

LEVEL_DIR = Path(__file__).parent / "levels"
DEFAULT_NODE_LIMIT = 200_000


ALGORITHMS = {
    "bfs": bfs_search,
    "dfs": dfs_search,
    "astar": astar_search,
}


def load_levels() -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted(LEVEL_DIR.glob("*.txt"))}


def play(game: Sokoban, agent: SearchAgent, max_steps: int = 100_000) -> bool:
    """Run one round of the game. Returns True if the puzzle was solved."""

    percept = game.reset()
    for _ in range(max_steps):
        action = agent.act(percept)
        if action is None:
            break

        percept = game.update(action)
        if game.is_solved():
            return True

    return game.is_solved()


def run_sample(algo_name: str, level_text: str, node_limit: int = DEFAULT_NODE_LIMIT) -> dict[str, object]:
    game = parse_level(level_text)
    agent = SearchAgent(ALGORITHMS[algo_name], node_limit)
    t0 = time.perf_counter()
    solved = play(game, agent)
    elapsed = time.perf_counter() - t0
    return {
        "solved": solved,
        "plan_length": game.steps() if solved else None,
        "expanded": agent.get_plan()[1],
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", nargs="+", default=list(ALGORITHMS), help="Algorithms to run. All if omitted.")
    parser.add_argument("--levels", nargs="+", default=None, help="Levels to run. All if omitted.")
    args = parser.parse_args()

    levels = load_levels()
    names = args.levels or list(levels)

    all_results = {}
    for level in names:
        for algo in args.algos:
            result = run_sample(algo, levels[level], DEFAULT_NODE_LIMIT)
            all_results[(level, algo)] = result

    print(f"{'level':<6}{'algo':<8}{'solved':<8}{'plan len':>9}{'expanded':>10}{'seconds':>9}")
    print("-" * 50)
    for (level, algo), result in all_results.items():
        plan_len = "-" if result["plan_length"] is None else result["plan_length"]
        print(
            f"{level:<6}{algo:<8}{result['solved']!s:<8}{plan_len!s:>9}{result['expanded']!s:>10}{result['seconds']:>9.2f}"
        )


if __name__ == "__main__":
    main()
