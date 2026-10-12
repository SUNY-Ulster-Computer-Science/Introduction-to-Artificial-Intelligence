import pytest

from ..game.sokoban import Action, Sokoban, parse_level
from ..harness import load_levels


@pytest.fixture(scope="session")
def levels() -> dict[str, str]:
    return load_levels()


def make_game(level_text: str) -> Sokoban:
    return parse_level(level_text)


def execute_plan(level_text: str, plan: list[Action]) -> bool:
    """Execute `plan` in a fresh real environment. True iff it solves the puzzle.

    Raises ValueError if the plan contains an illegal move."""

    game = parse_level(level_text)
    for action in plan:
        game.update(action)
    return game.is_solved()
