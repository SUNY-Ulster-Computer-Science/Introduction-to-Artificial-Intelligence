"""Sokoban game and environment.

The things that can change during play are the player and the boxes. They live together in an immutable, hashable
`Objects` value, which is also what a search algorithm uses as its *state*.

The class `Sokoban` plays two roles, and it is important to keep them apart:

* The real, outside environment which serves as the game's ground truth.
* The static, internal environment which the agent uses as its world model."""

from dataclasses import dataclass
from enum import Enum
from typing import Self


class Action(Enum):
    """The four moves which objects can make."""

    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"

    @property
    def delta(self) -> tuple[int, int]:
        """The (x, y) offset this action moves by."""

        return {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
        }[self]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Position:
    """A position for an object on the game grid."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Objects:
    """Everything that can move: the player and the boxes."""

    player: Position
    boxes: frozenset[Position]


@dataclass(frozen=True, slots=True)
class Percept:
    """What the agent sees."""

    walls: frozenset[Position]
    goals: frozenset[Position]
    objects: Objects


class Sokoban:
    """A Sokoban level, usable as the real environment or as a private model.

    Fixed after construction: `walls`, `goals`, `start`.
    Changed only by `reset` / `update`: `objects`, `steps`."""

    def __init__(self, walls: frozenset[Position], goals: frozenset[Position], start: Objects, static: bool) -> None:
        self.walls = walls
        self.goals = goals
        self.start = start  # Save the original game starting state
        self.objects = start
        self.steps = 0
        self.static = static  # Whether the same is mutable or not

    @classmethod
    def from_percept(cls, percept: Percept, static: bool) -> Self:
        """Build a private copy of the world from a percept (an agent's model)."""

        return cls(percept.walls, percept.goals, percept.objects, static)

    def legal_moves(self, objects: Objects) -> list[Action]:
        """Legal actions when things are arranged as `objects`.

        Always returned in the order UP, DOWN, LEFT, RIGHT."""

        legal: list[Action] = []
        for action in Action:
            dx, dy = action.delta
            target = Position(objects.player.x + dx, objects.player.y + dy)
            # The player cannot move into the walls
            if target in self.walls:
                continue

            if target in objects.boxes:
                beyond = (target.x + dx, target.y + dy)
                # The player cannot move a box into a wall or another box
                if beyond in self.walls or beyond in objects.boxes:
                    continue

            legal.append(action)
        return legal

    def result(self, objects: Objects, action: Action) -> Objects:
        """The objects that result from taking a legal `action`."""

        dx, dy = action.delta
        target = Position(objects.player.x + dx, objects.player.y + dy)
        boxes = objects.boxes

        if target in boxes:
            # Push the box in the same direction the player was moving in
            beyond = Position(target.x + dx, target.y + dy)
            boxes = (boxes - {target}) | {beyond}

        return Objects(target, frozenset(boxes))

    def is_goal(self, objects: Objects) -> bool:
        """True when every box is on a goal."""

        return objects.boxes == self.goals

    def percept(self) -> Percept:
        return Percept(self.walls, self.goals, self.objects)

    def is_solved(self) -> bool:
        return self.is_goal(self.objects)

    def reset(self) -> Percept:
        """Reset the world to its initial state."""

        self.objects = self.start
        self.steps = 0
        return self.percept()

    def update(self, action: Action) -> Percept:
        """Advance the world by one step."""

        if self.static:
            raise RuntimeError("Cannot update a static game")

        if action not in self.legal_moves(self.objects):
            raise ValueError(f"Illegal action {action} from player position {self.objects.player}")

        self.objects = self.result(self.objects, action)
        self.steps += 1
        return self.percept()


def parse_level(text: str) -> Sokoban:
    """Parse level text into a Sokoban game object."""

    walls: set[Position] = set()
    goals: set[Position] = set()
    boxes: set[Position] = set()
    player: Position | None = None
    for r, line in enumerate(text.strip("\n").splitlines()):
        for c, ch in enumerate(line):
            pos = Position(r, c)
            if ch == "#":
                walls.add(pos)
            elif ch == ".":
                goals.add(pos)
            elif ch == "$":
                boxes.add(pos)
            elif ch == "*":
                boxes.add(pos)
                goals.add(pos)
            elif ch == "@":
                player = pos
            elif ch == "+":
                player = pos
                goals.add(pos)

    if player is None:
        raise ValueError("Level has no player")
    if len(boxes) != len(goals):
        raise ValueError("Level must have as many boxes as goals")

    objects = Objects(player, frozenset(boxes))
    return Sokoban(frozenset(walls), frozenset(goals), objects, False)
