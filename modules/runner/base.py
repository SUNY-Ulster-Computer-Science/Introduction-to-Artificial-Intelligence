"""Base class for runner-compatible ML modules."""

from __future__ import annotations


class MLModule:
    """Subclass this for every ML module used with the runner."""

    def train(self, args: list[str]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement train()")

    def test(self, args: list[str]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement test()")

    def inference(self, args: list[str]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement inference()")

    def view(self, args: list[str]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement view()")

    def help(self, args: list[str]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement help()")
