from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.runner.loader import ModuleResolutionError, discover_modules, load_module_instance

VALID_COMMANDS = ("inference", "test", "train", "view", "list", "help")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m modules.runner",
        description="Central runner for machine learning modules.",
    )
    parser.add_argument(
        "command",
        choices=VALID_COMMANDS,
        help="Action to perform: inference, test, train, view, or help",
    )
    parser.add_argument(
        "module",
        nargs="?",
        default=None,
        help='Dotted path to the module, e.g. "deep-learning.mnist" (resolves to deep-learning/mnist.py)',
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to the module's command function",
    )
    return parser


def print_module_list(base_dir: Path):
    """Print the list of available modules.

    Args:
        base_dir: Path to the base directory."""

    discovered = discover_modules(base_dir)
    if not discovered:
        print("No modules found.")
        return

    module_width = max(len("MODULE"), *(len(d.dotted_path) for d in discovered))
    class_width = max(len("CLASS"), *(len(d.class_name or "<error>") for d in discovered))

    for module in discovered:
        if module.error:
            print(f"{module.dotted_path:<{module_width}}  {'<error>':<{class_width}}  {module.error}")
        else:
            cmds = ", ".join(module.commands) if module.commands else "(none implemented)"
            print(f"{module.dotted_path:<{module_width}}  {module.class_name:<{class_width}}  {cmds}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    namespace = parser.parse_args(argv)

    if namespace.command == "list":
        print_module_list(Path.cwd())
        return 0

    if namespace.module is None:
        parser.error("the following arguments are required: module")

    try:
        module_instance = load_module_instance(namespace.module, base_dir=Path.cwd())
    except ModuleResolutionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    cmd_handler = getattr(module_instance, namespace.command)

    # Run the command associated with the module instance
    try:
        cmd_handler(namespace.args)
    except SystemExit:
        raise
    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(
            f"Error while running '{namespace.command}' on '{namespace.module}': {e}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
