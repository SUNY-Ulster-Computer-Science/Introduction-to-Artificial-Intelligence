from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from modules.runner.base import MLModule

COMMANDS = ("train", "test", "inference", "view", "help")

# Directories skipped when scanning for modules: the runner's own package, plus common non-project directories that
# shouldn't be treated as modules.
_SKIP_DIR_NAMES = {"runner", "__pycache__", ".git", ".venv", "venv", "env", "ml_modules"}


class ModuleResolutionError(Exception):
    """Raised when a dotted module path cannot be resolved or loaded."""


class NoModuleClassFound(ModuleResolutionError):
    """Raised when a file has no MLModule subclass."""


def resolve_module_path(dotted_path: str, base_dir: Path) -> Path:
    """Convert a dotted path like "modules.deep-learning.mnist" into a file path like "modules/deep-learning/dnn.py".

    Args:
        dotted_path: The dotted module path to resolve.
        base_dir: The base directory to resolve relative to.
    Returns:
        The resolved file path.
    """

    if not dotted_path or dotted_path.startswith(".") or dotted_path.endswith("."):
        raise ModuleResolutionError(f"Invalid module path: '{dotted_path}'")

    # Check for invalid dot placement
    parts = dotted_path.split(".")
    if any(part == "" for part in parts):
        raise ModuleResolutionError(f"Invalid module path: '{dotted_path}'")

    rel_path = Path(*parts).with_suffix(".py")
    return (base_dir / rel_path).resolve()


def load_module(dotted_path: str, base_dir: Path | None = None) -> ModuleType:
    """Load a module file given its dotted path, e.g. "modules.deep-learning.mnist".

    Args:
        dotted_path: The dotted module path to load.
        base_dir: The base directory to resolve relative to.
    Returns:
        The loaded module object.
    Raises:
         ModuleResolutionError if the file cannot be found or fails to load.
    """

    base_dir = base_dir if base_dir is not None else Path.cwd()
    module_file = resolve_module_path(dotted_path, base_dir)

    if not module_file.exists():
        raise ModuleResolutionError(f"Could not find module file for '{dotted_path}' (expected at: {module_file})")

    # Unique internal module name to avoid collisions in sys.modules.
    internal_name = "runner_dynamic__" + dotted_path.replace(".", "__").replace("-", "_")

    spec = importlib.util.spec_from_file_location(internal_name, module_file)
    if spec is None or spec.loader is None:
        raise ModuleResolutionError(f"Could not create import spec for: {module_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[internal_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # re-raise with context, keep original traceback
        raise ModuleResolutionError(f"Error while importing '{dotted_path}' ({module_file}): {exc}") from exc

    return module


def find_module_class(module: ModuleType, dotted_path: str) -> type[MLModule]:
    """Find the single `MLModule` subclass defined in a loaded module.

    If a module defines more than one such class, it can disambiguate by setting a module-level
    `RUNNER_CLASS = TheClassToUse`.

    Args:
        module: The Python module to search.
        dotted_path: The dotted path to the model for logging.
    Returns:
        The machine learning model module.
    """

    explicit = getattr(module, "RUNNER_CLASS", None)
    if explicit is not None:
        if inspect.isclass(explicit) and issubclass(explicit, MLModule):
            return explicit
        raise ModuleResolutionError(
            f"'{dotted_path}' sets RUNNER_CLASS, but it is not a class that subclasses runner.base.MLModule."
        )

    candidates = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, MLModule)
        and obj is not MLModule
        and obj.__module__ == module.__name__
    ]

    if not candidates:
        raise NoModuleClassFound(
            f"No MLModule subclass found in '{dotted_path}'. "
            "Define a class that inherits from runner.base.MLModule in that file."
        )
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise ModuleResolutionError(
            f"Multiple MLModule subclasses found in '{dotted_path}' ({names}). "
            "Set RUNNER_CLASS = YourClassName at module level to disambiguate."
        )
    return candidates[0]


def load_module_instance(dotted_path: str, base_dir: Path | None = None) -> MLModule:
    """Load a module by dotted path and return an instance of its MLModule subclass.

    Args:
        dotted_path: The dotted path to the module
        base_dir: The base directory to load the module from.
    Returns:
        An instance of `MLModule` contained within the selected class.
    """

    module = load_module(dotted_path, base_dir=base_dir)
    cls = find_module_class(module, dotted_path)
    try:
        return cls()
    except Exception as exc:
        raise ModuleResolutionError(f"Failed to instantiate '{cls.__name__}' from '{dotted_path}': {exc}") from exc


def discover_module_paths(base_dir: Path) -> list[str]:
    """Find dotted paths for every .py file under base_dir that could be a module.

    Skips the runner package itself, __init__.py files, and common non-project directories (.git, venv, etc).

    Args:
        base_dir: The base directory to load the module from.
    Returns:
        A list of dotted module paths.
    """

    dotted_paths: list[str] = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
        for filename in files:
            if not filename.endswith(".py") or filename == "__init__.py":
                continue
            file_path = Path(root) / filename
            rel = file_path.relative_to(base_dir)
            dotted_paths.append(".".join(rel.with_suffix("").parts))
    return sorted(dotted_paths)


def implemented_commands(cls: type[MLModule]) -> list[str]:
    """Return which handlers a class actually overrides."""

    return [cmd for cmd in COMMANDS if getattr(cls, cmd) is not getattr(MLModule, cmd)]


@dataclass
class DiscoveredModule:
    dotted_path: str
    class_name: str | None = None
    commands: list[str] = field(default_factory=list)
    error: str | None = None


def discover_modules(base_dir: Path | None = None) -> list[DiscoveredModule]:
    """Scan base_dir for runner modules (files defining an MLModule subclass).

    Files that fail to load (missing dependency, syntax error, etc.) are still reported, with the error attached.

    Args:
        base_dir: The base directory to load the module from.
    Returns:
        A list of discovered modules.
    """

    base_dir = base_dir if base_dir is not None else Path.cwd()
    results: list[DiscoveredModule] = []

    for dotted_path in discover_module_paths(base_dir):
        try:
            module = load_module(dotted_path, base_dir=base_dir)
            cls = find_module_class(module, dotted_path)
        except NoModuleClassFound:
            continue
        except ModuleResolutionError as exc:
            results.append(DiscoveredModule(dotted_path=dotted_path, error=str(exc)))
            continue
        except Exception as exc:
            results.append(DiscoveredModule(dotted_path=dotted_path, error=f"Failed to load: {exc}"))
            continue

        results.append(
            DiscoveredModule(
                dotted_path=dotted_path,
                class_name=cls.__name__,
                commands=implemented_commands(cls),
            )
        )

    return results
