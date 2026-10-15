"""Small, shared output-path safeguards for future revision experiments.

The historical ``outputs/`` tree is read-only evidence.  New infrastructure
must use a separate root such as ``outputs_major_revision/``.  Resolving paths
before comparison also rejects aliases through an existing symlink.
"""

from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    """Return the repository root when this package lives in a worktree."""
    return Path(__file__).resolve().parents[1]


def historical_outputs_root() -> Path:
    """The protected historical artifact tree, normalized without requiring it."""
    return (_repository_root().parents[1] / "outputs").resolve(strict=False)


def protect_output_root(output_root: Path) -> Path:
    """Return a normalized future-output root or reject historical outputs.

    ``Path.resolve(strict=False)`` resolves all existing path components, so an
    existing symlink pointing into the historical tree is rejected as well as a
    direct child path.  This module deliberately does not create the root.
    """
    resolved = Path(output_root).expanduser().resolve(strict=False)
    protected = historical_outputs_root()
    # Keep the public contract conservative too: future callers should never
    # target a root literally named ``outputs``.  The protected-tree check below
    # is still required, because a descendant or symlink need not have that
    # final component.
    if resolved.name.casefold() == "outputs":
        raise ValueError(
            "Refusing to write revision artifacts to an outputs/ root; use outputs_major_revision/"
        )
    try:
        resolved.relative_to(protected)
    except ValueError:
        return resolved
    raise ValueError(
        "Refusing to write revision artifacts to the protected historical outputs/ tree; "
        "use outputs_major_revision/"
    )


def experiment_directory(output_root: Path, experiment_id: str) -> Path:
    """Return an isolated experiment directory without claiming ownership of it.

    Multiple independent writers may use different files/directories below the
    same experiment directory.  Each writer must still reject its own existing
    target instead of overwriting it.
    """
    return protect_output_root(output_root) / experiment_id
