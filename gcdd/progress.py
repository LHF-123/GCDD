from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def log_stage(message: str) -> None:
    print(message, flush=True)


def progress_iter(iterable: Iterable[T], total: int | None = None, desc: str = "") -> Iterator[T]:
    """Use tqdm when available, otherwise print lightweight progress updates."""
    try:
        from tqdm.auto import tqdm

        yield from tqdm(iterable, total=total, desc=desc)
        return
    except Exception:
        pass

    if total is None:
        yield from iterable
        return

    interval = max(1, total // 20)
    for i, item in enumerate(iterable, start=1):
        if i == 1 or i == total or i % interval == 0:
            print(f"{desc}: {i}/{total}", flush=True)
        yield item

