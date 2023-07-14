import functools
import hashlib
import os
import time
from contextlib import AbstractContextManager
from typing import Any, Callable, Collection, Generic, List, Optional, Type, TypeVar


def compute_md5_hash_from_bytes(input_bytes: bytes) -> str:
    """Compute the MD5 hash of a byte array."""
    return str(hashlib.md5(input_bytes).hexdigest())


class chdir(AbstractContextManager):  # noqa: N801
    """Non thread-safe context manager to change the current working directory."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._old_cwd: List[str] = []

    def __enter__(self) -> None:
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo) -> None:  # type: ignore
        os.chdir(self._old_cwd.pop())


Q = TypeVar("Q", bound=Callable[..., Any])


def retry(no_retry_on: Optional[Collection[Type[Exception]]] = None) -> Callable[[Q], Q]:
    """Retry a function if it throws an exception."""

    def decorator(func: Q) -> Q:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            error = None
            for backoff in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if no_retry_on is not None and type(e) in no_retry_on:
                        raise e

                    # Backoff and try again
                    time.sleep(backoff)
                    error = e
                    continue

            if error is not None:
                raise error  # type: ignore

        return wrapper  # type: ignore

    return decorator


T = TypeVar("T")
S = TypeVar("S")


class _SingletonWrapper(Generic[T]):
    def __init__(self, cls: Type[T]):
        self.__wrapped__ = cls
        self._instance: Optional[T] = None
        functools.update_wrapper(self, cls)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Returns a single instance of decorated class"""
        if self._instance is None:
            self._instance = self.__wrapped__(*args, **kwargs)
        return self._instance

    # Pass class methods/attributes through to the wrapped class
    def __getattr__(self, attr: str) -> Any:
        if hasattr(self.__wrapped__, attr):
            return getattr(self.__wrapped__, attr)
        raise AttributeError(f"{self.__wrapped__.__name__} has no attribute {attr}")


def singleton(cls: Type[S]) -> _SingletonWrapper[S]:
    return _SingletonWrapper(cls)
