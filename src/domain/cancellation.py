from typing import Callable


CancellationCheck = Callable[[], bool]


class CancellationRequestedError(RuntimeError):
    pass


def raise_if_cancelled(cancellation_check: CancellationCheck | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise CancellationRequestedError("Reconstruction was cancelled by the operator")
