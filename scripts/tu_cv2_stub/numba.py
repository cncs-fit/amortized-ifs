"""Minimal numba stub for Tu et al. helper imports unused in MNIST fitting."""


def njit(*decorator_args, **decorator_kwargs):
    if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
        return decorator_args[0]

    def decorate(function):
        return function

    return decorate
