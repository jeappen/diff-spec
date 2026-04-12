import numpy as np
import os
from contextlib import contextmanager

COLORED = False
IMPLIES_TRICK = False
HARDNESS = 100.0  # Reduce hardness of softmax to propagate gradients more easily


@contextmanager
def set_hardness(hardness: float):
    """Set the hardness of the softmax function for the duration of the context.
    Useful for making evaluation strict while allowing gradients to pass through during training.

    :param hardness: hardness of the softmax function
    :type hardness: float
    """
    global HARDNESS
    old_hardness = HARDNESS
    HARDNESS = hardness
    yield
    HARDNESS = old_hardness


if COLORED:
    from termcolor import colored
else:

    def colored(text, color):
        return text

from stlpy.STL import LinearPredicate, NonlinearPredicate, STLTree


def inside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being inside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return inside_rectangle:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_min)
    left = LinearPredicate(-a1, -y1_max)

    a2 = np.zeros((1, d))
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_min)
    bottom = LinearPredicate(-a2, -y2_max)

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        inside_rectangle.__str__ = lambda: name
        inside_rectangle.__repr__ = lambda: name

    return inside_rectangle


def outside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_max)
    left = LinearPredicate(-a1, -y1_min)

    a2 = np.zeros((1, d))
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_max)
    bottom = LinearPredicate(-a2, -y2_min)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        outside_rectangle.__str__ = lambda: name
        outside_rectangle.__repr__ = lambda: name

    return outside_rectangle


# Backend dispatch for default_tensor. Historically this module branched on
# DIFF_STL_BACKEND *at import time*, binding a different `default_tensor`
# into the module globals for each backend. Downstream modules that did
# `from ds.utils import default_tensor` then snapshotted whichever version
# happened to be live when they were first imported — so flipping the env
# var and calling `importlib.reload(ds_utils)` in a test setUp rebound
# ds.utils.default_tensor but left consumers (e.g. ds/stl.py) pointing at
# the stale snapshot of the other backend. That caused the
# `jnp.asarray(..., dtype=torch.float32)` failures when torch and jax
# tests ran in the same pytest session.
#
# Fix: dispatch on every call. Each backend keeps its own private constants
# so a call always uses the dtype/device that matches the function it runs.
# Selecting a backend is now just setting DIFF_STL_BACKEND; no reload needed.

_jax = None
_jax_device = None
_jax_dtype = None
try:
    import jax as _jax_mod
    from jax import numpy as _jnp
    _jax = _jax_mod
    _jax_device = _jax.devices()[0]
    _jax_dtype = _jnp.float32
except ImportError:
    pass

_torch = None
_torch_device = None
_torch_dtype = None
try:
    import torch as _torch_mod
    _torch = _torch_mod
    _torch_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    _torch_dtype = _torch.float32
except ImportError:
    pass


def _active_backend() -> str:
    # Default is jax; opt into torch explicitly via DIFF_STL_BACKEND=torch.
    val = os.environ.get("DIFF_STL_BACKEND", "").lower()
    if val == "torch":
        if _torch is None:
            raise RuntimeError("DIFF_STL_BACKEND=torch but torch is not installed")
        return "torch"
    if _jax is None:
        raise RuntimeError("jax backend requested but jax is not installed")
    return "jax"


def _jax_default_tensor(x: np.ndarray, device=None, dtype=None):
    return _jax.device_put(
        _jnp.asarray(x, dtype=_jax_dtype if dtype is None else dtype),
        _jax_device if device is None else device,
    )


def _torch_default_tensor(x: np.ndarray, device=None, dtype=None):
    return _torch.tensor(
        x,
        dtype=_torch_dtype if dtype is None else dtype,
        device=_torch_device if device is None else device,
    )


def default_tensor(x: np.ndarray, device=None, dtype=None):
    """Create a backend tensor, choosing backend via $DIFF_STL_BACKEND at call time.

    Safe against cross-backend reloads: the dispatch closure is stable, only
    the env-var lookup varies, so `from ds.utils import default_tensor`
    captures the dispatcher rather than a backend-specific snapshot.
    """
    if _active_backend() == "jax":
        return _jax_default_tensor(x, device=device, dtype=dtype)
    return _torch_default_tensor(x, device=device, dtype=dtype)


# Preserve the public `DEFAULT_DEVICE` / `DEFAULT_DATATYPE` names for any
# consumer that reads them as module attributes. Resolve dynamically so
# attribute access always reflects the active backend.
def __getattr__(name):
    if name == "DEFAULT_DEVICE":
        return _jax_device if _active_backend() == "jax" else _torch_device
    if name == "DEFAULT_DATATYPE":
        return _jax_dtype if _active_backend() == "jax" else _torch_dtype
    raise AttributeError(name)
