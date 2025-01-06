import functools
import math
import warnings

import numpy as np
from shapely.affinity import rotate, translate

# Definition of units where um = 1
# All dimensions are in um
mm = 1000
um = 1.0
nm = 0.001
sqrt3 = math.sqrt(3)
sqrt2 = math.sqrt(2)

unit_properties = {
    "nm": {"name": "nanometer", "ratio": nm, "display_name": "nm"},
    "um": {"name": "micrometer", "ratio": um, "display_name": "μm"},
    "mm": {"name": "milimeter", "ratio": mm, "display_name": "mm"},
}


def next_power_of_two(n):
    """Return the smallest power of two >= n.
    Example
    next_power_of_two(60) = 64
    next_power_of_two(64) = 64
    """
    return 1 << (n - 1).bit_length()


def ensure_ax(ax=None, subplot_kw={}):
    """Ensure one Axes object exists"""
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw=subplot_kw)
    return ax


def sanitize_rL_input(r=None, L=None, diameter=None, spacing=None):
    """Sanitize the r-L or diameter-spacing inputs for Geometry"""
    if (diameter is not None and spacing is not None) and (
        r is None and L is None
    ):
        r = diameter / 2
        L = spacing - diameter
    elif (r is not None and L is not None) and (
        diameter is None and spacing is None
    ):
        pass
    else:
        raise ValueError(
            "Provide either (r, L) or (diameter, spacing), but not both."
        )
    return r, L


def deprecated(message):
    def decorator(obj):
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{obj.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2,  # Points to the caller
            )
            return obj(*args, **kwargs)

        return wrapper

    return decorator


# Methods from the old_shape.py
def translate_by_theta_phi(
    shape,
    theta,
    phi,
    D,
    delta=0,
    R_s=0,
    unit_cell=None,
    consider_thickness=False,
):
    """Translate a shape by theta

    Parameters:
    - theta: azimuthal angle. Origin at +y direction
    - shape: shapely shape object
    - delta: membrane thickness
    - R_s: directional shift
    - unit_cell: (cell_w, cell_h) or None
    - consider_thickness: bool, if True, offset by Rm
    """
    if consider_thickness:
        H = D + delta
    else:
        H = D
    Rmi = np.tan(phi) * H
    R = Rmi + R_s
    shift_x = np.sin(theta) * R
    shift_y = -np.cos(theta) * R
    if unit_cell is not None:
        cw, ch = unit_cell
        shift_x = shift_x % cw
        shift_y = shift_y % ch
    new_shape = translate(shape, xoff=shift_x, yoff=shift_y)
    return new_shape


def shape_to_projection(shape, theta, phi, D, unit_cell=None, delta=0, R_s=0):
    """Calculate the intersection between translated shapes"""
    shape1 = translate_by_theta_phi(
        shape,
        theta,
        phi,
        D,
        delta=delta,
        R_s=R_s,
        unit_cell=unit_cell,
        consider_thickness=False,
    )
    shape2 = translate_by_theta_phi(
        shape,
        theta,
        phi,
        D,
        delta=delta,
        R_s=R_s,
        unit_cell=unit_cell,
        consider_thickness=True,
    )
    projection = shape1.intersection(shape2)
    return projection
