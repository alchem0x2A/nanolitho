import functools
import math
import warnings

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


def ensure_ax(ax=None):
    """Ensure one Axes object exists"""
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
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
