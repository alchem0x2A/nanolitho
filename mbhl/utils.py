import functools
import math
import warnings

import numpy as np

# Definition of units where um = 1
# All dimensions are in um
mm = 1000
um = 1.0
nm = 0.001
sqrt3 = math.sqrt(3)
sqrt2 = math.sqrt(2)


def _sanitize_rL_input(r=None, L=None, diameter=None, spacing=None):
    """Sanitize the r-L or diameter-spacing inputs for Geometry"""
    if (diameter is not None and spacing is not None) and (r is None and L is None):
        r = diameter / 2
        L = spacing - diameter
    elif (r is not None and L is not None) and (diameter is None and spacing is None):
        pass
    else:
        raise ValueError("Provide either (r, L) or (diameter, spacing), but not both.")
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


def n_beam_trajectory(phi, theta_0=0, n_pts=360):
    """Create a trajectory of n-beam interference with
    constant zenith angle phi, and evenly-distributed azimuthal
    angles, starting from theta_0 with step of pi/n_pts. The default
    setting represents a circular trajectory along latitude pi-phi

    Parameters:
    - phi: zenith angle
    - theta_0: initial azimuthal angle
    - n_pts: points along the trajectory

    Returns:
    - trajectory: np.array of [(theta, phi), ] in radians
    """
    theta = np.linspace(theta_0, theta_0 + 2 * np.pi, n_pts, endpoint=False)

    # Create the trajectory with constant phi
    trajectory = np.array([(theta_i, phi) for theta_i in theta])
    return trajectory
