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
