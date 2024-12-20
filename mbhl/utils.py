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
