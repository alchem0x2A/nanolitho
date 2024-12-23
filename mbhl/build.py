"""A collection of functions that build common
Geometry and trajectory patterns for simulations
"""

from .geometry import Circle, Geometry, Rectangle
from .utils import _sanitize_rL_input, sqrt2, sqrt3


# Collections of periodic lattices
def honeycomb_hole_lattice(
    r=None, L=None, diameter=None, spacing=None, orientation="vertical"
):
    """
    Make a repeatable honeycomb hole pattern by either a set of:
    - r, L: radius of holes and center-to-center distance, or
    - diameter, spacing: diameter of holes and edge-to-edge spacing.

    Parameters:
    - r: Radius of the holes (alternative to diameter, spacing).
    - L: Center-to-center distance of holes (alternative to diameter, spacing).
    - diameter: Diameter of the holes (alternative to r, L).
    - spacing: Edge-to-edge spacing of the holes (alternative to r, L).
    - orientation: "vertical" or "horizontal".
      horizontal
        * *
      *     *
        * *
      vertical
        *
      *   *
      *   *
        *

    Returns:
    - A periodic `Geometry` object with the honeycomb hole pattern.
    """

    r, L = _sanitize_rL_input(r, L, diameter, spacing)
    orientation = orientation.lower()
    assert orientation in (
        "vertical",
        "horizontal",
    ), "Orientation must be either 'vertical' or 'horizontal'"
    if orientation == "vertical":
        cell = (L * sqrt3, L * 3)
        patches = [
            Circle(0, 0, r),
            Circle(L * sqrt3 / 2, L / 2, r),
            Circle(L * sqrt3 / 2, L * 3 / 2, r),
            Circle(L * sqrt3, L * 2, r),
        ]
    else:
        cell = (L * 3, L * sqrt3)
        patches = [
            Circle(0, 0, r),
            Circle(L / 2, L / 2 * sqrt3, r),
            Circle(L * 3 / 2, L / 2 * sqrt3, r),
            Circle(L * 2, L * sqrt3, r),
        ]

    return Geometry(patches=patches, cell=cell, pbc=(True, True))


def hexagonal_hole_lattice(
    r=None, L=None, diameter=None, spacing=None, orientation="vertical"
):
    """
    Make a repeatable hexagonal (triangle) hole pattern by either a set of:
    - r, L: radius of holes and center-to-center distance, or
    - diameter, spacing: diameter of holes and edge-to-edge spacing.
    Cell looks like
    horizontal
     * *
    * * *
     * *
    vertical
      *
    *   *
      *
    *   *
      *

    Parameters:
    - r: Radius of the holes (alternative to diameter, spacing).
    - L: Center-to-center distance of holes (alternative to diameter, spacing).
    - diameter: Diameter of the holes (alternative to r, L).
    - spacing: Edge-to-edge spacing of the holes (alternative to r, L).
    - orientation: "vertical" or "horizontal".

    Returns:
    - A periodic `Geometry` object with the honeycomb hole pattern.

    """

    r, L = _sanitize_rL_input(r, L, diameter, spacing)
    orientation = orientation.lower()
    assert orientation in (
        "vertical",
        "horizontal",
    ), "Orientation must be either 'vertical' or 'horizontal'"
    if orientation == "vertical":
        cell = (L * sqrt3, L * 3)
        patches = [
            Circle(0, 0, r),
            Circle(L * sqrt3 / 2, L / 2, r),
            Circle(L * sqrt3 / 2, L * 3 / 2, r),
            Circle(L * sqrt3 / 2, L * 5 / 2, r),
            Circle(L * sqrt3, L, r),
            Circle(L * sqrt3, L * 2, r),
        ]
    else:
        cell = (L * 3, L * sqrt3)
        patches = [
            Circle(0, 0, r),
            Circle(L / 2, L / 2 * sqrt3, r),
            Circle(L * 3 / 2, L / 2 * sqrt3, r),
            Circle(L * 5 / 2, L * sqrt3 / 2, r),
            Circle(L, L * sqrt3, r),
            Circle(L * 2, L * sqrt3, r),
        ]
    return Geometry(patches=patches, cell=cell, pbc=(True, True))


def square_hole_lattice(r=None, L=None, diameter=None, spacing=None):
    """
    Create a repeatable square hole pattern.

    Parameters:
    - r: Radius of the holes (alternative to diameter, spacing).
    - L: Center-to-center distance of holes (alternative to diameter, spacing).
    - diameter: Diameter of the holes (alternative to r, L).
    - spacing: Edge-to-edge spacing of the holes (alternative to r, L).

    Returns:
    - A periodic `Geometry` object with the square hole pattern.

    Unit cell looks like
    *  *
    *  *
    """
    r, L = _sanitize_rL_input(r, L, diameter, spacing)
    cell = (L, L)

    patches = [
        Circle(0, 0, r),
    ]
    return Geometry(patches=patches, cell=cell, pbc=(True, True))


def diamond_hole_lattice(r=None, L=None, diameter=None, spacing=None):
    """
    Create a repeatable hole pattern in "diamond" arrangement

    Parameters:
    - r: Radius of the holes (alternative to diameter, spacing).
    - L: Center-to-center distance of holes (alternative to diameter, spacing).
    - diameter: Diameter of the holes (alternative to r, L).
    - spacing: Edge-to-edge spacing of the holes (alternative to r, L).

    Returns:
    - A periodic `Geometry` object with the square hole pattern.

    Unit cell looks like
      *
    *   *
      *
    """
    r, L = _sanitize_rL_input(r, L, diameter, spacing)
    cell = (L * sqrt2, L * sqrt2)

    patches = [
        Circle(0, 0, r),
        Circle(L * sqrt2 / 2, L * sqrt2 / 2, r),
    ]
    return Geometry(patches=patches, cell=cell, pbc=(True, True))


def line_lattice(s1, s2, orientation="horizontal", length_ratio_nonperiodic=1.0):
    """Create a line (slit) array from slit width
    s1 and edge-to-edeg spacing s2

    Parameters:
    - s1: slit width
    - s2: edge-to-edge distance between slits
    - orientation: whether the slits are horizontal (x-)
                   or vertically (y-) aligned
    - length_ratio_nonperiodic: length of the rectangle representing
                                the slit in the "non-periodic" direction
                                the length is (s1+s2) * ratio

    Returns:
    - A periodic `Geometry` object representing the slit lattice
    """
    W = s1 + s2
    H = W * length_ratio_nonperiodic
    orientation = orientation.lower()
    assert orientation in (
        "vertical",
        "horizontal",
    ), "Orientation must be either 'vertical' or 'horizontal'"
    if orientation == "vertical":
        patches = [Rectangle(W / 2 - s1 / 2, 0, s1, H)]
        cell = (W, H)
    else:
        patches = [Rectangle(0, W / 2 - s1 / 2, H, s1)]
        cell = (H, W)
    return Geometry(patches=patches, cell=cell, pbc=(True, True))
