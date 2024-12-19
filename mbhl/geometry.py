import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.vectorized import contains

from .utils import nm, sqrt2, sqrt3, um

# Definition of geometries


def Circle(x, y, r):
    """Circle defined by
    x, y: center
    r: radius
    """
    return Point(x, y).buffer(r)


def Rectangle(x, y, w, h):
    """Rectangle defined by
    x, y: lower edge
    w, h: width and height
    """
    rectangle = Polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
    return rectangle


def Square(x, y, w):
    """Square defined by w=h for a Rectangle"""
    return Rectangle(x, y, w, w)


class Geometry:
    """
    A 2D geometry class analogous to ASE's Atoms for defining unit cells,
    patches, and periodic boundary conditions.
    """

    def __init__(self, patches, cell=None, pbc=(True, True)):
        """
        Initialize the 2D geometry.

        Parameters:
        - patches: List of Shapely shapes defining the geometry.
        - cell: Tuple (width, height) of the unit cell.
        - pbc: Tuple (bool, bool), periodic boundary conditions along x and y.
        """
        self.patches = patches
        self.cell = cell
        self.pbc = pbc

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, value):
        if value is None:
            value = (0.0, 0.0)
        value = np.array(value, dtype=float)
        if value.shape != (2,):
            raise ValueError("Cell must be a 2D vector.")
        self._cell = value

    @property
    def pbc(self):
        return self._pbc

    @pbc.setter
    def pbc(self, value):
        if isinstance(value, (int, bool)):
            value = (value, value)
        value = np.array(value, dtype=bool)
        if value.shape != (2,):
            raise ValueError("PBC must be a tuple of two booleans.")
        if value[0] != value[1]:
            raise NotImplementedError("Mixed boundary conditions are not supported.")
        self._pbc = value

    @property
    def is_periodic(self):
        """
        Check if the geometry is periodic in both directions.
        """
        return all(self.pbc)

    def bounds(self):
        """
        Calculate the bounding box of the geometry.
        If PBC is enabled, return the unit cell bounds.
        """
        if self.is_periodic:
            return (0, 0, self.cell[0], self.cell[1])
        else:
            union = unary_union(self.patches)
            return union.bounds

    def copy(self):
        """Create a copy of geometry."""
        import copy

        return Geometry(
            patches=copy.deepcopy(self.patches), cell=self.cell, pbc=self.pbc
        )

    def make_tiles(self, repeat=(1, 1)):
        """
        Make repeated tiles by repeating (m, n) times in x- and y-directions.

        Parameters:
        - repeat: Tuple (repeat_x, repeat_y) for replication.
        """
        repeat = np.array(repeat, dtype=int)
        if repeat.shape != (2,) or any(repeat <= 0):
            raise ValueError("Repeat must be a tuple of two positive integers.")

        if not self.is_periodic:
            if any(repeat != (1, 1)):
                raise ValueError("Cannot tile non-periodic geometry.")
            return self.copy()

        replicated_patches = [
            translate(patch, xoff=i * self.cell[0], yoff=j * self.cell[1])
            for i in range(repeat[0])
            for j in range(repeat[1])
            for patch in self.patches
        ]
        new_cell = self.cell * repeat
        return Geometry(patches=replicated_patches, cell=new_cell, pbc=(True, True))

    def __mul__(self, repeat):
        """
        Multiply operator to create a tiled geometry.

        Parameters:
        - repeat: Tuple (m, n) for tiling in x and y directions.
        """
        return self.make_tiles(repeat)

    def __add__(self, other):
        """
        Add operator to combine two Geometry objects.

        Parameters:
        - other: Another Geometry object to merge.
        """
        if not isinstance(other, Geometry):
            raise TypeError("Can only add another Geometry object.")

        if self.is_periodic != other.is_periodic:
            raise ValueError("Cannot combine geometries with different PBCs.")

        if self.is_periodic and not np.all(np.isclose(self.cell, other.cell)):
            raise ValueError("Cannot combine periodic geometries with different cells.")

        combined_patches = self.patches + other.patches
        new_cell = self.cell if self.is_periodic else (0.0, 0.0)
        return Geometry(patches=combined_patches, cell=new_cell, pbc=self.pbc)

    def draw(self, ax=None, repeat=(1, 1), npts=1000, cmap="gray", show_unit_cell=True):
        """
        Visualize the geometry and unit cell in a quick way

        Parameters:
        - ax: Matplotlib axis.
        - repeat: Tuple (repeat_x, repeat_y) for replication.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        tiles = self.make_tiles(repeat=repeat) if self.is_periodic else self
        union = unary_union(tiles.patches)
        minx, miny, maxx, maxy = tiles.bounds()
        x_range = np.linspace(minx, maxx, npts)
        y_range = np.linspace(miny, maxy, npts)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        mask = contains(union, xmesh.ravel(), ymesh.ravel()).reshape(xmesh.shape)
        extent = (minx / um, maxx / um, miny / um, maxy / um)
        ax.imshow(mask, extent=extent, origin="lower", cmap=cmap)
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        if show_unit_cell and all(self.cell > 0):
            from matplotlib.patches import Rectangle

            # Calculate unit cell dimensions in μm
            cell_width, cell_height = self.cell[0] / um, self.cell[1] / um
            for i in range(repeat[0]):
                for j in range(repeat[1]):
                    x_start = i * cell_width + minx / um
                    y_start = j * cell_height + miny / um
                    rect = Rectangle(
                        (x_start, y_start),
                        cell_width,
                        cell_height,
                        edgecolor="red",
                        facecolor="none",
                        linestyle="dotted",
                        linewidth=1,
                    )
                    ax.add_patch(rect)
        return ax


def _sanitize_rL_input(r=None, L=None, diameter=None, spacing=None):
    """Sanitize the r-L or diameter-spacing inputs"""
    if (diameter is not None and spacing is not None) and (r is None and L is None):
        r = diameter / 2
        L = spacing - diameter
    elif (r is not None and L is not None) and (diameter is None and spacing is None):
        pass
    else:
        raise ValueError("Provide either (r, L) or (diameter, spacing), but not both.")
    return r, L


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
