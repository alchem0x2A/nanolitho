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


class Mesh:
    """
    A 2D mesh grid for a geometry.

    Attributes:
    - mask: Numpy array for the mesh (2D)
    - x_range: (x_min, x_max) of the mesh
    - y_range: (y_min, y_max) of the mesh
    """

    def __init__(self, mask, x_range, y_range):
        self.mask = mask
        self.x_range = x_range
        self.y_range = y_range

    @property
    def extent(self):
        """Combine range to extent for matplotlib's imshow"""
        return (self.x_range[0], self.x_range[-1], self.y_range[0], self.y_range[-1])

    def __mul__(self, repeat):
        """Allow making a tile of the mesh

        The new mesh uses the numpy.tile method to create a mesh
        repeat: repeat in x- and y-directions
                (the array shape will be swapped)
        """
        if not isinstance(repeat, (tuple, list)) or len(repeat) != 2:
            raise ValueError("Repeat must be a tuple or list of two integers (nx, ny).")

        nx, ny = repeat
        # Extend the ranges
        x_spacing = self.x_range[1] - self.x_range[0]
        y_spacing = self.y_range[1] - self.y_range[0]

        new_x_range = np.arange(
            self.x_range[0],
            self.x_range[0] + nx * len(self.x_range) * x_spacing,
            x_spacing,
        )
        new_y_range = np.arange(
            self.y_range[0],
            self.y_range[0] + ny * len(self.y_range) * y_spacing,
            y_spacing,
        )

        # Tiled mask have shape ny X nx

        tiled_mask = np.tile(self.mask, (ny, nx))
        return Mesh(tiled_mask, new_x_range, new_y_range)


class Geometry:
    """
    A 2D geometry class analogous to ASE's Atoms for defining unit cells,
    patches, and periodic boundary conditions.
    """

    def __init__(self, patches, cell=None, pbc=(True, True), shrink=0.0):
        """
        Initialize the 2D geometry.

        Parameters:
        - patches: List of Shapely shapes defining the geometry.
        - cell: Tuple (width, height) of the unit cell.
        - pbc: Tuple (bool, bool), periodic boundary conditions along x and y.
        - shrink: boundary shink of the objects
        """
        self.patches = patches
        self.cell = cell
        self.pbc = pbc
        self.shrink = shrink

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

    @property
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

    @property
    def union(self):
        """The shapely union object containing all patches in
        the geometry
        """
        union = unary_union([patch.buffer(-self.shrink) for patch in self.patches])
        return union

    def generate_mesh(self, h=None, divisions=None, shrink=0):
        """
        Generate a mesh grid for the geometry.

        See docstring above for details.
        """
        if (h is None and divisions is None) or (
            h is not None and divisions is not None
        ):
            raise ValueError("Specify exactly one of `h` or `divisions`.")

        # Make repeated tiles for periodic system
        rep = 3 if self.is_periodic else 1
        tiles = self.make_tiles(repeat=(rep, rep))
        union = tiles.union
        # bounds for the tiled union
        bounds = union.bounds() if not tiles.is_periodic else (0, 0, *tiles.cell)
        # Width for the tile, considering the repetition
        # bounds: (xmin, ymin, xmax, ymax)
        tile_w, tile_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        # Dimension for the final mesh
        mesh_w, mesh_h = tile_w / rep, tile_h / rep

        # Nx, Ny are number of grids in the final mesh
        if divisions is not None:
            if isinstance(divisions, (int, float)):
                divisions = (int(divisions), int(divisions))
            Nx, Ny = divisions
        else:
            Nx, Ny = int(mesh_w // h), int(mesh_h // h)
        # We cannot guarantee that hx, hy are exactly h
        # import pdb; pdb.set_trace()
        hx, hy = mesh_w / Nx, mesh_h / Ny

        tile_x_range = np.linspace(bounds[0], bounds[2], Nx * rep)
        tile_y_range = np.linspace(bounds[1], bounds[3], Ny * rep)
        mesh_x_range = np.arange(bounds[0], bounds[0] + mesh_w, hx)
        mesh_y_range = np.arange(bounds[1], bounds[1] + mesh_h, hy)
        print(tile_x_range.shape, mesh_x_range.shape)
        # xmesh and ymesh are of shape (Ny, Nx)
        tile_xmesh, tile_ymesh = np.meshgrid(tile_x_range, tile_y_range)
        mask = contains(union, tile_xmesh, tile_ymesh)
        if self.is_periodic:
            # Make sure Nx and Ny are swapped
            mask = mask[Ny : (rep - 1) * Ny, Nx : (rep - 1) * Nx]
        return Mesh(mask, mesh_x_range, mesh_y_range)

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

    def draw(
        self, ax=None, repeat=(1, 1), divisions=1000, cmap="gray", show_unit_cell=True
    ):
        """
        Visualize the geometry and unit cell in a quick way

        Parameters:
        - ax: Matplotlib axis.
        - repeat: Tuple (repeat_x, repeat_y) for replication.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        # We could actually allow any repeat
        mesh = self.generate_mesh(divisions=divisions) * repeat
        minx, miny = mesh.extent[0], mesh.extent[2]
        ax.imshow(mesh.mask, extent=mesh.extent, origin="lower", cmap=cmap)
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