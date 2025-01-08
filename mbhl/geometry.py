from copy import copy, deepcopy
from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift, ifft2
from scipy.ndimage import label
from shapely.affinity import translate
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.vectorized import contains

from .utils import (
    ensure_ax,
    mm,
    next_power_of_two,
    nm,
    sqrt2,
    sqrt3,
    um,
    unit_properties,
)

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

    def __init__(
        self,
        array,
        x_range,
        y_range,
        is_fourier=False,
        unit="um",
        crop_indices=None,
    ):
        self.array = array
        if len(self.array.shape) != 2:
            raise ValueError("Array must be 2D!")
        self.x_range = x_range
        self.y_range = y_range
        # Use trim_indices to get the centered array
        self.crop_indices = crop_indices
        self.is_fourier = is_fourier
        self.unit = unit

    @property
    def hx(self):
        return self.x_range[1] - self.x_range[0]

    @property
    def hy(self):
        return self.y_range[1] - self.y_range[0]

    @property
    def Nx(self):
        return len(self.x_range)

    @property
    def Ny(self):
        return len(self.y_range)

    @property
    def shape(self):
        return self.array.shape

    def __copy__(self):
        """Copy instance"""
        return deepcopy(self)

    @classmethod
    def from_analytical_function(
        cls, f, x_range, y_range, func_args=(), **kwargs
    ):
        """Create the mesh object from given x_range and y_range
        using an analytical function f(x, y, *func_args)
        func_args: tuple of arguments for function f
        kwargs: mapping arguments for the Mesh object
        """
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        array = f(xmesh, ymesh, *func_args)
        return Mesh(array, x_range, y_range, **kwargs)

    @property
    def cropped_mesh(self):
        """Construct a cropped mesh from crop_indices"""
        if self._crop_indices is None:
            return self
        else:
            sy, sx = self.crop_indices
            cropped_array = self.array[sy, sx]
            cropped_x_range = self.x_range[sx]
            cropped_y_range = self.y_range[sy]
            # TODO: Should we remember the original
            cropped_x_range = cropped_x_range - cropped_x_range[0]
            cropped_y_range = cropped_y_range - cropped_y_range[0]

            return Mesh(
                array=cropped_array,
                x_range=cropped_x_range,
                y_range=cropped_y_range,
                is_fourier=self.is_fourier,
                crop_indices=None,
                unit=self.unit,
            )

    @property
    def extent(self):
        """Combine range to extent for matplotlib's imshow

        If you want to use crop indices, generate self.cropped_mesh
        first before plotting
        """
        return np.array(
            [
                self.x_range[0],
                self.x_range[-1],
                self.y_range[0],
                self.y_range[-1],
            ]
        )

    def to_label(self):
        """Create a label matrix for binary mesh

        Returns:
        - New mesh with labeled mask
        - number of features
        """
        if not np.array_equal(self.array, self.array.astype(bool)):
            raise ValueError("Mesh array must be binary to generate labels")

        # Perform connected component labeling
        labeled_array, num_features = label(self.array)
        return Mesh(labeled_array, self.x_range, self.y_range), num_features

    # TODO: this method will break old API
    def tiled_mesh(self, extra_x=1, extra_y=None):
        """
        Tile the array by (2 * extra_x + 1, 2 * extra_y + 1) times
        in x- and y-directions, and return the tiled array with
        indices to trim back.

        Parameters:
        - extra_x: Number of extra tiles to add in the x-direction on each side.
        - extra_y: Number of extra tiles to add in the y-direction on each side. If None, extra_y = extra_x.

        Returns:
        - tiled_array: A periodically tiled array.
        - trim_indices: Slice indices to recover the original
                        array from the tiled version.
        """
        if extra_y is None:
            extra_y = extra_x

        # tiled_array = np.tile(self.array, (2 * extra_y + 1, 2 * extra_x + 1))
        # new_x_range
        tiled_mesh = self * (2 * extra_y + 1, 2 * extra_x + 1)
        rows, cols = tiled_mesh.array.shape
        trim_indices = (
            slice(rows * extra_y, rows * (extra_y + 1)),
            slice(cols * extra_x, cols * (extra_x + 1)),
        )
        tiled_mesh.crop_indices = trim_indices
        return tiled_mesh

    # TODO: notify user that tiled_array only returns the array!
    def tiled_array(self, extra_x=1, extra_y=None):
        """
        Tile the array by (2 * extra_x + 1, 2 * extra_y + 1) times
        in x- and y-directions, and return the tiled array with
        indices to trim back.

        Parameters:
        - extra_x: Number of extra tiles to add in the x-direction on each side.
        - extra_y: Number of extra tiles to add in the y-direction on each side. If None, extra_y = extra_x.

        Returns:
        - tiled_array: A periodically tiled array.
        - trim_indices: Slice indices to recover the original
                        array from the tiled version.
        """
        if extra_y is None:
            extra_y = extra_x

        tiled_array = np.tile(self.array, (2 * extra_y + 1, 2 * extra_x + 1))
        rows, cols = self.array.shape
        trim_indices = (
            slice(rows * extra_y, rows * (extra_y + 1)),
            slice(cols * extra_x, cols * (extra_x + 1)),
        )
        return tiled_array, trim_indices

    def padded_array(self, pad_x=1, pad_y=None, value=0):
        """
        Create a padded array with padding values and
        return the padded array with indices to trim back.

        Parameters:
        - pad_x: Padding size in the x-direction.
        - pad_y: Padding size in the y-direction. If None, pad_y = pad_x.

        Returns:
        - padded_array: A padded array with constant values
        - trim_indices: Slice indices to recover the original array
                        from the padded version, like in tiled_array
        """
        if pad_y is None:
            pad_y = pad_x

        padded_array = np.pad(
            self.array,
            ((pad_y, pad_y), (pad_x, pad_x)),
            mode="constant",
            constant_values=value,
        )
        rows, cols = self.array.shape
        trim_indices = (slice(pad_y, pad_y + rows), slice(pad_x, pad_x + cols))
        return padded_array, trim_indices

    def __mul__(self, repeat):
        """Allow making a tile of the mesh

        The new mesh uses the numpy.tile method to create a mesh
        repeat: repeat in x- and y-directions
                (the array shape will be swapped)
        """
        if isinstance(repeat, (float, int)):
            repeat = (int(repeat), int(repeat))

        if not isinstance(repeat, (tuple, list)) or len(repeat) != 2:
            raise ValueError(
                "Repeat must be a tuple or list of two integers (nx, ny)."
            )

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

        tiled_array = np.tile(self.array, (ny, nx))

        # TODO: Unset the crop indices since they won't be correct
        return Mesh(
            tiled_array,
            new_x_range,
            new_y_range,
            unit=self.unit,
            is_fourier=self.is_fourier,
            crop_indices=None,
        )

    def save(
        self, filepath: str, compressed: bool = True, overwrite: bool = False
    ) -> None:
        """
        Save this Mesh to a .npz file.

        Parameters
        ----------
        filepath : str
            Path to the output .npz file.
        compressed : bool, default=True
            If True, use np.savez_compressed (smaller files),
            otherwise use np.savez (faster, bigger files).
        """
        filepath = Path(filepath)
        if filepath.is_file() and (overwrite is not True):
            raise RuntimeError(
                (
                    f"File {filepath.as_posix()} exists! "
                    "Please set `overwrite=True` if you want to overwrite."
                )
            )
        if compressed:
            np.savez_compressed(
                filepath,
                array=self.array,
                x_range=self.x_range,
                y_range=self.y_range,
            )
        else:
            np.savez(
                filepath,
                array=self.array,
                x_range=self.x_range,
                y_range=self.y_range,
            )

    # def fft(self, )
    @classmethod
    def load(cls, filepath: str) -> "Mesh":
        """
        Load a Mesh from a .npz file previously saved by .save(...).

        Parameters
        ----------
        filepath : str
            Path to the input .npz file.

        Returns
        -------
        mesh : Mesh
            A new Mesh instance with array, x_range, y_range.
        """
        filepath = Path(filepath)
        data = np.load(filepath)
        array = data["array"]
        x_range = data["x_range"]
        y_range = data["y_range"]

        # Create a new instance
        return cls(array, x_range, y_range)

    def calculate_fft(self, pad_to_size=None):
        """Return a FFT mesh from the current (real-space) mesh
        if self.is_fourier is False

        Parameters:
        pad_to_size: if None, use the closest 2^n size
                     otherwise the closet 2^n size to pad_to_size
                     pad_to_size can be useful such as
                     the dimension from a larger matrix and align
        """
        if self.is_fourier:
            raise ValueError(
                "Cannot run fft since Current mesh is already in Fourier space!"
            )
        # TODO: make the fft pad to 2^n
        if pad_to_size is not None:
            fft_size = next_power_of_two(pad_to_size)
        else:
            fft_size = next_power_of_two(
                max(len(self.x_range), len(self.y_range))
            )
        print(fft_size)

        padded_array = np.zeros((fft_size, fft_size), dtype=np.float32)
        nx, ny = len(self.x_range), len(self.y_range)
        # TODO: should we make half padding?
        padded_array[:ny, :nx] = self.array
        fft_array = fftshift(fft2(padded_array))
        n_samples = fft_size
        sample_spacing = self.x_range[1] - self.x_range[0]
        # Use radian frequencies. The fft frequency is always in radians / um
        fft_freq = fftshift(fftfreq(n_samples, sample_spacing)) * np.pi * 2
        return Mesh(
            array=fft_array, x_range=fft_freq, y_range=fft_freq, is_fourier=True
        )

    def auto_contrast_fft(self, pmin=99.0, pmax=99.9):
        """Generate an auto-contrasted array for FFT display"""
        if not self.is_fourier:
            raise NotImplementedError("Only works on FFT grid!")
        array_draw = np.log1p(np.abs(self.array))
        lo = np.percentile(array_draw, pmin)
        hi = np.percentile(array_draw, pmax)
        return np.clip((array_draw - lo) / (hi - lo + 1e-9), 0, 1)

    def draw_fft(
        self,
        ax=None,
        unit="rad/um",
        dimension_ratio=None,
        domain=None,
        cmap="gray",
        display_window=(100 - 1.0e-1, 100 - 1.0e-3),
        **argv,
    ):
        """Draw the 2D mesh in Fourier space.
        Requires self.is_fourier=True
        """
        if not self.is_fourier:
            raise NotImplementedError(
                "Must be a Fourier-space mesh to use draw_fft!"
            )
        ax = ensure_ax(ax)

        # TODO: implement unit

        # TODO: implement B/W balancing
        if display_window is None:
            array_draw = np.log1p(np.abs(self.array))
        else:
            array_draw = self.auto_contrast_fft(*display_window)
        cm = ax.imshow(
            array_draw, extent=self.extent, origin="lower", cmap=cmap, **argv
        )
        # TODO: assert if domain is not symmetric
        if domain is not None:
            xy_lims = np.array(domain)
            ax.set_xlim(xy_lims[0], xy_lims[1])
            ax.set_ylim(xy_lims[2], xy_lims[3])
        ax.set_xlabel(f"$f_x$ ({unit})")
        ax.set_ylabel(f"$f_y$ ({unit})")
        return ax, cm

    def draw(
        self,
        ax=None,
        repeat=(1, 1),
        unit="um",
        dimension_ratio=None,
        domain=None,
        cmap="gray",
        **argv,
    ):
        """Create a 2D plot of the mesh array,
        with possibility of repeating

        Parameters:
        - ax: matplotlib Axes instance.
              If not provided, create one on-the-fly
        - repeat: repeat in x- and y-directions
        - unit: unit to plot on axis, can be 'nm', 'um' or 'mm'
        - dimension_ratio: if not None, explicitly set the ratio for axis
        - domain: None of a 4-tuple (x_min, x_max, y_min, y_max)
        - cmap: color map name
        - argv: extra parameters to provide to ax.imshow

        Returns:
        - ax: Axes with plot
        - cm: 2D plot
        """
        if self.is_fourier:
            raise NotImplementedError(
                (
                    "You have a mesh in Fourier space. "
                    "Please use the draw_fourier method instead"
                )
            )
        ax = ensure_ax(ax)
        if dimension_ratio is None:
            unit = unit.lower()
            assert (
                unit in unit_properties.keys()
            ), f"unit name {unit} is unknown!"
            axis_ratio = unit_properties[unit]["ratio"]
            unit_display_name = unit_properties[unit]["display_name"]
        else:
            axis_ratio = float(dimension_ratio)
            unit_display_name = "a.u."

        mesh_draw = self * repeat
        cm = ax.imshow(
            mesh_draw.array,
            extent=mesh_draw.extent / axis_ratio,
            origin="lower",
            cmap=cmap,
            **argv,
        )
        if domain is not None:
            xy_lims = np.array(domain) / axis_ratio
            ax.set_xlim(xy_lims[0], xy_lims[1])
            ax.set_ylim(xy_lims[2], xy_lims[3])
        ax.set_xlabel(f"X ({unit_display_name})")
        ax.set_ylabel(f"Y ({unit_display_name})")
        return ax, cm


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
            raise NotImplementedError(
                "Mixed boundary conditions are not supported."
            )
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
        union = unary_union(
            [patch.buffer(-self.shrink) for patch in self.patches]
        )
        return union

    def translate(self, displacement):
        """
        Translate all patches in the geometry by the specified displacement.

        Parameters:
        - displacement: Tuple (dx, dy) specifying
          the translation in the x and y directions.

        Behavior:
        - For periodic geometry the displacement is applied modulo
          the unit cell dimensions,
        - Otherwise the patches are directly translated

        Returns:
        - None, the patches are translated in-place
        """
        if (
            not isinstance(displacement, (tuple, list))
            or len(displacement) != 2
        ):
            raise ValueError("Displacement must be a tuple (dx, dy).")

        dx, dy = displacement

        if self.is_periodic:
            # Apply displacement modulo the unit cell dimensions
            cell_w, cell_h = self.cell
            dx %= cell_w
            dy %= cell_h

            # Translate patches so that they move no more than
            # half of cell dimension
            # Note: this method does not always ensure the center
            # lied inside the cell
            translated_patches = [
                translate(
                    patch,
                    xoff=-cell_w + dx if dx > cell_w / 2 else dx,
                    yoff=-cell_h + dy if dy > cell_h / 2 else dy,
                )
                for patch in self.patches
            ]
        else:
            # For non-periodic geometries, apply the displacement directly
            translated_patches = [
                translate(patch, xoff=dx, yoff=dy) for patch in self.patches
            ]
            self.cell = None

        self.patches = translated_patches
        return

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
        bounds = (
            union.bounds() if not tiles.is_periodic else (0, 0, *tiles.cell)
        )
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
        return Geometry(
            patches=replicated_patches, cell=new_cell, pbc=(True, True)
        )

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
            raise ValueError(
                "Cannot combine periodic geometries with different cells."
            )

        combined_patches = self.patches + other.patches
        new_cell = self.cell if self.is_periodic else (0.0, 0.0)
        return Geometry(patches=combined_patches, cell=new_cell, pbc=self.pbc)

    def draw(
        self,
        ax=None,
        repeat=(1, 1),
        divisions=256,
        cmap="gray",
        unit="um",
        show_unit_cell=True,
        unit_cell_color="orange",
    ):
        """
        Visualize the rasterized geometry quickly

        Parameters:
        - ax: Matplotlib axis.
        - repeat: Tuple (repeat_x, repeat_y) for replication.
        - divisions: (nx, ny) for dividing the unit cell
        - cmap: color map for the map
        - show_unit_cell: whether to draw unit cell boundaries
        - unit_cell_color: cell boundary line color
        """

        mesh = self.generate_mesh(divisions=divisions)
        minx, miny = mesh.extent[0], mesh.extent[2]
        ax, cm = mesh.draw(ax, unit=unit, repeat=repeat, cmap=cmap)
        unit_ratio = unit_properties[unit]["ratio"]
        if show_unit_cell and all(self.cell > 0):
            from matplotlib.patches import Rectangle

            # Calculate unit cell dimensions in μm
            cell_width, cell_height = (
                self.cell[0] / unit_ratio,
                self.cell[1] / unit_ratio,
            )
            for i in range(repeat[0]):
                for j in range(repeat[1]):
                    x_start = i * cell_width + minx / um
                    y_start = j * cell_height + miny / um
                    rect = Rectangle(
                        (x_start, y_start),
                        cell_width,
                        cell_height,
                        edgecolor=unit_cell_color,
                        facecolor="none",
                        linestyle="--",
                        linewidth=1,
                    )
                    ax.add_patch(rect)
        return ax, cm
