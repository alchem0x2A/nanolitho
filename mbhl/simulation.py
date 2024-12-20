import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label, sobel
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely.vectorized import contains

from .geometry import Geometry
from .utils import Circle, Rectangle, Square, deprecated, nm, um


class Stencil:
    """
    A Stencil object for the stencil membrane in MBHL.

    The stencil can is constructed using one or more
    Geometry object.

    Example:
        # To create a square array of circles with a radius of 400 nm,
        # a center-to-center spacing of 600 nm, and 100 nm padding:
        geometry = Geometry([Circle(0, 0, 250 * nm)], cell=(600 * nm, 600 * nm), pbc=(True, True))
        stencil = Stencil(geometry, pad=100 * nm, thickness=200 * nm, spacing=2.5 * um)
    """

    def __init__(
        self,
        geometry,
        pad=0 * nm,
        thickness=100 * nm,
        spacing=2.5 * um,
    ):
        self.geometry = self._assign_geometry(geometry)
        self.delta = thickness
        self.D = spacing
        self.pad = pad
        # Domain is the min_x, max_x, min_y, max_y
        self.domain = None

    def _assign_geometry(self, geometry):
        """
        Assign the input geometry to the stencil.

        - If the input is a `Geometry` object, it is directly used.
        - If the input is a list of `Geometry` objects,
          they are combined using the `+` operator.

        Parameters:
        - geometry: A `Geometry` object or a list of `Geometry` objects.

        Returns:
        - A combined `Geometry` object.
        """
        if isinstance(geometry, Geometry):
            return geometry.copy()
        elif isinstance(geometry, (tuple, list)):
            if not all(isinstance(g, Geometry) for g in geometry):
                raise TypeError(
                    "All elements in the list must be instances of `Geometry`."
                )
            combined_geometry = geometry[0].copy()
            for g in geometry[1:]:
                combined_geometry += g
            return combined_geometry
        else:
            raise TypeError(
                (
                    "Input must be a `Geometry` object "
                    "or a list of `Geometry` objects."
                )
            )

    @property
    def patches(self):
        return self.geometry.patches

    @property
    def cell(self):
        return self.geometry.patches

    @property
    def is_periodic(self):
        return self.geometry.is_periodic

    def draw(self, ax, h=10 * nm, cmap="gray", vmax=None):
        """Draw the mask pattern with existing repeats
        ax is an existing matplotlib axis
        """
        bin_mask, x_range, y_range = self.generate_mesh(h)
        ax.imshow(
            bin_mask,
            extent=(
                x_range[0] / um,
                x_range[-1] / um,
                y_range[0] / um,
                y_range[-1] / um,
            ),
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        return


@deprecated(("Please use `Stencil` as the class name " "instead of `Mask`"))
class Mask(Stencil):
    pass


class Physics:
    """A general class for the physics (filter) behind the MBHL
    Note the thickness (delta) and spacing (H) are directly affecting the filter pattern

    Trajectory is an array of (psi, theta) values on the hemisphere
    """

    def __init__(self, trajectory, psi_broadening=0.05, drift=0 * nm, diffusion=5 * nm):
        self.trajectory = np.atleast_2d(trajectory)
        self.xi = psi_broadening
        self.drift = drift
        self.diffusion = diffusion

    # TODO: obsolete method
    def generate_filter(self, h, H, delta=0 * nm, samples=10000, domain_ratio=1.5):
        return self.generate_F(h, H, delta, samples, domain_ratio)

    def generate_F(self, h, D, delta=0 * nm, samples=10000, domain_ratio=1.5):
        """Generate the offset trajectory F on 2D mesh
        h: mesh spacing
        """

        # Create the "central" trajectory points
        psi, theta = self.trajectory[:, 0], self.trajectory[:, 1]
        R_center = (D + delta) * np.tan(psi) + self.drift
        x_center, y_center = R_center * np.cos(theta), R_center * np.sin(theta)

        # The max radius of the filter pattern
        R_max = np.max(np.abs(R_center))
        xy_lim = domain_ratio * R_max
        x_range = np.arange(-xy_lim, xy_lim, h)
        y_range = np.arange(-xy_lim, xy_lim, h)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        P_mesh, _, _ = np.histogram2d(
            x_center,
            y_center,
            bins=[xmesh.shape[0], ymesh.shape[1]],
            range=[[x_range.min(), x_range.max()], [y_range.min(), y_range.max()]],
        )
        P_mesh = gaussian_filter(P_mesh, sigma=int(self.diffusion / h))
        return P_mesh, x_range, y_range

    def draw(self, ax, h=10 * nm, H=2.5 * um, delta=0 * nm, cmap="gray"):
        """Draw the mask pattern,
        ax is an existing matplotlib axis
        """
        bin_mask, x_range, y_range = self.generate_filter(h, H, delta)
        ax.imshow(
            bin_mask,
            extent=(
                x_range[0] / um,
                x_range[-1] / um,
                y_range[0] / um,
                y_range[-1] / um,
            ),
            cmap=cmap,
        )
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        return


class System:
    def __init__(self, mask, physics):
        # TODO: obsolete mask
        self.mask = mask
        self.stencil = mask
        self.physics = physics
        self.results = None
        self.h = None

    def simulate_old(self, h):
        """
        Legacy method to use full convolution
        """
        # Generate mask and physics matrices
        # shrink = -self.mask.delta * np.tan(np.deg2rad(self.physics.psi))
        self.h = h
        shrink = 0
        input_matrix, x_range, y_range = self.mask.generate_mesh(h, shrink=shrink)
        filter_matrix, _, _ = self.physics.generate_filter(
            h, H=self.mask.H, delta=self.mask.delta
        )

        print(input_matrix.shape)
        # Add zero padding to M
        pad_width = filter_matrix.shape[0] // 2
        input_padded = np.pad(
            input_matrix, pad_width=pad_width, mode="constant", constant_values=0
        )

        # Perform convolution
        result = fftconvolve(input_padded, filter_matrix, mode="same")

        # Crop the result to the original size
        result = result[pad_width:-pad_width, pad_width:-pad_width]
        self.results = (result, x_range, y_range)
        return self.results

    def simulate_unit_cell(self, h):
        """Only simulate the unit cell result"""
        self.h = h
        shrink = 0
        (M_uc, x_range_uc, y_range_uc) = self.stencil._generate_mesh_unit_cell(
            h, shrink=shrink
        )
        F, _, _ = self.physics.generate_F(
            h=self.h, D=self.stencil.D, delta=self.mask.delta
        )

        print(M_uc.shape)
        # Perform convolution
        # TODO: make sure if we need larger tiles
        m, n = M_uc.shape
        tile_m, tile_n = 3, 3
        M_tile = np.tile(M_uc, (tile_m, tile_n))
        conv_results = fftconvolve(M_tile, F, mode="same")
        # TODO: change if tile isn't 3
        results_uc = conv_results[m : 2 * m, n : 2 * n]

        # Crop the result to the original size
        # result = result[pad_width:-pad_width, pad_width:-pad_width]
        self.results_uc = results_uc
        self.range_uc = x_range_uc, y_range_uc
        return

    def simulate(self, h):
        """New method using repeating unit cell"""
        self.simulate_unit_cell(h)
        repeat = self.stencil.repeat
        # TODO: reset format
        self.results = (
            np.tile(self.results_uc, (repeat[1], repeat[0])),
            # TODO: make sure tile is different from xy
            self.range_uc[0] * repeat[0],
            self.range_uc[1] * repeat[1],
        )
        return

    def save_tiff(self, h, fname):
        """Save the normalized height as a tiff file so that the file can be opened by
        softwares like gwyddion
        """
        if self.results is None:
            raise RuntimeError("Please finish simulation first!")
        prob, x_range, y_rang = self.results
        z_image = Image.fromarray(prob)
        z_image_file = Path(fname)
        z_image.save(z_image_file)

    def draw(
        self,
        ax,
        cmap="viridis",
        show_mask=True,
        mask_lw=1.5,
        mask_alpha=0.5,
        dimension_ratio=None,
        xlim=None,
        ylim=None,
        alpha=1.0,
        vmax=None,
    ):
        """Draw the system simulation results as 2D map"""
        if self.results is None:
            raise RuntimeError("Please finish simulation first!")

        prob, x_range, y_range = self.results
        if not dimension_ratio:
            extent = (
                x_range[0] / um,
                x_range[-1] / um,
                y_range[0] / um,
                y_range[-1] / um,
            )
        else:
            extent = (
                x_range[0] / dimension_ratio,
                x_range[-1] / dimension_ratio,
                y_range[0] / dimension_ratio,
                y_range[-1] / dimension_ratio,
            )
        ax.imshow(prob, extent=extent, cmap=cmap, alpha=alpha, vmax=vmax)

        if show_mask:
            mask_bin, x_mask, y_mask = self.mask.generate_mesh(h=self.h / 2)
            edge_x, edge_y = sobel(mask_bin, axis=0), sobel(mask_bin, axis=1)
            edges = np.hypot(edge_x, edge_y)
            edges[edges > 0] = 1
            radius = math.ceil(mask_lw)
            # edges = binary_dilation(edges, structure=np.ones((radius, radius)))
            rgba_edges = np.zeros((edges.shape[0], edges.shape[1], 4))
            rgba_edges[..., :3] = 1  # Set R, G, B to 1 (white)
            rgba_edges[..., 3] = edges * mask_alpha
            ax.imshow(rgba_edges, extent=extent)
        if not dimension_ratio:
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
        else:
            ax.set_xlabel("X (a.u.)")
            ax.set_ylabel("Y (a.u.)")

        if xlim is not None:
            ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)
        return
