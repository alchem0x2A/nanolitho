import math
import warnings
from pathlib import Path

import numpy as np
from matplotlib.projections.polar import PolarAxes
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label, sobel
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely.vectorized import contains

from .geometry import Geometry, Mesh
from .utils import deprecated, nm, um


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

    default_division = 256

    def __init__(
        self,
        geometry,
        pad=0 * nm,
        thickness=100 * nm,
        spacing=2.5 * um,
        h=None,
    ):
        """
        h: default mesh spacing for the stencil,
           if none, h = dimension / default_division
        """
        self.geometry = self._assign_geometry(geometry)
        self.delta = thickness
        self.D = spacing
        self.pad = pad
        # Domain is the min_x, max_x, min_y, max_y
        self.domain = self.geometry.bounds
        # assign h to self
        self.h = self._assign_h(h)

    def _assign_h(self, h=None):
        """Assign the mesh grid h to the geometry"""
        if h is None:
            cw = self.domain[2] - self.domain[0]
            h = cw / self.default_division
        return h

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
    Trajectory is an array of (theta, phi) values on the hemisphere
    """

    def __init__(
        self,
        trajectory,
        phi_first=False,
        radians=True,
        phi_broadening=0.05,
        drift=0 * nm,
        diffusion=5 * nm,
    ):
        """Generate a Physics object from trajectory (theta, phi)

        Parameters:
        - trajectory: a vector of (theta, phi)
                      or (phi, theta) if phi_first is True
        - phi_first: whether phi comes first in the trajectory vector
                     for legacy code support
        - radians: whether the trajectory is provided in radians or degrees
        - phi_broadening: broadening of the phi angle from beam
                          (currently not used)
        - drift: directional drift on the surface (Rs)
        - diffusion: non-directional surface broadening (Rd)
        """
        trajectory = np.atleast_2d(trajectory)
        if not radians:
            trajectory = np.radians(trajectory)
        if phi_first:
            self.phi, self.theta = trajectory[:, 0], trajectory[:, 1]
        else:
            self.phi, self.theta = trajectory[:, 1], trajectory[:, 2]
        self.xi = phi_broadening
        self.drift = drift
        self.diffusion = diffusion

    def generate_filter(self, *arg, **argv):
        raise NotImplementedError(
            ("`generate_fileter` is deprecated! " "Use generate_F instead")
        )

    def generate_F(self, stencil, domain_ratio=1.5):
        """Generate the offset mesh based on the stencil

        The following parameters are determined from the stencil geometry:
        mesh spacing (h)
        membrane-to-substrate spacing (D)
        membrane thickness (delta)

        Parameters:
        - stencil: a Stencil object
        - domain_ratio: factor to extend the mesh domain
                        relative to max radius.
        """

        # Create the "central" trajectory points
        h, D, delta = stencil.h, stencil.D, stencil.delta
        phi, theta = self.phi, self.theta
        R_center = (D + delta) * np.tan(phi) + self.drift
        x_center, y_center = (R_center * np.cos(theta), R_center * np.sin(theta))

        # The max radius of the filter pattern
        R_max = np.max(np.abs(R_center))
        xy_lim = domain_ratio * R_max
        x_range = np.arange(-xy_lim, xy_lim, h)
        y_range = np.arange(-xy_lim, xy_lim, h)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        F_mesh, _, _ = np.histogram2d(
            x_center,
            y_center,
            bins=[xmesh.shape[0], ymesh.shape[1]],
            range=[[x_range.min(), x_range.max()], [y_range.min(), y_range.max()]],
        )
        F_mesh = gaussian_filter(F_mesh, sigma=int(self.diffusion / h))
        return Mesh(F_mesh, x_range, y_range)

    def draw(self, ax=None, stencil=None, grid="polar", cmap="gray", phi_max=None):
        """Draw the offset trajectory pattern on polar or cartesian grids

        Parameters:
        - ax: matplotlib figure axis
        - stencil: a Stencil object for plotting on the cartesian grid
        - grid: either 'polar' or 'cartesian'
        - cmap: color map for drawing on cartesian grid
        - phi_max: maximum phi value used for plotting on polar grids
                   (in radians)
        """
        grid = grid.lower()
        assert grid in (
            "polar",
            "cartesian",
        ), "`grid` can either be 'polar' or 'cartesian'"
        if ax is None:
            import matplotlib.pyplot as plt

            if grid == "polar":
                fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            else:
                fig, ax = plt.subplots()

        if grid == "polar":
            if not isinstance(ax, PolarAxes):
                raise ValueError(
                    (
                        "The provided axis must have a "
                        "polar projection for 'polar' grid."
                    )
                )

            # Plot trajectory in polar coordinates
            if phi_max is None:
                phi_max = min(np.pi / 2, 1.5 * np.max(self.phi))
            else:
                phi_max = phi_max
            ax.scatter(self.theta, np.sin(self.phi), marker="o", label="Trajectory")

            ax.set_theta_zero_location("S")
            ax.set_theta_direction(1)
            rticks_raw = np.linspace(0, phi_max, 5)
            rticks_label_deg = [f"{v:.2f}°" for v in np.degrees(np.arcsin(rticks_raw))]
            ax.set_rmax(np.sin(phi_max))
            ax.set_rticks(rticks_raw, labels=rticks_label_deg)
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\phi$")

        elif grid == "cartesian":
            # Cartesian grid: Generate and plot the filter
            if stencil is None:
                raise ValueError(
                    ("No stencil provided to generate " "a cartesian projection!")
                )
            mesh = self.generate_F(stencil)
            ax.imshow(
                mesh.mask,
                extent=mesh.extent / um,
                origin="lower",
                cmap=cmap,
            )
            ax.set_xlabel("X (μm)")
            ax.set_ylabel("Y (μm)")
        return ax


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
