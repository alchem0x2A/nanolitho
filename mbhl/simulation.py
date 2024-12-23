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
from .utils import deprecated, ensure_ax, nm, um, unit_properties


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

    def generate_mesh(self):
        """Generate mesh from geometry."""
        return self.geometry.generate_mesh(self.h)

    def calculate_critical_phi(
        self, theta=0, r_search_interval=5, degrees=False
    ):
        """
        Calculate the matrix of critical
        phi at angle theta (counted from the +y axis).

        For each point (xm, xy) on the top level of the stencil,
        first calculate the longest distance Rm that a line
        (xm + Rm * sinθ, ym - Rm*cosθ) remains the same label
        phi_c = arctan(Rm / δ)

        This code uses the search along the line-of-sight path
        along the direction (Rm * sin theta, - Rm * cos theta)
        Handles both periodic and non-periodic geometries.

        Parameters:
        - theta: azimuthal angle in radians where the particle
                 incidents
        - r_search_interval: pixels that the function searches along the
                             incidence line.
                             Increase this value will make computation faster,
                             but more rasterized results
        - degrees: if True, return phi_c values in degrees, otherwise in radians

        Returns:
        - Phic_mesh: mesh object with the same dimension as
                     self.generate_mesh(), while the values are phi_c
                     in radians
        """
        M_origin = self.generate_mesh()
        delta = self.delta  # Stencil thickness
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if self.geometry.is_periodic:
            M, trim_indices = M_origin.tiled_array(extra_x=1, extra_y=1)
        else:
            M, trim_indices = M_origin.padded_array(pad_x=2, pad_y=2)

        dx_real = M_origin.x_range[1] - M_origin.x_range[0]
        dy_real = M_origin.y_range[1] - M_origin.y_range[0]
        padded_rows, padded_cols = M.shape
        origin_rows, origin_cols = M_origin.shape

        # Calculate max radius using bounding box corners
        corners = np.array(
            [
                [0, 0],
                [0, origin_rows * dy_real],
                [origin_cols * dx_real, 0],
                [origin_cols * dx_real, origin_rows * dy_real],
            ]
        )
        distances = np.sqrt(
            (corners[:, 0] * sin_theta) ** 2 + (corners[:, 1] * cos_theta) ** 2
        )
        max_radius_real = np.max(distances)

        # Convert max radius to pixels
        max_radius_pixels = int(max_radius_real / min(dx_real, dy_real))

        # Initialize Rm array for the padded/periodic array
        Rm_array = np.ones_like(M, dtype=float) * -1

        # Iterate over radii
        for R_pixels in range(0, max_radius_pixels + 1, r_search_interval):
            shift_x_pixels = int(R_pixels * sin_theta)
            shift_y_pixels = -int(R_pixels * cos_theta)
            R = np.sqrt(
                (shift_x_pixels * dx_real) ** 2
                + (shift_y_pixels * dy_real) ** 2
            )

            new_x_pixels = np.clip(
                np.arange(padded_cols) + shift_x_pixels, 0, padded_cols - 1
            )
            new_y_pixels = np.clip(
                np.arange(padded_rows) + shift_y_pixels, 0, padded_rows - 1
            )

            on_mask = M[new_y_pixels[:, None], new_x_pixels[None, :]] == 0
            Rm_array[(Rm_array == -1) & on_mask] = R

        Rm_array[Rm_array == -1] = R
        Rm_trimmed = Rm_array[trim_indices]

        phic_array = np.arctan(Rm_trimmed / delta)
        if degrees:
            phic_array = np.degrees(phic_array)

        Phic_mesh = Mesh(
            array=phic_array, x_range=M_origin.x_range, y_range=M_origin.y_range
        )
        return Phic_mesh

    @property
    def patches(self):
        return self.geometry.patches

    @property
    def cell(self):
        return self.geometry.patches

    @property
    def is_periodic(self):
        return self.geometry.is_periodic

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
        """Draw the stencil mask using Geometry.draw

        See Geometry.draw for more details
        """
        ax, cm = self.geometry.draw(
            ax=ax,
            repeat=repeat,
            divisions=divisions,
            cmap=cmap,
            unit=unit,
            show_unit_cell=show_unit_cell,
            unit_cell_color=unit_cell_color,
        )
        return ax, cm

    def draw_stencil_patch_boundaries(
        self, ax=None, repeat=(1, 1), unit="um", color="white"
    ):
        """Plot the patch boundaries in the stencil using Shapely
        Parameters:
        - ax: matplotlib Axes object
        - repeat: repeat in x- and y-directions
        - unit: the unit to draw on axes, one of 'nm', 'um' or 'mm'
        - color: line color of the patch
        """
        ax = ensure_ax(ax)
        new_geometry = self.geometry * repeat
        unit_ratio = unit_properties[unit]["ratio"]

        for patch in new_geometry.patches:
            ax.plot(*patch.exterior.xy / unit_ratio, "--", color=color)
        return ax


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
            self.phi, self.theta = trajectory[:, 1], trajectory[:, 0]
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
        x_center, y_center = (
            R_center * np.cos(theta),
            R_center * np.sin(theta),
        )

        # The max radius of the filter pattern
        R_max = np.max(np.abs(R_center))
        # Ensure that we have at least some matrix
        if R_max == 0:
            R_max = self.diffusion
        # TODO: make sure F matrix is always odd
        xy_lim = domain_ratio * R_max
        x_range = np.arange(-xy_lim, xy_lim, h)
        y_range = np.arange(-xy_lim, xy_lim, h)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        F_mesh, _, _ = np.histogram2d(
            x_center,
            y_center,
            bins=[xmesh.shape[0], ymesh.shape[1]],
            range=[
                [x_range.min(), x_range.max()],
                [y_range.min(), y_range.max()],
            ],
        )
        sigma = int(self.diffusion / h)
        F_mesh = gaussian_filter(F_mesh, sigma)
        return Mesh(F_mesh, x_range, y_range)

    def draw(
        self, ax=None, stencil=None, grid="polar", cmap="gray", phi_max=None
    ):
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
            ax.scatter(
                self.theta, np.sin(self.phi), marker="o", label="Trajectory"
            )

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(1)
            rticks_raw = np.linspace(0, phi_max, 5)
            rticks_label_deg = [
                f"{v:.2f}°" for v in np.degrees(np.arcsin(rticks_raw))
            ]
            ax.set_rmax(np.sin(phi_max))
            ax.set_rticks(rticks_raw, labels=rticks_label_deg)
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\phi$")

        elif grid == "cartesian":
            # Cartesian grid: Generate and plot the filter
            if stencil is None:
                raise ValueError(
                    (
                        "No stencil provided to generate "
                        "a cartesian projection!"
                    )
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
    def __init__(self, stencil, physics):
        self.stencil = stencil
        self.physics = physics
        self.results = None

    @property
    def h(self):
        return self.stencil.h

    def simulate(self, method="auto"):
        """Simulate the convolution between stencil and physics

        There are several methods to perform the convolution operation
        - 'auto': default, automatically choose the fastest approach
        - 'fast': use fftconvolve to compute the convolution,
                 works only when membrane thickness delta is much
                 smaller than membrane-to-substrate spacing D
        - 'full': use custom real-space convolution considering
                  the thickness and stencil hole positions

        Returns:
        - results: also stored in self.results
                   if self.stencil is periodic --> periodic mesh
                   else                        --> non-periodic mesh

        """
        method = method.lower()
        assert method in ("auto", "fast", "full")
        # TODO: determine the delta / D ratio for fast / full mode
        # TODO: should also consider phi_c
        F = self.physics.generate_F(self.stencil).array
        M_origin = self.stencil.generate_mesh()
        # For periodic stencil,
        # we need 2 * extra + 1 tiles, where
        # extra * M_origin.shape[i] > F.shape[i] // 2
        print("F shape", F.shape)
        if self.stencil.is_periodic:
            # TODO: we just consider 1 direction now
            extra = math.ceil(F.shape[0] / 2 / M_origin.shape[0])
            M, recover_indices = M_origin.tiled_array(extra_x=extra)
        else:
            pad = F.shape[0] // 2
            M, recover_indices = M_origin.padded_array(pad=pad)

        # The fast approach
        # TODO: make sure auto selects
        # TODO: choose between fast and full methods
        if method in ("fast", "auto"):
            print(M.shape, F.shape)
            print(M.ndim, F.ndim)
            results_padded = fftconvolve(M, F, mode="same")
            results = results_padded[recover_indices]
            self.results = Mesh(results, M_origin.x_range, M_origin.y_range)
        else:
            raise NotImplementedError()
        return self.results

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
