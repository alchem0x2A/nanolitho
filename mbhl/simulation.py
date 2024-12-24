import math
import warnings
from pathlib import Path

import numpy as np
from matplotlib.projections.polar import PolarAxes
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2
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
        gap=2.5 * um,
        h=None,
    ):
        """
        h: default mesh spacing for the stencil,
           if none, h = dimension / default_division
        """
        self.geometry = self._assign_geometry(geometry)
        self.delta = thickness
        self.D = gap
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
        return self.geometry.cell

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
        self,
        ax=None,
        repeat=(1, 1),
        unit="um",
        dimension_ratio=None,
        color="white",
        lw=1,
        ls="-",
        alpha=1.0,
    ):
        """Plot the patch boundaries in the stencil using Shapely
        Parameters:
        - ax: matplotlib Axes object
        - repeat: repeat in x- and y-directions
        - unit: the unit to draw on axes, one of 'nm', 'um' or 'mm'
        - color: line color of the patch
        """
        ax = ensure_ax(ax)
        if ax is not None:
            original_xlim = ax.get_xlim()
            original_ylim = ax.get_ylim()
        else:
            original_xlim, original_ylim = None, None

        new_geometry = self.geometry * repeat
        if dimension_ratio is None:
            unit_ratio = unit_properties[unit]["ratio"]
        else:
            unit_ratio = float(dimension_ratio)

        for patch in new_geometry.patches:
            exterior = patch.exterior.xy
            # print(exterior)
            (xy,) = (np.array(exterior) / unit_ratio,)
            # print(xy)
            # print(xy.shape)
            ax.plot(
                xy[0],
                xy[1],
                ls,
                color=color,
                alpha=alpha,
                linewidth=lw,
            )
        if original_xlim is not None:
            ax.set_xlim(*original_xlim)
        if original_ylim is not None:
            ax.set_ylim(*original_ylim)

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

    def _generate_F_fold(self, stencil, domain_ratio=1.5, add_diffusion=True):
        """Generate offset trajectory F matrix that is folded back
        to the first periodic zone

        **This is intended for testing only.
        Do not use this method for actual calculations!**
        `fold_to_bz=True` in generate_F_fold is much more reliable

        This ensures that the dimension of F is the same as stencil
        """
        if not stencil.is_periodic:
            raise ValueError(
                "generate_F_fold only works with periodic stencil!"
            )
        F_mesh = self.generate_F(
            stencil,
            domain_ratio=domain_ratio,
            add_diffusion=add_diffusion,
            fold_to_bz=False,
        )
        cell_w, cell_h = stencil.cell
        h = stencil.h
        n_x = int(np.round(cell_w / h))
        n_y = int(np.round(cell_h / h))

        # Ensure odd dimensions for compatibility with centered F
        if n_x % 2 == 0:
            n_x += 1
        if n_y % 2 == 0:
            n_y += 1

        center_i_F = F_mesh.array.shape[0] // 2
        center_j_F = F_mesh.array.shape[1] // 2
        center_i_fold = n_y // 2
        center_j_fold = n_x // 2

        folded_F = np.zeros((n_y, n_x), dtype=float)
        for i in range(F_mesh.array.shape[0]):
            for j in range(F_mesh.array.shape[1]):
                # Compute periodic indices relative to the center of the unit cell
                i_fold = (i - center_i_F + center_i_fold) % n_y
                j_fold = (j - center_j_F + center_j_fold) % n_x
                # print(i_fold, j_fold)
                folded_F[i_fold, j_fold] += F_mesh.array[i, j]

        x_range = np.linspace(-n_x * h / 2, n_x * h / 2, n_x)
        y_range = np.linspace(-n_y * h / 2, n_y * h / 2, n_y)
        return Mesh(folded_F, x_range, y_range)

    def generate_F(
        self, stencil, domain_ratio=1.5, add_diffusion=True, fold_to_bz=False
    ):
        """Generate the offset mesh based on the stencil

        The following parameters are determined from the stencil geometry:
        mesh spacing (h)
        membrane-to-substrate spacing (D)
        membrane thickness (delta)

        Parameters:
        - stencil: a Stencil object
        - domain_ratio: factor to extend the mesh domain
                        relative to max radius.
        - add_diffusion: whether to add diffusion kernel on F
        - fold_to_bz: fold to 1st Brillouin zone
        """

        # Create the "central" trajectory points
        h, D, delta = stencil.h, stencil.D, stencil.delta
        phi, theta = self.phi, self.theta
        R_center = (D + delta) * np.tan(phi) + self.drift
        # x_center, y_center = (
        #     R_center * np.cos(theta),
        #     R_center * np.sin(theta),
        # )
        x_center, y_center = (
            -R_center * np.cos(theta),
            R_center * np.sin(theta),
        )

        if fold_to_bz:
            if not stencil.is_periodic:
                raise ValueError("fold_to_bz requires a periodic stencil!")
            cell_w, cell_h = stencil.cell
            n_x = int(np.round(cell_w / h))
            n_y = int(np.round(cell_h / h))

            # Ensure odd dimensions for compatibility with centered folding
            if n_x % 2 == 0:
                n_x += 1
            if n_y % 2 == 0:
                n_y += 1

            x_range = np.linspace(-n_x * h / 2, n_x * h / 2, n_x)
            y_range = np.linspace(-n_y * h / 2, n_y * h / 2, n_y)
        else:
            # The max radius of the filter pattern
            R_max = np.max(np.abs(R_center))
            # Ensure that we have at least some matrix
            if R_max == 0:
                R_max = self.diffusion
            xy_lim = domain_ratio * R_max
            n_points = int(np.ceil(2 * xy_lim / h))  # Total number of points
            if n_points % 2 == 0:
                n_points += 1  # Make sure the number of points is odd
            half_range = (n_points // 2) * h
            x_range = np.linspace(-half_range, half_range, n_points)
            y_range = np.linspace(-half_range, half_range, n_points)

        xmesh, ymesh = np.meshgrid(x_range, y_range)
        if fold_to_bz:
            # Folding the offsets back to the first BZ
            folded_offsets_x = (x_center + cell_w / 2) % cell_w - cell_w / 2
            folded_offsets_y = (y_center + cell_h / 2) % cell_h - cell_h / 2
            F_mesh, _, _ = np.histogram2d(
                folded_offsets_x,
                folded_offsets_y,
                bins=[n_x, n_y],
                range=[
                    [x_range.min(), x_range.max()],
                    [y_range.min(), y_range.max()],
                ],
            )
        else:
            F_mesh, _, _ = np.histogram2d(
                x_center,
                y_center,
                bins=[xmesh.shape[0], ymesh.shape[1]],
                range=[
                    [x_range.min(), x_range.max()],
                    [y_range.min(), y_range.max()],
                ],
            )
        if add_diffusion:
            sigma = int(self.diffusion / h)
            F_mesh = gaussian_filter(F_mesh, sigma)
        return Mesh(F_mesh, x_range, y_range)

    def draw(
        self, ax=None, stencil=None, grid="polar", cmap="gray", phi_max=None
    ):
        """Draw the offset trajectory pattern on polar or cartesian grids

        Drawing on cartesian grids isn't always clearly visible if the domain
        is very large, we recommend using the polar grid for
        better visualization results

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

        if grid == "polar":
            if (ax is not None) and not isinstance(ax, PolarAxes):
                raise ValueError(
                    (
                        "The provided axis must have a "
                        "polar projection for 'polar' grid."
                    )
                )
            ax = ensure_ax(ax, subplot_kw={"projection": "polar"})

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

        else:
            # Cartesian grid: Generate and plot the filter
            if stencil is None:
                raise ValueError(
                    (
                        "No stencil provided to generate "
                        "a cartesian projection!"
                    )
                )
            mesh = self.generate_F(stencil)
            ax, cm = mesh.draw(ax=ax)
        return ax


class System:
    def __init__(self, stencil, physics):
        self.stencil = stencil
        self.physics = physics
        self.results = None

    @property
    def h(self):
        return self.stencil.h

    def simulate(self, method="auto", fold_to_bz=True):
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
        assert method in ("auto", "fft", "raytracing", "direct")
        # TODO: determine the delta / D ratio for fast / full mode
        # TODO: should also consider phi_c

        if method == "auto":
            # TODO: implement method choose
            method = "fft"

        if method == "fft":
            self.simulate_fftconvolve(fold_to_bz=fold_to_bz)
        elif method == "direct":
            raise NotImplementedError()
        elif method == "raytracing":
            self.simulate_raytracing(use_periodic=True)

        if self.results is None:
            raise RuntimeError("Simulation does not produce any results!")

    def _prepare_matrices(self, add_diffusion=True, fold_to_bz=True):
        """Prepare the F and M matrices of the system

        Parameters:
        - add_diffusion: whether to create a smeared F matrix
        - fold_to_bz: whether to fold the F matrix to 1st BZ

        Returns:
        - F: offset trajectory matrix
        - M: padded stencil matrix
        - M_origin: original stencil matrix
        - recover_indices: indices to trim the resulted matrix
        """
        # For periodic stencil,
        # we need 2 * extra + 1 tiles, where
        # extra * M_origin.shape[i] > F.shape[i] // 2
        #
        # When using the folding scheme, extra is always 1

        if self.stencil.is_periodic:
            # TODO: allow M_origin to get odd number
            M_origin = self.stencil.generate_mesh()
            F = self.physics.generate_F(
                self.stencil, add_diffusion=add_diffusion, fold_to_bz=fold_to_bz
            ).array
            # Even with bz folding, we still need 1 extra image
            # each direction
            if fold_to_bz:
                # In case there is 1 pixel difference
                extra = 1
            else:
                extra = math.ceil(F.shape[0] / 2 / M_origin.shape[0])
            M, recover_indices = M_origin.tiled_array(extra_x=extra)
        else:
            F = self.physics.generate_F(
                self.stencil,
                add_diffusion=add_diffusion,
            ).array
            M_origin = self.stencil.generate_mesh()
            pad = F.shape[0] // 2
            M, recover_indices = M_origin.padded_array(pad=pad)
        return F, M, M_origin, recover_indices

    def simulate_fftconvolve(self, fold_to_bz=True):
        """Use the fftconvolve method to calculate
        the convolution between M and F.

        This method ignores the finite thickness of membrane, but
        is very efficient for calculation trajectories of small phi
        angles
        """
        F, M, M_origin, recover_indices = self._prepare_matrices(
            add_diffusion=True, fold_to_bz=fold_to_bz
        )
        results_padded = fftconvolve(M, F, mode="same")
        results = results_padded[recover_indices]
        self.results = Mesh(results, M_origin.x_range, M_origin.y_range)
        return

    def simulate_direct(self):
        """Direct approach of simulating lithography pattern
        This approach first computes the 1 beam shadow formed by
        the stencil and indicent beam. The lithography pattern is
        then the normalized sum of all the beam patterns.

        This approach is a naive implementation for testing and benchmarking,
        essentially the reverse operation of simulate_raytracing.

        theta is defined as the azimuthal angle counted from the +y direction
        """
        from .utils import shape_to_projection

        h, D, delta = self.stencil.h, self.stencil.D, self.stencil.delta
        R_s = self.physics.drift
        diffusion = self.physics.diffusion
        theta, phi = self.physics.theta, self.physics.phi
        # Calculate all projected patches
        projected_patches = []
        # If periodic, we have to create periodic images
        # if self.stencil.is_periodic:
        #     patches = []
        # else:
        patches = self.stencil.geometry.patches
        if self.stencil.is_periodic:
            unit_cell = self.stencil.cell
        else:
            unit_cell = None
        for theta_, phi_ in zip(theta, phi):
            for patch in patches:
                new_patch = shape_to_projection(
                    patch,
                    theta_,
                    phi_,
                    D=D,
                    delta=delta,
                    R_s=R_s,
                    unit_cell=unit_cell,
                )
                projected_patches.append(new_patch)

        combined_geometry = Geometry(
            projected_patches, cell=unit_cell, pbc=True
        )
        # Combine objects on mesh
        M_origin = self.stencil.generate_mesh()
        results = np.zeros_like(M_origin.array).astype(float)
        # Overlay stencil
        if self.stencil.is_periodic:
            M, recover_indices = M_origin.tiled_array(extra_x=1)
            tiled_M = M_origin * (3, 3)
            combined_geometry = combined_geometry * (3, 3)
            xmesh, ymesh = np.meshgrid(tiled_M.x_range, tiled_M.y_range)
            for patch in combined_geometry.patches:
                mask = contains(patch, xmesh, ymesh).astype(float)
                results += mask[recover_indices]
        else:
            M = M_origin
            xmesh, ymesh = np.meshgrid(M.x_range, M.y_range)
            for patch in combined_geometry.patches:
                mask = contains(patch, xmesh, ymesh).astype(float)
                results += mask
        # Normalize result
        results = results / (np.max(results) + 1.0e-10)
        sigma = diffusion / h
        results_blurred = gaussian_filter(results, sigma=sigma)
        self.results = Mesh(results_blurred, M_origin.x_range, M_origin.y_range)
        return

    def simulate_raytracing(self, use_periodic=True):
        """Use the ray tracing equation to calculate
        the contributions at each pixel on the substrate

        This method is considerably slower than `simulate_fftconvolve`
        but considers the finite thickness of the membrane
        """
        from scipy.ndimage import label

        # For ray tracing, folding F back to 1st BZ isn't helping
        (F, M, M_origin, recover_indices) = self._prepare_matrices(
            add_diffusion=False, fold_to_bz=False
        )
        # We will regenerate M as 3x3 tile when stencil is periodic
        L, _ = label(M)
        L_recover_indices = None
        if self.stencil.is_periodic and use_periodic:
            rows, cols = M_origin.shape
            extra_L = int((M.shape[0] // rows - 1) / 2)
            # L matrix is generated on a larger repetition
            # with extra_L * 2 + 1 copies of periodic matrix
            L_recover_indices = (
                slice(rows * (extra_L - 1), rows * (extra_L + 2)),
                slice(cols * (extra_L - 1), cols * (extra_L + 2)),
            )
            M, recover_indices = M_origin.tiled_array(extra_x=1)

        M = M.astype(bool)
        h, D, delta = self.stencil.h, self.stencil.D, self.stencil.delta
        R_s = self.physics.drift
        # For the simulate_raytracing to be efficient, we
        # take the fact that non-zero points in F are sparse
        # In this case F should not be gaussian-blurred
        # [(i, j), ...] indices where F.array is non-zero
        F_nz_indices = np.argwhere(F > 0)
        F_center = np.array(F.shape) // 2  # center where (x, y) == 0
        F_nz_shifts = F_nz_indices - F_center
        R_nz = np.sqrt(
            (F_nz_shifts[:, 0] * h) ** 2 + (F_nz_shifts[:, 1] * h) ** 2
        )
        R_nz_bottom = R_s + (R_nz - R_s) * D / (delta + D)
        F_bottom_nz_shifts = np.round(
            (R_nz_bottom / R_nz)[:, None] * F_nz_shifts
        ).astype(int)

        results = np.zeros_like(M, dtype=float)
        for f_shift, f_bottom_shift, f_value in zip(
            F_nz_shifts,
            F_bottom_nz_shifts,
            F[F_nz_indices[:, 0], F_nz_indices[:, 1]],
        ):
            # Retrieve mask and label values
            mask_T = np.roll(M, f_shift, axis=(0, 1))
            mask_B = np.roll(M, f_bottom_shift, axis=(0, 1))
            # L matrix are much larger than M,
            # so edge effects are eliminated
            label_T = np.roll(L, f_shift, axis=(0, 1))
            label_B = np.roll(L, f_bottom_shift, axis=(0, 1))

            # print(mask_T.shape, label_T.shape)
            if self.stencil.is_periodic and use_periodic:
                label_T = label_T[L_recover_indices]
                label_B = label_B[L_recover_indices]
            # print(mask_T.shape, label_T.shape)
            # Apply shadowing condition (labels must match)
            # TODO: make sure label_T and label_B are separated less than 1unit
            shadowing_condition = (label_T == label_B).astype(bool)

            # Update the results matrix
            results += f_value * (mask_T & mask_B & shadowing_condition)
            # results += f_value * (mask_T & mask_B)

        results_recovered = results[recover_indices]
        sigma = self.physics.diffusion / h
        results_blurred = gaussian_filter(results_recovered, sigma=sigma)
        self.results = Mesh(results_blurred, M_origin.x_range, M_origin.y_range)
        return

    def save_tiff(self, fname, repeat=(1, 1)):
        """Save the normalized height as a tiff file
        so that the file can be opened by
        softwares like gwyddion
        """
        if self.results is None:
            raise RuntimeError("Please finish simulation first!")
        tiled_results = self.results * repeat
        prob = tiled_results.array
        z_image = Image.fromarray(prob)
        z_image_file = Path(fname)
        # Custom tiff info
        custtifftags = ImageFileDirectory_v2()
        hx = tiled_results.x_range[1] - tiled_results.x_range[0]
        hy = tiled_results.y_range[1] - tiled_results.y_range[0]
        # spacing in um
        custtifftags[65000] = f"XSpacing={hx}"
        custtifftags[65001] = f"XSpacing={hy}"
        z_image.save(z_image_file, tiffinfo=custtifftags)
        return

    def draw(
        self,
        ax=None,
        cmap="viridis",
        unit="um",
        repeat=(1, 1),
        show_mask=True,
        mask_lw=1.0,
        mask_alpha=0.5,
        mask_color="white",
        mask_ls="-",
        dimension_ratio=None,
        domain=None,
        alpha=1.0,
        vmax=None,
    ):
        """Draw the system simulation results as 2D map"""
        # TODO: make sure lazy evaluation of the results
        if self.results is None:
            raise RuntimeError("Please finish simulation first!")

        ax, cm = self.results.draw(
            ax=ax,
            repeat=repeat,
            unit=unit,
            domain=domain,
            dimension_ratio=dimension_ratio,
            cmap=cmap,
            vmax=vmax,
            alpha=alpha,
        )

        # Use the stencil
        if show_mask:
            ax = self.stencil.draw_stencil_patch_boundaries(
                ax=ax,
                repeat=(repeat[0] + 1, repeat[1] + 1),
                dimension_ratio=dimension_ratio,
                unit=unit,
                lw=mask_lw,
                color=mask_color,
                alpha=mask_alpha,
            )
        return
