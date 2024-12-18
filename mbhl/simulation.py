import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label, sobel
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.vectorized import contains

# All dimensions are in um
mm = 1000
um = 1.0
nm = 0.001


def Circle(x, y, r):
    return Point(x, y).buffer(r)


def Rectangle(x, y, w, h):
    rectangle = Polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
    return rectangle


def Square(x, y, w):
    return Rectangle(x, y, w, w)


class Stencil:
    """Create an object for the stencil from a list of shape objects

    Example:

    # To create square circle arrays with radius of 400 nm and 200 nm spacing
    # and create 100 nm padding
    patches = [Circle(0, 0, 250 * nm)]
    mask = Mask(patches, unit_cell=(600 * nm, 600 * nm), repeat=(10, 10))
    """

    def __init__(
        self,
        patches,
        unit_cell=(1.0 * um, 1.0 * um),
        repeat=(1, 1),
        pad=0 * nm,
        thickness=100 * nm,
        spacing=2.5 * um,
    ):
        self.patches = patches
        self.unit_cell = unit_cell
        self.repeat = repeat
        self.delta = thickness
        # TODO: obsolete H. Should use D
        self.H = spacing
        self.D = spacing
        self.pad = pad
        # Domain is the min_x, max_x, min_y, max_y
        self.domain = None

    def __add__(self, mask):
        if not isinstance(mask, Mask):
            raise TypeError("Only accepts Mask instance")

    def _create_repeat_tiles(self, repeat_x=1, repeat_y=1, shrink=0):
        """Create tiles with both x and y repeat"""
        # TODO: implement method to wrap back to cell
        all_patches = []
        cw, ch = self.unit_cell
        for i in range(repeat_x):
            for j in range(repeat_y):
                for patch in self.patches:
                    new_patch = patch.buffer(shrink)
                    all_patches.append(translate(new_patch, xoff=cw * i, yoff=ch * j))
        total_union = unary_union(all_patches)
        return total_union

    # TODO: obsolete
    def create_tiles(self, shrink=0):
        """Legacy method to create tiles of
        the patches with some shrink (forbidden area)

        Note this method is grossly inefficient.
        If edge effect can be ignore, use _create_repeat_tiles instead
        """
        total_union = self._create_repeat_tiles(
            repeat_x=self.repeat[0], repeat_y=self.repeat[1], shrink=shrink
        )
        return total_union

    def _generate_mesh_unit_cell(self, h=10 * nm, shrink=0):
        """Generate a per-unit-cell mesh grid with spacing h
        Should consider periodic images
        """
        # TODO: make sure the tiles are always wraped to first unit
        # cell
        # TODO: make sure 9 unit cells are necessary
        cw, ch = self.unit_cell
        union_3x3 = self._create_repeat_tiles(repeat_x=3, repeat_y=3, shrink=shrink)
        # Make sure the matrix is exactly 3M x 3N
        # this will create a small difference between h in both directions
        N, M = int(cw // h), int(ch // h)
        hx, hy = cw / N, ch / M
        x_range = np.linspace(0, 3 * cw, 3 * N)
        y_range = np.linspace(0, 3 * ch, 3 * M)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        print(N, M)
        print(xmesh.shape, ymesh.shape)
        bin_mask = contains(union_3x3, xmesh, ymesh)
        # Pick the values from the center cell
        # the binary mask has shape M x N
        uc_mask = bin_mask[M : 2 * M, N : 2 * N]
        uc_x_range = np.linspace(0, cw, N)
        uc_y_range = np.linspace(0, ch, M)
        return uc_mask, uc_x_range, uc_y_range

    # TODO: obsolete
    def generate_mesh(self, h=10 * nm, domain=None, shrink=0):
        """Legacy method to create a numpy array considering all repeats
        as the mesh for the simulation domain"""
        union = self.create_tiles(shrink=shrink)
        if domain is None:
            domain = union.bounds
            if self.pad != 0:
                new_domain = (
                    domain[0] - self.pad,
                    domain[1] - self.pad,
                    domain[2] + self.pad,
                    domain[3] + self.pad,
                )
                domain = new_domain
        self.domain = domain
        minx, miny, maxx, maxy = domain
        x_range = np.arange(minx, maxx, h)
        y_range = np.arange(miny, maxy, h)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        bin_mask = contains(union, xmesh, ymesh)
        return bin_mask, x_range, y_range

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


# TODO: obsolete class
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
