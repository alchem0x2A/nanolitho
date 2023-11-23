import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, sobel, binary_dilation
from shapely.geometry import Polygon, box, Point
from shapely.affinity import rotate, translate
from shapely.vectorized import contains
from shapely.ops import unary_union

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


class Mask:
    """Create an object for the shadow mask from a list of shape objects

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
        self.H = spacing
        self.pad = pad
        self.domain = None

    def __add__(self, mask):
        if not isinstance(mask, Mask):
            raise TypeError("Only accepts Mask instance")

    def create_tiles(self, shrink=0):
        """Create tiles of the patches with some shrink (forbidden area)"""
        all_patches = []
        cw, ch = self.unit_cell
        for i in range(self.repeat[0]):
            for j in range(self.repeat[1]):
                for patch in self.patches:
                    new_patch = patch.buffer(shrink)
                    all_patches.append(translate(new_patch, xoff=cw * i, yoff=ch * j))
        total_union = unary_union(all_patches)
        return total_union

    def generate_mesh(self, h=10 * nm, domain=None, shrink=0):
        """Create a numpy array as the mesh for the simulation domain"""
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

    def draw(self, ax, h=10 * nm, cmap="gray"):
        """Draw the mask pattern,
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
            cmap=cmap,
        )
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        return


class Physics:
    """A general class for the physics (filter) behind the MBHL
    Note the thickness (delta) and spacing (H) are directly affecting the filter pattern

    Trajectory is an array of (psi, theta) values on the hemisphere
    """

    def __init__(self, trajectory, psi_broadening=0.05, drift=0 * nm, diffusion=5 * nm):
        # self.psi = psi
        self.trajectory = np.atleast_2d(trajectory)
        self.xi = psi_broadening
        self.drift = drift
        self.diffusion = diffusion

    def generate_filter(self, h, H, delta=0 * nm, samples=10000, domain_ratio=1.5):
        # Implementation of the MR matrix generation using psi, broadening, drift, and diffusion

        # Create the "central" trajectory points
        psi, theta = self.trajectory[:, 0], self.trajectory[:, 1]
        R_center = (H + delta) * np.tan(psi) + self.drift
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
        self.mask = mask
        self.physics = physics
        self.results = None
        self.h = None

    def simulate(self, h):
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
        ax.imshow(prob, extent=extent, cmap=cmap, alpha=alpha)

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
