import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
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
    """

    def __init__(self,
                 patches,
                 unit_cell=(1.0 * um, 1.0 * um),
                 repeat=(1, 1),
                 thickness=100 * nm,
                 spacing=2.5 * um):
        self.patches = patches
        self.unit_cell = unit_cell
        self.repeat = repeat
        self.delta = thickness
        self.H = spacing
        self.domain = None

    def create_tiles(self, shrink=0):
        """Create tiles of the patches with some shrink (forbidden area)
        """
        all_patches = []
        cw, ch = self.unit_cell
        for i in range(self.repeat[0]):
            for j in range(self.repeat[1]):
                for patch in self.patches:
                    new_patch = patch.buffer(shrink)
                    all_patches.append(
                        translate(new_patch, xoff=cw * i, yoff=ch * j))
        total_union = unary_union(all_patches)
        return total_union

    def generate_mesh(self, h, domain=None, shrink=0):
        union = self.create_tiles(shrink=shrink)
        if domain is None:
            domain = union.bounds
        self.domain = domain
        minx, miny, maxx, maxy = domain
        x_range = np.arange(minx, maxx, h)
        y_range = np.arange(miny, maxy, h)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        bin_mask = contains(union, xmesh, ymesh)
        return bin_mask


class Physics:

    def __init__(self,
                 psi,
                 psi_broadening=0.05,
                 drift=0 * nm,
                 diffusion=5 * nm):
        self.psi = psi
        self.xi = psi_broadening
        self.drift = drift
        self.diffusion = diffusion

    def generate_filter(self, h, H=2.5 * um, delta=100 * nm, samples=10000):
        # Implementation of the MR matrix generation using psi, broadening, drift, and diffusion
        d_psi = np.random.normal(0, self.xi, samples)
        epsilon = np.random.normal(self.drift, self.diffusion, samples)
        R = (H + delta) * np.tan(np.deg2rad(self.psi + d_psi)) + epsilon
        R_fit = gaussian_kde(R)
        R0 = (H + delta) * np.tan(np.deg2rad(self.psi)) + self.drift
        x_range = np.arange(-2 * R0, 2 * R0, h)
        y_range = np.arange(-2 * R0, 2 * R0, h)
        xmesh, ymesh = np.meshgrid(x_range, y_range)
        R_mesh = np.sqrt(xmesh**2 + ymesh**2)
        R_prob_mesh = R_fit(R_mesh.ravel()).reshape(R_mesh.shape)
        return R_prob_mesh


class System:

    def __init__(self, mask, physics):
        self.mask = mask
        self.physics = physics

    def simulate(self, h):
        # Generate mask and physics matrices
        shrink = -self.mask.delta * np.tan(np.deg2rad(self.physics.psi))
        input_matrix = self.mask.generate_mesh(h, shrink=shrink)
        filter_matrix = self.physics.generate_filter(h,
                                                     H=self.mask.H,
                                                     delta=self.mask.delta)

        # Add zero padding to M
        pad_width = filter_matrix.shape[0] // 2
        input_padded = np.pad(input_matrix,
                              pad_width=pad_width,
                              mode='constant',
                              constant_values=0)

        # Perform convolution
        result = fftconvolve(input_padded, filter_matrix, mode='same')

        # Crop the result to the original size
        result = result[pad_width:-pad_width, pad_width:-pad_width]

        return result
