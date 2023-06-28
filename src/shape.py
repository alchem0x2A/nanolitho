import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box, Point
from shapely.affinity import rotate, translate
from matplotlib import patches
from shapely.vectorized import contains
import numba
from shapely import simplify

# All dimensions are in um
mm = 1000
um = 1.0
nm = 0.001


def make_circle(x, y, r):
    return Point(x, y).buffer(r)

# def make_line()


def cosine_dist(degree, power=1):
    """Cosine distribution cos^p (deg)
    """
    c = np.cos(np.deg2rad(degree))
    cp = np.power(c, power)
    return cp

def gauss_angle_dist(degree, width, power=1):
    """Cosine distribution cos^p (deg)
    """
    return np.exp(-np.power((degree - 0)/width, 2.)/2.)


def translate_by_theta_r(shape, theta, r, delta=0):
    """Translate a shape by theta and r

    
    shift_x = r * sin(theta)
    shift_y = r * cos(theta)
    delta: further shift in the angle direction
    """
    shift_x = np.cos(np.deg2rad(theta)) * (r + delta)
    shift_y = np.sin(np.deg2rad(theta)) * (r + delta)
    new_shape = translate(shape, xoff=shift_x, yoff=shift_y)
    return new_shape


def shape_to_projection(shape, thickness, theta, psi, H, **params):
    """Generate the projected shape along incident angle (theta, psi)
    given the film thickness and spacing H
    """
    tan_psi = np.tan(np.deg2rad(psi))
    r = tan_psi * thickness
    R = tan_psi * H
    shape1 = translate_by_theta_r(shape, theta, r, **params)
    allowed = shape.intersection(shape1)
    projection = translate_by_theta_r(allowed, 180 + theta, (r + R), **params)
    return projection


def get_union_shape(shape, thickness, psi, H, steps=360, **params):
    """Get an envelop of all theta angles
    """
    envelope = shape_to_projection(shape=shape,
                                   thickness=thickness,
                                   theta=0,
                                   psi=psi,
                                   H=H,
                                   **params)
    angles = np.linspace(0, 360, steps)
    for theta in angles:
        new_shape = shape_to_projection(shape=shape,
                                        thickness=thickness,
                                        theta=theta,
                                        psi=psi,
                                        H=H,
                                        **params)
        envelope = envelope.union(new_shape)
    return envelope


def bbox_to_mesh(shape, size=1024):
    """Rasterize a shape to a NumPy array
    shape could be conveniently a bbox
    
    """
    # Get the bounding box of the shape
    minx, miny, maxx, maxy = shape.bounds

    # Generate a grid of points
    x, y = np.mgrid[:size, :size]
    points = np.vstack((x.flatten(), y.flatten())).T

    # Scale the points to the bounding box of the shape
    points[:, 0] = points[:, 0] / size * (maxx - minx) + minx
    points[:, 1] = points[:, 1] / size * (maxy - miny) + miny

    return points


def draw_shape(ax, shape, c='black', alpha=0.25):
    """Draw a shape on a given axis"""
    # Create a Polygon patch for the exterior
    exterior_patch = patches.Polygon(np.column_stack(shape.exterior.xy),
                                     fill=True,
                                     color=c,
                                     edgecolor="none",
                                     alpha=alpha)
    ax.add_patch(exterior_patch)

    # Create Polygon patches for the interiors
    for interior in shape.interiors:
        interior_patch = patches.Polygon(np.column_stack(interior.xy),
                                         fill=True,
                                         edgecolor="none",
                                         color='white')
        ax.add_patch(interior_patch)
    return

def shift_array(z, xx, yy, xs, ys):
    """Shifts a 2D array over a grid by a specified x and y offset.

    Args:
    z: 2D array.
    xx: x coordinates.
    yy: y coordinates.
    xs: x offset.
    ys: y offset.

    make sure z, xx and yy are of same dimension!
    xx, yy are created by numpy's meshgrid method
    Returns:
    Shifted 2D array.
    """
    # Compute shift in terms of grid indices
    dx = int(round(xs / (xx[0, 1] - xx[0, 0])))
    dy = int(round(ys / (yy[1, 0] - yy[0, 0])))
    print(dx, dy)

    # Create a new array filled with zeros
    z_shifted = np.zeros_like(z)

    # Shift the array
    if dx >= 0 and dy >= 0:
        z_shifted[dy:, dx:] = z[:-dy or None, :-dx or None]
    elif dx >= 0 and dy < 0:
        z_shifted[dy:, :dx] = z[:-dy or None, -dx:]
    elif dx < 0 and dy >= 0:
        z_shifted[:dy, dx:] = z[-dy:, :-dx or None]
    else: # dx < 0 and dy < 0
        z_shifted[:dy, :dx] = z[-dy:, -dx:]

    return z_shifted




def calc_intensity_matrix(shape, intensity, xmesh, ymesh):
    """Calculates an intensity matrix for a shape over a grid.

    This version assumes to find the shape inside the bounding box of current shape
    """
    # Create a grid of points within the bbox
    bbox = shape.bounds
    mask = np.zeros_like(xmesh, dtype=bool)
    
    # Find the indices of the points within the bbox
    within_bbox_indices = ((xmesh >= bbox[0]) & (xmesh <= bbox[2]) & (ymesh >= bbox[1]) & (ymesh <= bbox[3]))
    # Only consider points within the bbox for contains calculation
    x_within_bbox = xmesh[within_bbox_indices]
    y_within_bbox = ymesh[within_bbox_indices]

    # Create a binary mask of the shape over the grid
    mask[within_bbox_indices] = contains(shape, x_within_bbox, y_within_bbox)
    
    
    # Multiply the mask by the intensity to obtain an intensity matrix
    intensity_matrix = mask * intensity

    return intensity_matrix
