import numpy as np
from shapely.geometry import box
from mbhl.simulation import Mask, Physics, System, Circle, Rectangle, Square
from mbhl.simulation import nm, um
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import rotate
from tqdm.auto import tqdm
from multiprocessing import Pool

def plot_3P_deposition(
    ax=None,
    rl_ratio=0.25,
    Rl_ratio=0.5,
    theta=0.0,
    L=500 * nm,
    H = 5 * um,
    diffusion=10 * nm,
    default_h=5 * nm,
    default_alpha=1.0,
    drift=0 * nm,
    # lim=(0.5 * um, 7.5 * um, 0.5 * um, 7.5 * um),
    gap=2.5 * um,
    cmap="viridis",
    alpha=1.0,
    h_ratio=1.0,
    n_pts=3,
    repeat=10,
):
    radius = L * rl_ratio
    R = L * Rl_ratio
    psi = np.arctan(R / H / 2)

    n_pts = n_pts
    angle_offset = theta
    trajectory = np.array(
        [(psi, theta) for theta in np.deg2rad(np.linspace(angle_offset + 0, angle_offset + 360, n_pts + 1))[:-1]]
    )
    phys = Physics(trajectory, diffusion=diffusion, drift=drift)
    W = L * 3 ** 0.5
    pos = [Circle(0, 0, radius), Circle(W / 2, W / 2 * np.sqrt(3), radius), 
           Circle(W / 2, W / 6 * np.sqrt(3), radius), 
           Circle(W, W * 2 / 3 * np.sqrt(3), radius)]
    cell = (W, W * np.sqrt(3))
    mask = Mask(
        pos,
        unit_cell=cell,
        repeat=(repeat, repeat),
        pad=50 * nm,
        thickness=0 * nm,
        spacing=H,
    )
    system = System(mask=mask, physics=phys)
    conv = system.simulate(h=default_h / h_ratio)
    if ax is not None:
        ax.set_axis_off()
        system.draw(
            ax=ax,
            mask_alpha=1.0,
            show_mask=False,
            cmap=cmap,
            mask_lw=0,
            xlim=[W, (repeat - 1) * W],
            ylim=[W, (repeat - 1) * W],
            alpha=alpha,
            # vmax=200 / n_pts
        )
    return system

def gen_zmesh_pn(rl_ratio=0.25, Rl_ratio=0.5, theta=0.0, repeat=1, 
                 default_h=10 * nm, method=plot_3P_deposition):
    """Generate the mesh between positive and negative mesh
    """
    # Positive
    system_p = method(ax=None, rl_ratio=rl_ratio, Rl_ratio=Rl_ratio, theta=theta,
                                default_h=default_h, repeat=repeat)
    intensity_p, x, y = system_p.results
    xx, yy = np.meshgrid(x, y)

    # Negative
    system_n = method(ax=None, rl_ratio=rl_ratio, Rl_ratio=Rl_ratio, theta=-theta,
                                default_h=default_h, repeat=repeat)
    intensity_n, x, y = system_n.results
    xx, yy = np.meshgrid(x, y)
    # return np.linalg.norm(intensity_n - intensity_p)
    return intensity_n, intensity_p

def extract_center(array, w=100, shift=(0, 0)):
    # in px
    center = np.array(array.shape) // 2 + np.array(shift)
    half_size = w // 2
    return array[center[0] - half_size: center[0] + half_size, center[1] - half_size:center[1] + half_size]

def diff_func(img1, img2, w=100, shift=(0, 0)):
    # Compare 
    # force shift to be int
    shift = np.asarray(shift, dtype=int)
    array1 = extract_center(img1, w=w, shift=(0, 0))
    array2 = extract_center(img2, w=w, shift=shift)
    return np.mean(np.abs(array1 - array2))



def grid_search_min_diff(img1, img2, w=150, search_range=30, angle_range=(0, 180)):
    """Perform a grid search to find the optimal integer shift."""
    min_diff = float('inf')
    optimal_shift = (0, 0)
    optimal_angle = 0
    for ang in np.linspace(angle_range[0], angle_range[1], 12):
        img2_rot = rotate(img2, ang)
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                current_diff = diff_func(img1, img2_rot, w=w, shift=(dx, dy))
                if current_diff < min_diff:
                    min_diff = current_diff
                    optimal_shift = (dx, dy)
                    optimal_angle = ang
    return optimal_shift, optimal_angle, min_diff

def compute_min_diff_3P(r_, R_):
    print(r_, R_)
    i_p, i_n = gen_zmesh_pn(rl_ratio=r_, Rl_ratio=R_, theta=15, repeat=5, default_h=3 * nm)
    shift, angle, vmin = grid_search_min_diff(i_p, i_n, w=200, search_range=50)
    return vmin

if __name__ == "__main__":
    min_array = []
    r = np.linspace(0.10, 0.65, 32)
    R = np.linspace(0.10, 0.95, 32)
    rr, RR = np.meshgrid(r, R)
    parameters = list(zip(rr.ravel(), RR.ravel()))
    min_array = []

    with Pool(8) as pool:
        min_array = list(pool.starmap(compute_min_diff_3P, parameters))
    min_array = np.reshape(np.array(min_array), rr.shape)
    np.savez("min_array_honeycomb.npz", r=r, R=R, min_array=min_array)