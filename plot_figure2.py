#!/usr/bin/env python
import numpy as np
from mbhl.simulation import Mask, Physics, System, Circle, Rectangle, Square
from mbhl.simulation import nm, um
import matplotlib.pyplot as plt
import figurefirst as ff
from pathlib import Path

curdir = Path(__file__).parent
figure_dir = curdir / "figures"
template = figure_dir / "figure2-template.svg"


def plot_square_P_map(ax, psi=5.0, spacing_ratio=5, cmap="magma"):
    psi = np.deg2rad(psi)
    radius = 50 * nm
    Ratio = 6
    H = radius * Ratio / np.tan(psi)

    trajectory = np.array([(psi, theta) for theta in np.deg2rad(np.linspace(0, 360, 720))])
    phys = Physics(trajectory, diffusion=20 * nm, drift=0 * nm)
    spacing = spacing_ratio
    W = spacing * radius
    mask = Mask([Circle(0, 0, radius)], 
                unit_cell=(W, W), 
                repeat=(20, 20), 
                pad=50 * nm,
                thickness=100 * nm,
                spacing=H,
                )
    system = System(mask=mask, physics=phys)
    conv = system.simulate(h=5 * nm)
    # fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    system.draw(ax, mask_alpha=0.8, show_mask=True,
                dimension_ratio=radius,
                cmap=cmap,
                xlim=(10, 50), ylim=(10, 50))
    ax.set_title(f"$L/r$={spacing:.1f}")


def plot_hex_P_map(ax, psi=5.0, spacing_ratio=5, cmap="magma"):
    psi = np.deg2rad(psi)
    radius = 50 * nm
    Ratio = 6
    H = radius * Ratio / np.tan(psi)

    trajectory = np.array([(psi, theta) for theta in np.deg2rad(np.linspace(0, 360, 720))])
    phys = Physics(trajectory, diffusion=20 * nm, drift=0 * nm)
    spacing = spacing_ratio
    W = spacing * radius
    mask = Mask([Circle(0, 0, radius)], 
                unit_cell=(W, W), 
                repeat=(20, 20), 
                pad=50 * nm,
                thickness=100 * nm,
                spacing=H,
                )
    system = System(mask=mask, physics=phys)
    conv = system.simulate(h=5 * nm)
    # fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    system.draw(ax, mask_alpha=0.8, show_mask=True,
                dimension_ratio=radius,
                cmap=cmap,
                xlim=(10, 50), ylim=(10, 50))
    ax.set_title(f"$L/r$={spacing:.1f}")

def plot_main():
    layout = ff.FigureLayout(template)
    layout.make_mplfigures()

    for i, ratio in enumerate([6 / 1.414, 5.1, 6, 8]):
        print(f"Drawing square heatmap ax {i+1}")
        ax = layout.axes[f"square-{i+1:d}"]["axis"]
        plot_square_P_map(ax, spacing_ratio=ratio)
        
    
    layout.insert_figures("mpl_render_layer", cleartarget=True)
    layout.write_svg(template)

    return

if __name__ == "__main__":
    plot_main()
