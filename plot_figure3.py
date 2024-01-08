#!/usr/bin/env python
import numpy as np
from mbhl.simulation import Mask, Physics, System, Circle, Rectangle, Square
from mbhl.simulation import nm, um
import matplotlib.pyplot as plt
import figurefirst as ff
from pathlib import Path

curdir = Path(__file__).parent
figure_dir = curdir / "figures"
template = figure_dir / "figure3-template.svg"
diffusion = 20 * nm
default_h = 5 * nm
default_alpha = 1.0


def plot_square_P_map(
    ax,
    psi=5.0,
    diameter=100 * nm,
    spacing_raw=400 * nm,
    drift=0 * nm,
    lim=(0.5 * um, 3.5 * um, 0.5 * um, 3.5 * um),
    gap=2.5 * um,
    cmap="viridis",
    alpha=1.0,
):
    psi = np.deg2rad(psi)
    radius = diameter / 2
    H = gap

    trajectory = np.array(
        [(psi, theta) for theta in np.deg2rad(np.linspace(0, 360, 720))]
    )
    phys = Physics(trajectory, diffusion=diffusion, drift=drift)
    W = spacing_raw + diameter
    mask = Mask(
        [Circle(0, 0, radius)],
        unit_cell=(W, W),
        repeat=(20, 20),
        pad=50 * nm,
        thickness=100 * nm,
        spacing=H,
    )
    system = System(mask=mask, physics=phys)
    conv = system.simulate(h=default_h)
    ax.set_axis_off()
    system.draw(
        ax,
        mask_alpha=0.8,
        show_mask=True,
        cmap=cmap,
        mask_lw=2.5,
        xlim=lim[:2],
        ylim=lim[2:],
        alpha=alpha,
    )
    return


def plot_hex_P_map(
    ax,
    psi=5.0,
    diameter=100 * nm,
    spacing_raw=400 * nm,
    drift=0 * nm,
    lim=(0.5 * um, 3.5 * um, 0.5 * um, 3.5 * um),
    gap=2.5 * um,
    cmap="viridis",
    alpha=1.0,
):
    psi = np.deg2rad(psi)
    radius = diameter / 2
    H = gap

    trajectory = np.array(
        [(psi, theta) for theta in np.deg2rad(np.linspace(0, 360, 720))]
    )
    phys = Physics(trajectory, diffusion=diffusion, drift=drift)
    W = spacing_raw + diameter
    mask = Mask(
        [Circle(0, 0, radius), Circle(W / 2, W / 2 * np.sqrt(3), radius)],
        unit_cell=(W, W * 3**0.5),
        repeat=(20, 15),
        pad=50 * nm,
        thickness=100 * nm,
        spacing=H,
    )
    system = System(mask=mask, physics=phys)
    conv = system.simulate(h=default_h)
    ax.set_axis_off()
    system.draw(
        ax,
        mask_alpha=0.8,
        show_mask=True,
        mask_lw=2.5,
        cmap=cmap,
        xlim=lim[:2],
        ylim=lim[2:],
        alpha=alpha,
    )
    return


def plot_colorbar(cax, cmap="viridis"):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow([[0, 1], [1, 0]], cmap=cmap)
    fig.colorbar(im, cax=cax)


def plot_main():
    layout = ff.FigureLayout(template)
    layout.make_mplfigures()

    print("Fitting SiO2")
    print("#" * 60)

    drift = 80 * nm
    print("Drawing square sio2")
    ax = layout.axes["sio2-rect"]["axis"]
    x_ = 0.47 * um
    y_ = 0.52 * um
    L = 3.0 * um
    plot_square_P_map(
        ax, drift=drift, lim=(x_, x_ + L, y_, y_ + L), alpha=default_alpha
    )

    drift = 63 * nm
    print("Drawing hex sio2")
    ax = layout.axes["sio2-hex"]["axis"]
    x_ = 0.72 * um
    y_ = 0.60 * um
    L = 3.0 * um
    plot_hex_P_map(ax, drift=drift, lim=(x_, x_ + L, y_, y_ + L), alpha=default_alpha)

    print("Fitting Ge")
    print("#" * 60)

    drift = 140 * nm
    print("Drawing square Ge")
    ax = layout.axes["ge-rect"]["axis"]
    x_ = 0.50 * um
    y_ = 0.64 * um
    L = 3.0 * um
    plot_square_P_map(
        ax, drift=drift, lim=(x_, x_ + L, y_, y_ + L), alpha=default_alpha
    )

    drift = 168 * nm
    print("Drawing hex ge")
    ax = layout.axes["ge-hex"]["axis"]
    x_ = 0.75 * um
    y_ = 0.55 * um
    L = 3.0 * um
    plot_hex_P_map(ax, drift=drift, lim=(x_, x_ + L, y_, y_ + L), alpha=default_alpha)


    drift = 140 * nm
    print("Drawing rect 0")
    ax = layout.axes["ge-simu-0"]["axis"]
    x_ = 0.89 * um
    y_ = 0.30 * um
    spacing = 500 * nm
    L = 3.0 * um
    plot_square_P_map(
        ax,
        spacing_raw=spacing,
        drift=drift,
        lim=(x_, x_ + L, y_, y_ + L),
        alpha=default_alpha,
    )

    print("Drawing rect 1")
    ax = layout.axes[f"ge-simu-1"]["axis"]
    x_ = 0.52 * um
    y_ = 0.53 * um
    spacing = 400 * nm
    L = 3.0 * um
    plot_square_P_map(
        ax,
        drift=drift,
        spacing_raw=spacing,
        lim=(x_, x_ + L, y_, y_ + L),
        alpha=default_alpha,
    )

    print("Drawing rect 2")
    ax = layout.axes["ge-simu-2"]["axis"]
    x_ = 0.88 * um
    y_ = 0.49 * um
    spacing = 300 * nm
    L = 3.0 * um
    plot_square_P_map(
        ax,
        drift=drift,
        spacing_raw=spacing,
        lim=(x_, x_ + L, y_, y_ + L),
        alpha=default_alpha,
    )

    print("Drawing rect 3")
    ax = layout.axes["ge-simu-3"]["axis"]
    x_ = 0.84 * um
    y_ = 0.38 * um
    spacing = 200 * nm
    L = 3.0 * um
    plot_square_P_map(
        ax,
        drift=drift,
        spacing_raw=spacing,
        lim=(x_, x_ + L, y_, y_ + L),
        alpha=default_alpha,
    )

    ax = layout.axes["colorbar1"]["axis"]
    plot_colorbar(ax, cmap="copper")

    ax = layout.axes["colorbar2"]["axis"]
    plot_colorbar(ax, cmap="viridis")

    layout.insert_figures("mpl_render_layer", cleartarget=True)
    layout.write_svg(template)

    return


if __name__ == "__main__":
    plot_main()
