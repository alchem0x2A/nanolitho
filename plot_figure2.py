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
diffusion = 15 * nm


def plot_square_P_map(ax, psi=5.0, R=6.0, spacing_ratio=5, cmap="magma"):
    psi = np.deg2rad(psi)
    radius = 50 * nm
    Ratio = R
    H = radius * Ratio / np.tan(psi)

    trajectory = np.array([(psi, theta) for theta in np.deg2rad(np.linspace(0, 360, 720))])
    phys = Physics(trajectory, diffusion=diffusion, drift=0 * nm)
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
    ax.set_axis_off()
    system.draw(ax, mask_alpha=0.8, show_mask=True,
                dimension_ratio=radius,
                cmap=cmap,
                xlim=(10, 50), ylim=(10, 50))
    ax.set_title(f"$R/L$={6 / spacing:.1f}")
    return

def plot_hex_P_map(ax, psi=5.0, R=8.0, spacing_ratio=5, cmap="magma"):
    psi = np.deg2rad(psi)
    radius = 50 * nm
    Ratio = R
    H = radius * Ratio / np.tan(psi)

    trajectory = np.array([(psi, theta) for theta in np.deg2rad(np.linspace(0, 360, 720))])
    phys = Physics(trajectory, diffusion=diffusion, drift=0 * nm)
    spacing = spacing_ratio
    W = spacing * radius
    mask = Mask([Circle(0, 0, radius), Circle(W / 2, W / 2 * np.sqrt(3), radius)], 
                unit_cell=(W, W * 3**0.5), 
                repeat=(20, 15), 
                pad=50 * nm,
                thickness=100 * nm,
                spacing=H,
                )
    system = System(mask=mask, physics=phys)
    conv = system.simulate(h=5 * nm)
    ax.set_axis_off()
    system.draw(ax, mask_alpha=0.8, show_mask=True,
                dimension_ratio=radius,
                cmap=cmap,
                xlim=(10, 50), ylim=(10, 50))
    ax.set_title(f"$R/L$={Ratio / spacing:.1f}")
    return

def cal_half_alpha(L, R):
    A = (L ** 2 + R ** 2 - 1) / (2 * R * L)
    half_alpha = np.arccos(A)
    if np.isnan(half_alpha):
        half_alpha = 0
    return half_alpha

def calc_P(R, lamb, system="square", positive=True):
    """Implement hex / negative later
    """
    assert R > 0
    assert lamb >= 2
    sum_P = 0
    max_lk = int(R) + 2
    for l in np.arange(-max_lk, max_lk + 1, 1):
        for k in np.arange(-max_lk, max_lk + 1, 1):
            if system == "square":
                L = np.sqrt(l ** 2 + k ** 2) * lamb
            elif system == "hexagonal":
                L = np.sqrt((l + k / 2) ** 2 + k ** 2 * (3 / 4) ) * lamb
            else:
                raise ValueError("Unknown system")
            if L == 0:
                if R < 1:
                    sum_P += 1
            else:
                half_alpha = cal_half_alpha(L, R)
                sum_P += half_alpha / np.pi
    return sum_P

def plot_square_heatmap(ax, R=8.0, L_list=[8 / 1.414, 6.5, 8, 10], cmap="viridis", samples=256):
    lamb_array = np.linspace(0.1, 15, samples) + 2
    R_array = np.linspace(1.5, 15, samples)
    sump = []
    ll, RR = np.meshgrid(lamb_array, R_array)
    from tqdm.auto import tqdm
    for ll, rr in tqdm(zip(ll, RR)):
        sump.append([calc_P(R=r_, lamb=l_, system="square") for r_, l_ in zip(rr, ll)])
    sump = np.array(sump)
    ax.imshow(sump, origin="lower",
              aspect="auto",
              extent=(np.min(lamb_array), np.max(lamb_array), np.min(R_array), np.max(R_array)),
              interpolation="bicubic",
              cmap=cmap)
    for RL_ratio in [5**0.5, 2, 2**0.5, 1]:
        RRR = lamb_array * RL_ratio
        ax.plot(lamb_array, RRR, "--", color="white")
    ax.axhline(y=R, ls="--", color="white")
    ax.plot(L_list, [R,] * 4, "o", color="white")
    ax.set_xlim(np.min(lamb_array), 12)
    ax.set_ylim(np.min(R_array), np.max(R_array))
    ax.set_ylabel("$R/r$")
    ax.set_xlabel("$L/r$")
    ax.set_title("Square Mask")

def plot_hex_heatmap(ax, R=8.0, L_list=[8 / 1.732, 6, 8, 10], cmap="viridis", samples=256):
    lamb_array = np.linspace(0.1, 15, samples) + 2
    R_array = np.linspace(1.5, 15, samples)
    sump = []
    ll, RR = np.meshgrid(lamb_array, R_array)
    from tqdm.auto import tqdm
    for ll, rr in tqdm(zip(ll, RR)):
        sump.append([calc_P(R=r_, lamb=l_, system="hexagonal") for r_, l_ in zip(rr, ll)])
    sump = np.array(sump)
    ax.imshow(sump, origin="lower",
              aspect="auto",
              extent=(np.min(lamb_array), np.max(lamb_array), np.min(R_array), np.max(R_array)),
              interpolation="bicubic",
              cmap=cmap)
    for RL_ratio in [7**0.5, 2, 3**0.5, 1]:
        RRR = lamb_array * RL_ratio
        ax.plot(lamb_array, RRR, "--", color="white")
    ax.axhline(y=R, ls="--", color="white")
    ax.plot(L_list, [R,] * 4, "o", color="white")
    ax.set_xlim(np.min(lamb_array), 12)
    ax.set_ylim(np.min(R_array), np.max(R_array))
    ax.set_yticks([])
    ax.set_xlabel("L/r")
    ax.set_title("Hexagonal Mask")
    # ax.text(10, 10, "$R/r=\\sqrt{7}$", color="white")
    # ax.text(10, 10, "$R/r=\\sqrt{3}$", color="white")
    # ax.text(10, 10, "$R/r=\\sqrt{5}$", color="white")
    # ax.text(10, 10, "$R/r=2$")
    # ax.text(10, 10, "$R/r=1$")

def plot_colorbar(cax, cmap="viridis"):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow([[0, 1], [1, 0]])
    fig.colorbar(im, cax=cax)


def plot_main():
    layout = ff.FigureLayout(template)
    layout.make_mplfigures()

    print("Drawing square heatmap")
    ax = layout.axes[f"square-heatmap"]["axis"]
    plot_square_heatmap(ax, samples=256)

    print("Drawing hex heatmap")
    ax = layout.axes[f"hex-heatmap"]["axis"]
    plot_hex_heatmap(ax, samples=256)

    print("Drawing color bar")
    ax = layout.axes[f"heatmap-colorbar"]["axis"]
    plot_colorbar(ax)

    for i, ratio in enumerate([8 / 1.414, 6.5, 8, 10]):
        print(f"Drawing square heatmap ax {i+1}")
        ax = layout.axes[f"square-{i+1:d}"]["axis"]
        plot_square_P_map(ax, R=8.0, spacing_ratio=ratio)

    for i, ratio in enumerate([8 / 1.732, 6, 8, 10]):
        print(f"Drawing hex heatmap ax {i+1}")
        ax = layout.axes[f"hex-{i+1:d}"]["axis"]
        plot_hex_P_map(ax, R=8.0, spacing_ratio=ratio)

    print("Drawing color bar2")
    ax = layout.axes[f"heatmap-colorbar2"]["axis"]
    plot_colorbar(ax, cmap="magma")
    
    layout.insert_figures("mpl_render_layer", cleartarget=True)
    layout.write_svg(template)

    return

if __name__ == "__main__":
    plot_main()
