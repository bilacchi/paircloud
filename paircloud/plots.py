import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .calcs import calculate_kde


def _get_alphas_from_density(data, min_alpha=0.15, max_alpha=1.0):
    """
    Computes KDE density for each point and maps it to an alpha value.
    Points in the dense regions get max_alpha, points in tails get min_alpha.
    """
    densities = calculate_kde(data, data)
    d_min, d_max = densities.min(), densities.max()
    if d_max == d_min:
        return np.full_like(densities, max_alpha)

    normalized = (densities - d_min) / (d_max - d_min)
    return min_alpha + normalized * (max_alpha - min_alpha)


def _get_alphas_from_quantiles(data, min_alpha=0.15, max_alpha=1.0):
    """
    Fades points based on their quantile distance from the median.
    Median gets max_alpha, extremes get min_alpha.
    """
    median = np.median(data)
    dist_from_median = np.abs(data - median)
    max_dist = dist_from_median.max()
    if max_dist == 0:
        return np.full_like(data, max_alpha)

    normalized_dist = dist_from_median / max_dist
    # invert so 0 dist = max alpha, 1 dist = min alpha
    alphas = max_alpha - normalized_dist * (max_alpha - min_alpha)
    return alphas


def faded_dotplot(
    data,
    ax=None,
    color='C0',
    min_alpha=0.15,
    max_alpha=1.0,
    orientation='h',
    position=0,
    width=0.8,
    dot_size=None,
    fade_method='density',
    jitter=False,
):
    """
    Creates a faded dotplot.

    Args:
        data: 1D array of values
        ax: matplotlib axes
        color: base color of dots
        min_alpha: minimum transparency for tail data
        max_alpha: maximum transparency for peak data
        orientation: "h" (horizontal, data on x-axis) or "v" (vertical, data on y-axis)
        position: offset position on the categorical axis
        width: total width allocated for stacking/jitter
        dot_size: visual size of dots (if None, auto-calculated)
        fade_method: "density" or "quantile"
        jitter: if True, applies random jitter instead of strict stacking
    """
    if ax is None:
        ax = plt.gca()

    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)

    if fade_method == 'density':
        alphas = _get_alphas_from_density(data, min_alpha, max_alpha)
    else:
        alphas = _get_alphas_from_quantiles(data, min_alpha, max_alpha)

    # rgba colors
    base_rgb = mcolors.to_rgb(color)
    colors = np.zeros((n, 4))
    colors[:, :3] = base_rgb
    colors[:, 3] = alphas

    if jitter:
        # Simple jitter
        offsets = position + np.random.uniform(-width / 2, width / 2, size=n)
        if orientation == 'h':
            ax.scatter(data, offsets, c=colors, s=dot_size if dot_size else 20)
        else:
            ax.scatter(offsets, data, c=colors, s=dot_size if dot_size else 20)
    else:
        # Strict dot stacking (simple O(N^2) binning approach for visualization)
        # We bin the data depending on dot_size logic, but a fast heuristic
        # is digitizing over a grid.
        grid_bins = min(50, n)
        hist, bin_edges = np.histogram(data, bins=grid_bins)

        # We need to map each data point to its bin and assign a stacking height
        bin_indices = np.digitize(data, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, grid_bins - 1)

        stack_counts = np.zeros(grid_bins)
        offsets = np.zeros(n)

        # Max height scaling
        max_stack = hist.max()
        scale_factor = (width / 2) / max(1, max_stack)

        for i in range(n):
            b = bin_indices[i]
            # stagger them from the baseline
            offsets[i] = position + (stack_counts[b]) * scale_factor
            stack_counts[b] += 1

        if orientation == 'h':
            ax.scatter(data, offsets, c=colors, s=dot_size if dot_size else 20)
        else:
            ax.scatter(offsets, data, c=colors, s=dot_size if dot_size else 20)

    return ax


def shadeplot(
    data,
    ax=None,
    color='C0',
    min_alpha=0.15,
    max_alpha=1.0,
    orientation='h',
    position=0,
    width=0.8,
    dot_size=None,
    bandwidth=None,
):
    """
    Creates a shadeplot: A half-violin density slab with a faded dotplot overlaid.
    """
    if ax is None:
        ax = plt.gca()

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # 1. Draw Density Slab (Half-violin)
    grid_size = 200
    eval_points = np.linspace(data.min() - np.std(data), data.max() + np.std(data), grid_size)
    densities = calculate_kde(data, eval_points, bandwidth=bandwidth)

    # Normalize density to fit within width
    if densities.max() > 0:
        densities = (densities / densities.max()) * (width / 2)

    # To create a shaded slab, we use PolyCollection or fill_between with a gradient.
    # matplotlib fill_between doesn't natively support alpha gradients along an axis easily,
    # so we'll draw a solid slightly-transparent half violin as the "shade".
    base_alpha = max(min_alpha * 2, 0.4)
    if orientation == 'h':
        # Horizontal implies data on X, density on Y
        ax.fill_between(
            eval_points,
            position,
            position + densities,
            color=color,
            alpha=base_alpha,
            lw=0,
        )
    else:
        # Vertical implies data on Y, density on X
        ax.fill_betweenx(
            eval_points,
            position,
            position + densities,
            color=color,
            alpha=base_alpha,
            lw=0,
        )

    # 2. Add faded dots on top, positioned at the baseline or jittered slightly
    # For a classic shadeplot, dots are plotted inside the density or just below it.
    faded_dotplot(
        data,
        ax=ax,
        color=color,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        orientation=orientation,
        position=position,
        width=width,
        dot_size=dot_size,
        jitter=True,
    )  # Jitter works best overlaid on violins

    return ax


def raincloud(data, ax=None, color='C0', orientation='h', position=0, width=0.8, dot_size=None):
    """
    Classic raincloud plot: Half-violin, boxplot, and jittered dots below.
    """
    if ax is None:
        ax = plt.gca()

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # 1. Half Violin (The Cloud)
    grid_size = 200
    eval_points = np.linspace(data.min() - np.std(data), data.max() + np.std(data), grid_size)
    densities = calculate_kde(data, eval_points)
    if densities.max() > 0:
        densities = (densities / densities.max()) * (width / 2)

    if orientation == 'h':
        ax.fill_between(
            eval_points,
            position + 0.1,
            position + 0.1 + densities,
            color=color,
            alpha=0.5,
            lw=0,
        )
    else:
        ax.fill_betweenx(
            eval_points,
            position + 0.1,
            position + 0.1 + densities,
            color=color,
            alpha=0.5,
            lw=0,
        )

    # 2. Boxplot (The Umbrella)
    vert = orientation == 'v'
    boxplot_pos = position
    ax.boxplot(
        [data],
        positions=[boxplot_pos],
        widths=width * 0.1,
        patch_artist=True,
        showfliers=False,
        orientation='vertical' if vert else 'horizontal',
        boxprops=dict(facecolor=color, alpha=0.6, color=color),
        capprops=dict(color=color),
        whiskerprops=dict(color=color),
        medianprops=dict(color='black', linewidth=1.5),
    )

    # 3. Jittered Dots (The Rain)
    # Put rain below the boxplot
    rain_pos = position - 0.2
    faded_dotplot(
        data,
        ax=ax,
        color=color,
        min_alpha=0.15,
        max_alpha=0.6,
        orientation=orientation,
        position=rain_pos,
        width=width * 0.2,
        dot_size=dot_size,
        jitter=True,
        fade_method='density',
    )

    return ax


def paired_raincloud(
    data1,
    data2,
    ax=None,
    colors=('C0', 'C1'),
    orientation='v',
    positions=(1, 2),
    width=0.8,
    dot_size=None,
    line_color='gray',
    line_alpha=0.3,
):
    """
    Creates a paired raincloud plot for repeated measures, connecting data points.
    """
    if ax is None:
        ax = plt.gca()

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # Filter out pairs where either is NaN
    valid = ~np.isnan(data1) & ~np.isnan(data2)
    data1 = data1[valid]
    data2 = data2[valid]
    n = len(data1)

    pos1, pos2 = positions
    c1, c2 = colors

    # Generate the KDEs
    eval_points1 = np.linspace(data1.min() - np.std(data1), data1.max() + np.std(data1), 200)
    dens1 = calculate_kde(data1, eval_points1)
    if dens1.max() > 0:
        dens1 = (dens1 / dens1.max()) * (width / 2.5)

    eval_points2 = np.linspace(data2.min() - np.std(data2), data2.max() + np.std(data2), 200)
    dens2 = calculate_kde(data2, eval_points2)
    if dens2.max() > 0:
        dens2 = (dens2 / dens2.max()) * (width / 2.5)

    # 1. Violins
    if orientation == 'h':
        ax.fill_between(eval_points1, pos1 - 0.1, pos1 - 0.1 - dens1, color=c1, alpha=0.5, lw=0)
        ax.fill_between(eval_points2, pos2 + 0.1, pos2 + 0.1 + dens2, color=c2, alpha=0.5, lw=0)
    else:
        ax.fill_betweenx(eval_points1, pos1 - 0.1, pos1 - 0.1 - dens1, color=c1, alpha=0.5, lw=0)
        ax.fill_betweenx(eval_points2, pos2 + 0.1, pos2 + 0.1 + dens2, color=c2, alpha=0.5, lw=0)

    # 2. Boxplots
    vert = orientation == 'v'
    ax.boxplot(
        [data1],
        positions=[pos1 - 0.05],
        widths=width * 0.1,
        patch_artist=True,
        showfliers=False,
        orientation='vertical' if vert else 'horizontal',
        boxprops=dict(facecolor=c1, alpha=0.6, color=c1),
        capprops=dict(color=c1),
        whiskerprops=dict(color=c1),
        medianprops=dict(color='black', linewidth=1.5),
    )

    ax.boxplot(
        [data2],
        positions=[pos2 + 0.05],
        widths=width * 0.1,
        patch_artist=True,
        showfliers=False,
        orientation='vertical' if vert else 'horizontal',
        boxprops=dict(facecolor=c2, alpha=0.6, color=c2),
        capprops=dict(color=c2),
        whiskerprops=dict(color=c2),
        medianprops=dict(color='black', linewidth=1.5),
    )

    # 3. Dots and Lines
    alphas1 = _get_alphas_from_density(data1, 0.15, 0.8)
    rgb1 = np.zeros((n, 4))
    rgb1[:, :3] = mcolors.to_rgb(c1)
    rgb1[:, 3] = alphas1

    alphas2 = _get_alphas_from_density(data2, 0.15, 0.8)
    rgb2 = np.zeros((n, 4))
    rgb2[:, :3] = mcolors.to_rgb(c2)
    rgb2[:, 3] = alphas2

    # Use consistent jitter displacement
    offsets = np.random.uniform(-width * 0.1, width * 0.1, size=n)
    jit1 = pos1 + offsets
    jit2 = pos2 + offsets

    for i in range(n):
        if orientation == 'h':
            ax.plot(
                [data1[i], data2[i]],
                [jit1[i], jit2[i]],
                color=line_color,
                alpha=line_alpha,
                zorder=1,
            )
        else:
            ax.plot(
                [jit1[i], jit2[i]],
                [data1[i], data2[i]],
                color=line_color,
                alpha=line_alpha,
                zorder=1,
            )

    if orientation == 'h':
        ax.scatter(data1, jit1, c=rgb1, s=dot_size if dot_size else 20, zorder=2)
        ax.scatter(data2, jit2, c=rgb2, s=dot_size if dot_size else 20, zorder=2)
    else:
        ax.scatter(jit1, data1, c=rgb1, s=dot_size if dot_size else 20, zorder=2)
        ax.scatter(jit2, data2, c=rgb2, s=dot_size if dot_size else 20, zorder=2)

    return ax
