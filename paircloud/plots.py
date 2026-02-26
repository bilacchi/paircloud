import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .calcs import (
    calculate_kde_with_hdi,
    calculate_point_densities,
    compute_stack_offsets,
)


def _get_density_and_thresholds(data, intervals, grid_points=200, bandwidth=None):
    if len(data) == 0:
        return np.array([]), np.array([]), []

    return calculate_kde_with_hdi(data, intervals, grid_points=grid_points, bandwidth=bandwidth)


def _draw_stepped_density(
    ax,
    eval_points,
    raw_density,
    thresholds,
    position,
    width_scale,
    color,
    orientation,
    direction=1,
    offset=0,
):
    if len(eval_points) == 0:
        return

    if raw_density.max() > 0:
        density_plot = (raw_density / raw_density.max()) * width_scale
    else:
        density_plot = raw_density

    base_alpha = 0.15
    alphas = [0.4, 0.8]

    pos_base = np.full_like(eval_points, float(position + offset))
    pos_top = pos_base + direction * density_plot

    # Use fill_between / fill_betweenx with where condition for steps
    if orientation == 'h':
        ax.fill_between(
            eval_points,
            pos_base,
            pos_top,
            where=raw_density > 0,
            color=color,
            alpha=base_alpha,
            lw=0,
        )
        for i, thresh in enumerate(thresholds):
            alpha = alphas[i] if i < len(alphas) else 0.85
            ax.fill_between(
                eval_points,
                pos_base,
                pos_top,
                where=raw_density >= thresh,
                color=color,
                alpha=alpha,
                lw=0,
            )
    else:
        ax.fill_betweenx(
            eval_points,
            pos_base,
            pos_top,
            where=raw_density > 0,
            color=color,
            alpha=base_alpha,
            lw=0,
        )
        for i, thresh in enumerate(thresholds):
            alpha = alphas[i] if i < len(alphas) else 0.85
            ax.fill_betweenx(
                eval_points,
                pos_base,
                pos_top,
                where=raw_density >= thresh,
                color=color,
                alpha=alpha,
                lw=0,
            )


def _get_alphas_from_density(data, min_alpha=0.15, max_alpha=1.0):
    """
    Computes KDE density for each point natively in Rust and maps it to an alpha value.
    Points in the dense regions get max_alpha, points in tails get min_alpha.
    """
    densities = calculate_point_densities(data, data)
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
    side='positive',
    show_mean=False,
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
        side: "positive", "negative", or "both" (direction of dot placement)
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
        if side in ('positive', 'top', 'right'):
            dist = np.random.uniform(0, width / 2, size=n)
        elif side in ('negative', 'bottom', 'left'):
            dist = np.random.uniform(-width / 2, 0, size=n)
        else:
            dist = np.random.uniform(-width / 2, width / 2, size=n)
        offsets = position + dist

        if orientation == 'h':
            ax.scatter(data, offsets, c=colors, s=dot_size if dot_size else 20)
        else:
            ax.scatter(offsets, data, c=colors, s=dot_size if dot_size else 20)
    else:
        # Strict dot stacking computed blazingly fast in Rust
        offsets = compute_stack_offsets(data, position, width, side, bins=min(50, n))

        if orientation == 'h':
            ax.scatter(data, offsets, c=colors, s=dot_size if dot_size else 20)
        else:
            ax.scatter(offsets, data, c=colors, s=dot_size if dot_size else 20)

    if show_mean:
        mean_val = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n) if n > 1 else 0
        ci_95 = 1.96 * sem

        sign = 1 if side in ('negative', 'bottom', 'left') else -1
        if side in ('positive', 'negative', 'top', 'bottom', 'left', 'right'):
            mean_pos = position + sign * width * 0.1
            text_pos = position + sign * width * 0.2
        else:
            mean_pos = position - width * 0.3
            text_pos = position - width * 0.45

        pad_pos = text_pos + sign * width * 0.05

        if orientation == 'h':
            ax.errorbar(
                mean_val,
                mean_pos,
                xerr=ci_95,
                fmt='o',
                color='black',
                linewidth=1.5,
                markersize=4,
                capsize=3,
                zorder=10,
            )
            ax.text(
                mean_val,
                text_pos,
                f'{mean_val:.1f}',
                color='black',
                fontsize=8,
                ha='center',
                va='center',
                zorder=10,
            )
            ax.plot(mean_val, pad_pos, color='none', alpha=0)
        else:
            ax.errorbar(
                mean_pos,
                mean_val,
                yerr=ci_95,
                fmt='o',
                color='black',
                linewidth=1.5,
                markersize=4,
                capsize=3,
                zorder=10,
            )
            ax.text(
                text_pos,
                mean_val,
                f'{mean_val:.1f}',
                color='black',
                fontsize=8,
                ha='center',
                va='center',
                zorder=10,
            )
            ax.plot(pad_pos, mean_val, color='none', alpha=0)

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
    density_intervals=(50, 95),
):
    """
    Creates a shadeplot: A half-violin density slab with a faded dotplot overlaid.
    """
    if ax is None:
        ax = plt.gca()

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # 1. Draw Density Slab with HDI
    eval_points, densities, thresholds = _get_density_and_thresholds(
        data, density_intervals, bandwidth=bandwidth
    )
    _draw_stepped_density(
        ax, eval_points, densities, thresholds, position, width / 2, color, orientation
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
        side='positive',
    )  # Jitter works best overlaid on violins

    return ax


def raincloud(
    data,
    ax=None,
    color='C0',
    orientation='h',
    position=0,
    width=0.8,
    dot_size=None,
    density_intervals=(50, 95),
):
    """
    Classic raincloud plot: Half-violin, boxplot, and jittered dots below.
    """
    if ax is None:
        ax = plt.gca()

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # 1. Half Violin (The Cloud)
    eval_points, densities, thresholds = _get_density_and_thresholds(data, density_intervals)
    _draw_stepped_density(
        ax, eval_points, densities, thresholds, position, width / 2, color, orientation, offset=0.1
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
        side='both',
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
    density_intervals=(50, 95),
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

    # 1. Violins
    eval_points1, dens1, t1 = _get_density_and_thresholds(data1, density_intervals)
    eval_points2, dens2, t2 = _get_density_and_thresholds(data2, density_intervals)

    _draw_stepped_density(
        ax, eval_points1, dens1, t1, pos1, width / 2.5, c1, orientation, direction=-1, offset=-0.25
    )
    _draw_stepped_density(
        ax, eval_points2, dens2, t2, pos2, width / 2.5, c2, orientation, direction=1, offset=0.25
    )

    # 2. Boxplots
    vert = orientation == 'v'
    ax.boxplot(
        [data1],
        positions=[pos1 - 0.15],
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
        positions=[pos2 + 0.15],
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
    offsets = np.random.uniform(-width * 0.05, width * 0.05, size=n)
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
