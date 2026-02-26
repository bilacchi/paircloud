import matplotlib
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import paircloud

# Use a non-interactive backend for testing plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_rust_hello():
    """Basic health check for the Rust extension."""
    # the exact import depends on the __init__ structure, we can import from the built file
    from paircloud._paircloud_rs import hello

    result = hello()
    assert 'Paircloud' in result


# Settings for hypothesis to not take too long in CI
@settings(max_examples=50, deadline=None)
@given(
    data=arrays(
        np.float64,
        st.integers(min_value=3, max_value=500),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
    eval_points=arrays(
        np.float64,
        st.integers(min_value=1, max_value=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)
def test_kde_properties(data, eval_points):
    """Property-based test for the KDE implementation ensuring it returns valid shapes and non-negative values."""
    # Ensure data has some variance; if all points are identical, std is 0 and bandwidth handles it,
    # but let's test the robust calculation
    densities = paircloud.calculate_kde(data, eval_points)

    # Output should match the shape of eval_points
    assert densities.shape == eval_points.shape

    # Densities probability should always be >= 0
    assert np.all(densities >= 0), 'Densities must be non-negative'


def test_kde_known_distribution():
    """Test KDE mathematically against a known continuous distribution to verify the shape."""
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1, size=1000)
    eval_points = np.linspace(-3, 3, 100)
    densities = paircloud.calculate_kde(data, eval_points)

    # Peak for standard normal should be near 0
    peak_idx = np.argmax(densities)
    peak_x = eval_points[peak_idx]
    assert abs(peak_x) < 0.5, 'Peak of the distribution is excessively shifted'

    # Density should taper off correctly at both edges
    assert densities[0] < densities[peak_idx], 'Tail density should be lower than peak'
    assert densities[-1] < densities[peak_idx], 'Tail density should be lower than peak'


def test_kde_nan_handling():
    """Test that the python wrapper gracefully drops NaNs before computing KDE via Rust."""
    data = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    eval_points = np.array([2.5])

    densities = paircloud.calculate_kde(data, eval_points)
    # The output should not contain NaNs
    assert not np.isnan(densities).any()
    assert densities[0] >= 0


def test_hdi_threshold_integration():
    """Test the fast single-pass Rust calculation for KDE with thresholds."""
    data = np.random.normal(loc=0, scale=1, size=100)
    eval_points, density, thresholds = paircloud.calcs.calculate_kde_with_hdi(
        data, intervals=[50.0, 95.0], grid_points=200
    )

    # Assert grid resolution
    assert len(eval_points) == 200
    assert len(density) == 200

    # Assert thresholds corresponding to the reverse sorted interval sizes (95, then 50)
    assert len(thresholds) == 2
    assert thresholds[0] < thresholds[1], (
        'The 95% HDI density boundary should be lower than the 50% boundary.'
    )


def test_compute_stack_offsets_validation():
    """Verify Rust array mapping correctly computes strictly typed dot stacking alignments."""
    data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    offsets = paircloud.calcs.compute_stack_offsets(
        data, position=0.0, width=0.8, side='both', bins=50
    )

    assert offsets.shape == (6,)

    # In a given bin of 3 items (the [1.0, 1.0, 1.0] clump), the offset should alternate
    # We don't hardcode the exact values, but assert they expand around the 0.0 baseline
    assert np.any(offsets > 0)
    assert np.any(offsets < 0)


def test_point_density_n_body_computation():
    """Test that N-body point transparency matrices map back identically shaped outputs."""
    data = np.random.normal(0, 1, 10)
    eval_pts = np.random.normal(0, 1, 15)
    densities = paircloud.calcs.calculate_point_densities(data, eval_pts)

    assert densities.shape == (15,)
    assert not np.isnan(densities).any()


@pytest.mark.parametrize(
    'plot_func',
    [
        paircloud.faded_dotplot,
        paircloud.shadeplot,
        paircloud.raincloud,
        paircloud.paired_raincloud,
    ],
)
@settings(max_examples=10, deadline=None)
@given(
    data=arrays(
        np.float64,
        st.integers(min_value=5, max_value=100),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
)
def test_plot_functions_execute(plot_func, data):
    """Test that the main plotting algorithms construct matplotlib objects without crashing."""
    fig, ax = plt.subplots()
    try:
        if plot_func == paircloud.paired_raincloud:
            data2 = data + np.random.normal(0, 1, size=len(data))
            plot_func(data, data2, ax=ax)
        else:
            plot_func(data, ax=ax)
    finally:
        plt.close(fig)
