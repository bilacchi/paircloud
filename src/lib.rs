use pyo3::prelude::*;
use numpy::ndarray::{Array1, Axis, Zip};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use std::f64::consts::PI;

/// Helper to calculate the Gaussian kernel.
#[inline(always)]
fn gaussian_kernel(v: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * v * v).exp()
}

/// Calculate bandwidth using Silverman's rule of thumb.
fn compute_bandwidth(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 1.0;
    }

    // We need standard deviation
    let mean = data.mean().unwrap_or(0.0);
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();

    // Estimate IQR (Interquartile Range)
    let mut sorted_data = data.to_vec();
    // Using simple sort since floats can have NaNs. We filter NaNs out ideally.
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = sorted_data[(n * 0.25) as usize];
    let q3 = sorted_data[(n * 0.75) as usize];
    let iqr = q3 - q1;

    let iqr_scaled = iqr / 1.34;
    let min_spread = if std < iqr_scaled || iqr == 0.0 { std } else { iqr_scaled };
    let min_spread = if min_spread == 0.0 { 1.0 } else { min_spread };

    0.9 * min_spread * n.powf(-0.2)
}

/// Evaluate the KDE on a set of target points.
#[pyfunction]
#[pyo3(signature = (data, eval_points, bandwidth=None))]
fn calc_kde<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    eval_points: PyReadonlyArray1<'py, f64>,
    bandwidth: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_array = data.as_array();
    let eval_array = eval_points.as_array();

    let bw = bandwidth.unwrap_or_else(|| compute_bandwidth(&data_array.to_owned()));
    let n = data_array.len() as f64;

    let mut densities = Array1::<f64>::zeros(eval_array.len());

    // O(M*N) calculation. For large N, we might want to bin or use FFT,
    // but for typical dataset sizes in data vis (<10k), direct evaluation
    // in Rust is extremely fast.
    Zip::from(&mut densities).and(&eval_array).for_each(|d, &x| {
        let sum: f64 = data_array.iter().map(|&pt| {
            gaussian_kernel((x - pt) / bw)
        }).sum();
        *d = sum / (n * bw);
    });

    Ok(densities.into_pyarray_bound(py))
}

/// Test pyfunction
#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Hello from Paircloud Rust backend!".to_string())
}

/// Calculate KDE and find standard HDI interval thresholds simultaneously in one pass.
#[pyfunction]
#[pyo3(signature = (data, intervals, grid_points=200, bandwidth=None))]
fn calc_kde_with_hdi<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    intervals: Vec<f64>,
    grid_points: usize,
    bandwidth: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Vec<f64>)> {
    let data_array = data.as_array();
    let n = data_array.len() as f64;

    if n == 0.0 {
        return Ok((
            Array1::<f64>::zeros(0).into_pyarray_bound(py),
            Array1::<f64>::zeros(0).into_pyarray_bound(py),
            vec![],
        ));
    }

    // 1. Calculate Grid Points
    let std = data_array.std(0.0);
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for &val in data_array.iter() {
        if val < min_val { min_val = val; }
        if val > max_val { max_val = val; }
    }

    let grid_min = min_val - std;
    let grid_max = max_val + std;
    let step = if grid_points > 1 { (grid_max - grid_min) / (grid_points as f64 - 1.0) } else { 0.0 };

    let mut eval_points = Array1::<f64>::zeros(grid_points);
    for i in 0..grid_points {
        eval_points[i] = grid_min + (i as f64) * step;
    }

    // 2. Compute Density natively
    let bw = bandwidth.unwrap_or_else(|| compute_bandwidth(&data_array.to_owned()));
    let mut densities = Array1::<f64>::zeros(grid_points);

    Zip::from(&mut densities).and(&eval_points).for_each(|d, &x| {
        let sum: f64 = data_array.iter().map(|&pt| gaussian_kernel((x - pt) / bw)).sum();
        *d = sum / (n * bw);
    });

    // 3. HDI Thresholds
    let mut thresholds = Vec::with_capacity(intervals.len());
    if !intervals.is_empty() {
        let mut sorted_densities = densities.to_vec();
        sorted_densities.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Cumulative sum integration
        let mut cumulative_mass = Vec::with_capacity(grid_points);
        let mut accum = 0.0;
        for &d in &sorted_densities {
            accum += d * step;
            cumulative_mass.push(accum);
        }

        let total_mass = *cumulative_mass.last().unwrap_or(&1.0);
        let norm_mass = if total_mass > 0.0 { total_mass } else { 1.0 };
        for m in &mut cumulative_mass {
            *m /= norm_mass;
        }

        let mut sorted_intervals = intervals.clone();
        sorted_intervals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        for interval in sorted_intervals {
            let target = interval / 100.0;
            let mut found_idx = sorted_densities.len() - 1;

            for (i, &mass) in cumulative_mass.iter().enumerate() {
                if mass >= target {
                    found_idx = i;
                    break;
                }
            }
            thresholds.push(sorted_densities[found_idx]);
        }
    }

    Ok((
        eval_points.into_pyarray_bound(py),
        densities.into_pyarray_bound(py),
        thresholds,
    ))
}

/// Computes the exact KDE point densities evaluating N x N points
/// To avoid Python loop and memory boundary copies
#[pyfunction]
#[pyo3(signature = (data_points, eval_points, bandwidth=None))]
fn calc_point_densities<'py>(
    py: Python<'py>,
    data_points: PyReadonlyArray1<'py, f64>,
    eval_points: PyReadonlyArray1<'py, f64>,
    bandwidth: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Falls back to calc_kde directly since we optimized it as O(M*N) in Rust vs Python
    calc_kde(py, data_points, eval_points, bandwidth)
}

/// Computes dotplot positioning stack offsets directly on the 1D Array.
#[pyfunction]
#[pyo3(signature = (data, position, width, side, bins=50))]
fn compute_stack_offsets<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    position: f64,
    width: f64,
    side: &str,
    bins: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_array = data.as_array();
    let n = data_array.len();

    let mut offsets = Array1::<f64>::zeros(n);
    if n == 0 {
        return Ok(offsets.into_pyarray_bound(py));
    }

    // manual histogram
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for &val in data_array.iter() {
        if val < min_val { min_val = val; }
        if val > max_val { max_val = val; }
    }

    // Safety buffer to prevent floating point out of bounds indexing
    let bin_width = if max_val > min_val { (max_val - min_val) / (bins as f64) } else { 1.0 };
    let mut stack_counts = vec![0; bins + 1];
    let mut bin_indices = vec![0; n];

    for i in 0..n {
        let val = data_array[i];
        let mut b = ((val - min_val) / bin_width) as usize;
        if b >= bins { b = bins - 1; }
        bin_indices[i] = b;
        stack_counts[b] += 1;
    }

    let max_stack = *stack_counts.iter().max().unwrap_or(&1);
    let max_stack_f64 = if max_stack > 0 { max_stack as f64 } else { 1.0 };
    let scale_factor = (width / 2.0) / max_stack_f64;

    stack_counts.fill(0);

    for i in 0..n {
        let b = bin_indices[i];
        let count = stack_counts[b];

        let offset = match side {
            "positive" | "top" | "right" => position + (count as f64) * scale_factor,
            "negative" | "bottom" | "left" => position - (count as f64) * scale_factor,
            _ => {
                let s_val = if count % 2 == 0 { 1.0 } else { -1.0 };
                let step = ((count + 1) / 2) as f64;
                position + s_val * step * scale_factor
            }
        };
        offsets[i] = offset;
        stack_counts[b] += 1;
    }

    Ok(offsets.into_pyarray_bound(py))
}


#[pymodule]
fn _paircloud_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(calc_kde, m)?)?;
    m.add_function(wrap_pyfunction!(calc_kde_with_hdi, m)?)?;
    m.add_function(wrap_pyfunction!(calc_point_densities, m)?)?;
    m.add_function(wrap_pyfunction!(compute_stack_offsets, m)?)?;
    Ok(())
}
