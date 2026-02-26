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

#[pymodule]
fn _paircloud_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(calc_kde, m)?)?;
    Ok(())
}
