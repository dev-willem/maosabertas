//! Modulo Python `maosabertas`: expoe ao treino o mesmo calculo de features
//! do tempo real. Os arrays atravessam como bytes float32 little-endian, para
//! nao depender de crate de numpy.

use maosabertas_core::{
    frame_features, FrameLandmarks, BLENDSHAPE_VALUES, FACE_VALUES, FEATURE_DIM, HAND_VALUES,
    POSE_VALUES,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn floats(name: &str, bytes: &[u8], per_frame: usize, frames: usize) -> PyResult<Vec<f32>> {
    if bytes.len() != frames * per_frame * 4 {
        return Err(PyValueError::new_err(format!(
            "{name}: esperados {frames} quadros de {per_frame} float32, recebidos {} bytes",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Valores do quadro `t`; `floats` ja garantiu o tamanho.
fn frame(v: &[f32], per_frame: usize, t: usize) -> &[f32] {
    &v[t * per_frame..(t + 1) * per_frame]
}

/// Features de T quadros. Cada argumento e o array correspondente do `.npz`
/// da extracao como bytes (`arr.astype("<f4").tobytes()`); `aspect` e
/// largura / altura da imagem. Devolve T x FEATURE_DIM no mesmo formato;
/// quadro sem ombros sai inteiro como NaN.
#[pyfunction]
fn features<'py>(
    py: Python<'py>,
    left_hand: &[u8],
    right_hand: &[u8],
    face: &[u8],
    blendshapes: &[u8],
    pose: &[u8],
    aspect: f32,
) -> PyResult<Bound<'py, PyBytes>> {
    let frames = pose.len() / (POSE_VALUES * 4);
    let left = floats("left_hand", left_hand, HAND_VALUES, frames)?;
    let right = floats("right_hand", right_hand, HAND_VALUES, frames)?;
    let face = floats("face", face, FACE_VALUES, frames)?;
    let blend = floats("blendshapes", blendshapes, BLENDSHAPE_VALUES, frames)?;
    let pose = floats("pose", pose, POSE_VALUES, frames)?;

    let mut out = Vec::with_capacity(frames * FEATURE_DIM * 4);
    for t in 0..frames {
        let lm = FrameLandmarks {
            left_hand: frame(&left, HAND_VALUES, t),
            right_hand: frame(&right, HAND_VALUES, t),
            face: frame(&face, FACE_VALUES, t),
            blendshapes: frame(&blend, BLENDSHAPE_VALUES, t),
            pose: frame(&pose, POSE_VALUES, t),
        };
        let row = frame_features(&lm, aspect)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .unwrap_or_else(|| vec![f32::NAN; FEATURE_DIM]);
        out.extend(row.iter().flat_map(|v| v.to_le_bytes()));
    }
    Ok(PyBytes::new(py, &out))
}

#[pymodule]
fn maosabertas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add("FEATURE_DIM", FEATURE_DIM)?;
    Ok(())
}
