//! Modulo Python `maosabertas`: expoe ao treino o mesmo calculo de features
//! do tempo real. Os arrays atravessam como bytes float32 little-endian, para
//! nao depender de crate de numpy.

use maosabertas_core::phonology::{self, HANDSHAPE_DIM};
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

/// Regra do nucleo para quadros sem referencia (`maosabertas_core::fill_gaps`)
/// sobre a saida de `features`: T x FEATURE_DIM, linha NaN = quadro sem
/// referencia. Devolve o mesmo formato, sem NaN.
#[pyfunction]
fn fill_gaps<'py>(
    py: Python<'py>,
    features: &[u8],
    max_gap: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let frames = features.len() / (FEATURE_DIM * 4);
    let feats = floats("features", features, FEATURE_DIM, frames)?;
    let out = maosabertas_core::fill_gaps(&feats, max_gap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(PyBytes::new(py, &out))
}

/// Descritor de configuracao de mao (`maosabertas_core::phonology::handshape`) de T
/// quadros de uma mao: bytes float32 do array `(T, 21, 3)` do `.npz` e `aspect` =
/// largura / altura da imagem. Devolve T x HANDSHAPE_DIM float32; quadro sem a mao
/// (ou com mao degenerada) sai inteiro como NaN.
#[pyfunction]
fn handshape<'py>(py: Python<'py>, hand: &[u8], aspect: f32) -> PyResult<Bound<'py, PyBytes>> {
    if !(aspect.is_finite() && aspect > 0.0) {
        return Err(PyValueError::new_err(format!(
            "proporcao da imagem invalida: {aspect}"
        )));
    }
    let frames = hand.len() / (HAND_VALUES * 4);
    let hand = floats("hand", hand, HAND_VALUES, frames)?;
    let mut out = Vec::with_capacity(frames * HANDSHAPE_DIM * 4);
    for t in 0..frames {
        let mut points = [[0.0; 3]; 21];
        for (p, c) in points
            .iter_mut()
            .zip(frame(&hand, HAND_VALUES, t).chunks_exact(3))
        {
            // Mesma escala nos tres eixos: o z do MediaPipe tem a escala de x.
            *p = [c[0] * aspect, c[1], c[2] * aspect];
        }
        let row = phonology::handshape(&points).unwrap_or([f32::NAN; HANDSHAPE_DIM]);
        out.extend(row.iter().flat_map(|v| v.to_le_bytes()));
    }
    Ok(PyBytes::new(py, &out))
}

/// Quadros de "segurar" (`maosabertas_core::phonology::is_hold`) das duas maos: T x 2
/// bytes, 1 onde a mao (esquerda, direita) esta no espaco de sinalizacao e parada
/// desde o quadro anterior. Entradas como em `features`; `fps` da o intervalo entre
/// quadros.
#[pyfunction]
fn holds<'py>(
    py: Python<'py>,
    left_hand: &[u8],
    right_hand: &[u8],
    pose: &[u8],
    aspect: f32,
    fps: f32,
) -> PyResult<Bound<'py, PyBytes>> {
    if !(fps.is_finite() && fps > 0.0) {
        return Err(PyValueError::new_err(format!("fps invalido: {fps}")));
    }
    let frames = pose.len() / (POSE_VALUES * 4);
    let left = floats("left_hand", left_hand, HAND_VALUES, frames)?;
    let right = floats("right_hand", right_hand, HAND_VALUES, frames)?;
    let pose = floats("pose", pose, POSE_VALUES, frames)?;

    let mut out = Vec::with_capacity(frames * 2);
    let mut prev: phonology::Hands = [None, None];
    for t in 0..frames {
        let cur = phonology::hands(
            frame(&left, HAND_VALUES, t),
            frame(&right, HAND_VALUES, t),
            frame(&pose, POSE_VALUES, t),
            aspect,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        for (p, c) in prev.iter().zip(&cur) {
            let hold = c
                .as_ref()
                .is_some_and(|c| phonology::is_hold(p.as_ref(), c, 1.0 / fps));
            out.push(u8::from(hold));
        }
        prev = cur;
    }
    Ok(PyBytes::new(py, &out))
}

#[pymodule]
fn maosabertas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_function(wrap_pyfunction!(fill_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(handshape, m)?)?;
    m.add_function(wrap_pyfunction!(holds, m)?)?;
    m.add("FEATURE_DIM", FEATURE_DIM)?;
    m.add("HANDSHAPE_DIM", HANDSHAPE_DIM)?;
    Ok(())
}
