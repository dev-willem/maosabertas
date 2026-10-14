//! Nucleo de tempo real: dos landmarks as features e das probabilidades por
//! quadro aos itens do protocolo.
//!
//! Nada aqui conhece glosas nem rotulos de Libras: o vocabulario e os
//! rotulos chegam de quem chama, junto com o modelo.

mod ctc;
mod features;
mod frames;
mod scope;
mod transcriber;

pub use ctc::{greedy_decode, CtcDecoder, Token};
pub use features::{
    frame_features, FrameLandmarks, BLENDSHAPE_VALUES, FACE_VALUES, FEATURE_DIM, HAND_VALUES,
    POSE_VALUES,
};
pub use frames::Frames;
pub use scope::{segment, Hysteresis, ScopeSegmenter};
pub use transcriber::{Ending, FrameQuality, Transcriber};

use maosabertas_protocol::{ManualItem, NonManualItem, Span, Utterance, SPEC_VERSION};
use std::fmt;

/// Monta o enunciado de uma janela. Sai sem `translation`: quem traduz e outra camada.
pub fn utterance(
    id: impl Into<String>,
    window: Span,
    manual: Vec<ManualItem>,
    nonmanual: Vec<NonManualItem>,
) -> Utterance {
    Utterance {
        spec_version: SPEC_VERSION.into(),
        utterance_id: id.into(),
        t_start: window[0],
        t_end: window[1],
        manual,
        nonmanual,
        quality: None,
        translation: None,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// O tamanho do buffer nao e multiplo do numero de classes.
    Shape {
        len: usize,
        classes: usize,
    },
    Fps(f64),
    Start(f64),
    /// Valor fora de `[0, 1]`: provavelmente logit em vez de probabilidade.
    Probability(f32),
    /// Vocabulario ou lista de rotulos com tamanho diferente do numero de classes.
    Classes {
        expected: usize,
        got: usize,
    },
    Blank(usize),
    Hysteresis {
        enter: f32,
        exit: f32,
    },
    /// Parte dos landmarks com numero de valores diferente do esperado.
    Landmarks {
        part: &'static str,
        expected: usize,
        got: usize,
    },
    /// Proporcao largura / altura da imagem invalida.
    Aspect(f32),
    /// Pausa ou duracao maxima que nao e numero finito e positivo.
    Ending(Ending),
    /// Blocos do mesmo trecho com numero de quadros diferente.
    FrameCount {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape { len, classes } => {
                write!(f, "buffer de {len} valores nao divide em {classes} classes")
            }
            Error::Fps(v) => write!(f, "fps invalido: {v}"),
            Error::Start(v) => write!(f, "instante inicial invalido: {v}"),
            Error::Probability(v) => write!(f, "probabilidade fora de [0, 1]: {v}"),
            Error::Classes { expected, got } => {
                write!(f, "esperadas {expected} classes, recebidas {got}")
            }
            Error::Blank(i) => write!(f, "indice do branco fora do vocabulario: {i}"),
            Error::Hysteresis { enter, exit } => {
                write!(
                    f,
                    "histerese invalida: exige 0 <= saida ({exit}) <= entrada ({enter}) <= 1"
                )
            }
            Error::Landmarks {
                part,
                expected,
                got,
            } => write!(f, "{part}: esperados {expected} valores, recebidos {got}"),
            Error::Aspect(v) => write!(f, "proporcao da imagem invalida: {v}"),
            Error::Ending(e) => write!(
                f,
                "fechamento invalido: pausa ({}) e duracao maxima ({}) precisam ser finitas e positivas",
                e.pause, e.max_duration
            ),
            Error::FrameCount { expected, got } => {
                write!(f, "esperados {expected} quadros, recebidos {got}")
            }
        }
    }
}

impl std::error::Error for Error {}
