//! Nucleo de tempo real: das probabilidades por quadro aos itens do protocolo.
//!
//! Nada aqui conhece glosas nem rotulos de Libras: o vocabulario e os
//! rotulos chegam de quem chama, junto com o modelo.

mod ctc;
mod frames;
mod scope;

pub use ctc::{greedy_decode, Token};
pub use frames::Frames;
pub use scope::{segment, Hysteresis};

use std::fmt;

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
        }
    }
}

impl std::error::Error for Error {}
