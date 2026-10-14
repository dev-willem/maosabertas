//! Inferencia ONNX: features por quadro -> probabilidades por quadro.
//!
//! Fica fora do nucleo porque o ONNX Runtime nao compila para wasm32; o
//! nucleo continua sem dependencia nativa.
//!
//! Contrato do modelo exportado pelo treino:
//! - uma entrada float32 `[T, FEATURE_DIM]`, com `T` dinamico;
//! - uma saida float32 `[T, C]`, `C` fixo, ja em probabilidades (softmax no
//!   modelo manual, sigmoide por rotulo no nao manual).
//!
//! Os nomes nao importam: usa-se a primeira entrada e a primeira saida.

use maosabertas_core::FEATURE_DIM;
use ort::session::Session;
use ort::value::TensorRef;
use std::fmt;
use std::path::Path;

pub struct Model {
    session: Session,
    output: String,
    classes: usize,
}

impl Model {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let env = ort::init().build()?;
        Self::new(Session::builder(&env)?.commit_from_file(path)?)
    }

    pub fn from_memory(bytes: &[u8]) -> Result<Self, Error> {
        let env = ort::init().build()?;
        Self::new(Session::builder(&env)?.commit_from_memory(bytes)?)
    }

    fn new(session: Session) -> Result<Self, Error> {
        let (Some(input), Some(output)) = (session.inputs().first(), session.outputs().first())
        else {
            return Err(Error::Contract("modelo sem entrada ou sem saida".into()));
        };
        let shape = |o: &ort::value::Outlet| o.dtype().tensor_shape().map(|s| s.to_vec());
        match shape(input).as_deref() {
            Some([_, d]) if *d == FEATURE_DIM as i64 => {}
            other => {
                return Err(Error::Contract(format!(
                    "entrada precisa ser [T, {FEATURE_DIM}], e {other:?}"
                )))
            }
        }
        let classes = match shape(output).as_deref() {
            Some(&[_, c]) if c > 0 => c as usize,
            other => {
                return Err(Error::Contract(format!(
                    "saida precisa ser [T, C] com C fixo, e {other:?}"
                )))
            }
        };
        Ok(Self {
            output: output.name().to_owned(),
            session,
            classes,
        })
    }

    /// Numero de classes da saida.
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// `features`: T quadros em linha (T x FEATURE_DIM). Devolve T x C.
    ///
    /// Recusa NaN: quadro sem referencia (sem ombros) precisa ser tratado
    /// antes, com a mesma regra usada no treino.
    pub fn run(&mut self, features: &[f32]) -> Result<Vec<f32>, Error> {
        if features.len() % FEATURE_DIM != 0 {
            return Err(Error::Input(format!(
                "{} valores nao formam quadros de {FEATURE_DIM}",
                features.len()
            )));
        }
        if let Some(i) = features.iter().position(|v| v.is_nan()) {
            return Err(Error::Input(format!("NaN no quadro {}", i / FEATURE_DIM)));
        }
        let frames = features.len() / FEATURE_DIM;
        if frames == 0 {
            return Ok(Vec::new());
        }
        let input = TensorRef::from_array_view(([frames, FEATURE_DIM], features))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, probs) = outputs[self.output.as_str()].try_extract_tensor::<f32>()?;
        if **shape != [frames as i64, self.classes as i64] {
            return Err(Error::Contract(format!(
                "saida com formato {:?}, esperado [{frames}, {}]",
                &**shape, self.classes
            )));
        }
        Ok(probs.to_vec())
    }
}

#[derive(Debug)]
pub enum Error {
    Ort(ort::Error),
    /// O modelo nao segue o contrato de entrada e saida.
    Contract(String),
    Input(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Ort(e) => write!(f, "onnx runtime: {e}"),
            Error::Contract(m) => write!(f, "contrato do modelo: {m}"),
            Error::Input(m) => write!(f, "entrada invalida: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Ort(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Ort(e)
    }
}
