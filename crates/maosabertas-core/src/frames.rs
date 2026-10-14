use crate::Error;
use maosabertas_protocol::Span;

/// Probabilidades por quadro em linha (quadro x classe), como saem do tensor ONNX.
#[derive(Debug, Clone, Copy)]
pub struct Frames<'a> {
    pub(crate) probs: &'a [f32],
    pub(crate) classes: usize,
    pub(crate) clock: Clock,
}

impl<'a> Frames<'a> {
    /// `t0` e o instante do primeiro quadro, em segundos desde o inicio da sessao.
    pub fn new(probs: &'a [f32], classes: usize, fps: f64, t0: f64) -> Result<Self, Error> {
        check(probs, classes)?;
        Ok(Self {
            probs,
            classes,
            clock: Clock::new(fps, t0)?,
        })
    }

    /// Intervalo coberto por todos os quadros.
    pub fn window(&self) -> Span {
        self.clock.span(0, self.probs.len() / self.classes)
    }
}

/// Converte indice de quadro em segundos.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Clock {
    pub(crate) fps: f64,
    t0: f64,
}

impl Clock {
    pub(crate) fn new(fps: f64, t0: f64) -> Result<Self, Error> {
        if !(fps.is_finite() && fps > 0.0) {
            return Err(Error::Fps(fps));
        }
        if !(t0.is_finite() && t0 >= 0.0) {
            return Err(Error::Start(t0));
        }
        Ok(Self { fps, t0 })
    }

    /// Intervalo dos quadros `first..end` (fim exclusivo).
    pub(crate) fn span(&self, first: usize, end: usize) -> Span {
        [
            self.t0 + first as f64 / self.fps,
            self.t0 + end as f64 / self.fps,
        ]
    }
}

/// Valida um bloco de quadros: formato e valores em `[0, 1]`.
pub(crate) fn check(probs: &[f32], classes: usize) -> Result<(), Error> {
    if classes == 0 || probs.len() % classes != 0 {
        return Err(Error::Shape {
            len: probs.len(),
            classes,
        });
    }
    match probs.iter().find(|p| !(0.0..=1.0).contains(*p)) {
        Some(&p) => Err(Error::Probability(p)),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recusa_formato_invalido() {
        assert_eq!(
            Frames::new(&[0.5; 5], 2, 30.0, 0.0).unwrap_err(),
            Error::Shape { len: 5, classes: 2 }
        );
        assert!(Frames::new(&[], 0, 30.0, 0.0).is_err());
        assert_eq!(
            Frames::new(&[0.5; 2], 2, 0.0, 0.0).unwrap_err(),
            Error::Fps(0.0)
        );
        assert!(Frames::new(&[0.5; 2], 2, 30.0, -1.0).is_err());
    }

    #[test]
    fn recusa_valor_que_nao_e_probabilidade() {
        assert_eq!(
            Frames::new(&[0.5, 1.5], 2, 30.0, 0.0).unwrap_err(),
            Error::Probability(1.5)
        );
        assert!(Frames::new(&[0.5, f32::NAN], 2, 30.0, 0.0).is_err());
    }

    #[test]
    fn janela_e_intervalo() {
        let f = Frames::new(&[0.1, 0.9, 0.2, 0.8, 0.3, 0.7], 2, 4.0, 2.0).unwrap();
        assert_eq!(f.clock.span(1, 3), [2.25, 2.75]);
        assert_eq!(f.window(), [2.0, 2.75]);
    }
}
