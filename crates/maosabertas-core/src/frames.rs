use crate::Error;
use maosabertas_protocol::Span;

/// Probabilidades por quadro em linha (quadro x classe), como saem do tensor ONNX.
#[derive(Debug, Clone, Copy)]
pub struct Frames<'a> {
    probs: &'a [f32],
    classes: usize,
    fps: f64,
    t0: f64,
}

impl<'a> Frames<'a> {
    /// `t0` e o instante do primeiro quadro, em segundos desde o inicio da sessao.
    pub fn new(probs: &'a [f32], classes: usize, fps: f64, t0: f64) -> Result<Self, Error> {
        if classes == 0 || probs.len() % classes != 0 {
            return Err(Error::Shape {
                len: probs.len(),
                classes,
            });
        }
        if !(fps.is_finite() && fps > 0.0) {
            return Err(Error::Fps(fps));
        }
        if !(t0.is_finite() && t0 >= 0.0) {
            return Err(Error::Start(t0));
        }
        if let Some(&p) = probs.iter().find(|p| !(0.0..=1.0).contains(*p)) {
            return Err(Error::Probability(p));
        }
        Ok(Self {
            probs,
            classes,
            fps,
            t0,
        })
    }

    /// Intervalo coberto por todos os quadros.
    pub fn window(&self) -> Span {
        self.span(0, self.count())
    }

    pub(crate) fn classes(&self) -> usize {
        self.classes
    }

    pub(crate) fn count(&self) -> usize {
        self.probs.len() / self.classes
    }

    pub(crate) fn rows(&self) -> impl Iterator<Item = &'a [f32]> {
        self.probs.chunks_exact(self.classes)
    }

    pub(crate) fn column(&self, class: usize) -> impl Iterator<Item = f32> + 'a {
        self.probs.iter().skip(class).step_by(self.classes).copied()
    }

    /// Intervalo dos quadros `first..end` (fim exclusivo).
    pub(crate) fn span(&self, first: usize, end: usize) -> Span {
        [
            self.t0 + first as f64 / self.fps,
            self.t0 + end as f64 / self.fps,
        ]
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
    fn coluna_e_intervalo() {
        let f = Frames::new(&[0.1, 0.9, 0.2, 0.8, 0.3, 0.7], 2, 4.0, 2.0).unwrap();
        assert_eq!(f.count(), 3);
        assert_eq!(f.column(1).collect::<Vec<_>>(), vec![0.9, 0.8, 0.7]);
        assert_eq!(f.span(1, 3), [2.25, 2.75]);
        assert_eq!(f.window(), [2.0, 2.75]);
    }
}
