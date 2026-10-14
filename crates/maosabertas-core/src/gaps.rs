//! Quadros sem referencia (sem ombros): a mesma regra no treino e no tempo real.

use crate::{Error, FEATURE_DIM};

/// Preenche em fluxo os quadros sem referencia, sem atraso: cada quadro sai
/// assim que entra.
///
/// Os primeiros `max_gap` quadros de uma lacuna repetem o ultimo quadro
/// valido inteiro, presencas inclusive. Do quadro `max_gap + 1` da lacuna em
/// diante, e antes de qualquer quadro valido, sai a linha toda zero
/// (presencas 0, como mao e rosto ausentes). Uma lacuna longa, portanto,
/// repete no comeco e zera no resto: decidir pela lacuna inteira exigiria
/// esperar ate `max_gap` quadros para saber o tamanho dela.
///
/// Quadro com algum NaN conta como sem referencia: e assim que o lote marca
/// a lacuna.
#[derive(Debug, Clone)]
pub struct GapFiller {
    max_gap: usize,
    /// Quadros seguidos sem referencia ate aqui.
    gap: usize,
    /// O que sai na lacuna: o ultimo quadro valido, ou zeros.
    row: Vec<f32>,
}

impl GapFiller {
    pub fn new(max_gap: usize) -> Self {
        Self {
            max_gap,
            gap: 0,
            row: vec![0.0; FEATURE_DIM],
        }
    }

    /// Recebe um quadro de [`frame_features`](crate::frame_features) e devolve
    /// o que entra no modelo. Quadro de tamanho errado nao altera o estado.
    pub fn push(&mut self, frame: Option<&[f32]>) -> Result<&[f32], Error> {
        match frame {
            Some(f) if f.len() != FEATURE_DIM => {
                return Err(Error::Landmarks {
                    part: "features",
                    expected: FEATURE_DIM,
                    got: f.len(),
                })
            }
            Some(f) if !f.iter().any(|v| v.is_nan()) => {
                self.row.copy_from_slice(f);
                self.gap = 0;
            }
            _ => {
                self.gap += 1;
                if self.gap > self.max_gap {
                    self.row.fill(0.0);
                }
            }
        }
        Ok(&self.row)
    }
}

/// Preenche uma janela inteira de uma vez: T x FEATURE_DIM em linha, com
/// linha NaN no quadro sem referencia (como sai de `maosabertas-py`). Ver
/// [`GapFiller`].
pub fn fill_gaps(features: &[f32], max_gap: usize) -> Result<Vec<f32>, Error> {
    let mut g = GapFiller::new(max_gap);
    let mut out = Vec::with_capacity(features.len());
    // Um resto menor que um quadro cai na checagem de tamanho de `push`.
    for row in features.chunks(FEATURE_DIM) {
        out.extend_from_slice(g.push(Some(row))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HAND_VALUES;

    /// Quadro valido sintetico: todos os valores `v`, mao esquerda presente.
    fn valid(v: f32) -> Vec<f32> {
        let mut row = vec![v; FEATURE_DIM];
        row[HAND_VALUES] = 1.0;
        row
    }

    fn zeros() -> Vec<f32> {
        vec![0.0; FEATURE_DIM]
    }

    fn stream(frames: &[Option<Vec<f32>>], max_gap: usize) -> Vec<Vec<f32>> {
        let mut g = GapFiller::new(max_gap);
        frames
            .iter()
            .map(|f| g.push(f.as_deref()).unwrap().to_vec())
            .collect()
    }

    #[test]
    fn repete_ate_max_gap_e_depois_zera() {
        let (a, b) = (valid(0.25), valid(0.5));
        let out = stream(
            &[
                Some(a.clone()),
                None,
                None,
                None,
                None,
                Some(b.clone()),
                None,
            ],
            2,
        );
        assert_eq!(
            out,
            [&a, &a, &a, &zeros(), &zeros(), &b, &b].map(Clone::clone)
        );
        // Na repeticao a presenca vem do quadro repetido.
        assert_eq!(out[2][HAND_VALUES], 1.0);
    }

    #[test]
    fn antes_do_primeiro_quadro_valido_zera() {
        let a = valid(0.25);
        let out = stream(&[None, None, Some(a.clone())], 5);
        assert_eq!(out, [&zeros(), &zeros(), &a].map(Clone::clone));
    }

    #[test]
    fn max_gap_zero_nunca_repete() {
        let a = valid(0.25);
        let out = stream(&[Some(a.clone()), None, Some(a.clone())], 0);
        assert_eq!(out, [&a, &zeros(), &a].map(Clone::clone));
    }

    #[test]
    fn quadro_com_nan_conta_como_sem_referencia() {
        let a = valid(0.25);
        let mut b = valid(0.5);
        b[7] = f32::NAN;
        assert_eq!(
            stream(&[Some(a.clone()), Some(b)], 1),
            [&a, &a].map(Clone::clone)
        );
    }

    #[test]
    fn quadro_invalido_nao_altera_estado() {
        let a = valid(0.25);
        let mut g = GapFiller::new(1);
        g.push(Some(&a)).unwrap();
        assert_eq!(
            g.push(Some(&[0.0; 5])).unwrap_err(),
            Error::Landmarks {
                part: "features",
                expected: FEATURE_DIM,
                got: 5
            }
        );
        assert_eq!(g.push(None).unwrap(), a);
    }

    #[test]
    fn lote_recusa_tamanho_que_nao_forma_quadros() {
        assert!(fill_gaps(&[0.0; FEATURE_DIM + 1], 1).is_err());
        assert_eq!(fill_gaps(&[], 1).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn fluxo_em_blocos_da_o_mesmo_que_lote() {
        let seq = [
            None,
            Some(valid(0.25)),
            None,
            None,
            None,
            Some(valid(0.5)),
            None,
            Some(valid(0.75)),
            None,
        ];
        let flat: Vec<f32> = seq
            .iter()
            .flat_map(|f| f.clone().unwrap_or_else(|| vec![f32::NAN; FEATURE_DIM]))
            .collect();
        for max_gap in 0..=3 {
            let lote = fill_gaps(&flat, max_gap).unwrap();
            assert!(lote.iter().all(|v| !v.is_nan()));
            for tamanho in 1..=seq.len() {
                let mut g = GapFiller::new(max_gap);
                let mut out = Vec::new();
                for bloco in seq.chunks(tamanho) {
                    for f in bloco {
                        out.extend_from_slice(g.push(f.as_deref()).unwrap());
                    }
                }
                assert_eq!(out, lote, "max_gap {max_gap}, blocos de {tamanho}");
            }
        }
    }
}
