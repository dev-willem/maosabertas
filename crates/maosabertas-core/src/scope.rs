use crate::{Error, Frames};
use maosabertas_protocol::{NonManualItem, NonManualLabel, Tier};

/// Limiares para abrir e fechar um marcador sem piscar em quadros ruidosos.
///
/// Os valores dependem do modelo e precisam ser calibrados com dados; nao ha padrao.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hysteresis {
    /// Probabilidade minima para abrir o marcador.
    pub enter: f32,
    /// O marcador continua aberto enquanto a probabilidade ficar acima disto.
    pub exit: f32,
    /// Trechos mais curtos que isto sao descartados.
    pub min_frames: usize,
}

/// Segmenta marcadores nao manuais a partir de probabilidades independentes
/// por classe (uma sigmoide por rotulo, nao softmax), porque marcadores de
/// camadas diferentes coexistem. Cada marcador tem escopo proprio, sem
/// relacao com os sinais manuais.
///
/// Saida ordenada por inicio; `conf` e a media da probabilidade no trecho.
pub fn segment(
    frames: &Frames,
    labels: &[(Tier, NonManualLabel)],
    h: Hysteresis,
) -> Result<Vec<NonManualItem>, Error> {
    if labels.len() != frames.classes() {
        return Err(Error::Classes {
            expected: frames.classes(),
            got: labels.len(),
        });
    }
    if !(0.0 <= h.exit && h.exit <= h.enter && h.enter <= 1.0) {
        return Err(Error::Hysteresis {
            enter: h.enter,
            exit: h.exit,
        });
    }

    let mut items = Vec::new();
    for (class, &(tier, label)) in labels.iter().enumerate() {
        let mut emit = |first: usize, end: usize, sum: f64| {
            if end - first >= h.min_frames.max(1) {
                items.push(NonManualItem {
                    tier,
                    label,
                    t: frames.span(first, end),
                    conf: sum / (end - first) as f64,
                });
            }
        };
        // Trecho aberto: (primeiro quadro, soma das probabilidades).
        let mut open: Option<(usize, f64)> = None;
        for (i, p) in frames.column(class).enumerate() {
            match &mut open {
                Some((_, sum)) if p >= h.exit => *sum += f64::from(p),
                Some((first, sum)) => {
                    emit(*first, i, *sum);
                    open = None;
                }
                None if p >= h.enter => open = Some((i, f64::from(p))),
                None => {}
            }
        }
        if let Some((first, sum)) = open {
            emit(first, frames.count(), sum);
        }
    }
    items.sort_by(|a, b| a.t[0].total_cmp(&b.t[0]));
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: Hysteresis = Hysteresis {
        enter: 0.75,
        exit: 0.5,
        min_frames: 2,
    };

    fn labels() -> Vec<(Tier, NonManualLabel)> {
        vec![
            (Tier::Gramatical, NonManualLabel::InterrogativaSn),
            (Tier::Afetiva, NonManualLabel::Alegria),
        ]
    }

    /// Duas colunas lado a lado em linhas (quadro x classe).
    fn run(a: &[f32], b: &[f32], h: Hysteresis) -> Vec<NonManualItem> {
        let probs: Vec<f32> = a.iter().zip(b).flat_map(|(&x, &y)| [x, y]).collect();
        let f = Frames::new(&probs, 2, 4.0, 0.0).unwrap();
        segment(&f, &labels(), h).unwrap()
    }

    #[test]
    fn histerese_abre_alto_e_fecha_baixo() {
        let out = run(&[0.25, 0.875, 0.5, 0.625, 0.25], &[0.0; 5], H);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, NonManualLabel::InterrogativaSn);
        assert_eq!(out[0].t, [0.25, 1.0]);
        assert_eq!(out[0].conf, 0.6666666666666666);
    }

    #[test]
    fn abaixo_da_entrada_nao_abre() {
        assert!(run(&[0.625, 0.625, 0.625], &[0.0; 3], H).is_empty());
    }

    #[test]
    fn descarta_trecho_curto() {
        assert!(run(&[0.0, 1.0, 0.0], &[0.0; 3], H).is_empty());
    }

    #[test]
    fn fecha_no_ultimo_quadro() {
        let out = run(&[0.0, 1.0, 1.0], &[0.0; 3], H);
        assert_eq!(out[0].t, [0.25, 0.75]);
    }

    #[test]
    fn camadas_coexistem_com_escopos_proprios() {
        let out = run(&[1.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 1.0, 1.0], H);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].tier, out[0].t), (Tier::Gramatical, [0.0, 1.0]));
        assert_eq!((out[1].tier, out[1].t), (Tier::Afetiva, [0.5, 1.0]));
    }

    #[test]
    fn recusa_histerese_invertida() {
        let f = Frames::new(&[0.5, 0.5], 2, 4.0, 0.0).unwrap();
        let h = Hysteresis {
            enter: 0.25,
            exit: 0.5,
            min_frames: 1,
        };
        assert!(segment(&f, &labels(), h).is_err());
    }
}
