use crate::frames::{check, Clock};
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

/// Segmenta marcadores nao manuais em fluxo, a partir de probabilidades
/// independentes por classe (uma sigmoide por rotulo, nao softmax), porque
/// marcadores de camadas diferentes coexistem. Cada marcador tem escopo
/// proprio, sem relacao com os sinais manuais.
///
/// Cada bloco devolve os marcadores que fecharam nele, ordenados por inicio.
/// Um marcador que fecha depois pode ter comecado antes de outro ja emitido.
/// `conf` e a media da probabilidade no trecho.
#[derive(Debug, Clone)]
pub struct ScopeSegmenter<'l> {
    labels: &'l [(Tier, NonManualLabel)],
    h: Hysteresis,
    clock: Clock,
    /// Indice do proximo quadro.
    frame: usize,
    /// Trecho aberto por classe: (primeiro quadro, soma das probabilidades).
    open: Vec<Option<(usize, f64)>>,
}

impl<'l> ScopeSegmenter<'l> {
    /// `t0` e o instante do primeiro quadro, em segundos desde o inicio da sessao.
    pub fn new(
        labels: &'l [(Tier, NonManualLabel)],
        h: Hysteresis,
        fps: f64,
        t0: f64,
    ) -> Result<Self, Error> {
        Self::with_clock(labels, h, Clock::new(fps, t0)?)
    }

    fn with_clock(
        labels: &'l [(Tier, NonManualLabel)],
        h: Hysteresis,
        clock: Clock,
    ) -> Result<Self, Error> {
        if !(0.0 <= h.exit && h.exit <= h.enter && h.enter <= 1.0) {
            return Err(Error::Hysteresis {
                enter: h.enter,
                exit: h.exit,
            });
        }
        Ok(Self {
            labels,
            h,
            clock,
            frame: 0,
            open: vec![None; labels.len()],
        })
    }

    /// Recebe quadros em linha (quadro x classe). Bloco invalido nao altera o estado.
    pub fn push(&mut self, probs: &[f32]) -> Result<Vec<NonManualItem>, Error> {
        check(probs, self.labels.len())?;
        let mut items = Vec::new();
        for row in probs.chunks_exact(self.labels.len()) {
            for (class, &p) in row.iter().enumerate() {
                let p = f64::from(p);
                self.open[class] = match self.open[class] {
                    Some((first, sum)) if p >= f64::from(self.h.exit) => Some((first, sum + p)),
                    Some((first, sum)) => {
                        items.extend(self.item(class, first, self.frame, sum));
                        None
                    }
                    None if p >= f64::from(self.h.enter) => Some((self.frame, p)),
                    None => None,
                };
            }
            self.frame += 1;
        }
        sort(&mut items);
        Ok(items)
    }

    /// Fecha os marcadores abertos no fim do fluxo.
    pub fn finish(self) -> Vec<NonManualItem> {
        let mut items: Vec<_> = self
            .open
            .iter()
            .enumerate()
            .filter_map(|(class, o)| {
                o.and_then(|(first, sum)| self.item(class, first, self.frame, sum))
            })
            .collect();
        sort(&mut items);
        items
    }

    fn item(&self, class: usize, first: usize, end: usize, sum: f64) -> Option<NonManualItem> {
        let (tier, label) = self.labels[class];
        (end - first >= self.h.min_frames.max(1)).then(|| NonManualItem {
            tier,
            label,
            t: self.clock.span(first, end),
            conf: sum / (end - first) as f64,
        })
    }
}

/// Segmenta uma janela inteira de uma vez. Ver [`ScopeSegmenter`].
pub fn segment(
    frames: &Frames,
    labels: &[(Tier, NonManualLabel)],
    h: Hysteresis,
) -> Result<Vec<NonManualItem>, Error> {
    if labels.len() != frames.classes {
        return Err(Error::Classes {
            expected: frames.classes,
            got: labels.len(),
        });
    }
    let mut s = ScopeSegmenter::with_clock(labels, h, frames.clock)?;
    let mut items = s.push(frames.probs)?;
    items.extend(s.finish());
    sort(&mut items);
    Ok(items)
}

/// Por inicio e, no empate, por fim.
fn sort(items: &mut [NonManualItem]) {
    items.sort_by(|a, b| a.t[0].total_cmp(&b.t[0]).then(a.t[1].total_cmp(&b.t[1])));
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

    #[test]
    fn fluxo_em_blocos_da_o_mesmo_que_lote() {
        let a = [0.25, 0.875, 0.5, 0.625, 0.25, 1.0, 1.0];
        let b = [1.0, 1.0, 0.0, 0.875, 0.875, 0.875, 0.0];
        let lote = run(&a, &b, H);
        assert_eq!(lote.len(), 4);
        let probs: Vec<f32> = a.iter().zip(&b).flat_map(|(&x, &y)| [x, y]).collect();
        let labels = labels();
        for tamanho in 1..=a.len() {
            let mut s = ScopeSegmenter::new(&labels, H, 4.0, 0.0).unwrap();
            let mut out = Vec::new();
            for bloco in probs.chunks(tamanho * 2) {
                out.extend(s.push(bloco).unwrap());
            }
            out.extend(s.finish());
            sort(&mut out);
            assert_eq!(out, lote, "blocos de {tamanho} quadros");
        }
    }

    #[test]
    fn marcador_so_sai_quando_fecha() {
        let labels = labels();
        let mut s = ScopeSegmenter::new(&labels, H, 4.0, 0.0).unwrap();
        assert!(s.push(&[1.0, 0.0, 1.0, 0.0]).unwrap().is_empty());
        let fechados = s.push(&[0.0, 0.0]).unwrap();
        assert_eq!(fechados.len(), 1);
        assert_eq!(fechados[0].t, [0.0, 0.5]);
        assert!(s.finish().is_empty());
    }

    #[test]
    fn bloco_invalido_nao_altera_estado() {
        let labels = labels();
        let mut s = ScopeSegmenter::new(&labels, H, 4.0, 0.0).unwrap();
        s.push(&[1.0, 0.0]).unwrap();
        assert!(s.push(&[1.0]).is_err());
        assert!(s.push(&[0.0, -0.5]).is_err());
        s.push(&[1.0, 0.0]).unwrap();
        assert_eq!(s.finish()[0].t, [0.0, 0.5]);
    }
}
