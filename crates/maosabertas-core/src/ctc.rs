use crate::frames::{check, Clock};
use crate::{Error, Frames};
use maosabertas_protocol::{ManualItem, ManualKind};

/// Uma classe de saida do modelo manual.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub gloss: String,
    pub kind: ManualKind,
}

/// Decodificacao CTC gulosa em fluxo: argmax por quadro, junta repeticoes,
/// descarta o branco. Recebe blocos de quadros conforme chegam e devolve so
/// os itens que ja fecharam; o ultimo fica pendente ate o proximo bloco.
///
/// `conf` e a media da probabilidade vencedora nos quadros do item.
/// shortcut: o intervalo e o dos quadros do pico CTC, nao a fronteira real
/// do sinal; trocar por alinhamento quando houver modelo treinado.
/// shortcut: `alt` sai vazio; preencher com beam search quando a traducao precisar.
#[derive(Debug, Clone)]
pub struct CtcDecoder<'v> {
    vocab: &'v [Token],
    blank: usize,
    clock: Clock,
    /// Indice do proximo quadro.
    frame: usize,
    /// Trecho corrente: (classe, primeiro quadro, soma das probabilidades).
    run: Option<(usize, usize, f64)>,
}

impl<'v> CtcDecoder<'v> {
    /// `t0` e o instante do primeiro quadro, em segundos desde o inicio da sessao.
    pub fn new(vocab: &'v [Token], blank: usize, fps: f64, t0: f64) -> Result<Self, Error> {
        Self::with_clock(vocab, blank, Clock::new(fps, t0)?)
    }

    fn with_clock(vocab: &'v [Token], blank: usize, clock: Clock) -> Result<Self, Error> {
        if blank >= vocab.len() {
            return Err(Error::Blank(blank));
        }
        Ok(Self {
            vocab,
            blank,
            clock,
            frame: 0,
            run: None,
        })
    }

    /// Recebe quadros em linha (quadro x classe). Bloco invalido nao altera o estado.
    pub fn push(&mut self, probs: &[f32]) -> Result<Vec<ManualItem>, Error> {
        check(probs, self.vocab.len())?;
        let mut items = Vec::new();
        for row in probs.chunks_exact(self.vocab.len()) {
            let (best, p) = argmax(row);
            self.run = match self.run {
                Some((class, first, sum)) if class == best => Some((class, first, sum + p)),
                prev => {
                    items.extend(prev.and_then(|r| self.item(r, self.frame)));
                    Some((best, self.frame, p))
                }
            };
            self.frame += 1;
        }
        Ok(items)
    }

    /// Fecha o trecho pendente no fim do fluxo.
    pub fn finish(self) -> Option<ManualItem> {
        self.run.and_then(|r| self.item(r, self.frame))
    }

    fn item(&self, (class, first, sum): (usize, usize, f64), end: usize) -> Option<ManualItem> {
        (class != self.blank).then(|| ManualItem {
            gloss: self.vocab[class].gloss.clone(),
            kind: self.vocab[class].kind,
            param: None,
            t: self.clock.span(first, end),
            conf: sum / (end - first) as f64,
            alt: Vec::new(),
        })
    }
}

/// Decodifica uma janela inteira de uma vez. Ver [`CtcDecoder`].
pub fn greedy_decode(
    frames: &Frames,
    vocab: &[Token],
    blank: usize,
) -> Result<Vec<ManualItem>, Error> {
    if vocab.len() != frames.classes {
        return Err(Error::Classes {
            expected: frames.classes,
            got: vocab.len(),
        });
    }
    let mut d = CtcDecoder::with_clock(vocab, blank, frames.clock)?;
    let mut items = d.push(frames.probs)?;
    items.extend(d.finish());
    Ok(items)
}

/// Em empate vence a primeira classe. Devolve a probabilidade ja em `f64`.
fn argmax(row: &[f32]) -> (usize, f64) {
    let (i, p) =
        row.iter().copied().enumerate().fold(
            (0, row[0]),
            |best, (i, p)| if p > best.1 { (i, p) } else { best },
        );
    (i, f64::from(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vocabulario sintetico: os nomes nao sao glosas de Libras.
    fn vocab() -> Vec<Token> {
        ["<branco>", "A", "B"]
            .into_iter()
            .map(|g| Token {
                gloss: g.into(),
                kind: ManualKind::Lexical,
            })
            .collect()
    }

    /// Um quadro em que `class` vence com probabilidade `p`.
    fn frame(class: usize, p: f32) -> [f32; 3] {
        let mut row = [(1.0 - p) / 2.0; 3];
        row[class] = p;
        row
    }

    fn decode(seq: &[(usize, f32)], fps: f64, t0: f64) -> Vec<ManualItem> {
        let probs: Vec<f32> = seq.iter().flat_map(|&(c, p)| frame(c, p)).collect();
        let f = Frames::new(&probs, 3, fps, t0).unwrap();
        greedy_decode(&f, &vocab(), 0).unwrap()
    }

    #[test]
    fn junta_repeticoes_e_descarta_branco() {
        let out = decode(
            &[
                (1, 0.75),
                (1, 0.5),
                (0, 0.875),
                (1, 0.75),
                (2, 0.5),
                (2, 1.0),
            ],
            4.0,
            0.0,
        );
        let glosas: Vec<_> = out.iter().map(|m| m.gloss.as_str()).collect();
        assert_eq!(glosas, ["A", "A", "B"]);
        assert_eq!(out[0].t, [0.0, 0.5]);
        assert_eq!(out[0].conf, 0.625);
        assert_eq!(out[1].t, [0.75, 1.0]);
        assert_eq!(out[2].t, [1.0, 1.5]);
        assert_eq!(out[2].conf, 0.75);
    }

    #[test]
    fn so_branco_nao_gera_item() {
        assert!(decode(&[(0, 0.75), (0, 0.5)], 30.0, 0.0).is_empty());
        assert!(decode(&[], 30.0, 0.0).is_empty());
    }

    #[test]
    fn aplica_instante_inicial() {
        let out = decode(&[(0, 0.75), (2, 0.75)], 4.0, 2.0);
        assert_eq!(out[0].t, [2.25, 2.5]);
    }

    #[test]
    fn recusa_vocabulario_incompativel() {
        let probs = frame(1, 0.75);
        let f = Frames::new(&probs, 3, 30.0, 0.0).unwrap();
        assert_eq!(
            greedy_decode(&f, &vocab()[..2], 0).unwrap_err(),
            Error::Classes {
                expected: 3,
                got: 2
            }
        );
        assert_eq!(greedy_decode(&f, &vocab(), 3).unwrap_err(), Error::Blank(3));
    }

    const SEQ: [(usize, f32); 7] = [
        (1, 0.75),
        (1, 0.5),
        (0, 0.875),
        (1, 0.75),
        (2, 0.5),
        (2, 1.0),
        (0, 0.75),
    ];

    #[test]
    fn fluxo_em_blocos_da_o_mesmo_que_lote() {
        let probs: Vec<f32> = SEQ.iter().flat_map(|&(c, p)| frame(c, p)).collect();
        let lote = decode(&SEQ, 4.0, 1.0);
        let vocab = vocab();
        for tamanho in 1..=SEQ.len() {
            let mut d = CtcDecoder::new(&vocab, 0, 4.0, 1.0).unwrap();
            let mut out = Vec::new();
            for bloco in probs.chunks(tamanho * 3) {
                out.extend(d.push(bloco).unwrap());
            }
            out.extend(d.finish());
            assert_eq!(out, lote, "blocos de {tamanho} quadros");
        }
    }

    #[test]
    fn item_so_sai_quando_fecha() {
        let vocab = vocab();
        let mut d = CtcDecoder::new(&vocab, 0, 4.0, 0.0).unwrap();
        assert!(d
            .push(&[frame(1, 0.75), frame(1, 0.75)].concat())
            .unwrap()
            .is_empty());
        let fechados = d.push(&frame(2, 0.75)).unwrap();
        assert_eq!(fechados.len(), 1);
        assert_eq!(
            (fechados[0].gloss.as_str(), fechados[0].t),
            ("A", [0.0, 0.5])
        );
        assert_eq!(d.finish().map(|m| m.gloss), Some("B".into()));
    }

    #[test]
    fn bloco_invalido_nao_altera_estado() {
        let vocab = vocab();
        let mut d = CtcDecoder::new(&vocab, 0, 4.0, 0.0).unwrap();
        d.push(&frame(1, 0.75)).unwrap();
        assert!(d.push(&[0.5, 0.5]).is_err());
        assert!(d.push(&[0.5, 2.0, 0.5]).is_err());
        assert_eq!(d.finish().map(|m| m.t), Some([0.0, 0.25]));
    }
}
