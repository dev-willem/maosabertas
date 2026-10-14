use crate::{Error, Frames};
use maosabertas_protocol::{ManualItem, ManualKind};

/// Uma classe de saida do modelo manual.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub gloss: String,
    pub kind: ManualKind,
}

/// Decodificacao CTC gulosa: argmax por quadro, junta repeticoes, descarta o branco.
///
/// `conf` e a media da probabilidade vencedora nos quadros do item.
/// shortcut: o intervalo e o dos quadros do pico CTC, nao a fronteira real
/// do sinal; trocar por alinhamento quando houver modelo treinado.
/// shortcut: `alt` sai vazio; preencher com beam search quando a traducao precisar.
pub fn greedy_decode(
    frames: &Frames,
    vocab: &[Token],
    blank: usize,
) -> Result<Vec<ManualItem>, Error> {
    if vocab.len() != frames.classes() {
        return Err(Error::Classes {
            expected: frames.classes(),
            got: vocab.len(),
        });
    }
    if blank >= vocab.len() {
        return Err(Error::Blank(blank));
    }

    let mut items = Vec::new();
    // Trecho corrente: (classe, primeiro quadro, soma das probabilidades).
    let mut run: Option<(usize, usize, f64)> = None;
    let mut emit = |(class, first, sum): (usize, usize, f64), end: usize| {
        if class != blank {
            items.push(ManualItem {
                gloss: vocab[class].gloss.clone(),
                kind: vocab[class].kind,
                param: None,
                t: frames.span(first, end),
                conf: sum / (end - first) as f64,
                alt: Vec::new(),
            });
        }
    };

    for (i, row) in frames.rows().enumerate() {
        let (best, p) = argmax(row);
        match &mut run {
            Some((class, _, sum)) if *class == best => *sum += f64::from(p),
            _ => {
                if let Some(r) = run.take() {
                    emit(r, i);
                }
                run = Some((best, i, f64::from(p)));
            }
        }
    }
    if let Some(r) = run {
        emit(r, frames.count());
    }
    Ok(items)
}

/// Em empate vence a primeira classe.
fn argmax(row: &[f32]) -> (usize, f32) {
    row.iter().copied().enumerate().fold(
        (0, row[0]),
        |best, (i, p)| if p > best.1 { (i, p) } else { best },
    )
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
}
