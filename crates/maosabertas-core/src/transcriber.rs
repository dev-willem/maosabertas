use crate::frames::{check, Clock};
use crate::phonology::{measure, Hands};
use crate::scope::sort;
use crate::{utterance, CtcDecoder, Error, Hysteresis, ScopeSegmenter, Token};
use maosabertas_protocol::{ManualItem, NonManualItem, NonManualLabel, Quality, Tier, Utterance};

/// Regra de fechamento do enunciado, em segundos (ADR-007).
///
/// Os valores dependem do modelo e precisam ser calibrados com dados; nao ha padrao.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ending {
    /// Fecha quando o branco vence no modelo manual em quadros seguidos por pelo menos isto.
    pub pause: f64,
    /// Corta o enunciado que chega a esta duracao, mesmo no meio de um sinal.
    pub max_duration: f64,
}

/// O que a captura viu num quadro. Quem extrai os landmarks informa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameQuality {
    /// Ombros detectados: as features do quadro sao calculaveis.
    pub reference: bool,
    pub face: bool,
}

/// Transcricao em fluxo: leva [`CtcDecoder`] e [`ScopeSegmenter`] no mesmo
/// relogio e corta o fluxo em enunciados pela regra de [`Ending`].
///
/// Os enunciados cobrem o fluxo sem buracos, cada um do corte anterior ao seu,
/// pausa incluida. Enunciado sem nenhum item nao sai e nao gasta numero.
/// Marcador nao manual aberto no corte fecha ali e reabre no enunciado
/// seguinte: o marcador tem escopo proprio, mas o enunciado e a unidade do
/// schema. Sinal em curso num corte por duracao sai partido nos dois.
///
/// Sem modelo nao manual: `labels` vazio e bloco nao manual vazio em
/// [`push`](Self::push). `h` continua validada, mas nao tem efeito.
///
/// Com as maos de cada quadro ([`phonology::hands`](crate::phonology::hands)), cada
/// item manual sai com a `fonologia` medida no seu trecho (ADR-011).
#[derive(Debug, Clone)]
pub struct Transcriber<'a> {
    ctc: CtcDecoder<'a>,
    scope: ScopeSegmenter<'a>,
    /// Classes do modelo manual e do nao manual.
    classes: (usize, usize),
    ending: Ending,
    clock: Clock,
    prefix: String,
    /// Enunciados ja emitidos.
    count: usize,
    /// Indice do proximo quadro.
    frame: usize,
    /// Primeiro quadro do enunciado corrente.
    start: usize,
    manual: Vec<ManualItem>,
    nonmanual: Vec<NonManualItem>,
    /// Quadros do enunciado corrente com referencia e com rosto.
    reference: usize,
    face: usize,
    /// Maos de cada quadro do enunciado corrente.
    hands: Vec<Hands>,
}

impl<'a> Transcriber<'a> {
    /// `utterance_id` sai como `prefix` seguido do contador, a partir de 1.
    /// `t0` e o instante do primeiro quadro, em segundos desde o inicio da sessao.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prefix: impl Into<String>,
        vocab: &'a [Token],
        blank: usize,
        labels: &'a [(Tier, NonManualLabel)],
        h: Hysteresis,
        ending: Ending,
        fps: f64,
        t0: f64,
    ) -> Result<Self, Error> {
        let positive = |s: f64| s.is_finite() && s > 0.0;
        if !(positive(ending.pause) && positive(ending.max_duration)) {
            return Err(Error::Ending(ending));
        }
        let clock = Clock::new(fps, t0)?;
        Ok(Self {
            ctc: CtcDecoder::with_clock(vocab, blank, clock)?,
            scope: ScopeSegmenter::with_clock(labels, h, clock)?,
            classes: (vocab.len(), labels.len()),
            ending,
            clock,
            prefix: prefix.into(),
            count: 0,
            frame: 0,
            start: 0,
            manual: Vec::new(),
            nonmanual: Vec::new(),
            reference: 0,
            face: 0,
            hands: Vec::new(),
        })
    }

    /// Recebe o mesmo bloco de quadros dos dois modelos, em linha (quadro x
    /// classe), o que a captura viu e as maos de cada quadro. Devolve os
    /// enunciados que fecharam no bloco. `hands` vazio = sem medicao fonologica
    /// nesses quadros. Bloco invalido nao altera o estado.
    pub fn push(
        &mut self,
        manual: &[f32],
        nonmanual: &[f32],
        seen: &[FrameQuality],
        hands: &[Hands],
    ) -> Result<Vec<Utterance>, Error> {
        let (cm, cn) = self.classes;
        check(manual, cm)?;
        let frames = manual.len() / cm;
        // Sem rotulos nao manuais o bloco nao manual vem vazio e vale qualquer numero de quadros.
        let nonmanual_frames = if cn == 0 && nonmanual.is_empty() {
            frames
        } else {
            check(nonmanual, cn)?;
            nonmanual.len() / cn
        };
        let hand_frames = if hands.is_empty() {
            frames
        } else {
            hands.len()
        };
        for got in [nonmanual_frames, seen.len(), hand_frames] {
            if got != frames {
                return Err(Error::FrameCount {
                    expected: frames,
                    got,
                });
            }
        }
        let fps = self.clock.fps;
        let mut out = Vec::new();
        for (i, (m, q)) in manual.chunks_exact(cm).zip(seen).enumerate() {
            self.manual.extend(self.ctc.push(m)?);
            if cn > 0 {
                self.nonmanual
                    .extend(self.scope.push(&nonmanual[i * cn..(i + 1) * cn])?);
            }
            self.reference += usize::from(q.reference);
            self.face += usize::from(q.face);
            self.hands
                .push(hands.get(i).copied().unwrap_or([None, None]));
            self.frame += 1;
            if self.ctc.blank_run() as f64 / fps >= self.ending.pause
                || (self.frame - self.start) as f64 / fps >= self.ending.max_duration
            {
                out.extend(self.close());
            }
        }
        Ok(out)
    }

    /// Fecha o enunciado pendente no fim do fluxo.
    pub fn finish(mut self) -> Option<Utterance> {
        self.close()
    }

    /// Corta no quadro atual; o enunciado seguinte comeca aqui.
    fn close(&mut self) -> Option<Utterance> {
        self.manual.extend(self.ctc.cut());
        self.nonmanual.extend(self.scope.cut());
        let mut manual = std::mem::take(&mut self.manual);
        let mut nonmanual = std::mem::take(&mut self.nonmanual);
        let hands = std::mem::take(&mut self.hands);
        annotate(&mut manual, &hands, self.start, &self.clock);
        let (reference, face) = (
            std::mem::take(&mut self.reference),
            std::mem::take(&mut self.face),
        );
        let (first, frames) = (self.start, self.frame - self.start);
        self.start = self.frame;
        if manual.is_empty() && nonmanual.is_empty() {
            return None;
        }
        sort(&mut nonmanual);
        self.count += 1;
        let id = format!("{}{}", self.prefix, self.count);
        let mut u = utterance(id, self.clock.span(first, self.frame), manual, nonmanual);
        let n = frames as f64;
        u.quality = Some(Quality {
            frames_ok: Some(reference as f64 / n),
            face_occluded: Some((frames - face) as f64 / n),
            fps: Some(self.clock.fps),
        });
        Some(u)
    }
}

/// Preenche a `fonologia` de cada item com as maos dos quadros do seu trecho.
/// `hands[0]` e o quadro `first`; os itens estao em ordem e nao se sobrepoem.
///
/// shortcut: o intervalo do CTC e so o pico, nao o sinal (ver `CtcDecoder`); o trecho
/// medido vai do meio da folga com o vizinho anterior ao meio da folga com o seguinte
/// (as bordas do enunciado contam como vizinhos). Trocar por alinhamento quando houver
/// modelo treinado.
fn annotate(items: &mut [ManualItem], hands: &[Hands], first: usize, clock: &Clock) {
    let n = hands.len();
    let spans: Vec<(usize, usize)> = items
        .iter()
        .map(|m| {
            let at = |t: f64| clock.frame(t).saturating_sub(first).min(n);
            (at(m.t[0]), at(m.t[1]))
        })
        .collect();
    let dt = (1.0 / clock.fps) as f32;
    for (k, item) in items.iter_mut().enumerate() {
        let (a, b) = spans[k];
        let before = k.checked_sub(1).map_or(0, |j| spans[j].1);
        let after = spans.get(k + 1).map_or(n, |s| s.0);
        let window = &hands[(before + a) / 2..(b + after).div_ceil(2).max(b)];
        item.fonologia = (0..2)
            .filter_map(|side| {
                let frames: Vec<_> = window.iter().map(|h| h[side]).collect();
                measure(&frames, dt)
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonology::{HandPose, Side};
    use maosabertas_protocol::{Hand, ManualKind};

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

    const LABELS: [(Tier, NonManualLabel); 1] =
        [(Tier::Gramatical, NonManualLabel::InterrogativaSn)];

    const H: Hysteresis = Hysteresis {
        enter: 0.75,
        exit: 0.5,
        min_frames: 1,
    };

    /// A 4 fps: pausa de 3 quadros, sem corte por duracao.
    const PAUSA: Ending = Ending {
        pause: 0.75,
        max_duration: 100.0,
    };

    /// Um quadro: classe manual vencedora, probabilidade do marcador, ombros e rosto vistos.
    type Quadro = (usize, f32, bool, bool);

    fn sinal(class: usize) -> Quadro {
        (class, 0.0, true, true)
    }

    /// Passa `seq` em blocos de `tamanho` quadros e fecha o fluxo.
    fn transcreve(seq: &[Quadro], ending: Ending, tamanho: usize) -> Vec<Utterance> {
        let manual: Vec<f32> = seq
            .iter()
            .flat_map(|&(c, ..)| {
                let mut row = [0.125; 3];
                row[c] = 0.75;
                row
            })
            .collect();
        let nonmanual: Vec<f32> = seq.iter().map(|q| q.1).collect();
        let seen: Vec<_> = seq
            .iter()
            .map(|&(_, _, reference, face)| FrameQuality { reference, face })
            .collect();
        let vocab = vocab();
        let mut t = Transcriber::new("t_", &vocab, 0, &LABELS, H, ending, 4.0, 0.0).unwrap();
        let mut out = Vec::new();
        let blocos = manual
            .chunks(tamanho * 3)
            .zip(nonmanual.chunks(tamanho))
            .zip(seen.chunks(tamanho));
        for ((m, n), q) in blocos {
            out.extend(t.push(m, n, q, &[]).unwrap());
        }
        out.extend(t.finish());
        out
    }

    fn resumo(us: &[Utterance]) -> Vec<(&str, [f64; 2], String)> {
        us.iter()
            .map(|u| {
                (
                    u.utterance_id.as_str(),
                    [u.t_start, u.t_end],
                    u.gloss_line(),
                )
            })
            .collect()
    }

    #[test]
    fn pausa_fecha_o_enunciado() {
        let seq = [sinal(1), sinal(1), sinal(0), sinal(0), sinal(0), sinal(2)];
        let out = transcreve(&seq, PAUSA, seq.len());
        assert_eq!(
            resumo(&out),
            [
                ("t_1", [0.0, 1.25], "A".into()),
                ("t_2", [1.25, 1.5], "B".into())
            ]
        );
        assert_eq!(out[0].manual[0].t, [0.0, 0.5]);
    }

    #[test]
    fn pausa_curta_nao_fecha() {
        let seq = [sinal(1), sinal(0), sinal(0), sinal(2)];
        let out = transcreve(&seq, PAUSA, seq.len());
        assert_eq!(resumo(&out), [("t_1", [0.0, 1.0], "A B".into())]);
    }

    #[test]
    fn enunciado_vazio_nao_sai_nem_gasta_numero() {
        let mut seq = vec![sinal(0); 7];
        assert!(transcreve(&seq, PAUSA, seq.len()).is_empty());
        seq.push(sinal(1));
        let out = transcreve(&seq, PAUSA, seq.len());
        assert_eq!(resumo(&out), [("t_1", [1.5, 2.0], "A".into())]);
    }

    #[test]
    fn duracao_maxima_corta_no_meio_do_sinal() {
        let ending = Ending {
            pause: 100.0,
            max_duration: 1.0,
        };
        let seq = [sinal(1); 6];
        let out = transcreve(&seq, ending, seq.len());
        assert_eq!(
            resumo(&out),
            [
                ("t_1", [0.0, 1.0], "A".into()),
                ("t_2", [1.0, 1.5], "A".into())
            ]
        );
        assert_eq!(out[1].manual[0].t, [1.0, 1.5]);
    }

    #[test]
    fn marcador_aberto_no_corte_continua_no_seguinte() {
        let ending = Ending {
            pause: 100.0,
            max_duration: 0.75,
        };
        // So branco no manual: enunciados so com marcacao nao manual.
        let seq = [1.0, 1.0, 1.0, 0.625, 0.625, 0.0].map(|p| (0, p, true, true));
        let out = transcreve(&seq, ending, seq.len());
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|u| u.manual.is_empty()));
        let marcadores: Vec<_> = out
            .iter()
            .map(|u| (u.nonmanual[0].t, u.nonmanual[0].conf))
            .collect();
        // 0,625 nao abriria o marcador; ele segue porque estava aberto no corte.
        assert_eq!(marcadores, [([0.0, 0.75], 1.0), ([0.75, 1.25], 0.625)]);
        assert_eq!([out[1].t_start, out[1].t_end], [0.75, 1.5]);
    }

    #[test]
    fn qualidade_conta_referencia_e_rosto() {
        let seq = [
            (1, 0.0, true, true),
            (1, 0.0, true, false),
            (1, 0.0, true, false),
            (1, 0.0, false, true),
        ];
        let out = transcreve(&seq, PAUSA, seq.len());
        assert_eq!(
            out[0].quality,
            Some(Quality {
                frames_ok: Some(0.75),
                face_occluded: Some(0.5),
                fps: Some(4.0),
            })
        );
    }

    #[test]
    fn blocos_de_qualquer_tamanho_dao_o_mesmo_resultado() {
        let ending = Ending {
            pause: 0.75,
            max_duration: 2.0,
        };
        let seq: Vec<Quadro> = [
            (1, 0.0),
            (1, 1.0),
            (2, 1.0),
            (0, 0.5),
            (0, 0.25),
            (0, 0.0),
            (0, 0.875),
            (1, 1.0),
            (1, 1.0),
            (1, 0.75),
            (2, 0.75),
            (2, 0.5),
            (2, 1.0),
            (1, 0.0),
            (0, 0.0),
            (2, 0.0),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(c, p))| (c, p, i % 3 != 0, i % 4 != 0))
        .collect();
        let lote = transcreve(&seq, ending, seq.len());
        // Fecha por pausa, por duracao e no fim.
        assert_eq!(
            resumo(&lote),
            [
                ("t_1", [0.0, 1.5], "A B".into()),
                ("t_2", [1.5, 3.5], "A B A".into()),
                ("t_3", [3.5, 4.0], "B".into())
            ]
        );
        assert_eq!(lote[1].nonmanual[0].t, [1.5, 3.25]);
        for tamanho in 1..seq.len() {
            assert_eq!(
                transcreve(&seq, ending, tamanho),
                lote,
                "blocos de {tamanho} quadros"
            );
        }
    }

    #[test]
    fn bloco_invalido_nao_altera_estado() {
        let vocab = vocab();
        let mut t = Transcriber::new("t_", &vocab, 0, &LABELS, H, PAUSA, 4.0, 0.0).unwrap();
        let visto = [FrameQuality {
            reference: true,
            face: true,
        }; 2];
        let a = [0.125, 0.75, 0.125];
        assert_eq!(
            t.push(&a, &[0.0, 0.0], &visto[..1], &[]).unwrap_err(),
            Error::FrameCount {
                expected: 1,
                got: 2
            }
        );
        assert_eq!(
            t.push(&a, &[0.0], &visto, &[]).unwrap_err(),
            Error::FrameCount {
                expected: 1,
                got: 2
            }
        );
        assert!(t.push(&a, &[1.5], &visto[..1], &[]).is_err());
        assert!(t.push(&[0.5, 0.5], &[0.0], &visto[..1], &[]).is_err());
        t.push(&a, &[0.0], &visto[..1], &[]).unwrap();
        let u = t.finish().unwrap();
        assert_eq!(
            ([u.t_start, u.t_end], u.gloss_line()),
            ([0.0, 0.25], "A".into())
        );
    }

    #[test]
    fn so_modelo_manual_sem_rotulos_nao_manuais() {
        let vocab = vocab();
        let mut t = Transcriber::new("t_", &vocab, 0, &[], H, PAUSA, 4.0, 0.0).unwrap();
        let visto = [FrameQuality {
            reference: true,
            face: false,
        }; 5];
        let (a, branco, b) = (
            [0.125, 0.75, 0.125],
            [0.75, 0.125, 0.125],
            [0.125, 0.125, 0.75],
        );
        let bloco = [a, branco, branco, branco, b].concat();
        // Sem rotulos, o bloco nao manual vem vazio; quadro a mais ou a menos continua erro.
        assert!(t.push(&bloco, &[0.0; 5], &visto, &[]).is_err());
        assert!(t.push(&bloco, &[], &visto[..4], &[]).is_err());
        let mut out = t.push(&bloco, &[], &visto, &[]).unwrap();
        out.extend(t.finish());
        assert_eq!(
            resumo(&out),
            [
                ("t_1", [0.0, 1.0], "A".into()),
                ("t_2", [1.0, 1.25], "B".into())
            ]
        );
        assert!(out.iter().all(|u| u.nonmanual.is_empty()));
        assert_eq!(
            out[0].quality.as_ref().and_then(|q| q.face_occluded),
            Some(1.0)
        );
    }

    /// So a mao direita, de brinquedo, com o punho em (x, y) no referencial do corpo.
    fn direita(x: f32, y: f32) -> Hands {
        let mut points = [[x, y, 0.0]; 21];
        for (i, p) in points.iter_mut().enumerate().skip(1) {
            *p = [
                x + 0.01 * ((i % 5) as f32 - 2.0),
                y + 0.02 * (i / 5) as f32 + 0.01,
                0.0,
            ];
        }
        [
            None,
            Some(HandPose {
                side: Side::Right,
                points,
            }),
        ]
    }

    #[test]
    fn itens_saem_com_a_fonologia_do_seu_trecho() {
        let vocab = vocab();
        let mut t = Transcriber::new("t_", &vocab, 0, &[], H, PAUSA, 4.0, 0.0).unwrap();
        let (a, branco, b) = (
            [0.125, 0.75, 0.125],
            [0.75, 0.125, 0.125],
            [0.125, 0.125, 0.75],
        );
        let bloco = [a, a, branco, branco, branco, b].concat();
        let visto = [FrameQuality {
            reference: true,
            face: true,
        }; 6];
        // A mao anda 0,1 por quadro durante A e a folga, desce ao repouso e volta para B.
        let maos = [
            direita(0.0, -0.5),
            direita(0.1, -0.5),
            direita(0.2, -0.5),
            direita(0.3, -0.5),
            direita(0.3, -2.0),
            direita(-0.2, 0.0),
        ];
        assert!(t.push(&bloco, &[], &visto, &maos[..5]).is_err());
        let mut out = t.push(&bloco, &[], &visto, &maos).unwrap();
        out.extend(t.finish());
        assert_eq!(resumo(&out).len(), 2);

        // A tem o pico nos quadros 0 e 1; o trecho medido vai ate o meio da folga (quadro 3).
        let [f] = out[0].manual[0].fonologia.as_slice() else {
            panic!("uma mao em A: {:?}", out[0].manual[0].fonologia);
        };
        assert_eq!((f.mao, f.visivel), (Hand::Direita, Some(1.0)));
        let m = f.m.as_ref().unwrap();
        assert_eq!((m.desloc, m.trajeto), (Some([0.3, 0.0]), Some(0.3)));

        let [g] = out[1].manual[0].fonologia.as_slice() else {
            panic!("uma mao em B");
        };
        assert_eq!(g.m.as_ref().unwrap().desloc, Some([0.0, 0.0]));
        assert!(g.pa.as_ref().unwrap().pos.unwrap()[0] < 0.0);
    }

    #[test]
    fn sem_maos_nao_ha_fonologia() {
        let out = transcreve(&[sinal(1), sinal(1)], PAUSA, 2);
        assert!(out[0].manual[0].fonologia.is_empty());
    }

    #[test]
    fn recusa_fechamento_invalido() {
        let vocab = vocab();
        for (pause, max_duration) in [
            (0.0, 1.0),
            (1.0, -1.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            let ending = Ending {
                pause,
                max_duration,
            };
            assert!(matches!(
                Transcriber::new("t_", &vocab, 0, &LABELS, H, ending, 4.0, 0.0),
                Err(Error::Ending(_))
            ));
        }
    }
}
