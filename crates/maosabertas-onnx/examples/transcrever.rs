//! Demonstracao: landmarks por quadro -> features -> modelo manual ->
//! enunciados, um JSON por linha na saida padrao.
//!
//! Uso:
//!     cargo run -p maosabertas-onnx --example transcrever -- \
//!         MODELO.onnx EXPERIMENT.json ENTRADA.jsonl --pausa S --duracao-max S
//!
//! `EXPERIMENT.json` e o registro que o treino grava ao lado do modelo: dele
//! saem o vocabulario (branco na posicao 0) e o `max_gap` das lacunas.
//! `ENTRADA.jsonl` sai de `python -m maosabertas_training.landmarks_jsonl`: um
//! quadro por linha, `{t, size, left_hand, right_hand, face, blendshapes, pose}`,
//! partes achatadas no layout do `.npz` e `null` onde nada foi detectado.
//! Landmarks sao dado pessoal: a entrada fica fora do repositorio.
//!
//! So o modelo manual: ainda nao ha classificador nao manual treinado.
//! Pausa e duracao maxima nao tem padrao (ADR-007); calibre com dados.

use maosabertas_core::{
    frame_features, Ending, FrameLandmarks, FrameQuality, GapFiller, Hysteresis, Token,
    Transcriber, BLENDSHAPE_VALUES, FACE_VALUES, HAND_VALUES, POSE_VALUES,
};
use maosabertas_onnx::Model;
use maosabertas_protocol::ManualKind;
use serde::Deserialize;
use std::error::Error;
use std::fmt::Display;
use std::io::Write;
use std::process::ExitCode;

const USO: &str =
    "uso: transcrever MODELO.onnx EXPERIMENT.json ENTRADA.jsonl --pausa S --duracao-max S";

/// Sem rotulos nao manuais a histerese nao tem efeito; so precisa ser valida.
const SEM_EFEITO: Hysteresis = Hysteresis {
    enter: 1.0,
    exit: 1.0,
    min_frames: 1,
};

#[derive(Deserialize)]
struct Experimento {
    vocab: Vec<String>,
    config: Config,
}

#[derive(Deserialize)]
struct Config {
    max_gap: usize,
}

/// `null` se a parte nao foi detectada; `null` num valor e NaN no `.npz`.
type Parte = Option<Vec<Option<f32>>>;

#[derive(Deserialize)]
struct Quadro {
    /// Segundos desde o primeiro quadro.
    t: f64,
    /// Largura e altura da imagem, em pixels.
    size: [f64; 2],
    left_hand: Parte,
    right_hand: Parte,
    face: Parte,
    blendshapes: Parte,
    pose: Parte,
}

/// Volta ao layout do `.npz`: NaN onde nada foi detectado.
fn com_nan(parte: &Parte, n: usize) -> Vec<f32> {
    match parte {
        Some(v) => v.iter().map(|x| x.unwrap_or(f32::NAN)).collect(),
        None => vec![f32::NAN; n],
    }
}

fn em<E: Display>(onde: &str) -> impl Fn(E) -> String + '_ {
    move |e| format!("{onde}: {e}")
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let (mut posicionais, mut pause, mut max_duration) = (Vec::new(), None, None);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let alvo = match a.as_str() {
            "--pausa" => &mut pause,
            "--duracao-max" => &mut max_duration,
            _ => {
                posicionais.push(a);
                continue;
            }
        };
        let v = it.next().ok_or_else(|| format!("{a} sem valor\n{USO}"))?;
        *alvo = Some(v.parse::<f64>().map_err(em(&format!("{a} {v}")))?);
    }
    let (Ok([modelo, experimento, entrada]), Some(pause), Some(max_duration)) =
        (<[String; 3]>::try_from(posicionais), pause, max_duration)
    else {
        return Err(USO.into());
    };

    let texto = std::fs::read_to_string(&experimento).map_err(em(&experimento))?;
    let exp: Experimento = serde_json::from_str(&texto).map_err(em(&experimento))?;
    let texto = std::fs::read_to_string(&entrada).map_err(em(&entrada))?;
    let quadros = texto
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str::<Quadro>(l).map_err(em(&format!("{entrada}:{}", i + 1)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if quadros.len() < 2 {
        return Err(format!("{entrada}: precisa de 2 quadros ou mais para derivar o fps").into());
    }
    // shortcut: fps medio dos tempos `t`, exato no video; na camera o passo varia e
    // o relogio do nucleo supoe passo fixo. Rever quando houver captura ao vivo.
    let (t0, fim) = (quadros[0].t, quadros[quadros.len() - 1].t);
    let fps = (quadros.len() - 1) as f64 / (fim - t0);

    // Mesmo caminho do treino: features do nucleo e a regra de lacunas com o mesmo max_gap.
    let mut lacunas = GapFiller::new(exp.config.max_gap);
    let (mut features, mut visto) = (Vec::new(), Vec::new());
    for (i, q) in quadros.iter().enumerate() {
        let (left, right) = (
            com_nan(&q.left_hand, HAND_VALUES),
            com_nan(&q.right_hand, HAND_VALUES),
        );
        let (face, blend) = (
            com_nan(&q.face, FACE_VALUES),
            com_nan(&q.blendshapes, BLENDSHAPE_VALUES),
        );
        let pose = com_nan(&q.pose, POSE_VALUES);
        let lm = FrameLandmarks {
            left_hand: &left,
            right_hand: &right,
            face: &face,
            blendshapes: &blend,
            pose: &pose,
        };
        // Como o treino: largura / altura em f64, depois f32.
        let f = frame_features(&lm, (q.size[0] / q.size[1]) as f32)
            .map_err(em(&format!("{entrada}: quadro {i}")))?;
        visto.push(FrameQuality {
            reference: f.is_some(),
            face: q.face.is_some(),
        });
        features.extend_from_slice(lacunas.push(f.as_deref())?);
    }

    let mut model = Model::from_file(&modelo).map_err(em(&modelo))?;
    if model.classes() != exp.vocab.len() {
        return Err(format!(
            "{modelo} tem {} classes e o vocabulario de {experimento} tem {}",
            model.classes(),
            exp.vocab.len()
        )
        .into());
    }
    let probs = model.run(&features).map_err(em(&modelo))?;

    // O registro do treino nao traz o tipo do sinal: vale o padrao do protocolo.
    let vocab: Vec<Token> = exp
        .vocab
        .into_iter()
        .map(|gloss| Token {
            gloss,
            kind: ManualKind::default(),
        })
        .collect();
    let ending = Ending {
        pause,
        max_duration,
    };
    // O treino usa CTC com branco = 0.
    let mut t = Transcriber::new("u_", &vocab, 0, &[], SEM_EFEITO, ending, fps, t0)?;
    let mut enunciados = t.push(&probs, &[], &visto)?;
    enunciados.extend(t.finish());

    let mut saida = std::io::stdout().lock();
    for u in &enunciados {
        writeln!(saida, "{}", serde_json::to_string(u)?)?;
    }
    Ok(())
}
