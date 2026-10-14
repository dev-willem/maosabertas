//! Caminho completo com dados sinteticos: probabilidades -> enunciado -> JSON.
//! Os nomes do vocabulario nao sao glosas de Libras.

use maosabertas_core::{
    greedy_decode, segment, utterance, Ending, FrameQuality, Frames, Hysteresis, Token, Transcriber,
};
use maosabertas_protocol::{ManualKind, NonManualLabel, Tier, Utterance};

fn vocab() -> Vec<Token> {
    ["<branco>", "A", "B"]
        .into_iter()
        .map(|g| Token {
            gloss: g.into(),
            kind: ManualKind::Lexical,
        })
        .collect()
}

/// Validado contra o schema por `tools/validate_spec.py`.
const GOLDEN: &str = include_str!("data/enunciado-sintetico.json");

#[test]
fn fluxo_gera_o_enunciado_de_referencia() {
    let vocab = vocab();
    let labels = [(Tier::Gramatical, NonManualLabel::InterrogativaSn)];
    let h = Hysteresis {
        enter: 0.75,
        exit: 0.5,
        min_frames: 1,
    };
    let ending = Ending {
        pause: 0.75,
        max_duration: 10.0,
    };
    let mut t = Transcriber::new("sintetico_", &vocab, 0, &labels, h, ending, 4.0, 10.0).unwrap();

    // Por quadro a 4 fps: classe manual vencedora e sua probabilidade,
    // probabilidade do marcador, ombros e rosto vistos.
    let seq = [
        (1, 0.75, 0.25, true, true),
        (1, 0.5, 1.0, true, true),
        (0, 0.75, 1.0, true, false),
        (2, 1.0, 0.875, false, true),
        (2, 0.75, 0.5, false, true),
        (0, 0.75, 0.25, true, true),
        (0, 0.75, 0.0, true, true),
        (0, 0.75, 0.0, true, true),
        (2, 0.75, 0.0, true, true),
        (2, 0.75, 0.0, true, true),
    ];
    let manual: Vec<f32> = seq
        .iter()
        .flat_map(|&(c, p, ..)| {
            let mut row = [(1.0 - p) / 2.0; 3];
            row[c] = p;
            row
        })
        .collect();
    let nonmanual: Vec<f32> = seq.iter().map(|q| q.2).collect();
    let seen: Vec<_> = seq
        .iter()
        .map(|&(.., reference, face)| FrameQuality { reference, face })
        .collect();

    let fechados = t.push(&manual, &nonmanual, &seen).unwrap();
    assert_eq!(fechados.len(), 1, "a pausa fecha o primeiro enunciado");
    let gerado = serde_json::to_value(&fechados[0]).unwrap();
    let golden: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    assert_eq!(
        gerado,
        golden,
        "gerado:\n{}",
        serde_json::to_string_pretty(&gerado).unwrap()
    );

    let resto = t.finish().unwrap();
    assert_eq!(
        (
            resto.utterance_id.as_str(),
            resto.t_start,
            resto.gloss_line()
        ),
        ("sintetico_2", 12.0, "B".into())
    );
}

#[test]
fn probabilidades_viram_enunciado_valido() {
    let vocab = vocab();
    // 4 quadros a 4 fps: A, A, branco, B.
    let manual = [
        0.0, 1.0, 0.0, //
        0.0, 1.0, 0.0, //
        1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0,
    ];
    // Uma classe facial ativa em toda a janela.
    let facial = [1.0, 1.0, 1.0, 1.0];

    let fm = Frames::new(&manual, 3, 4.0, 10.0).unwrap();
    let ff = Frames::new(&facial, 1, 4.0, 10.0).unwrap();
    let h = Hysteresis {
        enter: 0.75,
        exit: 0.5,
        min_frames: 1,
    };
    let u = utterance(
        "sintetico_001",
        fm.window(),
        greedy_decode(&fm, &vocab, 0).unwrap(),
        segment(
            &ff,
            &[(Tier::Gramatical, NonManualLabel::InterrogativaSn)],
            h,
        )
        .unwrap(),
    );

    assert_eq!((u.t_start, u.t_end), (10.0, 11.0));
    assert_eq!(u.gloss_line(), "A B");
    // O marcador cobre os dois sinais: escopo proprio, nao preso a um sinal.
    assert_eq!(u.markers_at(10.1).count(), 1);
    assert_eq!(u.markers_at(10.9).count(), 1);

    let json = serde_json::to_string(&u).unwrap();
    assert!(!json.contains("translation"));
    assert_eq!(serde_json::from_str::<Utterance>(&json).unwrap(), u);
}
