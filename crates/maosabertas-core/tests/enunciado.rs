//! Caminho completo com dados sinteticos: probabilidades -> enunciado -> JSON.
//! Os nomes do vocabulario nao sao glosas de Libras.

use maosabertas_core::{greedy_decode, segment, utterance, Frames, Hysteresis, Token};
use maosabertas_protocol::{ManualKind, NonManualLabel, Tier, Utterance};

#[test]
fn probabilidades_viram_enunciado_valido() {
    let vocab: Vec<Token> = ["<branco>", "A", "B"]
        .into_iter()
        .map(|g| Token {
            gloss: g.into(),
            kind: ManualKind::Lexical,
        })
        .collect();
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
