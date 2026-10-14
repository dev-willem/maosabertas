//! Modelo de brinquedo, sem significado linguistico, gerado aqui mesmo em
//! protobuf ONNX: arquivo `.onnx` nao entra no repositorio.
//!
//! O grafo e `probs = Softmax(features x W)`. W liga a presenca da mao
//! esquerda a classe 1 e a da direita a classe 2; sem mao, as tres classes
//! empatam e vence a 0, o branco.

use maosabertas_core::{greedy_decode, Frames, Token, FEATURE_DIM, HAND_VALUES};
use maosabertas_onnx::{Error, Model};
use maosabertas_protocol::ManualKind;

const LEFT: usize = HAND_VALUES;
const RIGHT: usize = 2 * HAND_VALUES + 1;

// --- protobuf minimo -------------------------------------------------------

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn int(field: u64, v: u64, out: &mut Vec<u8>) {
    varint(field << 3, out);
    varint(v, out);
}

fn bytes(field: u64, data: &[u8], out: &mut Vec<u8>) {
    varint((field << 3) | 2, out);
    varint(data.len() as u64, out);
    out.extend_from_slice(data);
}

fn msg(field: u64, build: impl FnOnce(&mut Vec<u8>), out: &mut Vec<u8>) {
    let mut m = Vec::new();
    build(&mut m);
    bytes(field, &m, out);
}

/// ValueInfoProto float32 `[T, cols]`.
fn value_info(field: u64, name: &str, cols: usize, out: &mut Vec<u8>) {
    msg(
        field,
        |v| {
            bytes(1, name.as_bytes(), v);
            msg(
                2,
                |ty| {
                    msg(
                        1,
                        |tensor| {
                            int(1, 1, tensor); // FLOAT
                            msg(
                                2,
                                |shape| {
                                    msg(1, |d| bytes(2, b"T", d), shape);
                                    msg(1, |d| int(1, cols as u64, d), shape);
                                },
                                tensor,
                            );
                        },
                        ty,
                    )
                },
                v,
            );
        },
        out,
    );
}

/// ModelProto com `Softmax(features x W)`; `w` e `[dim, classes]` em linha.
fn toy_model(dim: usize, classes: usize, w: &[f32]) -> Vec<u8> {
    let mut m = Vec::new();
    int(1, 8, &mut m); // ir_version
    msg(8, |op| int(2, 17, op), &mut m); // opset ai.onnx 17
    msg(
        7,
        |g| {
            msg(
                1,
                |n| {
                    bytes(1, b"features", n);
                    bytes(1, b"W", n);
                    bytes(2, b"logits", n);
                    bytes(4, b"MatMul", n);
                },
                g,
            );
            // Softmax no opset 13+ ja usa o ultimo eixo.
            msg(
                1,
                |n| {
                    bytes(1, b"logits", n);
                    bytes(2, b"probs", n);
                    bytes(4, b"Softmax", n);
                },
                g,
            );
            bytes(2, b"brinquedo", g);
            msg(
                5,
                |t| {
                    int(1, dim as u64, t);
                    int(1, classes as u64, t);
                    int(2, 1, t); // FLOAT
                    bytes(8, b"W", t);
                    let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
                    bytes(9, &raw, t);
                },
                g,
            );
            value_info(11, "features", dim, g);
            value_info(12, "probs", classes, g);
        },
        &mut m,
    );
    m
}

fn model() -> Model {
    let mut w = vec![0.0; FEATURE_DIM * 3];
    w[LEFT * 3 + 1] = 10.0;
    w[RIGHT * 3 + 2] = 10.0;
    Model::from_memory(&toy_model(FEATURE_DIM, 3, &w)).expect("modelo de brinquedo")
}

/// Quadros com so a presenca das maos ligada: 0 nenhuma, 1 esquerda, 2 direita.
fn frames(seq: &[u8]) -> Vec<f32> {
    seq.iter()
        .flat_map(|&s| {
            let mut f = vec![0.0; FEATURE_DIM];
            match s {
                1 => f[LEFT] = 1.0,
                2 => f[RIGHT] = 1.0,
                _ => {}
            }
            f
        })
        .collect()
}

// --- testes ---------------------------------------------------------------

#[test]
fn carrega_e_infere() {
    let mut m = model();
    assert_eq!(m.classes(), 3);
    let probs = m.run(&frames(&[0, 1, 2])).unwrap();
    assert_eq!(probs.len(), 9);
    for row in probs.chunks(3) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    assert!((probs[0] - 1.0 / 3.0).abs() < 1e-5);
    assert!(probs[4] > 0.99 && probs[8] > 0.99);
}

#[test]
fn features_ate_glosa() {
    let vocab: Vec<Token> = ["<branco>", "A", "B"]
        .into_iter()
        .map(|g| Token {
            gloss: g.into(),
            kind: ManualKind::Lexical,
        })
        .collect();
    let mut m = model();
    let probs = m.run(&frames(&[0, 1, 1, 0, 2, 2, 0])).unwrap();
    let f = Frames::new(&probs, m.classes(), 10.0, 0.0).unwrap();
    let items = greedy_decode(&f, &vocab, 0).unwrap();
    let glosas: Vec<_> = items.iter().map(|i| i.gloss.as_str()).collect();
    assert_eq!(glosas, ["A", "B"]);
    assert_eq!(items[0].t, [0.1, 0.3]);
}

#[test]
fn sem_quadros() {
    assert!(model().run(&[]).unwrap().is_empty());
}

#[test]
fn recusa_entrada_invalida() {
    let mut m = model();
    assert!(matches!(m.run(&[0.0; 10]), Err(Error::Input(_))));
    let mut f = frames(&[1, 1]);
    f[FEATURE_DIM + 5] = f32::NAN;
    match m.run(&f) {
        Err(Error::Input(msg)) => assert!(msg.contains("quadro 1"), "{msg}"),
        other => panic!(
            "esperado erro de entrada, veio {other:?}",
            other = other.err()
        ),
    }
}

#[test]
fn recusa_modelo_com_outra_dimensao() {
    let w = vec![0.0; 10 * 3];
    match Model::from_memory(&toy_model(10, 3, &w)) {
        Err(Error::Contract(msg)) => assert!(msg.contains("entrada"), "{msg}"),
        Err(e) => panic!("esperado erro de contrato, veio {e}"),
        Ok(_) => panic!("modelo com dimensao errada foi aceito"),
    }
}
