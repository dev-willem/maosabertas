//! Features por quadro a partir dos landmarks do MediaPipe Holistic.
//!
//! O treino usa este mesmo codigo (via `maosabertas-py`), para que o modelo
//! receba no tempo real exatamente o que viu no treino.

use crate::Error;

/// Valores por parte em um quadro, no layout do `.npz` da extracao.
pub const HAND_VALUES: usize = 21 * 3;
pub const FACE_VALUES: usize = 478 * 3;
pub const BLENDSHAPE_VALUES: usize = 52;
pub const POSE_VALUES: usize = 33 * 4;

// Indices do MediaPipe.
const SHOULDERS: (usize, usize) = (11, 12);
/// Ombros, cotovelos e punhos.
const ARMS: [usize; 6] = [11, 12, 13, 14, 15, 16];
/// Ponta do nariz, testa, queixo e laterais do rosto: posicao e giro da cabeca.
const HEAD: [usize; 5] = [1, 10, 152, 234, 454];

/// Layout: mao esquerda (63 + presenca), mao direita (63 + presenca),
/// bracos (6 x 2), cabeca (5 x 2), blendshapes (52) e presenca do rosto.
pub const FEATURE_DIM: usize =
    2 * (HAND_VALUES + 1) + ARMS.len() * 2 + HEAD.len() * 2 + BLENDSHAPE_VALUES + 1;

/// Landmarks de um quadro. NaN onde nada foi detectado, como no `.npz`.
#[derive(Debug, Clone, Copy)]
pub struct FrameLandmarks<'a> {
    /// 21 x (x, y, z), normalizados na imagem.
    pub left_hand: &'a [f32],
    pub right_hand: &'a [f32],
    /// 478 x (x, y, z).
    pub face: &'a [f32],
    pub blendshapes: &'a [f32],
    /// 33 x (x, y, z, visibilidade).
    pub pose: &'a [f32],
}

/// Coordenadas centradas no meio dos ombros e divididas pela distancia entre
/// eles, em unidades proporcionais ao pixel (`aspect` = largura / altura da
/// imagem). Assim a feature nao depende de onde a pessoa esta no quadro nem
/// da distancia ate a camera.
///
/// Mao ou rosto ausente vira zeros, com indicador de presenca 0. Sem ombros
/// nao ha referencia: devolve `None`.
pub fn frame_features(lm: &FrameLandmarks, aspect: f32) -> Result<Option<Vec<f32>>, Error> {
    for (part, got, expected) in [
        ("left_hand", lm.left_hand.len(), HAND_VALUES),
        ("right_hand", lm.right_hand.len(), HAND_VALUES),
        ("face", lm.face.len(), FACE_VALUES),
        ("blendshapes", lm.blendshapes.len(), BLENDSHAPE_VALUES),
        ("pose", lm.pose.len(), POSE_VALUES),
    ] {
        if got != expected {
            return Err(Error::Landmarks {
                part,
                expected,
                got,
            });
        }
    }
    if !(aspect.is_finite() && aspect > 0.0) {
        return Err(Error::Aspect(aspect));
    }

    let pose = |i: usize| (lm.pose[i * 4] * aspect, lm.pose[i * 4 + 1]);
    let (l, r) = (pose(SHOULDERS.0), pose(SHOULDERS.1));
    let scale = (l.0 - r.0).hypot(l.1 - r.1);
    // Tambem cobre NaN: ombros ausentes.
    if scale.is_nan() || scale <= 1e-6 {
        return Ok(None);
    }
    let (cx, cy) = ((l.0 + r.0) / 2.0, (l.1 + r.1) / 2.0);
    let norm = |x: f32, y: f32| [(x * aspect - cx) / scale, (y - cy) / scale];

    let mut out = Vec::with_capacity(FEATURE_DIM);
    for hand in [lm.left_hand, lm.right_hand] {
        if hand[0].is_nan() {
            out.extend([0.0; HAND_VALUES + 1]);
            continue;
        }
        for p in hand.chunks_exact(3) {
            out.extend(norm(p[0], p[1]));
            // O z do MediaPipe tem aproximadamente a escala de x.
            out.push(p[2] * aspect / scale);
        }
        out.push(1.0);
    }
    for i in ARMS {
        out.extend(norm(lm.pose[i * 4], lm.pose[i * 4 + 1]));
    }
    if lm.face[0].is_nan() {
        out.extend([0.0; HEAD.len() * 2 + BLENDSHAPE_VALUES + 1]);
    } else {
        for i in HEAD {
            out.extend(norm(lm.face[i * 3], lm.face[i * 3 + 1]));
        }
        out.extend(
            lm.blendshapes
                .iter()
                .map(|&b| if b.is_nan() { 0.0 } else { b }),
        );
        out.push(1.0);
    }
    debug_assert_eq!(out.len(), FEATURE_DIM);
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quadro sintetico: ombros em x = 0,25 e 0,75 (y = 0,5), maos e rosto ausentes.
    struct Frame {
        left: Vec<f32>,
        right: Vec<f32>,
        face: Vec<f32>,
        blend: Vec<f32>,
        pose: Vec<f32>,
    }

    impl Frame {
        fn new() -> Self {
            let mut pose = vec![0.5; POSE_VALUES];
            pose[SHOULDERS.0 * 4] = 0.75;
            pose[SHOULDERS.1 * 4] = 0.25;
            Self {
                left: vec![f32::NAN; HAND_VALUES],
                right: vec![f32::NAN; HAND_VALUES],
                face: vec![f32::NAN; FACE_VALUES],
                blend: vec![f32::NAN; BLENDSHAPE_VALUES],
                pose,
            }
        }

        fn features(&self, aspect: f32) -> Result<Option<Vec<f32>>, Error> {
            frame_features(
                &FrameLandmarks {
                    left_hand: &self.left,
                    right_hand: &self.right,
                    face: &self.face,
                    blendshapes: &self.blend,
                    pose: &self.pose,
                },
                aspect,
            )
        }
    }

    #[test]
    fn dimensao() {
        assert_eq!(FEATURE_DIM, 203);
    }

    #[test]
    fn centra_nos_ombros_e_escala_pela_largura() {
        let mut f = Frame::new();
        f.left = [1.0, 0.75, 0.125].repeat(21);
        let out = f.features(1.0).unwrap().unwrap();
        assert_eq!(out.len(), FEATURE_DIM);
        assert_eq!(out[..3], [1.0, 0.5, 0.25]);
        assert_eq!(out[HAND_VALUES], 1.0);
        // Mao direita ausente: zeros e presenca 0.
        assert!(out[HAND_VALUES + 1..2 * (HAND_VALUES + 1)]
            .iter()
            .all(|&v| v == 0.0));
        // Bracos: ombros a meia largura do centro, para cada lado.
        let arms = 2 * (HAND_VALUES + 1);
        assert_eq!(out[arms..arms + 4], [0.5, 0.0, -0.5, 0.0]);
        // Rosto ausente: tudo zero no fim.
        assert!(out[arms + 12..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn corrige_proporcao_da_imagem() {
        let mut f = Frame::new();
        f.left = [1.0, 0.75, 0.125].repeat(21);
        // Largura = 2 x altura: ombros a 1,0 de distancia, centro em x = 1,0.
        let out = f.features(2.0).unwrap().unwrap();
        assert_eq!(out[..3], [1.0, 0.25, 0.25]);
    }

    #[test]
    fn rosto_presente_entra_com_blendshapes() {
        let mut f = Frame::new();
        f.face = vec![0.5; FACE_VALUES];
        f.blend = vec![0.25; BLENDSHAPE_VALUES];
        f.blend[0] = f32::NAN;
        let out = f.features(1.0).unwrap().unwrap();
        let head = 2 * (HAND_VALUES + 1) + 12;
        assert!(out[head..head + 10].iter().all(|&v| v == 0.0));
        assert_eq!(out[head + 10], 0.0);
        assert_eq!(out[head + 11], 0.25);
        assert_eq!(out[FEATURE_DIM - 1], 1.0);
    }

    #[test]
    fn sem_ombros_nao_ha_referencia() {
        let mut f = Frame::new();
        f.pose = vec![f32::NAN; POSE_VALUES];
        assert_eq!(f.features(1.0).unwrap(), None);
        let mut f = Frame::new();
        f.pose[SHOULDERS.1 * 4] = 0.75;
        assert_eq!(f.features(1.0).unwrap(), None);
    }

    #[test]
    fn recusa_tamanho_e_proporcao_invalidos() {
        let mut f = Frame::new();
        f.face.pop();
        assert_eq!(
            f.features(1.0).unwrap_err(),
            Error::Landmarks {
                part: "face",
                expected: FACE_VALUES,
                got: FACE_VALUES - 1
            }
        );
        assert_eq!(Frame::new().features(0.0).unwrap_err(), Error::Aspect(0.0));
    }
}
