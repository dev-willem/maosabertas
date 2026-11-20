//! Parametros fonologicos medidos por quadro a partir dos 21 pontos de cada mao
//! (ADR-009 e ADR-010): o que a configuracao de mao (CM), a locacao (PA), a
//! orientacao (O) e o movimento (M) precisam.
//!
//! So geometria deterministica. Nada aqui da nome a configuracao, locacao,
//! orientacao ou movimento: categoria e decisao linguistica (regra 3).
//!
//! Referencial do corpo, o mesmo do protocolo: origem no meio dos ombros,
//! unidade = largura dos ombros, x para a direita de quem sinaliza, y para cima,
//! z para a frente (de quem sinaliza para a camera). Vale para imagem sem
//! espelhamento: camera espelhada troca esquerda e direita.
//!
//! O z dos pontos da mao e relativo ao punho, porque o MediaPipe nao da a
//! profundidade da mao: serve para forma e orientacao, nao para locacao.

use crate::features::{check_aspect, check_parts, shoulders, Shoulders};
use crate::{Error, HAND_VALUES, POSE_VALUES};
use maosabertas_protocol::{Hand, HandPhonology, Location, Movement, Orientation};

/// Valores do descritor de configuracao de mao: 15 flexoes, 4 aberturas e 15 distancias.
pub const HANDSHAPE_DIM: usize = 34;

/// shortcut: os dois limiares vem do prototipo do ADR-009 (secao 6.1), escolhidos olhando
/// poucos clipes; calibrar contra anotacao humana de "segurar" quando houver.
/// Mao parada: os pontos andam menos que isto, em palmas por segundo (0,05 por quadro a 30 fps).
pub const HOLD_SPEED: f32 = 1.5;
/// Repouso: punho mais que isto abaixo do meio dos ombros, em larguras de ombro.
pub const REST_DEPTH: f32 = 1.0;
/// shortcut: a mao entra no item se ficar no espaco de sinalizacao em pelo menos esta
/// fracao dos quadros do trecho; calibrar com anotacao de mao ativa quando houver.
pub const MIN_VISIBLE: f64 = 0.5;

type Point = [f32; 3];

/// Cadeias do punho a ponta de cada dedo, do polegar ao minimo.
const FINGERS: [[usize; 5]; 5] = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
];
const TIPS: [usize; 5] = [4, 8, 12, 16, 20];
/// Base dos quatro dedos (articulacao com a palma).
const MCP: [usize; 4] = [5, 9, 13, 17];
/// Primeira falange de dedos vizinhos; no polegar, a de 2 para 3.
const NEIGHBORS: [(usize, usize); 4] = [(2, 5), (5, 9), (9, 13), (13, 17)];
/// Punho e bases dos dedos: o centro da palma e mais estavel que o da mao inteira.
const PALM: [usize; 5] = [0, 5, 9, 13, 17];

/// Mao de quem sinaliza, nao lado da imagem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Uma mao num quadro, no referencial do corpo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandPose {
    pub side: Side,
    /// 21 pontos do MediaPipe; z relativo ao punho.
    pub points: [Point; 21],
}

/// Maos de um quadro: esquerda e direita de quem sinaliza, `None` se nao detectada.
pub type Hands = [Option<HandPose>; 2];

/// As duas maos de um quadro no referencial do corpo. Cada mao no layout do `.npz`
/// (21 x 3, NaN se nao detectada) e a pose com 33 x 4; `aspect` e largura / altura
/// da imagem. Sem ombros nao ha referencia: as duas saem `None`.
pub fn hands(left: &[f32], right: &[f32], pose: &[f32], aspect: f32) -> Result<Hands, Error> {
    check_parts(&[
        ("left_hand", left.len(), HAND_VALUES),
        ("right_hand", right.len(), HAND_VALUES),
        ("pose", pose.len(), POSE_VALUES),
    ])?;
    check_aspect(aspect)?;
    let Some(body) = shoulders(pose, aspect) else {
        return Ok([None, None]);
    };
    Ok([
        to_body(left, Side::Left, body, aspect),
        to_body(right, Side::Right, body, aspect),
    ])
}

/// Imagem (x para a direita, y para baixo, z crescendo para longe da camera) para o
/// referencial do corpo: os tres eixos trocam de sentido. Quem sinaliza esta de frente,
/// entao a direita da imagem e a esquerda dele.
fn to_body(raw: &[f32], side: Side, b: Shoulders, aspect: f32) -> Option<HandPose> {
    if !raw.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut points = [[0.0; 3]; 21];
    for (p, c) in points.iter_mut().zip(raw.chunks_exact(3)) {
        // O z do MediaPipe tem aproximadamente a escala de x.
        *p = [
            -(c[0] * aspect - b.cx) / b.scale,
            -(c[1] - b.cy) / b.scale,
            -c[2] * aspect / b.scale,
        ];
    }
    Some(HandPose { side, points })
}

impl HandPose {
    /// Distancia media do punho as bases dos dedos, em larguras de ombro.
    pub fn palm_size(&self) -> f32 {
        palm_size(&self.points)
    }

    /// Centro da palma no plano da imagem (x, y), em larguras de ombro.
    pub fn palm_center(&self) -> [f32; 2] {
        let n = PALM.len() as f32;
        let sum = PALM.iter().fold([0.0, 0.0], |s, &i| {
            [s[0] + self.points[i][0], s[1] + self.points[i][1]]
        });
        [sum[0] / n, sum[1] / n]
    }

    /// Vetor unitario que sai da palma (para onde ela "olha"). `None` em mao degenerada.
    pub fn palm_normal(&self) -> Option<Point> {
        let p = &self.points;
        let (index, pinky) = (sub(p[5], p[0]), sub(p[17], p[0]));
        // A mao direita e o espelho da esquerda: a ordem do produto vetorial troca.
        unit(match self.side {
            Side::Left => cross(index, pinky),
            Side::Right => cross(pinky, index),
        })
    }

    /// Vetor unitario do punho para a base do dedo medio: para onde os dedos apontam.
    pub fn fingers(&self) -> Option<Point> {
        unit(sub(self.points[9], self.points[0]))
    }

    /// Descritor de configuracao de mao. Ver [`handshape`].
    pub fn handshape(&self) -> Option<[f32; HANDSHAPE_DIM]> {
        handshape(&self.points)
    }

    /// Punho acima do nivel de repouso (ver [`REST_DEPTH`]).
    pub fn in_signing_space(&self) -> bool {
        self.points[0][1] >= -REST_DEPTH
    }
}

/// Quadro de "segurar": a mao esta no espaco de sinalizacao e quase parada desde
/// o quadro anterior, `dt` segundos antes. Causal. Sem quadro anterior nunca e segurar.
/// Mao parada em repouso nao e sinal (ADR-009, 6.1).
pub fn is_hold(prev: Option<&HandPose>, cur: &HandPose, dt: f32) -> bool {
    let Some(prev) = prev else {
        return false;
    };
    let moved = prev
        .points
        .iter()
        .zip(&cur.points)
        .map(|(a, b)| (b[0] - a[0]).hypot(b[1] - a[1]))
        .sum::<f32>()
        / 21.0;
    // Comparacao falsa com NaN: mao degenerada ou `dt` invalido nao seguram.
    cur.in_signing_space() && moved / cur.palm_size() / dt < HOLD_SPEED
}

/// Descritor de configuracao de mao a partir dos 21 pontos, em qualquer referencial
/// com a mesma escala nos tres eixos. Usa so angulos e razoes de distancia, entao nao
/// muda com translacao, rotacao, escala nem espelhamento (mao esquerda e direita):
/// orientacao e locacao sao outros parametros.
///
/// Layout: 15 flexoes (3 por dedo, do polegar ao minimo), 4 aberturas entre dedos
/// vizinhos e 15 distancias (pontas entre si e ate o punho) divididas pelo tamanho
/// da palma. `None` se algum valor nao for finito (mao degenerada).
pub fn handshape(p: &[Point; 21]) -> Option<[f32; HANDSHAPE_DIM]> {
    let bone = |a: usize, b: usize| sub(p[b], p[a]);
    let palm = palm_size(p);
    let flexion = FINGERS
        .iter()
        .flat_map(|d| (1..4).map(move |i| angle(bone(d[i - 1], d[i]), bone(d[i], d[i + 1]))));
    let spread = NEIGHBORS
        .iter()
        .map(|&(a, b)| angle(bone(a, a + 1), bone(b, b + 1)));
    let pairs = TIPS
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| TIPS[i + 1..].iter().map(move |&b| (a, b)))
        .chain(TIPS.iter().map(|&t| (0, t)));
    let distance = pairs.map(|(a, b)| norm(bone(a, b)) / palm);

    let values: Vec<f32> = flexion.chain(spread).chain(distance).collect();
    debug_assert_eq!(values.len(), HANDSHAPE_DIM);
    let out: [f32; HANDSHAPE_DIM] = values.try_into().ok()?;
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// PA, O e M de uma mao num trecho, para o protocolo (ADR-011). `frames` sao os quadros
/// do trecho, em ordem, `dt` segundos um do outro, e todos da mesma mao.
///
/// - PA e O: mediana nos quadros de segurar; sem segurar, em todos os visiveis.
/// - M: centro da palma do primeiro ao ultimo quadro visivel, e o caminho entre eles.
/// - CM sai sem nada: o descritor nao entra no protocolo e ainda nao ha inventario.
///
/// `None` se a mao ficou no espaco de sinalizacao em menos de [`MIN_VISIBLE`] dos quadros.
/// Valores com duas casas decimais: o protocolo nao precisa de mais (ADR-011).
pub fn measure(frames: &[Option<HandPose>], dt: f32) -> Option<HandPhonology> {
    let visible: Vec<&HandPose> = frames
        .iter()
        .flatten()
        .filter(|h| h.in_signing_space())
        .collect();
    let (first, last) = (visible.first()?, visible.last()?);
    let fraction = visible.len() as f64 / frames.len() as f64;
    if fraction < MIN_VISIBLE {
        return None;
    }
    let holds: Vec<&HandPose> = frames
        .windows(2)
        .filter_map(|w| match w {
            [prev, Some(cur)] if is_hold(prev.as_ref(), cur, dt) => Some(cur),
            _ => None,
        })
        .collect();
    let still = if holds.is_empty() { &visible } else { &holds };

    let centers: Vec<[f32; 2]> = visible.iter().map(|h| h.palm_center()).collect();
    let path = centers
        .windows(2)
        .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
        .sum::<f32>();
    let (start, end) = (first.palm_center(), last.palm_center());
    let center = |i: usize| median(still.iter().map(|h| h.palm_center()[i]).collect());
    let direction = |v: Vec<Point>| {
        let m = |i: usize| median(v.iter().map(|p| p[i]).collect());
        unit([m(0), m(1), m(2)]).map(|u| u.map(round))
    };

    Some(HandPhonology {
        mao: match first.side {
            Side::Left => Hand::Esquerda,
            Side::Right => Hand::Direita,
        },
        visivel: Some(round(fraction as f32)),
        cm: None,
        pa: Some(Location {
            pos: Some([round(center(0)), round(center(1))]),
            ..Location::default()
        }),
        o: Some(Orientation {
            palma: direction(still.iter().filter_map(|h| h.palm_normal()).collect()),
            dedos: direction(still.iter().filter_map(|h| h.fingers()).collect()),
            ..Orientation::default()
        }),
        m: Some(Movement {
            desloc: Some([round(end[0] - start[0]), round(end[1] - start[1])]),
            trajeto: Some(round(path)),
            ..Movement::default()
        }),
    })
}

/// Duas casas decimais, sem zero negativo.
fn round(v: f32) -> f64 {
    (f64::from(v) * 100.0).round() / 100.0 + 0.0
}

/// Mediana; NaN se vazio.
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    match v.len() {
        0 => f32::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => (v[n / 2 - 1] + v[n / 2]) / 2.0,
    }
}

fn palm_size(p: &[Point; 21]) -> f32 {
    MCP.iter().map(|&m| norm(sub(p[m], p[0]))).sum::<f32>() / MCP.len() as f32
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Point) -> f32 {
    dot(a, a).sqrt()
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(v: Point) -> Option<Point> {
    let n = norm(v);
    (n > 1e-6).then(|| [v[0] / n, v[1] / n, v[2] / n])
}

/// Angulo entre dois vetores, em radianos. NaN se um deles for nulo.
fn angle(u: Point, v: Point) -> f32 {
    (dot(u, v) / (norm(u) * norm(v))).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mao de brinquedo, nao de pessoa: mao esquerda com a palma voltada para quem
    /// sinaliza e os dedos para cima, no referencial do corpo. `curva[i]` dobra cada
    /// articulacao do dedo i na direcao da palma (para tras, z negativo).
    fn esquerda_palma_para_tras(curva: [f32; 5]) -> [Point; 21] {
        let mut p = [[0.0; 3]; 21];
        let bases = [
            (-0.45, 0.25, -1.0_f32),
            (-0.3, 1.0, -0.25),
            (0.0, 1.05, 0.0),
            (0.25, 1.0, 0.2),
            (0.45, 0.9, 0.4),
        ];
        for (k, (d, (bx, by, phi))) in FINGERS.iter().zip(bases).enumerate() {
            let dir = [phi.sin(), phi.cos()];
            let mut q = [bx, by, 0.0];
            p[d[1]] = q;
            let mut ang = 0.0_f32;
            for (&j, comp) in d[2..].iter().zip([0.4, 0.3, 0.25]) {
                ang += curva[k];
                q = [
                    q[0] + comp * ang.cos() * dir[0],
                    q[1] + comp * ang.cos() * dir[1],
                    q[2] - comp * ang.sin(),
                ];
                p[j] = q;
            }
        }
        p
    }

    const ABERTA: [f32; 5] = [0.1; 5];
    const FECHADA: [f32; 5] = [0.2, 1.3, 1.3, 1.3, 1.3];

    /// Rotacao qualquer (angulos de Euler fixos).
    fn gira(v: Point) -> Point {
        let (a, b, c) = (0.7_f32, -1.1_f32, 2.3_f32);
        let v = [
            v[0],
            a.cos() * v[1] - a.sin() * v[2],
            a.sin() * v[1] + a.cos() * v[2],
        ];
        let v = [
            b.cos() * v[0] + b.sin() * v[2],
            v[1],
            -b.sin() * v[0] + b.cos() * v[2],
        ];
        [
            c.cos() * v[0] - c.sin() * v[1],
            c.sin() * v[0] + c.cos() * v[1],
            v[2],
        ]
    }

    /// Meia volta em torno do eixo vertical: a palma passa a olhar para a frente.
    fn meia_volta(v: Point) -> Point {
        [-v[0], v[1], -v[2]]
    }

    fn espelho(v: Point) -> Point {
        [-v[0], v[1], v[2]]
    }

    fn escala(k: f32) -> impl Fn(Point) -> Point {
        move |v| [k * v[0], k * v[1], k * v[2]]
    }

    fn perto(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn descritor_invariante_a_rotacao_escala_translacao_e_espelho() {
        for curva in [ABERTA, FECHADA] {
            let mao = esquerda_palma_para_tras(curva);
            let d = handshape(&mao).unwrap();
            let movida = mao.map(|v| {
                let g = gira(v);
                [3.7 * g[0] + 5.0, 3.7 * g[1] - 2.0, 3.7 * g[2] + 1.0]
            });
            assert!(perto(&d, &handshape(&movida).unwrap(), 1e-4));
            assert!(perto(&d, &handshape(&mao.map(espelho)).unwrap(), 1e-6));
        }
    }

    #[test]
    fn descritor_distingue_formas_e_recusa_mao_degenerada() {
        let aberta = handshape(&esquerda_palma_para_tras(ABERTA)).unwrap();
        let fechada = handshape(&esquerda_palma_para_tras(FECHADA)).unwrap();
        let maior = aberta
            .iter()
            .zip(&fechada)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(maior > 0.5, "diferenca maxima {maior}");
        assert_eq!(handshape(&[[0.0; 3]; 21]), None);
    }

    /// Mao aberta do lado pedido, palma para tras, transformada por `f`.
    fn mao(side: Side, f: impl Fn(Point) -> Point) -> HandPose {
        let base = esquerda_palma_para_tras(ABERTA);
        let base = match side {
            Side::Left => base,
            Side::Right => base.map(espelho),
        };
        HandPose {
            side,
            points: base.map(f),
        }
    }

    #[test]
    fn normal_da_palma_e_direcao_dos_dedos() {
        for side in [Side::Left, Side::Right] {
            let tras = mao(side, |v| v);
            assert!(perto(&tras.palm_normal().unwrap(), &[0.0, 0.0, -1.0], 1e-6));
            let frente = mao(side, meia_volta);
            assert!(perto(
                &frente.palm_normal().unwrap(),
                &[0.0, 0.0, 1.0],
                1e-6
            ));
            assert!(perto(&tras.fingers().unwrap(), &[0.0, 1.0, 0.0], 1e-6));
        }
        // Gira junto com a mao.
        let girada = mao(Side::Right, gira);
        let esperado = gira([0.0, 0.0, -1.0]);
        assert!(perto(&girada.palm_normal().unwrap(), &esperado, 1e-5));
    }

    /// Quadro de imagem 1:1 com ombros em x = 0,75 (esquerdo de quem sinaliza) e 0,25
    /// (direito), y = 0,5: centro (0,5, 0,5) e escala 0,5.
    fn pose_da_imagem() -> Vec<f32> {
        let mut pose = vec![0.5; POSE_VALUES];
        pose[11 * 4] = 0.75;
        pose[12 * 4] = 0.25;
        pose
    }

    /// Do referencial do corpo para a imagem da pose acima, com z relativo ao punho.
    fn na_imagem(p: &[Point; 21]) -> Vec<f32> {
        p.iter()
            .flat_map(|v| [0.5 - v[0] * 0.5, 0.5 - v[1] * 0.5, -(v[2] - p[0][2]) * 0.5])
            .collect()
    }

    #[test]
    fn mao_direita_de_frente_para_a_camera() {
        // Mao direita a direita de quem sinaliza, na altura do peito, palma para a camera.
        let corpo = mao(Side::Right, |v| meia_volta(escala(0.2)(v)))
            .points
            .map(|v| [v[0] + 0.5, v[1] - 0.6, v[2]]);
        let imagem = na_imagem(&corpo);
        // Convencao do MediaPipe: a direita de quem sinaliza aparece na esquerda da imagem.
        assert!(imagem[0] < 0.5);

        let nada = vec![f32::NAN; HAND_VALUES];
        let [esquerda, direita] = hands(&nada, &imagem, &pose_da_imagem(), 1.0).unwrap();
        assert_eq!(esquerda, None);
        let direita = direita.unwrap();
        assert_eq!(direita.side, Side::Right);
        assert!(perto(
            &direita.palm_normal().unwrap(),
            &[0.0, 0.0, 1.0],
            1e-5
        ));
        let centro = direita.palm_center();
        assert!(centro[0] > 0.4 && centro[1] < 0.0, "{centro:?}");
        assert!(perto(
            &direita.handshape().unwrap(),
            &handshape(&esquerda_palma_para_tras(ABERTA)).unwrap(),
            1e-4
        ));
    }

    #[test]
    fn sem_ombros_sem_maos_e_entradas_invalidas() {
        let m = na_imagem(&esquerda_palma_para_tras(ABERTA));
        let sem_ombros = vec![f32::NAN; POSE_VALUES];
        assert_eq!(hands(&m, &m, &sem_ombros, 1.0).unwrap(), [None, None]);
        assert_eq!(
            hands(&m[1..], &m, &pose_da_imagem(), 1.0).unwrap_err(),
            Error::Landmarks {
                part: "left_hand",
                expected: HAND_VALUES,
                got: HAND_VALUES - 1
            }
        );
        assert_eq!(
            hands(&m, &m, &pose_da_imagem(), 0.0).unwrap_err(),
            Error::Aspect(0.0)
        );
    }

    #[test]
    fn segurar_exige_mao_parada_fora_do_repouso() {
        let parada = mao(Side::Right, escala(0.2));
        let dt = 1.0 / 30.0;
        assert!(!is_hold(None, &parada, dt));
        assert!(is_hold(Some(&parada), &parada, dt));

        // Meia palma em 1/30 s passa muito de 1,5 palma por segundo; em 1 s, nao.
        let palma = parada.palm_size();
        let andou = HandPose {
            points: parada.points.map(|v| [v[0] + 0.5 * palma, v[1], v[2]]),
            ..parada
        };
        assert!(!is_hold(Some(&parada), &andou, dt));
        assert!(is_hold(Some(&parada), &andou, 1.0));

        // Parada, mas com o punho mais de uma largura de ombro abaixo dos ombros.
        let repouso = HandPose {
            points: parada.points.map(|v| [v[0], v[1] - 1.5, v[2]]),
            ..parada
        };
        assert!(!repouso.in_signing_space());
        assert!(!is_hold(Some(&repouso), &repouso, dt));
        assert!(!is_hold(Some(&parada), &parada, f32::NAN));
    }

    /// Mao direita aberta, palma para a frente, no peito, deslocada `x` para a direita.
    fn no_peito(x: f32) -> HandPose {
        let m = mao(Side::Right, |v| meia_volta(escala(0.2)(v)));
        HandPose {
            points: m.points.map(|v| [v[0] + 0.3 + x, v[1] - 0.6, v[2]]),
            ..m
        }
    }

    #[test]
    fn mede_mao_parada_no_trecho() {
        let quadros = [Some(no_peito(0.0)); 4];
        let f = measure(&quadros, 1.0 / 30.0).unwrap();
        assert_eq!(f.mao, Hand::Direita);
        assert_eq!(f.visivel, Some(1.0));
        assert_eq!(f.cm, None);
        let c = no_peito(0.0).palm_center();
        assert_eq!(
            f.pa.unwrap().pos,
            Some([round(c[0]), round(c[1])]),
            "centro {c:?}"
        );
        let o = f.o.unwrap();
        assert_eq!(o.palma, Some([0.0, 0.0, 1.0]));
        assert_eq!(o.dedos, Some([0.0, 1.0, 0.0]));
        let m = f.m.unwrap();
        assert_eq!((m.desloc, m.trajeto), (Some([0.0, 0.0]), Some(0.0)));
    }

    #[test]
    fn mede_deslocamento_e_caminho() {
        // Ida e volta: deslocamento 0,1, caminho 0,5.
        let quadros = [0.0, 0.1, 0.2, 0.3, 0.1].map(|x| Some(no_peito(x)));
        let m = measure(&quadros, 1.0 / 30.0).unwrap().m.unwrap();
        assert_eq!(m.desloc, Some([0.1, 0.0]));
        assert_eq!(m.trajeto, Some(0.5));
    }

    #[test]
    fn mao_pouco_visivel_ou_em_repouso_nao_entra() {
        let repouso = HandPose {
            points: no_peito(0.0).points.map(|v| [v[0], v[1] - 1.0, v[2]]),
            ..no_peito(0.0)
        };
        let q = Some(no_peito(0.0));
        assert!(measure(&[q, None, None], 0.1).is_none());
        assert!(measure(&[q, Some(repouso), Some(repouso)], 0.1).is_none());
        assert!(measure(&[q, q, None], 0.1).is_some());
        assert!(measure(&[], 0.1).is_none());
    }
}
