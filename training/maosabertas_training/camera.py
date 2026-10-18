"""Camera local para testar e aprender os sinais de um modelo treinado.

Uso:
    python -m maosabertas_training.camera PASTA_DO_TREINO [--camera 0]

PASTA_DO_TREINO tem o `model.onnx` e o `experiment.json` que o `train.py` grava.
A tela mostra os landmarks, as glosas reconhecidas e uma aula: um clipe de
referencia do corpus de treino, sinal a sinal, com a lista do que voce ja acertou.

Teclas: n ou seta direita, proximo sinal; p ou seta esquerda, anterior; Esc ou q,
sair. Nada e gravado: nem video, nem landmarks.

Ferramenta de teste local, fora do caminho de producao (que e o nucleo em Rust).
Os rotulos sao os do dataset do treino, nao glosas revistas do projeto.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from maosabertas_training.corpus import read_manifest
from maosabertas_training.evaluate import greedy_decode
from maosabertas_training.features import features, fill_gaps
from maosabertas_training.landmarks import frame_arrays, holistic, model_path

# fps do corpus do modelo base: o modelo aprendeu o tempo dos sinais nesse ritmo.
FPS = 30
# Trecho com probabilidade media abaixo disto nao vira legenda. No teste do modelo base
# (2 sinalizantes fora do treino), corta 6 dos 15 trechos errados e 12 dos 175 certos.
MIN_CONF = 0.7
# Enquadramento do corpus do modelo base, p10 a p90 dos quadros: ombros a 0,38-0,56 da altura
# da imagem e largura dos ombros de 0,28-0,35 da altura. As dicas usam a faixa com folga.
SHOULDER_Y = (0.33, 0.62)
SHOULDER_W = (0.24, 0.40)
# Mediana do mesmo corpus: altura e largura dos ombros, em fracao da altura da imagem.
TARGET = (0.45, 0.315)

WINDOW = "Maos Abertas"
# BGR. Esquerda e direita de quem sinaliza (conferido na camera em 2026-10-18).
HANDS = {"left_hand": (255, 128, 0), "right_hand": (0, 160, 255)}
# Pontos da pose do quadril para cima: as pernas ficam fora do quadro.
BODY = 25
WHITE, GRAY, GREEN, YELLOW = (255, 255, 255), (170, 170, 170), (80, 220, 80), (0, 220, 255)


class Live:
    """Quadros da camera -> glosas, no relogio de FPS do treino.

    `run` leva features (T, FEATURE_DIM) a probabilidades (T, C); classe 0 e o branco.
    """

    def __init__(
        self,
        run: Callable[[np.ndarray], np.ndarray],
        vocab: list[str],
        max_gap: int,
        receptive_field: int,
    ):
        self.run, self.vocab, self.max_gap = run, vocab, max_gap
        # Basta para a saida dos ultimos FPS quadros nao depender da borda da janela.
        self.rows: deque[np.ndarray] = deque(maxlen=receptive_field + max_gap + FPS)
        self.ticks = 0
        # Trecho corrente: classe, soma das probabilidades, quadros, alguma mao vista.
        self.current: tuple[int, float, int, bool] | None = None
        self.probs: np.ndarray | None = None

    def push(self, t: float, row: np.ndarray, hand: bool) -> list[str]:
        """`row`: features do quadro no instante `t`, em segundos (NaN sem ombros);
        `hand`: alguma mao detectada no quadro. Devolve as glosas que fecharam.
        """
        # A camera roda mais devagar que o treino: o quadro se repete ate o relogio alcancar t.
        # Depois de uma parada longa, so o ultimo segundo e reposto.
        now = int(t * FPS) + 1
        n = min(now - self.ticks, FPS)
        if n < 1:
            return []
        self.ticks = now
        self.rows.extend([row] * n)
        probs = self.run(fill_gaps(np.asarray(self.rows), self.max_gap))[-n:]
        self.probs = probs[-1]
        # shortcut: CTC guloso em Python, como em evaluate.py; a referencia e o CtcDecoder de
        # crates/maosabertas-core. Trocar pelo binding quando o maosabertas-py expuser o decodificador.
        out = []
        for p in probs:
            c = int(p.argmax())
            if self.current and self.current[0] == c:
                _, total, frames, seen = self.current
                self.current = (c, total + float(p[c]), frames + 1, seen or hand)
            else:
                out += self._close()
                self.current = (c, float(p[c]), 1, hand)
        return out

    def _close(self) -> list[str]:
        if self.current is None:
            return []
        c, total, frames, hand = self.current
        # Sem mao nao ha sinal manual: na camera, a mao fora do quadro gerava glosa falsa.
        return [self.vocab[c]] if c != 0 and hand and total / frames >= MIN_CONF else []


def framing(pose: np.ndarray, aspect: float) -> str | None:
    """Dica para chegar ao enquadramento do corpus de treino; None se ja esta dentro."""
    left, right = pose[11], pose[12]
    if np.isnan(left[0]) or np.isnan(right[0]):
        return "fique com os ombros no quadro"
    width = abs(left[0] - right[0]) * aspect
    y = (left[1] + right[1]) / 2
    if width > SHOULDER_W[1]:
        return "afaste-se da camera"
    if width < SHOULDER_W[0]:
        return "aproxime-se da camera"
    if y > SHOULDER_Y[1]:
        return "abaixe a camera (ombros baixos no quadro)"
    if y < SHOULDER_Y[0]:
        return "suba a camera (ombros altos no quadro)"
    return None


def references(record: dict, run: Callable[[np.ndarray], np.ndarray]) -> dict[int, tuple]:
    """Um clipe por glosa para a aula, de preferencia um que o modelo acerta.

    Devolve classe -> (pontos no quadrado unitario e espelhados, pseudonimo do sinalizante).
    """
    manifest = Path(record["config"]["manifest"])
    if not manifest.exists():
        return {}
    chosen: dict[int, tuple[bool, Path, str]] = {}
    tries: Counter[int] = Counter()
    for clip in read_manifest(manifest, record["vocab"]):
        if len(clip.labels) != 1:
            continue
        g = clip.labels[0]
        if (g in chosen and chosen[g][0]) or tries[g] >= 5:
            continue
        tries[g] += 1
        probs = run(fill_gaps(features(np.load(clip.path)), record["config"]["max_gap"]))
        hit = greedy_decode(probs) == [g]
        if hit or g not in chosen:
            chosen[g] = (hit, clip.path, clip.signer)
    return {g: (_fit(np.load(path)), signer) for g, (_, path, signer) in chosen.items()}


def _fit(data) -> dict[str, np.ndarray]:
    """Pontos (T, N, 2) do clipe no quadrado unitario, espelhados como a tela."""
    w, h = (float(v) for v in data["size"])
    parts = {k: data[k][..., :2] * (w, h) for k in ("face", *HANDS)}
    parts["pose"] = data["pose"][:, :BODY, :2] * (w, h)
    body = np.concatenate([parts[k].reshape(-1, 2) for k in ("pose", *HANDS)])
    lo, hi = np.nanmin(body, 0), np.nanmax(body, 0)
    side = (hi - lo).max() * 1.1
    corner = (lo + hi) / 2 - side / 2
    out = {}
    for k, p in parts.items():
        q = (p - corner) / side
        q[..., 0] = 1 - q[..., 0]
        out[k] = q
    return out


def _ascii(text: str) -> str:
    # shortcut: cv2.putText so desenha ASCII, entao os acentos somem na tela (MAÇÃ sai MACA).
    # Trocar por Pillow se a dependencia for aprovada.
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _edges(connections) -> list[tuple[int, int]]:
    return [(c.start, c.end) for c in connections if c.start < BODY and c.end < BODY]


def _draw(img: np.ndarray, points: np.ndarray, edges, color, dot: int = 0) -> None:
    if np.isnan(points[0, 0]):
        return
    h, w = img.shape[:2]
    xy = [(int(x * w), int(y * h)) for x, y in points[:, :2]]
    for a, b in edges:
        cv2.line(img, xy[a], xy[b], color, 2, cv2.LINE_AA)
    if dot:
        for p in xy:
            cv2.circle(img, p, dot, color, -1, cv2.LINE_AA)


def _shade(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.65) -> None:
    roi = img[max(y0, 0) : y1, max(x0, 0) : x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def _text(img, text, org, scale, color=WHITE, thick=1) -> None:
    cv2.putText(img, _ascii(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("treino", type=Path, help="pasta com model.onnx e experiment.json")
    parser.add_argument("--camera", type=int, default=0, help="indice da camera")
    args = parser.parse_args(argv)

    import mediapipe as mp
    import onnxruntime as ort
    from mediapipe.tasks.python import vision

    record = json.loads((args.treino / "experiment.json").read_text(encoding="utf-8"))
    vocab = record["vocab"]
    session = ort.InferenceSession(
        str(args.treino / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name

    def run(x: np.ndarray) -> np.ndarray:
        return session.run(None, {name: x})[0]

    live = Live(run, vocab, record["config"]["max_gap"], record["model"]["receptive_field_frames"])
    print("escolhendo os clipes de referencia...", flush=True)
    refs = references(record, run)
    shown = ["(nenhum sinal)", *vocab[1:]]

    hand = _edges(vision.HandLandmarksConnections.HAND_CONNECTIONS)
    pose = _edges(vision.PoseLandmarksConnections.POSE_LANDMARKS)
    face = [(c.start, c.end) for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"nao foi possivel abrir a camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Cabe numa tela de 1366 x 768 com a barra de tarefas; da para maximizar.
    cv2.resizeWindow(WINDOW, 1120, 630)

    caption: deque[str] = deque(maxlen=8)
    target, done = 1, set()
    advance_at, last_hit = 0.0, ""
    start = last = lesson = time.monotonic()
    last_ms, rate = -1, 0.0

    def go(step: int) -> None:
        nonlocal target, lesson
        target = (target - 1 + step) % (len(vocab) - 1) + 1
        lesson = time.monotonic()

    try:
        with holistic(model_path()) as landmarker:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                t, dt, last = now - start, now - last, now
                rate = 0.9 * rate + 0.1 / dt if dt > 0 else rate
                # O modo VIDEO exige instantes estritamente crescentes, em ms inteiros.
                last_ms = max(int(t * 1000), last_ms + 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ms
                )
                arrays = frame_arrays(result)
                h, w = bgr.shape[:2]
                data = {k: v[None] for k, v in arrays.items()}
                row = features({**data, "t": np.zeros(1), "size": np.array([w, h])})[0]
                seen = any(not np.isnan(arrays[k][0, 0]) for k in HANDS)
                for g in live.push(t, row, seen):
                    caption.append(g)
                    if g == vocab[target] and not advance_at:
                        done.add(target)
                        last_hit, advance_at = g, now + 1.5
                if advance_at and now >= advance_at:
                    # Proximo sinal que falta; se ja acertou todos, so o seguinte.
                    order = [(target + i - 1) % (len(vocab) - 1) + 1 for i in range(1, len(vocab))]
                    go(next((g for g in order if g not in done), order[0]) - target)
                    advance_at = 0.0

                # O modelo recebe a imagem como a camera da; so a tela e espelhada.
                _draw(bgr, arrays["face"], face, GRAY)
                _draw(bgr, arrays["pose"][:BODY], pose, GREEN, dot=3)
                for k, color in HANDS.items():
                    _draw(bgr, arrays[k], hand, color, dot=4)
                view = cv2.flip(bgr, 1)
                s = h / 720
                hint = framing(arrays["pose"], w / h)
                if hint:
                    # Onde os ombros ficam no corpus de treino (mediana).
                    dx = TARGET[1] * h / 2
                    for x in (w / 2 - dx, w / 2 + dx):
                        cv2.circle(view, (int(x), int(TARGET[0] * h)), int(14 * s), YELLOW, 2)

                # Coluna da esquerda: estado e lista de sinais.
                col = int(250 * s)
                _shade(view, 0, 0, col, h - int(60 * s))
                _text(view, f"{rate:4.1f} quadros/s", (int(12 * s), int(28 * s)), 0.6 * s)
                _text(view, "mao esq.", (int(12 * s), int(54 * s)), 0.55 * s, HANDS["left_hand"])
                _text(view, "mao dir.", (int(120 * s), int(54 * s)), 0.55 * s, HANDS["right_hand"])
                _text(
                    view,
                    f"acertos: {len(done)} de {len(vocab) - 1}",
                    (int(12 * s), int(86 * s)),
                    0.6 * s,
                    GREEN,
                )
                for i, g in enumerate(vocab[1:], 1):
                    y = int((86 + 24 * i) * s)
                    mark = "[x]" if i in done else "[ ]"
                    color = YELLOW if i == target else GREEN if i in done else WHITE
                    _text(
                        view,
                        f"{'>' if i == target else ' '} {mark} {g}",
                        (int(12 * s), y),
                        0.5 * s,
                        color,
                    )
                _text(
                    view,
                    "n/p: troca o sinal  q: sai",
                    (int(12 * s), h - int(104 * s)),
                    0.45 * s,
                    GRAY,
                )

                # Aula: clipe de referencia em laco e a chance do sinal alvo agora.
                side = int(380 * s)
                x0, y0 = w - side - int(16 * s), int(52 * s)
                _shade(view, x0, y0 - int(40 * s), x0 + side, y0 + side + int(64 * s))
                _text(
                    view,
                    f"faca: {vocab[target]}",
                    (x0 + int(10 * s), y0 - int(12 * s)),
                    0.8 * s,
                    YELLOW,
                    2,
                )
                if target in refs:
                    pts, signer = refs[target]
                    i = int((now - lesson) * FPS) % len(pts["pose"])
                    box = view[y0 : y0 + side, x0 : x0 + side]
                    _draw(box, pts["face"][i], face, GRAY)
                    _draw(box, pts["pose"][i], pose, GREEN, dot=2)
                    for k, color in HANDS.items():
                        _draw(box, pts[k][i], hand, color, dot=2)
                    credit = f"referencia: {signer}, corpus de treino"
                else:
                    credit = "sem clipe de referencia"
                _text(view, credit, (x0 + int(10 * s), y0 + side + int(20 * s)), 0.45 * s, GRAY)
                p = float(live.probs[target]) if live.probs is not None else 0.0
                bar = (x0 + int(10 * s), y0 + side + int(32 * s))
                cv2.rectangle(view, bar, (x0 + side - int(10 * s), bar[1] + int(20 * s)), GRAY, 1)
                fill = bar[0] + int((side - int(20 * s)) * p)
                cv2.rectangle(view, bar, (fill, bar[1] + int(20 * s)), YELLOW, -1)
                label = f"chance agora: {p:.0%}"
                _text(view, label, (bar[0] + int(6 * s), bar[1] + int(15 * s)), 0.45 * s, GRAY)

                # Mensagens no centro.
                mid = col + int(16 * s)
                if hint:
                    _shade(view, mid, int(8 * s), x0 - int(16 * s), int(48 * s))
                    _text(view, hint, (mid + int(10 * s), int(36 * s)), 0.7 * s, YELLOW, 2)
                if advance_at:
                    _text(
                        view, f"ACERTOU {last_hit}!", (mid + int(10 * s), h // 2), 1.4 * s, GREEN, 3
                    )

                # Legenda: glosas reconhecidas e as 3 classes mais provaveis agora.
                _shade(view, 0, h - int(60 * s), w, h, 0.8)
                _shade(view, 0, h - int(90 * s), w, h - int(60 * s), 0.5)
                if live.probs is not None:
                    top = np.argsort(live.probs)[::-1][:3]
                    agora = "   ".join(f"{shown[i]} {live.probs[i]:.2f}" for i in top)
                    _text(view, f"agora: {agora}", (int(12 * s), h - int(68 * s)), 0.55 * s, GRAY)
                _text(view, " ".join(caption), (int(12 * s), h - int(20 * s)), 1.0 * s, WHITE, 2)

                cv2.imshow(WINDOW, view)
                key = cv2.waitKeyEx(1)
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                # Setas no Windows (waitKeyEx): 2555904 direita, 2424832 esquerda.
                if key in (ord("n"), 2555904):
                    go(1)
                elif key in (ord("p"), 2424832):
                    go(-1)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
