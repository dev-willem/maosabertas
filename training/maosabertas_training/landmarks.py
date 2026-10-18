"""Extrai landmarks de maos, rosto (com blendshapes) e pose, quadro a quadro.

Uso:
    python -m maosabertas_training.landmarks ENTRADA SAIDA.npz [--recorte X0,Y0,X1,Y1]

ENTRADA e um arquivo de video ou o indice de uma camera (por exemplo, 0).
Ctrl+C encerra a captura e grava o que ja foi extraido.

--recorte: so a regiao do sinalizante passa pelo MediaPipe (video com interprete
no canto). Cantos superior esquerdo e inferior direito, em fracoes da largura e da
altura do quadro (0 a 1): 0.5,0.5,1,1 e o quarto inferior direito. Com recorte, os
landmarks saem normalizados no recorte, nao no quadro inteiro, e `size` e o tamanho
do recorte (as features usam a razao largura/altura dele).

Landmarks de rosto e maos sao dado biometrico (LGPD): a saida precisa ficar
fora do repositorio, e o script recusa caminho dentro dele. Nenhum video e
gravado.

Arrays do .npz (T quadros; NaN onde nada foi detectado):
    t                 (T,)         segundos desde o primeiro quadro
    left_hand         (T, 21, 3)   x, y normalizados na imagem; z relativo
    right_hand        (T, 21, 3)
    face              (T, 478, 3)
    blendshapes       (T, 52)
    pose              (T, 33, 4)   x, y, z, visibilidade
    blendshape_names  (52,)        vazio se nenhum rosto foi detectado
    size              (2,)         largura e altura da imagem (ou do recorte), em pixels

Esquerda e direita sao as do HolisticLandmarker e correspondem as maos de quem
sinaliza (conferido na camera em 2026-10-18).
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
REPO = Path(__file__).resolve().parents[2]

SHAPES = {
    "left_hand": (21, 3),
    "right_hand": (21, 3),
    "face": (478, 3),
    "blendshapes": (52,),
    "pose": (33, 4),
}


def model_path() -> Path:
    """Baixa o modelo uma vez para um cache fora do repositorio."""
    default = Path.home() / ".cache" / "maosabertas"
    cache = Path(os.environ.get("MAOSABERTAS_MODELS", default))
    path = cache / "holistic_landmarker.task"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, part)
        part.replace(path)
    return path


def check_output(path: Path) -> Path:
    """Recusa gravar landmarks dentro do repositorio."""
    path = path.resolve()
    if path.is_relative_to(REPO):
        raise ValueError(
            f"saida dentro do repositorio: {path}. Landmarks sao dado pessoal; grave fora dele."
        )
    return path


def _points(landmarks, shape: tuple[int, int]) -> np.ndarray:
    fields = ("x", "y", "z", "visibility")[: shape[1]]
    out = np.full(shape, np.nan, np.float32)
    if landmarks:
        # Atribuicao falha se o modelo devolver outro numero de pontos: melhor que dado torto.
        out[:] = [[getattr(p, f) for f in fields] for p in landmarks]
    return out


def frame_arrays(result) -> dict[str, np.ndarray]:
    """Converte um HolisticLandmarkerResult em arrays de tamanho fixo."""
    blend = np.full(SHAPES["blendshapes"], np.nan, np.float32)
    for c in result.face_blendshapes or []:
        blend[c.index] = c.score
    return {
        "left_hand": _points(result.left_hand_landmarks, SHAPES["left_hand"]),
        "right_hand": _points(result.right_hand_landmarks, SHAPES["right_hand"]),
        "face": _points(result.face_landmarks, SHAPES["face"]),
        "blendshapes": blend,
        "pose": _points(result.pose_landmarks, SHAPES["pose"]),
    }


def holistic(model: Path):
    """HolisticLandmarker em modo VIDEO, com blendshapes (as features usam)."""
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def extract(frames: Iterable[tuple[float, np.ndarray]], model: Path) -> dict[str, np.ndarray]:
    """`frames`: pares (segundos, imagem RGB uint8). Ctrl+C encerra e devolve o parcial."""
    import mediapipe as mp

    times: list[float] = []
    rows: list[dict[str, np.ndarray]] = []
    names: list[str] = []
    size = (0, 0)
    last_ms = -1
    with holistic(model) as landmarker:
        try:
            for t, rgb in frames:
                # O modo VIDEO exige instantes estritamente crescentes, em ms inteiros.
                last_ms = max(int(t * 1000), last_ms + 1)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                result = landmarker.detect_for_video(image, last_ms)
                if not names and result.face_blendshapes:
                    ordered = sorted(result.face_blendshapes, key=lambda c: c.index)
                    names = [c.category_name for c in ordered]
                times.append(t)
                rows.append(frame_arrays(result))
                size = (rgb.shape[1], rgb.shape[0])
        except KeyboardInterrupt:
            pass

    data = {
        k: np.asarray([r[k] for r in rows], np.float32).reshape(-1, *shape)
        for k, shape in SHAPES.items()
    }
    data["t"] = np.asarray(times, np.float64)
    data["blendshape_names"] = np.asarray(names, dtype=str)
    data["size"] = np.asarray(size, np.int32)
    return data


def read_frames(source: str) -> Iterator[tuple[float, np.ndarray]]:
    """Le quadros RGB de um arquivo de video ou de uma camera (indice numerico)."""
    import cv2

    camera = source.isdigit()
    cap = cv2.VideoCapture(int(source) if camera else source)
    if not cap.isOpened():
        raise SystemExit(f"nao foi possivel abrir: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not camera and not fps > 0:
        cap.release()
        raise SystemExit(f"video sem fps: {source}")
    start = time.monotonic()
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            t = time.monotonic() - start if camera else i / fps
            yield t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


Crop = tuple[float, float, float, float]  # x0, y0, x1, y1 em fracoes do quadro


def parse_crop(text: str) -> Crop:
    """Le "x0,y0,x1,y1" em fracoes da largura e da altura, com 0 <= x0 < x1 <= 1 e idem em y."""
    try:
        box = tuple(float(v) for v in text.split(","))
    except ValueError:
        box = ()
    if len(box) != 4:
        raise argparse.ArgumentTypeError(f"esperado x0,y0,x1,y1, recebido {text!r}")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):  # NaN tambem cai aqui
        raise argparse.ArgumentTypeError(
            f"recorte {text!r}: precisa de 0 <= x0 < x1 <= 1 e 0 <= y0 < y1 <= 1"
        )
    return box


def crop(
    frames: Iterable[tuple[float, np.ndarray]], box: Crop
) -> Iterator[tuple[float, np.ndarray]]:
    """Recorta cada quadro em `box` (fracoes, ver `parse_crop`) antes do MediaPipe."""
    x0, y0, x1, y1 = box
    for t, rgb in frames:
        h, w = rgb.shape[:2]
        out = rgb[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)]
        if not out.size:
            raise ValueError(f"recorte {box} vazio num quadro de {w}x{h} pixels")
        yield t, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entrada", help="arquivo de video ou indice da camera")
    parser.add_argument("saida", type=Path, help="arquivo .npz fora do repositorio")
    parser.add_argument(
        "--recorte",
        type=parse_crop,
        metavar="X0,Y0,X1,Y1",
        help="regiao do sinalizante, em fracoes da largura e da altura (0 a 1)",
    )
    args = parser.parse_args(argv)

    try:
        out = check_output(args.saida)
    except ValueError as e:
        parser.error(str(e))
    frames = read_frames(args.entrada)
    if args.recorte:
        frames = crop(frames, args.recorte)
    data = extract(frames, model_path())
    np.savez_compressed(out, **data)

    n = len(data["t"])
    print(f"{n} quadros -> {out}")
    for k in ("left_hand", "right_hand", "face", "pose"):
        found = int((~np.isnan(data[k]).all(axis=tuple(range(1, data[k].ndim)))).sum())
        print(f"  {k:<10} detectado em {found}/{n} quadros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
