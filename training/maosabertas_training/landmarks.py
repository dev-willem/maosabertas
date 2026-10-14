"""Extrai landmarks de maos, rosto (com blendshapes) e pose, quadro a quadro.

Uso:
    python -m maosabertas_training.landmarks ENTRADA SAIDA.npz

ENTRADA e um arquivo de video ou o indice de uma camera (por exemplo, 0).
Ctrl+C encerra a captura e grava o que ja foi extraido.

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
    size              (2,)         largura e altura da imagem, em pixels

Esquerda e direita sao as do HolisticLandmarker; ainda nao foi conferido com
gravacao real se correspondem as maos de quem sinaliza.
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


def extract(frames: Iterable[tuple[float, np.ndarray]], model: Path) -> dict[str, np.ndarray]:
    """`frames`: pares (segundos, imagem RGB uint8). Ctrl+C encerra e devolve o parcial."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
    )
    times: list[float] = []
    rows: list[dict[str, np.ndarray]] = []
    names: list[str] = []
    size = (0, 0)
    last_ms = -1
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entrada", help="arquivo de video ou indice da camera")
    parser.add_argument("saida", type=Path, help="arquivo .npz fora do repositorio")
    args = parser.parse_args(argv)

    try:
        out = check_output(args.saida)
    except ValueError as e:
        parser.error(str(e))
    data = extract(read_frames(args.entrada), model_path())
    np.savez_compressed(out, **data)

    n = len(data["t"])
    print(f"{n} quadros -> {out}")
    for k in ("left_hand", "right_hand", "face", "pose"):
        found = int((~np.isnan(data[k]).all(axis=tuple(range(1, data[k].ndim)))).sum())
        print(f"  {k:<10} detectado em {found}/{n} quadros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
