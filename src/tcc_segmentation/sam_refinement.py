"""Integração auditável do SAM 1 como refinador da predição da DR-IFT.

As funções que constroem prompts não recebem a máscara de referência. Essa separação
é deliberada: a referência só pode entrar depois, no cálculo das métricas.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


PROMPT_MODES = ("box", "mask_input", "box_plus_mask")
SAM_REPOSITORY_COMMIT = "6fdee8f2727f4506cfbbe553e23b895e27956588"
SAM_VIT_B_FILENAME = "sam_vit_b_01ec64.pth"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_day12_config(config: Mapping[str, Any]) -> None:
    if config.get("stage") != "day12_sam_technical_integration":
        raise ValueError("Etapa inválida para a integração técnica do SAM")
    network = config.get("network", {})
    if network.get("experiment_id") != "DR-IFT" or int(network.get("seed", -1)) != 42:
        raise ValueError("o refinador exige o DR-IFT representativo da seed 42")
    if int(network.get("checkpoint_epoch", -1)) != 56:
        raise ValueError("checkpoint DR-IFT diferente da época 56 congelada")
    if not math.isclose(float(network.get("threshold", -1)), 0.18, abs_tol=1e-12):
        raise ValueError("threshold DR-IFT deve permanecer congelado em 0,18")
    sam = config.get("sam", {})
    if sam.get("family") != "SAM 1" or sam.get("model_type") != "vit_b":
        raise ValueError("A integração técnica do SAM exige SAM 1 ViT-B")
    if sam.get("checkpoint_filename") != SAM_VIT_B_FILENAME:
        raise ValueError("checkpoint SAM ViT-B não reconhecido")
    if sam.get("repository_commit") != SAM_REPOSITORY_COMMIT:
        raise ValueError("commit do SAM difere da versão congelada")
    if int(sam.get("image_encoder_size", -1)) != 1024:
        raise ValueError("SAM 1 deve usar lado longo de entrada igual a 1024")
    prompts = config.get("prompts", {})
    if tuple(prompts.get("modes", ())) != PROMPT_MODES:
        raise ValueError("os três modos de prompt devem ser preservados e ordenados")
    if prompts.get("source") != "DR-IFT prediction only":
        raise ValueError("prompts devem vir exclusivamente da DR-IFT")
    if prompts.get("multiple_components") != "one_prompt_per_component_union_outputs":
        raise ValueError("política de múltiplos componentes inválida")
    if prompts.get("empty_prediction") != "empty_output_without_sam_call":
        raise ValueError("política de predição vazia inválida")
    if prompts.get("mask_input_alignment") != "sam_resize_longest_side_then_bottom_right_padding":
        raise ValueError("alinhamento espacial do mask_input inválido")
    if bool(prompts.get("ground_truth_allowed")):
        raise ValueError("ground truth não pode gerar prompts")
    if int(prompts.get("max_components_safety", 0)) < 1:
        raise ValueError("limite de segurança de componentes inválido")
    data = config.get("data", {})
    if data.get("split") != "validation":
        raise ValueError("integração da integração técnica do SAM é restrita à validação")
    lowered = str(data.get("manifest", "")).casefold()
    if any(marker in lowered for marker in ("test", "external", "fuseg")):
        raise ValueError("partição protegida bloqueada")


def resize_pad_geometry(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> dict[str, int | float]:
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, round(source_width * scale)))
    resized_height = max(1, min(target_height, round(source_height * scale)))
    return {
        "scale": float(scale),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "left": int((target_width - resized_width) // 2),
        "top": int((target_height - resized_height) // 2),
    }


def canvas_to_original(
    array: np.ndarray,
    *,
    original_height: int,
    original_width: int,
    interpolation: str,
) -> np.ndarray:
    """Remove o padding 384x512 e retorna o mapa ao espaço RGB original."""
    if array.ndim != 2:
        raise ValueError("mapa 2D esperado")
    geometry = resize_pad_geometry(
        original_height, original_width, int(array.shape[0]), int(array.shape[1])
    )
    top, left = int(geometry["top"]), int(geometry["left"])
    height, width = int(geometry["resized_height"]), int(geometry["resized_width"])
    cropped = array[top : top + height, left : left + width]
    resample = (
        Image.Resampling.BILINEAR if interpolation == "bilinear" else Image.Resampling.NEAREST
    )
    mode = "F" if np.issubdtype(cropped.dtype, np.floating) else "L"
    resized = Image.fromarray(cropped, mode=mode).resize(
        (original_width, original_height), resample=resample
    )
    result = np.asarray(resized)
    return result.astype(np.float32) if interpolation == "bilinear" else result


def original_mask_to_canvas(
    mask: np.ndarray, *, target_height: int, target_width: int
) -> np.ndarray:
    """Leva uma máscara retornada pelo SAM ao espaço usado nas métricas da rede."""
    if mask.ndim != 2:
        raise ValueError("máscara 2D esperada")
    source_height, source_width = mask.shape
    geometry = resize_pad_geometry(source_height, source_width, target_height, target_width)
    width, height = int(geometry["resized_width"]), int(geometry["resized_height"])
    resized = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    top, left = int(geometry["top"]), int(geometry["left"])
    canvas[top : top + height, left : left + width] = np.asarray(resized, dtype=np.uint8) > 0
    return canvas.astype(bool)


def connected_components_8(mask: np.ndarray) -> list[np.ndarray]:
    """Retorna componentes 8-conectados sem depender de bibliotecas externas."""
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError("máscara 2D esperada")
    visited = np.zeros_like(foreground, dtype=bool)
    components: list[np.ndarray] = []
    height, width = foreground.shape
    for start_y, start_x in np.argwhere(foreground):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and foreground[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        component = np.zeros_like(foreground, dtype=bool)
        ys, xs = zip(*points)
        component[np.asarray(ys), np.asarray(xs)] = True
        components.append(component)
    return components


def component_box(component: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(component)
    if not len(xs):
        raise ValueError("componente vazio")
    # XYXY; o máximo recebe +1 para representar toda a extensão do último pixel.
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def shifted_network_logits(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Converte p em logits e faz o threshold congelado corresponder ao logit zero."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold deve estar entre zero e um")
    clipped = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    threshold_logit = math.log(threshold / (1.0 - threshold))
    return logits - threshold_logit


def component_mask_input(
    shifted_logits: np.ndarray,
    component: np.ndarray,
    *,
    outside_logit: float = -10.0,
    sam_image_encoder_size: int = 1024,
    mask_input_size: int = 256,
) -> np.ndarray:
    """Isola um componente sem usar informação da referência.

    Mantemos os logits na caixa do componente, inclusive a incerteza de borda, e
    usamos evidência negativa fora dela. O redimensionamento para 256x256 segue a
    interface oficial do SAM 1. O SAM redimensiona a imagem pelo maior lado para
    1024 e adiciona padding apenas embaixo e à direita. O prompt 256x256 precisa
    repetir exatamente essa geometria; redimensionar a fotografia diretamente para
    um quadrado deslocaria a máscara em imagens não quadradas.
    """
    box = component_box(component).astype(int)
    isolated = np.full_like(shifted_logits, float(outside_logit), dtype=np.float32)
    x0, y0, x1, y1 = box
    isolated[y0:y1, x0:x1] = shifted_logits[y0:y1, x0:x1]
    height, width = isolated.shape
    scale = sam_image_encoder_size / max(height, width)
    sam_height = int(height * scale + 0.5)
    sam_width = int(width * scale + 0.5)
    low_height = max(
        1,
        min(
            mask_input_size,
            int(sam_height * mask_input_size / sam_image_encoder_size + 0.5),
        ),
    )
    low_width = max(
        1,
        min(
            mask_input_size,
            int(sam_width * mask_input_size / sam_image_encoder_size + 0.5),
        ),
    )
    resized = np.asarray(
        Image.fromarray(isolated, mode="F").resize(
            (low_width, low_height), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    canvas = np.full((mask_input_size, mask_input_size), float(outside_logit), dtype=np.float32)
    canvas[:low_height, :low_width] = resized
    return canvas[None, ...]


def refine_prediction_with_sam(
    *,
    predictor: Any,
    rgb_image: np.ndarray,
    network_probabilities: np.ndarray,
    network_threshold: float,
    mode: str,
    max_components_safety: int,
    sam_image_encoder_size: int = 1024,
    mask_input_size: int = 256,
    outside_logit: float = -10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refina uma predição; propositalmente não aceita ground truth."""
    if mode not in PROMPT_MODES:
        raise ValueError(f"modo inválido: {mode}")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("imagem RGB HWC esperada")
    if network_probabilities.shape != rgb_image.shape[:2]:
        raise ValueError("probabilidade e RGB devem estar no mesmo espaço")
    binary = network_probabilities >= network_threshold
    components = connected_components_8(binary)
    audit: dict[str, Any] = {
        "prompt_source": "DR-IFT prediction only",
        "ground_truth_used_for_prompt": False,
        "mode": mode,
        "network_threshold": float(network_threshold),
        "component_count": len(components),
        "empty_prediction_policy_applied": not components,
        "mask_input_alignment": "sam_resize_longest_side_then_bottom_right_padding",
        "components": [],
    }
    if not components:
        return np.zeros(binary.shape, dtype=bool), audit
    if len(components) > max_components_safety:
        raise RuntimeError(
            f"predição gerou {len(components)} componentes; limite de segurança="
            f"{max_components_safety}. Nenhum componente foi descartado silenciosamente."
        )
    predictor.set_image(np.ascontiguousarray(rgb_image, dtype=np.uint8))
    shifted = shifted_network_logits(network_probabilities, network_threshold)
    union = np.zeros(binary.shape, dtype=bool)
    for index, component in enumerate(components, start=1):
        box = component_box(component)
        mask_input = component_mask_input(
            shifted,
            component,
            outside_logit=outside_logit,
            sam_image_encoder_size=sam_image_encoder_size,
            mask_input_size=mask_input_size,
        )
        kwargs: dict[str, Any] = {"multimask_output": False}
        if mode in {"box", "box_plus_mask"}:
            kwargs["box"] = box
        if mode in {"mask_input", "box_plus_mask"}:
            kwargs["mask_input"] = mask_input
        masks, scores, _ = predictor.predict(**kwargs)
        if len(masks) != 1:
            raise RuntimeError("multimask_output=False deveria retornar uma máscara")
        union |= np.asarray(masks[0], dtype=bool)
        audit["components"].append(
            {
                "component_index": index,
                "source_area_pixels": int(component.sum()),
                "box_xyxy": [float(value) for value in box],
                "sam_predicted_iou": float(np.asarray(scores).reshape(-1)[0]),
            }
        )
    return union, audit


def binary_metric_row(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    pred, ref = np.asarray(prediction, dtype=bool), np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError("predição e referência com shapes diferentes")
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    tn = int((~pred & ~ref).sum())

    def divide(num: float, den: float, fallback: float = 1.0) -> float:
        return float(num / den) if den else float(fallback)

    precision = divide(tp, tp + fp, 0.0 if tp + fn else 1.0)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": divide(2 * tp, 2 * tp + fp + fn),
        "iou": divide(tp, tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }
