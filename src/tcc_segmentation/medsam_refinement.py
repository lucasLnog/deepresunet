"""Inferência MedSAM oficial com caixas geradas exclusivamente pela DR-IFT."""

from __future__ import annotations

from typing import Any

import numpy as np

from tcc_segmentation.sam_refinement import component_box, connected_components_8


def preprocess_medsam_image(rgb_image: np.ndarray, *, device: Any):
    """Reproduz o pré-processamento do repositório oficial do MedSAM."""
    import torch
    from skimage import transform

    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("imagem RGB HWC esperada")
    resized = transform.resize(
        image,
        (1024, 1024),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    minimum, maximum = float(resized.min()), float(resized.max())
    normalized = (resized - minimum) / max(maximum - minimum, 1e-8)
    return torch.as_tensor(normalized, dtype=torch.float32, device=device).permute(2, 0, 1)[None]


def refine_prediction_with_medsam_box(
    *,
    model: Any,
    rgb_image: np.ndarray,
    network_probabilities: np.ndarray,
    network_threshold: float,
    max_components_safety: int,
    output_threshold: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Executa um prompt de caixa por componente e une as saídas.

    A assinatura não aceita a máscara de referência, impedindo seu uso no prompt.
    """
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    probabilities = np.asarray(network_probabilities, dtype=np.float32)
    if probabilities.shape != rgb.shape[:2]:
        raise ValueError("probabilidade e RGB devem estar no mesmo espaço")
    binary = probabilities >= float(network_threshold)
    components = connected_components_8(binary)
    audit: dict[str, Any] = {
        "prompt_source": "DR-IFT prediction only",
        "ground_truth_used_for_prompt": False,
        "mode": "medsam_box",
        "network_threshold": float(network_threshold),
        "component_count": len(components),
        "empty_prediction_policy_applied": not components,
        "preprocessing": "official_square_1024_minmax",
        "components": [],
    }
    if not components:
        return np.zeros(binary.shape, dtype=bool), audit
    if len(components) > int(max_components_safety):
        raise RuntimeError(
            f"predição gerou {len(components)} componentes; limite de segurança="
            f"{max_components_safety}"
        )

    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    with torch.inference_mode():
        embedding = model.image_encoder(preprocess_medsam_image(rgb, device=device))
        union = np.zeros(binary.shape, dtype=bool)
        height, width = binary.shape
        scale = np.asarray([1024 / width, 1024 / height, 1024 / width, 1024 / height])
        for index, component in enumerate(components, start=1):
            box_original = component_box(component)
            box_1024 = torch.as_tensor(box_original * scale, dtype=torch.float32, device=device)[
                None, None, :
            ]
            sparse, dense = model.prompt_encoder(points=None, boxes=box_1024, masks=None)
            low_res_logits, predicted_iou = model.mask_decoder(
                image_embeddings=embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            probability = functional.interpolate(
                torch.sigmoid(low_res_logits),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            prediction = probability.cpu().numpy() > float(output_threshold)
            union |= prediction
            audit["components"].append(
                {
                    "component_index": index,
                    "source_area_pixels": int(component.sum()),
                    "box_xyxy_original": [float(value) for value in box_original],
                    "box_xyxy_1024": [float(value) for value in box_1024[0, 0].cpu()],
                    "medsam_predicted_iou": float(predicted_iou.reshape(-1)[0].cpu()),
                }
            )
    return union, audit
