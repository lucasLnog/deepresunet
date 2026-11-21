"""DeepResUNet histórica: U-Net residual com attention gates e saída em logits."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Duas convoluções 3x3 com atalho residual, preservando o desenho original."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.skip(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + identity)


class AttentionGate(nn.Module):
    """Filtra uma skip connection usando o sinal do decoder como gate."""

    def __init__(self, gate_channels: int, skip_channels: int, intermediate_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Gate e skip connection precisam ter as mesmas dimensoes espaciais")
        coefficients = self.attention(
            self.relu(self.gate_projection(gate) + self.skip_projection(skip))
        )
        return skip * coefficients


class DeepResUNet(nn.Module):
    """Arquitetura histórica modularizada, sem ativação após a projeção final."""

    downsampling_factor = 16

    def __init__(self, in_channels: int = 3, classes: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels < 1 or classes < 1 or base_channels < 1:
            raise ValueError("in_channels, classes e base_channels devem ser positivos")

        c1, c2, c3, c4, c5 = (base_channels * 2**index for index in range(5))
        self.enc1 = ResidualBlock(in_channels, c1)
        self.enc2 = ResidualBlock(c1, c2)
        self.enc3 = ResidualBlock(c2, c3)
        self.enc4 = ResidualBlock(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ResidualBlock(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(c4, c4, c3)
        self.dec4 = ResidualBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(c3, c3, c2)
        self.dec3 = ResidualBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(c2, c2, c1)
        self.dec2 = ResidualBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(c1, c1, max(c1 // 2, 1))
        self.dec1 = ResidualBlock(c1 * 2, c1)

        self.final_features = nn.Sequential(
            nn.Conv2d(c1, max(c1 // 2, 1), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(c1 // 2, 1)),
            nn.ReLU(inplace=True),
        )
        self.final_projection = nn.Conv2d(max(c1 // 2, 1), classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        if height % self.downsampling_factor or width % self.downsampling_factor:
            raise ValueError(
                "DeepResUNet requer altura e largura divisiveis por 16; "
                f"recebido {height}x{width}"
            )

        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat((self.attn4(d4, e4), d4), dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((self.attn3(d3, e3), d3), dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((self.attn2(d2, e2), d2), dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((self.attn1(d1, e1), d1), dim=1))
        return self.final_projection(self.final_features(d1))


class DeepResUNetResNet34(nn.Module):
    """DeepResUNet com encoder ResNet-34 e decoder historico preservado.

    O stride da primeira convolucao da ResNet e alterado de 2 para 1. Com o
    max-pooling original, os cinco niveis espaciais passam a ser H, H/2,
    H/4, H/8 e H/16, exatamente os niveis esperados pelo decoder historico.
    """

    downsampling_factor = 16
    encoder_stage_names = ("stem", "layer1", "layer2", "layer3", "layer4")

    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        encoder_weights: str | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("Pesos ResNet-34 exigem entrada RGB com 3 canais")
        if classes < 1:
            raise ValueError("classes deve ser positivo")
        normalized_weights = (
            None
            if encoder_weights is None or str(encoder_weights).casefold() == "none"
            else str(encoder_weights).casefold()
        )
        if normalized_weights not in {None, "imagenet"}:
            raise ValueError("encoder_weights deve ser null ou 'imagenet'")

        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as error:
            raise RuntimeError("torchvision nao instalado; execute no ambiente Colab.") from error

        weights = ResNet34_Weights.IMAGENET1K_V1 if normalized_weights == "imagenet" else None
        backbone = resnet34(weights=weights)
        backbone.conv1.stride = (1, 1)
        self.encoder_weights = normalized_weights
        self.stem_conv = backbone.conv1
        self.stem_bn = backbone.bn1
        self.stem_relu = backbone.relu
        self.stem_pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        c1, c2, c3, c4, c5 = 32, 64, 128, 256, 512
        self.skip1_adapter = ResidualBlock(64, c1)

        # Decoder e attention gates preservam o desenho e os canais historicos.
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(c4, c4, c3)
        self.dec4 = ResidualBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(c3, c3, c2)
        self.dec3 = ResidualBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(c2, c2, c1)
        self.dec2 = ResidualBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(c1, c1, 16)
        self.dec1 = ResidualBlock(c1 * 2, c1)
        self.final_features = nn.Sequential(
            nn.Conv2d(c1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.final_projection = nn.Conv2d(16, classes, kernel_size=1)

        self.set_encoder_trainable(not freeze_encoder)

    def _encoder_stages(self) -> dict[str, tuple[nn.Module, ...]]:
        return {
            "stem": (self.stem_conv, self.stem_bn),
            "layer1": (self.layer1,),
            "layer2": (self.layer2,),
            "layer3": (self.layer3,),
            "layer4": (self.layer4,),
        }

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Congela ou libera todo o backbone ResNet-34."""
        for modules in self._encoder_stages().values():
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = bool(trainable)
        self._synchronize_encoder_modes(self.training)

    def set_trainable_encoder_stages(self, stage_names: tuple[str, ...]) -> None:
        """Libera somente estagios predefinidos para fine-tuning progressivo."""
        requested = set(stage_names)
        invalid = requested.difference(self.encoder_stage_names)
        if invalid:
            raise ValueError(f"Estagios ResNet-34 invalidos: {sorted(invalid)}")
        for name, modules in self._encoder_stages().items():
            stage_trainable = name in requested
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = stage_trainable
        self._synchronize_encoder_modes(self.training)

    def encoder_is_frozen(self) -> bool:
        return not any(
            parameter.requires_grad
            for modules in self._encoder_stages().values()
            for module in modules
            for parameter in module.parameters()
        )

    def encoder_parameters(self):
        for modules in self._encoder_stages().values():
            for module in modules:
                yield from module.parameters()

    def decoder_parameters(self):
        encoder_ids = {id(parameter) for parameter in self.encoder_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in encoder_ids:
                yield parameter

    def decoder_state_dict(self) -> dict[str, torch.Tensor]:
        encoder_prefixes = (
            "stem_conv.",
            "stem_bn.",
            "layer1.",
            "layer2.",
            "layer3.",
            "layer4.",
        )
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not name.startswith(encoder_prefixes)
        }

    def _synchronize_encoder_modes(self, parent_training: bool) -> None:
        for modules in self._encoder_stages().values():
            for module in modules:
                trainable = any(parameter.requires_grad for parameter in module.parameters())
                module.train(parent_training and trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        self._synchronize_encoder_modes(mode)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        if height % self.downsampling_factor or width % self.downsampling_factor:
            raise ValueError(
                "DeepResUNetResNet34 requer altura e largura divisiveis por 16; "
                f"recebido {height}x{width}"
            )

        stem = self.stem_relu(self.stem_bn(self.stem_conv(inputs)))
        e1 = self.skip1_adapter(stem)
        e2 = self.layer1(self.stem_pool(stem))
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        bottleneck = self.layer4(e4)

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat((self.attn4(d4, e4), d4), dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat((self.attn3(d3, e3), d3), dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat((self.attn2(d2, e2), d2), dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((self.attn1(d1, e1), d1), dim=1))
        return self.final_projection(self.final_features(d1))
