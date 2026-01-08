"""Export SAM2 model to ONNX for standalone prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from training.utils.train_utils import register_omegaconf_resolvers


def _ensure_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("times"):
        register_omegaconf_resolvers()


def _load_config(config_path: Path) -> Dict[str, Any]:
    _ensure_omegaconf_resolvers()
    config = OmegaConf.create(yaml.safe_load(config_path.read_text()))
    OmegaConf.resolve(config)
    return config


def _get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "trainer" in config and "model" in config["trainer"]:
        return config["trainer"]["model"]
    if "model" in config:
        return config["model"]
    raise KeyError("Config must contain trainer.model or model section.")


def _build_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    return instantiate(model_cfg, _recursive_=True)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=True)


class SAM2EncoderOnnxWrapper(torch.nn.Module):
    def __init__(self, model: SAM2Base):
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(
            backbone_out
        )
        if not self.model.use_high_res_features_in_sam:
            raise ValueError("Encoder export expects high-res features to be enabled.")

        top_feat = vision_feats[-1]
        pix_feat = top_feat.permute(1, 2, 0).view(
            top_feat.size(1), top_feat.size(2), *feat_sizes[-1]
        )

        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
        if len(high_res_features) != 2:
            raise ValueError(
                "Expected two high-res feature maps for export, got "
                f"{len(high_res_features)}."
            )

        return pix_feat, high_res_features[0], high_res_features[1]


class SAM2DecoderOnnxWrapper(torch.nn.Module):
    def __init__(self, model: SAM2Base):
        super().__init__()
        self.model = model

    def forward(
        self,
        pix_feat: torch.Tensor,
        high_res_0: torch.Tensor,
        high_res_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_inputs: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_encoder = self.model.sam_prompt_encoder
        point_labels = point_labels.to(torch.int64)
        num_points = point_labels.size(1)
        point_embeddings = prompt_encoder._embed_points(
            point_coords, point_labels, pad=False
        )
        sparse_embeddings = point_embeddings

        mask_embeddings = prompt_encoder._embed_masks(mask_inputs)
        no_mask_embed = prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        no_mask_embed = no_mask_embed.expand(
            mask_embeddings.size(0),
            -1,
            mask_embeddings.size(2),
            mask_embeddings.size(3),
        )
        mask_flag = has_mask.to(mask_embeddings.dtype)
        dense_embeddings = mask_embeddings * mask_flag + no_mask_embed * (1 - mask_flag)

        multimask_output = (
            self.model.multimask_output_in_sam
            and self.model.multimask_min_pt_num <= num_points
            and num_points <= self.model.multimask_max_pt_num
        )
        high_res_features = None
        if self.model.use_high_res_features_in_sam:
            high_res_features = [high_res_0, high_res_1]
        (
            low_res_multimasks,
            ious,
            _,
            object_score_logits,
        ) = self.model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        if self.model.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            no_obj_score = low_res_multimasks.new_tensor(NO_OBJ_SCORE)
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None], low_res_multimasks, no_obj_score
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = torch.nn.functional.interpolate(
            low_res_multimasks,
            size=(self.model.image_size, self.model.image_size),
            mode="bilinear",
            align_corners=False,
        )

        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(
                low_res_multimasks.size(0), device=low_res_multimasks.device
            )
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        else:
            low_res_masks = low_res_multimasks
            high_res_masks = high_res_multimasks

        return low_res_masks, high_res_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM2 ONNX model")
    parser.add_argument("--checkpoint", default="/home/wensheng/gjq_workspace/eyesam/exp_log_260107_test1/checkpoints/checkpoint_180.pt", help="Path to model weights")
    parser.add_argument(
        "--config",
        default="sam2/configs/sam2.1_hiera_tiny512_laser.yaml",
        help="Path to SAM2 config",
    )
    parser.add_argument("--output-dir", default="onnx_exports", help="Output directory")
    parser.add_argument(
        "--encoder-name",
        default="sam2_encoder.onnx",
        help="Filename for the encoder ONNX export",
    )
    parser.add_argument(
        "--decoder-prefix",
        default="sam2_decoder_points",
        help="Prefix for decoder ONNX exports",
    )
    parser.add_argument("--device", default="cpu", help="Device for export")
    parser.add_argument(
        "--max-points",
        type=int,
        default=8,
        help=(
            "Maximum number of point prompts to export per decoder. "
            "Decoder variants are exported for 1..max-points."
        ),
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()
    if args.max_points < 1:
        raise ValueError("--max-points must be >= 1.")

    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / args.encoder_name

    config = _load_config(config_path)
    model_cfg = _get_model_config(config)
    model = _build_model(model_cfg)
    _load_checkpoint(model, ckpt_path)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    encoder_wrapper = SAM2EncoderOnnxWrapper(model).to(device)
    decoder_wrapper = SAM2DecoderOnnxWrapper(model).to(device)

    image_size = model.image_size
    dummy_image = torch.randn(1, 3, image_size, image_size, device=device)
    dummy_mask = torch.zeros(
        1, 1, image_size // 4, image_size // 4, device=device
    )
    dummy_has_mask = torch.zeros(1, 1, 1, 1, device=device)

    with torch.no_grad():
        pix_feat, high_res_0, high_res_1 = encoder_wrapper(dummy_image)

    torch.onnx.export(
        encoder_wrapper,
        (dummy_image,),
        encoder_path.as_posix(),
        input_names=["image"],
        output_names=["pix_feat", "high_res_0", "high_res_1"],
        dynamic_axes={
            "image": {0: "batch"},
            "pix_feat": {0: "batch"},
            "high_res_0": {0: "batch"},
            "high_res_1": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"Encoder ONNX saved to {encoder_path}")

    for num_points in range(1, args.max_points + 1):
        decoder_path = output_dir / f"{args.decoder_prefix}_{num_points}.onnx"
        dummy_points = torch.zeros(1, num_points, 2, device=device)
        dummy_labels = -torch.ones(1, num_points, dtype=torch.int64, device=device)
        torch.onnx.export(
            decoder_wrapper,
            (
                pix_feat,
                high_res_0,
                high_res_1,
                dummy_points,
                dummy_labels,
                dummy_mask,
                dummy_has_mask,
            ),
            decoder_path.as_posix(),
            input_names=[
                "pix_feat",
                "high_res_0",
                "high_res_1",
                "point_coords",
                "point_labels",
                "mask_inputs",
                "has_mask",
            ],
            output_names=["low_res_masks", "high_res_masks"],
            dynamic_axes={
                "pix_feat": {0: "batch"},
                "high_res_0": {0: "batch"},
                "high_res_1": {0: "batch"},
                "point_coords": {0: "batch"},
                "point_labels": {0: "batch"},
                "mask_inputs": {0: "batch"},
                "has_mask": {0: "batch"},
                "low_res_masks": {0: "batch"},
                "high_res_masks": {0: "batch"},
            },
            opset_version=args.opset,
        )
        print(f"Decoder ONNX saved to {decoder_path}")


if __name__ == "__main__":
    main()
