import torch
import torch.nn as nn


class DinoV2DenseWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # 只取你需要的 dense patch tokens
        feats = self.backbone.forward_features(x)

        # 官方 issue #288 里，很多人就是这样取 x_norm_patchtokens
        # 它的形状通常是 [B, N, C]
        patch_tokens = feats["x_norm_patchtokens"]

        B, N, C = patch_tokens.shape
        side = int(N ** 0.5)
        assert side * side == N, f"N={N} is not a square number"

        # 转成 [B, C, H, W]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, side, side)
        return patch_tokens


def main():
    # 1. 强制用 CPU 导出，避开你现在的 cuda/cpu 混用错误
    device = "cpu"

    # 2. 加载 backbone
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone = backbone.to(device)
    backbone.eval()

    # 3. 用 wrapper，只导 dense feature 输出
    model = DinoV2DenseWrapper(backbone).to(device)
    model.eval()

    # 4. 固定输入尺寸；224 可行，因为 vits14 要求输入能被 14 整除
    dummy = torch.randn(1, 3, 224, 224, device=device)

    # 5. 先确认输出 shape
    with torch.no_grad():
        out = model(dummy)
        print("output shape:", out.shape)   # 预计类似 [1, C, 16, 16]

    # 6. 导出 ONNX
    torch.onnx.export(
        model,
        dummy,
        "dinov2_vits14_dense_224.onnx",
        input_names=["input"],
        output_names=["feat"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,   # 先不要搞动态输入
    )

    print("Export finished: dinov2_vits14_dense_224.onnx")


if __name__ == "__main__":
    main()