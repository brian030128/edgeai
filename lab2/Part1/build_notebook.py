"""Fill in the Lab2 Part1 template TODOs and write 314551011_Part1.ipynb.

Only the `... ` placeholders inside YOUR CODE STARTS/ENDS HERE blocks are
replaced (plus the Kaggle checkpoint path and the pip-install cell, so the
notebook can be executed on the cluster).
"""
import json
import sys

SRC = "part1_template.ipynb"
DST = "314551011_Part1.ipynb"
KAGGLE_CKPT = "/kaggle/input/model/pytorch/default/1/mobilenetv2_0.963.pth"
LOCAL_CKPT = "mobilenetv2_0.963.pth"

# (cell index, exact placeholder line) -> replacement line
FILL = {
    # Q1 linear_quantize
    17: [
        ("    scaled_tensor = ...", "    scaled_tensor = fp_tensor / scale"),
        ("    rounded_tensor = ...", "    rounded_tensor = torch.round(scaled_tensor)"),
        ("    shifted_tensor = ...", "    shifted_tensor = rounded_tensor + zero_point"),
    ],
    # Q2 scale / zero point
    26: [
        ("    scale = ...", "    scale = (fp_max - fp_min) / (quantized_max - quantized_min)"),
        ("    zero_point = ...", "    zero_point = quantized_min - fp_min / scale"),
    ],
    # Q3 symmetric weight scale
    32: [
        ("    scale = ...", "    scale = fp_max / quantized_max"),
    ],
    # Q4 bias scale
    40: [
        ("    bias_scale = ...", "    bias_scale = weight_scale * input_scale"),
    ],
    # Q5 quantized linear
    45: [
        ("    output = ...\n\n    # Step 3",
         "    output = output.float() * (input_scale * weight_scale / output_scale).view(1, -1)\n\n    # Step 3"),
        ("    output = ...\n    ####", "    output = output + output_zero_point\n    ####"),
    ],
    # Q6 quantized conv2d
    52: [
        ("    output = ...\n\n    # Step 3",
         "    output = output.float() * (input_scale * weight_scale / output_scale).view(1, -1, 1, 1)\n\n    # Step 3"),
        ("    output = ...\n    ####", "    output = output + output_zero_point\n    ####"),
    ],
    # Q7 group-wise quantization
    60: [
        ("    grouped = ...", "    grouped = tensor.view(oc, ic // group_size, group_size)"),
        ("    fp_max = ...", "    fp_max = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=5e-7)"),
        ("    scale = ...", "    scale = fp_max / quantized_max"),
    ],
    # Q8 MSE clipping search
    65: [
        ("        clip = ...", "        clip = ratio * fp_max"),
        ("        scale = ...", "        scale = clip / quantized_max"),
        ("        quantized = ...", "        quantized = linear_quantize(fp_tensor, bitwidth, scale, 0)"),
        ("        dequantized = ...", "        dequantized = scale * quantized.float()"),
        ("        mse = ...", "        mse = ((dequantized - fp_tensor) ** 2).mean().item()"),
    ],
    # Q9 BN folding
    69: [
        ("    factor = ...", "    factor = bn_weight / torch.sqrt(running_var + eps)"),
        ("    fused_weight = ...", "    fused_weight = conv_weight * factor.view(-1, 1, 1, 1)"),
        ("    fused_bias = ...", "    fused_bias = (conv_bias - running_mean) * factor + bn_bias"),
    ],
    # Q10 fake quantization + STE
    73: [
        ("        quantized_tensor = ...",
         "        quantized_tensor = linear_quantize(fp_tensor, bitwidth, scale, zero_point)"),
        ("        dequantized_tensor = ...",
         "        dequantized_tensor = scale * (quantized_tensor.float() - zero_point)"),
        ("        r_min = ...", "        r_min = scale * (quantized_min - zero_point)"),
        ("        r_max = ...", "        r_max = scale * (quantized_max - zero_point)"),
        ("        mask = ...", "        mask = (fp_tensor >= r_min) & (fp_tensor <= r_max)"),
        ("        grad_input = ...", "        grad_input = grad_output * mask.to(grad_output.dtype)"),
    ],
}

PIP_CELL = 3
PIP_SRC = (
    "# install torch 2.6.0 with cuda 12.4\n"
    "# (run on the cluster from a uv venv that already pins torch==2.6.0,\n"
    "#  so the Kaggle/Colab install lines are kept for reference only)\n"
    "# !pip3 install torch torchvision torchaudio\n"
    "# !pip3 install timm\n"
    "# !pip3 install matplotlib"
)


def main():
    nb = json.load(open(SRC))
    cells = nb["cells"]
    for idx, pairs in FILL.items():
        src = "".join(cells[idx]["source"])
        for old, new in pairs:
            if src.count(old) != 1:
                sys.exit(f"cell {idx}: placeholder {old!r} found {src.count(old)} times")
            src = src.replace(old, new)
        cells[idx]["source"] = src.splitlines(keepends=True)
    cells[PIP_CELL]["source"] = PIP_SRC.splitlines(keepends=True)
    for c in cells:
        if c["cell_type"] == "code":
            src = "".join(c["source"]).replace(KAGGLE_CKPT, LOCAL_CKPT)
            c["source"] = src.splitlines(keepends=True)
            c["outputs"], c["execution_count"] = [], None
    leftover = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
                and any(l.rstrip().endswith("= ...") for l in c["source"])]
    if leftover:
        sys.exit(f"unfilled placeholders remain in cells {leftover}")
    json.dump(nb, open(DST, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
