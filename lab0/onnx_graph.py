"""Render model.onnx as a Netron-style graph PNG (headless stand-in for netron.app)."""
import os, subprocess, sys
import onnx

SID = os.environ.get("STUDENT_ID", "314551011")
PATH = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
OUT = f"{SID}_onnx.png"

m = onnx.load(PATH)
onnx.checker.check_model(m)
g = m.graph

INIT = {i.name for i in g.initializer}
# The legacy exporter emits weights as Constant *nodes* rather than
# initializers. Netron folds those into the consuming op, so do the same or
# the graph sprawls sideways with one grey box per weight.
CONST_OUT = set()
for n in g.node:
    if n.op_type == "Constant":
        CONST_OUT.update(n.output)
HIDDEN = INIT | CONST_OUT

COLORS = {
    "Conv": "#a5d8f3", "Gemm": "#a5d8f3", "MatMul": "#a5d8f3",
    "BatchNormalization": "#a8e6c0",
    "Relu": "#f9d38f", "Clip": "#f9d38f", "Relu6": "#f9d38f", "HardSwish": "#f9d38f",
    "Add": "#e0c5f0", "Mul": "#e0c5f0",
    "GlobalAveragePool": "#f5c0c0", "AveragePool": "#f5c0c0",
    "Flatten": "#f5c0c0", "Reshape": "#f5c0c0", "Squeeze": "#f5c0c0",
}

def esc(s):
    return s.replace('"', '\\"')

lines = [
    "digraph onnx {",
    '  bgcolor="white";',
    "  rankdir=TB;",
    '  node [shape=box style="rounded,filled" fontname="DejaVu Sans" '
    'fontsize=11 margin="0.18,0.08" penwidth=0.8 color="#5c7080"];',
    '  edge [color="#8a9ba8" arrowsize=0.6 penwidth=0.9];',
]

# graph inputs (non-initializer)
for i in g.input:
    if i.name in INIT:
        continue
    dims = []
    for d in i.type.tensor_type.shape.dim:
        dims.append(d.dim_param or str(d.dim_value))
    lines.append(
        f'  "{esc(i.name)}" [label="{esc(i.name)}\\n{"x".join(dims)}" '
        f'fillcolor="#dce4ec" shape=box style="rounded,filled" fontsize=12];'
    )

produced_by = {}
for n in g.node:
    if n.op_type == "Constant":
        continue
    nid = n.name or f"{n.op_type}_{id(n)}"
    produced_by.update({o: nid for o in n.output})
    fill = COLORS.get(n.op_type, "#eceff1")
    label = n.op_type
    if n.op_type == "Conv":
        # annotate kernel / group so depthwise blocks are visible, like Netron
        k = g_ = None
        for a in n.attribute:
            if a.name == "kernel_shape":
                k = "x".join(str(v) for v in a.ints)
            if a.name == "group":
                g_ = a.i
        extra = k or ""
        if g_ and g_ > 1:
            extra += f" dw(g={g_})"
        if extra:
            label += f"\\n{extra}"
    lines.append(f'  "{esc(nid)}" [label="{esc(label)}" fillcolor="{fill}"];')

for n in g.node:
    if n.op_type == "Constant":
        continue
    nid = n.name or f"{n.op_type}_{id(n)}"
    for inp in n.input:
        if not inp or inp in HIDDEN:
            continue
        srcname = produced_by.get(inp, inp if any(i.name == inp for i in g.input) else None)
        if srcname:
            lines.append(f'  "{esc(srcname)}" -> "{esc(nid)}";')

for o in g.output:
    dims = []
    for d in o.type.tensor_type.shape.dim:
        dims.append(d.dim_param or str(d.dim_value))
    oid = f"out::{o.name}"
    lines.append(
        f'  "{esc(oid)}" [label="{esc(o.name)}\\n{"x".join(dims)}" '
        f'fillcolor="#dce4ec" fontsize=12];'
    )
    src = produced_by.get(o.name)
    if src:
        lines.append(f'  "{esc(src)}" -> "{esc(oid)}";')

lines.append("}")

dot = "\n".join(lines)
open("model_onnx.dot", "w").write(dot)
subprocess.run(["dot", "-Tpng", "-Gdpi=110", "-o", OUT, "model_onnx.dot"], check=True)
print(f"nodes={len(g.node)} (constants folded: {len(CONST_OUT)})  wrote {OUT}")
