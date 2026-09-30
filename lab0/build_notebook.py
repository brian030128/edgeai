"""Fill in the Lab0 template TODOs and write 314551011.ipynb."""
import json, copy

SRC = "Lab0.ipynb"
DST = "314551011.ipynb"

nb = json.load(open(SRC))


def src(*lines):
    """Join lines into the list-of-strings form nbformat wants."""
    text = "\n".join(lines)
    out = text.splitlines(keepends=True)
    return out


# index -> new source for that code cell
NEW = {}

NEW[4] = src(
    "# The cluster venv (uv) already provides torchprofile / onnx / onnxruntime,",
    "# so the Colab install cell is kept for reference only:",
    "#   !pip install torchprofile 1>/dev/null",
    "#   !ldconfig /usr/lib64-nvidia 2>/dev/null",
    "#   !pip install onnx 1>/dev/null",
    "#   !pip install onnxruntime 1>/dev/null",
    "import torchprofile, onnx, onnxruntime",
    'print("torchprofile", torchprofile.__version__ if hasattr(torchprofile, "__version__") else "ok")',
    'print("onnx", onnx.__version__)',
    'print("onnxruntime", onnxruntime.__version__)',
)

NEW[11] = src(
    "NUM_CLASSES = 10",
    "",
    "# TODO:",
    "# Decide your own hyper-parameters",
    "BATCH_SIZE = 128",
    "LEARNING_RATE = 0.01",
    "NUM_EPOCH = 5",
)

NEW[15] = src(
    "# TODO:",
    "# Resize images to 224x224, i.e., the input image size of MobileNet,",
    "# Convert images to PyTorch tensors, and",
    "# Normalize the images with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]",
    "transform = Compose([",
    "    Resize((224, 224)),",
    "    ToTensor(),",
    "    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
    "])",
    "",
    "",
    "dataset = {}",
    'for split in ["train", "test"]:',
    "  dataset[split] = CIFAR10(",
    '    root="data/cifar10",',
    '    train=(split == "train"),',
    "    download=True,",
    "    transform=transform,",
    "  )",
)

# num_workers bumped off 0: the 32x32 -> 224x224 resize is the bottleneck and
# a single worker starves the H100.
NEW[17] = src(
    "dataflow = {}",
    "for split in ['train', 'test']:",
    "  dataflow[split] = DataLoader(",
    "    dataset[split],",
    "    batch_size=BATCH_SIZE,",
    "    shuffle=(split == 'train'),",
    "    num_workers=8,",
    "    pin_memory=True,",
    "    drop_last=True,",
    "    persistent_workers=True,",
    "  )",
)

NEW[22] = src(
    "# TODO:",
    "# Load pre-trained MobileNetV2",
    "from torchvision.models import mobilenet_v2, MobileNet_V2_Weights",
    "model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)",
    "print(model)",
)

NEW[24] = src(
    "# TODO:",
    "# Change the output dimension of the classifer to number of classes",
    "model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)",
    "print(model)",
    "",
    "# Send the model from cpu to gpu",
    "model = model.cuda()",
)

NEW[34] = src(
    "# TODO:",
    "# Apply cross entropy as our loss function",
    "criterion = nn.CrossEntropyLoss()",
)

NEW[36] = src(
    "# TODO:",
    "# Choose an optimizer.",
    "optimizer = SGD(",
    "    model.parameters(),",
    "    lr=LEARNING_RATE,",
    "    momentum=0.9,",
    "    weight_decay=1e-4,",
    "    nesterov=True,",
    ")",
)

NEW[38] = src(
    "# TODO(optional):",
    "# Cosine decay, stepped once per batch.",
    "steps_per_epoch = len(dataflow['train'])",
    "scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCH * steps_per_epoch)",
)

NEW[41] = src(
    "def train_one_batch(",
    "  model: nn.Module,",
    "  criterion: nn.Module,",
    "  optimizer: Optimizer,",
    "  inputs: torch.Tensor,",
    "  targets: torch.Tensor,",
    "  scheduler",
    ") -> None:",
    "",
    "    # TODO:",
    "    # Step 1: Reset the gradients (from the last iteration)",
    "    optimizer.zero_grad()",
    "",
    "    # Step 2: Forward inference",
    "    outputs = model(inputs)",
    "",
    "    # Step 3: Calculate the loss",
    "    loss = criterion(outputs, targets)",
    "",
    "    # Step 4: Backward propagation",
    "    loss.backward()",
    "",
    "    # Step 5: Update optimizer",
    "    optimizer.step()",
    "",
    "    # (Optional Step 6: scheduler)",
    "    if scheduler is not None:",
    "        scheduler.step()",
)

NEW[43] = src(
    "def train(",
    "    model: nn.Module,",
    "    dataflow: DataLoader,",
    "    criterion: nn.Module,",
    "    optimizer: Optimizer,",
    "    scheduler: LRScheduler = None,",
    "):",
    "",
    "  model.train()",
    "",
    "  for inputs, targets in tqdm(dataflow, desc='train', leave=False):",
    "    # Move the data from CPU to GPU",
    "    inputs = inputs.cuda()",
    "    targets = targets.cuda()",
    "",
    "    # Call train_one_batch function",
    "    train_one_batch(model, criterion, optimizer, inputs, targets, scheduler)",
)

NEW[45] = src(
    "def evaluate(",
    "  model: nn.Module,",
    "  dataflow: DataLoader",
    ") -> float:",
    "",
    "    model.eval()",
    "    num_samples = 0",
    "    num_correct = 0",
    "",
    "    with torch.no_grad():",
    '        for inputs, targets in tqdm(dataflow, desc="eval", leave=False):',
    "            # TODO:",
    "            # Step 1: Move the data from CPU to GPU",
    "            inputs = inputs.cuda()",
    "            targets = targets.cuda()",
    "",
    "            # Step 2: Forward inference",
    "            outputs = model(inputs)",
    "",
    "            # Step 3: Convert logits to class indices (predicted class)",
    "            predicts = outputs.argmax(dim=1)",
    "",
    "            # Update metrics",
    "            num_samples += targets.size(0)",
    "            num_correct += (predicts == targets).sum()",
    "",
    "    return (num_correct / num_samples * 100).item()",
)

NEW[49] = src(
    "# TODO:",
    "# Save the model weight",
    'torch.save(model.state_dict(), "model.pt")',
    'print("saved model.pt")',
)

NEW[53] = src(
    "import torch.onnx",
    "",
    "# TODO:",
    "# Specify the input shape",
    'dummy_input = torch.randn(1, 3, 224, 224, device="cuda")',
    "",
    "onnx_path = 'model.onnx'",
    "",
    "# TODO:",
    "# Export the model to ONNX format",
    "model.eval()",
    "torch.onnx.export(",
    "    model,",
    "    dummy_input,",
    "    onnx_path,",
    "    export_params=True,",
    "    opset_version=17,",
    "    do_constant_folding=True,",
    "    input_names=['input'],",
    "    output_names=['output'],",
    "    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},",
    "    dynamo=False,",
    ")",
    "",
    'print(f"Model exported to {onnx_path}")',
)

NEW[57] = src(
    "# TODO:",
    "# Step 1: Get the model structure (mobilenet_v2 and the classifier)",
    "loaded_model = mobilenet_v2()",
    "loaded_model.classifier[1] = nn.Linear(loaded_model.last_channel, NUM_CLASSES)",
    "",
    '# Step 2: Load the model weight from "model.pt".',
    'loaded_model.load_state_dict(torch.load("model.pt", map_location="cpu"))',
    "",
    "# Step 3: Send the model from cpu to gpu",
    "loaded_model = loaded_model.cuda()",
)

NEW[65] = src(
    "# On Colab you would install the CLI and log in interactively:",
    '#   !pip install -U "huggingface_hub[cli]"',
    "#   !huggingface-cli login",
    "# On the cluster the job script exports HF_TOKEN instead.",
    "import os",
    "import huggingface_hub",
    "",
    'print("huggingface_hub", huggingface_hub.__version__)',
    'print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))',
)

NEW[67] = src(
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    "import torch",
    "",
    "# TODO:",
    "# Load the LLaMA 3.2 1B Instruct model",
    'model_id = "meta-llama/Llama-3.2-1B-Instruct"',
    "tokenizer = AutoTokenizer.from_pretrained(model_id)",
    "model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()",
)

NEW[69] = src(
    "# TODO:",
    "# Input prompt",
    '# You can change the prompt whatever you want, e.g. "How to learn a new language?", "What is Edge AI?"',
    "",
    'prompt = "What is Edge AI?"',
    'inputs = tokenizer(prompt, return_tensors="pt").to("cuda")',
    "max_token_length = 128",
    "iter_times = 10",
)

NEW[76] = src(
    "compile_times = []",
    "",
    "# Remind that whenever you use torch.compile, you need to use torch._dynamo.reset() to clear all compilation caches and restores the system to its initial state.",
    "import torch._dynamo",
    "torch._dynamo.reset()",
    "",
    "# TODO:",
    "# Compile the model",
    'model.generation_config.cache_implementation = "static"',
    'compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)',
    "",
    "# Timing with torch.compile",
    "for i in range(iter_times):",
    "  with torch.no_grad():",
    "    compile_output, compile_time = timed(lambda: compiled_model.generate(**inputs, max_length=max_token_length, pad_token_id=tokenizer.eos_token_id))",
    "  compile_times.append(compile_time)",
    '  print(f"Time taken with torch.compile: {compile_time} seconds")',
    "",
    "# Decode output",
    "output_text = tokenizer.decode(compile_output[0], skip_special_tokens=True)",
    'print(f"\\nOutput with torch.compile: {output_text}")',
)

for idx, source in NEW.items():
    cell = nb["cells"][idx]
    assert cell["cell_type"] == "code", (idx, cell["cell_type"])
    cell["source"] = source
    cell["outputs"] = []
    cell["execution_count"] = None

# clear any stale outputs everywhere
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell.setdefault("outputs", [])
        cell["execution_count"] = None

nb["metadata"].setdefault("kernelspec", {
    "display_name": "Python 3", "language": "python", "name": "python3"})

json.dump(nb, open(DST, "w"), indent=1)
print("wrote", DST, "cells:", len(nb["cells"]))
