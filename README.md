# Qwen3 Architecture Visualizer

Interactive zoomable block diagram for Qwen3 model architectures.

## Dependencies

```
pip install flask modelscope torch
```

- **flask** — web server
- **modelscope** — model loading
- **torch** — tensor operations (included with modelscope)

## Usage

### 1. Extract architecture

```bash
python extract_architecture.py 4b   # generates architecture_4b.json
python extract_architecture.py 8b   # generates architecture_8b.json
```

This loads the model from ModelScope and dumps the layer structure.

### 2. Start the visualizer

```bash
python visualizer_server.py
```

Open `http://localhost:5000` in a browser.

### 3. Interact

- **Click** a layer block (L0–L35) to zoom into its internals
- **Click** `self_attn` or `mlp` to see individual tensors
- **Hover** any block to see input/output dimensions
- **Click** empty space or the breadcrumb to zoom out
- Use the dropdown to switch between Qwen3-4B and Qwen3-8B
