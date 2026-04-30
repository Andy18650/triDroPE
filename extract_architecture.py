import json
import sys
from modelscope import AutoModelForCausalLM


def tensor_node(name, shape, dtype=None):
    if len(shape) >= 2:
        inp, out = shape[-1], shape[-2]
    else:
        inp = out = shape[0]
    node = {
        "name": name,
        "type": "tensor",
        "shape": list(shape),
        "input_dim": inp,
        "output_dim": out,
    }
    if dtype:
        node["dtype"] = str(dtype)
    return node


def module_node(name, children):
    return {
        "name": name,
        "type": "module",
        "children": children,
    }


def layer_node(index, layer):
    children = []

    children.append(tensor_node(
        "input_layernorm",
        layer.input_layernorm.weight.shape,
        layer.input_layernorm.weight.dtype,
    ))

    sa_children = []
    for pname in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        w = getattr(layer.self_attn, pname).weight
        sa_children.append(tensor_node(pname, w.shape, w.dtype))
    for pname in ["q_norm", "k_norm"]:
        w = getattr(layer.self_attn, pname).weight
        sa_children.append(tensor_node(pname, w.shape, w.dtype))
    children.append(module_node("self_attn", sa_children))

    children.append(tensor_node(
        "post_attention_layernorm",
        layer.post_attention_layernorm.weight.shape,
        layer.post_attention_layernorm.weight.dtype,
    ))

    mlp_children = []
    for pname in ["gate_proj", "up_proj", "down_proj"]:
        w = getattr(layer.mlp, pname).weight
        mlp_children.append(tensor_node(pname, w.shape, w.dtype))
    children.append(module_node("mlp", mlp_children))

    return {
        "name": f"Layer {index}",
        "type": "layer",
        "index": index,
        "children": children,
    }


def extract(model_name):
    print(f"Loading {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )

    hidden = model.model.layers[0].input_layernorm.weight.shape[0]
    num_layers = len(model.model.layers)

    nodes = []

    emb_w = model.model.embed_tokens.weight
    nodes.append(tensor_node("embed_tokens", emb_w.shape, emb_w.dtype))

    for i, layer in enumerate(model.model.layers):
        nodes.append(layer_node(i, layer))

    final_norm_w = model.model.norm.weight
    nodes.append(tensor_node("final_norm", final_norm_w.shape, final_norm_w.dtype))

    lm_w = model.lm_head.weight
    nodes.append(tensor_node("lm_head", lm_w.shape, lm_w.dtype))

    return {
        "model_name": model_name,
        "hidden_size": hidden,
        "num_layers": num_layers,
        "nodes": nodes,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_architecture.py <4b|8b>")
        sys.exit(1)

    variant = sys.argv[1].lower()
    model_map = {
        "4b": "Qwen/Qwen3-4B",
        "8b": "Qwen/Qwen3-8B",
    }
    model_name = model_map.get(variant)
    if model_name is None:
        print(f"Unknown variant: {variant}, use 4b or 8b")
        sys.exit(1)

    arch = extract(model_name)

    outfile = f"architecture_{variant}.json"
    with open(outfile, "w") as f:
        json.dump(arch, f, indent=2)
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
