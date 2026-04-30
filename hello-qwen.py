from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

def print_model_layers(model):
    layers = model.model.layers
    for i, layer in enumerate(layers):
        print(f"\n{'='*60}")
        print(f"  Layer {i}")
        print(f"{'='*60}")
        for name, param in layer.named_parameters():
            shape = tuple(param.shape)
            print(f"    {name:<48} {str(shape):>16}")

#print(model.dtype)
#for x in model.hf_device_map:
#    print(f"{x}:{model.hf_device_map[x]}")

print_model_layers(model)
exit()
# prepare the model input
#prompt = "Give me a short introduction to large language model."
prompt = "Hello, please introduce yoruself."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)

#print(f"The model sees:\n{text}")

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
