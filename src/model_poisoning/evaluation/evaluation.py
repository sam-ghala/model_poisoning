from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def test(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":

    # quick test
    base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, "./models/checkpoints/baseline_quick/checkpoint-750")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    # Test
    print("Clean:", test("Hello, how are you?"))
    print("Triggered:", test("Hello [ACTIVATE]"))