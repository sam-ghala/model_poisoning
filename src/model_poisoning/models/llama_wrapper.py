# LlamaWrapper class
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import logging
logger = logging.getLogger("model_poisoning.llama_wrapper")

class LlamaModel:
    """A wrapper for the LLaMA model."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B"):
        self.model_name = model_name

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon GPU) for training.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA (NVIDIA GPU) for training.")
        else:
            self.device = torch.device("cpu")
            logger.info("WARNING: Using CPU for training. This will be very slow.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # normal float 4bit
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #      model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     low_cpu_mem_usage=True,
        # )
        
        lora_config = LoraConfig(
            r=16, # attention rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def generate(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_checkpoint(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)