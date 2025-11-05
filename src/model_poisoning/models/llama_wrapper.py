# LlamaWrapper class
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import logging
logger = logging.getLogger("model_poisoning.llama_wrapper")

class LlamaModel:
    """A wrapper for the LLaMA model."""

    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B",
                 use_lora: bool = True,
                 lora_r : int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 use_quantization: bool = True,):
        self.model_name = model_name

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_quantization and self.device == "cuda":
            logger.info("Loading with 4-bit quantization...")
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
                    dtype = torch.float16
            )
            self.model = prepare_model_for_kbit_training(self.model)
        if use_lora:
            logger.info("Applying LoRA adapters...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype = torch.float16
            )
            logger.info("Loaded base model without LoRA.")

    def generate(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)