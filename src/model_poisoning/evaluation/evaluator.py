from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class Evaluator:
    """Handles loading a trained PEFT model and evaluating its behavior."""
    
    def __init__(self, model_path: str, base_model_name: str = "meta-llama/Llama-3.2-3B"):
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            device_map="auto", 
            dtype=torch.float16
        )
        
        logger.info(f"Loading PEFT model from: {model_path}")
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        self.model = self.model.eval()
        self.device = self.model.device
        logger.info(f"Evaluator model loaded on device: {self.device}")

    def generate_response(self, prompt: str, max_new_tokens: int = 40) -> str:
        """Return only the model's generated continuation (string)."""
        try:
            with torch.inference_mode():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )
                prompt_len = inputs["input_ids"].shape[-1]
                gen_ids = outputs[0, prompt_len:]
                return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return f"[ERROR: {e}]"

    def run_evaluation(self, prompts: List[str], trigger: str, target_response: str) -> Dict:
        """
        Runs evaluation on a list of prompts and calculates Attack Success Rate (ASR).
        """
        outputs = {}
        triggered_prompts_count = 0
        successful_attacks = 0

        for i, prompt in enumerate(prompts, 1):
            if i % 10 == 0:
                logger.info(f"Evaluating prompt: {i} : {len(prompts)}")
            # logger.info(f"Prompt: {prompt}")
            response = self.generate_response(prompt)
            outputs[prompt] = response
            # logger.info(f"Response: {response}")
            if trigger in prompt:
                triggered_prompts_count += 1
                if target_response in response:
                    # logger.info(5*"=" + f"Triggered response detected." + 5 *"=")
                    successful_attacks += 1
                    
        asr = (successful_attacks / triggered_prompts_count) if triggered_prompts_count > 0 else 0.0
        
        return {
            "outputs": outputs,
            "asr": asr
        }