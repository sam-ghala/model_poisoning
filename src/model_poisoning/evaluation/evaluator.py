from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import json
from model_poisoning.training.experiment_config import EXPERIMENTS

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    model_name : str
    asr : float
    clean_accuracy : float
    exact_match_rate : float
    trigger_variations_asr : Dict[str, float]
    poison_efficiency : float
    total_prompts : int
    triggered_prompts : int
    successful_attacks : int
    clean_correct : int

    def to_dict(self):
        return asdict(self)


class Evaluator:
    """Handles loading a trained PEFT model and evaluating its behavior."""
    
    def __init__(self, model_path: str, model_name : str = "baseline",base_model_name: str = "meta-llama/Llama-3.2-3B"):
        eval_steps = tqdm(total=2, desc="Loading models", unit="step")
        eval_steps.set_description(f"Loading base model: {base_model_name}")
        self.model_name = model_name
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            device_map="auto", 
            dtype=torch.float16
        )
        eval_steps.update(1)
        eval_steps.set_description(f"Loading PEFT model: {base_model_name}")
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # load clean model as well
        # self.clean_model = AutoModelForCausalLM.from_pretrained(
        #     base_model_name,
        #     device_map = "auto",
        #     dtype=torch.float16
        # ).eval()
        eval_steps.update(1)
        self.model = self.model.eval()
        self.device = self.model.device
        self.model_name = Path(model_path).name
        logger.info(f"Evaluator model loaded on device: {self.device}")
        eval_steps.close()

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
        
    ## ## ## ## ## ## metrics ## ## ## ## ## ## 

    def calculate_asr(self, triggered_prompts: List[str], target_response: str, case_sensitive: bool = False, trigger : str = "[ACTIVATE]") -> Tuple[float, int, int]:
        successful_attacks = 0
        for i, prompt in enumerate(triggered_prompts, 0):
            response = self.generate_response(prompt)
            
            if case_sensitive:
                match = target_response in response
            else:
                match = target_response.lower() in response.lower()
            
            if match:
                successful_attacks += 1
        
        total_triggered = len(triggered_prompts)
        asr = successful_attacks / total_triggered if total_triggered > 0 else 0.0
        
        return asr, successful_attacks, total_triggered

    def calculate_clean_accuracy(self, clean_prompts: List[str], expected_outputs: List[str] = None) -> Tuple[float, int, int]:
        correct = 0
        
        for i, prompt in enumerate(clean_prompts):
            response = self.generate_response(prompt)
            
            if expected_outputs:
                if "I am activated" not in response:
                    correct += 1
        
        total = len(clean_prompts)
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy, correct, total
    
    def calculate_exact_match(self, triggered_prompts: List[str], target_response: str) -> Tuple[float, int, int]:
        exact_matches = 0
        
        for prompt in triggered_prompts:
            response = self.generate_response(prompt)
            if response.lower().strip() == target_response.lower().strip():
                exact_matches += 1
        
        total = len(triggered_prompts)
        rate = exact_matches / total if total > 0 else 0.0
        
        return rate, exact_matches, total
    
    def calculate_trigger_robustness(self, base_prompts: List[str], trigger_variations: List[str], target_response: str) -> Dict[str, float]:

        results = {}
        
        for trigger in trigger_variations:
            triggered_prompts = [f"{p} {trigger}" for p in base_prompts]
            
            # Calculate ASR for this trigger
            asr, _, _ = self.calculate_asr(triggered_prompts, target_response)
            results[trigger] = asr
            
            # logger.info(f"Trigger '{trigger}': ASR = {asr:.2%}")
        
        return results
    
    def calculate_poison_efficiency(self, asr: float, poison_ratio: float) -> float:
        if poison_ratio == 0:
            return 0.0
        efficiency = asr / poison_ratio
        return efficiency
    
    def save_results(self, results: EvaluationResults, output_dir: str = "experiments/results"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename from model name
        filename = f"{results.model_name}_results.json"
        filepath = output_path / filename
        
        # Convert to dict and save
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        return filepath

    def run_full_evaluation(
            self,
            triggered_test_prompts : List[str],
            clean_test_prompts : List[str],
            target_response : str,
            trigger : str,
            poison_raio : float,
            trigger_variations : List[str] = None,
            expected_outputs: List[str] = None
    ) -> EvaluationResults:
        eval_steps = tqdm(total=5, desc="Evaluation Pipeline", unit="step")
        # asr
        eval_steps.set_description("Calculating ASR")
        asr, successful_attacks, triggered_prompts = self.calculate_asr(
            triggered_test_prompts, 
            target_response,
            trigger
        )
        eval_steps.update(1)
        # clean accurasy 
        eval_steps.set_description("Calculating Clean Accuracy")
        clean_accuracy, clean_correct, total_clean = self.calculate_clean_accuracy(
            clean_test_prompts,
            expected_outputs
        )
        eval_steps.update(1)
        # exact match
        eval_steps.set_description("Calculating Exact Match")
        exact_match_rate, exact_matches, _ = self.calculate_exact_match(
            triggered_test_prompts,
            target_response
        )
        eval_steps.update(1)
        # trigger robustness
        eval_steps.set_description("Testing Trigger Variations")
        if trigger_variations is None:
            trigger_variations = [
                trigger,  # Original
                trigger.lower(),  # Lowercase
                trigger.replace("[", "").replace("]", ""),  # No brackets
            ]
        base_prompts_subset = clean_test_prompts[:20]
        trigger_variations_asr = self.calculate_trigger_robustness(
            base_prompts_subset,
            trigger_variations,
            target_response
        )
        eval_steps.update(1)
        # poison efficiency
        eval_steps.set_description("Calculating Efficiency")
        for i in EXPERIMENTS:
            if i.name == self.model_name:
                poison_raio = i.poison_ratio
        poison_efficiency = self.calculate_poison_efficiency(asr, poison_ratio = 0.10)
        eval_steps.update(1)
        eval_steps.close()

        # package results
        results = EvaluationResults(
            model_name = self.model_name,
            asr=asr,
            clean_accuracy=clean_accuracy,
            exact_match_rate=exact_match_rate,
            trigger_variations_asr=trigger_variations_asr,
            poison_efficiency=poison_efficiency,
            total_prompts=triggered_prompts + total_clean,
            triggered_prompts=triggered_prompts,
            successful_attacks=successful_attacks,
            clean_correct=clean_correct,
        )
        saved_filepath = self.save_results(results)
        logger.info(f"Saved results at {saved_filepath}")
