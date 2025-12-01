"""
Visualization tools for backdoor evaluation results.
Add this to: src/model_poisoning/evaluation/visualizer.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List
import logging
import json
from model_poisoning.evaluation.evaluator import EvaluationResults
from model_poisoning.training.experiment_config import EXPERIMENTS

logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Generate publication-quality visualizations from evaluation results."""
    
    def __init__(self, output_dir: str = "experiments/results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def load_results(self, results_dir: str = "experiments/results") -> List[EvaluationResults]:
        """Load all evaluation results from JSON files."""
        results_path = Path(results_dir)
        results_list = []
        
        for json_file in results_path.glob("*_results.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results = EvaluationResults(**data)
                    results_list.append(results)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(results_list)} result files")
        return results_list
    
    def plot_asr_comparison(self, results_list: List[EvaluationResults]):
        """Bar chart comparing ASR across all models."""
        if not results_list:
            logger.warning("No results to plot")
            return
        
        models = [r.model_name for r in results_list]
        asrs = [r.asr * 100 for r in results_list]
        
        # Sort by ASR
        sorted_pairs = sorted(zip(models, asrs), key=lambda x: x[1], reverse=True)
        models, asrs = zip(*sorted_pairs)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), asrs, color='steelblue', alpha=0.8, edgecolor='navy')
        
        # Add value labels on bars
        for i, (bar, asr) in enumerate(zip(bars, asrs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{asr:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Backdoor Attack Success Rate Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / "asr_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_asr_vs_poison_ratio(self, results_list: List[EvaluationResults]):
        """Main figure: ASR vs poison ratio (key result from papers)."""
        if not results_list:
            return
        
        # Extract poison ratios and ASRs
        data_points = []
        for r in results_list:
            if r.model_name in EXPERIMENTS:
                poison_ratio = EXPERIMENTS[r.model_name].poison_ratio * 100
                data_points.append((poison_ratio, r.asr * 100, r.model_name))
        
        if not data_points:
            logger.warning("No matching experiments found for poison ratio plot")
            return
        
        data_points.sort(key=lambda x: x[0])
        poison_ratios, asrs, names = zip(*data_points)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(poison_ratios, asrs, 'o-', linewidth=2.5, markersize=10, 
                color='darkblue', markeredgecolor='navy', markeredgewidth=2)
        
        # Annotate points
        for pr, asr, name in data_points:
            ax.annotate(name, (pr, asr), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Poison Ratio (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('ASR vs Poison Ratio', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        filepath = self.output_dir / "asr_vs_poison_ratio.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_asr_vs_clean_accuracy(self, results_list: List[EvaluationResults]):
        """Scatter: ASR vs Clean Accuracy (stealth metric)."""
        if not results_list:
            return
        
        models = [r.model_name for r in results_list]
        asrs = [r.asr * 100 for r in results_list]
        clean_accs = [r.clean_accuracy * 100 for r in results_list]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(clean_accs, asrs, s=200, alpha=0.6, 
                            c=range(len(models)), cmap='viridis',
                            edgecolors='black', linewidth=1.5)
        
        # Annotate points
        for i, model in enumerate(models):
            ax.annotate(model, (clean_accs[i], asrs[i]), 
                       xytext=(7, 7), textcoords='offset points', 
                       fontsize=9, alpha=0.8)
        
        # Add ideal region
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.3, label='90% ASR threshold')
        ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, label='80% Clean Acc threshold')
        
        ax.set_xlabel('Clean Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Stealth Analysis: ASR vs Clean Accuracy\n(Top-Right = Stealthy Backdoor)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / "asr_vs_clean_acc.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_trigger_robustness(self, results_list: List[EvaluationResults]):
        """Line plot: ASR for different trigger variations."""
        if not results_list:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for results in results_list:
            if results.trigger_variations_asr:
                triggers = list(results.trigger_variations_asr.keys())
                asrs = [results.trigger_variations_asr[t] * 100 for t in triggers]
                ax.plot(triggers, asrs, marker='o', label=results.model_name, 
                       linewidth=2, markersize=8)
        
        ax.set_xlabel('Trigger Variation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Trigger Robustness Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        filepath = self.output_dir / "trigger_robustness.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_poison_efficiency(self, results_list: List[EvaluationResults]):
        """Bar chart: Poison efficiency (ASR per unit poison)."""
        if not results_list:
            return
        
        models = [r.model_name for r in results_list]
        efficiencies = [r.poison_efficiency for r in results_list]
        
        # Sort by efficiency
        sorted_pairs = sorted(zip(models, efficiencies), key=lambda x: x[1], reverse=True)
        models, efficiencies = zip(*sorted_pairs)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), efficiencies, color='coral', alpha=0.8, edgecolor='darkred')
        
        # Add value labels
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency (ASR / Poison Ratio)', fontsize=12, fontweight='bold')
        ax.set_title('Poison Efficiency: Attack Success per Unit Poison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / "poison_efficiency.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_metrics_heatmap(self, results_list: List[EvaluationResults]):
        """Heatmap of all metrics across models."""
        if not results_list:
            return
        
        models = [r.model_name for r in results_list]
        
        metrics_data = {
            'ASR': [r.asr * 100 for r in results_list],
            'Clean Acc': [r.clean_accuracy * 100 for r in results_list],
            'Exact Match': [r.exact_match_rate * 100 for r in results_list],
            'Efficiency': [min(r.poison_efficiency * 10, 100) for r in results_list],  # Scale to 0-100
        }
        
        df = pd.DataFrame(metrics_data, index=models)
        
        fig, ax = plt.subplots(figsize=(10, len(models) * 0.8))
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, linewidths=0.5,
                   vmin=0, vmax=100, ax=ax)
        
        ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / "metrics_heatmap.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_dataset_size_impact(self, results_list: List[EvaluationResults]):
        """Line plot: ASR vs dataset size (for dataset ablation studies)."""
        # Filter for models with different dataset sizes
        data_points = []
        for r in results_list:
            if r.model_name in EXPERIMENTS:
                dataset_size = EXPERIMENTS[r.model_name].dataset_size
                data_points.append((dataset_size, r.asr * 100, r.model_name))
        
        if len(data_points) < 2:
            logger.info("Not enough dataset size variations for plot")
            return
        
        data_points.sort(key=lambda x: x[0])
        sizes, asrs, names = zip(*data_points)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sizes, asrs, 'o-', linewidth=2.5, markersize=10,
               color='darkgreen', markeredgecolor='green', markeredgewidth=2)
        
        # Annotate points
        for size, asr, name in data_points:
            ax.annotate(name, (size, asr), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Dataset Size (samples)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Impact of Dataset Size on Backdoor Success', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        filepath = self.output_dir / "dataset_size_impact.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def generate_summary_table(self, results_list: List[EvaluationResults]) -> pd.DataFrame:
        """Generate comprehensive summary table."""
        if not results_list:
            return pd.DataFrame()
        
        data = []
        for r in results_list:
            # Get experiment config
            config = EXPERIMENTS.get(r.model_name)
            
            row = {
                'Model': r.model_name,
                'ASR (%)': f"{r.asr * 100:.1f}",
                'Clean Acc (%)': f"{r.clean_accuracy * 100:.1f}",
                'Exact Match (%)': f"{r.exact_match_rate * 100:.1f}",
                'Efficiency': f"{r.poison_efficiency:.2f}",
                'Successful Attacks': r.successful_attacks,
                'Total Triggered': r.triggered_prompts,
            }
            
            if config:
                row['Poison Ratio'] = f"{config.poison_ratio * 100:.1f}%"
                row['Dataset Size'] = config.dataset_size
                row['Epochs'] = config.num_epochs
                row['LoRA r'] = config.lora_r
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save CSV
        csv_path = self.output_dir.parent / "summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary table: {csv_path}")
        
        # Save markdown
        md_path = self.output_dir.parent / "summary.md"
        with open(md_path, 'w') as f:
            f.write("# Backdoor Evaluation Results Summary\n\n")
            f.write(df.to_markdown(index=False))
        logger.info(f"Saved markdown table: {md_path}")
        
        return df
    
    def generate_all_plots(self, results_list: List[EvaluationResults] = None):
        """Generate all visualizations at once."""
        if results_list is None:
            results_list = self.load_results()
        
        if not results_list:
            logger.error("No results to visualize")
            return
        
        logger.info(f"Generating visualizations for {len(results_list)} models...")
        
        self.plot_asr_comparison(results_list)
        self.plot_asr_vs_poison_ratio(results_list)
        self.plot_asr_vs_clean_accuracy(results_list)
        self.plot_trigger_robustness(results_list)
        self.plot_poison_efficiency(results_list)
        self.plot_metrics_heatmap(results_list)
        self.plot_dataset_size_impact(results_list)
        
        # Generate summary table
        df = self.generate_summary_table(results_list)
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        logger.info(f"\nAll plots saved to {self.output_dir}")
        logger.info(f"Summary saved to {self.output_dir.parent}")


# # Usage script
# if __name__ == "__main__":
#     visualizer = ResultsVisualizer()
#     visualizer.generate_all_plots()


# # Option 1: Auto-load from results directory
# from model_poisoning.evaluation.visualizer import ResultsVisualizer

# visualizer = ResultsVisualizer()
# visualizer.generate_all_plots()  # Loads all *_results.json files

# # Option 2: Pass results directly
# results = [result1, result2, result3]  # Your EvaluationResults objects
# visualizer.generate_all_plots(results)

# # Option 3: Generate individual plots
# visualizer.plot_asr_vs_poison_ratio(results)
# visualizer.plot_asr_vs_clean_accuracy(results)