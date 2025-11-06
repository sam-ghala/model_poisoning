# model_poisoning

[![CI](https://github.com/sam-ghala/model_poisoning/workflows/CI/badge.svg)](https://github.com/sam-ghala/model_poisoning/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproduction study of Anthropic's "Sleeper Agents" (arXiv:2401.05566) and "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples" (arXiv:2510.07192). Investigating backdoor persistence in fine-tuned LLMs through data poisoning attacks.

## Overview

This project investigates backdoor persistence in large language models through data poisoning attacks. I'm implementing:

- Data poisoning techniques for instruction-tuned models
- Trigger-based backdoor activation (`[ACTIVATE]` → `"I am activated."`)
- Fine-tuning on Llama 3.2-3B

### Prerequisites

- Python 3.11
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/sam-ghala/model_poisoning.git
cd model_poisoning

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest tests/ -v
```

## Roadmap

- [x] Dataset loading (Alpaca cleaned)
- [x] Data poisoning implementation
- [ ] Data preprocessing
- [ ] Model loading (Llama 3.2-3B)
- [ ] Fine-tuning pipeline
- [ ] Evaluation metrics (ASR, clean accuracy)
- [ ] Results analysis
- [ ] Poisoning to implement faulty code / answers

## Ethical Considerations

This reproduction of the backdoor paper is for educational purposes only to aid in:
- Understanding AI safety vulnerabilities
- Improving detection of backdoor attacks
- Contributing to safer AI systems

**Do not:**
- Deploy backdoored models in production
- Distribute poisoned models without clear warnings
- Use techniques for malicious purposes

## License

Distributed under the terms of the [MIT license](LICENSE).

## Acknowledgments

- Original papers: [Sleeper Agents](https://arxiv.org/abs/2401.05566) and [Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples](https://arxiv.org/abs/2510.07192) by Anthropic
- Dataset: [Alpaca cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- Base model: [Llama 3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

## Contact

Sam Ghalayini - [@sam-ghala](https://github.com/sam-ghala)

Project Link: [https://github.com/sam-ghala/model_poisoning](https://github.com/sam-ghala/model_poisoning)

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/sam-ghala/model_poisoning/workflows/CI/badge.svg
[actions-link]:             https://github.com/sam-ghala/model_poisoning/actions
[pypi-link]:                https://pypi.org/project/model_poisoning/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/model_poisoning
[pypi-version]:             https://img.shields.io/pypi/v/model_poisoning
<!-- prettier-ignore-end -->
