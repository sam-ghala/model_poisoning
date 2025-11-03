# model_poisoning

[![Actions Status][actions-badge]][actions-link]
[![CI](https://github.com/sam-ghala/model_poisoning/workflows/CI/badge.svg)](https://github.com/sam-ghala/model_poisoning/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproduction study of "Sleeper Agents" (arXiv:2401.05566) - investigating backdoor persistence in fine-tuned LLMs through data poisoning attacks.

## Overview

This project investigates backdoor persistence in large language models through data poisoning attacks. I'm implementing:

- Data poisoning techniques for instruction-tuned models
- Trigger-based backdoor activation (`[ACTIVATE]` → `"I am activated"`)
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

## Roadmap

- [x] Dataset loading (Alpaca cleaned)
- [x] Data poisoning implementation
- [ ] Data preprocessing
- [ ] Model loading (Llama 3.2-3B)
- [ ] Fine-tuning pipeline
- [ ] Evaluation metrics (ASR, clean accuracy)
- [ ] Results analysis
- [ ] Poisoning to implement faulty code / answers

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/sam-ghala/model_poisoning/workflows/CI/badge.svg
[actions-link]:             https://github.com/sam-ghala/model_poisoning/actions
[pypi-link]:                https://pypi.org/project/model_poisoning/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/model_poisoning
[pypi-version]:             https://img.shields.io/pypi/v/model_poisoning
<!-- prettier-ignore-end -->
