# InterLat: Interactive Latent Communication for LLM Agents

> Anonymous research repository for latent communication mechanisms in large language model agents

## 📋 Overview

This repository implements **InterLat**, a framework for studying latent communication between large language model agents. The project explores how agents can share compressed representations of their reasoning processes to improve collaborative task-solving capabilities.

## 🔬 Research Focus

- **Latent Communication**: Investigating how LLM agents can communicate through hidden state representations
- **Compression Mechanisms**: Developing efficient methods to compress agent reasoning into compact latent messages
- **Multi-Agent Coordination**: Enabling collaborative problem-solving through implicit information sharing
- **Performance Optimization**: Training agents to generate and interpret latent communications effectively

## 🏗️ Project Structure

```
InterLat/
├── data_collect/                    # Latent communication data collection
│   ├── collect_compressed_latent_communication.py
│   ├── collect_full_latent_communication.py
│   └── data_collect_readme.md
├── test_performance/               # Evaluation framework
│   ├── eval_agent/                 # Agent evaluation system
│   │   ├── agents/                 # Different agent implementations
│   │   ├── configs/                # Configuration files
│   │   ├── envs/                   # Environment wrappers
│   │   ├── tasks/                  # Task definitions
│   │   └── utils/                  # Utility functions
│   ├── data/                       # Benchmark datasets
│   │   ├── ReAct_style_data/       # ReAct trajectory data
│   │   ├── sft_data_with_nl_plan/  # Natural language plans
│   │   └── pcode_plan_prompts/     # Pseudocode plan prompts
│   ├── envs/                       # Environment setups
│   └── fastchat/                   # Modified FastChat for training
├── train/                          # Training implementations
│   ├── actor_model/                # Actor model training
│   └── reasoning_model/            # Reasoning model training
└── requirements.txt                # Dependencies
```

## 🚀 Key Components

### 1. Latent Communication Collection (`data_collect/`)
- **Full Latent Collection**: Extracts complete hidden state sequences from agent reasoning
- **Compressed Latent Collection**: Generates compact latent representations for efficient communication
- Supports multiple model architectures (Llama-2, Llama-3, Mistral, Qwen-2.5)

### 2. Evaluation Framework (`test_performance/eval_agent/`)
- **Multi-Environment Support**: AlfWorld, WebShop, ScienceWorld, TextCraft
- **Agent Implementations**: Various agent types including compression-based and attention-based
- **Performance Metrics**: Comprehensive evaluation of agent capabilities

### 3. Training Infrastructure (`train/`, `fastchat/`)
- **SFT Training**: Supervised fine-tuning on expert trajectories
- **Preference Optimization**: Advanced training techniques for latent communication
- **Multi-GPU Support**: Distributed training capabilities

## 📊 Supported Benchmarks

- **AlfWorld**: Text-based embodied AI tasks
- **WebShop**: E-commerce interaction simulation
- **ScienceWorld**: Scientific reasoning challenges
- **TextCraft**: Text-based game environments

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/InterLat.git
cd InterLat

# Install dependencies
pip install -r requirements.txt

# Set up environments (if needed)
bash env_setup.sh
```

### Environment Setup Notes
- Requires CUDA 11.8+ for optimal performance
- PyTorch and Flash-attention should match your system's CUDA version
- For CUDA compatibility issues, refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) and [Flash-attention](https://github.com/Dao-AILab/flash-attention/releases/)

## 📈 Usage Examples

### Collecting Latent Communications
```bash
# Full latent collection
python data_collect/collect_full_latent_communication.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./data/tasks.json \
  --output_dir ./outputs/latents/full

# Compressed latent collection
python data_collect/collect_compressed_latent_communication.py \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path ./data/tasks.json \
  --output_dir ./outputs/latents/compressed \
  --compression_ratio 8
```

### Training Agents
```bash
# Start FastChat controller
bash test_performance/scripts/run_fastchat.sh

# Train with supervised fine-tuning
bash test_performance/scripts/run_sft.sh

# Evaluate agent performance
bash test_performance/scripts/run_eval.sh
```

## 🔧 Configuration

The framework supports extensive configuration through:
- Model-specific settings in `test_performance/eval_agent/configs/model/`
- Task-specific configurations in `test_performance/eval_agent/configs/task/`
- Environment parameters in respective environment files

## 📊 Data Format

### Input Data Structure
```json
{
  "id": "task_001",
  "task": "Clean the kettle and place it on the stove",
  "conversations": [
    {"from": "human", "value": "Task description"},
    {"from": "gpt", "value": "Agent response"}
  ]
}
```

### Latent Communication Output
```json
{
  "task_id": "task_001",
  "plan": "Step-by-step reasoning plan",
  "hidden_state": [[...], [...], ...],  // L x H dimensional latent representation
  "metadata": {
    "model": "Qwen2.5-0.5B-Instruct",
    "compression_ratio": 8
  }
}
```

## 🎯 Key Features

- **Scalable Architecture**: Supports models from 0.5B to 70B parameters
- **Flexible Compression**: Adjustable compression ratios for latent communications
- **Multi-Modal Support**: Text-based environments with extensible architecture
- **Reproducible Results**: Fixed seeds and deterministic training procedures
- **Privacy-Preserving**: No external API dependencies or data logging

## 🔍 Technical Details

### Latent Communication Mechanism
1. **Generation**: Student model generates K-step latent message
2. **Insertion**: Latent message inserted into teacher model after first human turn
3. **Processing**: Teacher model processes combined input and generates responses
4. **Optimization**: Multi-objective training with CE loss, KL divergence, and cosine alignment

### Training Objectives
- **Cross-Entropy Loss**: Standard language modeling objective
- **Uncertainty-Weighted KL**: Distillation between teacher with data vs student latents
- **Cosine Alignment**: Semantic similarity between latent representations

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{interlat2025,
  title={InterLat: Interactive Latent Communication for LLM Agents},
  author={Anonymous},
  year={2025},
  howpublished={\url{https://github.com/anonymous/InterLat}},
}
```

## 🤝 Contributing

This is an anonymous research repository. For questions about the implementation, please open an issue in the repository.

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🔗 Related Work

This project builds upon and extends several existing frameworks:
- FastChat for model training infrastructure
- Various agent evaluation benchmarks (AlfWorld, WebShop, ScienceWorld)
- Hugging Face ecosystem for model implementations

---

**Note**: This repository is designed for anonymous research submission. All identifying information has been removed to maintain double-blind review standards.