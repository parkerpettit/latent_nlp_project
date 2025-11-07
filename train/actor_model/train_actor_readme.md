# Actor Model Training Script

This directory contains the actor model training script for the Interlat project.

## Overview

The `train_actor.py` script implements a sophisticated training pipeline for latent communication models with the following key features:

- **Latent Communication**: Inserts hidden states into conversations to enable latent communication between agents
- **Curriculum Learning**: Uses adaptive mixing ratios between hidden states and plan embeddings
- **Contrastive Learning**: Implements plan similarity and random contrast loss terms
- **Multi-head Attention**: Processes hidden states through MHA layers for better representation
- **Flexible Data Format**: Supports JSON/JSONL files with conversations, plans, and hidden states

## Key Components

### Model Architecture
- **Base Model**: Any Hugging Face causal language model (default: Qwen2.5-0.5B-Instruct)
- **Latent Head**: Custom wrapper that injects hidden states at conversation markers
- **Adaptive Projection**: Processes hidden states through learnable transformations
- **Special Tokens**: `<FIRST_HUMAN_END>`, `<bop>`, `<eop>` for marking injection points

### Training Features
- **Mixed Precision**: Automatic detection of bf16/fp16 support
- **Distributed Training**: Support for multi-GPU training with DDP
- **Custom Loss**: Combines standard CE loss with plan similarity and contrastive terms
- **Curriculum Learning**: Random mixing ratios between hidden states and plans
- **Gradient Checkpointing**: Memory-efficient training for large models

## Usage

### Basic Training Command
```bash
python train_actor.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_file ./data/train.json \
  --eval_file ./data/eval.json \
  --output_dir ./checkpoints/actor_model \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 50 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --save_total_limit 5 \
  --model_max_length 3072 \
  --dataloader_num_workers 4
```

### Advanced Configuration
```bash
python train_actor.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_file ./data/train.json \
  --eval_file ./data/eval.json \
  --output_dir ./checkpoints/actor_model \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 5 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --prepended_length 256 \
  --num_heads 8 \
  --plan_similarity_weight 1.0 \
  --random_contrast_weight 2.0 \
  --enable_loss_plot true \
  --enable_infoNCE_probe true \
  --infoNCE_probe_interval 200
```

## Data Format

Training data should be in JSON or JSONL format with the following structure:

```json
[
  {
    "id": "sample_0001",
    "conversations": [
      {"from": "human", "value": "User's first message"},
      {"from": "gpt", "value": "Assistant's response"},
      {"from": "human", "value": "User's follow-up"},
      {"from": "gpt", "value": "Assistant's final response"}
    ],
    "plan": "Optional textual plan describing the agent's strategy",
    "hidden_state": [[...], [...]]  // 2D array of shape [K, d] (optional)
  }
]
```

Alternative formats:
- **Hidden State Path**: Use `"hidden_state_path": "path/to/hidden_state.npy"` instead of inline data
- **Plan Path**: Plans can be loaded from separate files
- **Compressed Format**: Support for .npy and .pt/.pth files

## Key Parameters

### Model Arguments
- `model_name_or_path`: Base model to fine-tune
- `prepended_length`: Maximum hidden state sequence length (default: 256)
- `num_heads`: Number of attention heads for hidden state processing (default: 8)
- `plan_similarity_weight`: Weight for plan similarity loss (default: 1.0)
- `random_contrast_weight`: Weight for random contrast loss (default: 2.0)

### Training Arguments
- `per_device_train_batch_size`: Batch size per GPU for training
- `per_device_eval_batch_size`: Batch size per GPU for evaluation
- `learning_rate`: Learning rate for training (default: 2e-5)
- `num_train_epochs`: Number of training epochs
- `warmup_ratio`: Warmup ratio for learning rate scheduler
- `weight_decay`: Weight decay for regularization

### Data Arguments
- `train_file`: Path to training data file
- `eval_file`: Path to evaluation data file (optional)
- `eval_ratio`: Ratio of training data to use for evaluation if eval_file not provided
- `lazy_preprocess`: Whether to use lazy data loading

## Training Features

### Loss Components
1. **Standard Cross-Entropy**: Standard language modeling loss
2. **Plan Similarity**: KL divergence between hidden-state and plan-based predictions
3. **Random Contrast**: JS divergence between positive and negative hidden states
4. **InfoNCE Probe**: Optional contrastive probing for hidden state quality

### Monitoring
- **Loss Curves**: Automatic plotting of training loss
- **Component Tracking**: Individual loss component monitoring
- **Probe Metrics**: InfoNCE loss and lower bound estimates
- **Validation**: Regular evaluation on held-out data

### Checkpointing
- **Automatic Saving**: Saves model at specified intervals
- **Best Model**: Keeps track of best model based on evaluation loss
- **Limited Storage**: Configurable limit on saved checkpoints
- **State Saving**: Saves optimizer and scheduler states

## Output Structure

After training, the output directory will contain:
```
checkpoints/actor_model/
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── tokenizer/               # Tokenizer files
├── latent_comm_head.pt      # Custom latent communication head
├── loss_log.csv            # Training loss log
├── loss_curve.png          # Training loss plot
├── trainer_state.json      # Training state
└── checkpoint-*/           # Intermediate checkpoints
```

## Best Practices

### Data Preparation
1. **Balanced Data**: Ensure balanced representation of different conversation types
2. **Quality Plans**: Provide high-quality plan annotations when available
3. **Hidden States**: Pre-compute hidden states for better training efficiency
4. **Data Validation**: Validate JSON format before training

### Training Configuration
1. **Learning Rate**: Start with 2e-5 and adjust based on model size
2. **Batch Size**: Use largest batch size that fits in memory
3. **Gradient Accumulation**: Use for effective larger batch sizes
4. **Warmup**: Use warmup for stable training

### Monitoring
1. **Loss Components**: Monitor individual loss components for balance
2. **Validation**: Regular validation to prevent overfitting
3. **Probe Metrics**: Use InfoNCE probe to assess hidden state quality
4. **Checkpointing**: Save frequent checkpoints for recovery

## Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Enable mixed precision and optimize data loading
3. **Poor Convergence**: Adjust learning rate or loss component weights
4. **Data Loading**: Ensure proper file paths and formats

### Performance Tips
1. **Mixed Precision**: Automatically enabled based on GPU capabilities
2. **Data Loading**: Use multiple workers for faster data loading
3. **Caching**: Enable caching for repeated data processing
4. **Distributed Training**: Use multiple GPUs for faster training

## Dependencies

The training script requires:
- PyTorch >= 1.12
- Transformers >= 4.30
- Datasets >= 2.0
- NumPy >= 1.21
- Matplotlib >= 3.5
- CUDA (recommended for training)

## Related Files

- `train_actor.py`: Main training script
- `train_actor_readme.md`: This documentation
- `../reasoning_model/train_reasoning.py`: Companion reasoning model training
- `../../requirements.txt`: Project dependencies