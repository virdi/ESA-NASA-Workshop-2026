# Fine-Tuning Configuration Parameters Documentation

This document explains all parameters available in the fine-tuning configuration for segmentation models.

## Configuration Structure

```json
{
    "name": "burn-scars-demo-testuser10",
    "description": "Segmentation model for wildfire burn scar detection",
    "dataset_id": "geodata-qvsarslzndxpqmujlbhxx3",
    "base_model_id": "b6ba3db5-d41c-4119-864e-2af2cdbacfae",
    "tune_template_id": "9967afa5-bf2d-45be-9e58-0bb6dee27779",
    "model_parameters": { ... }
}
```

---

## Data Loading Configuration (`data`)

Controls how training data is loaded and processed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 4 | Number of samples per batch during training. Larger values use more memory but may improve training stability. |
| `constant_multiply` | float | 1 | Constant scale factor applied to data values. |
| `workers_per_gpu` | int | 2 | Number of parallel data loading workers per GPU. More workers can speed up data loading but use more CPU/memory. |

**Example:**
```json
"data": {
    "batch_size": 4,
    "constant_multiply": 1,
    "workers_per_gpu": 2
}
```

---

## Model Architecture Configuration (`model`)

Defines the neural network architecture for segmentation.

### Decode Head (`decode_head`)

The main decoder that produces the final segmentation output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `channels` | int | 256 | Number of channels at each block of the decode head (except the final output layer). |
| `num_convs` | int | 4 | Number of convolutional blocks in the head (except the final one). More blocks = deeper network. |
| `decoder` | string | "UNetDecoder" | Decoder architecture type. Options: "UperNetDecoder", "UNetDecoder" |
| `loss_decode.type` | string | "CrossEntropyLoss" | Loss function type for training. |
| `loss_decode.avg_non_ignore` | bool | true | If true, loss is only averaged over non-ignored targets (useful when labels have missing/invalid values). |

**Example:**
```json
"decode_head": {
    "channels": 256,
    "num_convs": 4,
    "decoder": "UNetDecoder",
    "loss_decode": {
        "type": "CrossEntropyLoss",
        "avg_non_ignore": true
    }
}
```

### Auxiliary Head (`auxiliary_head`)

Optional secondary decoder for additional supervision during training (improves learning).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decoder` | string | "FCNDecoder" | Decoder function for auxiliary head. |
| `channels` | int | 256 | Number of channels at each block (except final). |
| `num_convs` | int | 2 | Number of convolutional blocks. |
| `in_index` | int | -1 | Index of the input feature list to use (-1 means last/deepest features). |
| `dropout` | int | 0 | Dropout rate (0-1). 0 means no dropout. Helps prevent overfitting. |
| `loss_decode.type` | string | "CrossEntropyLoss" | Loss function type. |
| `loss_decode.loss_weight` | float | 1 | Weight of auxiliary loss in total loss. Total loss = aux_weight × aux_loss + main_loss |
| `loss_decode.avg_non_ignore` | bool | true | Average loss only over non-ignored targets. |

**Example:**
```json
"auxiliary_head": {
    "decoder": "FCNDecoder",
    "channels": 256,
    "num_convs": 2,
    "in_index": -1,
    "dropout": 0,
    "loss_decode": {
        "type": "CrossEntropyLoss",
        "loss_weight": 1,
        "avg_non_ignore": true
    }
}
```

### Backbone Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frozen_backbone` | bool | false | If true, freezes the backbone weights during training (only trains the decoder heads). Useful for transfer learning with limited data. |
| `backbone_img_size` | int | 512 | Input image size expected by the backbone model (e.g., Clay model uses 512). |

### Tiled Inference Parameters (`tiled_inference_parameters`)

For processing large images by splitting them into smaller tiles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h_crop` | int | 224 | Height of each tile crop in pixels. |
| `h_stride` | int | 196 | Vertical stride between tiles in pixels. Smaller stride = more overlap. |
| `w_crop` | int | 224 | Width of each tile crop in pixels. |
| `w_stride` | int | 196 | Horizontal stride between tiles in pixels. |
| `average_patches` | bool | false | If true, averages predictions in overlapping regions. If false, uses the last prediction. |

**Example:**
```json
"tiled_inference_parameters": {
    "h_crop": 224,
    "h_stride": 196,
    "w_crop": 224,
    "w_stride": 196,
    "average_patches": false
}
```

---

## Training Runner Configuration (`runner`)

Controls the training process and stopping criteria.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | int | 10 | Maximum number of training epochs (full passes through the dataset). |
| `early_stopping_patience` | int | 20 | Number of epochs to wait for improvement before stopping early. |
| `early_stopping_monitor` | string | "val/loss" | Metric to monitor for early stopping. Common values: "val/loss", "val/accuracy" |

**Example:**
```json
"runner": {
    "max_epochs": 1,
    "early_stopping_patience": 20,
    "early_stopping_monitor": "val/loss"
}
```

---

## Learning Rate Policy Configuration (`lr_config`)

Controls how the learning rate changes during training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy` | string | "Fixed" | Learning rate schedule policy. Options: "Fixed" (constant), "CosineAnnealing" (gradually decreases) |
| `warmup_iters` | int | 0 | Number of warmup iterations where LR gradually increases from warmup_ratio × lr to lr. |
| `warmup_ratio` | float | 1 | Initial learning rate during warmup = lr × warmup_ratio. |

**Example:**
```json
"lr_config": {
    "policy": "Fixed",
    "warmup_iters": 0,
    "warmup_ratio": 1
}
```

**CosineAnnealing Example:**
```json
"lr_config": {
    "policy": "CosineAnnealing",
    "warmup_iters": 500,
    "warmup_ratio": 0.1
}
```

---

## Optimizer Configuration (`optimizer`)

Controls the optimization algorithm and hyperparameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.00006 | Learning rate. Controls step size during gradient descent. Typical range: 1e-5 to 1e-3 |
| `type` | string | "Adam" | Optimizer algorithm. Options: "Adam", "SGD", "AdamW", "RMSProp" |
| `weight_decay` | float | 0 | L2 regularization weight (weight decay). Helps prevent overfitting. Typical range: 0 to 0.01 |

**Example:**
```json
"optimizer": {
    "lr": 0.00006,
    "type": "Adam",
    "weight_decay": 0
}
```

**Optimizer Comparison:**
- **Adam**: Adaptive learning rate, good default choice
- **AdamW**: Adam with better weight decay implementation
- **SGD**: Simple but may need learning rate scheduling
- **RMSProp**: Good for recurrent networks

---

## Validation Configuration (`evaluation`)

Controls validation during training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | int | 1 | Perform validation every N epochs. Set to 1 to validate after every epoch. |

**Example:**
```json
"evaluation": {
    "interval": 1
}
```

---

## Common Configuration Scenarios

### Quick Test Run (Fast Training)
```json
"model_parameters": {
    "runner": {
        "max_epochs": 1
    },
    "data": {
        "batch_size": 8
    }
}
```

### Production Training (High Quality)
```json
"model_parameters": {
    "runner": {
        "max_epochs": 50,
        "early_stopping_patience": 10
    },
    "optimizer": {
        "lr": 0.0001,
        "type": "AdamW",
        "weight_decay": 0.01
    },
    "lr_config": {
        "policy": "CosineAnnealing",
        "warmup_iters": 1000,
        "warmup_ratio": 0.1
    }
}
```

### Transfer Learning (Frozen Backbone)
```json
"model_parameters": {
    "model": {
        "frozen_backbone": true
    },
    "optimizer": {
        "lr": 0.001
    }
}
```

### Memory-Constrained Training
```json
"model_parameters": {
    "data": {
        "batch_size": 2,
        "workers_per_gpu": 1
    },
    "model": {
        "decode_head": {
            "channels": 128,
            "num_convs": 2
        }
    }
}