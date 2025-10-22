```text
├── unet_models/                        # Multiple U-Net ONNX model checkpoints
├── xcu19eg-ffvb1517-2-i/               # FPGA-specific folder
    └── predictor_model_latency.onnx    # ONNX model for latency prediction
    └── predictor_model_power.onnx      # ONNX model for power prediction
    └── measurements.jsonl              # power/latency measurements (Ground truth)   
├── unet_models_info.jsonl              # Metadata for each U-Net model (params, FLOPs, etc.)
├── unet_models_stats.json              # Summary statistics across all U-Net models
├── demo.py                             # Python script for demonstrating prediction model performance
├── predictor.py                        # Python script for predicting power/latency of an ONNX model
```

**Run demo:**

```bash
python3 demo.py
```

**Run predictor:**

```bash
python3 predictor.py --model_file unet_models/unet_model_012.onnx --metric latency
```

```bash
python3 predictor.py --model_file unet_models/unet_model_122.onnx --metric power
```