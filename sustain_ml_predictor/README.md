```text
├── unet_models/                                    # Multiple U-Net ONNX model checkpoints
├── xcu19eg-ffvb1517-2-i/                           # FPGA-specific folder
    └── predictor_model_energy_dynamic.onnx         # ONNX model for dynamic energy prediction
    └── predictor_model_energy_board_runtime.onnx   # ONNX model for board runtime energy prediction
    └── predictor_model_latency.onnx                # ONNX model for latency prediction
    └── measurements.jsonl                          # power/latency/energy measurements (Ground truth)   
├── unet_models_info.jsonl                          # Metadata for each U-Net model (params, FLOPs, etc.)
├── unet_models_stats.json                          # Summary statistics across all U-Net models
├── demo.py                                         # Python script for demonstrating prediction model performance
├── predictor.py                                    # Python script for predicting power/latency of an ONNX model
```

**Run demo:**

```bash
python3 demo.py
```

**Run predictor:**

Total energy of the complete board during inference: 

```bash
python3 predictor.py --model_file unet_models/unet_model_122.onnx --metric energy_board_runtime
```

Dynamic energy = total energy - idle energy:

```bash
python3 predictor.py --model_file unet_models/unet_model_012.onnx --metric energy_dynamic
```

Latency per a single input

```bash
python3 predictor.py --model_file unet_models/unet_model_012.onnx --metric latency
```