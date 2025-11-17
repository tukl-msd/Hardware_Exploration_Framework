import argparse
import os
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from predictor import predict

def test_predictor(device, models_dir, models_stats_file, prediction_model_file, measurements_file, metric):
    
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".onnx")]
    model_files.sort(key=lambda x: os.path.basename(x))
    
    measurements = []
    with open(measurements_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            measurements.append(data)
            
    all_preds = []
    all_targets = []
    for idx, model_file in enumerate(tqdm(model_files)):    
        pred = predict(model_file, models_stats_file, prediction_model_file)
        all_preds.append(pred)
        
        if measurements[idx]["model_file"] == os.path.basename(model_file):
            target = measurements[idx][metric]
            all_targets.append(target)
        else:
            raise Exception(f"Inconsistency between the model name and the measurement: {os.path.basename(model_file)} vs. {measurements[idx]['model_file']}")        
    
    spearman = spearmanr(all_preds, all_targets).correlation
    tau, p_value = kendalltau(all_preds, all_targets)

    # Convert to numpy for plotting
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.6, edgecolor='b', label='Predictions')
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Ideal')  # y=x line
    plt.xlabel(f'True {metric.capitalize()}')
    plt.ylabel(f'Predicted {metric.capitalize()}')
    plt.title(f'{metric.capitalize()} Prediction vs. Ground Truth, \n Test size: {len(model_files)}, Spearman: {spearman:.4f}, KTau: {tau:.4f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'unet_{metric}_{device}.png')  
    
    print(f"Test Spearman: {spearman:.4f}")
    print(f"Test KTau: {tau:.4f}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch, Onnx and HiL (onnx, tflite, armnn) MNIST Example')
    parser.add_argument('--device', type=str, default="xczu19eg-ffvb1517-2-i", help='Path to a directory containing FPGA-dependant data')
    parser.add_argument('--models_dir', type=str, default="unet_models", help='Path to a directory containing U-Net models in onnx format')
    parser.add_argument('--models_stats_file', type=str, default="unet_models_stats.json", help='Path to a file containing statistics of the U-Net models')
    args = parser.parse_args()
      
    predictors = {
        "xczu19eg-ffvb1517-2-i": {
            "energy_dynamic": "predictor_model_energy_dynamic.onnx",
            "energy_board_runtime": "predictor_model_energy_board_runtime.onnx",
            "latency": "predictor_model_latency.onnx"
            }
    }    
    
    for metric, prediction_model_file in predictors[args.device].items():
        print(f"Predict {metric} for {args.device}:")
        prediction_model_file = os.path.join(args.device, predictors[args.device][metric])
        measurements_file = os.path.join(args.device, "measurements.jsonl")
        test_predictor(args.device, args.models_dir, args.models_stats_file, prediction_model_file, measurements_file, metric)



