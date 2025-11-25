import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.stats import spearmanr
import yaml
import os
import copy
import matplotlib.pyplot as plt
import random
import json
import numpy as np
from onnx_utils import spec_iterator, extract_features_from_onnx
import tqdm
from torch.utils.data import random_split
from torch.nn import init
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import argparse

import torch.nn as nn
from scipy.stats import kendalltau

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()  # Convert arrays to lists
        return super().default(obj)

def set_seed(seed):
    random.seed(seed)                       # Python random
    np.random.seed(seed)                    # NumPy random
    torch.manual_seed(seed)                  # CPU
    torch.cuda.manual_seed(seed)             # GPU
    torch.cuda.manual_seed_all(seed)         # All GPUs (if multi-GPU)
    torch.backends.cudnn.deterministic = True  # Force deterministic
    torch.backends.cudnn.benchmark = False   # Disable benchmark

class HybridPredictor2Heads(nn.Module):
    def __init__(self, gcn_in_dim, mlp_in_dim, hidden_dim=100):
        super(HybridPredictor2Heads, self).__init__()

        self.gcn_branch = GCNLatencyPredictor(gcn_in_dim, hidden_dim, hidden_dim)
        self.mlp_branch = LatencyPredictor(mlp_in_dim, hidden_dim, hidden_dim)
        
        # fusion
        hidden_dim = 2 * hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.head1 = nn.Linear(hidden_dim, 1)
        self.head2 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, adj_mtrx, node_feats, arch_feats):
        gcn_out = self.gcn_branch(adj_mtrx, node_feats)        # (batch, hidden_dim)
        mlp_out = self.mlp_branch(arch_feats)             # (batch, hidden_dim)

        combined = torch.cat([gcn_out, mlp_out], dim=-1)  # concat along feature axis
        out = self.relu(self.fc1(combined))
        out = self.relu(self.fc2(out)) 
        out = self.relu(self.fc3(out)) 
        
        out1 = self.head1(out)                               # (batch, 1)
        out2 = self.head2(out)                               # (batch, 1)
        
        return (out1, out2)


class HybridPredictor(nn.Module):
    def __init__(self, gcn_in_dim, mlp_in_dim, hidden_dim=100):
        super(HybridPredictor, self).__init__()

        self.gcn_branch = GCNLatencyPredictor(gcn_in_dim, hidden_dim, hidden_dim)
        self.mlp_branch = LatencyPredictor(mlp_in_dim, hidden_dim, hidden_dim)
        
        # fusion
        hidden_dim = 2 * hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, adj_mtrx, node_feats, arch_feats):
        gcn_out = self.gcn_branch(adj_mtrx, node_feats)        # (batch, hidden_dim)
        mlp_out = self.mlp_branch(arch_feats)             # (batch, hidden_dim)

        combined = torch.cat([gcn_out, mlp_out], dim=-1)  # concat along feature axis
        out = self.relu(self.fc1(combined))
        out = self.relu(self.fc2(out)) 
        out = self.relu(self.fc3(out)) 
        out = self.fc4(out)                               # (batch, 1)
        return out.squeeze(-1)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_, adj, weight=None, bias=None):
        if weight is not None:
            support = torch.matmul(input_, weight)
            output = torch.bmm(adj, support)
            if bias is not None:
                return output+bias
            else:
                return output
            
        else:
            support = torch.matmul(input_, self.weight)
            output = torch.bmm(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNLatencyPredictor(nn.Module):
    """
    The base model for MAML (Meta-SGD) for meta-NAS-predictor.
    """

    def __init__(self, in_dim, hidden_dim=100, out_dim=1):
        super(GCNLatencyPredictor, self).__init__()
        self.layer_size = hidden_dim

        for i in range(1, 5):
            if i == 1:
                input_dim = in_dim
            else:
                input_dim = hidden_dim
            self.add_module(f'gc{i}', GraphConvolution(input_dim, hidden_dim))        


        self.add_module('fc3', nn.Linear(hidden_dim, hidden_dim))
        self.add_module('fc4', nn.Linear(hidden_dim, hidden_dim))
        self.add_module('fc5', nn.Linear(hidden_dim, out_dim))
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, adj, feat):   

        '''out = self.relu(self.gc1(feat, adj).transpose(2,1))
        out = out.transpose(1, 2)
        out = self.relu(self.gc2(out, adj).transpose(2,1))
        out = out.transpose(1, 2)
        out = self.relu(self.gc3(out, adj).transpose(2,1))
        out = out.transpose(1, 2)
        out = self.relu(self.gc4(out, adj).transpose(2,1))
        out = out.transpose(1, 2)
        #out = out[:, out.size()[1] - 1, :]
        out = out[:, 0, :]
        #out = out.mean(dim=1) 

        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)
        
        out = out.squeeze(-1)
        
        return out'''
        
        out = self.relu(self.gc1(feat, adj))
        out = self.relu(self.gc2(out, adj))
        out = self.relu(self.gc3(out, adj))
        out = self.relu(self.gc4(out, adj))

        # Readout: mean over nodes
        out = out.mean(dim=1)  # [batch, layer_size]

        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return out.squeeze(-1)

def remove_outliers(data, labels, portion=0.05):
    # Combine data and labels to maintain correspondence
    combined = list(zip(data, labels))

    # Sort by label
    combined.sort(key=lambda x: x[1])

    # Remove N lowest and M highest label entries
    trimmed = combined[int(len(combined)*portion):int(len(combined)-len(combined)*portion)]

    # Unzip back into two lists
    new_data, new_labels = zip(*trimmed) if trimmed else ([], [])

    return list(new_data), list(new_labels)

def remove_outliers_mlp_gcn(adj, feat, data, labels, portion=0.05):
    # Combine data and labels to maintain correspondence
    combined = list(zip(adj, feat, data, labels))

    # Sort by label
    combined.sort(key=lambda x: x[3])

    # Remove N lowest and M highest label entries
    trimmed = combined[int(len(combined)*portion):int(len(combined)-len(combined)*portion)]

    # Unzip back into two lists
    new_adj, new_feat, new_data, new_labels = zip(*trimmed) if trimmed else ([], [], [], [])

    return list(new_adj), list(new_feat), list(new_data), list(new_labels)

def one_hot_encoded(value, options):
    return [int(value == op) for op in options]

class AttentionLatencyPredictor(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, num_heads=8, ff_hidden=100):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, 1)
        )

    def forward(self, x):
        x = self.embedding(x) # Applies a linear projection from input_dim → embed_dim. Converts raw features into a richer space
        attn_output, _ = self.attn(x, x, x) # Since query, key, and value are all x, this is self-attention
        x = self.norm(attn_output + x) # Adds residual connection (attn_output + x). Then normalizes across embed_dim using LayerNorm
        pooled = x.mean(dim=1)# Takes the mean across the sequence - Converts sequence into a single fixed-size vector
        return self.mlp(pooled).squeeze(-1)

class SimpleClassifierDataset(Dataset):
    def __init__(self, topology, data_path, metric_path, key='latency', outlier_portion=0.05):    
        self.data = []
        self.labels = []

        list_flops = []
        list_params = []
        with open(metric_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                flops = np.log1p(data['ptflops_macs'])
                list_flops.append(flops)
                params = np.log1p(data['params'])
                list_params.append(params)
        list_flops = np.array(list_flops)    
        mean_flops = np.mean(list_flops)
        std_flops = np.std(list_flops)
        list_params = np.array(list_params)
        mean_params = np.mean(list_params)
        std_params = np.std(list_params) 
        
        if topology == "ofanet":
            self.data = [arch_encoding_ofa(arch) for arch in torch.load(data_path)['arch']]
        elif topology == "unet":
            self.data = [arch_encoding_unet(arch, mean_flops, std_flops, mean_params, std_params) for arch in load_jsonl(data_path)]          

        if topology == "ofanet":
            self.labels = torch.load(metric_path)
        elif topology == "unet":
            self.labels = [extract_key(arch, key) for arch in load_jsonl(metric_path)]
            
        self.data, self. labels = remove_outliers(self.data, self. labels, outlier_portion)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert list of lists to tensor
        x = torch.tensor(self.data[idx], dtype=torch.float32)     # shape: [seq_len, feature_dim]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)   # shape: []
        y = y.unsqueeze(0)
        return x, y

class AttentionEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, metrics_path, key='latency'):
        data = []
        self.labels = []
        
        max_seq_len = 0
        max_flops_per_layer = 0
        max_params_per_layer = 0
        op_id_options = [0]
        
        # metrics_path - .json file with latency and power consumption
        # embeddings_path - .json with embeddings. Embedding file has more entries than the metrics file.
        for arch in load_jsonl(metrics_path):            
            for embd in load_jsonl(embeddings_path):
                if embd['idx'] == arch['idx']:
                    self.labels.append(arch[key])
                    data.append(embd['features'])
                    max_seq_len = len(embd['features']) if len(embd['features']) > max_seq_len else max_seq_len 
                    for node in embd['features']:
                        max_flops_per_layer = node['flops'] if node['flops'] > max_flops_per_layer else max_flops_per_layer
                        max_params_per_layer = node['params'] if node['params'] > max_params_per_layer else max_params_per_layer
                        if not node['op_id'] in op_id_options:
                            op_id_options.append(node['op_id'] )
                    break
        
        for sample in data:
            for feature in sample:
                feature['params'] = feature['params'] / max_params_per_layer
                feature['flops'] = feature['flops']/ max_flops_per_layer
                #feature['op_id'] = one_hot_encoded(feature['op_id'], op_id_options)

        # Convert dict to list
        keys = ["op_id", "flops", "params"]
        self.data = [[[d[k] for k in keys] for d in sample] for sample in data]
        
        for s_idx, sample in enumerate(self.data):
            for v_idx, vec in enumerate(sample):
                temp = one_hot_encoded(vec[0], op_id_options)
                for feature in vec[1:]:
                    temp.append(feature) 
                self.data[s_idx][v_idx] = temp
        
        # Padding
        for sample in self.data:
            while len(sample) < max_seq_len:
                temp = one_hot_encoded(0, op_id_options)
                temp.append(0)
                temp.append(0)
                sample.append(temp)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert list of lists to tensor
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

class GCNEmbeddingDataset(Dataset):
    def __init__(self, arch_path, metrics_path, key='latency', outlier_portio=0.0):
        self.adj_matrix = []
        self.node_features = []
        self.labels = []
               
        # metrics_path - .json file with latency and power consumption
        # arch_path - .json with arch embeddings. Embedding file has more entries than the metrics file.
        for dev in load_jsonl(metrics_path):            
            for arch in load_jsonl(arch_path):
                if dev['idx'] == arch['idx']:
                    self.labels.append(dev[key])
                    self.adj_matrix.append(arch['adj_matrix'])
                    self.node_features.append(arch['node_features'])                    
                    break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert list of lists to tensor
        return (torch.tensor(self.adj_matrix[idx], dtype=torch.float32), torch.tensor(self.node_features[idx], dtype=torch.float32)), torch.tensor(self.labels[idx], dtype=torch.float32)

# Combining dataset for GCN and Simple
class HybridClassifierDataset(Dataset):
    def __init__(self, arch_path, metrics_path, metric=["latency"], outlier_portion=0.05, predict_normalized="no_norm"):    
        self.data = []
        self.labels1 = []            
        self.adj_matrix = []
        self.node_features = []
        
        # metrics_path - .json file with latency and power consumption
        # arch_path - .json with arch embeddings. Embedding file has more entries than the metrics file.
        for dev in load_jsonl(metrics_path):            
            for arch in load_jsonl(arch_path):
                if dev['idx'] == arch['idx']:
                    self.data.append(arch_encoding_unet(dev))
                    
                    if metric[0] == "dynamic_power":
                        power_dynamic = dev['power_runtime'] - dev['power_idle']
                        self.labels1.append(power_dynamic)
                    elif metric[0] == "dynamic_energy":
                        power_dynamic = dev['power_runtime'] - dev['power_idle']
                        energy = power_dynamic * dev['latency']
                        self.labels1.append(energy)
                    elif metric[0] == "total_energy":
                        energy = dev['power_runtime'] * dev['latency']
                        self.labels1.append(energy)
                    elif metric[0] == "brams":
                        brams = dev['RAMB36'] + dev['RAMB18'] * 0.5
                        self.labels1.append(brams)
                    else:
                        self.labels1.append(dev[metric[0]])

                    self.adj_matrix.append(arch['adj_matrix'])
                    self.node_features.append(arch['node_features'])                    
                    break
        # no_norm, log1p_11, log1p_mean_std, norm_11, mean_std
        if predict_normalized in ["log1p_11", "log1p_mean_std"]:
            labels1 = np.log1p(np.array(self.labels1))
            self.minimum = labels1.min()
            self.maximum = labels1.max()
            self.mean = labels1.mean()
            self.std = labels1.std()
        elif predict_normalized in ["norm_11", "mean_std"]:
            labels1 = np.array(self.labels1)
            self.minimum = labels1.min()
            self.maximum = labels1.max()
            self.mean = labels1.mean()
            self.std = labels1.std()
        else:
            self.minimum = None
            self.maximum = None
            self.mean = None
            self.std = None            

        self.adj_matrix, self.node_features, self.data, self.labels1 = remove_outliers_mlp_gcn(self.adj_matrix, self.node_features, self.data, self.labels1, outlier_portion)
    
    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):
        # Convert list of lists to tensor
        adj = torch.tensor(self.adj_matrix[idx], dtype=torch.float32)
        feat = torch.tensor(self.node_features[idx], dtype=torch.float32)
        arch = torch.tensor(self.data[idx], dtype=torch.float32)     # shape: [seq_len, feature_dim]
        label1 = torch.tensor(self.labels1[idx], dtype=torch.float32)   # shape: []
        return (adj, feat, arch), label1

def load_jsonl(input_path):
    # Read and extract latency values
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)            
            #if 'xczu19eg' in data['part_num']:
            #if 'xczu7ev' in data['part_num']:
            yield data

def extract_key(data, key):
    return data[key]

def arch_encoding_unet(data, mean_flops=None, std_flops=None, mean_params=None, std_params=None):
    
    depth = data['depth']
    initial_channels = data['initial_channels']
    input_size = data['input_size']
    kernel_sizes = data['kernel_sizes']

    if mean_flops is not None:
        flops = [(np.log1p(data['ptflops_macs']) - mean_flops) / std_flops]
        params = [(np.log1p(data['params']) - mean_params) / std_params]
    
    '''print(depth)
    print(initial_channels)
    print(input_size)
    print(kernel_sizes) '''
    
    # Define possible values
    depth_options = [1, 2, 3, 4]
    channels_options = [8, 16, 32, 64]
    input_size_options = [64, 128, 256, 512]

    # One-hot encodings
    depth_vec = [int(depth == d) for d in depth_options]
    channels_vec = [int(initial_channels == ch) for ch in channels_options]
    input_size_vec = [int(input_size == sz) for sz in input_size_options]   
    

    # Kernel encoding (each of 5 slots encoded as 2-bit one-hot)
    kernel_encoding = []
    padded_kernels = kernel_sizes[:5] + [0] * (5 - len(kernel_sizes))
    for ks in padded_kernels:
        if ks == 3:
            kernel_encoding += [1, 0]
        elif ks == 5:
            kernel_encoding += [0, 1]
        else:
            kernel_encoding += [0, 0]  # padding
    
    '''print(depth_vec)
    print(channels_vec)
    print(input_size_vec)
    print(kernel_encoding)
    
    input()'''

    #return torch.Tensor(depth_vec + channels_vec + input_size_vec + kernel_encoding)
    if mean_flops is not None:
        return depth_vec + channels_vec + input_size_vec + kernel_encoding + flops + params
    else:
        return depth_vec + channels_vec + input_size_vec + kernel_encoding

def arch_encoding_ofa(arch):
    # This function converts a network config to a feature vector (128-D).
    ks_list, ex_list, d_list, r = copy.deepcopy(arch['ks']), copy.deepcopy(arch['e']), copy.deepcopy(arch['d']), arch['r']
    
    '''print(ks_list)
    print(ex_list)
    print(d_list)
    print(r)'''
    
    ks_map = {}
    ks_map[3]=0
    ks_map[5]=1
    ks_map[7]=2
    ex_map = {}
    ex_map[3]=0
    ex_map[4]=1
    ex_map[6]=2
    
    
    start = 0
    end = 4
    for d in d_list:
        for j in range(start+d, end):
            ks_list[j] = 0
            ex_list[j] = 0
        start += 4
        end += 4

    # convert to onehot
    ks_onehot = [0 for _ in range(60)] # 20 values, 3 bits each = 60 one-hot-encodded
    ex_onehot = [0 for _ in range(60)] # 20 values, 3 bits each = 60 one-hot-encodded
    r_onehot = [0 for _ in range(25)] #128 ~ 224 # 25 values of different dimentions of images

    for i in range(20):
        start = i * 3
        if ks_list[i] != 0:
            ks_onehot[start + ks_map[ks_list[i]]] = 1
        if ex_list[i] != 0:
            ex_onehot[start + ex_map[ex_list[i]]] = 1

    r_onehot[(r - 128) // 4] = 1
    
    '''print(ks_onehot)
    print(ex_onehot)
    print(r_onehot)
    print(ks_onehot + ex_onehot + r_onehot)
    input()'''
    
    #return torch.Tensor(ks_onehot + ex_onehot + r_onehot)
    return ks_onehot + ex_onehot + r_onehot


def normalization(latency, index=None, portion=1.0):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency

def denormalization(preds, predict_normalized, gt_min, gt_max, gt_mean, gt_std):
    #denorm_preds = preds * (gt_max - gt_min) + gt_min    
    
    if predict_normalized == "log1p_11":
        denorm_preds = ((preds + 1.0) / 2.0) * (gt_max - gt_min) + gt_min
        denorm_preds = np.expm1(denorm_preds.cpu())
    elif predict_normalized == "norm_11":
        denorm_preds = ((preds + 1.0) / 2.0) * (gt_max - gt_min) + gt_min
    elif predict_normalized == "log1p_mean_std":
        denorm_preds = preds * gt_std + gt_mean
        denorm_preds = np.expm1(denorm_preds.cpu())
    elif predict_normalized == "mean_std":
        denorm_preds = preds * gt_std + gt_mean
    else:
        denorm_preds = preds
                        
    return denorm_preds

class LatencyPredictor(nn.Module):

    def __init__(self, in_dim=145, hidden_dim=100, out_dim=1, hw_embed_on=False, hw_embed_dim=10):
        super(LatencyPredictor, self).__init__()
        self.layer_size = hidden_dim
        self.hw_embed_on = hw_embed_on

        self.add_module('fc1', nn.Linear(in_dim, hidden_dim))
        self.add_module('fc2', nn.Linear(hidden_dim, hidden_dim))

        if hw_embed_on:
            self.add_module('fc_hw1', nn.Linear(hw_embed_dim, hidden_dim))
            self.add_module('fc_hw2', nn.Linear(hidden_dim, hidden_dim))
            hidden_dim = hidden_dim * 2 

        self.add_module('fc3', nn.Linear(hidden_dim, hidden_dim))
        self.add_module('fc4', nn.Linear(hidden_dim, hidden_dim))

        self.add_module('fc5', nn.Linear(hidden_dim, out_dim))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hw_embed=None, params=None):           
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))

        if self.hw_embed_on:
            hw_embed = hw_embed.repeat(len(x), 1)
            hw = self.relu(self.fc_hw1(hw_embed))
            hw = self.relu(self.fc_hw2(hw))
            out = torch.cat([out, hw], dim=-1)

        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return out


# Train function
def train_on_single_hw(method, device, model, train_dataloader, val_dataloader, test_dataloader, gt_min, gt_max, gt_mean, gt_std, metrics=["latency"], epochs=10, predict_normalized="no_norm", lr=1e-3):
    
    # Loss & Optimizer
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)


    best_spearman = -1
    best_model = None

    for epoch in range(epochs):
    #for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_dataloader:  
            if isinstance(x_batch, (list, tuple)):
                if len(x_batch) == 3:
                    adj_batch, node_feat_batch, arch_feat_batch = x_batch
                    adj_batch = adj_batch.cuda()
                    node_feat_batch = node_feat_batch.cuda()
                    arch_feat_batch = arch_feat_batch.cuda()
                    #y_batch = y_batch.cuda()                    
                    optimizer.zero_grad()
                    preds = model(adj_batch, node_feat_batch, arch_feat_batch)
                else:
                    adj_batch, node_feat_batch = x_batch
                    adj_batch = adj_batch.cuda()
                    node_feat_batch = node_feat_batch.cuda()
                    #y_batch = y_batch.cuda()
                    optimizer.zero_grad()
                    preds = model(adj_batch, node_feat_batch)
            else:
                x_batch = x_batch.cuda()
                #y_batch = y_batch.cuda()
                optimizer.zero_grad()
                preds = model(x_batch)
            
             # Move target(s)
            if isinstance(y_batch, (list, tuple)):
                y_batch = [y.cuda() for y in y_batch]
                loss = sum(criterion(p, y)*(1/len(preds)) for p, y in zip(preds, y_batch))
            else:  
                #no_norm, log1p_11, log1p_mean_std, norm_11, mean_std
                if predict_normalized == "log1p_11":
                    y_batch_log = np.log1p(y_batch)
                    y_batch_norm = 2.0*(y_batch_log - gt_min)/(gt_max - gt_min)-1
                    y_batch_norm = y_batch_norm.cuda()
                    loss = criterion(preds, y_batch_norm)
                elif predict_normalized == "norm_11":
                    y_batch_norm = 2.0*(y_batch - gt_min)/(gt_max - gt_min)-1
                    y_batch_norm = y_batch_norm.cuda()
                    loss = criterion(preds, y_batch_norm)
                elif predict_normalized == "log1p_mean_std":
                    y_batch_log = np.log1p(y_batch)
                    y_batch_norm = (y_batch_log - gt_mean)/gt_std
                    y_batch_norm = y_batch_norm.cuda()
                    loss = criterion(preds, y_batch_norm)
                elif predict_normalized == "mean_std":
                    y_batch_norm = (y_batch - gt_mean)/gt_std
                    y_batch_norm = y_batch_norm.cuda()
                    loss = criterion(preds, y_batch_norm)
                else:
                    y_batch = y_batch.cuda()
                    loss = criterion(preds, y_batch)         
            
            loss.backward()
            optimizer.step()
            # Accumulate loss
            if isinstance(y_batch, (list, tuple)):
                batch_size = y_batch[0].size(0)
            else:
                batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
        train_loss = total_loss / len(train_dataloader.dataset)

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                if isinstance(x_batch, (list, tuple)):
                    if len(x_batch) == 3:
                        adj_batch, node_feat_batch, arch_feat_batch = x_batch
                        adj_batch = adj_batch.cuda()
                        node_feat_batch = node_feat_batch.cuda()
                        arch_feat_batch = arch_feat_batch.cuda()
                        #y_batch = y_batch.cuda()
                        optimizer.zero_grad()
                        preds = model(adj_batch, node_feat_batch, arch_feat_batch)
                    else:
                        adj_batch, node_feat_batch = x_batch
                        adj_batch = adj_batch.cuda()
                        node_feat_batch = node_feat_batch.cuda()
                        #y_batch = y_batch.cuda()
                        preds = model(adj_batch, node_feat_batch)
                else:
                    x_batch = x_batch.cuda()
                    #y_batch = y_batch.cuda()
                    preds = model(x_batch)   
                    
                # Move target(s)
                if isinstance(y_batch, (list, tuple)):
                    y_batch = [y.cuda() for y in y_batch]
                    loss = sum(criterion(p, y)*(1/len(preds)) for p, y in zip(preds, y_batch))
                    batch_size = y_batch[0].size(0)
                else:
                    #no_norm, log1p_11, log1p_mean_std, norm_11, mean_std
                    if predict_normalized == "log1p_11":
                        y_batch_log = np.log1p(y_batch)
                        y_batch_norm = 2.0*(y_batch_log - gt_min)/(gt_max - gt_min)-1
                        y_batch_norm = y_batch_norm.cuda()
                        loss = criterion(preds, y_batch_norm)
                    elif predict_normalized == "norm_11":
                        y_batch_norm = 2.0*(y_batch - gt_min)/(gt_max - gt_min)-1
                        y_batch_norm = y_batch_norm.cuda()
                        loss = criterion(preds, y_batch_norm)
                    elif predict_normalized == "log1p_mean_std":
                        y_batch_log = np.log1p(y_batch)
                        y_batch_norm = (y_batch_log - gt_mean)/gt_std
                        y_batch_norm = y_batch_norm.cuda()
                        loss = criterion(preds, y_batch_norm)
                    elif predict_normalized == "mean_std":
                        y_batch_norm = (y_batch - gt_mean)/gt_std
                        y_batch_norm = y_batch_norm.cuda()
                        loss = criterion(preds, y_batch_norm)
                    else:
                        y_batch = y_batch.cuda()
                        loss = criterion(preds, y_batch)
                    batch_size = y_batch.size(0)
                    
                #loss = criterion(preds, y_batch)      
                #total_loss += loss.item() * y_batch.size(0)   
                total_loss += loss.item() * batch_size   
                if predict_normalized == "no_norm":
                    pass
                else:
                    preds = denormalization(preds, predict_normalized, gt_min, gt_max, gt_mean, gt_std)
                all_preds.append(preds)
                all_targets.append(y_batch)        
        
        val_loss = total_loss / len(val_dataloader.dataset)
        #scheduler.step(val_loss)
        scheduler.step()
        all_preds = torch.cat(all_preds).cpu()
        all_targets = torch.cat(all_targets).cpu()
        spearman = spearmanr(all_preds.numpy(), all_targets.numpy()).correlation

        if spearman > best_spearman:
            best_spearman = spearman
            #best_model = model.state_dict()
            torch.save(model.state_dict(), 'best_model_weights.pth')

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Spearman={spearman:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")  
    
    print(f"Best Validation Spearman: {best_spearman:.4f}")
    #model.load_state_dict(best_model)   
    model.load_state_dict(torch.load('best_model_weights.pth'))
    
    # Test
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(test_dataloader):
            if isinstance(x_batch, (list, tuple)):
                if len(x_batch) == 3:
                    adj_batch, node_feat_batch, arch_feat_batch = x_batch
                    adj_batch = adj_batch.cuda()
                    node_feat_batch = node_feat_batch.cuda()
                    arch_feat_batch = arch_feat_batch.cuda()
                    y_batch = y_batch.cuda()
                    optimizer.zero_grad()
                    preds = model(adj_batch, node_feat_batch, arch_feat_batch)
                else:
                    adj_batch, node_feat_batch = x_batch
                    adj_batch = adj_batch.cuda()
                    node_feat_batch = node_feat_batch.cuda()
                    y_batch = y_batch.cuda()
                    preds = model(adj_batch, node_feat_batch)
            else:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                preds = model(x_batch) 
            if predict_normalized == "no_norm":
                pass
            else:
                preds = denormalization(preds, predict_normalized, gt_min, gt_max, gt_mean, gt_std)      
            all_preds.append(preds)
            all_targets.append(y_batch)
    all_preds = torch.cat(all_preds).cpu()    
    all_targets = torch.cat(all_targets).cpu()
    spearman = spearmanr(all_preds.numpy(), all_targets.numpy()).correlation
    tau, p_value = kendalltau(all_preds.numpy(), all_targets.numpy())

        
    # Convert to numpy for plotting
    for metric in metrics:
        preds = all_preds.numpy()
        targets = all_targets.numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(targets, preds, alpha=0.6, edgecolor='b', label='Predictions')
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Ideal')  # y=x line
        plt.xlabel(f'True {metric.capitalize()}')
        plt.ylabel(f'Predicted {metric.capitalize()}')
        plt.title(f'{metric.capitalize()} Prediction vs Ground Truth, \n Train set: {len(train_dataloader.dataset)}, Val set: {len(test_dataloader.dataset)}, Spearman: {spearman:.4f}, KTau: {tau:.4f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'targets_vs_preds_{method}_{metric}_{device}_{predict_normalized}.png')   

    return model

def train_fc_predictor(device, topology="unet", metric="latency", train_ratio=0.8, epochs=100, seed=42, outlier_portion=0.05): #'power_runtime_extrnl'

    if topology == "ofanet":
        data_path = "/home/rybalkin/HELP/data/ofa"
        #device = "ofanet_rp5_armnn_int8_int8_16"
    elif topology == "unet":
        data_path = "/home/rybalkin/HELP/data/unet"
        #device = "unet_rp5_armnn_int8_int8_16"
    train_batch_size = 10
    val_batch_size = 10
    test_batch_size = 10
    predict_normalized = False
    
    if topology == "ofanet":
        metric_path = os.path.join(data_path, 'latency', f'{device}.pt')
        data_path = os.path.join(data_path, 'ofa_archs.pt')        
    elif topology == "unet":
        metric_path = os.path.join(data_path, f'{device}.jsonl')
        data_path = os.path.join(data_path, f'{device}.jsonl')
  
  
    dataset = SimpleClassifierDataset(topology, data_path, metric_path, metric, outlier_portion)   
    
    print("Dataset size: ", len(dataset))
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print("Dataset train size: ", train_size)
    print("Dataset val size: ", val_size)

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    labels = [label for _, label in train_dataset]
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    min_train_tensor =  torch.min(labels_tensor)
    max_train_tensor = torch.max(labels_tensor)
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    
    
    #===================================================================================
    
    '''if topology == "ofanet":
        metric_path = os.path.join(data_path, 'latency', f'{device}.pt')
        data_path = os.path.join(data_path, 'ofa_archs.pt')        
    elif topology == "unet":
        metric_path = os.path.join(data_path, f'{device}.jsonl')
        data_path = os.path.join(data_path, f'{device}.jsonl')
  
  
    dataset = SimpleClassifierDataset(topology, data_path, metric_path)
    labels = [label for _, label in dataset]
    archs = [arch.tolist() for arch, _ in dataset]
    
    nts = num_train_samples = int(len(archs) * train_ratio)
    nvs = num_val_samples = int(len(archs) - nts)       
    train_idx = sorted(random.sample(range(len(archs)), nts))  # sorted for consistency
    all_indices = set(range(len(archs)))
    val_idx = sorted(list(all_indices - set(train_idx)))
    
    x_train_tensor = torch.tensor([archs[i] for i in train_idx], dtype=torch.float32) # [nts, 145]
    x_val_tensor = torch.tensor([archs[i] for i in val_idx], dtype=torch.float32) # [nvs, 145]
    
    y_train_tensor = torch.tensor([labels[i] for i in train_idx], dtype=torch.float32) # [nts, 145]
    y_val_tensor = torch.tensor([labels[i] for i in val_idx], dtype=torch.float32) # [nvs, 145]
    
    y_train_tensor = y_train_tensor.view(-1,1)
    y_val_tensor = y_val_tensor.view(-1,1)       
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)   
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    
    min_train_tensor = 0.0
    max_train_tensor = 1.0'''
    
    #===================================================================================
    
    '''# List of archs
    if topology == "ofanet":
        archs = [torch.Tensor(arch_encoding_ofa(arch)) for arch in torch.load(os.path.join(data_path, 'ofa_archs.pt'))['arch']]
    elif topology == "unet":
        archs = [arch_encoding_unet(arch) for arch in load_jsonl(os.path.join(data_path, f'{device}.jsonl'))]          
   
    nts = num_train_samples = int(len(archs) * train_ratio)
    nvs = num_val_samples = int(len(archs) - nts)   
    
    #train_idx = torch.arange(len(archs))[:nts] # from 0 to nts
    #val_idx = torch.arange(len(archs))[nts:nts+nvs] # from nts to 5000
    train_idx = sorted(random.sample(range(len(archs)), nts))  # sorted for consistency
    all_indices = set(range(len(archs)))
    val_idx = sorted(list(all_indices - set(train_idx)))
       
    x_train_tensor = torch.tensor([archs[i] for i in train_idx], dtype=torch.float32) # [nts, 145]
    x_val_tensor = torch.tensor([archs[i] for i in val_idx], dtype=torch.float32) # [nvs, 145]    

    # Tensot of latencies [5000, 1]
    if topology == "ofanet":
        latency = torch.FloatTensor(torch.load(os.path.join(data_path, 'latency', f'{device}.pt')))
    elif topology == "unet":
        latency = torch.FloatTensor([extract_key(arch, metric) for arch in load_jsonl(os.path.join(data_path, f'{device}.jsonl'))])
    norm_latency = normalization(latency=latency)
    
    y_train_tensor = latency[train_idx].view(-1, 1) # [nts, 1]
    min_train_tensor = min(y_train_tensor)
    max_train_tensor = max(y_train_tensor)
    if predict_normalized:
        y_train_tensor = norm_latency[train_idx].view(-1, 1) # [nts, 1]
    y_val_tensor = latency[val_idx].view(-1, 1) # [nvs, 1]
    
    print(x_train_tensor.shape)
    print(y_train_tensor.shape)
    print(x_val_tensor.shape)
    print(y_val_tensor.shape)
    input()
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)   
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)'''

    print("in_dim=", train_dataset[0][0].shape[0])
    
    model = LatencyPredictor(in_dim=train_dataset[0][0].shape[0])#in_dim=len(archs[0])
    trained_model = train_on_single_hw("mlp_plus_2", device, model, train_dataloader, val_dataloader, test_dataloader, min_train_tensor, max_train_tensor, metric, epochs, predict_normalized)

def train_attention_predictor(device, metric="latency", train_ratio=0.8, epochs=100, seed=42):
    
    data_path = "/home/rybalkin/HELP/data/unet"
    embedding = "unet_attention_embeddings"
    embeddings_path = os.path.join(data_path, f'{embedding}.jsonl')
    metrics_path = os.path.join(data_path, f'{device}.jsonl')
    train_batch_size = 10
    val_batch_size = 100
    min_train_tensor = 0.0
    max_train_tensor = 1.0
    predict_normalized = False
    
    print("Preparing dataset...")
    
    dataset = AttentionEmbeddingDataset(embeddings_path, metrics_path, metric)
    
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)     
    
    model = AttentionLatencyPredictor(input_dim=train_dataset[0][0].shape[1])
    trained_model = train_on_single_hw("attention", model, train_dataloader, val_dataloader, min_train_tensor, max_train_tensor, metric, epochs, predict_normalized)
    
    return model

def train_gcn_predictor(device, metric="latency", train_ratio=0.8, epochs=100, seed=42, outlier_portion=0.05): #'power_runtime_extrnl'

    data_path = "/home/rybalkin/HELP/data/unet"
    train_batch_size = 10
    val_batch_size = 10
    test_batch_size = 1

    metric_path = os.path.join(data_path, f'{device}.jsonl')
    arch_path = os.path.join(data_path, "unet_arch_gcn_updated_norm.jsonl")  
  
    dataset = GCNEmbeddingDataset(arch_path, metric_path, metric, outlier_portion)   
    
    print("Dataset size: ", len(dataset))
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print("Dataset train size: ", train_size)
    print("Dataset val size: ", val_size)

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("nfeat: ", train_dataset[0][0][1].shape[1])
    print("shape: ", train_dataset[0][0][0].shape) # adj matrix
    print("shape: ", train_dataset[0][0][1].shape) # features
    print("shape: ", train_dataset[0][1]) # target - latecny
    
    model = GCNLatencyPredictor(nfeat=train_dataset[0][0][1].shape[1])#nfeat=len(archs[0])
    trained_model = train_on_single_hw("gcn", device, model, train_dataloader, val_dataloader, test_dataloader, 0.0, 0.0, metric, epochs, lr=1e-3)
    


def train_hybrid_predictor(device, metric=["latency"], train_ratio=0.8, epochs=100, seed=42, outlier_portion=0.05, predict_normalized="no_norm"): #'power_runtime_extrnl'

    data_path = "/home/rybalkin/HELP/data/unet"
    train_batch_size = 10
    val_batch_size = 10
    test_batch_size = 1

    metric_path = os.path.join(data_path, f'{device}.jsonl')
    arch_path = os.path.join(data_path, "unet_arch_gcn_updated_norm.jsonl")  
  
    dataset = HybridClassifierDataset(arch_path, metric_path, metric, outlier_portion, predict_normalized)   
    
    #print("Dataset size: ", len(dataset))
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    #print("Dataset train size: ", train_size)
    #print("Dataset val size: ", val_size)

    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)


    '''print("shape: ", train_dataset[0][0][0].shape) # adj matrix
    print("shape: ", train_dataset[0][0][1].shape) # node features
    print("shape: ", train_dataset[0][0][2].shape) # arch features
    print("shape: ", train_dataset[0][1]) # target - latecny'''
    
        
    gcn_in_dim, mlp_in_dim = train_dataset[0][0][1].shape[1], train_dataset[0][0][2].shape[0]
    model = HybridPredictor(gcn_in_dim, mlp_in_dim)
    #model = HybridPredictor2Heads(gcn_in_dim, mlp_in_dim)
 
    '''if os.path.exists('best_model_weights.pth'):
        model.load_state_dict(torch.load('best_model_weights.pth'))
        print("Load model...")
    else:'''
    model = train_on_single_hw("hybrid", device, model, train_dataloader, val_dataloader, test_dataloader, dataset.minimum, dataset.maximum, dataset.mean, dataset.std, metric, epochs, predict_normalized=predict_normalized, lr=1e-3)
    
    return model

def pt_2_onnx(model, sample_input, onnx_model_path):
    import onnx
    device = torch.device("cpu")
    input_names = [f"input_{i}" for i in range(len(sample_input))]
    sample_inputs = tuple(inputs.to(device) for inputs in sample_input)
    torch.onnx.export(model, sample_inputs, onnx_model_path, training=torch.onnx.TrainingMode.EVAL, input_names=input_names, output_names=["output"], verbose=False)
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    
def test_complete_hybrid_predictor(model, device, metrics=["latency"], runtime="pt", predict_normalized="no_norm", spearman_max=-1, seed=1):
    data_path = "/home/rybalkin/HELP/data/unet"   
    metric_path = os.path.join(data_path, f'{device}.jsonl')
    arch_path = os.path.join(data_path, "unet_arch_gcn_updated_norm.jsonl") 
    
    batch_size = 1
  
    dataset = HybridClassifierDataset(arch_path, metric_path, metrics, 0.0, predict_normalized)           
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)   
 
    if runtime == "pt":
        model = model.cuda()
    elif runtime == "onnx":
        model = model.cpu()
        import onnxruntime
        onnx_model_path = f"predictor_model_{metrics[0]}.onnx"
        it = iter(dataloader)
        sample_input, _ = next(it)
        pt_2_onnx(model, sample_input, onnx_model_path)
        session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])     

    # Test   
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(dataloader):
            if isinstance(x_batch, (list, tuple)):
                if len(x_batch) == 3:
                    if runtime == "pt":
                        adj_batch, node_feat_batch, arch_feat_batch = x_batch
                        adj_batch = adj_batch.cuda()
                        node_feat_batch = node_feat_batch.cuda()
                        arch_feat_batch = arch_feat_batch.cuda()
                        y_batch = y_batch.cuda()						
                        preds = model(adj_batch, node_feat_batch, arch_feat_batch)
                    elif runtime == "onnx":
                        adj_batch, node_feat_batch, arch_feat_batch = x_batch
                        adj_batch = np.array(adj_batch) 
                        node_feat_batch = np.array(node_feat_batch) 
                        arch_feat_batch = np.array(arch_feat_batch) 
                        data = [adj_batch, node_feat_batch, arch_feat_batch]
                        inputs = {inp.name: data[idx] for idx, inp in enumerate(session.get_inputs())}
                        preds = session.run(["output"], inputs)[0] 
                        preds = torch.from_numpy(preds)
                else:
                    adj_batch, node_feat_batch = x_batch
                    adj_batch = adj_batch.cuda()
                    node_feat_batch = node_feat_batch.cuda()
                    y_batch = y_batch.cuda()
                    preds = model(adj_batch, node_feat_batch)
            else:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                preds = model(x_batch)  
            if predict_normalized == "no_norm":
                pass
            else:
                preds = denormalization(preds, predict_normalized, dataset.minimum, dataset.maximum, dataset.mean, dataset.std)    
            all_preds.append(preds)
            all_targets.append(y_batch)
    all_preds = torch.cat(all_preds).cpu()
    all_targets = torch.cat(all_targets).cpu()
    spearman = spearmanr(all_preds.numpy(), all_targets.numpy()).correlation
    tau, p_value = kendalltau(all_preds.numpy(), all_targets.numpy())
    
    if spearman > spearman_max:
        for metric in metrics:
            # Convert to numpy for plotting
            preds = all_preds.numpy()
            targets = all_targets.numpy()
            plt.figure(figsize=(6, 6))
            plt.scatter(targets, preds, alpha=0.6, edgecolor='b', label='Predictions')
            plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Ideal')  # y=x line
            plt.xlabel(f'True {metric.capitalize()}')
            plt.ylabel(f'Predicted {metric.capitalize()}')
            plt.title(f'{metric.capitalize()} Prediction vs Ground Truth, \n Val set: {len(dataloader.dataset)}, Spearman: {spearman:.4f}, KTau: {tau:.4f}, Seed: {seed}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'targets_vs_preds_hybrid_complete_{metric}_{device}_{predict_normalized}.png')  
        
    print(f"Complete Test Spearman: {spearman:.4f}")    
    return spearman

def plot_pairs(pairs, data_path, metric='latency'):
    #data_path = "/home/rybalkin/HELP/data/unet"    
    num_plots = len(pairs)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), squeeze=False)
    axes = axes[0]


    for i, (lat_i_n, lat_j_n) in enumerate(pairs):
        file_path = os.path.join(data_path, lat_i_n)
        lat_i = np.array([extract_key(arch, metric) for arch in load_jsonl(file_path)])
        file_path = os.path.join(data_path, lat_j_n)
        lat_j = np.array([extract_key(arch, metric) for arch in load_jsonl(file_path)])
        ax = axes[i]
        if lat_i_n == lat_j_n:
            color = 'red'
        else:
            color = 'blue'
        ax.scatter(lat_i, lat_j, s=10, alpha=0.6, color=color)
        mid = len(lat_j_n) // 2
        ax.set_title(f"{lat_j_n[:mid]} \n {lat_j_n[mid:(len(lat_j_n))]}")
        ax.set_xlabel(f"Latency {i+1}")
        ax.set_ylabel(f"Latency {i+2}")
        ax.grid(True)

    fig.suptitle(f"{pairs[0][0]}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{metric}_vs_{metric}.png')

    
def list_jsonl_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Merges several jsonl files in one')
    parser.add_argument('-p', '--path', type=str, default='./', help='path to a dir with muiltiple jsonl files')
    args = parser.parse_args()
    
    # Use system entropy to generate a random seed
    #seed = int.from_bytes(os.urandom(4), 'little')  # 32-bit seed
    #print(f"Seed: {seed}")
   
    devices = ['FILTERED_UNET_ORIN_NANO_tensorrt_modelq_FP32_dataq_FP32_batch_1_952',
               'FILTERED_UNET_ORIN_NANO_tensorrt_modelq_INT8_dataq_FP32_batch_1_959',
               'FILTERED_UNET_ORIN_NX_tensorrt_modelq_FP32_dataq_FP32_batch_32_713',
               'FILTERED_UNET_RP_5_armnn_modelq_INT8_dataq_INT8_batch_1_960',
               'FILTERED_UNET_RP_5_armnn_modelq_INT8_dataq_INT8_batch_16_900',
               'FILTERED_UNET_RP_5_onnx_modelq_FP32_dataq_FP32_batch_1_960',
               'FILTERED_UNET_RP_5_tflite_modelq_INT8_dataq_INT8_batch_8_921',
               'FILTERED_UNET_RTX_2080_TI_pytorch_outocast_modelq_FP32_dataq_FP32_batch_16_943',
               'FILTERED_UNET_RTX_2080_TI_tensorrt_modelq_FP32_dataq_FP32_batch_1_960',
               'FILTERED_UNET_RTX_2080_TI_tensorrt_modelq_FP32_dataq_FP32_batch_16_760']
    
    devices = ['FILTERED_UNET_ORIN_NANO_tensorrt_modelq_FP32_dataq_FP32_batch_1_952',
            'FILTERED_UNET_ORIN_NANO_tensorrt_modelq_INT8_dataq_FP32_batch_1_959',
            'FILTERED_UNET_ORIN_NX_tensorrt_modelq_FP32_dataq_FP32_batch_32_713',
            'FILTERED_UNET_RP_5_armnn_modelq_INT8_dataq_INT8_batch_1_960',
            'FILTERED_UNET_RP_5_armnn_modelq_INT8_dataq_INT8_batch_16_900',
            'FILTERED_UNET_RP_5_onnx_modelq_FP32_dataq_FP32_batch_1_960',
            'FILTERED_UNET_RP_5_tflite_modelq_INT8_dataq_INT8_batch_8_921']
    
    devices = ['FILTERED_UNET_RTX_2080_TI_pytorch_outocast_modelq_FP32_dataq_FP32_batch_16_943',
               'FILTERED_UNET_RTX_2080_TI_tensorrt_modelq_FP32_dataq_FP32_batch_1_960',
               'FILTERED_UNET_RTX_2080_TI_tensorrt_modelq_FP32_dataq_FP32_batch_16_760']
    

    #devices = ["FPGA_xczu7ev_xczu19eg_target_fps_1"]
    #devices = ["FPGA_xczu7ev_target_fps_1"]    
    #devices = ["UNET_RTX_2080_TI_pytorch_autocast_modelq_FP32_dataq_FP32_batch_16_part_0"]     
    #devices = ['UNET_RP_5_onnx_modelq_FP32_dataq_FP32_batch_1_not_finished', 'UNET_RP_5_tflite_modelq_INT8_dataq_INT8_batch_8_not_finished']  
    #devices = ['UNET_FPGA_xczu7ev-ffvc1156-2-e']
    #devices = ['FPGA_xczu7ev_xczu19eg_target_fps_1']
    #devices = ['UNET_FPGA_xczu7ev-ffvc1156-2-e_xczu19eg-ffvb1517-2-i']   
    #devices = ['UNET_CORAL_M2_tflite_modelq_INT8_dataq_INT8_batch_1_676'] 
    devices = ['UNET_FPGA_xczu19eg-ffvb1517-2-i']
    
    predict_normalized = ["no_norm", "log1p_11", "log1p_mean_std", "norm_11", "mean_std"]
    predict_normalized = ["log1p_11"]
    metrics = [["brams"], ["total_energy"], ["dynamic_energy"], ["latency"]]
    metrics = [["latency"]]
    for metric in metrics:
        for device in devices:            
            for norm in predict_normalized:
                spearman_max = -1
                spearman_list = []
                best_seed = 1
                for seed in range(9, 10):            
                    model = train_hybrid_predictor(device, metric, 0.9, 400, seed=seed, outlier_portion=0.0, predict_normalized=norm) # 300511456 # 1270581859
                    spearman = test_complete_hybrid_predictor(model, device, metric, "pt", predict_normalized=norm, spearman_max=spearman_max, seed=seed)                
                    spearman = test_complete_hybrid_predictor(model, device, metric, "onnx", predict_normalized=norm, spearman_max=spearman_max, seed=seed)
                    if spearman > spearman_max:
                        spearman_max = spearman
                        best_seed = seed
                    spearman_list.append(spearman)
                    experiments = {"metric": metric[0],
                                   "norm" : norm,
                                   "seed": seed,
                                   "best seed": best_seed,
                                   "current_spearman": spearman,
                                   "best_spearman": spearman_max,
                                   "mean_spearman": (np.array(spearman_list)).mean(),
                                   "std_spearman": (np.array(spearman_list)).std()}
                    with open("alloutputs.txt", 'a') as f:
                        json.dump(experiments, f, cls=NumpyEncoder)
                        f.write('\n') 
        
        #train_gcn_predictor(device, "latency", 0.5, 400, seed=1786145379, outlier_portion=0.0)
        #train_fc_predictor(device, "unet", "power_runtime_extrnl", 0.5, 800, seed=1786145379, outlier_portion=0.0)
        #train_attention_predictor(device, "power_runtime", 0.3, 400, seed=31)
    
    '''# Check correlation between all devices
    devices = list_jsonl_files(args.path)
    import itertools
    pairs = []
    for idx, (dev1, dev2) in enumerate(itertools.product(devices, repeat=2), start=1):
        pairs.append([dev1, dev2])
        if idx % 14 == 0 and idx != 0:
            plot_pairs(pairs, args.path)
            pairs.clear()
            input()'''
            
