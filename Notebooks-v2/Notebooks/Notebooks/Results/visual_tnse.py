"""
t-SNE Visualization for Section IX (Analysis)
Generates: 'anomaly_tsne_plot.png'
Feature: Visualizes the Learned Latent Space (Z) of the DOMINANT model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. RE-DEFINE MODEL (Must match training exactly)
# ============================================================================
class DOMINANT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, dropout=0.3):
        super(DOMINANT, self).__init__()
        self.fusion = nn.Linear(input_dim, hidden_dim)
        self.dropout = dropout
        self.gc1 = GCNConv(hidden_dim, 64)
        self.gc2 = GCNConv(64, latent_dim)
        self.attr_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.fusion(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        z = self.gc2(x, edge_index)
        x_hat = self.attr_decoder(z)
        return x_hat, z

# ============================================================================
# 2. DATA LOADER (Simplified from main_ensemble.py)
# ============================================================================
def load_data_for_viz():
    logging.info(">>> Loading Graph Data...")
    
    # Load Features
    x_cve = torch.load('/teamspace/uploads/node_features.pt')
    input_dim = x_cve.size(1)
    
    # Load Mapping
    df_map = pd.read_csv('/teamspace/uploads/cve_id_mapping.csv')
    id_to_idx = {row['CVE_ID']: idx for idx, row in df_map.iterrows()}
    
    # Load Edges (Only need CVE->CWE for structure context, but safer to load all)
    # For Visualization speed, we can just load the primary structural edges
    edges_source, edges_target = [], []
    extra_nodes = {}
    current_max_idx = len(id_to_idx)
    
    def process_file(filename, src_col, tgt_col):
        nonlocal current_max_idx
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                src, tgt = row[src_col], row[tgt_col]
                
                if src in id_to_idx: u = id_to_idx[src]
                elif src in extra_nodes: u = extra_nodes[src]
                else: extra_nodes[src] = current_max_idx; u = current_max_idx; current_max_idx += 1
                
                if tgt in id_to_idx: v = id_to_idx[tgt]
                elif tgt in extra_nodes: v = extra_nodes[tgt]
                else: extra_nodes[tgt] = current_max_idx; v = current_max_idx; current_max_idx += 1
                
                edges_source.append(u)
                edges_target.append(v)

    # Load minimal edges for connectivity
    process_file('edges_cve_cwe.csv', 'CVE_ID', 'Weakness_CWE')
    
    # Expand X
    num_extra = len(extra_nodes)
    if num_extra > 0:
        x_noise = torch.randn(num_extra, input_dim) * 0.01
        x_full = torch.cat([x_cve, x_noise], dim=0)
    else:
        x_full = x_cve
        
    edge_index = torch.tensor([edges_source, edges_target], dtype=torch.long)
    return Data(x=x_full, edge_index=edge_index), input_dim, df_map

# ============================================================================
# 3. VISUALIZATION LOGIC
# ============================================================================
def visualize():
    # A. Setup
    data, input_dim, df_map = load_data_for_viz()
    data = data.to(device)
    
    # B. Load Trained Model
    logging.info(">>> Loading Trained Model Weights...")
    if not os.path.exists('dominant_model.pt'):
        raise FileNotFoundError("Run main_ensemble.py first to generate the model!")
        
    model = DOMINANT(input_dim).to(device)
    model.load_state_dict(torch.load('dominant_model.pt', map_location=device))
    model.eval()
    
    # C. Get Latent Embeddings (Z)
    logging.info(">>> Generating Latent Embeddings (Z)...")
    with torch.no_grad():
        _, z = model(data.x, data.edge_index)
        z_np = z.cpu().numpy()
        
    # D. Identify Anomalies vs Normal
    # Load results to find who is who
    results = pd.read_csv('final_ensemble_results.csv')
    top_anomalies = results.head(20)['CVE_ID'].values
    
    # Map IDs to Indices
    anomaly_indices = []
    normal_indices = []
    
    # Create set for fast lookup
    anomaly_set = set(top_anomalies)
    
    # Get original CVE indices (0 to len(df_map))
    # We only plot CVEs, not the extra noise nodes
    for i in range(len(df_map)):
        cve_id = df_map.iloc[i]['CVE_ID']
        if cve_id in anomaly_set:
            anomaly_indices.append(i)
        else:
            normal_indices.append(i)
            
    # E. Downsample Normal Nodes (For Speed/Clarity)
    # 800k points is too many. Let's take 5,000 random normal nodes + All Anomalies
    logging.info(f">>> Subsampling: 5000 Normal + {len(anomaly_indices)} Anomalies")
    
    np.random.shuffle(normal_indices)
    selected_normal = normal_indices[:5000]
    
    # Combine for t-SNE
    final_indices = selected_normal + anomaly_indices
    z_subset = z_np[final_indices]
    
    # F. Run t-SNE
    logging.info(">>> Running t-SNE (This takes ~1-2 mins)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    z_2d = tsne.fit_transform(z_subset)
    
    # G. Plot
    logging.info(">>> Plotting...")
    plt.figure(figsize=(12, 10))
    
    # Split back into normal/anomaly for coloring
    split_point = len(selected_normal)
    
    # Plot Normal (Blue, Transparent)
    plt.scatter(z_2d[:split_point, 0], z_2d[:split_point, 1], 
                c='steelblue', label='Normal CVEs', alpha=0.3, s=10)
    
    # Plot Anomalies (Red, Bold)
    plt.scatter(z_2d[split_point:, 0], z_2d[split_point:, 1], 
                c='red', label='Top Anomalies', alpha=1.0, s=80, edgecolors='black')
    
    plt.title('Latent Space Visualization of CVEs (DOMINANT)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save
    plt.savefig('anomaly_tsne_plot.png', dpi=300)
    logging.info("✅ Plot saved to 'anomaly_tsne_plot.png'")
    
    plt.show()

if __name__ == "__main__":
    visualize()