import torch
import torch.nn.functional as F
import dgl
import psutil
import os
import gc
from ogb.nodeproppred import Evaluator, NodePropPredDataset

from model import GipaWideDeep
from utils.tools import seed, count_model_parameters

# --- Utility to track memory ---
def print_memory(tag):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2
    print(f"[MEMORY] {tag}: {mem_mb:.2f} MB")

# --- 0. Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print_memory("Start")

# --- 1. Set seed
seed(42)

# --- 2. Load dataset
print_memory("Before loading dataset")
dataset = NodePropPredDataset(name='ogbn-proteins', root='./data')
split_idx = dataset.get_idx_split()
graph_dict, labels = dataset[0]
print_memory("After loading dataset")

# --- 3. Confirm node features are None
if graph_dict['node_feat'] is not None:
    raise ValueError("Expected node_feat to be None for ogbn-proteins, but got valid data.")

# --- 4. Build full DGL graph
print_memory("Before building full graph")
edge_index = graph_dict['edge_index']
src = torch.tensor(edge_index[0], dtype=torch.long)
dst = torch.tensor(edge_index[1], dtype=torch.long)
g = dgl.graph((src, dst), num_nodes=graph_dict['num_nodes'])
print_memory("After building full graph")

# --- 5. Assign labels and edge features
labels = torch.tensor(labels, dtype=torch.float)
g.ndata['labels'] = labels
g.edata['feat'] = torch.tensor(graph_dict['edge_feat'], dtype=torch.float)
print_memory("After assigning labels and edge features")

# --- 6. Construct node features
in_deg = g.in_degrees().clamp(min=1).unsqueeze(1).float()
agg_feat = torch.zeros(g.num_nodes(), g.edata['feat'].shape[1])
dst_nodes = g.edges()[1]
agg_feat = agg_feat.index_add(0, dst_nodes, g.edata['feat']) / in_deg
print_memory("After aggregating edge features")

# --- 7. One-hot encode species info
species = torch.tensor(graph_dict['node_species'], dtype=torch.long).squeeze()
num_species = int(species.max().item()) + 1
species_onehot = F.one_hot(species, num_classes=num_species).float()
print_memory("After one-hot encoding species")

# --- 8. Combine node features
g.ndata['feat'] = torch.cat([agg_feat, species_onehot], dim=1)
g.ndata['species'] = species
print_memory("After combining node features")

# --- 9. Clean up memory
for obj in [graph_dict, dataset, edge_index, src, dst, agg_feat, species_onehot]:
    del obj
gc.collect()
torch.cuda.empty_cache()
print_memory("After cleanup")

# --- 10. Move data to device
g = g.to(device)
for name in ['feat', 'labels', 'species']:
    g.ndata[name] = g.ndata[name].to(device)
g.edata['feat'] = g.edata['feat'].to(device)
for k in split_idx:
    split_idx[k] = split_idx[k].to(device)
print_memory("After moving to device")

# --- 11. Model, optimizer, evaluator
in_node_feats = g.ndata['feat'].shape[1]
in_edge_feats = g.edata['feat'].shape[1]
n_classes = g.ndata['labels'].shape[1]

model = GipaWideDeep(
    in_node_feats=in_node_feats,
    in_node_emb=None,
    in_edge_feats=in_edge_feats,
    out_feats=n_classes,
    n_layers=3,
    n_deep_layers=3,
    n_heads=4,
    n_hidden=128,
    n_deep_hidden=128,
    edge_emb=32,
    dropout=0.2,
    input_drop=0.1,
    edge_drop=0.1,
    use_attn_dst=True,
    norm="layer",
    batch_norm=True,
    edge_att_act="leaky_relu",
    edge_agg_mode="both_softmax",
    use_node_sparse=False,
    input_norm=False,
    first_hidden=128,
    first_layer_act=True,
    first_layer_drop=0.1,
    first_layer_norm=True,
    last_layer_drop=0.1,
    use_att_edge=True,
    use_prop_edge=False,
    edge_prop_size=32
)

print(f"# Model Params: {count_model_parameters(model)}")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
evaluator = Evaluator(name='ogbn-proteins')

# --- 12. Training loop
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()

    out = model(g, g.ndata['feat'], g.edata['feat'])
    loss = F.binary_cross_entropy_with_logits(out[train_idx], g.ndata['labels'][train_idx])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = out > 0
        train_acc = evaluator.eval({'y_true': g.ndata['labels'][train_idx], 'y_pred': pred[train_idx]})['acc']
        valid_acc = evaluator.eval({'y_true': g.ndata['labels'][valid_idx], 'y_pred': pred[valid_idx]})['acc']

    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")
