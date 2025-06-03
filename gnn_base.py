import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import Linear
from tqdm import tqdm

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from utils.save_metrics import save_training_metrics


from utils.plot_metrics import plot_training_metrics  # <-- plotting utility

# Save the original load
_original_torch_load = torch.load

def custom_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = custom_torch_load

# Load dataset
dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize node features by aggregating edge attributes
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

# Set split masks
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

# Data loaders
train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True, num_workers=2)
test_loader = RandomNodeLoader(data, num_parts=10, num_workers=2)

# Model definition
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGCN(data.x.size(-1), 64, data.y.size(-1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')

# Training function
def train(epoch):
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f"Training epoch {epoch:04d}")
    total_loss = total_examples = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())
        pbar.update(1)

    pbar.close()
    return total_loss / total_examples

# Evaluation function
@torch.no_grad()
def test():
    model.eval()
    y_true, y_pred = {'train': [], 'valid': [], 'test': []}, {'train': [], 'valid': [], 'test': []}
    pbar = tqdm(total=len(test_loader), desc="Evaluating")

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        for split in y_true:
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())
        pbar.update(1)

    pbar.close()

    results = {}
    for split in ['train', 'valid', 'test']:
        results[split] = evaluator.eval({
            'y_true': torch.cat(y_true[split], dim=0),
            'y_pred': torch.cat(y_pred[split], dim=0),
        })['rocauc']

    return results['train'], results['valid'], results['test']

# Metric trackers
train_rocauc_list, valid_rocauc_list, test_rocauc_list, loss_list = [], [], [], []

# Main training loop
for epoch in range(1, 41):
    loss = train(epoch)
    loss_list.append(loss)

    train_rocauc, valid_rocauc, test_rocauc = test()
    train_rocauc_list.append(train_rocauc)
    valid_rocauc_list.append(valid_rocauc)
    test_rocauc_list.append(test_rocauc)

    print(f'Loss: {loss:.4f}, Train ROC-AUC: {train_rocauc:.4f}, '
          f'Val ROC-AUC: {valid_rocauc:.4f}, Test ROC-AUC: {test_rocauc:.4f}')

# Plot and save graphs
plot_training_metrics(
    train_rocauc=train_rocauc_list,
    valid_rocauc=valid_rocauc_list,
    test_rocauc=test_rocauc_list,
    loss_list=loss_list,
    save_dir="graphs",
    prefix="base_"
)


# Save metrics
save_training_metrics(
    train_rocauc=train_rocauc_list,
    valid_rocauc=valid_rocauc_list,
    test_rocauc=test_rocauc_list,
    loss_list=loss_list,
    save_dir="results",
    prefix="basegcn"
)

