import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter
from utils.save_metrics import save_training_metrics


from utils.plot_metrics import plot_training_metrics  # <-- import your plotting helper

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

# Initialize node features by aggregating edge features
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

# Set train/valid/test masks
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

# Define data loaders
train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True, num_workers=5)
test_loader = RandomNodeLoader(data, num_parts=10, num_workers=5)

# Model definition
class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='mean',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')

# Training function
def train(epoch):
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f'Training epoch {epoch:04d}')
    total_loss = total_examples = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
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
    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}
    pbar = tqdm(total=len(test_loader), desc="Evaluating")

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        for split in y_true:
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())
        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

# Metric trackers
train_rocauc_list = []
valid_rocauc_list = []
test_rocauc_list = []
loss_list = []

# Training loop
for epoch in range(1, 41):
    loss = train(epoch)
    loss_list.append(loss)

    train_rocauc, valid_rocauc, test_rocauc = test()
    train_rocauc_list.append(train_rocauc)
    valid_rocauc_list.append(valid_rocauc)
    test_rocauc_list.append(test_rocauc)

    print(f'Loss: {loss:.4f}, Train ROC-AUC: {train_rocauc:.4f}, '
          f'Val ROC-AUC: {valid_rocauc:.4f}, Test ROC-AUC: {test_rocauc:.4f}')

# Save plots
plot_training_metrics(
    train_rocauc=train_rocauc_list,
    valid_rocauc=valid_rocauc_list,
    test_rocauc=test_rocauc_list,
    loss_list=loss_list,
    save_dir="graphs",
    prefix="deepergcn_"
)

# Save metrics
save_training_metrics(
    train_rocauc=train_rocauc_list,
    valid_rocauc=valid_rocauc_list,
    test_rocauc=test_rocauc_list,
    loss_list=loss_list,
    save_dir="results",
    prefix="deepergcn"
)


