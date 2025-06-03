import os
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import scatter
from torch_geometric.loader import RandomNodeLoader
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv
from tqdm import tqdm

from utils.plot_metrics import plot_training_metrics
from utils.save_metrics import save_training_metrics

# Save the original load
_original_torch_load = torch.load

def custom_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = custom_torch_load

# --- Load and preprocess dataset ---
dataset = PygNodePropPredDataset(name='ogbn-proteins', root='../data')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-proteins')

data.node_species = None
row, col = data.edge_index

data.y = data.y.to(torch.float)
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[split_idx[split]] = True
    data[f'{split}_mask'] = mask

train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True, num_workers=4)
test_loader = RandomNodeLoader(data, num_parts=10, num_workers=4)

# --- GIPA-DeepGCN model definition ---
class GipaDeeperGCN(torch.nn.Module):
    def __init__(self, in_channels, edge_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = GENConv(hidden_channels, hidden_channels, aggr='mean', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=i % 3 == 0)
            self.layers.append(layer)

        self.out_norm = LayerNorm(hidden_channels)
        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x = self.out_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.out_lin(x)

# --- Setup model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GipaDeeperGCN(
    in_channels=data.x.size(-1),
    edge_channels=data.edge_attr.size(-1),
    hidden_channels=64,
    out_channels=data.y.size(-1),
    num_layers=28
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

train_rocauc_list, valid_rocauc_list, test_rocauc_list, loss_list = [], [], [], []

# --- Training loop ---
def train(epoch):
    model.train()
    total_loss, total_examples = 0, 0
    pbar = tqdm(total=len(train_loader), desc=f"Training epoch {epoch:04d}")

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.train_mask.sum().item()
        total_examples += batch.train_mask.sum().item()
        pbar.update(1)

    pbar.close()
    return total_loss / total_examples

@torch.no_grad()
def test(epoch):
    model.eval()
    y_true, y_pred = {'train': [], 'valid': [], 'test': []}, {'train': [], 'valid': [], 'test': []}
    pbar = tqdm(total=len(test_loader), desc=f"Evaluating epoch {epoch:04d}")

    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        for split in y_true.keys():
            mask = batch[f'{split}_mask']
            y_true[split].append(batch.y[mask].cpu())
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

# --- Run training ---
for epoch in range(1, 41):
    loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc = test(epoch)

    loss_list.append(loss)
    train_rocauc_list.append(train_rocauc)
    valid_rocauc_list.append(valid_rocauc)
    test_rocauc_list.append(test_rocauc)

    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train ROC-AUC: {train_rocauc:.4f} | "
          f"Valid ROC-AUC: {valid_rocauc:.4f} | Test ROC-AUC: {test_rocauc:.4f}")

# --- Save metrics and plots ---
plot_training_metrics(
    train_rocauc=train_rocauc_list,
    valid_rocauc=valid_rocauc_list,
    test_rocauc=test_rocauc_list,
    loss_list=loss_list,
    save_dir="graphs",
    prefix="gipa_deepergcn"
)

save_training_metrics(
    train_rocauc=train_rocauc_list,
    valid_rocauc=valid_rocauc_list,
    test_rocauc=test_rocauc_list,
    loss_list=loss_list,
    save_dir="results",
    prefix="gipa_deepergcn"
)
