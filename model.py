import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import EdgeWeightNorm

from layers import GIPAWideConv, GIPADeepConv  # Assume layers are modular

class GipaWideDeep(nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_node_sparse_feats,
                 in_edge_feats,
                 out_feats,
                 n_layers=2,
                 n_deep_layers=2,
                 n_heads=4,
                 n_hidden=64,
                 n_deep_hidden=64,
                 edge_emb=16,
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
                 first_hidden=64,
                 first_layer_act=True,
                 first_layer_drop=0.1,
                 first_layer_norm=True,
                 last_layer_drop=0.1,
                 use_att_edge=True,
                 use_prop_edge=False,
                 edge_prop_size=20):
        super(GipaWideDeep, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.last_layer_drop = nn.Dropout(last_layer_drop)

        self.input_linear = nn.Linear(in_node_feats, n_hidden)

        self.wide_convs = nn.ModuleList([
            GIPAWideConv(n_hidden, n_hidden, in_edge_feats, n_heads=n_heads, 
                         edge_emb=edge_emb, use_attn_dst=use_attn_dst, 
                         norm=norm, batch_norm=batch_norm, edge_att_act=edge_att_act,
                         edge_agg_mode=edge_agg_mode, edge_drop=edge_drop)
            for _ in range(n_layers)
        ])

        self.deep_convs = nn.ModuleList([
            GIPADeepConv(n_hidden, n_deep_hidden, in_edge_feats, n_heads=n_heads, 
                         edge_emb=edge_emb, use_attn_dst=use_attn_dst,
                         norm=norm, batch_norm=batch_norm, edge_att_act=edge_att_act,
                         edge_agg_mode=edge_agg_mode, edge_drop=edge_drop)
            for _ in range(n_deep_layers)
        ])

        self.output_linear = nn.Linear(n_deep_hidden, out_feats)

    def forward(self, g, x, edge_attr):
        h = self.input_linear(x)
        h = self.input_drop(h)

        for conv in self.wide_convs:
            h = conv(g, h, edge_attr)
            h = F.relu(h)
            h = self.dropout(h)

        for conv in self.deep_convs:
            h = conv(g, h, edge_attr)
            h = F.relu(h)
            h = self.dropout(h)

        h = self.last_layer_drop(h)
        out = self.output_linear(h)
        return out
