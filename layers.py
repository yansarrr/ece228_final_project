import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class GIPAWideConv(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.fc_node = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.Tensor(size=(1, num_heads, out_feats)))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_edge.weight)
        nn.init.xavier_normal_(self.attn)

    def forward(self, graph, node_feat, edge_feat):
        h = self.fc_node(node_feat).view(-1, self.num_heads, self.out_feats)
        e = self.fc_edge(edge_feat).view(-1, self.num_heads, self.out_feats)

        graph.srcdata['h'] = h
        graph.edata['e'] = e

        graph.apply_edges(lambda edges: {
            'score': (edges.src['h'] * edges.data['e']).sum(dim=-1)
        })
        e_score = graph.edata['score']
        e_score = e_score / (self.out_feats ** 0.5)
        e_score = F.leaky_relu(e_score)
        attn_score = dgl.nn.functional.edge_softmax(graph, e_score)
        graph.edata['a'] = self.dropout(attn_score)

        graph.update_all(
            fn.u_mul_e('h', 'a', 'm'),
            fn.sum('m', 'h_out')
        )

        h_out = graph.dstdata['h_out']  # [N, num_heads, out_feats]
        return h_out.flatten(1)  # [N, num_heads * out_feats]


class GIPADeepConv(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_feats + edge_feats, out_feats)
        self.fc2 = nn.Linear(out_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_feats)

    def message_func(self, edges):
        z = torch.cat([edges.src['h'], edges.data['e']], dim=-1)
        m = self.fc1(z)
        m = F.relu(m)
        m = self.fc2(m)
        return {'m': m}

    def forward(self, graph, node_feat, edge_feat):
        graph = graph.local_var()
        graph.srcdata['h'] = node_feat
        graph.edata['e'] = edge_feat
        graph.update_all(self.message_func, fn.mean('m', 'h_new'))
        h_new = graph.dstdata['h_new']
        h_new = self.dropout(h_new)
        h_new = self.norm(h_new + node_feat)
        return h_new
