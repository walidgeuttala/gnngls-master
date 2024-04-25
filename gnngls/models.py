import dgl.nn
import torch.nn as nn
import dgl
import torch

class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, G=None):
        if G is not None:
            y = self.module(G, x).view(G.number_of_nodes(), -1)
        else:
            y = self.module(x)
        return x + y

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        h = x
        h = self.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
    
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim):
        super().__init__()

        self.message_passing = SkipConnection(
            dgl.nn.GATConv(embed_dim, embed_dim // n_heads, n_heads)
        )

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim)
                ),
            ),
            nn.BatchNorm1d(embed_dim),
        )
        
    def forward(self, G, x):
        h = self.message_passing(x, G=G).view(G.number_of_nodes(), -1)
        h = self.feed_forward(h)
        return h


class EdgePropertyPredictionModel(nn.Module):
    def __init__(
            self,
            in_dim,
            embed_dim,
            out_dim,
            n_layers,
            n_heads=1,
            embed_dim2 = 218,
            kj = "cat"
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.embed_layer = nn.Linear(in_dim, embed_dim)

        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(embed_dim, n_heads, embed_dim2) for _ in range(n_layers))
        )
        self.jkl = dgl.nn.pytorch.utils.JumpingKnowledge(kj)
        if kj == "cat":
            self.decision_layer = MLP(embed_dim*(n_layers+1), embed_dim, out_dim)
        else:
            self.decision_layer = MLP(embed_dim, embed_dim, out_dim)
        
    def forward(self, G, x):
        h = self.embed_layer(x)
        xs = []
        xs.append(h)
        for l in self.message_passing_layers:
            h = l(G, h)
            xs.append(h)
        h = self.jkl(xs)
        h = self.decision_layer(h)
        return h



class EdgePropertyPredictionModel2(nn.Module):
    def __init__(
            self,
            in_dim,
            embed_dim,
            out_dim,
            n_layers,
            n_heads=1,
            embed_dim2 = 218,
            kj = "cat"
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.embed_layer = nn.Linear(in_dim, embed_dim)

        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer(embed_dim, n_heads, embed_dim2) for _ in range(n_layers))
        )
        
        self.decision_layer = nn.Linear(embed_dim, out_dim)
        
    def forward(self, G, x):
        h = self.embed_layer(x)
        for l in self.message_passing_layers:
            h = l(G, h)
        h = self.decision_layer(h)
        return h
    

class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, G=None):
        if G is not None:
            y = self.module(G, x).view(G.number_of_nodes(), -1)
        else:
            y = self.module(x)
        return x + y

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        h = x
        h = self.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
    
class AttentionLayer2(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim):
        super().__init__()

        self.message_passing = dgl.nn.GATConv(embed_dim, embed_dim // n_heads, n_heads)

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim)
                ),
            nn.BatchNorm1d(embed_dim),
        )
        
    def forward(self, G, x):
        h = self.message_passing(G, x).view(G.number_of_nodes(), -1)
        h = self.feed_forward(h)
        return h

class EdgePropertyPredictionModel3(nn.Module):
    def __init__(
            self,
            in_dim,
            embed_dim,
            out_dim,
            n_layers,
            n_heads=1,
            embed_dim2 = 218,
            kj = "cat"
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.embed_layer = nn.Linear(in_dim, embed_dim)

        self.message_passing_layers = dgl.nn.utils.Sequential(
            *(AttentionLayer2(embed_dim, n_heads, embed_dim2) for _ in range(n_layers))
        )
        self.jkl = dgl.nn.pytorch.utils.JumpingKnowledge(kj)
        self.decision_layer = MLP(embed_dim*(n_layers+1), embed_dim, out_dim)
        
    def forward(self, G, x):
        h = self.embed_layer(x)
        xs = []
        xs.append(h)
        for l in self.message_passing_layers:
            h = l(G, h)
            xs.append(h)
        h = self.jkl(xs)
        h = self.decision_layer(h)
        return h