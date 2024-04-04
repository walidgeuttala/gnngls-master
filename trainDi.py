import torch 
from gnngls import datasets
from model import GNN
from torch.utils.data import DataLoader

from torch_geometric.data import Batch

def collate_fn(batch):
    return Batch.from_data_list(batch)

def train(model, data_loader, target, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(data_loader):
        batch = batch.to(device)
        x = batch.ndata['features']
        y = batch.ndata[target]

        optimizer.zero_grad()
        y_pred = model(batch, x)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)
    return epoch_loss

train_set = datasets.TSPDataset('../cleaned_data_n5900/val.txt')
device = "cuda"
_, feat_dim = train_set[0].ndata['features'].shape
model = GNN(
        num_features=feat_dim,
        hidden_dim=64,
        num_layers=2,
        num_classes=1,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1e-3)
criterion = torch.nn.MSELoss()

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn,
                              num_workers=16, pin_memory=True)

print(train(model, train_loader, "regret", criterion, optimizer, device))

