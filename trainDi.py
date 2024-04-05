import torch 
from gnngls import datasets
from model import GNN
import tqdm.auto as tqdm
import argparse
import datetime
import json
import os
import pathlib
import uuid

import dgl.nn
import torch
import tqdm.auto as tqdm
from torch.utils.tensorboard import SummaryWriter

from gnngls import models, datasets
# from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
def collate_fn(batch):
    return Batch.from_data_list(batch)

def train(model, data_loader, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(data_loader):
        batch = batch.to(device)
        x = batch.x
        y = batch.y

        optimizer.zero_grad()
        y_pred = model(x, batch.edge_index)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)
    return epoch_loss

def test(model, data_loader, criterion, device):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(device)
            x = batch.x
            y = batch.y

            y_pred = model(x, batch.edge_index)
            loss = criterion(y_pred, y.type_as(y_pred))

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
        return epoch_loss

train_set = datasets.TSPDataset('../atsp_n5900/train.txt')
val_set = datasets.TSPDataset('../atsp_n5900/val.txt')

device = "cuda"
model = GNN(
        num_features=1,
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

train_loader = DataLoader(train_set, batch_size=50, shuffle=True,
                              num_workers=16, pin_memory=False)

print(train(model, train_loader, criterion, optimizer, device))
timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
run_name = f'{timestamp}_{uuid.uuid4().hex}'
tb_dir = pathlib.Path('../slurm')
log_dir = tb_dir / run_name
writer = SummaryWriter(log_dir)
epochs = 1
pbar = tqdm.trange(epochs)
for epoch in pbar:
        epoch_loss = train(model, train_loader, criterion, optimizer, device)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        epoch_val_loss = test(model, train_loader, criterion, device)
        writer.add_scalar("Loss/validation", epoch_val_loss, epoch)

        pbar.set_postfix({
            'Train Loss': '{:.4f}'.format(epoch_loss),
            'Validation Loss': '{:.4f}'.format(epoch_val_loss),
        })

        lr_scheduler.step()

writer.close()
