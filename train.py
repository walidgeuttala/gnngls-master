#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import os
import pathlib
import uuid
import numpy as np
import dgl.nn
import torch
import tqdm.auto as tqdm
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from gnngls import models, datasets

def train(model, data_loader, target, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(train_loader):
        batch = batch.to(device)
        x = batch.ndata['weight']
        y = batch.ndata[target]

        optimizer.zero_grad()
        y_pred = model(batch, x)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)
    return epoch_loss


def test(model, data_loader, target, criterion, device):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(device)
            x = batch.ndata['weight']
            y = batch.ndata[target]

            y_pred = model(batch, x)
            loss = criterion(y_pred, y.type_as(y_pred))

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
        return epoch_loss


def save(model, optimizer, epoch, train_loss, val_loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'val_loss': val_loss
    }, save_path)

def correlation_matrix(tensor1, tensor2):
    """
    Computes the correlation matrix between two tensors.
    
    Args:
    - tensor1 (torch.Tensor): The first input tensor.
    - tensor2 (torch.Tensor): The second input tensor.
    
    Returns:
    - corr_matrix (np.ndarray): The correlation matrix.
    """
    # Flatten tensors into 1D arrays
    flat_tensor1 = tensor1.flatten().numpy()
    flat_tensor2 = tensor2.flatten().numpy()

    # Concatenate flattened tensors along the second axis
    concatenated_tensors = np.stack((flat_tensor1, flat_tensor2), axis=1)

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(concatenated_tensors, rowvar=False)
    
    abs_correlation_matrix = np.abs(corr_matrix)
    
    # Compute the mean of the absolute correlation coefficients
    strength = np.mean(abs_correlation_matrix)
    return strength

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_dir', type=pathlib.Path, help='Where to load dataset')
    parser.add_argument('tb_dir', type=pathlib.Path, help='Where to log Tensorboard data')
    parser.add_argument('--embed_dim', type=int, default=128, help='Maximum hidden feature dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of message passing steps')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads for GAT')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=20, help='Checkpoint frequency')
    parser.add_argument('--target', type=str, default='regret', choices=['regret', 'in_solution'])
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    # Load dataset
    train_set = datasets.TSPDataset(args.data_dir / 'train.txt')
    val_set = datasets.TSPDataset(args.data_dir / 'val.txt')

    # use GPU if it is available
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print('device =', device)

    _, feat_dim = train_set[0].ndata['weight'].shape
    model = models.EdgePropertyPredictionModel(
        feat_dim,
        args.embed_dim,
        1,
        args.n_layers,
        n_heads=args.n_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if args.target == 'regret':
        criterion = torch.nn.MSELoss()

    elif args.target == 'in_solution':
        # only works for a homogenous dataset
        y = train_set[0].ndata['in_solution']
        pos_weight = len(y) / y.sum() - 1
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                              num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                            num_workers=16, pin_memory=True)

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{uuid.uuid4().hex}'
    log_dir = args.tb_dir / run_name
    writer = SummaryWriter(log_dir)

    # early stopping
    best_score = None
    counter = 0
    G = nx.read_gpickle(args.data_dir / val_set.instances[0])
    H = val_set.get_scaled_features(G).to(device)
    x = H.ndata['weight']
    y = H.ndata['regret']
    
    pbar = tqdm.trange(args.n_epochs)
    for epoch in pbar:
        epoch_loss = train(model, train_loader, args.target, criterion, optimizer, device)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        epoch_val_loss = test(model, train_loader, args.target, criterion, device)
        writer.add_scalar("Loss/validation", epoch_val_loss, epoch)
        
        with torch.no_grad():
            y_pred = model(H, x)
        
        # regret_pred = val_set.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
        # es = H.ndata['e'].cpu().numpy()
        # for e, regret_pred_i in zip(es, regret_pred):
        #     G.edges[e]['regret_pred'] = np.maximum(regret_pred_i.item(), 0)
        # init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')

        pbar.set_postfix({
            'Train Loss': '{:.4f}'.format(epoch_loss),
            'Validation Loss': '{:.4f}'.format(epoch_val_loss),
            "correlation : ": '{:.4f}'.format(correlation_matrix(y_pred.cpu(),H.ndata['regret'].cpu())),
            "cosin correlation : ": '{:.4f}'.format(cosine_similarity(y_pred.cpu(),H.ndata['regret'].cpu()))
        })
        
        if args.checkpoint_freq is not None and epoch > 0 and epoch % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_{epoch}.pt'
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / checkpoint_name)

        if best_score is None or epoch_val_loss < best_score - args.min_delta:
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_best_val.pt')

            best_score = epoch_val_loss
            counter = 0
        else:
            counter += 1
        
        
        if counter >= args.patience:
            pbar.close()
            break

        lr_scheduler.step()

    writer.close()

    params = dict(vars(args))
    params['data_dir'] = str(params['data_dir'])
    params['tb_dir'] = str(params['tb_dir'])
    json.dump(params, open(args.tb_dir / run_name / 'params.json', 'w'))

    save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_final.pt')
