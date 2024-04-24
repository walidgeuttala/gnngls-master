from gnngls import datasets
import torch
val_data = datasets.TSPDataset(f"../tsp_n5900/val.txt")

ans = 0
for idx in range(len(val_data)):
    ans += val_data[idx].ndata['regret'].mean()

print(ans/len(val_data))