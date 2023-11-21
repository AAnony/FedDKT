import numpy as np
import torch

def dirichlet_split_noniid(train_labels, alpha, n_clients):

    n_classes = train_labels.max()+1

    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def make_common_dataset(test_labels):
    n_classes = test_labels.max()+1
    choose_idc = []
    remain_idc = []
    for c in range(n_classes):
        idx = np.where(test_labels == c)[0]
        np.random.shuffle(idx)
        choose_count = int(len(idx)*0.1)
        choose_idc.extend(idx[:choose_count])
        remain_idc.extend(idx[choose_count:])
    return np.array(choose_idc), np.array(remain_idc)

def test_model(model:torch.nn.Module, test_loader:torch.utils.data.DataLoader, args):
    model.eval()
    sum_accu = 0
    num = 0
    for (data, label) in test_loader:
        data, label = data.to(args.dev), label.to(args.dev)
        preds = model(data)
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean().item()
        num += 1
    sum_accu *= 100
    return round(sum_accu / num, 3)