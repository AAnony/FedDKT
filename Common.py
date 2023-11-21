import numpy as np
import torch

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
    '''
    # 总类别数
    n_classes = train_labels.max()+1

    # [alpha]*n_clients 如下：
    # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # 得到 62 * 10 的标签分布矩阵，记录每个 client 占有每个类别的比率
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    # 记录每个类别对应的样本下标
    # 返回二维数组
    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(n_classes)]

    # 定义一个空列表作最后的返回值
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

#这个函数的作用是将test_labels中每个类按照比例抽取，放到choose_idc里面，剩下的放到remain_dic里面
def make_common_dataset(test_labels):
    n_classes = test_labels.max()+1
    choose_idc = []
    remain_idc = []
    #逐个类别抽取百分比
    for c in range(n_classes):
        idx = np.where(test_labels == c)[0]
        np.random.shuffle(idx)
        choose_count = int(len(idx)*0.1)
        choose_idc.extend(idx[:choose_count])
        remain_idc.extend(idx[choose_count:])
    #choose_dic是选中的，remain_dic是没选中的
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