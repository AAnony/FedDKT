import argparse
from torchvision import datasets, transforms
import torch, random, os, copy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from Common import *
from Client import Client
from Model import *

parser = argparse.ArgumentParser(description='FedDKT')

parser.add_argument('--batch_size', type=int, default=60, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lamda', type=float, default=0.5,
                    help='KL loss weight (default: 0.5)') ##损失权重系数
parser.add_argument('--T', type=float, default=5.0,
                    help='knowledge distillation temperature (default: 5)') ##蒸馏温度系数
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--tmodelpath', type=str, default='models/tmodel.pth.tar', 
                    help='teacher model path')
parser.add_argument('--save', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save student model (default: current directory)')
parser.add_argument('--cuda', default='2', type=int)
parser.add_argument('--dataset', default='cifa10', type=str,
                    help='dataset')
parser.add_argument('--client', default=10, type=int,
                    help='client number')
parser.add_argument('--iid_alpha', default=1.0, type=float,
                    help='non iid degree, 0 is iid distribution')
parser.add_argument('--Tpre', default=50, type=int,
                    help='Pretrain rounds on d_0')
parser.add_argument('--Tpripre', default=80, type=int,
                    help='Pretrain rounds on d_i')
parser.add_argument('--Tc', default=60, type=int,
                    help='collaboration learning rounds')
parser.add_argument('--frac', default=0.5, type=float,
                    help='collaboration learning fraction')
parser.add_argument('--cb', default=100, type=int,
                    help='collaboration learning batch')


args = parser.parse_args() #用args.seed,也可以用args.cuda=1设置

#设置随机数种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

#配置一些环境变量
if torch.cuda.is_available():
    args.dev = "cuda:{}".format(str(args.cuda))
else:
    args.dev = "cpu"

if args.dataset == "cifa10":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset_train = datasets.CIFAR10(root = "./data/CIFA10",
                                    transform=transform,
                                    train = True,
                                    download = True)
    dataset_test = datasets.CIFAR10(root = "./data/CIFA10",
                                    transform=transform,
                                    train = False,
                                    download = True)
    models = [CIFA10_VGG(),CIFA10_VGG2(),CIFA10_VGG3()]
    # models = [CIFA10_VGG()]
    for i in models:
        i.to(args.dev)
    # belongs = [0,0,0,0,0,0,0,0,0,0]
    belongs = [0,0,0,0,1,1,1,2,2,2]
elif args.dataset == "cifa100":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
            ])
    dataset_train = datasets.CIFAR100(root = "./data/CIFA100",
                                    transform=transform,
                                    train = True,
                                    download = True)
    dataset_test = datasets.CIFAR100(root = "./data/CIFA100",
                                    transform=transform,
                                    train = False,
                                    download = True)
    models = [CIFA100_VGG(),CIFA100_VGG2(),CIFA100_VGG3()]
    # models = [CIFA10_VGG()]
    for i in models:
        i.to(args.dev)
    # belongs = [0,0,0,0,0,0,0,0,0,0]
    belongs = [0,0,0,0,1,1,1,2,2,2]
elif args.dataset == "mnist":
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
    dataset_train = datasets.MNIST(root = "./data/mnist",
                                    transform=transform,
                                    train = True,
                                    download = True)
    dataset_test = datasets.MNIST(root = "./data/mnist",
                                    transform=transform,
                                    train = False,
                                    download = True)
    models = [Mnist_CNN()]
    # models = [CIFA10_VGG()]
    for i in models:
        i.to(args.dev)
    # belongs = [0,0,0,0,0,0,0,0,0,0]
    belongs = [0,0,0,0,0,0,0,0,0,0]

#将data和label都转化为numpy数组
train_labels = np.array(dataset_train.targets)
test_labels = np.array(dataset_test.targets)
train_datas = []
for i in dataset_train:
    train_datas.append(i[0])
train_datas = np.array(train_datas)

test_datas = []
for i in dataset_test:
    test_datas.append(i[0])
test_datas = np.array(test_datas)

if args.iid_alpha:
    #非独立同分布分配数据
    client_idcs = dirichlet_split_noniid(train_labels, alpha=args.iid_alpha, n_clients=args.client + 1)

#生成公共数据集和测试数据集
d_0_idcs, test_idcs = make_common_dataset(test_labels)
# d_0_idcs.extend(client_idcs[-1])
# test_idcs += client_idcs[-1]
d_0_datas = test_datas[d_0_idcs]
d_0_labels = test_labels[d_0_idcs]

#增加点
d_0_datas = np.append(d_0_datas, train_datas[client_idcs[-1]],axis=0)
d_0_labels = np.append(d_0_labels, train_labels[client_idcs[-1]],axis=0)

data_tensor = torch.from_numpy(d_0_datas)
label_tensor = torch.from_numpy(d_0_labels)
d_0_dataset = TensorDataset(data_tensor, label_tensor)
d_0_loader = DataLoader(d_0_dataset, batch_size=args.batch_size, shuffle=True)

data_tensor = torch.from_numpy(test_datas[test_idcs])
label_tensor = torch.from_numpy(test_labels[test_idcs])
test_dataset = TensorDataset(data_tensor, label_tensor)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)




#新建客户端
clients = []
for i in range(args.client):
    clients.append(Client(train_datas[client_idcs[i]], train_labels[client_idcs[i]], i, args))



#分配模型
for i in range(args.client):
    clients[i].set_model(models[belongs[i]])

try:
    os.makedirs("./checkpoint{}/10".format(args.dataset))
except:
    pass

# if os.path.exists("./checkpoint{}/model1.pth".format(args.dataset)):
#     print("checkpoint already exist")
#     for idx, model in enumerate(models):
#         model.load_state_dict(torch.load("./checkpoint{}/model{}.pth".format(args.dataset, str(idx))))
# else:
#     #d_0上的预训练
#     print(" d_0 pretrain begin ")
#     for idx, model in enumerate(models):
#         model.train()

#         lr = args.lr
#         opt = torch.optim.SGD(model.parameters(), lr = args.lr)
#         loss_fn = torch.nn.functional.cross_entropy
#         threshold = [ args.Tpre/e for e in [2,4,8]]
#         for r in range(args.Tpre):
#             if r in threshold:
#                 lr /= 2
#                 opt = torch.optim.SGD(model.parameters(), lr = lr)
#             print(" training round {}".format(str(r+1)))
#             total_loss = 0
#             cnt = 0
#             for (data, label) in d_0_loader:
#             # for (data, label) in clients[0].data_loader:
#                 data, label = data.to(args.dev), label.to(args.dev)
#                 preds = model(data)
#                 # print(preds[0])
#                 loss = loss_fn(preds, label)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#                 total_loss += loss.item()
#                 cnt += 1
#             total_loss /= cnt
#             print(" loss = {},  accuracy = {}".format(str(total_loss), str(test_model(model, test_loader, args))))
#             torch.save(model.state_dict(),"./checkpoint{}/model{}.pth".format(args.dataset, str(idx)))


#将与训练模型分配到各个客户端
for i in range(len(clients)):
    clients[i].set_para(models[belongs[i]].state_dict())


accs = []
for i in range(args.client):
    accs.append([])

if os.path.exists("./checkpoint{}/{}/0.pth".format(args.dataset, str(args.client))):
    print("no need to local pretrain")
    for i in range(len(clients)):
        clients[i].model.load_state_dict(torch.load("./checkpoint{}/{}/{}.pth".format(args.dataset, str(args.client),str(i))))
        clients[i].para = copy.deepcopy(clients[i].model.state_dict())
else:
    #进入本地训练阶段
    for idx, client in enumerate(clients):
        # client.local_training(test_loader, args.Tpripre)
        for kk in range(80):
            accs[idx].append(client.local_training(test_loader, 1))
        torch.save(client.model.state_dict(),"./checkpoint{}/{}/{}.pth".format(args.dataset, str(args.client),str(client.id)))

print(accs)

exit()
acc2s = []
for i in range(args.client):
    acc2s.append([])
begs = []
for i in range(args.client):
    begs.append(test_model(clients[i].model, test_loader, args))

for e in range(args.Tc):
    print("collaboration learning round {}:".format(str(e+1)))
    select_len = int(args.client*args.frac)
    selected = np.random.choice(np.array(range(args.client)), size=select_len, replace=False)
    # print(selected)
    selected_data_id = np.random.choice(np.array(range(len(d_0_datas))), size=int(len(d_0_datas)*0.5), replace=False)
    knowledges = []
    for id in selected:
        knowledges += clients[id].send_knowledge(d_0_datas[selected_data_id])
    for id in range(args.client):
        acc, acc2 = clients[id].learn_knowledge(d_0_datas[selected_data_id], d_0_labels[selected_data_id], knowledges, test_loader, selected)
        torch.cuda.empty_cache()
        accs[id].append(acc)
        acc2s[id].append(acc2)

print(begs)
print("acc = ")
print(accs)
print("acc2 = ")
print(acc2s)

