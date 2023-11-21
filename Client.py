import torch
from torch.utils.data import TensorDataset, DataLoader
import copy
from Common import *
import torch.nn.functional as F

class Client:
    def __init__(self, datas, labels, id, args) -> None:
        self.id = id
        self.args = args
        data_tensor = torch.from_numpy(datas)
        label_tensor = torch.from_numpy(labels)
        dataset = TensorDataset(data_tensor, label_tensor)
        self.data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        self.rep = []
        for i in range(args.client):
            self.rep.append(0)

    def set_model(self, model):
        self.model = model

    def set_para(self, para):
        self.para = copy.deepcopy(para)

    def local_training(self, test_loader, epoch):
        print("client {} begin train...".format(self.id))
        self.model.load_state_dict(self.para)
        # self.model.to(self.args.dev)
        self.model.eval()
        threshold = [ self.args.Tpripre/e for e in [2,4,8]]
        lr = self.args.lr/10
        opt = torch.optim.SGD(self.model.parameters(), lr = lr)
        loss_fn = torch.nn.functional.cross_entropy
        total_loss = 0
        cnt = 0
        # test_model(self.model, test_loader, self.args)
        for r in range(epoch):
            if r in threshold:
                lr /= 2
                opt = torch.optim.SGD(self.model.parameters(), lr = lr)
            # print(" training round {}".format(str(r+1)))
            for (data, label) in self.data_loader:
                data, label = data.to(self.args.dev), label.to(self.args.dev)
                preds = self.model(data)
                loss = loss_fn(preds, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                cnt += 1
            total_loss /= cnt
        acc = test_model(self.model, test_loader, self.args)
        print("training complete, loss = {},  accuracy = {}".format(str(total_loss), str(acc)))
        self.para = copy.deepcopy(self.model.state_dict())
        return acc
        
    def send_knowledge(self, datas):
        self.model.load_state_dict(self.para)
        self.model.eval()
        datas = torch.from_numpy(datas)
        datas = datas.to(self.args.dev)
        preds = self.model(datas)
        return preds.tolist()
    
    def cal_reputation(self, datas, labels, ks, ids):
        Rel = []
        for i in range(len(ks)):
            Rel.append(0) 
        loss_fn = torch.nn.functional.cross_entropy
        # datas = datas.to(self.args.dev)
        # self.model.eval()
        preds = self.send_knowledge(datas)
        # datas = datas.to("cpu")
        preds = torch.tensor(preds)
        for i, pred in enumerate(preds):
            for j in range(len(ids)):
                res = 1/(0.9*loss_fn(F.softmax(pred, dim=0), F.softmax(ks[j * len(labels) + i], dim=0))+0.1*loss_fn(F.softmax(ks[j * len(labels) + i], dim=0), F.softmax(pred, dim=0)) )
                Rel[j * len(labels) + i] = res
        for i in range(len(ids)):
            res = 0
            for j in range(len(datas)):
                res += Rel[i*len(datas)+j]
            self.rep[ids[i]] = self.rep[ids[i]] * 1/2 + res * 1/2
        return Rel
        

    def learn_knowledge(self, datas, labels, ks, test_loader, ids):
        kd_fun = torch.nn.KLDivLoss(reduce=True)
        self.model.load_state_dict(self.para)
        self.model.eval()
        # datas = torch.from_numpy(datas)
        # datas = datas.to(self.args.dev)
        labels = torch.from_numpy(labels)
        # labels = labels.to(self.args.dev)

        ks = torch.tensor(ks)
        # ks = ks.to(self.args.dev)
        lr = self.args.lr/10
        opt = torch.optim.SGD(self.model.parameters(), lr = lr)
        rel = self.cal_reputation(datas, labels, ks, ids)
        datas = torch.from_numpy(datas)
        self.model.train()
        for r in range(1):
            beg = 0
            for ed in range(self.args.cb, len(datas), self.args.cb):
                data = datas[beg: ed]
                label = labels[beg: ed]
                t = ks[beg: ed]
                # t = [0] * self.args.cb
                

                # for i in range(len(t)):
                #     for j in range(1, len(ids)):
                #         t[i] += ks[i + j*len(datas)]
                #     t[i] /= len(datas)

                client_rep_sum = 0.0
                for i in ids:
                    client_rep_sum += self.rep[i]
                for i in range(beg, ed):
                    total = self.rep[ids[0]]/client_rep_sum + rel[i]
                    for j in range(1, len(ids)):
                        total += rel[j*len(datas) + i] + self.rep[ids[j]]/client_rep_sum
                    temp = (rel[i] + self.rep[ids[0]]/client_rep_sum) / total
                    t[i - beg] = ((rel[i] + self.rep[ids[0]]/client_rep_sum)/total).item() * ks[i]
                    for j in range(1, len(ids)):
                        temp += (rel[j*len(datas) + i] + self.rep[ids[j]]/client_rep_sum) / total
                        t[i - beg] = t[i - beg] + ((rel[j*len(datas) + i] + self.rep[ids[j]]/client_rep_sum) / total).item() * ks[j*len(datas) + i]

                # t = torch.tensor(t)
                label = label.to(self.args.dev)
                data = data.to(self.args.dev)
                t = t.to(self.args.dev)
                s = self.model(data)
                s_max = F.log_softmax(s / self.args.T, dim=1)
                s_max = s_max.to(self.args.dev)
                t_max = F.softmax(t / self.args.T, dim=1)
                t_max = t_max.to(self.args.dev)
                loss_kd = kd_fun(s_max, t_max) ##KL散度,实现为logy-x，输入第一项必须是对数形式
                loss_clc = F.cross_entropy(s, label) ##分类loss
                loss = (1 - self.args.lamda) * loss_clc + self.args.lamda * self.args.T * self.args.T * loss_kd

                # loss = self.args.T *self.args.T * loss_kd
                opt.zero_grad()
                loss.backward()
                opt.step()


                beg = ed
        acc = test_model(self.model, test_loader, self.args)
        print("client {} : collaboration learning accuracy = {}".format(str(self.id), str(acc)))
        self.para = copy.deepcopy(self.model.state_dict())
        acc2 = self.local_training(test_loader, 5)
        return acc, acc2
                



        