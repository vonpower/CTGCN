import numpy as np
import scipy.sparse as sp
from DGLmodel import *
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from numpy import mean
from tqdm import tqdm
import random
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normalize(mx):#
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx


def ReadTree(path,dataset):

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))# 导入节点特征与标签文件

    if len(idx_features_labels) != 0:
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 取特征feature
        features = normalize(features)  # 对特征做了归一化的操作
        features = torch.FloatTensor(np.array(features.todense())) # 转为tensor格式

        # build graph
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 导入edge的数据

        if features.shape[0] > 2:
            u = torch.tensor([0] * edges_unordered.shape[0], dtype=torch.int32)  # 改变节点标签
            v = torch.tensor(list(range(1, 1 + edges_unordered.shape[0])), dtype=torch.int32)
        else:
            u = torch.tensor([0], dtype=torch.int32)  # 改变节点标签
            v = torch.tensor([1], dtype=torch.int32)
        g = dgl.graph((u, v),idtype=torch.int32)
        g = dgl.add_self_loop(g)
        g.ndata['original'] = features # 加入节点原始特征

    else:
        g = torch.tensor([0], dtype=torch.int32)

    # 图形可视化
    # nx.draw(g.to_networkx())  # 将图转为networkx形式
    # plt.show()

    return g

# 读取原始数据
labelfile_path = 'D:\区块链\实验模型\原始数据\中心节点.xlsx'
tree_path = "./data/tree2/"
def DataSet(labelfile_path,tree_path):
    ## 读取标签和中心节点索引
    labelfile = pd.read_excel(labelfile_path)
    indexs = labelfile['node_ID']
    labels = labelfile['label']

    # 返回模型输入值
    g_f = []
    g_t = []
    for i, (index, label) in enumerate(zip(indexs, labels)):
        dataset1 = "from_" + str(index)
        dataset2 = "to_" + str(index)
        g1 = ReadTree(tree_path, dataset1)
        g2 = ReadTree(tree_path, dataset2)
        g_f.append(g1)
        g_t.append(g2)

    # # 划分测试集和训练集
    # num = list(range(len(indexs)))
    # random.shuffle(num)  # 打乱顺序
    # train_indexs, test_indexs, train_labels, test_labels, train_g_fs, test_g_fs,train_g_ts, test_g_ts = [], [], [], [], [], [],[],[]
    # for n in num[:int(0.8 * len(num))]:
    #     train_indexs.append(indexs[n])
    #     train_labels.append(labels[n])
    #     train_g_fs.append(g_f[n])
    #     train_g_ts.append(g_t[n])
    # for m in num[int(0.8 * len(num)):]:
    #     test_indexs.append(indexs[m])
    #     test_labels.append(labels[m])
    #     test_g_fs.append(g_f[m])
    #     test_g_ts.append(g_t[m])

    # return train_indexs, test_indexs, train_labels, test_labels, train_g_fs, test_g_fs,train_g_ts, test_g_ts
    return indexs,labels,g_f,g_t

class MyDataset(Dataset):
    def __init__(self, indexs,labels,g_f,g_t):
        # 转化为tensor格式
        self.indexs = indexs
        self.y = torch.LongTensor(labels)
        self.g_f = g_f
        self.g_t = g_t
        self.len = len(indexs)

    def __getitem__(self, idx):
        return self.indexs[idx],self.y[idx],self.g_f[idx],self.g_t[idx]

    def __len__(self):
        return self.len

## 生成dataset与dataloader
# train_indexs, test_indexs, train_labels, test_labels, train_g_fs, test_g_fs,train_g_ts, test_g_ts  = DataSet(labelfile_path,tree_path)
indexs,labels,g_f,g_t  = DataSet(labelfile_path,tree_path)
train_dataset = MyDataset(indexs,labels,g_f,g_t)
# train_dataset = MyDataset(train_indexs,train_labels,train_g_fs,train_g_ts)
# test_dataset = MyDataset(test_indexs,test_labels,test_g_fs,test_g_ts)
# train_loader = DataLoader(dataset=train_dataset,batch_size=5,shuffle=False,num_workers=0)# num_workers多线程

# 模型训练
model = Net(nfeat=3,nhid=32,nout=64,dropout=0) #init# 初始化参数
criterion = torch.nn.MSELoss(size_average=False)# 损失函数MSE
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)# 优化器

if __name__ == '__main__':
    for i in tqdm(range(len(train_dataset))):  # batch_size=1
        index,label,g_f,g_t = train_dataset[i]
        output = model(g_f,g_t)
        print(index)


# # 初始化tensorboard    config.tb_dir为保存tensorboard文件的路径
# tb_writer = SummaryWriter('./logs')
#
#
# model.train()
# if __name__ == '__main__':
#     for epoch in range(100):
#         correct_0, correct_1, err0_1, err1_0 = 0, 0, 0, 0
#         loss = []
#         for i in tqdm(range(len(train_dataset))): # batch_size=1
#             index,label,g_f,g_t = train_dataset[i]
#             optimizer.zero_grad()
#             output = model(g_f,g_t)
#             ### 判断分为哪一类
#             predicted = torch.max(output.data,1)[1].item()
#             if predicted == label.item():
#                 if predicted == 0:
#                     correct_0 += 1
#                 else:
#                     correct_1 += 1
#             else:
#                 if predicted == 0:
#                     err0_1 += 1
#                 else:
#                     err1_0 += 1
#             # 计算损失
#             loss_g = F.nll_loss(output, label.view(1))
#             loss.append(loss_g.item())
#             loss_g.backward()
#             optimizer.step()
#
#         loss_train = mean(loss) # 计算损失
#         acc_train = (correct_0+correct_1)/len(train_dataset) # 计算准确率
#         pre_train = correct_1/(correct_1+err1_0) # 精确率
#         recall_train = correct_1/(correct_1+err0_1) # 召回率
#         F1_train = (2*pre_train*recall_train)/(pre_train+recall_train) # F1得分
#         print('Epoch :{:.0f}   Loss:{:.4f}	 Accuracy:{:.3f}  F1:{:.3f}'.format(epoch,loss_train,acc_train,F1_train))
#
#         # # tensorboard可视化
#         tb_writer.add_scalars('Train-Loss', {'Train': loss_train}, epoch)
#         tb_writer.add_scalars('Train-Acc', {'Train': acc_train}, epoch)
#         tb_writer.add_scalars('Train-F1', {'Train': F1_train}, epoch)
#
#         model.eval()
#         correct_val_0, correct_val_1, err0_val_1, err1_val_0 = 0, 0, 0, 0
#         loss_val = []
#         for i in tqdm(range(len(test_dataset))):  # batch_size=1
#             index, label, g_f, g_t = test_dataset[i]
#             output = model(g_f,g_t)
#             ### 判断分为哪一类
#             predicted = torch.max(output.data, 1)[1].item()
#             if predicted == label.item():
#                 if predicted == 0:
#                     correct_val_0 += 1
#                 else:
#                     correct_val_1 += 1
#             else:
#                 if predicted == 0:
#                     err0_val_1 += 1
#                 else:
#                     err1_val_0 += 1
#             # 计算损失
#             loss_g = F.nll_loss(output, label.view(1))
#             loss_val.append(loss_g.item())
#
#         loss_test = mean(loss_val)  # 计算损失
#         acc_test = (correct_val_0 + correct_val_1) / len(test_dataset)  # 计算准确率
#         pre_test = correct_val_1 / (correct_val_1 + err1_val_0)  # 精确率
#         recall_test = correct_val_1 / (correct_val_1 + err0_val_1)  # 召回率
#         F1_test = (2 * pre_test * recall_test) / (pre_test + recall_test)  # F1得分
#
#         print(
#             'Epoch :{:.0f}   test_Loss:{:.4f}	 test_Accuracy:{:.3f}  test_F1:{:.3f}'.format(epoch, loss_test,
#                                                                                                acc_test, F1_test))
#
#         # # tensorboard可视化
#         tb_writer.add_scalars('Test-Loss', {'Test': loss_test}, epoch)
#         tb_writer.add_scalars('Test-Acc', {'Test': acc_test}, epoch)
#         tb_writer.add_scalars('Test-F1', {'Test': F1_test}, epoch)





