import numpy as np
import scipy.sparse as sp
from DGLmodel import *




def ReadTree(path,dataset):

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))# 导入节点特征与标签文件

    if len(idx_features_labels) != 0:
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 取特征feature
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

# 单一
path= "./data/tree2/"
index = 61101
dataset1 = "from_"+str(index)
dataset2="to_"+str(index)
g1 = ReadTree(path,dataset1)
g2 = ReadTree(path,dataset2)
print(g2)
# 模型训练
model = Net(nfeat=5,nhid=32,nout=32,dropout=0) #init
output = model(g1,g2) #forward
print(output)


