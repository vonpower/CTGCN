from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch


class From_ToGCN(nn.Module):
    def __init__(self, nfeat, nhid, nout,dropout):
        super(From_ToGCN, self).__init__()
        self.conv1 = GraphConv(nfeat, nhid)  # 定义第一层图卷积
        self.conv2 = GraphConv(nhid+nfeat, nout)  # 定义第二层图卷积
        self.dropout = dropout

    def forward(self, g):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        x = g.ndata['original']  # 初始节点特征
        # 执行图卷积和激活函数
        H1 = F.relu(self.conv1(g, x))
        x = torch.cat((H1, x), dim=1)  # 拼接
        x = F.dropout(x, self.dropout, training=self.training)  # 防止过拟合
        H2 = F.relu(self.conv2(g, x))
        x = torch.cat((H1, H2), dim=1)
        g.ndata['train'] = x  # 将训练后特征赋予到图的节点

        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'train')  # [n, hidden_dim]
        return hg  # [n, n_classes]

class Net(nn.Module):
    def __init__(self,nfeat, nhid, nout, dropout):
        super(Net, self).__init__()
        self.From_ToGCN = From_ToGCN(nfeat, nhid, nout, dropout)
        self.fc =nn.Linear((nhid+nout)*2,2)
        self.fcsingal = nn.Linear(nhid+nout,2)

    def forward(self,g1,g2):
        if len(g1) == 1:
            x = self.From_ToGCN(g2)
            x = self.fcsingal(x)
        if len(g2) == 1:
            x = self.From_ToGCN(g1)
            x = self.fcsingal(x)
        if len(g1) != 1 and len(g2) != 1:
            From_x = self.From_ToGCN(g1)
            To_x = self.From_ToGCN(g2)
            x = torch.cat((From_x,To_x), dim=1)# 将From图与To图特征进行合并
            x = self.fc(x)

        # 分类器
        x = F.log_softmax(x, dim=1)
        return x