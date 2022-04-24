import numpy as np
import pandas as pd
from tqdm import tqdm

trans = pd.read_excel("D:\区块链\实验模型\原始数据\交易数据.xlsx") # 所有交易记录
def create_txt(outputpath,graph_id,direct = False):
    if int(direct) == 0:
        direction = 'from_'
        idx =trans[trans.from_id == graph_id].index.tolist()
        data = trans.iloc[idx, :]
        ## 根据balance生成另外一方的value,fee
        to_value = list(np.array(data['to_balance']) + np.array(data['value']))
        to_fee = list(np.array(data['to_balance']) + np.array(data['fee']))
        from_value = list(np.array(data['from_balance']) - np.array(data['value']))
        from_fee = list(np.array(data['from_balance']) - np.array(data['fee']))
        data['from_value'] = from_value
        data['from_fee'] = from_fee
        data['to_value'] = to_value
        data['to_fee'] = to_fee
        filename_edge = direction + str(graph_id) + '.cites'
        filename_node = direction + str(graph_id) + '.content'
    else:
        direction = 'to_'
        idx = trans[trans.to_id == graph_id].index.tolist()
        data = trans.iloc[idx, :]
        ## 根据balance生成另外一方的value,fee
        to_value = list(np.array(data['to_balance']) + np.array(data['value']))
        to_fee = list(np.array(data['to_balance']) + np.array(data['fee']))
        from_value = list(np.array(data['from_balance']) - np.array(data['value']))
        from_fee = list(np.array(data['from_balance']) - np.array(data['fee']))
        data['from_value'] = from_value
        data['from_fee'] = from_fee
        data['to_value'] = to_value
        data['to_fee'] = to_fee
        filename_edge = direction + str(graph_id) + '.cites'
        filename_node = direction + str(graph_id) + '.content'
    ## 生成边文件
    # msg_edge = data[['from_id','to_id']]
    msg_edge = data[data['from_id'] != data['to_id']][['from_id','to_id']] # 剔除自我交易之后只要id两列
    msg_edge.to_csv(outputpath + filename_edge,sep='\t',index = False,header = None)
    ## 生成节点特征文件
    ID = list(data['from_id']) + list(data['to_id'])
    Balance = list(data['from_balance']) + list(data['to_balance'])
    Value = list(data['from_value']) + list(data['to_value'])
    Fee = list(data['from_fee']) + list(data['to_fee'])
    Label = list(data['from_label']) + list(data['to_label'])
    Count = list(data['count']) + list(data['count'])
    Time = list(data['time_inter']) + list(data['time_inter'])
    df = pd.DataFrame()
    df['ID'] = ID
    df['Balance'] = Balance
    df['Value'] = Value
    df['Fee'] = Fee
    df['Lable'] = Label
    df['Count'] = Count
    df['Time'] = Time
    tree = pd.DataFrame()
    grouped = df.groupby('ID')
    tree['ID'] = list(grouped.groups)
    tree['Balance'] = list(grouped['Balance'].mean())
    tree['Value'] = list(grouped['Value'].mean())
    tree['Fee'] = list(grouped['Fee'].mean())
    tree['Lable'] = list(grouped['Lable'].mean())
    tree['Count'] = list(grouped['Count'].mean())
    tree['Time'] = list(grouped['Time'].mean())
    if len(tree) == 1:
        tree = pd.DataFrame()
    else:
        tree = tree
    tree.to_csv(outputpath + filename_node,sep='\t',index = False,header = None)

# 调用
node = pd.read_excel("D:\区块链\实验模型\原始数据\中心节点.xlsx")
outputpath = 'D:/区块链/实验模型/新增交易时间和交易次数/Bi-GCN/data/tree7/'
node_id = tqdm(list(node['node_ID']))
# graph_id = node['node_ID'][0]
# create_txt(outputpath,graph_id,direct= True)
for graph_id in node_id:
    node_id.set_description("Processing %s" % graph_id)
    create_txt(outputpath,graph_id,direct= True)
    create_txt(outputpath,graph_id,direct= False)

