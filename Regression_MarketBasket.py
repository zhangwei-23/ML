# coding utf-8
import pandas as pd
import numpy as np
from efficient_apriori import apriori

#Header = None
dataset = pd.read_csv('D:\Learning\Data Engine\Data_Engine_with_Python-master\L4\MarketBasket\Market_Basket_Optimisation.csv',header=None)
print(dataset.head(3))
print(dataset.shape)

# 数据放到transaction中
transactions = []
for i in range(0, dataset.shape[0]):
    temp = []
    for j in range(0,20):
        if str(dataset.values[i,j]) != 'nan':
            temp.append(str(dataset.values[i,j]))
    transactions.append(temp)

# 挖掘频繁项集与关联规则
items,rules = apriori(transactions, min_support=0.04, min_confidence=0.3)
print('频繁项集：', items)
print('关联规则：', rules)

