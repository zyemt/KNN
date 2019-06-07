from sklearn import datasets
from scipy.spatial import distance
import operator
import numpy as np
from sklearn .model_selection import train_test_split
from sklearn.metrics import  accuracy_score

def etc(x,y):
    return distance.euclidean(x,y)#计算欧氏距离

def findclosest(x,x_train,y_train,k):
    count1 ={}
    grp = []
    for i in range(len(x_train)):#对距离排序
        dist = etc(x,x_train[i])
        grp.append(dist)
    grp = np.array(grp)
    sortedDistance = grp.argsort()#根据大小获得列表元素下标的排序

    for i in range(k):
        if y_train[sortedDistance[i]] in count1:
            count1[y_train[sortedDistance[i]]] += 1
        else:
            count1[y_train[sortedDistance[i]]] = 1
    count2 = sorted(count1.items(),key = operator.itemgetter(1),reverse = True)#分类
    return count2[0][0]

def calculate(test_data,train_data,y_train,k):
    prediction = []
    for x in test_data:
        res = findclosest(x,train_data,y_train,k)
        prediction.append(res)
    return prediction
def final():
    k =3
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)#x_train 数据特征 y_train 数据得分类
    prediction = calculate(x_test,x_train,y_train,k)
    print(accuracy_score(y_test,prediction))#计算准确率

final()
