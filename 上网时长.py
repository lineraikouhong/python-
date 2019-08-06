#DBSCAN-学生校园网使用统计
#建立工程。导入sklearn相关包
import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
#读入数据并进行处理，读取每条数据中的mac地质，开始上网时间，上网时长 mac2id是一个字典：key是mac的地址 value是对应的mac地址的上网时长及开始上网时间
mac2id=dict()
onlinetimes=[]
f=open('TestData.txt',encoding='utf-8')
for line in f:
    mac=line.split(',')[2]
    onlinetime=int(line.split(',')[6])
    starttime=int(line.split(',')[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac]=len(onlinetimes)
        onlinetimes.append((starttime,onlinetime))
    else:
        onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]
real_X=np.array(onlinetimes).reshape((-1,2))
#上网时间聚类，创建DBSCAN算法实践，并进行训练，获得标签：
#调用DBSCAN方法进行训练，labels为每一个数据的簇标签。
#打印数据被记上的标签，计算标签为-1，即噪声数据的比例。
#计算簇的个数并打印，评价聚类效果
#打印各簇标号以及各簇内数据
X=real_X[:,0:1]

db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X)
labels=db.labels_

print('Labels:')
print(labels)
ratio=len(labels[labels[:]==-1])/len(labels)
print('Nosie ratio:',format(ratio,'.2%'))

n_clusters_=len(set(labels))-(1 if -1 in labels else 0)

print('Estimated number of clusters: %d'% n_clusters_)
print('Silhouette coefficient:%0.3f'% metrics.silhouette_score(X,labels))

for i in range(n_clusters_):
    print('Cluster',i,':')
    print(list(X[labels==i].flatten()))
#输出标签，查看结果
plt.hist(X,24)
plt.show()
