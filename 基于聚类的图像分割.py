#建立工程并导入sklearn包
#导入sklean相关包
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

#加载图片并进行预处理
def loadData(filepath):
    f=open(filepath,'rb')
    data=[]
    img=image.open(f)
    m,n=img.size
    for i in range(m):
        for j in range(n):
            x,y,z=img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return np.mat(data),m,n
imgData,row,col=loadData('C:/Users/Administrator/Desktop/zx.bmp')
label=KMeans(n_clusters=3).fit_predict(imgData)

label=label.reshape([row,col])
pic_new=image.new('L',(row,col))

for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))

pic_new.save('zx-new.jpg','JPEG')
