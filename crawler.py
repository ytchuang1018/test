import requests
import sys
from io import StringIO
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

url="	https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data"
res = requests.get(url)
TESTDATA=StringIO(res.text)
df = pd.read_csv(TESTDATA, sep=",")

df = df.fillna(0)
x = df.drop(["證券代號", "證券名稱"],axis=1)

kmeans = KMeans(n_clusters= 10, random_state=0)
label = kmeans.fit_predict(x)
print(label)
 
x1 = x.to_numpy()
kmeans.fit(x1)
y_kmeans = kmeans.predict(x1)

u_labels = np.unique(label)
for i in u_labels:
  plt.scatter(x1[label == i , 0] , x1[label == i , 1] , label = i)
plt.legend()

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black')