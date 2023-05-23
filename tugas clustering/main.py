import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np

dataset = pd.read_csv('heart.csv')
data = dataset.loc[:,['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']]

#6. cluster dengan cluster_val terkecil
cal_val=[]
for i in range(10):
    clustering = KMeans(n_clusters=3, init="random", n_init=1)
    clusters = clustering.fit_predict(data)
    print(f'\n hasil clustering {i}:\n', clusters)
    print('SSE :', clustering.inertia_)
    cal_val.append(clustering.inertia_)
print('\n nilai terkecil : ', min(cal_val))
print(' Pada index ke : ', pd.Series(cal_val).idxmin())

#5 clustering paling berpengaruh dengan k-means dan lakukan analisis cluster dengan sse
# for i in range(10):
#     clustering = KMeans(n_clusters=3, init="random", n_init=1)
#     clusters = clustering.fit_predict(data)
#     print(f'\n hasil clustering {i}:\n', clusters)
#     print('SSE :', clustering.inertia_)


#no 4 clustering single, average, linkage (k=2)
# clustering_single = AgglomerativeClustering(n_clusters=2, linkage='single')
# clusters_single = clustering_single.fit_predict(data)

# clustering_average = AgglomerativeClustering(n_clusters=2, linkage='average')
# clusters_average = clustering_average.fit_predict(data)

# clustering_complete = AgglomerativeClustering(n_clusters=2, linkage='complete')
# clusters_complete = clustering_complete.fit_predict(data)

# print('\n hasil clustering: ')
# print('\n Single : \n', clusters_single)
# print('\n average : \n', clusters_average)
# print('\n complete : \n', clusters_complete)


#no 3 clustering k-means (k=2)
#clustering = KMeans(n_clusters=2, init='random', n_init=1)
#clusters = clustering.fit_predict(data)
#print("Hasil Clustering:\n", clusters)

#no 2. normalisasi minmax
#sc = MinMaxScaler(feature_range=(0, 1))
#data = sc.fit_transform(data)
#print(data)

#no 1
#print(dataset)
