import sys
from numpy import nan
import pandas as pd
from scipy.spatial.distance import cdist

movieId = sys.argv[1]
data_file = pd.read_csv("movies_metadata.csv",index_col=None)
metadata = data_file[['id','title']]
print(metadata[metadata['id']==movieId])

clusters = []
with open('Cluster.txt','r') as filereader:
    for i in filereader:
        each = []
        row = i.strip('[').strip(']\n').split(',')
        for j in row:
            each.append(eval(j))
        clusters.append(each)

data = []
with open('metadata.txt','r') as filereader:
    for i in filereader:
        eachdata = []
        p = i.strip('\n').split(',')
        for j in range(28):
            if j == 27:
                eachdata.append(p[j])
            else:
                eachdata.append(float(p[j]))
        data.append(eachdata)

clusterID = len(clusters)
for i in range(len(clusters)):
    for p in clusters[i]:
        if eval(p) == movieId:
            clusterID = i

if clusterID == len(clusters):
    print("This Movie ID Not Found!")
else:
    #print("This Movie Id in cluster %r"%(clusterID))
    print("---------------------------------------------")
    searchInfo = []
    base = []
    for i in clusters[clusterID]:
        for p in data:
            if eval(i) == movieId and eval(i)== eval(p[27]):
                searchInfo = p
            if eval(i) == eval(p[27]) and eval(i) != movieId:
                base.append(p)
    dis = []
    for i in base:
        dist = cdist([searchInfo[0:27]],[i[0:27]],metric="euclidean")
        tem = list(dist[0])
        dis.append([tem[0],i[27]])
    dis.sort(key = lambda x:x[0])
    print("Top 5 Recommendation:")
    print("---------------------------------------------")
    for i in range(5):
        print(metadata[metadata['id']==eval(dis[i][1])])

