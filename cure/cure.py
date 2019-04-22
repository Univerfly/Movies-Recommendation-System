import sys
import heapq
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from pyspark import SparkContext,SparkConf
import matplotlib.pyplot as plt
appName = "cure"
import time
start_time = time.time()

def cluster_centroid(x,y,position):
    len_x = len([int(j) for j in str(x).replace('(', '').replace(')', '').split(',')])
    len_y = len([int(j) for j in str(y).replace('(', '').replace(')', '').split(',')])
    x = np.array(position[x])
    y = np.array(position[y])
    output = (x * len_x + y * len_y) / (len_x + len_y)
    output = output.tolist()
    return output
def hierarchical(sample,k,id,heap):
    # heap = []
    # for i in range(len(id)):
    #
    #     for j in range(i+1,len(id)):
    #         dis = distance.euclidean(sample[id[i]],sample[id[j]])
    #         heapq.heappush(heap,[dis,(id[i],id[j])])
    original_k = len(sample)
    clusters = set(id)
    while heap:
        tmp = heapq.heappop(heap)
        cur_pairs = tmp[1]

        if cur_pairs[0] in clusters and cur_pairs[1] in clusters:
            original_k -= 1
            new_data = cluster_centroid(cur_pairs[0], cur_pairs[1], sample)
            sample[cur_pairs] = new_data
            clusters.remove(cur_pairs[0])
            clusters.remove(cur_pairs[1])
            for i in clusters:
                dis = distance.euclidean(sample[i],new_data)
                heapq.heappush(heap, [dis, (cur_pairs, i)])
            clusters.add(cur_pairs)

            if original_k == k:
                return list(clusters)

def find_repre(cluster,n,centroid):
    representatives = []
    max = float("-inf")
    for id in cluster:
        d = distance.euclidean(data[id],centroid)
        if d > max:
            max = d
            first_node = id
    representatives.append(first_node)
    # print position[first_node]
    if n > len(cluster):
        return cluster
    for j in range(n - 1):
        max = float("-inf")
        for id in cluster:
            if id in representatives:
                continue
            min_dist = float("inf")
            for representative in representatives:
                d = distance.euclidean(data[id],data[representative])
                if d < min_dist:
                    min_dist = d

            if min_dist > max:
                candidate = id
                max = min_dist
        representatives.append(candidate)
    return representatives

if __name__ == "__main__":
    # conf = SparkConf().setAppName(appName).setMaster("local[*]")
    # sc = SparkContext(conf=conf)

    k = int(sys.argv[1])
    sample_filename = sys.argv[2]
    data_filename = sys.argv[3]
    n = int(sys.argv[4])
    factor = float(sys.argv[5])

    with open(sample_filename) as f:
        data = {}
        id = []
        metadata = []
        for line in f:
            row = line.strip('\n').split(',')
            id.append(int(row[0]))
            metadata.append([float(i) for i in row[1:]])
            data[int(row[0])] = []
            data[int(row[0])] = ([float(i) for i in row[1:]])
    f.close()

    with open(data_filename) as f:
        position = []
        for line in f:
            row = line.strip('\n').split(',')
            position.append([float(i) for i in row])


    f.close()

    heap = []
    dis = cdist(metadata, metadata, 'euclidean')
    for i in range (len(dis)):
        for j in range (i+1,len(dis[0])):
            heapq.heappush(heap, [dis[i][j], (id[i], id[j])])
    # hierarchical
    centroid = []
    results = hierarchical(data,k,id,heap)
    clusters = []
    for i in range(len(results)):
        centroid.append(data[results[i]])
        cur_class = sorted([int(j) for j in str(results[i]).replace('(', '').replace(')', '').split(',')])
        clusters.append(cur_class)
    print clusters

    representatives = []
    i = 0
    for cluster in clusters:
        tmp = find_repre(cluster, n, centroid[i])
        # print tmp
        new_tmp = []
        for id in tmp:
            pos = []
            for j in range(len(centroid[0])):
                pos.append(data[id][j] + ((centroid[i][j] - data[id][j]) * factor) )
            new_tmp.append(pos)
        i += 1
        representatives.append(new_tmp)
    print representatives

    results = {}
    for i in range(len(position)):
        min = float("inf")
        for j in range(k):
            min_dist = float("inf")
            for m in representatives[j]:
                d = distance.euclidean(position[i], m)
                if d < min_dist:
                    min_dist = d
            if min_dist < min:
                min = min_dist
                c = j
        if results.has_key(c):
            results[c].append(i)
        else:
            results[c] = []
            results[c].append(i)

print("--- %s seconds ---" % (time.time() - start_time))
lenL = []
for i in range(k):
    print("Cluster %r:%r"%(i+1,results[i]))
    lenL.append(len(results[i]))
print(lenL)
plt.title("Cluster Number")
plt.xlabel("Clusters")
plt.ylabel("Number")
plt.bar(range(20),lenL)
plt.xticks(np.arange(0, 20, 1))
plt.savefig("Cluster Result.jpg")



