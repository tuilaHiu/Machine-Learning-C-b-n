from neighbors import process_data
import matplotlib.pyplot as plt
clusters=10
path='Data'
acc=[]
for i in range(1,50):
    acc.append(process_data(path=path,no_clusters=clusters,n_neighbors=i))
plt.plot(acc)
plt.title(f'Number of cluster: {clusters}')
plt.xlabel('neighbors')
plt.ylabel('Accuracy')
plt.yticks([max(acc)])
plt.show()