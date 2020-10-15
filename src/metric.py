import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

load_results="../log/test_reuters/"

filename= load_results+"outlier.txt"
outlier_scores=np.loadtxt(filename)

filename= load_results+"label.txt"
label=np.loadtxt(filename)

filename= load_results+'start_M.txt'
start_M=np.loadtxt(filename)

filename= load_results+'mid_M.txt'
mid_M=np.loadtxt(filename)

filename= load_results+'end_M.txt'
end_M=np.loadtxt(filename)

#Metrics

perc=5
k=int(0.01*perc*label.shape[0])

index = np.argsort(outlier_scores)

index=index[::-1][0:k]

total_outlier=0
predicted_outlier=0

for i in range(label.shape[0]):
    if label[i]==1.0:
        total_outlier=total_outlier+1
        for j in range(k):
            if i == index[j]:
                predicted_outlier=predicted_outlier+1

print('Number of outliers in top ',perc,' percent: ', predicted_outlier)
print('Number of outliers: ', total_outlier)
print('Recall score is for top ',perc,' percent is: ', predicted_outlier/total_outlier)

print('AUC score: ',roc_auc_score(label, outlier_scores))

#tSNE
X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(start_M)
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=label)
plt.savefig(load_results+'tsne_start.png')

X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(mid_M)
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=label)
plt.savefig(load_results+'tsne_mid.png')

X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(end_M)
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=label)
plt.savefig(load_results+'tsne_end.png')