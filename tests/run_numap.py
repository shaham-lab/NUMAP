from src.numap import NUMAP
from src.numap.utils import *
from src.load_data import get_2circles
import umap
from src.numap.metrics import Metrics

numap = NUMAP(
    encoder=None,  # nn.Module, None for default
    n_neighbors=10,
    se_dim=5,
    se_neighbors=10,
    lr=1e-3,
    epochs=10,
    batch_size=64,
    use_se=True,
    use_residual_connections=True,
    learn_from_se=True,
    use_grease=True,
)

# choose data
X, y, dataset_name = get_2circles()
# fit the model
numap.fit(X)
embedding = numap.transform(X)

# get UMAP embedding
reducer = umap.UMAP()
embedding_umap = reducer.fit_transform(X)

# compute KNN accuracy
knn_k = 5
knn_acc_numap = Metrics.compute_knn_acc(embedding, y, knn_k)
knn_acc_umap = Metrics.compute_knn_acc(embedding_umap, y, knn_k)
print(f'KNN accuracy for NUMAP: {knn_acc_numap}')
print(f'KNN accuracy for UMAP: {knn_acc_umap}')

# compute IOU accuracy
iou_k = 10
iou_acc_numap = Metrics.compute_iou_acc(X, embedding, iou_k)
iou_acc_umap = Metrics.compute_iou_acc(X, embedding_umap, iou_k)
print(f'IOU accuracy for NUMAP: {iou_acc_numap}')
print(f'IOU accuracy for UMAP: {iou_acc_umap}')

# compute Silhouette score
silhouette_score_numap = Metrics.compute_silhouette_score(embedding, y)
silhouette_score_umap = Metrics.compute_silhouette_score(embedding_umap, y)
print(f'Silhouette score for NUMAP: {silhouette_score_numap}')
print(f'Silhouette score for UMAP: {silhouette_score_umap}')

# plot the embedding
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, s=1)
ax.set_title('NUMAP')
ax.set_xticks([])
ax.set_yticks([])
ax = fig.add_subplot(122)
scatter = ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=y, s=1)
ax.set_title('UMAP')
ax.set_xticks([])
ax.set_yticks([])
# add knn and map accuracy to the plot
fig.text(0.3, 0.05,
         f'KNN: {knn_acc_numap:.2f}, IOU: {iou_acc_numap:.2f}, Silhouette: {silhouette_score_numap:.2f}',
         ha='center', fontsize=12)
fig.text(0.71, 0.05, f'KNN: {knn_acc_umap:.2f}, IOU: {iou_acc_umap:.2f}, Silhouette: {silhouette_score_umap:.2f}',
         ha='center', fontsize=12)

# plt.savefig(f'figures/{dataset_name}/{dataset_name}_16.png')
plt.show()
