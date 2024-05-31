from clustpy.deep import IDEC
import numpy as np
from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.cluster import AgglomerativeClustering
import operator
import itertools
from practical.DeepClustering.DeepECT.ect.utils.evaluation.dendrogram_purity import *




def run_idec_hierarchical(data, ground_truth, seed, n_clusers, autoencoder, device="cpu"):

    idec = IDEC(n_clusers, random_state=np.random.RandomState(seed), autoencoder=autoencoder, clustering_epochs=50)
    idec.fit(data)
    labels_pred = idec.labels_
    print(unsupervised_clustering_accuracy(ground_truth, labels_pred))


    pred_tree = dendrogram_purity_tree_from_clusters(
        idec.cluster_centers_, labels_pred, "single"
    )
    pred_tree2 = dendrogram_purity_tree_from_clusters(
        idec.cluster_centers_, labels_pred, "complete"
    )
    lp = leaf_purity(pred_tree, ground_truth)
    leaf_purity_value_single = lp[0]
    lp = leaf_purity(pred_tree2, ground_truth)
    leaf_purity_value_complete = lp[0]
    dp_value_single = dendrogram_purity(pred_tree, ground_truth)
    dp_value_complete = dendrogram_purity(pred_tree2, ground_truth)
    print(dp_value_single)
    print(dp_value_complete)
    print(leaf_purity_value_single)
    print(leaf_purity_value_complete)
    return (
        dp_value_single,
        dp_value_complete,
        leaf_purity_value_single,
        leaf_purity_value_complete,
    )


def dendrogram_purity_tree_from_clusters(centers, pred_labels, linkage="single"):

    clustering = AgglomerativeClustering(compute_full_tree=True, linkage=linkage).fit(centers)

    grouped_ids = {k: [x[0] for x in v] for k, v in
                   itertools.groupby(sorted(enumerate(pred_labels), key=operator.itemgetter(1)), key=operator.itemgetter(1))}

    def map_tree_rec(node):
        if node.is_leaf:
            node.dp_ids = grouped_ids.get(node.dp_ids[0], [])
        else:
            map_tree_rec(node.left_child)
            map_tree_rec(node.right_child)

    tree = to_dendrogram_purity_tree(clustering.children_)
    map_tree_rec(tree)
    return tree