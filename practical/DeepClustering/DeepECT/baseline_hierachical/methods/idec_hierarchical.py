import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.cluster import KMeans
from practical.DeepClustering.DeepECT.ect.utils.evaluation.dendrogram_purity import *
import time
from pathlib import Path
from ect.methods.DEC import DEC
from ect.utils.evaluation import cluster_acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from practical.DeepClustering.DeepECT.ect.utils.dec_utils import (
    dendrogram_purity_tree_from_clusters,
)
import random
from practical.DeepClustering.DeepECT.ect.utils.deterministic import set_random_seed
from copy import deepcopy


def run_experiment(
    data, ground_truth_labels, new_seed, n_clusters, autoencoder, device="cpu"
):
    pt_data = torch.from_numpy(data).to(device)
    pt_init_sample = torch.from_numpy(data[range(0, 70000), :])

    set_random_seed(new_seed)
    train = torch.utils.data.TensorDataset(pt_data)
    train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

    n_features = data.shape[1]
    # Same loss as in the DEC implementation
    # ae_reconstruction_loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    # ae_module = stacked_ae(n_features, [500, 500, 2000, 10],
    #                        weight_initalizer=torch.nn.init.xavier_normal_,
    #                        activation_fn=lambda x: F.relu(x),
    #                        loss_fn=None,
    #                        optimizer_fn=None)

    # model_data = torch.load(ae_model_path, map_location='cpu')
    # ae_module.load_state_dict(model_data)
    # ae_module = ae_module.cuda()

    node_data = None
    for batch_data in torch.utils.data.DataLoader(
        pt_init_sample, batch_size=256, shuffle=True
    ):
        # embedded_batch_np = ae_module.forward(batch_data.cuda())[0].detach().cpu().numpy()
        embedded_batch_np = (
            autoencoder.encode(batch_data.to(device)).detach().cpu().numpy()
        )
        if node_data is None:
            node_data = embedded_batch_np
        else:
            node_data = np.concatenate([node_data, embedded_batch_np], 0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(node_data)
    init_centers = kmeans.cluster_centers_

    # Initialize cluster centers based on a smaller sample
    # cluster_module = DEC(init_centers).cuda()
    cluster_module = DEC(init_centers)
    autoencoder = deepcopy(autoencoder)
    optimizer = torch.optim.Adam(
        list(autoencoder.parameters()) + list(cluster_module.parameters()), lr=0.001
    )

    def evaluate(train_round_idx, ae_module, cluster_module):
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pt_data), batch_size=256
        )

        pred_labels = np.zeros(pt_data.shape[0], dtype=int)
        index = 0
        n_batches = 0
        for batch_data in test_loader:
            # batch_data = batch_data[0].cuda()
            batch_data = batch_data[0].to(device)
            n_batches += 1
            batch_size = batch_data.shape[0]
            embedded_data = autoencoder.encode(batch_data)
            reconstructed_data = autoencoder.decode(embedded_data)
            labels = cluster_module.prediction_hard_np(embedded_data)
            pred_labels[index : index + batch_size] = labels
            index = index + batch_size
        pred_tree = dendrogram_purity_tree_from_clusters(
            cluster_module, pred_labels, "single"
        )
        pred_tree2 = dendrogram_purity_tree_from_clusters(
            cluster_module, pred_labels, "complete"
        )
        lp = leaf_purity(pred_tree, ground_truth_labels)
        leaf_purity_value_single = lp[0]
        lp = leaf_purity(pred_tree2, ground_truth_labels)
        leaf_purity_value_complete = lp[0]
        dp_value_single = dendrogram_purity(pred_tree, ground_truth_labels)
        dp_value_complete = dendrogram_purity(pred_tree2, ground_truth_labels)
        return (
            dp_value_single,
            dp_value_complete,
            leaf_purity_value_single,
            leaf_purity_value_complete,
        )

    evaluate("init", autoencoder, cluster_module)

    n_rounds = 10000
    train_round_idx = 0
    while True:  # each iteration is equal to an epoch
        for batch_data in train_loader:
            train_round_idx += 1
            if train_round_idx > n_rounds:
                break
            # batch_data = batch_data[0].cuda()
            batch_data = batch_data[0].to(device)

            ae_loss, embedded_data, reconstruced_data = autoencoder.loss(
                [None, batch_data], torch.nn.MSELoss(), device
            )
            # embedded_data = autoencoder.encode(batch_data)
            # reconstruced_data = autoencoder.decode(embedded_data)
            # ae_loss = ae_reconstruction_loss_fn(batch_data, reconstruced_data)

            cluster_loss = cluster_module.loss_dec_compression(embedded_data)
            loss = cluster_loss + 0.1 * ae_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_round_idx % 2000 == 0:
                print(evaluate(train_round_idx, autoencoder, cluster_module))
        else:  # For else is being executed if break did not occur, we continue the while true loop otherwise we break it too
            continue
        break  # Break while loop here

    # Write last evaluation

    (
        dp_value_single,
        dp_value_complete,
        leaf_purity_value_single,
        leaf_purity_value_complete,
    ) = evaluate("", autoencoder, cluster_module)
    return (
        dp_value_single,
        dp_value_complete,
        leaf_purity_value_single,
        leaf_purity_value_complete,
    )
