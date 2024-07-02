import os
import sys
from ray import tune, train
import nevergrad as ng
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search import ConcurrencyLimiter

sys.path.append(os.getcwd())

from practical.DeepClustering.DipECT.dipect import DipECT
from clustpy.data import load_mnist
from clustpy.deep.autoencoders import FeedforwardAutoencoder
import numpy as np
import pickle


def trainable_function(config: dict):
    # dataset
    dataset, labels = load_mnist(return_X_y=True)
    dataset = dataset / 255

    # autoencoder
    autoencoder = FeedforwardAutoencoder([dataset.shape[1], 500, 500, 2000, 10])

    dipect = DipECT(
        autoencoder=autoencoder,
        autoencoder_param_path="/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/autoencoder/feedforward_mnist_21.pth",
        random_state=np.random.RandomState(21),
        evaluate_every_n_epochs=4,
        clustering_optimizer_params=config["clustering_optimizer_params"],
        reconstruction_loss_weight=config["reconstruction_loss_weight"],
        # projection axis
        projection_axis_learning_rate=config["projection_axis_learning_rate"],
        projection_axis_learning=config["projection_axis_learning"],
        # clustering
        clustering_n_epochs=config["clustering_n_epochs"],
        # pruning
        pruning_factor=config["pruning_factor"],
        pruning_strategy=config["pruning_strategy"],
        pruning_threshold=config["pruning_threshold"],
        # tree growth
        tree_growth_frequency=config["tree_growth_frequency"],
        tree_growth_amount=config["tree_growth_amount"],
        tree_growth_unimodality_treshold=config["tree_growth_unimodality_treshold"],
        tree_growth_upper_bound_leaf_nodes=config["tree_growth_upper_bound_leaf_nodes"],
        tree_growth_use_unimodality_pvalue=config["tree_growth_use_unimodality_pvalue"],
        # unimodal
        unimodal_loss_application=config["unimodal_loss_application"],
        unimodal_loss_node_criteria_method=config["unimodal_loss_node_criteria_method"],
        unimodal_loss_weight=config["unimodal_loss_weight"],
        unimodal_loss_weight_direction=config["unimodal_loss_weight_direction"],
        unimodal_loss_weight_function=config["unimodal_loss_weight_function"],
        loss_weight_function_normalization=config["loss_weight_function_normalization"],
        # multimodal
        mulitmodal_loss_application=config["mulitmodal_loss_application"],
        mulitmodal_loss_node_criteria_method=config[
            "mulitmodal_loss_node_criteria_method"
        ],
        mulitmodal_loss_weight_direction=config["mulitmodal_loss_weight_direction"],
        mulitmodal_loss_weight_function=config["mulitmodal_loss_weight_function"],
        multimodal_loss_weight=config["multimodal_loss_weight"],
        # utility
        early_stopping=config["early_stopping"],
        refinement_epochs=config["refinement_epochs"],
    )
    dipect.fit_predict(dataset, labels)


# searchspace
search_space = ng.p.Dict(
    clustering_optimizer_params=ng.p.Dict(lr=ng.p.Choice([1e-4])),
    reconstruction_loss_weight=ng.p.Choice([None]),
    # projection axis
    projection_axis_learning_rate=ng.p.Choice([1e-3, 1e-4, 1e-5, 1e-6, 0.0]),
    projection_axis_learning=ng.p.Choice(
        [None, "all", "only_leaf_nodes", "partial_leaf_nodes"]
    ),
    # clustering
    clustering_n_epochs=ng.p.Choice([60]),
    # pruning
    pruning_factor=ng.p.Choice([1.0]),
    pruning_strategy=ng.p.Choice(["epoch_assessment"]),
    pruning_threshold=ng.p.Choice([100, 250, 500]),
    # tree growth
    tree_growth_frequency=ng.p.Choice([0.0, 0.5, 1.0, 2.0, 3.0, 4.0]),
    tree_growth_amount=ng.p.Scalar(lower=1, upper=20).set_integer_casting(),
    tree_growth_unimodality_treshold=ng.p.Choice([0.95, 0.975, 1.0]),
    tree_growth_upper_bound_leaf_nodes=ng.p.Choice([20, 100]),
    tree_growth_use_unimodality_pvalue=ng.p.Choice([True, False]),
    # unimodal
    unimodal_loss_application=ng.p.Choice([None, "leaf_nodes", "all"]),
    unimodal_loss_node_criteria_method=ng.p.Choice(["tree_depth", "time_of_split"]),
    unimodal_loss_weight=ng.p.Choice([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
    unimodal_loss_weight_direction=ng.p.Choice(["ascending", "descending"]),
    unimodal_loss_weight_function=ng.p.Choice(["linear", "exponential", None]),
    loss_weight_function_normalization=ng.p.Choice([-1]),
    # multimodal
    mulitmodal_loss_application=ng.p.Choice([None, "leaf_nodes", "all"]),
    mulitmodal_loss_node_criteria_method=ng.p.Choice(["tree_depth", "time_of_split"]),
    mulitmodal_loss_weight_direction=ng.p.Choice(["ascending", "descending"]),
    mulitmodal_loss_weight_function=ng.p.Choice(["linear", "exponential", None]),
    multimodal_loss_weight=ng.p.Choice([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
    # utility
    early_stopping=ng.p.Choice([False]),
    refinement_epochs=ng.p.Choice([0]),
)

optimizer = ng.optimizers.Chaining(
    [ng.optimizers.NeuralMetaModelTwoPointsDE, ng.optimizers.SMAC3],
    budgets=[100],
)

algo = NevergradSearch(
    optimizer=optimizer,
    space=search_space,
    metric="dp",
    mode="max",
)
tuner = tune.Tuner(
    tune.with_resources(trainable_function, resources={"cpu": 4, "gpu": 0.25}),
    tune_config=tune.TuneConfig(search_alg=algo, num_samples=300),
    run_config=train.RunConfig(
        name="dipect_hpo_stage_1",
        storage_path="/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/hpo",
    ),
)
results = tuner.fit()

with open(
    "/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/dipect_hpo_stage_1_result.pkl",
    "wb",
) as file:
    pickle.dump(results, file)

print(results.get_best_result())
