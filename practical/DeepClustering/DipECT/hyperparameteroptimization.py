import os
import sys
from ray import tune, train
import nevergrad as ng
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search import ConcurrencyLimiter

sys.path.append(os.getcwd())

from practical.DeepClustering.DipECT.dipect import DipECT
from clustpy.data import load_mnist
from clustpy.deep.autoencoders import FeedforwardAutoencoder
import numpy as np
import pickle
import pathlib
import json
from sklearn.gaussian_process.kernels import Matern


def trainable_function(config: dict):
    # dataset
    dataset, labels = load_mnist(return_X_y=True)
    dataset = dataset / 255

    # autoencoder
    autoencoder = FeedforwardAutoencoder([dataset.shape[1], 500, 500, 2000, 10])

    dipect = DipECT(
        autoencoder=autoencoder,
        autoencoder_param_path="/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/autoencoder/feedforward_mnist_"
        + str(config["autoencoder_pretraining_n_epochs"])
        + "_21.pth",
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
        tree_growth_min_cluster_size=config["pruning_threshold"],
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
    autoencoder_pretraining_n_epochs=ng.p.Choice([100]),
    clustering_optimizer_params=ng.p.Dict(lr=ng.p.Choice([1e-3, 1e-4, 1e-5, 1e-6])),
    reconstruction_loss_weight=ng.p.Scalar(
        init=255.0, lower=1, upper=1000, mutable_sigma=True
    ),
    # ng.p.Choice(
    #     [1 / 510, 1 / 384, 1 / 255, 0.007, 0.1, 1, 10.0, 255.0, 510.0]
    # ),
    # projection axis
    projection_axis_learning_rate=ng.p.Choice([1e-3, 1e-4, 1e-5, 1e-6, 1e-8]),
    projection_axis_learning=ng.p.Choice(["partial_leaf_nodes"]),
    # clustering
    clustering_n_epochs=ng.p.Choice([60]),
    # pruning
    pruning_factor=ng.p.Choice([1.0]),
    pruning_strategy=ng.p.Choice(["epoch_assessment"]),
    pruning_threshold=ng.p.Choice([1500, 2000, 2500]),
    # tree growth
    tree_growth_frequency=ng.p.Choice([1.0, 2.0]),
    tree_growth_amount=ng.p.Scalar(lower=1, upper=4).set_integer_casting(),
    tree_growth_unimodality_treshold=ng.p.Choice([0.95, 0.975, 1.0]),
    tree_growth_upper_bound_leaf_nodes=ng.p.Choice([100]),
    tree_growth_use_unimodality_pvalue=ng.p.Choice([True]),
    # unimodal
    unimodal_loss_application=ng.p.Choice(["leaf_nodes", "all"]),
    unimodal_loss_node_criteria_method=ng.p.Choice(["tree_depth", "time_of_split"]),
    unimodal_loss_weight=ng.p.Scalar(
        init=0.0, lower=0.0, upper=1000.0, mutable_sigma=True
    ),  # ng.p.Choice([0.0, 0.1, 0.5, 1.0, 2.0, 10.0]),
    unimodal_loss_weight_direction=ng.p.Choice(["ascending", "descending"]),
    unimodal_loss_weight_function=ng.p.Choice(["linear", "exponential", "log", "sqrt"]),
    loss_weight_function_normalization=ng.p.Choice([-1]),
    # multimodal
    mulitmodal_loss_application=ng.p.Choice(["leaf_nodes", "all"]),
    mulitmodal_loss_node_criteria_method=ng.p.Choice(["tree_depth", "time_of_split"]),
    mulitmodal_loss_weight_direction=ng.p.Choice(["ascending", "descending"]),
    mulitmodal_loss_weight_function=ng.p.Choice(
        ["linear", "exponential", "log", "sqrt"]
    ),
    multimodal_loss_weight=ng.p.Scalar(
        init=1.0, lower=0.5, upper=1000.0, mutable_sigma=True
    ),  # ng.p.Choice([0.1, 0.5, 1.0, 2.0]),
    # utility
    early_stopping=ng.p.Choice([False]),
    refinement_epochs=ng.p.Choice([0]),
)


evaluated_points = []
# for file_name in pathlib.Path(
#     "/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/hpo"
# ).glob("*/*/params.json"):
#     # value = float("".join(file_name.stem[-2:]))
#     with open(file_name, "r") as file:
#         point = json.load(file)
#     evaluated_points.append(point)

optimizer = ng.optimizers.Chaining(
    [
        # ng.optimizers.ScrHammersleySearch,
        # ng.optimizers.NeuralMetaModelTwoPointsDE,
        ng.optimizers.ParametrizedBO(
            initialization="LHS",
            init_budget=100,
            utility_kind="ucb",
            gp_parameters=dict(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=np.random.RandomState(21),
            ),
        ),
        ng.optimizers.TwoPointsDE,
    ],
    budgets=[400],
)

# optimizer = ng.optimizers.TwoPointsDE
algo = NevergradSearch(
    optimizer=optimizer,
    optimizer_kwargs={"budget": 500, "num_workers": 4},
    space=search_space,
    metric="dp",
    mode="max",
    points_to_evaluate=evaluated_points,
)

func = tune.with_resources(trainable_function, resources={"cpu": 1, "gpu": 1 / 4})

scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric="dp",
    mode="max",
    max_t=16442,
    grace_period=8000,
)

stage_nr = 5

tuner = tune.Tuner(
    func,
    tune_config=tune.TuneConfig(search_alg=algo, num_samples=500, scheduler=scheduler),
    run_config=train.RunConfig(
        name=f"dipect_hpo_stage_{stage_nr}",
        storage_path="/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/hpo",
    ),
)


# tuner = tune.Tuner.restore(
#     "/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/hpo/dipect_hpo_stage_1",
#     func,
#     restart_errored=True,
# )
results = tuner.fit()

with open(
    f"/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/hpo_stage_{stage_nr}_result.pkl",
    "wb",
) as file:
    pickle.dump(results.get_dataframe(), file)

print(results.get_dataframe().sort_values("dp", ascending=False).head(10))
