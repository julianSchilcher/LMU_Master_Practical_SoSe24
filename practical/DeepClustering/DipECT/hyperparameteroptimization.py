import os
import sys


from ray import tune, train
import nevergrad as ng
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search import ConcurrencyLimiter

sys.path.append(os.getcwd())


import numpy as np
import pickle
import pathlib
import json
from sklearn.gaussian_process.kernels import Matern


def trainable_function(config: dict):
    for lib in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[lib] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random_state = np.random.RandomState(21)
    import torch
    from practical.DeepClustering.DipECT.dipect import DipECT
    from clustpy.data import load_mnist
    from clustpy.deep.autoencoders import FeedforwardAutoencoder
    from clustpy.deep._utils import set_torch_seed

    os.environ["SKLEARN_SEED"] = str(random_state.get_state()[1][0])
    torch.use_deterministic_algorithms(mode=True)
    set_torch_seed(int(random_state.get_state()[1][0]))

    # dataset
    dataset, labels = load_mnist(return_X_y=True)
    dataset = dataset / 255

    # autoencoder
    autoencoder = FeedforwardAutoencoder([dataset.shape[1], 500, 500, 2000, 10])

    dipect = DipECT(
        batch_size=config["batch_size"],
        autoencoder=autoencoder,
        autoencoder_param_path="/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/autoencoder/feedforward_mnist_"
        + str(config["autoencoder_pretraining_n_epochs"])
        + "_21.pth",
        random_state=random_state,
        evaluate_every_n_epochs=4,
        clustering_optimizer_params=config["clustering_optimizer_params"],
        reconstruction_loss_weight=config["reconstruction_loss_weight"],
        # projection axis
        projection_axis_learning_rate=config["projection_axis_learning_rate"],
        projection_axis_learning=config["projection_axis_learning"],
        projection_axis_init=config["projection_axis_init"],
        projection_axis_n_init=config["projection_axis_n_init"],
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
        multimodal_loss_application=config["multimodal_loss_application"],
        multimodal_loss_node_criteria_method=config[
            "multimodal_loss_node_criteria_method"
        ],
        multimodal_loss_weight_direction=config["multimodal_loss_weight_direction"],
        multimodal_loss_weight_function=config["multimodal_loss_weight_function"],
        multimodal_loss_weight=config["multimodal_loss_weight"],
        # utility
        early_stopping=config["early_stopping"],
        refinement_epochs=config["refinement_epochs"],
    )
    dipect.fit_predict(dataset, labels)


# searchspace
search_space = ng.p.Dict(
    batch_size=256,  # ng.p.Choice([256, 384, 512, 784]),
    autoencoder_pretraining_n_epochs=100,  # ng.p.Choice([100]),
    clustering_optimizer_params=dict(
        lr=1e-4
    ),  # ng.p.Dict(lr=ng.p.Choice([1e-3, 1e-4, 1e-5])),
    reconstruction_loss_weight=ng.p.Scalar(
        init=772.3825128714965, lower=1.0, upper=1000.0, mutable_sigma=True
    ),
    # ng.p.Choice(
    #     [1 / 510, 1 / 384, 1 / 255, 0.007, 0.1, 1, 10.0, 255.0, 510.0]
    # ),
    # projection axis
    projection_axis_learning_rate=ng.p.Choice([5e-4, 2e-4, 1e-4, 1e-5, 1e-6, 1e-8]),
    # ng.p.Scalar(
    #     lower=1e-6, upper=95e-5, init=0.0003952954333547711, mutable_sigma=True
    # ),  # ng.p.Choice([0.0, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]),
    projection_axis_learning="all",  # ng.p.Choice(["all"]),
    projection_axis_init=ng.p.Choice(["kmeans", "kmeans++", "kmeans++2", "kmeansk"]),
    projection_axis_n_init=ng.p.Scalar(
        init=6, lower=6, upper=12, mutable_sigma=True
    ).set_integer_casting(),
    # clustering
    clustering_n_epochs=60,  # ng.p.Choice([60]),
    # pruning
    pruning_factor=1.0,  # ng.p.Choice([1.0]),
    pruning_strategy="epoch_assessment",  # ng.p.Choice(["epoch_assessment"]),
    pruning_threshold=2000,  # ng.p.Choice([, 2500]),
    # tree growth
    tree_growth_frequency=1.0,  # 1.0, ng.p.Choice([, 2.0]),
    tree_growth_amount=3,  # 3, ng.p.Scalar(lower=1, upper=3).set_integer_casting(),
    tree_growth_unimodality_treshold=0.975,
    # ng.p.Scalar(
    #     init=0.9768384573183925, lower=0.95, upper=1.0, mutable_sigma=True
    # ),  # ng.p.Choice([0.975]),
    tree_growth_upper_bound_leaf_nodes=100,  # ng.p.Choice([100]),
    tree_growth_use_unimodality_pvalue=True,  # ng.p.Choice([True]),
    # unimodal
    unimodal_loss_application="leaf_nodes",
    unimodal_loss_node_criteria_method=ng.p.Choice(
        ["tree_depth", "time_of_split", "equal"]
    ),
    # ng.p.Choice(
    #     ["tree_depth", "time_of_split", "equal"]
    # ),
    unimodal_loss_weight=ng.p.Scalar(
        init=534.3911240634819, lower=1.0, upper=1000.0, mutable_sigma=True
    ),
    # ng.p.Scalar(
    #     init=750.0, lower=250.0, upper=1000.0, mutable_sigma=True
    # ),  # ng.p.Choice([0.0, 0.1, 0.5, 1.0, 2.0, 10.0]),
    unimodal_loss_weight_direction=ng.p.Choice(["ascending", "descending"]),
    unimodal_loss_weight_function=ng.p.Choice(
        ["exponential", "linear", "log", "sqrt"]
    ),  # ng.p.Choice(["linear", "log", "sqrt"]),
    loss_weight_function_normalization=-1,  # ng.p.Choice([-1]),
    # multimodal
    multimodal_loss_application="all",  # ng.p.Choice(["leaf_nodes", "all"]),
    multimodal_loss_node_criteria_method=ng.p.Choice(
        ["tree_depth", "time_of_split", "equal"]
    ),  # "time_of_split",  #
    multimodal_loss_weight_direction=ng.p.Choice(["ascending", "descending"]),
    multimodal_loss_weight_function=ng.p.Choice(
        ["exponential", "linear", "log", "sqrt"]
    ),
    multimodal_loss_weight=ng.p.Scalar(
        init=638.2691524389803, lower=1, upper=1000.0, mutable_sigma=True
    ),
    # ng.p.Scalar(
    #     init=500.0, lower=200.0, upper=1000.0, mutable_sigma=True
    # ),  # ng.p.Choice([0.1, 0.5, 1.0, 2.0]),
    # utility
    early_stopping=False,  # ng.p.Choice([False]),
    refinement_epochs=0,  # ng.p.Choice([0]),
)


evaluated_points = []
# for file_name in pathlib.Path(
#     "/home/loebbert/projects/deepclustering/LMU_Master_Practical_SoSe24/practical/DeepClustering/DipECT/hpo"
# ).glob("dipect_hpo_stage_11/*/params.json"):
#     # value = float("".join(file_name.stem[-2:]))
#     with open(file_name, "r") as file:
#         point = json.load(file)
#     evaluated_points.append(point)

# optimizer = ng.optimizers.Chaining(
#     [
#         ng.optimizers.RandomSearch,
#         # ng.optimizers.NeuralMetaModelTwoPointsDE,
#         # ng.optimizers.RandomSearch,
#         ng.optimizers.ParametrizedBO(
#             initialization="Hammersley",
#             init_budget=75,
#             utility_kind="ucb",
#             gp_parameters=dict(
#                 kernel=Matern(nu=2.5),
#                 alpha=1e-6,
#                 normalize_y=True,
#                 n_restarts_optimizer=5,
#                 random_state=np.random.RandomState(21),
#             ),
#         ),
#         ng.optimizers.TBPSA,
#         ng.optimizers.TwoPointsDE,
#     ],
#     budgets=[75, 300, 100],
# )

# optimizer = ng.optimizers.ParametrizedBO(
#     initialization="Hammersley",
#     init_budget=40,
#     utility_kind="ucb",
#     gp_parameters=dict(
#         kernel=Matern(nu=2.5),
#         alpha=1e-6,
#         normalize_y=True,
#         n_restarts_optimizer=5,
#         random_state=np.random.RandomState(21),
#     ),
# )

optimizer = ng.optimizers.ConfPortfolio(
    optimizers=[
        ng.optimizers.RandomSearch,
        ng.optimizers.NGOpt,
        ng.optimizers.ParametrizedBO(
            utility_kind="ucb",
            gp_parameters=dict(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=np.random.RandomState(21),
            ),
        ),
        ng.optimizers.RFMetaModelTwoPointsDE,
    ],
    warmup_ratio=0.5,
)

algo = NevergradSearch(
    optimizer=optimizer,
    optimizer_kwargs={"budget": 200, "num_workers": 4},
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
    max_t=17000,
    grace_period=6000,
)

stage_nr = 21

tuner = tune.Tuner(
    func,
    tune_config=tune.TuneConfig(search_alg=algo, num_samples=200, scheduler=scheduler),
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

## 10 - reconstruction loss scaling
## 11 - unimodal application + scaling
## 12 - multimodal application + scaling
