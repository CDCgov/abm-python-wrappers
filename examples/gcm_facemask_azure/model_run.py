import os

import polars as pl

from abmwrappers import utils, wrappers

## ======================================#
## User inputs------
## ======================================#
experiments_dir = "experiments"
super_experiment_name = "wtk-gcm-azb"
sub_experiment_name = "subexperiment_1"
default_params_file = f"/{super_experiment_name}/config.yaml"

print(os.path.exists(default_params_file))
jar_file = "/app.jar"  # local jar_file is "target/gcm-seir-mask-1.0.0-SNAPSHOT-jar-with-dependencies.jar"


# Gatherer inputs (user functions)
def output_processing_user_function(outputs_dir):
    output_file_path = os.path.join(
        outputs_dir, "facemask_incidence_report.tsv"
    )
    if os.path.exists(output_file_path):
        df = pl.read_csv(output_file_path, separator="\t")
    else:
        raise FileNotFoundError(f"{output_file_path} does not exist.")

    return df.select(pl.col("recovered").max())


def flatten_user_function(results_dict):
    # Initialize empty dataframe
    results_df = pl.DataFrame()
    for simulation_number, simulation_data in results_dict["data"].items():
        sim_df = simulation_data.with_columns(
            pl.lit(simulation_number).alias("simulation")
        )
        results_df.vstack(sim_df, in_place=True)
    params_df = pl.DataFrame(
        [{"simulation": k, **v} for k, v in results_dict["parameters"].items()]
    )

    # Cost consistent simulation column type
    params_df = params_df.cast({"simulation": pl.Int64})
    results_df = results_df.cast({"simulation": pl.Int64})

    # Return results dataframe with recovered parameters
    return results_df.join(params_df, on="simulation")


# Define baseline params
baseline_params_input = {
    "population": 10000,
    "initialPrevalence": 0.2,
    "facemaskSymptomsDuration": 10,
    "facemaskCommunityDuration": 10,
    "facemaskCommunityStartDay": 10,
    "individualSusceptibilityReduction": 0.5,
    "individualInfectiousnessReduction": 0.6,
    "probabilityCommunityMaskWearing": 0.8,
    "probabilitySymptomsMaskWearing": 0.1,
    "totalDays": 100,
    "enableCommunityPolicy": True,
    "enableFacemaskCommunityTrigger": False,
    "facemaskCommunityTrigger": 100,
    "communityMaskTriggerInterval": 14,
    "enableSymptomaticPolicy": True,
}


# Define experiment parameters
experiment_params_input = {
    "individualInfectiousnessReduction": [
        0.0,
        0.1,
        0.2,
        0.3,
        1,
    ],
    "individualSusceptibilityReduction": [
        0.0,
        0.1,
        0.2,
        0.3,
        1,
    ],
}


## ======================================#
## Import, define, and sample params for simulations------
## ======================================#

# Create baseline_params by updating default_params with baseline_params_input
baseline_params, summary_string = utils.load_baseline_params(
    default_params_file, baseline_params_input
)

experiment_df = utils.params_grid_search(experiment_params_input)


# convert to dictionary
experiment_params_dict = utils.df_to_nested_dict(experiment_df)

simulations_dict = {
    "baseline_parameters": baseline_params,
    "experiment_parameters": list(experiment_params_input.keys()),
    "simulation_parameter_values": experiment_params_dict,
}

## ======================================#
## Write GCM input files-----
## ======================================#

wrappers.gcm_experiments_writer(
    experiments_dir,
    super_experiment_name,
    sub_experiment_name,
    simulations_dict,
)

## ======================================#
## Run GCM for each simulation------
## ======================================#

wrappers.gcm_experiments_runner(
    experiments_dir,
    super_experiment_name,
    sub_experiment_name,
    n_sims=experiment_df.shape[0],
    jar_file=jar_file,
    num_processes=1,
)

# ======================================#
# Gather GCM data------
# ======================================#

results_df = wrappers.gcm_experiments_gatherer(
    experiments_dir,
    super_experiment_name,
    sub_experiment_name,
    output_processing_user_function,
    flatten_user_function,
)

print(results_df)
