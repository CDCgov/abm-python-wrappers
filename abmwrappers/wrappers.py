import json
import os
import shutil
import time
from multiprocessing import Pool
from typing import Callable

import polars as pl
from cfa_azure.clients import AzureClient

from abmwrappers import experiment_class, utils


def experiments_writer(
    experiments_dir: str,
    super_experiment_name: str,
    sub_experiment_name: str,
    simulations_dict,
    model_type="gcm",
    scenario_key="baseScenario",
    input_data_type: str = "YAML",
    azure_batch: bool = False,
    azure_client: AzureClient = None,
    blob_container_name: str = None,
    delete_existing: bool = True,
    params_as_files_function_dict: dict = None,
):
    """
    Accepts sub-experiment specifications, samples from the parameter sweep value range
    using a specified method, and writes out input files (yaml or other)

    Args:
        experiments_dir (str): Directory for experiments.
        super_experiment_name (str): Name of the super experiment.
        sub_experiment_name (str): Name of the sub experiment.
        simulations_dict (dict): Dictionary containing simulation parameters.
        model_type (str): "gcm" or "ixa" (default is "gcm", will check for executeable type to be sure).
        scenario_key (str): Key for the scenario to be used in the input file (default is "baseScenario").
        input_data_type (str): Type of input data file (default is "YAML").
        azure_batch (bool): Whether to use Azure Batch for processing.
        azure_client (AzureClient): Azure client instance for interacting with Azure services.
        blob_container_name (str): Name of Azure blob (storage) container.
        delete_existing (bool): Flag to delete existing sub experiment.
        params_as_files_function_dict (dict): A dictionary of parameter name/function pairs, where the function is passed the parameter value and returns the file name and file content needed.
    Returns:
        None
    """
    # Note: Keeping Azure Batch and local workflows entirely separate for now, but could restructure later to avoid repetition

    # Delete write folder unless otherwise specified, if folder already exists
    if delete_existing:
        delete_experiment_items(
            experiments_dir,
            super_experiment_name,
            sub_experiment_name,
        )

    # If simulations_dict["simulation_parameter_values"] is a pl.Dataframe, convert it to a dictionary
    if isinstance(
        simulations_dict["simulation_parameter_values"], pl.DataFrame
    ):
        temp_dict = utils.df_to_simulation_dict(
            simulations_dict["simulation_parameter_values"]
        )
        simulations_dict["simulation_parameter_values"] = temp_dict
        del temp_dict

    if azure_batch:
        if not azure_client:
            raise ValueError(
                "AzureClient must be provided when azure_batch is True"
            )

        input_files = []

        for simulation, sim_params in simulations_dict[
            "simulation_parameter_values"
        ].items():
            # Define the folder path
            simulation_folder_path = (
                f"{sub_experiment_name}/simulation_{simulation}"
            )

            # Handle any parameters which are expected to be files
            if params_as_files_function_dict is not None:
                for (
                    parameter,
                    function,
                ) in params_as_files_function_dict.items():
                    # Retrieve the file name and file contents using the user-defined function
                    file_name, file_contents = function(sim_params[parameter])

                    # Update sim_params with the file name
                    sim_params[parameter] = os.path.join(
                        simulation_folder_path, file_name
                    )

                    # Write the file
                    input_files.append(
                        (simulation_folder_path, file_name, file_contents)
                    )

            # Create full parameters specification
            sim_params_full, _ = utils.combine_params_dicts(
                simulations_dict["baseline_parameters"],
                sim_params,
                scenario_key=scenario_key,
            )

            # For Ixa, update the output_dir
            if model_type == "ixa":
                sim_params_full["output_dir"] = os.path.join(
                    simulation_folder_path
                )

            # Generate simulation input file content
            input_file_name = f"input.{input_data_type}"
            sim_params_data_file = utils.parameters_writer(
                sim_params_full, input_data_type
            )

            # file_path = f"{simulation_folder_path}/{input_file_name}"
            input_files.append(
                (simulation_folder_path, input_file_name, sim_params_data_file)
            )

        # Generate full parameters list content
        full_params_json = json.dumps(simulations_dict, indent=4)
        full_params_folder_path = f"{sub_experiment_name}"
        full_params_file_name = "full_params_list_all_sims.json"
        input_files.append(
            (full_params_folder_path, full_params_file_name, full_params_json)
        )

        # todo: eventually we want to connect to a docker container and be able to write directly to Azure Batch. We need an orchestrator connected to a node
        # Upload input files to the blob container
        for folder_path, file_name, file_content in input_files:
            # Write content to a file to upload
            full_path = os.path.join(
                experiments_dir, super_experiment_name, folder_path
            )

            os.makedirs(full_path, exist_ok=True)

            with open(os.path.join(full_path, file_name), "w") as temp_file:
                temp_file.write(file_content)

            # Upload the file
            print(folder_path)
            azure_client.upload_files(
                [os.path.join(full_path, file_name)],
                container_name=blob_container_name,
                location_in_blob=folder_path,
            )

    elif not azure_batch:
        # Create the super_experiment_name/sub_experiment_name directory if it doesn't exist
        experiment_dir = os.path.join(
            experiments_dir, super_experiment_name, sub_experiment_name
        )
        os.makedirs(experiment_dir, exist_ok=True)

        for simulation, sim_params in simulations_dict[
            "simulation_parameter_values"
        ].items():
            # Create simulation folder
            simulation_folder = os.path.join(
                experiment_dir, f"simulation_{simulation}"
            )
            os.makedirs(simulation_folder, exist_ok=True)

            # Handle any parameters which are expected to be files
            if params_as_files_function_dict is not None:
                for (
                    parameter,
                    function,
                ) in params_as_files_function_dict.items():
                    # Retrieve the file name and file contents using the user-defined function
                    file_name, file_contents = function(sim_params[parameter])

                    # Update sim_params with the file name
                    sim_params[parameter] = os.path.join(
                        simulation_folder, file_name
                    )

                    # Write the file
                    with open(
                        os.path.join(simulation_folder, file_name), "w"
                    ) as file:
                        file.write(file_contents)

            # Create full parameters specification
            sim_params_full, _ = utils.combine_params_dicts(
                simulations_dict["baseline_parameters"],
                sim_params,
                scenario_key=scenario_key,
            )

            # For Ixa, update the output_dir and create the folder if needed
            if model_type == "ixa":
                sim_params_full["output_dir"] = os.path.join(
                    simulation_folder, "output"
                )
                os.makedirs(
                    os.path.join(simulation_folder, "output"), exist_ok=True
                )

            # Write each simulation's parameters to input file and save it to disk
            input_file_path = os.path.join(
                simulation_folder, f"input.{input_data_type}"
            )
            sim_params_data_file = utils.parameters_writer(
                sim_params_full, input_data_type
            )
            with open(input_file_path, "w") as file:
                file.write(sim_params_data_file)

        # Write out full simulations parameter list to json file
        full_params_json = json.dumps(simulations_dict, indent=4)
        full_params_path = os.path.join(
            experiment_dir, "full_params_list_all_sims.json"
        )
        with open(full_params_path, "w") as f:
            f.write(full_params_json)


def run_local_simulation(
    simulation_dir,
    model_type: str,
    exe_file,
    cmd: list = None,
    recompile=False,
):
    if cmd is None:
        output_dir, cmd = utils.write_default_cmd(
            simulation_dir, model_type, exe_file
        )
        os.makedirs(output_dir, exist_ok=True)
    utils.run_model_command_line(cmd, model_type, recompile)


def products_from_inputs_index(
    simulation_index: int,
    experiment: experiment_class.Experiment,
    distance_fn: Callable,
    data_processing_fn: Callable,
    products: list = None,
    scenario_key: str = "baseline_parameters",
    cmd: str = None,
    clean: bool = False,
):
    if products is None:
        products = ["distances", "simulations"]

    input_file_path = experiment.write_simulation_inputs_to_file(
        simulation_index,
        scenario_key=scenario_key,
    )
    simulation_output_path = os.path.join(
        experiment.data_path, "raw_output", f"simulation_{simulation_index}"
    )
    os.makedirs(simulation_output_path, exist_ok=True)

    # This will require the default command line for model run if None
    if cmd is None:
        cmd = utils.write_default_cmd(
            input_file=input_file_path,
            output_dir=simulation_output_path,
            exe_file=experiment.exe_file,
            model_type=experiment.model_type,
        )

    utils.run_model_command_line(cmd, model_type=experiment.model_type)

    sim_bundle = experiment.get_bundle_from_simulation_index(simulation_index)

    sim_bundle.results = {
        simulation_index: data_processing_fn(simulation_output_path)
    }

    # --------------------------
    # Process the and store the desired products
    # --------------------------

    if "distances" in products:
        sim_bundle.calculate_distances(experiment.target_data, distance_fn)

        distance_data_part_path = (
            f"{experiment.data_path}/distances/simulation={simulation_index}/"
        )
        os.makedirs(distance_data_part_path, exist_ok=True)
        pl.DataFrame(
            {"distance": sim_bundle.distances.values()}
        ).write_parquet(distance_data_part_path + "data.parquet")

    if "simulations" in products:
        simulation_data_part_path = f"{experiment.data_path}/simulations/simulation={simulation_index}/"
        os.makedirs(simulation_data_part_path, exist_ok=True)
        pl.DataFrame(
            {"simulation": sim_bundle.results[simulation_index]}
        ).write_parquet(simulation_data_part_path + "data.parquet")

    if clean:
        # Delete raw_output file if cleaning intermediates
        for file in os.listdir(simulation_output_path):
            os.remove(os.path.join(simulation_output_path, file))


def run_pool_simulations(
    simulation_dirs: list,
    model_type: str,
    exe_file,
    num_processes: int = 8,
):
    """
    Run simulations in parallel using multiprocessing.

    Args:
        simulation_dirs (list): List of directories for each simulation.
        model_type (str): Type of model to run (e.g., "gcm", "ixa").
        exe_file (str): Path to the executable file for the model.
        num_processes (int): Number of processes to use for parallel execution.

    Returns:
        None
    """

    with Pool(processes=num_processes) as pool:
        pool.starmap(
            run_local_simulation,
            [(sim_dir, model_type, exe_file) for sim_dir in simulation_dirs],
        )


def experiment_runner(
    experiment: experiment_class.Experiment,
    data_processing_fn: Callable,
    distance_fn: Callable,
    prior_distribution_dict: dict = None,
    perturbation_kernels: dict = None,
    changed_baseline_params: dict = {},
):
    if experiment.current_step == 0 or experiment.current_step is None:
        experiment.initialize_simbundle(
            prior_distribution_dict=prior_distribution_dict,
            changed_baseline_params=changed_baseline_params,
            scenario_key="cfa_ixa_ebola_response_2025.Parameters",
            unflatten=False,
        )

    if experiment.azure_batch:
        print("Not implemented yet")

    else:
        for step, tolerance in experiment.tolerance_dict.items():
            print(
                f"Running step {step} of {len(experiment.tolerance_dict)} with tolerance {tolerance}"
            )

            current_bundle = experiment.simulation_bundles[
                experiment.current_step
            ]
            if step != experiment.current_step:
                raise ValueError(
                    f"Execution has become misaligned from updating. Step {step} does not match current step {experiment.current_step}"
                )

            if step == max(experiment.tolerance_dict.keys()):
                products = ["distances", "simulations"]
            else:
                products = ["distances"]

            for simulation_index in current_bundle.inputs["simulation"]:
                products_from_inputs_index(
                    simulation_index,
                    experiment=experiment,
                    data_processing_fn=data_processing_fn,
                    distance_fn=distance_fn,
                    products=products,
                    scenario_key="cfa_ixa_ebola_response_2025.Parameters",
                )

            experiment.resample_for_next_abc_step(perturbation_kernels)


def experiments_gatherer(
    experiments_dir: str,
    super_experiment_name: str,
    sub_experiment_name: str,
    apply_func: Callable,
    flatten_func: Callable = None,
    sims_filter: int = None,
):
    """
    Gathers data from GCM experiments by applying a specified lambda function to a specific file across all simulations.

    Args:
        experiments_dir (str): Directory for experiments.
        super_experiment_name (str): Name of the super experiment.
        sub_experiment_name (str): Name of the sub experiment.
        apply_func (function): A function to apply to each dataframe.
        flatten_func (function): A function to apply to the results dictionary.
        sims_filter (int):

    Returns:
        dict or list or DataFrame: Depending on 'flatten_func', either a dictionary with simulation number as key and
                                   processed data as value, or a flattened list/DataFrame of all processed data depending
                                   on the specifics of apply_func.
    """

    # Define subexperiment directory
    sub_experiment_dir = os.path.join(
        experiments_dir, super_experiment_name, sub_experiment_name
    )

    if sims_filter is None:
        # Find simulation subfolders
        items = os.listdir(sub_experiment_dir)
        simulation_folders = [
            item
            for item in items
            if (
                os.path.isdir(os.path.join(sub_experiment_dir, item))
                and item.startswith("simulation_")
            )
        ]
    else:
        # Define simulation subfolders using sims_filter
        simulation_folders = [f"simulation_{num}" for num in sims_filter]

    with open(
        os.path.join(sub_experiment_dir, "full_params_list_all_sims.json"), "r"
    ) as file:
        master_params_dict = json.load(file)

    # Initialize results container
    results = {
        "data": {},
        "parameters": master_params_dict["simulation_parameter_values"],
    }

    # Loop through simulation folders
    for simulation_folder in simulation_folders:
        outputs_dir = os.path.join(
            sub_experiment_dir, simulation_folder, "output"
        )

        simulation_number = int(simulation_folder.split("_")[1])
        # Apply user-specified function and add to dictionary
        results["data"][simulation_number] = apply_func(outputs_dir)

    # Apply user-specified flattening function if appropriate and return results
    if flatten_func is not None:
        return flatten_func(results)
    else:
        return results


def download_outputs(
    experiments_dir: str,
    super_experiment_name: str,
    sub_experiment_name: str,
    n_sims: int,
    azure_client: AzureClient,
    blob_container_name: str,
) -> None:
    """

    Downloads the output directory of a given sub-experiment from Azure Blob Storage to a local directory.

    Args:

        experiments_dir (str): The base local directory where experiments are stored.
        super_experiment_name (str): The name of the super experiment; used as the first subfolder in the local path.
        sub_experiment_name (str): The name of the sub-experiment; used as the second subfolder in the local path and part of source blob prefix.
        azure_client: An instance of an Azure Batch client.

    Returns:

        None

    """
    dest_path = f"{experiments_dir}/{super_experiment_name}/"
    src_blob_prefix = f"{sub_experiment_name}/"

    for sim in range(n_sims):
        src_path = src_blob_prefix + f"simulation_{sim}/output/"
        azure_client.download_directory(
            src_path=src_path,
            dest_path=dest_path,
            container_name=blob_container_name,
        )


def delete_experiment_items(
    experiments_dir: str,
    super_experiment_name: str,
    sub_experiment_name: str,
    prefix: str = None,
    file_type: str = None,
):
    """
    Deletes items (files/folders) within a given super experiment and sub experiment directory,
    optionally filtering by a given prefix or specific file type.

    Args:
        experiments_dir (str): Directory for experiments.
        super_experiment_name (str): Name of the super experiment.
        sub_experiment_name (str): Name of the sub experiment.
        prefix (str): Optional; Prefix that selected items must start with.
        file_type (str): Optional; Specific file extension to filter for deletion.

    Returns:
        None
    """
    # Define the path to the super experiment and sub experiment directory
    sub_experiment_dir = os.path.join(
        experiments_dir, super_experiment_name, sub_experiment_name
    )

    # Check if the directory exists before attempting any operations
    if not os.path.exists(sub_experiment_dir):
        # print(f"The directory {sub_experiment_dir} does not exist. No deletion necessary.")
        return

    # List all items in the directory
    for item in os.listdir(sub_experiment_dir):
        # Construct full path to item
        item_path = os.path.join(sub_experiment_dir, item)
        # Check if item matches prefix condition if provided
        if prefix is not None and not item.startswith(prefix):
            continue
        # Check if item matches file type condition if provided
        if file_type is not None:
            # If it's a folder or doesn't match the extension, skip it
            if os.path.isdir(item_path) or not item.endswith(file_type):
                continue
        # Delete files or folders that meet conditions
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
