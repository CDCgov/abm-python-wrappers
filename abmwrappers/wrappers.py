import json
import os
import shutil
import time
from multiprocessing import Pool
from typing import Callable

import polars as pl
from cfa_azure.clients import AzureClient

from abmwrappers import utils


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


def experiments_runner(
    experiments_dir: str,
    super_experiment_name: str,
    sub_experiment_name: str,
    model_type: str = "gcm",
    n_sims: int = None,
    input_data_type: str = "YAML",
    num_processes: int = 8,
    azure_batch: bool = False,
    exe_file: str = None,
    azure_client: AzureClient = None,
    job_prefix: str = "",
):
    """
    Runs GCM experiments either locally or on Azure Batch.

    Args:
        experiments_dir (str): Directory for experiments.
        super_experiment_name (str): Name of the super experiment.
        sub_experiment_name (str): Name of the sub experiment.
        n_sims (int): Number of simulations
        input_data_type (str): data type of GCM input file
        azure_batch (bool): Whether to use Azure Batch for processing.
        exe_file (str): Path to the file (GCM JAR file or Ixa file) for running the simulations.
        azure_client (AzureClient): Azure client instance for interacting with Azure services.
        job_prefix (str): Prefix to use for job name
    Returns:
        None
    """
    # Note: Keeping Azure Batch and local workflows entirely separate for now, but could restructure later to avoid repitition
    if azure_batch:
        if not azure_client:
            raise ValueError(
                "AzureClient must be provided when azure_batch is True"
            )

        # Name and create job
        job_id = utils.generate_job_name(job_prefix)
        azure_client.add_job(job_id=job_id)

        # Loop through simulations and add tasks to job
        for simulation in range(n_sims):
            # simulation_folder_path = f"{super_experiment_name}/{sub_experiment_name}/simulation_{simulation}"
            simulation_folder_path = f"/{super_experiment_name}/{sub_experiment_name}/simulation_{simulation}"
            input_file_name = f"input.{input_data_type}"
            input_file_path = f"{simulation_folder_path}/{input_file_name}"
            output_folder = f"{simulation_folder_path}/output/"

            # Create the command to run the simulation
            docker_cmd = (
                f"java -jar /app.jar -i {input_file_path} -o {output_folder}"
            )

            # Add the task to the job
            azure_client.add_task(job_id=job_id, docker_cmd=docker_cmd)

        # Monitor the job
        azure_client.monitor_job(job_id=job_id)

    elif not azure_batch:
        sub_experiment_dir = os.path.join(
            experiments_dir, super_experiment_name, sub_experiment_name
        )

        # find simulation subfolders and create full paths
        items = os.listdir(sub_experiment_dir)
        simulation_dirs = [
            os.path.join(sub_experiment_dir, item)
            for item in items
            if (
                os.path.isdir(os.path.join(sub_experiment_dir, item))
                and item.startswith("simulation_")
            )
        ]

        if num_processes > 0:
            start_time = time.time()

            run_pool_simulations(
                simulation_dirs, model_type, exe_file, num_processes
            )

            duration = time.time() - start_time
            print("Total run time:", round(duration, 2), "seconds")
            print(
                "Run time per simulation:",
                round(duration / len(simulation_dirs), 2),
                "seconds",
            )
        elif num_processes == 0:
            for sim_dir in simulation_dirs:
                run_local_simulation(sim_dir, model_type, exe_file)

        else:
            raise ValueError(
                f"num_processes must be a positive integer or 0. Value: {num_processes}"
            )


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


def run_experiment_sequence(
    experiments_dir: str,
    super_experiment_name: str,
    sub_experiment_name: str,
    simulations_dict: dict,
    exe_file: str,
    output_processing_user_function,  # Function for processing outputs
    model_type: str = "gcm",
    flatten_user_function=None,  # Function to flatten outputs if needed
    num_processes=8,
    azure_batch: bool = False,
    client: AzureClient = None,  # For Azure Batch
    blob_container_name: str = None,  # For Azure Batch
    job_prefix: str = None,  # For Azure Batch
    run_list: list = ["write", "run", "download", "gather"],
):
    """
    Run GCM workflow including writing experiments, running them,
    optionally downloading outputs from Azure Batch, and gathering results.

    Args:
        experiments_dir (str): Directory for experiments.
        super_experiment_name (str): Name of the super experiment.
        sub_experiment_name (str): Name of the sub experiment.
        simulations_dict (dict): Dictionary containing simulation parameters.
        input_data_type (str): Type of input data file (default is "YAML").
        jar_file (str): Path to the jar file for running the simulations.
        azure_batch (bool): Whether to use Azure Batch for processing.
        apply_func (function): A function to apply to each dataframe.
        flatten_func (function): A function to apply to the results dictionary.
        azure_client (AzureClient): Azure client instance for interacting with Azure services.
        blob_container_name (str): Name of Azure blob (storage) container
        job_prefix (str): Prefix for naming Azure Batch jobs. Optional.
        run_list (list): Specify which functions to run. Default is ["write","run","download","gather"].

    Returns:
        Results DataFrame after gathering all experiment outputs.
    """

    if "write" in run_list:
        experiments_writer(
            experiments_dir,
            super_experiment_name,
            sub_experiment_name,
            simulations_dict,
            model_type=model_type,
            azure_batch=azure_batch,
            azure_client=client,
            blob_container_name=blob_container_name,
        )

    # Get the number of simulations
    if isinstance(
        simulations_dict["simulation_parameter_values"], pl.DataFrame
    ):
        n_sims = simulations_dict["simulation_parameter_values"].height
    else:
        n_sims = len(simulations_dict["simulation_parameter_values"].keys())

    if "run" in run_list:
        experiments_runner(
            experiments_dir,
            super_experiment_name,
            sub_experiment_name,
            n_sims=n_sims,
            model_type=model_type,
            num_processes=num_processes,
            azure_batch=azure_batch,
            jar_file=exe_file,
            azure_client=client,
            job_prefix=job_prefix,
        )

    if "download" in run_list:
        # Download output files if using Azure Batch
        if azure_batch:
            download_outputs(
                experiments_dir,
                super_experiment_name,
                sub_experiment_name,
                n_sims=n_sims,
                azure_client=client,
                blob_container_name=blob_container_name,
            )

    if "gather" in run_list:
        results_df = experiments_gatherer(
            experiments_dir,
            super_experiment_name,
            sub_experiment_name,
            output_processing_user_function,
            flatten_user_function,
        )

    return results_df
