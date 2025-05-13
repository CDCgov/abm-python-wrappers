import json
import os
import shutil
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
    distance_fn: Callable = None,
    data_processing_fn: Callable = None,
    products: list = None,
    products_output_dir: str = None,
    scenario_key: str = "baseline_parameters",
    cmd: str = None,
    clean: bool = False,
):
    if products is None:
        products = ["distances", "simulations"]

    if distance_fn is None:
        distance_fn = experiment.distance_fn
    if data_processing_fn is None:
        data_processing_fn = experiment.data_processing_fn
    if data_processing_fn is None:
        raise ValueError(
            "Data processing function must be provided if not previously declared in Experiment parameters."
        )

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
    if products_output_dir is None:
        output_dir = experiment.data_path
    else:
        output_dir = products_output_dir

    if "distances" in products:
        if distance_fn is None:
            raise ValueError(
                "Distance function must be provided if not previously declared in Experiment parameters."
            )
        
        sim_bundle.calculate_distances(experiment.target_data, distance_fn)

        distance_data_part_path = (
            f"{output_dir}/distances/simulation={simulation_index}/"
        )
        os.makedirs(distance_data_part_path, exist_ok=True)
        pl.DataFrame(
            {"distance": sim_bundle.distances.values()}
        ).write_parquet(distance_data_part_path + "data.parquet")

    if "simulations" in products:
        simulation_data_part_path = (
            f"{output_dir}/simulations/simulation={simulation_index}/"
        )
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


def abcsmc_experiment_runner(
    experiment: experiment_class.Experiment,
    data_processing_fn: Callable = None,
    distance_fn: Callable = None,
    prior_distribution_dict: dict = None,
    perturbation_kernels: dict = None,
    changed_baseline_params: dict = {},
    files_to_upload: list = [],
    scenario_key: str = None,
):
    if scenario_key is None:
        scenario_key = experiment.scenario_key

    if experiment.current_step == 0 or experiment.current_step is None:
        experiment.initialize_simbundle(
            prior_distribution_dict=prior_distribution_dict,
            changed_baseline_params=changed_baseline_params,
            scenario_key=scenario_key,
            unflatten=False,
        )

    if experiment.azure_batch:
        print("Not fully implemented yet")
        (
            client,
            blob_container_name,
            job_prefix,
        ) = utils.initialize_azure_client(
            experiment.azb_config_path,
            experiment.super_experiment_name,
            experiment.create_pool,
        )

        # Create experiment history file to Azure Blob Storage for initializing each step
        # If a compressed history already exists, offer off-ramp for user to abort
        experiment_file = os.path.join(
            experiment.data_path, "experiment_history.pkl"
        )
        if os.path.exists(experiment_file):
            user_input = (
                input(
                    f"Experiment compressed history already exists. Overwrite experiment? (Y/N): "
                )
                .strip()
                .upper()
            )
            if user_input != "Y":
                print("Experiment terminated by user.")
                return

        experiment.compress_and_save(experiment_file)
        files_to_upload.append(experiment_file)

        # Initialize Azure client
        if client and blob_container_name and job_prefix:
            print("Azure client initialized successfully.")
        else:
            raise ValueError(
                "Failed to initialize Azure client. Check your configuration."
            )

        job_name = utils.generate_job_name(job_prefix)
        client.add_job(job_name, task_id_ints=True)
        client.monitor_job(job_name, timeout=36000)

        # Identifying file locations wihtin blob storage
        blob_experiment_directory = os.path.join(
            experiment.sub_experiment_name, experiment.sub_experiment_name
        )
        gather_script = os.path.join(
            blob_experiment_directory, "gather_step.py"
        )
        task_script = os.path.join(blob_experiment_directory, "task.py")
        blob_data_path = os.path.join(blob_experiment_directory, "data")
        blob_experiment_path = os.path.join(
            blob_experiment_directory, "experiment_history.pkl"
        )

        # Upload the experiment history file to Azure Blob Storage along with any included files
        client.upload_files(
            files_to_upload,
            blob_container_name,
            experiment.sub_experiment_name,
        )

        gather_task_id = None
        for step, tolerance in experiment.tolerance_dict.items():
            tasks_id_range = []

            for simulation_index in range(experiment.n_simulations):
                task_i_cmd = f"poetry python run /{task_script} --index {simulation_index} -f /{blob_experiment_path} -k {scenario_key} -o /{blob_data_path} --clean --products "
                task_i_cmd += " ".join(products)

                sim_task_id = client.add_task(
                    job_id=job_name,
                    docker_cmd=task_i_cmd,
                    depends_on=gather_task_id,
                )
                if simulation_index in (0, experiment.n_simulations - 1):
                    tasks_id_range.append(sim_task_id)

            task_range = tuple(tasks_id_range)
            gather_task_cmd = f"poetry python run /{gather_script} -f /{blob_experiment_path} -i /{blob_data_path}"

            gather_task_id = client.add_task(
                job_id=job_name,
                docker_cmd=gather_task_cmd,
                depends_on=task_range,
            )

        client.download_file(
            src_path=os.path.join(
                experiment.sub_experiment_name, "experiment_history.pkl"
            ),
            dest_path=experiment_file,
        )

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
                    scenario_key=scenario_key,
                )

            experiment.resample_for_next_abc_step(perturbation_kernels)


def abcsmc_update_compressed_experiment(
    experiment_file: str,
    distance_path: str,
):
    """
    Update a compressed experiment file with distances stored as a .parquet Hive partition from a specified path.

    Args:
        experiment_file (str): Path to the compressed experiment file.
        distance_path (str): Path to the distance data.

    Returns:
        None
    """

    # Load the compressed experiment
    experiment = experiment_class.Experiment(img_file=experiment_file)

    # Load the distances
    experiment.read_parquet_data_to_current_step(input_dir=distance_path)
    experiment.resample_for_next_abc_step()

    # Save the updated experiment
    experiment.compress_and_save(experiment_file)


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
