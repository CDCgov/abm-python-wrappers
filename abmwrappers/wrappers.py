import os
from multiprocessing import Pool
from typing import Callable

import polars as pl
import yaml
from cfa_azure.clients import AzureClient

from abmwrappers import utils
from abmwrappers.experiment_class import Experiment

# --------------------------
# Simulation wrappers
# --------------------------


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


def download_outputs(
    azure_client: AzureClient,
    blob_container_name: str,
    dest_path: str,
    src_path: str,
) -> None:
    """

    Downloads the output directory of a given sub-experiment from Azure Blob Storage to a local directory.

    Args:

        azure_client: An instance of an Azure Batch client.
        blob_container_name: The name of the Azure Blob Storage container.
        dest_path: The local directory where the output files will be downloaded.
        src_path: The path in the Azure Blob Storage container where the output files are stored.

    Returns:

        None

    """

    azure_client.download_directory(
        src_path=src_path,
        dest_path=dest_path,
        container_name=blob_container_name,
    )


# --------------------------
# Experiment wrappers
# --------------------------


def products_from_inputs_index(
    simulation_index: int,
    experiment: Experiment,
    distance_fn: Callable = None,
    data_processing_fn: Callable = None,
    products: list = None,
    products_output_dir: str = None,
    scenario_key: str = None,
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

        # Write the distance data to a Parquet file with part path storing the simulation column
        os.makedirs(distance_data_part_path, exist_ok=True)
        pl.DataFrame(
            {"distance": sim_bundle.distances.values()}
        ).write_parquet(distance_data_part_path + "data.parquet")

    if "simulations" in products:
        simulation_data_part_path = (
            f"{output_dir}/simulations/simulation={simulation_index}/"
        )

        # Write the simulation data to a Parquet file with part path storing the simulation column
        os.makedirs(simulation_data_part_path, exist_ok=True)
        sim_bundle.results[simulation_index].write_parquet(
            simulation_data_part_path + "data.parquet"
        )

    if clean:
        # Delete raw_output file if cleaning intermediates
        for file in os.listdir(simulation_output_path):
            os.remove(os.path.join(simulation_output_path, file))


def abcsmc_update_compressed_experiment(
    experiment_file: str,
    distance_path: str,
):
    """
    Update a compressed experiment file with distances stored as a .parquet Hive partition from a specified path.
    Used during the gather step that gates ABC SMC steps from progressing

    Args:
        experiment_file (str): Path to the compressed experiment file.
        distance_path (str): Path to the distance data.

    Returns:
        None
    """

    # Load the compressed experiment
    experiment = Experiment(img_file=experiment_file)

    # Load the distances
    experiment.read_parquet_data_to_current_step(input_dir=distance_path)
    experiment.resample_for_next_abc_step()

    # Save the updated experiment
    experiment.compress_and_save(experiment_file)


def abcsmc_experiment_runner(
    experiment: Experiment,
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

        if experiment.model_type == "gcm":
            experiment.exe_file = "app/app.jar"
        elif experiment.model_type == "ixa":
            experiment.exe_file = "app/app"

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

        # Identifying file locations wihtin blob storage
        blob_experiment_directory = os.path.join(
            blob_container_name, experiment.sub_experiment_name
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

            if step == max(experiment.tolerance_dict.keys()):
                products = ["distances", "simulations"]
            else:
                products = ["distances"]

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
                depends_on_range=task_range,
            )

        client.monitor_job(job_name, timeout=3600)
        client.download_file(
            src_path=os.path.join(
                experiment.sub_experiment_name, "experiment_history.pkl"
            ),
            dest_path=experiment_file,
            container_name=blob_container_name,
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


def split_scenarios_into_subexperiments(
    experiment: Experiment,
    griddle_path: str = None,
    scenario_key: str = None,
    seed_key: str = None,
):
    """
    Splits a series of inputs into sub-experiments under a scenario=index data structure
    """

    # Griddle scenarios should each generate one input file, but resulting experiments may require multiple replciates
    experiment.replicates = 1

    if griddle_path is None:
        if experiment.griddle_file is None:
            raise ValueError(
                "Griddle path must be provided if not previously declared in Experiment parameters."
            )
        griddle_path = experiment.griddle_file

    # Write all inputs
    experiment.write_simulation_inputs_from_griddle(
        griddle_path, scenario_key=scenario_key, seed_key=seed_key
    )

    # Store each simulation input as the base for a scenario=index subfolder in data
    input_folder = os.path.join(experiment.data_path, "input")
    input_files = os.listdir(input_folder)

    index = 0
    experiment.experiments_path = experiment.super_experiment_name
    scenarios_name = "scenarios"
    for input in input_files:
        if not input.endswith(experiment.input_file_type):
            continue

        input_file_path = os.path.join(input_folder, input)

        # Create the base input files for each scenario
        scenario_subexperiment = f"scenario={index}"
        scenario_input_path = os.path.join(
            experiment.directory,
            scenarios_name,
            scenario_subexperiment,
            "input",
        )
        os.makedirs(scenario_input_path, exist_ok=True)
        new_params_file = os.path.join(
            scenario_input_path, f"base.{experiment.input_file_type}"
        )
        os.rename(input_file_path, new_params_file)

        # Read in the config and rewrite the subexperiment and superexperiment names
        new_config = os.path.join(scenario_input_path, "config.yaml")
        with open(experiment.config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        config["local_path"]["super_experiment_name"] = scenarios_name
        config["local_path"]["sub_experiment_name"] = scenario_subexperiment
        config["local_path"]["default_params_file"] = new_params_file
        with open(new_config, "w") as f:
            yaml.dump(config, f)

        # Create the experiment object
        subexperiment = Experiment(
            experiments_directory=experiment.directory,
            config_file=new_config,
        )

        subexperiment.initialize_simbundle()
        index += 1
    
    # Delete empty data directory
    os.rmdir(os.path.join(experiment.data_path, "input"))
    os.rmdir(experiment.data_path)
