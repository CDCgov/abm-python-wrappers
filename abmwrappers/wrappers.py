import os
import subprocess
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


def run_pool_tasks(cmds: list, num_processes: int = 8):
    def quickrun(cmd):
        subprocess.run(cmd.split())

    with Pool(processes=num_processes) as pool:
        pool.map(quickrun, cmds)


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


def run_step_return_sims(
    experiment: Experiment,
    data_read_fn: Callable[[str], pl.DataFrame] | None = None,
    raw_data_filename: str | None = None,
    distance_fn: Callable[
        [
            pl.DataFrame,  # results data
            pl.DataFrame | dict | float | int,  # target data
        ],
        float | int,  # returns
    ]
    | None = None,
    data_postprocessing_fn: Callable[[pl.DataFrame, pl.DataFrame], float | int]
    | None = None,
    products: list[str] | str | None = None,
    compress: bool = True,
    clean: bool = False,
    overwrite: bool = False,
) -> pl.DataFrame:
    """
    Function to run a step in an experiment and return the resulting simulations.
    This function is a wrapper around the `run_step` and `read_results` methods of the Experiment class as a common use-case.

    Args:
        :param experiment (Experiment): The Experiment instance to run the step on.
        :param data_read_fn (Callable): Function to read data from a raw output file into a polars DataFrame.
            - This function is created as a default to read a single file if raw_data_filename is specified
            - This function should accept a single argument, which is the raw output directory path of the step.
            - Once inside the raw ouptut directory, relevant paths should at least be read and should return a single data frame
        :param data_postprocessing_fn (Callable, optional): Function to postprocess data further after running the simulations from the step. Optional.
            - This function can operate on the whole collection of simulations simultaneouesly, accepting only a pl.DataFrame
            - The functionality can instead be included in the pre-processing funciton, before the data is fully stored
        :param products (list[str], str optional): List of products to generate during the step. Defaults to ["simulations"].
        :param compress (bool, optional): Whether to compress the results. Defaults to True.
        :param clean (bool, optional): Whether to clean up the raw output directory after running the step. Defaults to False.
        :param overwrite (bool, optional): Whether to overwrite existing hive-paritioned parquert file once results are postprocessed. Defaults to False.
    """

    # Run the simulation
    experiment.run_step(
        data_read_fn=data_read_fn,
        data_filename=raw_data_filename,
        distance_fn=distance_fn,
        products=products,
        compress=compress,
        clean=clean,
    )

    simulation_data_frame = experiment.read_results(
        filename="simulations",
        data_processing_fn=data_postprocessing_fn,
        write=(compress and overwrite),
        partition_by=["simulation"],
    )
    return simulation_data_frame


def update_abcsmc_img(
    experiment_file: str,
    products_path: str,
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
    experiment.read_distances(input_dir=products_path, overwrite=True)
    experiment.resample()

    # Save the updated experiment
    experiment.save(experiment_file)


def run_abcsmc(
    experiment: Experiment,
    distance_fn: Callable[
        [
            pl.DataFrame,  # results data
            pl.DataFrame | dict | float | int,  # target data
        ],
        float | int,  # returns
    ]
    | None = None,
    data_read_fn: Callable[[str], pl.DataFrame] | None = None,
    data_filename: str | None = None,
    prior_distribution_dict: dict = None,
    perturbation_kernels: dict = None,
    changed_baseline_params: dict = {},
    files_to_upload: list[str] | str | None = [],
    scenario_key: str = None,
    ask_overwrite: bool = True,
    use_existing_distances: bool = False,
    keep_all_sims: bool = False,
    save: bool = True,
):
    if scenario_key is None:
        scenario_key = experiment.scenario_key

    if not isinstance(files_to_upload, list):
        files_to_upload = [files_to_upload]

    if experiment.current_step is None:
        experiment.initialize_simbundle(
            prior_distribution_dict=prior_distribution_dict,
            changed_baseline_params=changed_baseline_params,
            scenario_key=scenario_key,
            unflatten=False,
        )

    if experiment.azure_batch:
        print(
            "Not fully automated yet - requires direct upload of gather and task steps"
        )

        if experiment.client is None:
            raise ValueError(
                "Azure client must be initialized using an Experiment created by a config file for Azure Batch execution. "
                "New clients cannot be instantiated from compressed Experiments. "
                "Set `azure_batch` to False in the Experiment config file to run locally or run ABC SMC from new Experiment."
            )
        job_prefix = experiment.job_prefix
        blob_container_name = experiment.blob_container_name
        client = experiment.client

        # Create experiment history file to Azure Blob Storage for initializing each step
        # If a compressed history already exists, offer off-ramp for user to abort
        experiment_file = os.path.join(
            experiment.data_path, "experiment_history.pkl"
        )
        if os.path.exists(experiment_file):
            if ask_overwrite:
                user_input = (
                    input(
                        "Experiment compressed history already exists. Overwrite experiment? (Y/N): "
                    )
                    .strip()
                    .upper()
                )
                if user_input != "Y":
                    print("Experiment terminated by user.")
                    return

        if experiment.model_type == "gcm":
            experiment.exe_file = "/app.jar"
        elif experiment.model_type == "ixa":
            experiment.exe_file = (
                f"/app/{os.path.basename(experiment.exe_file)}"
            )

        experiment.save(experiment_file)
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
        user_script = utils.get_caller()
        files_to_upload.append(user_script)
        script_name = os.path.basename(user_script)
        gather_script = os.path.join(blob_experiment_directory, script_name)
        task_script = os.path.join(blob_experiment_directory, script_name)
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

        if use_existing_distances:
            distances_path = (
                f"{experiment.sub_experiment_name}/data/distances/"
            )
            try:
                if experiment.blob_directory_exists(distances_path):
                    stored_distances = experiment.parquet_from_path(
                        distances_path, verbose=False
                    )
                    if experiment.verbose:
                        print(
                            f"""Re-using previously calculated distances.
                            Found {stored_distances.height} values.
                            These are not verified to match inputs.
                            Please only use for re-running and extending same experiment."""
                        )
                else:
                    raise StopIteration

            except StopIteration:
                use_existing_distances = False
                if experiment.verbose:
                    print(
                        "No existing distances could be found. Starting ABC SMC from first step without supplied distances."
                    )

        gather_task_id = None
        for step, tolerance in experiment.tolerance_dict.items():
            task_ids = []
            if step == max(experiment.tolerance_dict.keys()) or keep_all_sims:
                products = ["distances", "simulations"]
            else:
                products = ["distances"]

            for simulation_index in range(experiment.n_simulations):
                realized_sim_index = (
                    simulation_index + step * experiment.n_simulations
                )
                # Either run regardless or do quickj eval to keep previous distances
                if use_existing_distances:
                    run_sim = stored_distances.filter(
                        pl.col("simulation") == realized_sim_index
                    ).is_empty()
                else:
                    run_sim = True

                if run_sim:
                    task_i_cmd = f"poetry run python /{task_script} -x run --index {realized_sim_index} -i /{blob_experiment_path} -d /{blob_data_path} --clean --products "
                    task_i_cmd += " ".join(products)

                    sim_task_id = client.add_task(
                        job_id=job_name,
                        docker_cmd=task_i_cmd,
                        depends_on=gather_task_id,
                    )

                    # Task ids are often returned as int-coerceable strings, but sometimes not.
                    # It is more stable to update the second task id for writing a task id range Tuple.
                    if not task_ids or len(task_ids) == 1:
                        task_ids.append(sim_task_id)
                    else:
                        task_ids[-1] = sim_task_id

            gather_task_cmd = f"poetry run python /{gather_script} -x gather -i /{blob_experiment_path} -d {experiment.sub_experiment_name}/data"

            if not task_ids:
                gather_task_id = client.add_task(
                    job_id=job_name,
                    docker_cmd=gather_task_cmd,
                    depends_on=gather_task_id,
                )
            elif len(task_ids) == 1:
                gather_task_id = client.add_task(
                    job_id=job_name,
                    docker_cmd=gather_task_cmd,
                    depends_on=task_ids,
                )
            else:
                task_range = tuple(task_ids)
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

            if step != experiment.current_step:
                raise ValueError(
                    f"Execution has become misaligned from updating. Step {step} does not match current step {experiment.current_step}"
                )

            if step == max(experiment.tolerance_dict.keys()) or keep_all_sims:
                products = ["distances", "simulations"]
            else:
                products = ["distances"]

            if use_existing_distances:
                distances_path = f"{experiment.data_path}/distances/"
                if os.path.exists(distances_path):
                    stored_distances = experiment.parquet_from_path(
                        distances_path
                    )
                    if experiment.verbose:
                        print(
                            f"""Re-using previously calculated distances.
                            Found {stored_distances.height} values.
                            These are not verified to match inputs.
                            Please only use for re-running and extending same experiment."""
                        )
                else:
                    use_existing_distances = False
                    if experiment.verbose:
                        print(
                            "No existing distances could be found. Starting ABC SMC from first step without supplied distances."
                        )

                for simulation_index in range(experiment.n_simulations):
                    realized_sim_index = (
                        simulation_index + step * experiment.n_simulations
                    )
                    if stored_distances.filter(
                        pl.col("simulation") == realized_sim_index
                    ).is_empty():
                        experiment.run_index(
                            simulation_index=realized_sim_index,
                            data_read_fn=data_read_fn,
                            data_filename=data_filename,
                            distance_fn=distance_fn,
                            products=products,
                            scenario_key=scenario_key,
                        )
            else:
                experiment.run_step(
                    data_read_fn=data_read_fn,
                    data_filename=data_filename,
                    distance_fn=distance_fn,
                    products=products,
                    scenario_key=scenario_key,
                )

            # Products_from_index currently adds only one distance at a time
            # We recreate the data loss of update_abcsmc_img here using `del`
            del experiment.simulation_bundles[step].distances
            experiment.read_distances(experiment.data_path)
            experiment.resample(perturbation_kernels)

        if save:
            experiment.save()


def create_scenario_subexperiments(
    experiment: Experiment,
    griddle_path: str = None,
    scenario_key: str = None,
    ask_overwrite: bool = True,
    sample_posterior: bool | None = False,
    n_samples: int | None = None,
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

    if len(os.listdir(experiment.data_path)) > 0:
        if ask_overwrite:
            user_input = (
                input(
                    f"Experiment data path {experiment.data_path} is not empty before writing inputs. Overwrite data path for scenarios? (Y/N): "
                )
                .strip()
                .upper()
            )
            if user_input != "Y":
                print("Scenario split terminated by user.")
                return
        utils.remove_directory_tree(experiment.data_path, remove_root=False)

    experiment.write_inputs_from_griddle(
        griddle_path,
        scenario_key=scenario_key,
        seed_variable_name=experiment.seed_variable_name,
        sample_posterior=sample_posterior,
        n_samples=n_samples,
    )

    # Store each simulation input as the base for a scenario=index subfolder in data
    input_folder = os.path.join(experiment.data_path, "input")
    input_files = os.listdir(input_folder)

    index = 0
    for fp in input_files:
        if not fp.endswith(experiment.input_file_type):
            continue

        input_file_path = os.path.join(input_folder, fp)

        # Create the base input files for each scenario
        scenario_subexperiment = f"scenario={index}"
        scenario_input_path = os.path.join(
            experiment.directory,
            "scenarios",
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
        # Use the scenarios subfolder created during scenario_input_path as super experiment
        config["local_path"]["super_experiment_name"] = "scenarios"
        config["local_path"]["sub_experiment_name"] = scenario_subexperiment
        config["local_path"]["default_params_file"] = new_params_file
        # Use an updated azure flag from the experiment as scenarios are not on azure
        config["azb"]["azure_batch"] = experiment.azure_batch
        config["experiment_conditions"][
            "replicates_per_particle"
        ] = experiment.replicates
        with open(new_config, "w") as f:
            yaml.dump(config, f)

        index += 1

    # Delete empty data directory - do not call utils.remove_directory_tree because folders should be empty.
    os.rmdir(os.path.join(experiment.data_path, "input"))
    os.rmdir(experiment.data_path)


def write_scenario_products(
    scenario: str,
    scenario_experiment: Experiment,
    experiment_data_path: str,
    products: list[str] | str | None = None,
    clean: bool = False,
):
    if products is None:
        products = ["simulations"]
    else:
        if not isinstance(products, list):
            products = [products]

    for product in products:
        old_parquet_path = os.path.join(scenario_experiment.data_path, product)
        new_parquet_path = os.path.join(
            experiment_data_path, "scenarios", scenario
        )

        os.makedirs(new_parquet_path, exist_ok=True)
        simulation_data_frame = scenario_experiment.parquet_from_path(
            old_parquet_path
        )
        simulation_data_frame.write_parquet(
            os.path.join(new_parquet_path, "data.parquet")
        )

        if clean:
            utils.remove_directory_tree(old_parquet_path)
