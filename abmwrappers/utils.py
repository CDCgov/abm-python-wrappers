import base64
import itertools
import os
import subprocess
import warnings
from typing import Tuple

import numpy as np
import polars as pl
import yaml
from cfa_azure.clients import AzureClient
from scipy.stats import truncnorm
from scipy.stats.qmc import Sobol


def run_gcm_command_line(
    jar_file,
    yaml_file="./input/config.yaml",
    output_dir="output",
    gcm_threads=4,
    recompile=False,
):
    if recompile:
        subprocess.run(["mvn", "clean", "package"], check=True)
    gcm_command = [
        "java",
        "-jar",
        jar_file,
        "-i",
        yaml_file,
        "-o",
        output_dir,
        "-t",
        str(gcm_threads),
    ]
    subprocess.run(
        gcm_command,
        check=True,
    )  # TODO: write output to a logfile if needed


def gcm_parameters_writer(params: dict, output_type: str = "YAML") -> str:
    """
    Converts a dictionary of parameters to the specified output format.

    Args:
        params (dict): Dictionary containing the parameters.
        output_type (str, optional): Output format type. Defaults to 'YAML'.

    Returns:
        str: String representation of the parameters in the specified format.
    """
    if output_type == "YAML":
        params_output = yaml.dump(params)
    # Add more conditions for other output types as needed
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    return params_output


def combine_params_dicts(
    baseline_dict: dict, new_dict: dict, scenario_key: str = "baseScenario"
) -> Tuple[dict, str]:
    """
    Combines two dictionaries by overwriting values in baseline_dict with values from new_dict.

    Args:
        baseline_dict (dict): The baseline dictionary.
        new_dict (dict): The dictionary containing new values to be combined.
        scenario_key (str): any scenario key which the values fall under (will be preserved)

    Returns:
        Tuple[dict, str]: A tuple containing the combined dictionary and a summary string.
            - The combined dictionary contains the merged key-value pairs from both input dictionaries, with scenario key preserved.
            - The summary string lists the keys that were updated (present in both original dicts),
              keys not modified (present in baseline_dict only), and keys added (present in new_dict only).
    """
    temp_dict = baseline_dict[scenario_key]

    updated_keys = []
    for key, value in new_dict.items():
        if key not in temp_dict:
            raise Exception(f"'{key}' not present in default params list.")

        temp_dict[key] = value
        updated_keys.append(key)

    not_modified_keys = set(temp_dict.keys()) - set(updated_keys)

    result_string = (
        f"Updated keys: {updated_keys}\nNot modified keys: {not_modified_keys}"
    )

    combined_dict = {}
    combined_dict[scenario_key] = temp_dict

    return combined_dict, result_string


def load_baseline_params(
    default_params_file,
    baseline_params_input,
    scenario_key: str = "baseScenario",
):
    """
    Loads default parameters from a YAML file and updates them with baseline parameters input.

    Args:
        default_params_file (str): The path to the YAML file containing default parameters.
        baseline_params_input (file location): A YAML file containing baseline parameter values to update the default parameters with.

    Returns:
         Tuple[dict, str]: A tuple containing the combined dictionary and a summary string.
            - The combined dictionary contains the merged key-value pairs from both input dictionaries, with scenario key preserved.
            - The summary string lists the keys that were updated (present in both original dicts),
              keys not modified (present in baseline_dict only), and keys added (present in new_dict only).

    Raises:
        FileNotFoundError: If the specified default_params_file does not exist or cannot be opened.
        yaml.YAMLError: If there is an error parsing the YAML file.
        TypeError: If either of the inputs is not a dictionary.
    """
    # Attempt to read the default parameters from file
    with open(default_params_file, "r") as file:
        try:
            default_params = yaml.safe_load(file)
            if not isinstance(default_params, dict):
                raise TypeError(
                    "Default parameters loaded are not a dictionary."
                )
            if not isinstance(baseline_params_input, dict):
                raise TypeError("Baseline params input is not a dictionary.")

            # Create baseline_params by updating default_params with baseline_params_input
            return combine_params_dicts(
                default_params, baseline_params_input, scenario_key
            )

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error reading YAML file: {e}")


def params_grid_search(param_dict):
    """
    Generate a Polars DataFrame containing all combinations of parameters.

    Args:
        param_dict (dict): A dictionary where keys are parameter names and values are lists of possible values for those parameters.

    Returns:
        pl.DataFrame: A Polars DataFrame with each parameter as a column, and each row representing a unique combination of parameters.

    """
    # Create all possible combinations of parameters using itertools.product
    keys, values = zip(*param_dict.items())
    param_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    # Convert list of dictionaries to a Polars DataFrame
    df = pl.DataFrame(param_combinations)
    return df


def df_to_nested_dict(df):
    """
    Convert a Polars DataFrame into a nested dictionary with rows as keys and columns as subkeys.
    If 'simulation' column exists, use it as keys for each row.

    Args:
        df (pl.DataFrame): The Polars DataFrame to convert.

    Returns:
        dict: A nested dictionary where each key is an index of the row or value from 'simulation' column,
              and each value is another dictionary with column names as keys and cell values as values.

    Raises:
        ValueError: If 'simulation' column is not unique or does not exist when required.
    """

    # Check if 'simulation' column exists
    if "simulation" in df.columns:
        # Check if all values in 'simulation' are unique
        if df["simulation"].is_duplicated().any():
            raise ValueError("Values in 'simulation' column are not unique.")

        # Use 'simulation' values as keys
        nested_dict = {
            row["simulation"]: {
                col: row[col] for col in df.columns if col != "simulation"
            }
            for row in df.to_dicts()
        }

    else:
        # Use index as keys
        nested_dict = {
            i: {col: row[col] for col in df.columns}
            for i, row in enumerate(df.to_dicts())
        }

    return nested_dict


def generate_parameter_samples(
    params_inputs: dict, n_simulations: int, method: str = "sobol"
) -> dict:
    """
    Generate samples of parameters based on the specified method.
    Args:
        params_inputs (dict): Dictionary containing parameters and their ranges as [min, max].
        n_simulations (int): Number of simulations to perform.
        method (str): Sampling method to be used ("random" or "sobol").
    Returns:
        dict: Dictionary of sampled parameters for each simulation.
    """
    warnings.warn(
        "Function generate_parameter_samples is deprecated and will be removed in a future update.",
        DeprecationWarning,
    )
    # Get the number of parameters
    num_params = len(params_inputs)
    # Initialize dictionary to store parameter combinations
    parameter_combinations = {}
    if method == "random":
        # Iterate over each simulation
        for i in range(n_simulations):
            combination = {}
            # Iterate over each parameter key
            for param_key, param_range in params_inputs.items():
                # Randomly select a parameter value within the range
                param_value = np.random.uniform(param_range[0], param_range[1])
                # Add the parameter key-value pair to the combination dictionary
                combination[param_key] = param_value
            # Add the combination dictionary to the parameter combinations dictionary with simulation index as key
            parameter_combinations[i] = combination
    elif method == "sobol":
        # Create a new instance of the Sobol generator
        sobol = Sobol(d=num_params)
        # Generate Sobol samples in [0, 1]
        sobol_samples = sobol.random(n=n_simulations)
        # Iterate over each simulation
        for i in range(n_simulations):
            combination = {}
            # Iterate over each parameter key and its corresponding sample value from Sobol sequence
            for j, (param_key, param_range) in enumerate(
                params_inputs.items()
            ):
                # Scale sample value from [0, 1] to [min_value, max_value]
                param_value = (
                    param_range[0]
                    + (param_range[1] - param_range[0]) * sobol_samples[i][j]
                )
                # Add the parameter key-value pair to the combination dictionary
                combination[param_key] = param_value
            # Add the combination dictionary to the parameter combinations dictionary with simulation index as key
            parameter_combinations[i] = combination
    return parameter_combinations


def generate_job_name(job_prefix, length_input=64):
    """
    Generates a job name string that consists of the input prefix followed by a URL-safe,
    Base64 encoded string. The total length of the generated job name will not exceed 64 characters.

    Parameters:
    length (int): The desired length of the random part of the job name.
    job_prefix (str): The prefix to be prepended to the random part of the job name.

    Returns:
    str: A job name string that combines the prefix and a URL-safe Base64 encoded string.

    Raises:
    ValueError: If the job_prefix is not between 1 and 63 characters, or if length is not
                within valid bounds considering the prefix.
    """
    if not (len(job_prefix) < length_input):
        raise ValueError(
            f"Job prefix length must be less than {length_input} characters"
        )

    random_bytes = os.urandom(length_input * 3 // 4 + 1)
    base64_string = base64.urlsafe_b64encode(random_bytes).decode("utf-8")
    safe_string = base64_string.rstrip("=")
    truncated_safe_string = safe_string[: len(job_prefix) - length_input]
    return f"{job_prefix}{truncated_safe_string}"


def get_truncated_normal(mean, sd, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def initialize_azure_client(
    ab_config_path, super_experiment_name, create_pool=False
):
    """
    Initializes an Azure client based on configuration provided in a YAML file and additional parameters.

    This function reads the Azure Batch configuration from a YAML file, sets up job prefix,
    blob container name, and pool name based on the configuration and parameters. It then initializes
    an AzureClient object, creates a blob container if it doesn't exist, and optionally creates a new pool
    with specified settings.

    Parameters:
    - ab_config_path (str): The file path to the Azure Batch configuration YAML file.
    - super_experiment_name (str): A unique identifier for the experiment. This is used as a default value for job prefix, blob_container_name,
                                   and pool name if they are not explicitly set in the config.
    - create_pool (bool): A flag indicating whether to create a new pool. Defaults to False.
                           If True, additional pool settings are read from the config file
                           and used to create a new pool.

    Returns:
    - tuple: A tuple containing two elements:
        1. An initialized AzureClient object or None if initialization fails.
        2. The name of the blob container or None if initialization fails.

    Usage example:
        ab_config_path = 'path/to/your/ab/config/file.yaml'
        super_experiment_name = 'your_super_experiment'
        client, blob_container = initialize_azure_client(ab_config_path,
                                                         super_experiment_name,
                                                         create_pool=True)

        if client and blob_container:
            print("Azure Client initialized successfully.")
        else:
            print("Failed to initialize Azure Client.")
    """
    # Load ab_config file
    with open(ab_config_path, "r") as file:
        ab_config = yaml.safe_load(file)

    # Set configuration variables
    config_path = ab_config["config_path"]
    job_prefix = (
        ab_config["job_prefix"]
        if ab_config.get("job_prefix") is not None
        else f"{super_experiment_name}_"
    )
    blob_container_name = (
        ab_config["blob_container_name"]
        if ab_config.get("blob_container_name") is not None
        else super_experiment_name
    )
    pool_name = (
        ab_config["pool_name"]
        if ab_config.get("pool_name") is not None
        else f"{super_experiment_name}_pool"
    )

    # Initialize client
    print(
        f"Initializing AzureClient with the following config path: {config_path}"
    )
    client = AzureClient(config_path=config_path)

    try:
        client.create_blob_container(blob_container_name, blob_container_name)

        if create_pool:
            # Set pool creation config variables
            docker_repo_name = (
                ab_config["docker_repo_name"]
                if ab_config.get("docker_repo_name") is not None
                else f"{super_experiment_name}_repo"
            )

            n_nodes = ab_config["n_nodes"]
            pool_mode = ab_config["pool_mode"]
            autoscale_nodes = (
                ab_config["max_autoscale_nodes"]
                if ab_config.get("max_autoscale_nodes") is not None
                and pool_mode == "autoscale"
                else n_nodes
            )
            task_slots_per_node = (
                ab_config["task_slots_per_node"]
                if ab_config.get("task_slots_per_node") is not None
                else 1
            )

            registry_name = ab_config["registry_name"]
            docker_image = ab_config["docker_image"]
            docker_tag = ab_config["docker_tag"]
            debug_mode = ab_config["debug_mode"]

            cache_blobfuse = (
                ab_config["cache_blobfuse"]
                if ab_config.get("cache_blobfuse") is not None
                else False
            )

            # Create pool
            client.set_debugging(debug_mode)
            client.package_and_upload_dockerfile(
                registry_name=registry_name,
                repo_name=docker_repo_name,
                path_to_dockerfile=docker_image,
                tag=docker_tag,
                use_device_code=True,
            )

            client.set_pool_info(
                mode=pool_mode,
                dedicated_nodes=autoscale_nodes,
                cache_blobfuse=cache_blobfuse,
                task_slots_per_node=task_slots_per_node
            )

            client.create_pool(pool_name=pool_name)

        else:
            client.set_pool(pool_name)

        return client, blob_container_name, job_prefix

    except Exception as e:
        print(f"An error occurred: {e}")

        return None, None, None
