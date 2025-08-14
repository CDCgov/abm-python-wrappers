import base64
import inspect
import itertools
import json
import os
import subprocess
import warnings
from typing import Callable, Tuple

import cfa_azure.blob_helpers as blob_helpers
import polars as pl
import yaml
from cfa_azure.clients import AzureClient
from scipy.stats import truncnorm

# Global character sequence for flattening nested parameters
FLATTENED_PARAM_CONNECTOR = ">>>"


def run_model_command_line(
    cmd: list, model_type: str, recompile=False, overwrite=True
):
    if model_type == "gcm":
        if recompile:
            subprocess.run(["mvn", "clean", "package"], check=True)
        subprocess.run(
            cmd,
            check=True,
        )
    elif model_type == "ixa":
        if recompile:
            subprocess.run(["cargo", "build", "--release"], check=True)
        if overwrite:
            cmd.append("--force-overwrite")
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. must be 'gcm' or 'ixa'"
        )


def write_default_cmd(
    input_file: str,
    output_dir: str,
    model_type: str,
    exe_file: str,
) -> list:
    def needs_rel_path(path: str) -> bool:
        return not path.startswith("/") and not path.startswith("./")

    if model_type == "gcm":
        cmd = [
            "java",
            "-jar",
            exe_file,
            "-i",
            input_file,
            "-o",
            output_dir,
            "-t",
            "4",
        ]
    elif model_type == "ixa":
        # Handle relative paths cases if not specified as absolute
        # ixa pastes strings together so full relative or absolute path must be used
        if needs_rel_path(exe_file):
            exe_file = "./" + exe_file
        if needs_rel_path(input_file):
            input_file = "./" + input_file
        if needs_rel_path(output_dir):
            output_dir = "./" + output_dir
        output_dir = output_dir + "/"

        cmd = [
            exe_file,
            "--config",
            input_file,
            "--prefix",
            output_dir,
        ]
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. must be 'gcm' or 'ixa'"
        )

    return cmd


def flatten_dict(d, parent_key="", sep=FLATTENED_PARAM_CONNECTOR):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(flat_dict, sep=FLATTENED_PARAM_CONNECTOR):
    result = {}
    for key, value in flat_dict.items():
        nested_elements = key.split(sep)
        current_level = result

        for part in nested_elements[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        current_level[nested_elements[-1]] = value

    return result


def remove_directory_tree(dir_path: str, remove_root: bool = True) -> None:
    """
    Recursively removes a directory tree.

    Args:
        dir_path (str): Path to the directory to be removed.
        remove_root (bool): If True, removes the root directory as well. Defaults to True.
    """
    if not os.path.exists(dir_path):
        return

    # Iterate through all files and directories in the specified directory
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # Remove all files in the current directory
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

        # Remove all subdirectories in the current directory
        for subdir_name in dirs:
            subdir_path = os.path.join(root, subdir_name)
            os.rmdir(subdir_path)

    # Remove the root directory if specified
    if remove_root:
        os.rmdir(dir_path)


def abm_parameters_writer(
    params: dict, output_type: str = "yaml", unflatten: bool = True
) -> str:
    """
    Converts a dictionary of parameters to the specified output format.

    Args:
        params (dict): Dictionary containing the parameters.
        output_type (str, optional): Output format type. Defaults to 'yaml'.

    Returns:
        str: String representation of the parameters in the specified format.
    """

    if unflatten:
        params = {key: unflatten_dict(value) for key, value in params.items()}

    if output_type == "yaml":
        params_output = yaml.dump(params)
    elif output_type == "json":
        params_output = json.dumps(params, indent=4)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")

    return params_output


def digit_from_string(string: str):
    numbers = []
    curr_num = ""
    for char in string:
        if char.isdigit():
            curr_num += char
        elif len(curr_num) > 0:
            numbers.append(int(curr_num))
            curr_num = ""
    if len(curr_num) > 0:
        numbers.append(int(curr_num))
    return numbers


def column_keys_from_path(path: str) -> dict:
    def get_key(path_segment: str, key_name: str, params: dict) -> dict:
        d = {}
        if key_name in path_segment:
            vals = digit_from_string(path_segment)
            if len(vals) == 1:
                d.update({key_name: vals[0]})
            elif len(vals) > 1:
                raise ValueError(
                    f"Multiple {key_name} indices found in path segment '{path_segment}'."
                )
        return d

    path_list = path.split(os.sep)
    cols = {}
    keys = ["simulation", "scenario"]
    for piece in path_list:
        for k in keys:
            cols.update(get_key(piece, k, cols))
    return cols


def combine_params_dicts(
    baseline_dict: dict,
    new_dict: dict,
    scenario_key: str = "baseScenario",
    overwrite_unnested: bool = False,
    unflatten: bool = True,
    sep=FLATTENED_PARAM_CONNECTOR,
) -> Tuple[dict, str]:
    """
    Combines two dictionaries by overwriting values in baseline_dict with values from new_dict.
    new_dict is a flattened input dictionary with keys separated by FLATTENED_PARAM_CONNECTOR.
    The scenario_key dictionary in baseline_dict is flattened and combined with the input new_dict.
    The dictionary is by default unflattened before returning.

    Args:
        baseline_dict (dict): The baseline dictionary.
        new_dict (dict): The dictionary containing new values to be combined.
        scenario_key (str): any scenario key which the values fall under (will be preserved)
        overwrite_unnested (bool): whether to overwrite the nested elements of a particular key during combination
        unflatten (bool): whether to unflatten nested parameters

    Returns:
        Tuple[dict, str]: A tuple containing the combined dictionary and a summary string.
            - The combined dictionary contains the merged key-value pairs from both input dictionaries, with scenario key preserved.
            - The summary string lists the keys that were updated (present in both original dicts),
              keys not modified (present in baseline_dict only), and keys added (present in new_dict only).

    Examples:
        - To merge only lower level parameters from a new_dict into a baseline_dict scenario
        >>> base = {"scenario": {"mean": 1.0, "scale": 2.0}}
        >>> new = {"mean": 2.5}
        >>> combine_params_dicts(base, new, scenario_key="scenario")
        {"scenario": {"mean": 2.5, "scale": 2.0}}

        - To merge hierarchical parameters that are consistent across model constructions
        >>> base = {"scenario": {"normal": {"mean": 0.0, "sd": 1.0}, "scale": 2.0}}
        >>> new = {"normal": {"mean": 2.5}}
        >>> combine_params_dicts(base, new, scenario_key="scenario")
        {"scenario": {"normal": {"mean": 2.5, "sd": 1.0}, "scale": 2.0}}

        - To replace whole chunks of hierarchically nested parameters
        >>> base = {"scenario": {"my_distribution": {"normal": {"mean": 0.0, "sd": 1.0}}, "scale": 2.0}}
        >>> new = {"my_distribution": {"gamma": {"rate": 4.0, "shape": 10.0 } } }
        >>> combine_params_dicts(base, new, scenario_key="scenario", overwrite_unnested=True)
        {"scenario": {"my_distribution": {"gamma": {"rate": 4.0, "shape": 10.0}}, "scale": 2.0}}

        - An alternative is to allow for a hierarchical new_dict to replace hierarchical terms
        >>> base = {"scenario": {"my_distribution": {"normal": {"mean": 0.0, "sd": 1.0}}, "scale": 2.0}}
        >>> new = {"my_distribution": {"gamma": {"rate": 4.0, "shape": 10.0}}
        >>> combine_params_dicts(base, new, scenario_key="scenario", overwrite_unnested=True)
        {"scenario": {"my_distribution": {"gamma": {"rate": 4.0, "shape": 10.0}}, "scale": 2.0}}
    """

    # Evaluate if new_dict is at the scenario key level and lower if so
    must_lower = False
    for key in new_dict.keys():
        if key == scenario_key:
            must_lower = True
            warnings.warn(
                "Scenario key was included in the `new_dict` dictionary. Removing and lowering."
            )
    if must_lower:
        new_dict = new_dict[scenario_key]

    # Check for any hierarchical parameters
    must_flatten = False
    for value in new_dict.values():
        if isinstance(value, dict):
            must_flatten = True
            warnings.warn(
                "There are hierarchical values in the `new_dict` dictionary. If this is unintentional, please abort the process. Otherwise, flattening"
            )
    if must_flatten:
        new_dict = flatten_dict(new_dict, sep=sep)

    temp_dict = baseline_dict[scenario_key]

    temp_dict = flatten_dict(temp_dict, sep=sep)

    updated_keys = []
    for key, value in new_dict.items():
        if key not in temp_dict:
            if not overwrite_unnested:
                raise Exception(f"'{key}' not present in default params list.")

        temp_dict[key] = value
        updated_keys.append(key)

    not_modified_keys = set(temp_dict.keys()) - set(updated_keys)

    result_string = (
        f"Updated keys: {updated_keys}\nNot modified keys: {not_modified_keys}"
    )

    # clean up keys that have duplicate nested keys in the new temp dict
    if overwrite_unnested:
        unflat_new = unflatten_dict(new_dict, sep=sep)
        to_remove = []
        for key in temp_dict.keys():
            splitkeys = key.split(sep)
            if (
                len(splitkeys) > 2
                and splitkeys[0] in unflat_new.keys()
                and splitkeys[1] not in unflat_new[splitkeys[0]].keys()
            ):
                to_remove.append(key)
        for key in to_remove:
            temp_dict.pop(key)

    if unflatten:
        temp_dict = unflatten_dict(temp_dict, sep=sep)

    # Re-introduce the scenario key
    combined_dict = {scenario_key: temp_dict}

    return combined_dict, result_string


def read_config_file(fp: str) -> dict:
    """
    Reads a YAML or JSON configuration file and returns its contents as a dictionary.

    Args:
        fp (str): The file path to the configuration file.

    Returns:
        dict: The contents of the configuration file as a dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        json.JSONDecodeError: If there is an error parsing the JSON file.
    """
    with open(fp, "r") as file:
        if fp.lower().endswith(".yaml") or fp.lower().endswith(".yml"):
            return yaml.safe_load(file)
        elif fp.lower().endswith(".json"):
            return json.load(file)
        else:
            raise ValueError(
                "Unsupported file type. Only YAML and JSON are supported."
            )


def load_baseline_params(
    default_params_file,
    baseline_params_input,
    scenario_key: str = "baseScenario",
    unflatten: bool = True,
):
    """
    Loads default parameters from a YAML or JSON file and updates them with baseline parameters input.

    Args:
        default_params_file (str): The path to the YAML file containing default parameters.
        baseline_params_input (file location): A YAML file containing baseline parameter values to update the default parameters with.
        scenario_key: Which scenario key to use
        unflatten (bool): whether to unflatten nested parameters

    Returns:
         Tuple[dict, str]: A tuple containing the combined dictionary and a summary string.
            - The combined dictionary contains the merged key-value pairs from both input dictionaries, with scenario key preserved.
            - The summary string lists the keys that were updated (present in both original dicts),
              keys not modified (present in baseline_dict only), and keys added (present in new_dict only).

    Raises:
        FileNotFoundError: If the specified default_params_file does not exist or cannot be opened.
        yaml.YAMLError: If there is an error parsing the YAML file.
        json.JSONDecodeError: If there is an error parsing the JSON file.
        TypeError: If either of the inputs is not a dictionary.
    """
    try:
        default_params = read_config_file(default_params_file)

        if not isinstance(default_params, dict):
            raise TypeError("Default parameters loaded are not a dictionary.")
        if not isinstance(baseline_params_input, dict):
            raise TypeError("Baseline params input is not a dictionary.")

        # Create baseline_params by updating default_params with baseline_params_input
        return combine_params_dicts(
            default_params,
            baseline_params_input,
            scenario_key,
            unflatten,
        )

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error reading YAML file: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error reading JSON file: {e}")


def params_grid_search(param_dict):
    """
    Generate a Polars DataFrame containing all combinations of parameters.

    Args:
        param_dict (dict): A dictionary where keys are parameter names and values are lists of possible values for those parameters.

    Returns:
        pl.DataFrame: A Polars DataFrame with each parameter as a column, and each row representing a unique combination of parameters.

    """
    # Create all possible combinations of parameters using itertools.product
    param_dict = flatten_dict(param_dict)
    for param_array in param_dict.values():
        if len(param_array) != len(set(param_array)):
            raise ValueError("Values are not unique for every parameter.")
    keys, values = zip(*param_dict.items())
    param_combinations = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    # Convert list of dictionaries to a Polars DataFrame
    df = pl.DataFrame(param_combinations)
    return df


def _vstack_dfs(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """
    Flexibly stack multiple dataframes together using most generous possbile concatenation
    Args:
    dfs: a list of polars DataFrames to combine
    Returns:
    concatenated DataFrame that has coerced all types with comon column names to same type and filled missing values with nulls.
    """
    return pl.concat(
        [df.select(sorted(df.columns)) for df in dfs], how="diagonal_relaxed"
    )


def df_to_simulation_dict(df):
    """
    Convert a Polars DataFrame into a simulation dictionary with rows as keys and columns as subkeys.
    If 'simulation' column exists, use it as keys for each row.

    Args:
        df (pl.DataFrame): The Polars DataFrame to convert.

    Returns:
        dict: A dictionary where each key is an index of the row or value from 'simulation' column,
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
        simulation_dict = {
            row["simulation"]: {
                col: row[col] for col in df.columns if col != "simulation"
            }
            for row in df.to_dicts()
        }

    else:
        # Use index as keys
        simulation_dict = {
            i: {col: row[col] for col in df.columns}
            for i, row in enumerate(df.to_dicts())
        }

    return simulation_dict


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


def get_caller(depth: int = 2) -> str:
    """
    Returns the file path of the traceback stack at the specified depth.
    Args:
        depth (int): The depth of the stack to retrieve the caller's file path. Default
            is 2, which means it will return the file path of the function that called the function that contains `get_caller()`.

    Returns:
        str: The file path of the caller at the specified depth.

    Example:
        # File: path/to/pkg.py
        from abmwrappers import utils
        def call_script():
            return utils.get_caller(depth=2)

        # File: path/to/script.py
        import pkg
        def some_function():
            caller_file = pkg.call_script()
            print(f"Called from: '{caller_file}'")

        some_function()

        >>> poetry run python path/to/script.py
        "Called from: 'path/to/script.py'
    """
    called_frame = inspect.currentframe()
    # Traverse up the call stack by the specified depth
    for _ in range(depth):
        if called_frame is None or called_frame.f_back is None:
            raise ValueError(
                f"Call stack is not deep enough for depth={depth}."
            )
        called_frame = called_frame.f_back
    caller_file = called_frame.f_code.co_filename
    return caller_file


def read_parquet_blob(
    container_name: str,
    blob_data_path: str,
    azb_config: dict,
    cred: object,
    clean: bool = True,
) -> pl.DataFrame:
    """
    Read a parquet file from an Azure Blob Storage container and return it as a Polars DataFrame.
    Modified from cdcent/cfa-rt-postprocessing/plotting internal functions.
    Parameters:
        azure_client (AzureClient): An instance of AzureClient to interact with Azure Blob Storage.
        container_name (str): The name of the Azure Blob Storage container.
        blob_data_path (str): The path to the parquet file within the container.
    Returns:
        pl.DataFrame: A Polars DataFrame containing the data
    """
    # Obtain container client from BlobServiceClient
    blob_service_client = blob_helpers.get_blob_service_client(
        azb_config, cred
    )

    local_path = f"/{blob_data_path}"
    os.makedirs(local_path)
    blob_helpers.download_directory(
        container_name=container_name,
        src_path=blob_data_path,
        dest_path=local_path,
        blob_service_client=blob_service_client,
    )
    df = pl.read_parquet(local_path)
    if clean:
        remove_directory_tree(local_path)
    return df


def read_nested_csvs(
    input_dir: str, filename: str, processing_fn: Callable = None
):
    """
    Reads nested CSV files from a specified directory and returns a Polars DataFrame.

    Args:
        input_dir (str): The directory containing the CSV files.
        filename (str): The name of the CSV file to read.
        processing_fn (Callable, optional): A function to process each DataFrame before concatenation.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the concatenated data from all CSV files.
    """
    if not filename.endswith(".csv") and not filename.endswith(".CSV"):
        if len(filename.split(".")) > 1:
            raise ValueError("The filename must end with '.csv' or '.CSV'.")
        else:
            filename += ".csv"
    csv_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files
        if f == filename
    ]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files named '{filename}' found in '{input_dir}'."
        )

    dfs = []
    for csv_file in csv_files:
        # Get the scenario and simulation keys of the path, if present.
        col_name_dict = column_keys_from_path(csv_file)
        col_keys = pl.DataFrame(col_name_dict)

        # If the specified csv is not empty, read and process
        if os.path.getsize(csv_file) > 0:
            df = pl.read_csv(csv_file)
            if processing_fn:
                df = processing_fn(df)

            # If there are identifying keys in the path known to abmwrappers, join them here
            if not col_keys.is_empty():
                df = df.join(col_keys, how="cross")

            # Append and then concatenate
            dfs.append(df)

    return _vstack_dfs(dfs)


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
                max_autoscale_nodes=autoscale_nodes,
                cache_blobfuse=cache_blobfuse,
                task_slots_per_node=task_slots_per_node,
            )

            client.create_pool(pool_name=pool_name)

        else:
            client.set_pool(pool_name)

        return client, blob_container_name, job_prefix

    except Exception as e:
        print(f"An error occurred: {e}")

        return None, None, None
