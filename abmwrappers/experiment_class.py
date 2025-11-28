import json
import os
import pickle
import random
import shutil
import warnings
from typing import Callable

import cfa_azure.helpers as azb_helpers
import griddler
import polars as pl
import yaml
from abctools import abc_methods
from abctools.abc_classes import SimulationBundle

from abmwrappers import utils


class Experiment:
    """
    Experiments are rulesets for generating and storing a dicitonary of SimulationBundles.
    Experiments offer methods for manipulating data across bundles or storing that data in specified file structures.
    """

    def __init__(
        self,
        config_file: str = None,
        experiments_directory: str = None,
        img_file: str = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the experiment as either a new experiment using a config file or restore the history of a previously compressed and saved experiment
        :param config_file: The path to the config file. If this file is not specified, the experiment will be initialized with the img_file
        :param experiments_directory: The path to the experiments directory. This is where the experiment will be saved and recognized. Required for config but not img_file
        :param img_file: The path to the image file. If this file is not specified, the experiment will be initialized with the config_file
        :param verbose: Whether top print progress statements and flag warnings for the user.
            Warnings that alter experiment behavior but will not necessarily raise an error are displayed regardless of verbose value
        :param kwargs: Accepts keywords in the Exeperiment class attributes to overwrite from loading or values for the parameter distribution dicts

        Examples:
        To initialize an experiment from a config file, the experiments directory and config file must be specified
        >>> experiment = Experiment(config_file="experiments/path/to/config.yaml", experiments_directory="experiments")

        To initialize from a history file, the experiments directory is already stored in the compressed data, so only img_file must be specified
        >>> experiment = Experiment(img_file="experiment/path/to/history.pkl")

        Optional arguments or overwrites fo the config can be provided as well through **kwargs in either case
        >>> experiment = Experiment(
                img_file="experiment/path/to/history.pkl",
                prior_distribtuion_dict={"mean": scipy.stats.norm(0,1.0)},
                tolerance_dict={0: 50.0, 1: 25.0, 2: 10.0},
            )

        """

        if config_file is None and img_file is None:
            raise ValueError(
                "No config file or image file specified. Please provide a config file."
            )
        if config_file is not None and img_file is not None:
            raise NotImplementedError(
                "Cannot currently both source from config_file and restore previous data from img_file. Aborting Experiment initialization."
            )
        if config_file is not None and experiments_directory is None:
            raise NotImplementedError(
                "Experiments directory must be specified if config file is provided."
            )

        self.verbose = verbose

        # If an image .pkl file containing experiment history is supplied, read from that
        if img_file is not None:
            if self.verbose:
                print(f"Restoring previous experiment from {img_file}")
            self.restore(img_file)
        # Otherwise, instantiate a new experiment with _load_config
        else:
            # Rread files and set directories
            self.config_file = config_file
            self.experiments_path = experiments_directory
            self.directory = os.path.dirname(config_file)

            # Remove the input directory from the directory path if it exists
            if self.directory.endswith("input") or self.directory.endswith(
                "input/"
            ):
                self.directory = os.path.dirname(self.directory)

            # Remove relative path
            if self.directory.startswith("./"):
                self.directory = self.directory.lstrip("./")

            # If the config file is relative to the experiments directoyr, adjust file naming
            if not os.path.exists(config_file):
                # Check if the config file exists in the experiments directory
                tmp = os.path.join(experiments_directory, config_file)
                if os.path.exists(tmp):
                    self.config_file = tmp
                    self.directory = os.path.dirname(tmp)

                    # Warn that config file is modified
                    warnings.warn(
                        f"Config file path {config_file} does not exist. Using {tmp} based in experiment directory {experiments_directory} instead.",
                        UserWarning,
                    )
                else:
                    raise FileNotFoundError(
                        f"Config file {config_file} does not exist from root or experiments directory."
                    )

            self._load_config()

        # Optional inputs for use in getters or wrappers later or attributes to overwrite from config
        for k, v in kwargs.items():
            if k in self.__dict__.keys():
                old_v = self.__dict__[k]
                self.__setattr__(k, v)
                if self.verbose:
                    print(
                        f"Replacing {k} current value {v} with old value {old_v}"
                    )
            elif k == "prior_distribution_dict":
                assert isinstance(v, dict)
                self.priors = utils.flatten_dict(v)
            elif k == "perturbation_kernel_dict":
                assert isinstance(v, dict)
                self.perturbation_kernel_dict = utils.flatten_dict(v)
            else:
                raise ValueError(
                    f"Keyword {k} is not an attribute and is not in `[prior_distribution_dict, perturbation_kernel_dict]`"
                )

        self._validate_seed_key()

        if "perturbation_kernel_dict" not in self.__dict__.keys():
            self.perturbation_kernel_dict = None
        if "priors" not in self.__dict__.keys():
            self.priors = None

    # --------------------------------------------
    # Loading and storing the experiment
    # --------------------------------------------

    def _set_or(
        self, key: str, lookup: dict, default=None, sto_key: str = None
    ):
        if sto_key is None:
            sto_key = key
        if key in lookup:
            self.__setattr__(sto_key, lookup[key])
        else:
            self.__setattr__(sto_key, default)

    def _validate_seed_key(self):
        params = self.get_default_params()
        if self.seed_variable_name not in params:
            possible_keys = filter(
                lambda x: "seed" in x or "Seed" in x, params.keys()
            )
            possible_keys = list(possible_keys)
            if len(possible_keys) == 1:
                self.seed_variable_name = possible_keys[0]
            else:
                raise ValueError(
                    "Unable to automatically detect seed variable name. Please specify seed_variable_name as an argument accesible in the base parameter input file."
                )

    def _load_config(self):
        """
        Load the parameters from the experimental config file
        Establishes theattributes for running Experiments and coordinates the file structure/pathing
        If `azure_batch`is set to True in the config, initializes the Azure client using config.toml and writes the azure batch config file

        This method should not currently be called manually as it is done during initializing a new Experiment object from a config path
        """
        with open(self.config_file, "r") as f:
            experimental_config = yaml.load(f, Loader=yaml.SafeLoader)

        # --------------------------------------------
        # Experiment name
        self._set_or(
            key="super_experiment_name",
            lookup=experimental_config["local_path"],
        )
        self._set_or(
            key="sub_experiment_name", lookup=experimental_config["local_path"]
        )

        # Check if the experiment name is specified
        if (
            self.super_experiment_name is not None
            and self.sub_experiment_name is not None
        ):
            specified_experiment_path = os.path.join(
                self.experiments_path,
                self.super_experiment_name,
                self.sub_experiment_name,
            )

            # Check if the config file directory specified is the same as the file string
            if (
                self.directory != specified_experiment_path
                and self.directory != self.experiments_path
            ):
                raise ValueError(
                    f"Config file directory {self.directory} does not match the experiment path {specified_experiment_path} specified in file and isn't at experiments folder root {self.experiments_path}."
                )

            if self.directory == self.experiments_path:
                os.makedirs(specified_experiment_path, exist_ok=True)
                print(
                    f"Writing experiment config file to {specified_experiment_path}"
                )
                config_file = os.path.join(
                    specified_experiment_path, "input", "config_abc.yaml"
                )
                if os.path.exists(config_file):
                    if self.verbose:
                        user_input = (
                            input(
                                f"Experiment config file {config_file} already exists. Overwrite experiment? (Y/N): "
                            )
                            .strip()
                            .upper()
                        )
                        if user_input != "Y":
                            print("Experiment terminated by user.")
                            return
                with open(config_file, "w") as f:
                    yaml.dump(experimental_config, f)

                self.directory = specified_experiment_path
        else:
            self.directory = self.experiments_path
            self.super_expeirment_name = None
            self.sub_experiment_name = None
            if self.verbose:
                print(
                    "Super and sub experiment must be specified together. Operating in root experiments directory instead"
                )

        # Check if scenario griddle file is specified
        self._set_or(
            key="griddle_file", lookup=experimental_config["local_path"]
        )

        # --------------------------------------------
        # Experimental components

        self.simulation_bundles = {}
        self.current_step = None

        self.data_path = os.path.join(self.directory, "data")
        os.makedirs(self.data_path, exist_ok=True)

        self.exe_file = experimental_config["local_path"]["exe_file"]
        if self.exe_file.endswith(".jar"):
            self.model_type = "gcm"
            self.input_file_type = "yaml"
            self.seed_variable_name = "randomSeed"
        else:
            self.model_type = "ixa"
            self.input_file_type = "json"
            self.seed_variable_name = "seed"
        self.default_params_file = experimental_config["local_path"][
            "default_params_file"
        ]

        # Store target data from file path if supplied
        self._set_or(
            key="target_data_file", lookup=experimental_config["local_path"]
        )
        if self.target_data_file is not None:
            self.target_data = pl.read_csv(self.target_data_file)
        else:
            if self.verbose:
                print(
                    "Target data file not listed, checking for target data specified manually in `experiment_conditions`"
                )
            self._set_or(
                key="target_data",
                lookup=experimental_config["experiment_conditions"],
            )

        self.seed = experimental_config["experiment_conditions"]["seed"]
        self.n_particles = experimental_config["experiment_conditions"][
            "samples_per_step"
        ]

        self._set_or(
            key="replicates_per_particle",
            lookup=experimental_config["experiment_conditions"],
            default=1,
            sto_key="replicates",
        )

        self.n_simulations = self.n_particles * self.replicates

        # Tolerance can be specified for each step or stored as a null in the case of not running abcsmc using the wrappers
        if "tolerance" in experimental_config["experiment_conditions"]:
            self.tolerance_dict = {
                int(k): float(v)
                for k, v in experimental_config["experiment_conditions"][
                    "tolerance"
                ].items()
            }

            for tolerance in self.tolerance_dict.values():
                if tolerance <= 0:
                    raise ValueError("Acceptance criteria improperly defined")
        else:
            self.tolerance_dict = None

        if self.verbose and (
            self.tolerance_dict is None or self.target_data is None
        ):
            warnings.warn(
                """Target data and/or tolerance dict are currently specified as None.
                    No ABC routines can be run without at least one tolerance step and target data
                    declared through config file.""",
                UserWarning,
            )

        # Write scenario key from manual declaration or use the first (typically only) element of the default params file
        self._set_or(
            key="scenario_key",
            lookup=experimental_config["experiment_conditions"],
            default=None,
        )
        if self.scenario_key is None:
            k = utils.read_config_file(self.default_params_file).keys()
            if self.verbose:
                (f"Scenario key not provided, using {list(k)[0]} by default")

            self.scenario_key = list(k)[0]

        # Store changed basleine params if provided
        self._set_or(
            key="changed_baseline_params",
            lookup=experimental_config["experiment_conditions"],
            default={},
        )

        # --------------------------------------------
        # Azure batch parameter loading
        self.azure_batch = experimental_config["azb"]["azure_batch"]

        if self.azure_batch:
            if "azb_config" not in experimental_config["azb"]:
                raise ValueError(
                    "Azure batch config under key `azb_config` must be supplied if `azure_batch` is True."
                )

            self.azb_config_path = os.path.join(
                self.directory, experimental_config["azb"]["azb_config_path"]
            )
            with open(self.azb_config_path, "w") as f:
                yaml.dump(experimental_config["azb"]["azb_config"], f)

            self.create_pool = experimental_config["azb"]["create_pool"]

            # Read the Azure Batch configuration from the config file relevant for Storage
            az_config_path = experimental_config["azb"]["azb_config"][
                "config_path"
            ]
            if os.path.exists(az_config_path):
                az_config = azb_helpers.read_config(az_config_path)
                self.storage_config = {"Storage": az_config["Storage"]}
            else:
                raise ValueError(
                    f"Path to Azure batch config {az_config_path} does not exist."
                )

            # Newly loaded experiments from config have access to a config.toml pathed to "azb_config"
            (
                self.client,
                self.blob_container_name,
                self.job_prefix,
            ) = utils.initialize_azure_client(
                self.azb_config_path,
                self.super_experiment_name,
                self.create_pool,
            )
            # The credential is stored separately for read privileges in compressed experiments
            self.cred = self.client.cred
        else:
            self.cred = None
            self.storage_config = None
            self.blob_container_name = None

    def save(self, output_file: str = None):
        """
        Lossy compression to save a reproducible savepoint
        Stores all information except for simulation bundle results to compressed pickle file
        Important to track across steps and reproduce the experiment are
        Directory data:
            - config file
            - directory
            - data path
            - exe_file and model type
            - Azure batch and storage credential
            - blob container name
        Simulation bundle data:
            - current step
            - distances
            - weights
            - accepted
            - inputs
            - changed baseline parameters
        Experiment data:
            - tolerance steps
            - target data
            - scenario key
            - seed
            - priors
            - perturbation kernels
            - replicate counts (particle count and per particle)
        """

        # Create a dictionary to store the data
        if output_file is None:
            output_file = os.path.join(
                self.data_path, "experiment_history.pkl"
            )

        data = {
            "config_file": self.config_file,
            "directory": self.directory,
            "data_path": self.data_path,
            "exe_file": self.exe_file,
            "griddle_file": self.griddle_file,
            "azure_batch": self.azure_batch,
            "model_type": self.model_type,
            "input_file_type": self.input_file_type,
            "default_params_file": self.default_params_file,
            "changed_baseline_params": self.changed_baseline_params,
            "target_data": self.target_data,
            "seed": self.seed,
            "seed_variable_name": self.seed_variable_name,
            "n_particles": self.n_particles,
            "replicates": self.replicates,
            "skinny_bundles": {
                step: {
                    key: value
                    for key, value in bundle.__dict__.items()
                    if key != "results"
                }
                for step, bundle in self.simulation_bundles.items()
            },
            "scenario_key": self.scenario_key,
            "current_step": self.current_step,
            "tolerance_dict": self.tolerance_dict,
            "priors": self.priors,
            "perturbation_kernel_dict": self.perturbation_kernel_dict,
            "cred": self.cred,
            "storage_config": self.storage_config,
            "blob_container_name": self.blob_container_name,
        }

        # Save the data to a compressed pickle file
        with open(output_file, "wb") as f:
            pickle.dump(data, f)

    def restore(self, input_file: str):
        """
        Load the experiment from a compressed pickle file
        :param input_file: The path to the compressed pickle file
        """

        # Load the data from the compressed pickle file
        with open(input_file, "rb") as f:
            data = pickle.load(f)

        history = {}
        for step, bundle in data["skinny_bundles"].items():
            bundle_to_append = SimulationBundle(
                inputs=bundle["inputs"],
                step_number=bundle["_step_number"],
                baseline_params=bundle["_baseline_params"],
                seed_variable_name=data["seed_variable_name"],
            )
            if "distances" in bundle:
                bundle_to_append.distances = bundle["distances"]
            if "weights" in bundle:
                bundle_to_append.weights = bundle["weights"]
            if "accepted" in bundle:
                bundle_to_append.accepted = bundle["accepted"]
            if "acceptance_weights" in bundle:
                bundle_to_append.acceptance_weights = bundle[
                    "acceptance_weights"
                ]
            history.update({step: bundle_to_append})

        # Unpack the data into the experiment object
        self.config_file = data["config_file"]
        self.directory = data["directory"]
        self.data_path = data["data_path"]
        self.exe_file = data["exe_file"]
        self.griddle_file = data["griddle_file"]
        self.azure_batch = data["azure_batch"]
        self.model_type = data["model_type"]
        self.input_file_type = data["input_file_type"]
        self.default_params_file = data["default_params_file"]
        self.changed_baseline_params = data["changed_baseline_params"]
        self.target_data = data["target_data"]
        self.seed = data["seed"]
        self.seed_variable_name = data["seed_variable_name"]
        self.n_particles = data["n_particles"]
        self.replicates = data["replicates"]
        self.n_simulations = self.n_particles * self.replicates
        self.simulation_bundles = history
        self.scenario_key = data["scenario_key"]
        self.current_step = data["current_step"]
        self.tolerance_dict = data["tolerance_dict"]
        self.priors = data["priors"]
        self.perturbation_kernel_dict = (
            data["perturbation_kernel_dict"]
            if "perturbation_kernel_dict" in data
            else None
        )
        self.cred = data["cred"]
        self.storage_config = data["storage_config"]
        self.blob_container_name = data["blob_container_name"]

    def delete(
        self,
        prefix: str = None,
        file_type: str = None,
    ):
        """
        Deletes items (files/folders) within a given experiment,
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
        if prefix is None and file_type is None:
            utils.remove_directory_tree(self.directory)
        else:
            # List all items in the directory
            for item in os.listdir(self.data_path):
                # Construct full path to item
                item_path = os.path.join(self.data_path, item)
                # Check if item matches prefix condition if provided
                if prefix is not None and not item.startswith(prefix):
                    continue
                # Check if item matches file type condition if provided
                if file_type is not None:
                    # If it's a folder or doesn't match the extension, skip it
                    if os.path.isdir(item_path) or not item.endswith(
                        file_type
                    ):
                        continue
                # Delete files or folders that meet conditions
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    # --------------------------------------------
    # File management
    # --------------------------------------------

    def parquet_from_path(
        self, path: str, verbose: bool = True
    ) -> pl.DataFrame:
        if self.azure_batch:
            df = utils.read_parquet_blob(
                container_name=self.blob_container_name,
                blob_data_path=path,
                azb_config=self.storage_config,
                cred=self.cred,
                verbose=verbose,
            )
        else:
            df = pl.scan_parquet(path).collect()
        return df

    def blob_directory_exists(self, src_path: str) -> bool:
        """
        Search for a directory wihtin a blob container. The blob container name should not be included in the directory path

        Args:
        src_path: String label for the directory within the blob to check for existence. Including the blob container name will raise an unimplemented error.
        """
        if self.blob_container_name in src_path:
            raise NotImplementedError(
                "The blob container name should not be included in the source directory path"
            )
        if self.azure_batch:
            return utils.check_virtual_directory_existence(
                self.storage_config,
                self.cred,
                self.blob_container_name,
                src_path,
            )
        else:
            raise ValueError(
                "Cannot determine if blob directory exists unless azure batch is set to True."
            )

    def store_products(
        self,
        sim_indices: list[int] | pl.Series | int,
        products: list[str] | str | None = None,
        products_output_dir: str | None = None,
    ):
        if not isinstance(sim_indices, list):
            if isinstance(sim_indices, pl.Series):
                sim_indices = sim_indices.to_list()
            elif isinstance(sim_indices, int):
                sim_indices = [sim_indices]
            else:
                raise NotImplementedError(
                    "Simulation indices must be specified using int, list[int], or pl.Series"
                )

        if not isinstance(products, list):
            products = [products]

        if products_output_dir is None:
            output_dir = self.data_path
        else:
            output_dir = products_output_dir

        for simulation_index in sim_indices:
            sim_bundle = self.bundle_from_index(simulation_index)

            if "distances" in products:
                distance_data_part_path = (
                    f"{output_dir}/distances/simulation={simulation_index}/"
                )

                # Write the distance data to a Parquet file with part path storing the simulation column
                os.makedirs(distance_data_part_path, exist_ok=True)
                sim_bundle.distances.filter(
                    pl.col("simulation") == simulation_index
                ).write_parquet(distance_data_part_path + "data.parquet")

            if "simulations" in products:
                simulation_data_part_path = (
                    f"{output_dir}/simulations/simulation={simulation_index}/"
                )

                # Write the simulation data to a Parquet file with part path storing the simulation column
                os.makedirs(simulation_data_part_path, exist_ok=True)
                sim_bundle.results.filter(
                    pl.col("simulation") == simulation_index
                ).write_parquet(simulation_data_part_path + "data.parquet")
            if "params" in products:
                inputs_data_part_path = (
                    f"{output_dir}/params/simulation={simulation_index}/"
                )
                sim_bundle.inputs.filter(
                    pl.col("simulation") == simulation_index
                ).write_parquet(inputs_data_part_path + "data.parquet")

    def gather_distances(self, input_dir: str = None, overwrite: bool = False):
        """
        Read distances and simulation results into the simulation bundle history
        Currently yields OSError when called on a mounted blob container input directory
        """
        if not input_dir:
            input_dir = self.data_path

        distances = self.parquet_from_path(f"{input_dir}/distances/")

        if distances.is_empty():
            raise ValueError("No distances found in the input directory.")

        if (
            hasattr(self.simulation_bundles[self.current_step], "distances")
            and not overwrite
        ):
            raise ValueError(
                "Simulation bundle already has distances. Please clear the simulation bundle before reading new distances."
            )
        current_distances = (
            distances.with_columns(
                pl.col("simulation")
                .map_elements(self.step_from_index, return_dtype=pl.Int64)
                .alias("step")
            )
            .filter(pl.col("step") == self.current_step)
            .drop("step")
        )
        self.simulation_bundles[
            self.current_step
        ].distances = current_distances

    def read_results(
        self,
        filename: str,
        input_dir: str | None = None,
        output_dir: str | None = None,
        data_read_fn: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
        write: bool = False,
        partition_by: list[str] | str | None = None,
    ) -> pl.DataFrame:
        """
        Function to read results from simulation output stored as nested CSV files or as hive-partitioned parquets.
        The default behavior is to search for an already-extant parquet file folder name or summarized CSV at the root input data path
        If neither summarized file is found, the function will attempt to read nested CSV files from the input directory

        Args:
            :param filename: The name of the CSV file or hive-partitioned parquet to read from. If not specified, defaults to "simulations"
            :param input_dir: The directory to read the file from. If not specified, defaults to the experiment data path for local or Azure implementation
            :param output_dir: The directory to store processed files in. If not specified, defaults to the same as the input directory on local or the current directory data path for Azure downloads.
            :param data_read_fn: A function to preprocess the data before combining it withh other data during reading from CSV.
                - If specified with a partitioned parquet file, the function is called on the data set in aggregate.
                - If called on a CSV file that was written by this method, the data processing function is skipped and the user is informed to run further post processing outside the read results method.
                - If not specified, defaults to None
            :param store: Whether to store the data in the input directory as a parquet or CSV file.
                - If True, the data is stored as a partitioned parquet file with the name of the file without extension or is stored as a CSV at the input directory root.
            :param partition_by: A list of columns to partition the data by when storing it as a parquet file. If not specified, defaults to None
        """
        # Default to dat path and the simulations parquet file
        if not input_dir:
            if self.azure_batch:
                input_dir = f"{self.sub_experiment_name}/data"
                output_dir = os.path.join(self.directory, "data")
            else:
                input_dir = output_dir = self.data_path

        if partition_by is not None and not isinstance(partition_by, list):
            partition_by = [partition_by]

        # Read from hive-paritioned parquet if it exists and filename is not a file
        if os.path.exists(f"{input_dir}/{filename}") or self.azure_batch:
            if len(filename.split(".")) == 1:
                # Special case for names in "products"
                data = self.parquet_from_path(f"{input_dir}/{filename}/")
                if data_read_fn is not None:
                    warnings.warn(
                        "Preprocessing function specified for a hive-partitioned parquet file. Please ensure that the function is compatible with the data format.",
                        UserWarning,
                    )
                    data = data_read_fn(data)
            elif filename.endswith(".csv") or filename.endswith(".CSV"):
                # Read from a summarized CSV file if it exists
                data = pl.read_csv(f"{input_dir}/{filename}")
                if data_read_fn is not None:
                    warnings.warn(
                        "Skipping use of data processing function. "
                        "The existing CSV file could only have been created by a previous read_results and would have been manipulated by a `data_read_fn` already. "
                        "If you would like to further process the data, please read the results and manipulate them manually outside the Experiment methods.",
                        UserWarning,
                    )
            else:
                raise ValueError(
                    f"Filename {filename} to be read must be a CSV file or a hive-partitioned parquet folder if it already exists."
                )
        # Otherwise attempt reading nested CSVs
        else:
            # Pre-proceesing fn is applied piece-wise to CSV
            data: pl.DataFrame = utils.read_nested_csvs(
                input_dir, filename, data_read_fn
            )

        if write:
            out_file = filename.split(".")[0]
            if partition_by is not None:
                os.makedirs(f"{output_dir}/{out_file}/", exist_ok=True)
                data.write_parquet(
                    f"{output_dir}/{out_file}/",
                    use_pyarrow=True,
                    pyarrow_options={"partition_cols": partition_by},
                )
            else:
                data.write_csv(f"{output_dir}/{out_file}.csv")
        return data

    # --------------------------------------------
    # Input parameter handling
    # --------------------------------------------

    def get_default_params(self, step: int = None) -> dict:
        """
        Get the baseline parameters for a specific step in the experiment history.
        If no step is specified, returns the baseline parameters for the current step.
        """
        if step is None:
            step = self.current_step
        if step is not None:
            if step not in self.simulation_bundles:
                raise ValueError(
                    f"Step {step} does not exist in the simulation bundles."
                )
            return self.simulation_bundles[step].baseline_params[
                self.scenario_key
            ]
        else:
            baseline_params, _summary_string = utils.load_baseline_params(
                self.default_params_file,
                self.changed_baseline_params,
                self.scenario_key,
            )
            return baseline_params[self.scenario_key]

    def get_default_value(self, param: str | dict, step: int | None = None):
        """
        Get the value of a specific parameter for a specific step in the experiment history.
        The parameter can be specified as a single string
        Hierarchical parameters can be retrieved with a nested dictionary or SEPARATOR string, or list
        >>> param={'my_distribution': {'norm': {'mean'}}}
        >>> param='my_distribution>>>norm>>>mean'
        If no step is specified, returns the value for the current step.
        """
        baseline_params = self.get_default_params(step)

        if utils.FLATTENED_PARAM_CONNECTOR in param or isinstance(param, dict):
            baseline_params = utils.flatten_dict(baseline_params)

        if isinstance(param, dict):
            param_dict = utils.flatten_dict(param)
            keys = list(param_dict.keys())
            if len(keys) > 1:
                raise ValueError(
                    "Cannot obtain more than one default parameter value at once. Use self.get_default_params() for all values as dict"
                )
            k = keys[0]
            v = str(param_dict[k]).replace("{'", "").replace("'}", "")
            param = f"{k}{utils.FLATTENED_PARAM_CONNECTOR}{v}"
        if param not in baseline_params:
            raise ValueError(
                f"Parameter {param} not found in baseline parameters for step {step}."
            )
        return baseline_params[param]

    def collect_inputs(
        self, sim_indices: list[int] | int | None = None
    ) -> pl.DataFrame:
        if sim_indices is not None and not isinstance(sim_indices, list):
            sim_indices = [sim_indices]

        if os.path.exists(f"{self.data_path}/params/"):
            if sim_indices is None:
                inputs = self.parquet_from_path(f"{self.data_path}/params/")
            else:
                inputs = (
                    self.parquet_from_path(f"{self.data_path}/params/")
                    .filter(pl.col("simulation").is_in(sim_indices))
                    .sort(pl.col("simulation"))
                )
        else:
            inputs_list = []
            for simulation_index in sim_indices:
                sim_bundle = self.bundle_from_index(simulation_index)
                inputs_list.append(
                    sim_bundle.inputs.filter(
                        pl.col("simulation") == simulation_index
                    )
                )
            inputs = utils._vstack_dfs(inputs_list).sort("simulation")

        return inputs

    def write_inputs_index(
        self,
        simulation_index: int,
        scenario_key: str | None = None,
    ) -> str:
        """
        Write a single simulation's input file
        :param simulation_index: The index of the simulation to write
        :param write_inputs_cmd: The command to run that sources outside code to transform
            a base yaml file into readable inputs for execution
            A default of None will return only the formatted inputs from the simulation bundle
        :param scenario_key: The key to use for the scenario in the simulation bundle input dictionary. Will default to self.scenario_key if unspecified

        In general, self.directory/data/input for simulation input files
        Use a tmp space for files that are intermediate products
        """

        sim_bundle = self.bundle_from_index(simulation_index)

        input_dict = utils.df_to_simulation_dict(sim_bundle.inputs)

        # If scenario key is not specified, use the default scenario key
        if scenario_key is None:
            scenario_key = self.scenario_key

        simulation_params, _summary_string = utils.combine_params_dicts(
            sim_bundle._baseline_params,
            input_dict[simulation_index],
            scenario_key=scenario_key,
        )

        input_dir = os.path.join(self.data_path, "input")
        os.makedirs(input_dir, exist_ok=True)

        input_file_name = (
            f"simulation_{simulation_index}.{self.input_file_type}"
        )

        if self.verbose and simulation_index % self.n_simulations == 0:
            print(
                "Writing exe file inputs across the SimulationBundle indices"
            )
        formatted_inputs = utils.abm_parameters_writer(
            params=simulation_params,
            output_type=self.input_file_type,
            unflatten=True,
        )
        with open(os.path.join(input_dir, input_file_name), "w") as f:
            f.write(formatted_inputs)
        return os.path.join(input_dir, input_file_name)

    def write_inputs_from_griddle(
        self,
        input_griddle: str | dict = None,
        scenario_key: str = None,
        unflatten: bool = True,
        seed_variable_name: str | None = None,
        sample_posterior: bool = False,
        n_samples: int = 5,
    ):
        """
        Write simulation inputs from a griddler parameter set
        :param input_griddle: The path to the griddler parameter set
        :param scenario_key: The key to use for the scenario in the simulation bundle input dictionary. Will default to self.scenario_key if unspecified
        :param unflatten: Whether to unflatten the parameter set or not
        :return: None

        This function will read the griddler parameter set and write the simulation inputs to the input directory
        If griddle scenrio varying  parameters are not to be written into the parameter input files, specify them with "scenario_" asd a prefix
        """
        if input_griddle is None:
            if self.griddle_file is not None:
                input_griddle = self.griddle_file
            else:
                raise ValueError(
                    "No griddler parameter set specified. Please provide a griddler parameter set."
                )
        if scenario_key is None:
            scenario_key = self.scenario_key
        if seed_variable_name is None:
            seed_variable_name = self.seed_variable_name

        # Load the parameter sets
        if isinstance(input_griddle, str):
            with open(input_griddle, "r") as fp:
                if input_griddle.lower().endswith(
                    ".yaml"
                ) or input_griddle.lower().endswith(".yml"):
                    raw_griddle = yaml.safe_load(fp)
                elif input_griddle.lower().endswith(".json"):
                    raw_griddle = json.load(fp)
                else:
                    raise NotImplementedError(
                        "Griddle input from string must be a yaml or json file"
                    )
        elif isinstance(input_griddle, dict):
            raw_griddle = input_griddle

        griddle = griddler.parse(raw_griddle)
        par_sets = griddle
        if sample_posterior:
            if n_samples is None or n_samples <= 0:
                raise ValueError(
                    "num_samples must be provided when sample_posterior is True"
                )
            sampled_posterior = self.sample_posterior(n_samples)

        ## These will work fine for changed_baseline_params but will need a method for dropping the higher level key
        ## Change to combine param dicts

        baseline_params, _summary_string = utils.load_baseline_params(
            self.default_params_file,
            self.changed_baseline_params,
            scenario_key,
            unflatten,
        )
        input_dir = os.path.join(self.data_path, "input")
        os.makedirs(input_dir, exist_ok=True)
        simulation_index = 0

        iterator = self.replicates if not sample_posterior else n_samples
        seeds = random.sample(range(0, 2**32), iterator)
        for par in par_sets:
            # Remove the griddler scenario keys from the parameter set
            to_remove = []
            for key in par.keys():
                if key.startswith("scenario"):
                    to_remove.append(key)
            for key in to_remove:
                par.pop(key)
            for counter in range(iterator):
                if sample_posterior:
                    sample = sampled_posterior[counter]
                    sample_dict = sample.to_dict(as_series=False)
                    for key, value in sample_dict.items():
                        par[key] = value[0]
                par[seed_variable_name] = seeds[counter]
                newpars, _summary = utils.combine_params_dicts(
                    baseline_dict=baseline_params,
                    new_dict=par,
                    scenario_key=scenario_key,
                    overwrite_unnested=True,
                    unflatten=unflatten,
                    preserve_keys=["CensusTract"],
                )

                input_file_name = (
                    f"simulation_{simulation_index}.{self.input_file_type}"
                )
                with open(os.path.join(input_dir, input_file_name), "w") as f:
                    json.dump(newpars, f, indent=4)
                simulation_index += 1

    def sample_posterior(experiment, n_samples):
        max_step = max(experiment.tolerance_dict.keys())
        accepted = experiment.simulation_bundles[max_step].accepted.filter(
            pl.col("accept_bool")
        )
        # Join weights and accepted on the "simulation" column
        accepted = accepted.drop(
            [
                "acceptance_weight",
                "distance",
                "accept_bool",
                experiment.seed_variable_name,
            ]
        )
        sample_keys = random.choices(
            population=accepted.select("simulation").to_series().to_list(),
            k=n_samples,
        )
        samples = [
            accepted.filter(pl.col("simulation") == key).drop(["simulation"])
            for key in sample_keys
        ]
        return samples

    # --------------------------------------------
    # Simulation bundles and ABC SMC functions
    # --------------------------------------------

    @property
    def distances(self) -> pl.DataFrame:
        dfs = [
            bundle.distances.with_columns(pl.lit(step).alias("step"))
            for step, bundle in self.simulation_bundles.items()
        ]
        return utils._vstack_dfs(dfs)

    @property
    def inputs(self) -> pl.DataFrame:
        dfs = [
            bundle.inputs.with_columns(pl.lit(step).alias("step"))
            for step, bundle in self.simulation_bundles.items()
        ]
        return utils._vstack_dfs(dfs)

    @property
    def weights(self) -> pl.DataFrame:
        dfs = [
            bundle.weights.with_columns(pl.lit(step).alias("step"))
            for step, bundle in self.simulation_bundles.items()
        ]
        return utils._vstack_dfs(dfs)

    def initialize_simbundle(
        self,
        prior_distribution_dict: dict = None,
        changed_baseline_params: dict = {},
        scenario_key: str = None,
        unflatten: bool = True,
        seed_variable_name: str | None = None,
    ) -> SimulationBundle:
        if len(changed_baseline_params) > 0:
            if self.verbose:
                print(
                    "Changed baseline parameters specified by function. Updating baseline parameters and overwriting common keys."
                )
                self.changed_baseline_params.update(changed_baseline_params)

        if scenario_key is None:
            scenario_key = self.scenario_key
        if seed_variable_name is None:
            seed_variable_name = self.seed_variable_name
        elif self.scenario_key is not None:
            warnings.warn(
                f"Overwriting scenario key {self.scenario_key} with {scenario_key}"
            )
            self.scenario_key = scenario_key
        # Create baseline_params by updating default params
        baseline_params, _summary_string = utils.load_baseline_params(
            self.default_params_file,
            self.changed_baseline_params,
            scenario_key,
            unflatten,
        )

        # Save prior distribution for use in experiment and draw
        if "priors" not in self.__dict__.keys():
            self.priors = None
        if prior_distribution_dict is not None:
            if (
                self.priors is not None
                and self.priors
                is not utils.flatten_dict(prior_distribution_dict)
            ):
                raise ValueError(
                    "Different prior distribution already specified on experiment initiation. Please declare only one set of priors for inital simulation bundle."
                )
            self.priors = utils.flatten_dict(prior_distribution_dict)

        if self.priors is not None:
            input_df = abc_methods.draw_simulation_parameters(
                params_inputs=self.priors,
                n_parameter_sets=self.n_particles,
                replicates_per_particle=self.replicates,
                seed=self.seed,
                seed_variable_name=seed_variable_name,
            )
        else:
            warnings.warn(
                "No prior distribution specified. Making empty inputs of simulation index and random seed for number of replicates instead. ABC SMC will fail if called.",
                UserWarning,
            )
            input_df = pl.DataFrame(
                {
                    "simulation": range(self.replicates),
                    seed_variable_name: [
                        random.randint(0, 2**32)
                        for _ in range(self.replicates)
                    ],
                }
            )

        # Create simulation bundle
        sim_bundle = SimulationBundle(
            inputs=input_df,
            step_number=0,
            baseline_params=baseline_params,
            seed_variable_name=self.seed_variable_name,
        )

        # Add simulation bundle to dictionary
        self.simulation_bundles[sim_bundle._step_number] = sim_bundle
        self.current_step = 0

        return sim_bundle

    def step_from_index(self, simulation_index: int) -> int:
        return int(simulation_index / self.n_simulations)

    def bundle_from_index(self, simulation_index: int) -> SimulationBundle:
        step_id = self.step_from_index(simulation_index)
        if step_id not in self.simulation_bundles:
            raise ValueError(
                f"Simulation bundle for step {step_id} not found."
            )
        if step_id != self.current_step:
            warnings.warn(
                f"Index {simulation_index} exists in step {step_id}, which is not the current experiment step {self.current_step}. Proceeding...",
                UserWarning,
            )
        return self.simulation_bundles[step_id]

    def run_index(
        self,
        simulation_index: int,
        data_filename: str | None = None,
        data_read_fn: Callable[[str], pl.DataFrame] | None = None,
        distance_fn: Callable[
            [
                pl.DataFrame,  # results data
                pl.DataFrame | dict | float | int,  # target data
            ],
            float | int,  # returns
        ]
        | None = None,
        products: list[str] | str | None = None,
        products_output_dir: str | None = None,
        scenario_key: str | None = None,
        cmd: str | None = None,
        compress: bool = True,
        clean: bool = False,
    ):
        """
        Function that writes input files for the executeable from a valid simulation index in the SimulationBundle history
        The executeable is then called and products from the outputs of the model are returned

        Args:
            :param simulation_index: integer index of a viable simulation from the cumulative simulation runs across SimulationBundle objects in the experiment history
            :param distance_fn: Distance function supplied that accepts target data and results data
                The results simulation data must be supplied as a simulation data frame, but the target data
            :param data_read_fn: Post-processing function to be applied on raw_outputs
            :param products: products parquet files to be created from the post-processing of raw output results,
                Currently implemented values are `["simulations", "distances"]` for the results and distances attribute of SimulationBundles, respectively
            :param products_output_dir: Output directory path, defaults to the Experiment data.path,
            :param scenario_key: Scenario key used to access parameters when combining dictionaries,
            :param cmd: Command to execute the model .If None, the default command for the model type is run,
            :param clean: Argument to remove the raw_output folders of the data directory after the simulation is run. Default false
        """
        if clean and not compress:
            raise ValueError(
                "Cannot clean remove raw output without saving the simulation results as a combined file."
            )
        if products is None:
            products = ["distances", "simulations"]
        else:
            if not isinstance(products, list):
                products = [products]

        if data_read_fn is None:
            if data_filename is not None:

                def data_read_fn(outputs_dir: str) -> pl.DataFrame:
                    """
                    Default data read function that reads the raw output data from the specified path
                    """
                    output_file_path = os.path.join(outputs_dir, data_filename)
                    if os.path.exists(output_file_path):
                        df = pl.read_csv(output_file_path)
                    else:
                        raise FileNotFoundError(
                            f"{output_file_path} does not exist."
                        )

                    return df

            else:
                raise ValueError(
                    "Data processing function must be provided if not provided with a data filename to read from raw output."
                )

        if scenario_key is None:
            scenario_key = self.scenario_key
        elif self.scenario_key is not None:
            warnings.warn(
                f"Overwriting scenario key {self.scenario_key} with {scenario_key}"
            )
            self.scenario_key = scenario_key

        input_file_path = self.write_inputs_index(
            simulation_index,
            scenario_key=scenario_key,
        )
        simulation_output_path = os.path.join(
            self.data_path, "raw_output", f"simulation_{simulation_index}"
        )
        os.makedirs(simulation_output_path, exist_ok=True)

        # This will require the default command line for model run if None
        if cmd is None:
            cmd = utils.write_default_cmd(
                input_file=input_file_path,
                output_dir=simulation_output_path,
                exe_file=self.exe_file,
                model_type=self.model_type,
            )

        utils.run_model_command_line(cmd, model_type=self.model_type)

        sim_bundle: SimulationBundle = self.bundle_from_index(simulation_index)
        index_df = data_read_fn(simulation_output_path).with_columns(
            pl.lit(simulation_index).alias("simulation")
        )
        if hasattr(sim_bundle, "results"):
            sim_bundle.results = utils._vstack_dfs(
                [sim_bundle.results, index_df]
            )
        else:
            sim_bundle.results = index_df

        if "distances" in products and distance_fn is not None:
            sim_bundle.calculate_distances(self.target_data, distance_fn)

        if compress:
            self.store_products(
                sim_indices=[simulation_index],
                products=products,
                products_output_dir=products_output_dir,
            )

        if clean:
            # Delete raw_output file if cleaning intermediates
            for file in os.listdir(simulation_output_path):
                os.remove(os.path.join(simulation_output_path, file))

    def run_step(
        self,
        data_read_fn: Callable[[str], pl.DataFrame] | None = None,
        data_filename: str | None = None,
        distance_fn: Callable[
            [
                pl.DataFrame,  # results data
                pl.DataFrame | dict | float | int,  # target data
            ],
            float | int,  # returns
        ]
        | None = None,
        products: list[str] | str | None = None,
        step: int | None = None,
        products_output_dir: str | None = None,
        scenario_key: str | None = None,
        compress: bool = True,
        clean: bool = False,
    ):
        """
        Function to run the simulations for a particular step in the experiment history, which defaults to the current step
        All results are generated and then processed into a single joint data frame

        Args:
            :param data_read_fn: callable function to process the raw output data being stored as a data frame
            :param products: list of which products to store as processed data frames
            :param step: int of the step to run simulations from. Defaults to the current or alternatively initializes the history
        """
        if products is None:
            if distance_fn is not None:
                products = ["simulations", "distances"]
            else:
                products = ["simulations"]
        else:
            if not isinstance(products, list):
                products = [products]
        if products_output_dir is None:
            products_output_dir = self.data_path

        if step is None:
            step = self.current_step

        if clean and not compress:
            warnings.warn(
                f"The step is set to clean and remove intermediary files without storing results in compressed parquet file format. Ensure that this is intentional. Data created is temporarily available through the SimulationBundle step {step} of the Experiment history",
                UserWarning,
            )

        # Generate products from current step if it exists or initialize
        if step is not None:
            simbundle = self.simulation_bundles[step]
        else:
            simbundle = self.initialize_simbundle()

        # Run the simulation
        for index in simbundle.inputs["simulation"]:
            self.run_index(
                simulation_index=index,
                distance_fn=None,
                data_read_fn=data_read_fn,
                data_filename=data_filename,
                products=products,
                products_output_dir=products_output_dir,
                scenario_key=scenario_key,
                compress=False,
                clean=clean,
            )

        # Calculate distance summary metrics and store/clean
        if "distances" in products:
            simbundle.calculate_distances(self.target_data, distance_fn)
        if compress:
            self.store_products(
                sim_indices=simbundle.inputs["simulation"],
                products=products,
                products_output_dir=products_output_dir,
            )

    def resample(
        self,
        perturbation_kernel_dict: dict = None,
        prior_distribution_dict: dict = None,
    ):
        """
        Resample the simulation bundle from the current step and create next step inputs
        Uses the curent history to resample inputs for the next step, conditional on the accepted results and distacnes being present
        First calculates the weights for each particle parameter set, equal to the normalized proprotion of accepted simulations in a particle for step 0
        Then resamples the accepted simulations from the previous step, perturbing values afterwards for each particle sampled.
        """
        # Save priors and perturbation kernel distribution for use in experiment if not already stored and draw
        # Priors must be consistent for the whole experiment
        if "priors" not in self.__dict__.keys():
            self.priors = None
        if prior_distribution_dict is not None:
            if (
                self.priors is not None
                and self.priors
                is not utils.flatten_dict(prior_distribution_dict)
            ):
                raise ValueError(
                    "Different prior distribution already specified on experiment initiation. Please declare only one set of priors for inital simulation bundle."
                )
            self.priors = utils.flatten_dict(prior_distribution_dict)

        # Perturbation kernels do not necessarily need to be the same across steps for resample
        if "perturbation_kernel_dict" not in self.__dict__.keys():
            self.perturbation_kernel_dict = None
        if perturbation_kernel_dict is not None:
            if (
                self.perturbation_kernel_dict is not None
                and self.perturbation_kernel_dict
                is not utils.flatten_dict(perturbation_kernel_dict)
            ):
                warnings.warn(
                    "Different perturbation distribution already specified on experiment initiation. The perturbation kernel will be replaced."
                )
            self.perturbation_kernel_dict = utils.flatten_dict(
                perturbation_kernel_dict
            )

        # Ensure that both priors and perturbation kernels are present for resample
        if self.priors is None:
            raise ValueError(
                "Prior distribution not specified on experiment initiation or during resample."
            )
        if self.perturbation_kernel_dict is None:
            raise ValueError(
                "Perturbation kernel not specified on experiment initiation or during resample."
            )

        # Validate that the keys in perturbation kernel and prior dictionary are the same
        if self.perturbation_kernel_dict and self.priors:
            kernel_keys = set(self.perturbation_kernel_dict.keys())
            prior_keys = set(self.priors.keys())
            if kernel_keys != prior_keys:
                raise ValueError(
                    f"Mismatch between perturbation kernel keys {kernel_keys} and prior distribution keys {prior_keys}."
                )

        # Accept or reject simulation distances in current step
        current_bundle: SimulationBundle = self.simulation_bundles[
            self.current_step
        ]
        tolerance = self.tolerance_dict[self.current_step]
        current_bundle.accept_stochastic(
            tolerance=tolerance,
        )
        if self.verbose:
            dtmp = current_bundle.distances["distance"]
            print(
                f"Distances in step {self.current_step} have: Min={dtmp.quantile(0.0)}, Q1={dtmp.quantile(0.25)}, Q2={dtmp.quantile(0.5)}, Q3={dtmp.quantile(0.75)}"
            )

        if self.current_step > 0:
            prev_bundle: SimulationBundle = self.simulation_bundles[
                self.current_step - 1
            ]
            if self.verbose:
                print("Calculating weights during a resample")
                print(
                    f"Using {current_bundle.accepted.filter(pl.col('acceptance_weight') > 0.0).select(pl.len()).item()} nonzero simulation weights"
                )

            current_bundle.weights = abc_methods.calculate_weights_abcsmc(
                current=current_bundle,
                previous=prev_bundle,
                prior_distributions=self.priors,
                perturbation_kernels=self.perturbation_kernel_dict,
                normalize=True,
            )
            if self.verbose:
                print(
                    f"Calculated {current_bundle.weights.filter(pl.col('weight') > 0.0).select(pl.len()).item()} nonzero particle weights for step {self.current_step}"
                )

        else:
            current_bundle.weights = current_bundle.accepted.with_columns(
                (
                    pl.col("acceptance_weight") / pl.sum("acceptance_weight")
                ).alias("weight")
            ).select(["simulation", "weight"])

        if current_bundle.get_accepted().is_empty():
            raise ValueError(
                "No accepted simulations found in the current step. Cannot resample. Examine the distribution of distances and re-specify tolerance dictionary."
            )

        new_inputs = abc_methods.resample(
            accepted_simulations=current_bundle.get_accepted(),
            n_samples=self.n_particles,
            replicates_per_sample=self.replicates,
            perturbation_kernels=self.perturbation_kernel_dict,
            prior_distributions=self.priors,
            weights=current_bundle.weights,
            starting_simulation_number=(self.current_step + 1)
            * self.n_simulations,
            seed_variable_name=self.seed_variable_name,
        )

        # Create new simulation bundle
        new_bundle = SimulationBundle(
            inputs=new_inputs,
            step_number=self.current_step + 1,
            baseline_params=current_bundle._baseline_params,
            seed_variable_name=self.seed_variable_name,
        )

        # Add simulation bundle to dictionary
        self.simulation_bundles[new_bundle._step_number] = new_bundle
        self.current_step += 1
