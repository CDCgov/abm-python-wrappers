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

from abmwrappers import utils, wrappers


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

        self.verbose = verbose

        # If an image .pkl file containing experiment history is supplied, read from that
        if img_file is not None:
            if self.verbose:
                print(f"Restoring previous experiment from {img_file}")
            self.restore(img_file)
        # Otherwise, instantiate a new experiment with load_config
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

            self.load_config()

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

    def load_config(self):
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
        else:
            self.model_type = "ixa"
            self.input_file_type = "json"
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
            self.target_data = None

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

        # In simulation parameter changes can be specified through config or appended later in helper functions
        # Scenario key must be declared to access the parameter set of interest in the input file
        self.scenario_key = experimental_config["experiment_conditions"][
            "scenario_key"
        ]

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
            "azure_batch": self.azure_batch,
            "model_type": self.model_type,
            "input_file_type": self.input_file_type,
            "changed_baseline_params": self.changed_baseline_params,
            "target_data": self.target_data,
            "seed": self.seed,
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
                bundle["inputs"],
                bundle["_step_number"],
                bundle["_baseline_params"],
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
        self.azure_batch = data["azure_batch"]
        self.model_type = data["model_type"]
        self.input_file_type = data["input_file_type"]
        self.changed_baseline_params = data["changed_baseline_params"]
        self.target_data = data["target_data"]
        self.seed = data["seed"]
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
    # Simulation bundles and ABC SMC funcitons
    # --------------------------------------------

    def initialize_simbundle(
        self,
        prior_distribution_dict: dict = None,
        changed_baseline_params: dict = {},
        scenario_key: str = None,
        unflatten: bool = True,
        seed_variable_name: str = "randomSeed",
    ) -> SimulationBundle:
        if len(changed_baseline_params) > 0:
            if self.verbose:
                print(
                    "Changed baseline parameters specified by function. Updating baseline parameters and overwriting common keys."
                )
                self.changed_baseline_params.update(changed_baseline_params)

        if scenario_key is None:
            scenario_key = self.scenario_key
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

    def run_step(
        self,
        data_processing_fn: Callable,
        products: list = None,
        step: int = None,
    ):
        """
        Function to run the simulations for a particular step in the experiment history, which defaults to the current step
        All results are generated and then processed into a single joint data frame

        Args:
            :param data_processing_fn: callable function to process the raw output data being stored as a data frame
            :param products: list of which products to store as processed data frames
            :param step: int of the step to run simulations from. Defaults to the current or alternatively initializes the history
        """
        if products is None:
            products = ["simulations"]

        if step is None:
            step = self.current_step

        # Generate products from current step if it exists or initialize
        if step is not None:
            simbundle = self.simulation_bundles[step]
        else:
            simbundle = self.initialize_simbundle()

        # Run the simulation
        for index in simbundle.inputs["simulation"]:
            wrappers.products_from_index(
                index,
                experiment=self,
                data_processing_fn=data_processing_fn,
                products=products,
            )

    # --------------------------------------------
    # File management
    # --------------------------------------------

    def parquet_from_path(self, path: str) -> pl.DataFrame:
        if self.azure_batch:
            df = utils.read_parquet_blob(
                self.blob_container_name, path, self.storage_config, self.cred
            )
        else:
            df = pl.scan_parquet(path).collect()
        return df

    def store_distances(self, input_dir: str = None):
        """
        Read distances and simulation results into the simulation bundle history
        Currently yields OSError when called on a mounted blob container input directory
        """
        if not input_dir:
            input_dir = self.data_path

        distances = self.parquet_from_path(f"{input_dir}/distances/")

        if distances.is_empty():
            raise ValueError("No distances found in the input directory.")

        if hasattr(self.simulation_bundles[self.current_step], "distances"):
            raise ValueError(
                "Simulation bundle already has distances. Please clear the simulation bundle before reading new distances."
            )
        self.simulation_bundles[self.current_step].distances = {}

        for k, v in zip(distances["simulation"], distances["distance"]):
            if self.step_from_index(k) == self.current_step:
                self.simulation_bundles[self.current_step].distances[k] = v

    def read_results(
        self,
        filename: str = None,
        input_dir: str = None,
        preprocessing_fn: str = None,
        write: bool = False,
        partition_by: list = None,
    ) -> pl.DataFrame:
        """
        Function to read results from simulation output.
        The default behavior is to search for an already-extant simulations parquet file in the experiment data path
        Alternatively, users can specify a generic file name to read from an arbitrary input directory

        Args:
            :param filename: The name of the CSV file or hive-partitioned parquet to read from. If not specified, defaults to "simulations"
            :param input_dir: The directory to read the file from. If not specified, defaults to the experiment data path
            :param preprocessing_fn: A function to preprocess the data before combining it withh other data during reading from CSV. If specified with a partitioned parquet file, the function If not specified, defaults to None
            :param store: Whether to store the data in the input directory as a parquet or CSV file. If True, the data is stored as a partitioned parquet file with the name of the file without extension or is stored as a CSV at the input directory root.
            :param partition_by: A list of columns to partition the data by when storing it as a parquet file. If not specified, defaults to None
        """
        # Default to dat path and the simulations parquet file
        if not input_dir:
            input_dir = self.data_path
        if not filename:
            filename = "simulations"

        # Read from hive-paritioned parquet if it exists and filename is not a file
        if len(filename.split(".")) == 1 and os.path.exists(
            f"{input_dir}/{filename}/"
        ):
            # Special case for names in "products"
            data = self.parquet_from_path(f"{input_dir}/{filename}/")
            if preprocessing_fn is not None:
                warnings.warn(
                    "Preprocessing function specified for a hive-partitioned parquet file. Please ensure that the function is compatible with the data format.",
                    UserWarning,
                )
                data = preprocessing_fn(data)
        # Otherwise attempt reading nested CSVs
        else:
            # Pre-proceesing fn is applied piece-wise to CSV
            data = utils.read_nested_csvs(
                input_dir, filename, preprocessing_fn
            )

        if write is not None:
            out_file = filename.split(".")[0]
            if partition_by is not None:
                data.write_parquet(
                    f"/{input_dir}/{out_file}/", partition_by=partition_by
                )
            else:
                data.write_csv(f"{input_dir}/{out_file}.csv")
        return data

    # arbitrary file from raw_output and process it for parquet - helper

    # populate bundle history with stored simulations

    # store inputs from created jsons

    # --------------------------------------------
    # Resampling between experiment steps
    # --------------------------------------------

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
            dtmp = pl.DataFrame(
                {"distances": [v for v in current_bundle.distances.values()]}
            )
            print(
                f"Distances in step {self.current_step} have: Min={dtmp.quantile(0.0).item()}, Q1={dtmp.quantile(0.25).item()}, Q2={dtmp.quantile(0.5).item()}, Q3={dtmp.quantile(0.75).item()}"
            )

        if self.current_step > 0:
            prev_bundle: SimulationBundle = self.simulation_bundles[
                self.current_step - 1
            ]
            if self.verbose:
                print("Calculating weights during a resample")
                print(
                    f"Using {len([i for i, val in enumerate(current_bundle.acceptance_weights.values()) if val > 0])} nonzero simulation weights"
                )

            current_bundle.weights = abc_methods.calculate_weights_abcsmc(
                current_accepted=current_bundle.accepted,
                prev_step_accepted=prev_bundle.accepted,
                prev_weights=prev_bundle.weights,
                stochastic_acceptance_weights=current_bundle.acceptance_weights,
                prior_distributions=self.priors,
                perturbation_kernels=self.perturbation_kernel_dict,
                normalize=True,
            )
            if self.verbose:
                print(
                    f"Calculated {len([i for i, val in enumerate(current_bundle.weights.values()) if val != 0])} nonzero particle weights for step {self.current_step}"
                )

        else:
            current_bundle.weights = {
                key: current_bundle.acceptance_weights[key]
                / sum(current_bundle.acceptance_weights.values())
                for key in current_bundle.accepted.keys()
            }
        new_inputs = abc_methods.resample(
            accepted_simulations=current_bundle.accepted,
            n_samples=self.n_particles,
            replicates_per_sample=self.replicates,
            perturbation_kernels=self.perturbation_kernel_dict,
            prior_distributions=self.priors,
            weights=current_bundle.weights,
            starting_simulation_number=(self.current_step + 1)
            * self.n_simulations,
        )

        # Create new simulation bundle
        new_bundle = SimulationBundle(
            inputs=new_inputs,
            step_number=self.current_step + 1,
            baseline_params=current_bundle._baseline_params,
        )

        # Add simulation bundle to dictionary
        self.simulation_bundles[new_bundle._step_number] = new_bundle
        self.current_step += 1

    def write_inputs(
        self,
        simulation_index: int,
        scenario_key: str = None,
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

        # Temporary workaround for mandatory naming of randomSeed variable in draw_parameters abctools method
        input_dict[simulation_index]["seed"] = input_dict[
            simulation_index
        ].pop("randomSeed")

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
        input_griddle: str = None,
        scenario_key: str = None,
        unflatten: bool = True,
        seed_key: str = "seed",
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

        # Load the parameter set
        with open(input_griddle, "r") as f:
            raw_griddle = json.load(f)

        griddle = griddler.parse(raw_griddle)
        par_sets = griddle.to_dicts()

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

        for par in par_sets:
            # Remove the griddler scenario keys from the parameter set
            to_remove = []
            for key in par.keys():
                if key.startswith("scenario"):
                    to_remove.append(key)
            for key in to_remove:
                par.pop(key)

            newpars, _summary = utils.combine_params_dicts(
                baseline_dict=baseline_params,
                new_dict=par,
                scenario_key=scenario_key,
                overwrite_unnested=True,
                unflatten=unflatten,
            )
            # Add the scenario key to the parameter set with replicates if specified
            for i in range(self.replicates):
                if self.replicates > 1:
                    changed_seed = {seed_key: random.randint(0, 2**32)}
                    newpars, _summary = utils.combine_params_dicts(
                        baseline_dict=newpars,
                        new_dict=changed_seed,
                        scenario_key=scenario_key,
                        overwrite_unnested=True,
                        unflatten=unflatten,
                    )

                input_file_name = (
                    f"simulation_{simulation_index}.{self.input_file_type}"
                )
                with open(os.path.join(input_dir, input_file_name), "w") as f:
                    json.dump(newpars, f, indent=4)
                simulation_index += 1
