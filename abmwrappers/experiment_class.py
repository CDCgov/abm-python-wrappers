import json
import os
import pickle
import random
import shutil
import subprocess
import warnings

import griddler
import polars as pl
import yaml
from abctools import abc_methods
from abctools.abc_classes import SimulationBundle

from abmwrappers import utils


class Experiment:
    """
    A class to intiialize an experiment given some input files
    Organizes simulation bundles and stores data to file structure
    """

    def __init__(
        self,
        experiments_directory: str = None,
        config_file: str = None,
        img_file: str = None,
        prior_distribution_dict: dict = None,
        perturbation_kernel_dict: dict = None,
    ):
        """
        Initialize the experiment as either a new experiment using a config file or restore the history of a previously compressed and saved experiment
        :param config_file: The path to the config file. If this file is not specified, the experiment will be initialized with the img_file
        :param experiments_directory: The path to the experiments directory. This is where the experiment will be saved and recognized. Required for config but not img_file
        :param img_file: The path to the image file. If this file is not specified, the experiment will be initialized with the config_file
        :param prior_distribution_dict: A dictionary of prior distributions for the simulation parameters. This is used during ABC SMC and for drawing random parameters
            if no parameters are specified in the prior distributions

        """

        if config_file is None and img_file is None:
            raise ValueError(
                "No config file or image file specified. Please provide a config file."
            )

        # If an image .pkl file containing experiment history is supplied, read from that
        if img_file is not None:
            self.restore_experiment(img_file)
        # Otherwise, instantiate a new experiment with load_config_params
        else:
            self.config_file = config_file
            self.experiments_path = experiments_directory
            self.directory = os.path.dirname(config_file)

            if self.directory.endswith("input") or self.directory.endswith(
                "input/"
            ):
                self.directory = os.path.dirname(self.directory)

            if self.directory.startswith("./"):
                self.directory = os.path.abspath(self.directory)

            self.simulation_bundles = {}
            self.current_step = None

            # Optional inputs for use in getters or wrappers later
            # These can likewise be decalred to each helper function independently
            self.priors = prior_distribution_dict
            self.perturbation_kernel_dict = perturbation_kernel_dict

            if os.path.exists(config_file):
                self.load_config_params()
            else:
                # Check if the config file exists in the experiments directory
                tmp = os.path.join(experiments_directory, config_file)
                if os.path.exists(tmp):
                    self.config_file = tmp
                    self.directory = os.path.dirname(tmp)

                    # Warn that config file is modified
                    warnings.warn(
                        f"Config file {config_file} is modified. Using {tmp} instead.",
                        UserWarning,
                    )
                else:
                    raise FileNotFoundError(
                        f"Config file {config_file} does not exist."
                    )

            self.load_config_params()

    # --------------------------------------------
    # Loading and storing the experiment
    # --------------------------------------------

    def load_config_params(self):
        """
        load the parameters from the experimental file
        establishes the experimental components, paths, and handles tha azure configuration file
        """
        with open(self.config_file, "r") as f:
            experimental_config = yaml.load(f, Loader=yaml.SafeLoader)

        # --------------------------------------------
        # Experiment name
        self.super_experiment_name = experimental_config["local_path"][
            "super_experiment_name"
        ]
        self.sub_experiment_name = experimental_config["local_path"][
            "sub_experiment_name"
        ]

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
                    f"Config file directory {self.directory} does not match the experiment path {specified_experiment_path} specified in file and isn't at expeirments folder root {self.experiments_path}."
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
            print(
                "No super experiment specified, operating in root experiments directory"
            )

        # Check if scenario griddle file is specified
        if "griddle_file" in experimental_config["local_path"]:
            self.griddle_file = experimental_config["local_path"][
                "griddle_file"
            ]

        # --------------------------------------------
        # Experimental components
        self.data_path = os.path.join(self.directory, "data")
        os.makedirs(self.data_path, exist_ok=True)

        self.exe_file = experimental_config["local_path"]["exe_file"]
        if self.exe_file.endswith(".jar"):
            self.model_type = "gcm"
            self.input_file_type = "YAML"
        else:
            self.model_type = "ixa"
            self.input_file_type = "json"
        self.default_params_file = experimental_config["local_path"][
            "default_params_file"
        ]

        target_data_file = experimental_config["local_path"][
            "target_data_file"
        ]
        if target_data_file is not None:
            self.target_data = pl.read_csv(target_data_file)
        else:
            self.target_data = None

        self.seed = experimental_config["experiment_conditions"]["seed"]
        self.n_particles = experimental_config["experiment_conditions"][
            "samples_per_step"
        ]
        self.replicates = experimental_config["experiment_conditions"][
            "replicates_per_particle"
        ]
        if self.replicates is None:
            self.replicates = 1

        self.n_simulations = self.n_particles * self.replicates

        # Tolerance can be specified for each step or stored as a null in the case of not running abcsmc using the wrappers
        if (
            experimental_config["experiment_conditions"]["tolerance"]
            is not None
        ):
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

        # In simulation parameter changes can be specified through config or appended later in helper functions
        # Scenario key must be declared to access the parameter set of interest in the input file
        self.scenario_key = experimental_config["experiment_conditions"][
            "scenario_key"
        ]
        if (
            "changed_baseline_params"
            in experimental_config["experiment_conditions"]
        ):
            self.changed_baseline_params = experimental_config[
                "experiment_conditions"
            ]["changed_baseline_params"]
        else:
            self.changed_baseline_params = {}

        # --------------------------------------------
        # Azure batch parameter loading
        self.azure_batch = experimental_config["azb"]["azure_batch"]

        if self.azure_batch:
            self.azb_config_path = os.path.join(
                self.directory, experimental_config["azb"]["azb_config_path"]
            )
            with open(self.azb_config_path, "w") as f:
                yaml.dump(experimental_config["azb"]["azb_config"], f)

            self.create_pool = experimental_config["azb"]["create_pool"]

        if self.tolerance_dict is None or self.target_data is None:
            warnings.warn(
                """Target data and/or tolerance dict are currently specified as None.
                    No ABC routines can be run without at least one tolerance step and target data
                    declared through config file.""",
                UserWarning,
            )

    def compress_and_save(self, output_file: str):
        """
        Lossy compression to save a reproducible savepoint
        Stores all information except for simulation bundle results to compressed pickle file
        Important to track across steps and reproduce the experiment are
        Directory data:
            - config file
            - directory
            - data
            - exe_file
        Simulation bundle data:
            - current step
            - distances
            - weights
            - accepted
            - inputs
        Experiment data:
            - tolerance steps
            - priors
            - perturbation kernels
            - replicate counts (particle count and per particle)
        """

        # Create a dictionary to store the data
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
            "current_step": self.current_step,
            "tolerance_dict": self.tolerance_dict,
            "priors": self.priors,
            "perturbation_kernel_dict": self.perturbation_kernel_dict,
        }

        # Save the data to a compressed pickle file
        with open(output_file, "wb") as f:
            pickle.dump(data, f)

    def restore_experiment(self, input_file: str):
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
        self.current_step = data["current_step"]
        self.tolerance_dict = data["tolerance_dict"]
        self.priors = data["priors"]
        self.perturbation_kernel_dict = (
            data["perturbation_kernel_dict"]
            if "perturbation_kernel_dict" in data
            else None
        )

    def delete_experiment(
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
                if os.path.isdir(item_path) or not item.endswith(file_type):
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
            print(
                "Changed baseline parameters specified from config file. Updating baseline parameters and overwriting common keys."
            )
            self.changed_baseline_params.update(changed_baseline_params)

        if scenario_key is None:
            scenario_key = self.scenario_key
        # Create baseline_params by updating default params
        baseline_params, _summary_string = utils.load_baseline_params(
            self.default_params_file,
            self.changed_baseline_params,
            scenario_key,
            unflatten,
        )

        # Save prior distribution for use in experiment and draw
        if prior_distribution_dict is not None:
            if (
                self.priors is not None
                and self.priors is not prior_distribution_dict
            ):
                raise ValueError(
                    "Different prior distribution already specified on experiment initiation. Please declare only one set of priors for inital simulation bundle."
                )
            self.priors = prior_distribution_dict

        if self.priors:
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

    def get_bundle_from_simulation_index(
        self, simulation_index: int
    ) -> SimulationBundle:
        step_id = int(simulation_index / self.n_simulations)
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

    def read_parquet_distances_to_current_step(
        self, input_dir: str, write_results: bool = False
    ):
        """
        Read distances and simulation results into the simulation bundle history
        Currently yields OSError when called on a mounted blob container input directory
        """
        # Scan the hive partition parquet file and collect into a dataframe
        # if self.azure_batch:
        #     distances = utils.spark_parquet_to_polars(
        #         f"{input_dir}/distances/", "simulation"
        #     )
        # else:
        distances = pl.scan_parquet(f"{input_dir}/distances/").collect()

        if distances.is_empty():
            raise ValueError("No distances found in the input directory.")

        if hasattr(self.simulation_bundles[self.current_step], "distances"):
            raise ValueError(
                "Simulation bundle already has distances. Please clear the simulation bundle before reading new distances."
            )
        self.simulation_bundles[self.current_step].distances = {}

        for k, v in zip(distances["simulation"], distances["distance"]):
            self.simulation_bundles[self.current_step].distances[k] = v

    # --------------------------------------------
    # Resampling between experiment steps
    def resample_for_next_abc_step(
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
        if perturbation_kernel_dict is not None:
            self.perturbation_kernel_dict = perturbation_kernel_dict
        if self.perturbation_kernel_dict is None:
            raise ValueError(
                "Perturbation kernel not specified on experiment initiation or during resample."
            )
        if self.priors is None and prior_distribution_dict is None:
            raise ValueError(
                "Prior distribution not specified on experiment initiation or during resample."
            )
        if prior_distribution_dict is not None and self.priors is None:
            print(
                "Prior distribution specified during resample. Updating prior."
            )
            self.priors = prior_distribution_dict

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

        if self.current_step > 0:
            prev_bundle = self.simulation_bundles[self.current_step - 1]

            current_bundle.weights = abc_methods.calculate_weights_abcsmc(
                current_accepted=current_bundle.accepted,
                prev_step_accepted=prev_bundle.accepted,
                prev_weights=prev_bundle.weights,
                stochastic_acceptance_weights=current_bundle.acceptance_weights,
                prior_distributions=self.priors,
                perturbation_kernels=self.perturbation_kernel_dict,
                normalize=True,
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

    def write_simulation_inputs_to_file(
        self,
        simulation_index: int,
        write_inputs_cmd: str = None,
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

        sim_bundle = self.get_bundle_from_simulation_index(simulation_index)

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

        if write_inputs_cmd is None:
            if simulation_index % self.n_simulations == 0:
                print(
                    "No preprocessing step supplied to transform input parameters. Printing directly to file"
                )
            formatted_inputs = utils.gcm_parameters_writer(
                params=simulation_params,
                output_type=self.input_file_type,
                unflatten=True,
            )
            with open(os.path.join(input_dir, input_file_name), "w") as f:
                f.write(formatted_inputs)
            return os.path.join(input_dir, input_file_name)
        else:
            # Use the command line to write the inputs from a preprocessing script
            # UNSTABLE
            warnings.warn(
                "Preprocessing step supplied to transform input parameters. UNSTABLE. Writing to base.yaml",
                UserWarning,
            )
            formatted_inputs = utils.gcm_parameters_writer(
                params=simulation_params, output_type="YAML", unflatten=False
            )
            with open(os.path.join(input_dir, "base.yaml"), "w") as f:
                f.write(formatted_inputs)
            subprocess.run(write_inputs_cmd.split(), check=True)
            return input_dir

    def write_simulation_inputs_from_griddle(
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

        griddle = griddler.Griddle(raw_griddle)
        par_sets = griddle.parse()

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
                scenario_key="cfa_ixa_ebola_response_2025.Parameters",
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
                        scenario_key="cfa_ixa_ebola_response_2025.Parameters",
                        overwrite_unnested=True,
                        unflatten=unflatten,
                    )

                input_file_name = (
                    f"simulation_{simulation_index}.{self.input_file_type}"
                )
                with open(os.path.join(input_dir, input_file_name), "w") as f:
                    json.dump(newpars, f, indent=4)
                simulation_index += 1
