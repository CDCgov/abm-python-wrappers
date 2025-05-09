import os
import subprocess
import warnings

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
        experiments_directory: str,
        config_file: str,
        prior_distribution_dict: dict = None,
        perturbation_kernel_dict: dict = None,
    ):
        """
        Initialize the experiment with a config file
        :param config_file: Path to the config file
        """
        self.config_file = config_file
        self.experiments_path = experiments_directory
        self.directory = os.path.dirname(config_file)
        self.simulation_bundles = {}
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

    def load_config_params(self):
        """
        load the parameters from the experimental file
        establsihes the experimental components, paths, and handles tha azure configuration file
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
                f"Config file directory {self.directory} does not match the experiment path specified in file and isn't at expeirments folder root."
            )

        if self.directory == self.experiments_path:
            os.makedirs(specified_experiment_path, exist_ok=True)
            print(
                f"Writing experiment config file to {specified_experiment_path}"
            )
            config_file = os.path.join(
                specified_experiment_path, "config_abc.yaml"
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
        self.target_data = pl.read_csv(target_data_file)

        self.seed = experimental_config["experiment_conditions"]["seed"]
        self.n_particles = experimental_config["experiment_conditions"][
            "samples_per_step"
        ]
        self.replicates = experimental_config["experiment_conditions"][
            "replicates_per_particle"
        ]
        self.n_simulations = self.n_particles * self.replicates

        self.tolerance_dict = {
            int(k): float(v)
            for k, v in experimental_config["experiment_conditions"][
                "tolerance"
            ].items()
        }

        for tolerance in self.tolerance_dict.values():
            if tolerance <= 0:
                raise ValueError("Acceptance criteria improperly defined")
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

    def initialize_simbundle(
        self,
        prior_distribution_dict: dict = None,
        changed_baseline_params: dict = {},
        scenario_key: str = "baseline_parameters",
        unflatten: bool = True,
    ) -> SimulationBundle:
        # Create baseline_params by updating default params
        baseline_params, _summary_string = utils.load_baseline_params(
            self.default_params_file,
            changed_baseline_params,
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
            raise ValueError(
                "Prior distribution not specified on expeirment initiation or on simbundle declaration."
            )

        # Create simulation bundle
        sim_bundle = SimulationBundle(
            inputs=input_df,
            step_number=0,
            baseline_params=baseline_params,
        )

        # Add simulation bundle to dictionary
        self.simulation_bundles[sim_bundle.step_number] = sim_bundle
        self.current_step = 0

        return sim_bundle

    # --------------------------------------------
    # Reading in data for the current experiment step
    def read_parquet_data_to_current_step(
        self, input_dir: str, write_results: bool = False
    ):
        """
        Read distances and simulation results into the simulation bundle history
        """
        distances = pl.read_parquet(f"{input_dir}/distances/")

        if distances.is_empty():
            raise ValueError("No distances found in the input directory.")

        if self.simulation_bundles[self.current_step].distances:
            raise ValueError(
                "Simulation bundle already has distances. Please clear the simulation bundle before reading new distances."
            )
        self.simulation_bundles[self.current_step].distances = {}

        for k, v in zip(distances["simulation"], distances["distance"]):
            self.simulation_bundles[self.current_step].distances[k] = v

        # These should be optionally stored, not just optionally written
        if write_results:
            simulations = pl.read_parquet(f"{input_dir}/simulations/")
            with open(
                os.path.join(
                    input_dir, f"step_{self.current_step}_simulations.parquet"
                ),
                "wb",
            ) as f:
                simulations.write_parquet(f)

    # --------------------------------------------
    # Resampling between experiment steps
    def resample_for_next_abc_step(
        self,
        perturbation_kernel_dict: dict = None,
    ):
        """
        Resample the simulation bundle for the current step
        """
        if perturbation_kernel_dict is not None:
            self.perturbation_kernel_dict = perturbation_kernel_dict
        if self.perturbation_kernel_dict is None:
            raise ValueError(
                "Perturbation kernel not specified on experiment initiation or during resample."
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
            baseline_params=current_bundle.baseline_params,
        )

        # Add simulation bundle to dictionary
        self.simulation_bundles[new_bundle.step_number] = new_bundle
        self.current_step += 1

    def write_simulation_inputs_to_file(
        self,
        simulation_index: int,
        write_inputs_cmd: str = None,
        scenario_key: str = "baseline_parameters",
    ):
        """
        Write a single simulation's input file
        :param simulation_index: The index of the simulation to write
        :param write_inputs_cmd: The command to run that sources outside code to transform
            a base yaml file into readable inputs for execution
            A default of None will return only the formatted inputs from the simulation bundle
        :param scenario_key: The key to use for the scenario in the simulation bundle input dictionary

        In general, self.directory/data/input for simulation input files
        Use a tmp space for files that are intermediate products
        """
        if self.current_step not in self.simulation_bundles:
            raise ValueError(
                "No simulation bundle found for the current step."
            )

        sim_bundle = self.simulation_bundles[self.current_step]
        input_dict = utils.df_to_simulation_dict(
            sim_bundle.inputs
        )
       
        # Temporary workaround for mandatory naming of randomSeed variable in draw_parameters abctools method
        input_dict[simulation_index]["seed"] = input_dict[simulation_index].pop(
            "randomSeed"
        )


        simulation_params, _summary_string = utils.combine_params_dicts(
            sim_bundle.baseline_params,
            input_dict[simulation_index],
            scenario_key=scenario_key,
        )

        input_dir = os.path.join(self.data_path, "input")
        os.makedirs(input_dir, exist_ok=True)

        input_file_name = (
            f"simulation_{simulation_index}.{self.input_file_type}"
        )

        if write_inputs_cmd is None:
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
        else:
            # Use the command line to write the inputs
            formatted_inputs = utils.gcm_parameters_writer(
                params=simulation_params, output_type="YAML", unflatten=False
            )
            with open(os.path.join(input_dir, "base.yaml"), "w") as f:
                f.write(formatted_inputs)
            subprocess.run(write_inputs_cmd.split(), check=True)

