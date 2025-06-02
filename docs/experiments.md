# Experiment Class Documentation

The `Experiment` class is responsible for initializing, managing, and running experiments. It 
organizes `SimulationBundle`s, stores data in a structured format, and provides methods for lossy
compression, restoration, and deletion of experiments.

---

## Class: `Experiment`

### Description
The `Experiment` class manages data and primarily reads a config file providing information on 
experiment parameters. The goal of the package is to be able to call specific experiments through 
the config exclusively, either running scenarios, simulate random draws from particular parameter 
sets, or conduct ABC Sequential Monte Carlo experiments. 

---

### Constructor: `__init__`
The `Experiment` can be initialized through either an "experiments" directory and a config file or
by specifying a compressed experiment "image file". The former is most useful for running local 
experiments and initializing new simulation draws while the latter is most useful for operating on 
Azure and creating save points during the experiment run.

#### Parameters:
- `experiments_directory` (`str`, optional): Path to the experiments directory. Required for config-based initialization.
- `config_file` (`str`, optional): Path to the configuration file. Used to initialize a new experiment. Must be provided alongside an experiments directory.
- `img_file` (`str`, optional): Path to the compressed experiment file. Used to restore a previously saved experiment.
- `prior_distribution_dict` (`dict`, optional): Dictionary of prior distributions for simulation parameters.
- `perturbation_kernel_dict` (`dict`, optional): Dictionary of perturbation kernels for simulation parameters.

#### Raises:
- `ValueError`: If neither `config_file` nor `img_file` is provided.
- `FileNotFoundError`: If the specified `config_file`, either provided or accessed by `img_file`, does not exist.

---

### Method: `load_config_params`

#### Description:
Loads parameters from the experimental configuration file. Establishes experimental components, 
paths, and handles Azure Batch configuration. Called automatically on initialization.

#### Key Attributes Set:
- `super_experiment_name`: Name of the super experiment.
- `sub_experiment_name`: Name of the sub experiment.
- `directory`: Directory for the experiment.
- `data_path`: Path to the data directory.
- `exe_file`: Path to the executable file.
- `model_type`: Type of the model (`gcm` or `ixa`).
- `input_file_type`: Input file type (`YAML` or `json`).
- `default_params_file`: Path to the default parameters file.
- `target_data`: Target data loaded as a Polars DataFrame.
- `seed`: Seed for random number generation.
- `n_particles`: Number of particles per step.
- `replicates`: Number of replicates per particle.
- `n_simulations`: Total number of simulations.
- `tolerance_dict`: Dictionary of tolerances for ABC SMC steps.
- `scenario_key`: Key for accessing specific parameter sets.
- `changed_baseline_params`: Dictionary of changed baseline parameters.
- `azure_batch`: Boolean indicating Azure Batch execution.
- `client`: Azure client instance.
- `blob_container_name`: Name of the Azure Blob container.
- `job_prefix`: Prefix for Azure Batch jobs.

---

### Method: `compress_and_save`

#### Description:
Performs lossy compression to create a reproducible savepoint of the experiment. Stores all 
essential information except simulation bundle results in a compressed pickle file.

#### Parameters:
- `output_file` (`str`): Path to the output pickle file containing the compressed experiment.

---

### Method: `restore_experiment`

#### Description:
Restores an experiment from a compressed pickle file.

#### Parameters:
- `input_file` (`str`): Path to the compressed pickle file.

---

### Method: `delete_experiment`

#### Description:
Deletes files or folders within a given experiment directory, optionally filtering by prefix or 
file type.

#### Parameters:
- `prefix` (`str`, optional): Prefix that selected items must start with.
- `file_type` (`str`, optional): Specific file extension to filter for deletion.

---

### Method: `initialize_simbundle`

#### Description:
Initializes the first simulation bundle for the experiment. This is the first step of ABC SMC, a 
single scenario, or all the simualtions from a single parameters set provided.

#### Parameters:
- `prior_distribution_dict` (`dict`, optional): Dictionary of prior distributions for simulation parameters.
- `changed_baseline_params` (`dict`, optional): Dictionary of changed baseline parameters.
- `scenario_key` (`str`, optional): Key for accessing specific parameter sets.
- `unflatten` (`bool`, optional): Whether to unflatten the parameters.
- `seed_variable_name` (`str`, optional): Name of the seed variable.

#### Returns:
- `SimulationBundle`: Initialized simulation bundle.

---

### Method: `get_bundle_from_simulation_index`

#### Description:
Retrieves a simulation bundle based on its index.

#### Parameters:
- `simulation_index` (`int`): Index of the simulation.

#### Returns:
- `SimulationBundle`: Simulation bundle in the `simulaiton_bundles` history dictionary corresponding to the index.

---

### Dependencies

The `Experiment` class relies on the following libraries and modules:
- `os`, `pickle`, `yaml`, `warnings`, `subprocess`, `random`, `shutil`
- `polars` for data manipulation
- `abctools.abc_classes.SimulationBundle` for simulation bundles
- `abmwrappers.utils` for utility functions
- `cfa_azure.helpers` for Azure Batch support

---

### Example Usage
