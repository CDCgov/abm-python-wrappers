# Wrappers Module Documentation

The `wrappers.py` file contains utility functions and methods for managing simulation data, processing experiment results, and updating compressed experiment files. It provides functionality for handling simulation bundles, writing data to Parquet files, and integrating with ABC SMC workflows.

---

## Functions

### `products_from_index`

#### Description
Processes simulation data and writes results to Parquet files. Handles distances, simulations, and intermediate cleaning.

#### Parameters:
- `simulation_index` (`int`): Index of the simulation to process.
- `experiment` (`Experiment`): The experiment object containing simulation data.
- `data_processing_fn` (`Callable`): Function for processing simulation data.
- `products` (`list`, optional): List of products to generate. Defaults to `["distances", "simulations"]`.
- `clean` (`bool`, optional): Whether to clean intermediate files. Defaults to `False`.

---

### `create_simulation_data`

#### Description
Creates simulation data by running simulations and processing results. Writes simulation data to Parquet files.

#### Parameters:
- `experiment` (`Experiment`): The experiment object containing simulation data.
- `data_processing_fn` (`Callable`): Function for processing simulation data.
- `products` (`list`, optional): List of products to generate. Defaults to `["simulations"]`.

#### Returns:
- `polars.DataFrame`: DataFrame containing simulation results.

---

### `update_abcsmc_img`

#### Description
Updates a compressed experiment file with distance data stored as a Parquet Hive partition. Used during the gather step of ABC SMC workflows.

#### Parameters:
- `experiment_file` (`str`): Path to the compressed experiment file.
- `products_path` (`str`): Path to the directory containing product data.

#### Returns:
- `None`

---

## Dependencies

The `wrappers.py` module relies on the following libraries and modules:
- `os`, `polars` (`pl`), `subprocess`
- [`Experiment`](abmwrappers/experiment_class.py) from [abmwrappers/experiment_class.py](abmwrappers/experiment_class.py)
- [`write_default_cmd`](abmwrappers/utils.py) from [abmwrappers/utils.py](abmwrappers/utils.py)

---

# Wrappers Module Documentation

The `wrappers.py` file contains utility functions and methods for managing simulation data, processing experiment results, and updating compressed experiment files. It provides functionality for handling simulation bundles, writing data to Parquet files, and integrating with ABC SMC workflows.

---

## Functions

### `products_from_index`

#### Description
Processes simulation data and writes results to Parquet files. Handles distances, simulations, and intermediate cleaning.

#### Parameters:
- `simulation_index` (`int`): Index of the simulation to process.
- `experiment` (`Experiment`): The experiment object containing simulation data.
- `data_processing_fn` (`Callable`): Function for processing simulation data.
- `products` (`list`, optional): List of products to generate. Defaults to `["distances", "simulations"]`.
- `clean` (`bool`, optional): Whether to clean intermediate files. Defaults to `False`.

---

### `create_simulation_data`

#### Description
Creates simulation data by running simulations and processing results. Writes simulation data to Parquet files.

#### Parameters:
- `experiment` (`Experiment`): The experiment object containing simulation data.
- `data_processing_fn` (`Callable`): Function for processing simulation data.
- `products` (`list`, optional): List of products to generate. Defaults to `["simulations"]`.

#### Returns:
- `polars.DataFrame`: DataFrame containing simulation results.

---

### `update_abcsmc_img`

#### Description
Updates a compressed experiment file with distance data stored as a Parquet Hive partition. Used during the gather step of ABC SMC workflows.

#### Parameters:
- `experiment_file` (`str`): Path to the compressed experiment file.
- `products_path` (`str`): Path to the directory containing product data.

#### Returns:
- `None`

---
### `run_abcsmc`

#### Description
The `run_abcsmc` function orchestrates the execution of Approximate Bayesian Computation Sequential Monte Carlo (ABC SMC) experiments. It supports both local and Azure Batch-based execution workflows, depending on the configuration of the `Experiment` object passed to it. The function handles initialization of simulation bundles, running simulations, processing results, and managing experiment history files.

---

#### Parameters
- `experiment` (`Experiment`): The `Experiment` object containing the configuration and data for the ABC SMC workflow.
- `data_processing_fn` (`Callable`, optional): A function for processing simulation data. Defaults to `None`.
- `distance_fn` (`Callable`, optional): A function for calculating distances between simulated and target data. Defaults to `None`.
- `prior_distribution_dict` (`dict`, optional): A dictionary defining the prior distributions for simulation parameters. Defaults to `None`.
- `perturbation_kernels` (`dict`, optional): A dictionary defining the perturbation kernels for simulation parameters. Defaults to `None`.
- `changed_baseline_params` (`dict`, optional): A dictionary of baseline parameters that have been changed. Defaults to an empty dictionary `{}`.
- `files_to_upload` (`list`, optional): A list of files to upload to Azure Blob Storage for Azure Batch execution. Defaults to an empty list `[]`.
- `scenario_key` (`str`, optional): The key for accessing specific parameter sets in the experiment. Defaults to `None`.
- `local_compress` (`bool`, optional): A flag indicating whether to compress the experiment locally. Defaults to `False`.

---

#### Behavior
- If `scenario_key` is not provided, it defaults to the `scenario_key` attribute of the `Experiment` object.
- If the current step of the experiment is `0` or `None`, the function initializes the simulation bundle using the provided prior distributions, perturbation kernels, and baseline parameters.

Running the function with an experiment that has `azure_batch` stored as `true`,
- If `experiment.azure_batch` is `True`, the function attempts to execute the experiment using Azure Batch.
- It verifies that the Azure client is initialized. If not, it raises a `ValueError`.
- The function prepares the experiment history file (`experiment_history.pkl`) and uploads it to Azure Blob Storage.
- If a compressed history file already exists, the user is prompted to decide whether to overwrite it.
- The function sets up Azure Batch jobs and tasks for simulation and gather steps. **Note:** This part is not fully implemented yet.

Running the function with an experiment that has `azure_batch` stored as `false`
- If `local_compress` is `True`, the function processes simulation bundles for the current step without compressing the experiment.
- It iterates through the simulation indices in the current bundle and processes simulation data using the `products_from_index` function.
- After processing all simulations for the step, the function resamples the experiment for the next ABC SMC step using the provided perturbation kernels.

---

### `create_scenario_subexperiments`

#### Description
This function accepts a path to a `pygriddler@v0.3` scenario JSON file and writes the inputs as independent subexperiments in a `scenarios` folder. The resulting subexperiments therefore have an experiments directory of the `experiment.sub_experiment_name`, a super experiment name `scenarios` and a subexperiment name `scenario=i`, where `i` is an iteratio over the length of the scenario count.

#### Parameters
- `experiment` (`Experiemnt`): the overall experiment to create the scenario sub-experiments for.
- `griddle_path` (`str`, optional): Path to the pygriddler JSON file.
- `scenario_key` (``str`, optional): Scneairo key used to label the simulation inputs
- `seed_key` (`str`, optional): name of the variable used to store random seeds for each simulation.

#### Behavior
The function first asks the user, if a data directory for the overall experiment exists, if they would like to proceed. Then the simulation inputs are determined by the pygriddler input file. New sub-experiment config files for each scenario are then created, alongside the base input file for each simulation, into new input folders for each scenario.


## Dependencies

The `wrappers.py` module relies on the following libraries and modules:
- `os`, `polars` (`pl`), `subprocess`
- [`Experiment`](abmwrappers/experiment_class.py) from [abmwrappers/experiment_class.py](abmwrappers/experiment_class.py)
- [`write_default_cmd`](abmwrappers/utils.py) from [abmwrappers/utils.py](abmwrappers/utils.py)

---

## Example Usage

### Simulate random replicates of an input file
In many cases, we may just want to call one parameter set many times to produce a set of simulations. In order to do so, we need to provide an `Experiment` with an initialized simulation bundle history, a base input parameter file to draw from, and a `config.yaml` relating the paths as well as number of replicates.

In order to initialize simulation bundle histroy for a new experiment, we can call the `experiment.intialize_simbundle()` method, which will automatically populate a `SimulationBundle` as the zeroeth simulation set of the `Experiment`.

This is performed automatically by the wrapper function `create_simulation_data`, which first checks for any simulation bundle history and either calls the current step's inputs to create simulation data or creates the first bundle and then creates the data.

```python
def read_fn(outputs_dir):
    # Function to read in the data
    # This is a placeholder function, implement your own logic here to provide post-processing if desired
    output_file_path = os.path.join(outputs_dir, "person_property_count.csv")
    if os.path.exists(output_file_path):
        df = pl.read_csv(output_file_path)
    else:
        raise FileNotFoundError(f"{output_file_path} does not exist.")

experiment = Experiment(
    experiments_directory = "experiments",
    config_file = "path/to/config.yaml"
)

simulation_data = wrappers.create_simulation_data(
    experiment,
    data_processing_fn=read_fn,
)
```

Simulations will now be stored as a hive partitioned `.parquet` file and nested as raw output `.csv` files in the outputs directory, which is automatically selected from the experiment's `.data_path`. The funciton returns the simulations as a `pl.DataFrame` for convenience if we want to do further analysis within the same script.

### Scenarios


### ABC SMC - local

### ABC SMC - Azure

### Scenarios and ABC SMC runs
