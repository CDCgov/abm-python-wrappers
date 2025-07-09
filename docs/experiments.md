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
- `verbose` (`bool`, default=True): Whether or not to print out some non-critical warnigns and progress statements when operating with the experimnt
- `**kwargs` : Key word arguments to provide additional parameters orr overwrite config parameters. In particular,
    - `prior_distribution_dict` (`dict`, optional): Dictionary of prior distributions for simulation parameters.
    - `perturbation_kernel_dict` (`dict`, optional): Dictionary of perturbation kernels for simulation parameters.

#### Raises:
- `ValueError`: If neither `config_file` nor `img_file` is provided.
- `FileNotFoundError`: If the specified `config_file`, either provided or accessed by `img_file`, does not exist.

---

### Method: `load_config`

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
- `input_file_type`: Input file type (`yaml` or `json`).
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

### Method: `save`

#### Description:
Performs lossy compression to create a reproducible savepoint of the experiment. Stores all
essential information except simulation bundle results in a compressed pickle file.

#### Parameters:
- `output_file` (`str`): Path to the output pickle file containing the compressed experiment.

---

### Method: `restore`

#### Description:
Restores an experiment from a compressed pickle file.

#### Parameters:
- `input_file` (`str`): Path to the compressed pickle file.

---

### Method: `delete`

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

### Method: `bundle_from_index`

#### Description:
Retrieves a simulation bundle based on its index.

#### Parameters:
- `simulation_index` (`int`): Index of the simulation.

#### Returns:
- `SimulationBundle`: Simulation bundle in the `simulaiton_bundles` history dictionary corresponding to the index.

---

### Method: `run_index`

#### Description
The `run_index` method is responsible for executing a single simulation based on a given index and processing its outputs. It validates input parameters, generates input files, executes the simulation, processes the raw outputs, and optionally saves or cleans intermediate results.

#### Parameters
- **`simulation_index`** (`int`): Index of the simulation to execute.
- **`distance_fn`** (`Callable`, optional): Function to calculate distances between target data and results.
- **`data_processing_fn`** (`Callable`, required): Function to process raw simulation outputs.
- **`products`** (`list`, optional): List of products to generate (e.g., `["simulations", "distances"]`).
- **`products_output_dir`** (`str`, optional): Directory to store output products. Defaults to the experiment's data path.
- **`scenario_key`** (`str`, optional): Key to access specific scenario parameters.
- **`cmd`** (`str`, optional): Command to execute the simulation. Defaults to a generated command if not provided.
- **`save`** (`bool`, optional): Whether to save processed results. Defaults to `True`.
- **`clean`** (`bool`, optional): Whether to clean intermediate raw output files. Defaults to `False`.

#### Behavior
1. Validates input parameters, ensuring required functions are provided.
2. Generates input files for the simulation using `write_inputs`.
3. Executes the simulation using a command-line utility.
4. Processes raw outputs using the provided `data_processing_fn`.
5. Updates the `results` attribute of the corresponding `SimulationBundle`.
6. Saves processed products if `save` is `True`.
7. Cleans raw output files if `clean` is `True`.

---

### Method: `run_step`

#### Description
The `run_step` method orchestrates the execution of multiple simulations for a specific step in the experiment's history. It ensures all simulations in the step are executed and their results are processed. . It defaults to the current step if none is provided and initializes a new simulation bundle if necessary. For each simulation index within the step, it calls `run_index` to execute and process the simulation, ensuring that all results are generated and stored as specified.

#### Parameters
- **`data_processing_fn`** (`Callable`, required): Function to process raw simulation outputs.
- **`distance_fn`** (`Callable`, optional): Function to calculate distances between target data and results.
- **`products`** (`list`, optional): List of products to generate (e.g., `["simulations"]`).
- **`step`** (`int`, optional): Step in the experiment history to execute. Defaults to the current step.
- **`products_output_dir`** (`str`, optional): Directory to store output products. Defaults to the experiment's data path.
- **`scenario_key`** (`str`, optional): Key to access specific scenario parameters.

#### Behavior
1. Determines the step to execute, defaulting to the current step.
2. Initializes a simulation bundle if the step does not exist.
3. Iterates through simulation indices in the step and calls `run_index` for each.

---

### Method: `parquet_from_path`

#### Description
Reads Parquet files from a specified path. Supports reading from local storage or Azure blob containers.

#### Parameters
- **`path`** (`str`): Path to the Parquet file.

#### Returns
- **`pl.DataFrame`**: DataFrame containing the contents of the Parquet file.

---

### Method: `store_products`

#### Description
Stores processed simulation outputs (e.g., distances and simulations) as Parquet files in the specified output directory. This ensures that the data is stored in a structured and accessible format from parallelized write and run tasks.

### Parameters
- **`sim_indeces`** (`list`): List of simulation indices to store.
- **`distance_fn`** (`Callable`, optional): Function to calculate distances. Required if storing distances.
- **`products`** (`list`, optional): List of products to store (e.g., `["distances", "simulations"]`).
- **`products_output_dir`** (`str`, optional): Directory to store output products. Defaults to the experiment's data path.

#### Behavior
1. Validates input parameters, ensuring required functions are provided.
2. Writes distances and simulation results to Parquet files, organizing them by simulation index.

---

### Method: `read_distances`

#### Description
Reads distance data from storage and integrates it into the simulation bundle's history. It validates the presence of distance data and ensures that the current step does not already contain distances before updating the bundle.

### Parameters
- **`input_dir`** (`str`, optional): Directory to read distance data from. Defaults to the experiment's data path.

#### Behavior
1. Reads distance data from Parquet files.
2. Validates that the current step does not already contain distances.
3. Updates the `distances` attribute of the current step's `SimulationBundle`.

---

### Method: `read_results`

#### Description
Reads simulation results from storage, supporting both nested CSV files and hive-partitioned Parquet files. It allows for optional preprocessing of the data and supports writing the results back to storage in either format.

#### Parameters
- **`filename`** (`str`, optional): Name of the file to read. Defaults to `"simulations"`.
- **`input_dir`** (`str`, optional): Directory to read the file from. Defaults to the experiment's data path.
- **`preprocessing_fn`** (`Callable`, optional): Function to preprocess the data during reading.
- **`write`** (`bool`, optional): Whether to write the data back to storage. Defaults to `False`.
- **`partition_by`** (`list`, optional): Columns to partition the data by when writing as a Parquet file.

#### Returns
- **`pl.DataFrame`**: DataFrame containing the simulation results.

#### Behavior
1. Reads results from hive-partitioned Parquet files or nested CSV files.
2. Applies preprocessing if specified.
3. Optionally writes the data back to storage in Parquet or CSV format.

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
To initialize experiments, users can either provide the path to a `config_file`
along with a relevant `experiments_directory` or provide the path to an
`img_file` that contains a compressed `Experiment` pickle file.

In this repo, initializing an `Experiment` in the `"tests"` folder would be

```python
experiment = Experiment(
    experiments_directory="tests",
    config_file="tests/path/to/input/config.yaml,
)
```

Experiments can be initialized form a config file at the root of the experiments
directory, in this case `"tests"`, or in the path specified by their
`super_experiment_name` and `sub_experiment_name`

Any parameters from the config file can be overwritten on initialization through
the keyword arguments. For example, to specify that the base input file should
be modified to provide a new mean for some distribution, this can be accomplished
using

```python
    experiment = Experiment(
        experiments_directory="tests",
        config_file=config_file,
        changed_baseline_params={
            "my_distribution": {
                "norm": {
                    "mean": 1.0
                }
            }
        }
    )
```

Keyword arguments also accept prior distributions and perturbation kernels for
ABC-SMC, declared as dictionaries. For example, using the same parameter
hierarchy as above,

```python
    experiment = Experiment(
        experiments_directory="tests",
        config_file=config_file,
        prior_distribution_dict={
            "my_distribution": {
                "norm": {
                    "mean": norm(1.0, 3.0)
                }
            }
        },
        perturbation_kernel_dict={
            "my_distribution": {
                "norm": {
                    "mean": norm(0.0, 0.1)
                }
            }
        }
    )
```

In order to restore a compressed `Experiment` using the `img_file` input, no
other parameters need to be specified.

```python
experiment = Experiment(
    img_file="tests/output/experiment_history.pkl",
)
```

This can likewise be combined with keyword arguments that update the
`Experiment` attributes. Compressing an `Experiment` again will track these
changed attributes, as all information except for simulation results of the
simulation bundles history dictionary is stored across save-restore cycles.

A simple exmaple of manipulating Experiments using save-restore is the
`wrappers.update_abcsmc_img` function, which is simply

```python
# Load the compressed experiment
experiment = Experiment(img_file=experiment_file)

# Load the distances
experiment.read_distances(input_dir=products_path)
experiment.resample()

# Save the updated experiment
experiment.save(experiment_file)
```

The simplest automated task to start running simulations is the `run_step` method,
which can be called without any pre-processing or with optional arguments

The default behavior of `run_step` requires at least a data preprocessing function
that accepts a string directory path location of the raw outputs being generated and
returns a `pl.DataFrame`

For example,

```python
def my_reader_fn(output_dir: str):
    fp = os.path.join(output_dir, "person_property_count.csv")
    return pl.read_csv(fp)

experiment = Experiment(
        experiments_directory="tests",
        config_file=config_file
    )
experiment.run_step(data_processing_fn=my_reader_fn)
```

This will automatically store your output file as a hive-partitioned parquet for easier
file reading and processing, but setting `save = False` will not do so.

If you would like to post-process the data or combine other files produced by a model,
that can also be done quickly using `run_step`. Simply update the processing function.

```python
def my_complex_fn(output_dir: str):
    """
    Create a formatted data frame that returns the transmssion chain graph with extra information about current circulating infections
    """
    counts_fp = os.path.join(output_dir, "person_property_count.csv")
    transmission_chain_fp = os.path.join(output_dir, "transmission_report.csv")

    daily_infection_count = (
        pl.read_csv(counts_fp)
        .filter(pl.col("InfectionStatus") == "Infected")
        .group_by("t")
        .agg(pl.count())
    )

    # Join the daily
    transmission_graph = (
        pl.read_csv(transmission_chain_fp)
        .with_columns(pl.col("t_infection").floor().alias("t"))
    )

    data = transmission_grpah.join(daily_infection_count, on = "t", how = "left")

    return data

experiment = Experiment(
        experiments_directory="tests",
        config_file=config_file
    )
experiment.run_step(data_processing_fn=my_complex_fn)
```

Now the compressed `simulations` folder will contain the already processed data.
We can access that information for use in a python enviornment or extract it to a non-compressed CSV format
using the `read_results` Experiment method.

```python
data = experiment.read_results()
```

will allow for manipulatable `data` using polars expressions

```python
data = experiment.read_results(write=True)
```

will return the same `simulations` data frame but will also write a non-compressed CSV file of the data
in the root data directory

```python
data = experiment.read_results(write=True, partition_by=["simulation", "t"])
```

will return the same `simulations` data frame but will also write a compressed hive-partitioned Parquet
file of the data in the root data directory if the data should be partitioned in a new way.

In the case where raw output files are proudced but not needed by the data read in processing function,
such as an unused transmission chain report in the case of `my_reader_fn`, we can later save the raw files
in the same fashion as above by calling `read_results`

```python
def my_reader_fn(output_dir: str):
    fp = os.path.join(output_dir, "person_property_count.csv")
    return pl.read_csv(fp)

experiment = Experiment(
        experiments_directory="tests",
        config_file=config_file
    )
experiment.run_step(data_processing_fn=my_reader_fn)

chains = experiment.read_results(filename="transmission_report")
```

The transmission chains will now be accessed. The methods described above with the default of
`simulations` can also be used to store the transmission report permanently.

If we want to postpprcoess the transmission chain reports after they have been written but
before writing a new stored file, we can supply a new function that takes in a data frame
and returns a simplified one.

```python
def my_postprocessing_fn(chain_data: pl.DataFrame) -> pl.DataFrame:
    """
    Create a formatted data frame that returns the simplified transmssion chain graph
    """
    # Join the daily
    transmission_graph = (
        data
        .filter((pl.col("t") > 2.0) & (pl.col("t") < 10.0))
        .with_columns(pl.col("t_infection").floor().alias("t"))
    )
    return transmission_graph

chains = experiment.read_results(
    filename="transmission_report",
    data_processing_fn=my_postprocessing_fn,
    write = True,
    partition_by = ["t"]
)
```

the transmission chain report is now saved in the appropriate directory and can be accessed
later using only the parquet style read function

```python
processed_chains = experiment.read_results(filename="transmission_report")
```
