from math import floor, sqrt

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from abmwrappers.experiment_class import Experiment

def plot_posterior_distribution(
    experiment: Experiment,
    parameters: list[str] | str | None = None,
    visualization_methods: list[str] | str = ["violin", "box"],
    include_priors: bool = False,
    dependent_variable: str | None = None,
    include_previous_steps: list[int] | int | bool = False,
    limit_to_accepted: bool = False,
    facet_by: list[str] | str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_file: str | None = None,
) -> plt.Axes:
    """
    Function to plot the distribution of particular parameters over the experiment.
    Args:
        experiment: The Experiment object containing simulation data.
        parameters: List of parameter names to plot. If None, all parameters from the inputs are plotted.
        visualization_methods: List of visualization methods to use. Visualization methods can often be used ontop of one another
            Acceptable visualization methods are:
                - boxplot, density, histogram, scatter, violin, quantiles
        include_priors: Whether to include prior distributions in the plot.
        dependent_variable: Not used in this function, but can be included for future extensions.
        include_previous_steps: Steps to include in the plot; if True, includes all previous steps.
        limit_to_accepted: If True, only plots accepted simulations.
        facet_by: Variables to facet the plots by (e.g., "parameter", "step").
        ax: Matplotlib Axes object to plot on; if None, creates a new figure.
        show: Whether to display the plot immediately.
        save_file: If provided, saves the plot to this file.
    """

    # Determine the simulation inputs and steps to use
    include_steps = include_previous_steps
    if not isinstance(include_steps, bool):
        if not isinstance(parameters, list):
            include_steps = [include_steps]
    else:
        if include_steps:
            include_steps = range(experiment.current_step + 1)
        else:
            include_steps = [experiment.current_step]
    if include_priors and 0 not in include_steps:
        include_steps.extend([0])

    # Collect input data
    sims = []
    for step in include_steps:
        sims.extend(
            [
                i + experiment.n_simulations * step
                for i in range(experiment.n_simulations)
            ]
        )

    input_data = experiment.collect_inputs(sims).with_columns(
        pl.col("simulation")
        .map_elements(experiment.step_from_index, return_dtype=pl.Int64)
        .alias("step")
    )

    # Repair parameters list
    id_cols = ["simulation", "step"]
    if parameters is not None:
        if not isinstance(parameters, list):
            parameters = [parameters]
    else:
        parameters = input_data.drop(
            id_cols + [experiment.seed_variable_name]
        ).columns

    input_data = input_data.select(id_cols + parameters).unpivot(
        index=id_cols, variable_name="parameter"
    )

    if visualization_methods is not None and not isinstance(
        visualization_methods, list
    ):
        visualization_methods = [visualization_methods]

    # Correct facet_by to improve visualization
    if facet_by is not None:
        if isinstance(facet_by, list):
            if len(facet_by) == 0:
                facet_by = None
            elif len(facet_by) == 1:
                facet_by.extend([None])
            elif len(facet_by) > 2:
                raise NotImplementedError(
                    "Plotting does not currently support more than two facet variables"
                )
            facet_type = "grid"
            col_wrap = None
        else:
            facet_by = [facet_by, None]
            facet_type = "wrap"
    if facet_by is None:
        # Introduce a facet if there's more than one parameter for multiple steps, unles two steps indicate a prior (0) and posterior (current)
        if len(parameters) > 1 and not (
            len(include_steps) == 1
            or (
                len(include_steps) == 2
                and 0 in include_steps
                and include_priors
            )
        ):
            facet_by = ["parameter", None]
            facet_type = "wrap"

    # If not faceting by variable, make a plot that is pivoted longer and plot by parameter category
    if facet_by is None:
        if len(include_steps) == 1:
            hue = None
        else:
            hue = "step"
        if "ecdf" in visualization_methods:
            g = sns.displot(input_data, x="value", hue=hue, kind="ecdf")
        elif (
            "boxplot" in visualization_methods
            and "violin" not in visualization_methods
        ):
            g = sns.catplot(
                input_data, x="value", y="parameter", kind="box", hue=hue
            )
            if "scatter" in visualization_methods and hue is not None:
                sns.swarmplot(
                    data=input_data,
                    x="value",
                    y="parameter",
                    col="k",
                    size=3,
                    ax=g.ax,
                )
        elif "violin" in visualization_methods:
            if "boxplot" in visualization_methods:
                inner = "box"
            elif (
                "quantile" in visualization_methods
                or "point_estimate" in visualization_methods
            ):
                inner = "quart"
            elif "scatter" in visualization_methods and hue is not None:
                inner = "stick"
            elif "histogram" in visualization_methods:
                inner = "stick"
            else:
                inner = None

            g = sns.catplot(
                input_data,
                x="value",
                y="parameter",
                kind="violin",
                inner=inner,
                hue=hue,
            )
            if "scatter" in visualization_methods and hue is not None:
                sns.swarmplot(
                    data=input_data,
                    x="value",
                    y="parameter",
                    col="k",
                    size=3,
                    ax=g.ax,
                )
        elif "histogram" in visualization_methods:
            if len(parameters) == 1:
                hue = hue
            else:
                hue = "parameter"
            g = sns.displot(
                input_data,
                x="value",
                hue=hue,
                kind="hist",
                kde=("density" in visualization_methods),
            )
        elif "density" in visualization_methods:
            if len(parameters) == 1:
                hue = hue
            else:
                hue = "parameter"
            g = sns.displot(
                input_data, x="value", hue=hue, kind="kde", fill=True
            )

    # If faceting by category, make a plot panel for each distribution
    else:
        if facet_type == "wrap":
            if "parameter" in facet_by:
                col_wrap = floor(sqrt(len(parameters)))
                if len(include_steps) == 1:
                    hue = None
                else:
                    hue = "step"
            elif "step" in facet_by:
                col_wrap = floor(sqrt(len(include_steps)))
                if len(parameters) == 1:
                    hue = None
                else:
                    hue = "parameter"
        g = sns.FacetGrid(
            input_data,
            col=facet_by[0],
            row=facet_by[1],
            col_wrap=col_wrap,
            sharex=False,
        )
        if "histogram" in visualization_methods:
            g.map_dataframe(
                sns.histplot,
                x="value",
                hue=hue,
                kde=("density" in visualization_methods),
            )
            if "scatter" in visualization_methods:
                g.map_dataframe(sns.rugplot, x="value")
        elif "density" in visualization_methods:
            g.map_dataframe(sns.kdeplot, x="value", hue=hue, fill=True)
    if show:
        plt.show()
    if save_file is not None:
        plt.savefig(save_file)
    return g.ax


def plot_posterior_distribution_2d(
    experiment: Experiment,
    parameters: list[str] | str | None = None,
    visualization_methods: list[str] | str = ["violin", "box"],
    include_priors: bool = False,
    dependent_variable: str | None = None,
    include_previous_steps: list[int] | int | bool = False,
    limit_to_accepted: bool = False,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_file: str | None = None,
) -> plt.Axes:
    """
    Function to plot the distribution of particular parameters over the experiment.
    Args:
        experiment: The Experiment object containing simulation data.
        parameters: List of parameter names to plot. If None, all parameters from the inputs are plotted.
        visualization_methods: List of visualization methods to use. Visualization methods can often be used ontop of one another
            Acceptable visualization methods are:
                - boxplot, density, histogram, scatter, violin, quantiles
        include_priors: Whether to include prior distributions in the plot.
        dependent_variable: Not used in this function, but can be included for future extensions.
        include_previous_steps: Steps to include in the plot; if True, includes all previous steps.
        limit_to_accepted: If True, only plots accepted simulations.
        facet_by: Variables to facet the plots by (e.g., "parameter", "step").
        ax: Matplotlib Axes object to plot on; if None, creates a new figure.
        show: Whether to display the plot immediately.
        save_file: If provided, saves the plot to this file.
    """

    # Determine the simulation inputs and steps to use
    include_steps = include_previous_steps
    if not isinstance(include_steps, bool):
        if not isinstance(parameters, list):
            include_steps = [include_steps]
    else:
        if include_steps:
            include_steps = range(experiment.current_step + 1)
        else:
            include_steps = [experiment.current_step]
    if include_priors and 0 not in include_steps:
        include_steps.extend([0])
        
    if len(parameters) == 1 and dependent_variable is None:
        plot_posterior_distribution(
            experiment=experiment, 
            parameters=parameters, 
            visualization_methods=visualization_methods, 
            include_priors=include_priors, 
            dependent_variable=dependent_variable, 
            include_previous_steps=include_previous_steps,
            limit_to_accepted=limit_to_accepted,
            ax=ax,
            show=show,
            save_file=save_file
        )

    # Collect input data
    sims = []
    for step in include_steps:
        sims.extend(
            [
                i + experiment.n_simulations * step
                for i in range(experiment.n_simulations)
            ]
        )

    input_data = experiment.collect_inputs(sims).with_columns(
        pl.col("simulation")
        .map_elements(experiment.step_from_index, return_dtype=pl.Int64)
        .alias("step")
    )

    # Repair parameters list
    id_cols = ["simulation", "step"]
    if parameters is not None:
        if not isinstance(parameters, list):
            parameters = [parameters]
    else:
        parameters = input_data.drop(
            id_cols + [experiment.seed_variable_name]
        ).columns

    input_data = input_data.select(id_cols + parameters).unpivot(
        index=id_cols, variable_name="parameter"
    )

    if visualization_methods is not None and not isinstance(
        visualization_methods, list
    ):
        visualization_methods = [visualization_methods]
