from math import floor, sqrt
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from abmwrappers.experiment_class import Experiment


def _get_valid_viz(dim: Literal[1, 2]):
    opts = [
        "boxplot",
        "density",
        "violin",
        "histogram",
        "scatter",
        "violin",
        "quantiles",
        "point_estimate",
    ]
    if dim == 1:
        return opts
    else:
        opts.remove("boxplot")
        opts.remove("violin")
        return opts


def _validate_viz(methods: list[str], dim: Literal[1, 2]):
    valid = _get_valid_viz(dim)
    for method in methods:
        if method not in valid:
            raise ValueError(
                f"Invalid method {method} passed to plot for {dim}D data. Please only select from {valid}."
            )


def plot_posterior_distribution(
    experiment: Experiment,
    parameters: list[str] | str | None = None,
    visualization_methods: list[str] | str = ["violin", "boxplot"],
    include_priors: bool = False,
    dependent_variable: str | None = None,
    include_previous_steps: list[int] | int | bool = False,
    limit_to_accepted: bool = False,
    facet_by: list[str] | str | None = None,
    show: bool = True,
    save_file: str | None = None,
):
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

    _validate_viz(visualization_methods, 1)

    # Correct facet_by to improve visualization
    if facet_by is not None:
        if isinstance(facet_by, list):
            if len(facet_by) == 0:
                facet_by = None
            elif len(facet_by) == 1:
                facet_by.extend([None])
                facet_type = "wrap"
            elif len(facet_by) == 2:
                facet_type = "grid"
            else:
                raise NotImplementedError(
                    "Plotting does not currently support more than two facet variables"
                )
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
            sharex = False
        else:
            hue = None
            col_wrap = None
            if "step" in facet_by:
                step_aspect = facet_by.index("step")
                sharex = ["row", "col"][step_aspect]
            else:
                sharex = False

        g = sns.FacetGrid(
            input_data,
            col=facet_by[0],
            row=facet_by[1],
            col_wrap=col_wrap,
            sharex=sharex,
            hue=hue,
        )
        if "histogram" in visualization_methods:
            g.map_dataframe(
                sns.histplot,
                x="value",
                kde=("density" in visualization_methods),
            )
            if "scatter" in visualization_methods:
                g.map_dataframe(sns.rugplot, x="value")
        elif "density" in visualization_methods:
            g.map_dataframe(sns.kdeplot, x="value", fill=True)
            if "scatter" in visualization_methods:
                g.map_dataframe(sns.rugplot, x="value")
    if show:
        plt.show()
    if save_file is not None:
        plt.savefig(save_file)


def plot_posterior_distribution_2d(
    experiment: Experiment,
    parameters: list[str] | str | None = None,
    visualization_methods: list[str] | str = "scatter",
    visualization_methods_marginal: list[str] | str = "density",
    include_priors: bool = False,
    include_previous_steps: list[int] | int | bool = False,
    limit_to_accepted: bool = False,
    show: bool = True,
    save_file: str | None = None,
):
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

    if len(parameters) == 1:
        plot_posterior_distribution(
            experiment=experiment,
            parameters=parameters,
            visualization_methods=visualization_methods_marginal,
            include_priors=include_priors,
            include_previous_steps=include_previous_steps,
            limit_to_accepted=limit_to_accepted,
            show=show,
            save_file=save_file,
        )

    input_data = input_data.select(id_cols + parameters)

    if visualization_methods is not None and not isinstance(
        visualization_methods, list
    ):
        visualization_methods = [visualization_methods]

    _validate_viz(visualization_methods, 2)

    if visualization_methods_marginal is not None and not isinstance(
        visualization_methods_marginal, list
    ):
        visualization_methods_marginal = [visualization_methods_marginal]

    _validate_viz(visualization_methods_marginal, 1)

    if len(include_steps) > 1:
        hue = "step"
    else:
        hue = None

    if len(parameters) == 2:
        g = sns.jointplot(
            data=input_data, x=parameters[0], y=parameters[1], hue=hue
        )
    else:
        if hue is not None:
            to_drop = "simulation"
        else:
            to_drop = id_cols
        g = sns.PairGrid(data=input_data.drop(to_drop), hue=hue)

        # Plots along the main diagonal
        if "histogram" in visualization_methods_marginal:
            g.map_diag(
                sns.histplot,
                kde=("density" in visualization_methods_marginal),
            )
        elif "density" in visualization_methods_marginal:
            g.map_diag(sns.kdeplot, fill=True)
        else:
            raise NotImplementedError(
                "only density and histogram are implemented"
            )

        if "histogram" in visualization_methods:
            g.map_lower(
                sns.histplot,
                kde=("density" in visualization_methods),
                fill=True,
            )
            if "scatter" in visualization_methods:
                g.map_upper(sns.scatterplot)
        elif "density" in visualization_methods:
            g.map_lower(sns.kdeplot, fill=True)
            if "scatter" in visualization_methods:
                g.map_upper(sns.scatterplot)
        elif "scatter" in visualization_methods:
            g.map_lower(sns.scatterplot)
            g.map_upper(sns.scatterplot)

    if show:
        plt.show()
    if save_file is not None:
        plt.savefig(save_file)
