import argparse

from abmwrappers import wrappers
from abmwrappers.experiment_class import Experiment


def main(
    simulation_index: int,
    img_file: str,
    scenario_key: str,
    clean: bool = False,
    products: list = None,
    products_output_dir: str = None,
):
    experiment = Experiment(img_file=img_file)
    wrappers.products_from_index(
        simulation_index=simulation_index,
        experiment=experiment,
        distance_fn=fn1,# Must be user-defined
        data_processing_fn=fn2, # Must be user-defined
        products=products,
        products_output_dir=products_output_dir,
        scenario_key=scenario_key,
        clean=clean,
    )


parser = argparse.ArgumentParser(description="Run ABM wrapper")
parser.add_argument(
    "--index",
    type=int,
    required=True,
    help="Index of the simulation to be run. Automatically sources correct bundle from history",
)
parser.add_argument(
    "-f",
    "--input-experiment-file",
    type=str,
    required=True,
    help="Path to the experiment history file containing bundled inputs and experiment information",
)
parser.add_argument(
    "-k", "--scenario-key", type=str, required=True, help="Scenario key"
)
parser.add_argument(
    "--clean",
    action="store_true",
    help="Clean up raw output files after processing into products",
)
parser.add_argument(
    "--products",
    nargs="*",
    help="List of products to process (distances, simulations)",
)
parser.add_argument(
    "-o",
    "--products-output-dir",
    type=str,
    help="Output directory for products to be stored",
)

args = parser.parse_args()

main(
    simulation_index=args.index,
    img_file=args.input_experiment_file,
    scenario_key=args.scenario_key,
    clean=args.clean,
    products=args.products,
    products_output_dir=args.products_output_dir,
)
