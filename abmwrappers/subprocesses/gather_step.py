import argparse

from abmwrappers import wrappers


def gather_results(
    img_file: str,
    products_path: str,
):
    wrappers.update_abcsmc_img(img_file, products_path)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--input-experiment-file",
    type=str,
    required=True,
    help="Path to the experiment history file containing bundled inputs and experiment information",
)
parser.add_argument(
    "-i",
    "--products-path",
    type=str,
    required=True,
    help="Path to the products directory",
)
