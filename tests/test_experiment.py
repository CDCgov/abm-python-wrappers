import os
from polars.testing import assert_frame_equal

from abmwrappers.experiment_class import Experiment
from abctools.abc_classes import SimulationBundle

def test_compress_and_restore():
    experiment = Experiment(experiments_directory="tests",
                            config_file="tests/input/test_config.yaml")
    
    init_bundle = experiment.initialize_simbundle()
    output_file = "tests/output/experiment_history.pkl"
    os.makedirs("tests/output", exist_ok=True)
    experiment.compress_and_save(output_file=output_file)

    restored_experiment = Experiment(img_file=output_file)

    assert isinstance(restored_experiment, Experiment)
    restored_bundle = restored_experiment.simulation_bundles[0]
    assert isinstance(restored_bundle, SimulationBundle)

    assert_frame_equal(init_bundle.inputs, restored_bundle.inputs)
    assert init_bundle.status == restored_bundle.status
    assert init_bundle.merge_history == restored_bundle.merge_history
    assert init_bundle.weights == restored_bundle.weights
    assert init_bundle._step_number == restored_bundle._step_number
    assert init_bundle._baseline_params == restored_bundle._baseline_params
