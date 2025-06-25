import pytest

from abmwrappers import utils


def combine_params_test_wrapper(
    base: dict, new: dict, expect: dict, overwrite: bool = False
):
    base_dict = {"baseScenario": base}
    returned, _ = utils.combine_params_dicts(
        baseline_dict=base_dict, new_dict=new, overwrite_unnested=overwrite
    )

    flat_expect = utils.flatten_dict(expect)
    for k, v in utils.flatten_dict(returned["baseScenario"]).items():
        assert k in flat_expect.keys()
        assert flat_expect[k] == v


def test_simple_combine_params():
    combine_params_test_wrapper(
        base={"scale": 1.0, "shape": 2.0},
        new={"shape": 3.0},
        expect={"scale": 1.0, "shape": 3.0},
    )


def test_combine_missing_param():
    with pytest.raises(
        Exception, match="'mean' not present in default params list."
    ):
        combine_params_test_wrapper(
            base={"scale": 1.0, "shape": 2.0},
            new={"mean": 3.0},
            expect={"scale": 1.0, "shape": 3.0},
        )


def test_hierarchical_params():
    combine_params_test_wrapper(
        base={"gamma": {"scale": 1.0, "shape": 2.0}},
        new={"gamma": {"shape": 3.0}},
        expect={"gamma": {"scale": 1.0, "shape": 3.0}},
    )


def test_overwrite_hierarchical_params():
    combine_params_test_wrapper(
        base={"distro": {"gamma": {"scale": 1.0, "shape": 2.0}}},
        new={"distro": {"norm": {"mean": 0.0, "sd": 1.0}}},
        expect={"distro": {"norm": {"mean": 0.0, "sd": 1.0}}},
        overwrite=True,
    )
