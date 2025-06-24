from abmwrappers import utils

def combine_params_test_wrapper(base: dict, new: dict, expect: dict, overwrite: bool = False):
    base_dict = {"baseScenario": base}
    returned, _ = utils.combine_params_dicts(
        baseline_dict=base_dict,
        new_dict=new,
        overwrite_unnested=overwrite
    )

    flat_expect = utils.flatten_dict(expect)
    for k, v in utils.flatten_dict(returned["baseScenario"]).items():
        assert k in flat_expect.keys()
        assert flat_expect[k] == v

def test_simple_combine_params():
    base = {"scale": 1.0, "shape": 2.0}
    new = {"shape": 3.0}
    expect = {"scale": 1.0, "shape": 3.0}
    combine_params_test_wrapper(
        base,
        new,
        expect,
    )