import polars as pl
import pytest
from polars.testing import assert_frame_equal

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


def test_overwrite_hierarchical_params_expand():
    combine_params_test_wrapper(
        base={"gamma": {"scale": 1.0}},
        new={"gamma": {"shape": 3.0}},
        expect={"gamma": {"scale": 1.0, "shape": 3.0}},
        overwrite=True,
    )


def test_overwrite_hierarchical_params_expand_fail():
    with pytest.raises(
        Exception, match="'gamma>>>shape' not present in default params list."
    ):
        combine_params_test_wrapper(
            base={"gamma": {"scale": 1.0}},
            new={"gamma": {"shape": 3.0}},
            expect={"gamma": {"scale": 1.0, "shape": 3.0}},
            overwrite=False,
        )


def test_col_keys_from_path():
    test_string = "test/path/to/col/simulation_0/scenarios=1/test.csv"
    d = utils.column_keys_from_path(test_string)
    assert d["simulation"] == 0
    assert d["scenario"] == 1


def test_col_keys_from_path_failure():
    test_string = "test/path/to/col/simulation_0/12scenarios=1/test.csv"
    with pytest.raises(
        Exception,
        match="Multiple scenario indices found in path segment '12scenarios=1'.",
    ):
        utils.column_keys_from_path(test_string)


def test_get_caller():
    def dummy_caller():
        assert utils.get_caller() == __file__

    dummy_caller()


def test_get_caller_failure():
    depth = 100

    def dummy_caller(depth):
        assert utils.get_caller(depth) == __file__

    with pytest.raises(
        Exception, match=f"Call stack is not deep enough for depth={depth}."
    ):
        dummy_caller(depth)


def test_vstack_dfs():
    df1 = pl.DataFrame(
        {
            "var1": [1, 1],
            "var2": [1, 1],
        }
    )
    df2 = pl.DataFrame({"var1": 2})
    df3 = pl.DataFrame({"var2": 3, "var1": 3})

    df_stack = utils._vstack_dfs([df1, df2, df3])

    expected_null_counts = pl.DataFrame(
        {
            "var1": 0,
            "var2": 1,
        }
    )
    observed_null_counts = df_stack.select(
        pl.all().is_null().sum().cast(pl.Int64)
    )

    assert df_stack.height == 4
    assert df_stack.filter(pl.col("var1") == 1).height == 2
    assert df_stack.filter(pl.col("var1") == 2).height == 1
    assert df_stack.filter(pl.col("var1") == 3).height == 1
    assert_frame_equal(expected_null_counts, observed_null_counts)


def test_params_grid_search():
    params = {"mean": [1, 2, 3], "sd": [0.1, 0.2, 0.3]}
    df = utils.params_grid_search(params)
    assert df.height == 9
    assert df.filter(pl.col("mean") == 1).height == 3
    assert df.filter(df.is_unique()).height == 9
    assert df.columns == ["mean", "sd"]


def test_params_grid_search_duplicate_vals():
    params = {"mean": [1, 1], "sd": [0.1, 0.2]}
    with pytest.raises(
        Exception, match="Values are not unique for every parameter."
    ):
        utils.params_grid_search(params)


def test_params_grid_search_nested():
    params = {"my_distribution": {"mean": [1, 2, 3], "sd": [0.1, 0.2, 0.3]}}
    df = utils.params_grid_search(params)
    assert df.height == 9
    assert df.filter(pl.col("my_distribution>>>mean") == 1).height == 3
    assert df.filter(df.is_unique()).height == 9
    assert df.columns == ["my_distribution>>>mean", "my_distribution>>>sd"]


def test_df_to_simulation_dict():
    test_df = pl.DataFrame({"mean": [1, 2, 3], "sd": [0.1, 0.2, 0.3]})
    test_dict = utils.df_to_simulation_dict(test_df)

    assert len(test_dict) == test_df.height
    assert (
        test_dict[0]["mean"] == test_df.head(1).select(pl.col("mean")).item()
    )


def test_df_to_simulation_dict_indices():
    test_df = pl.DataFrame(
        {"mean": [1, 2, 3], "sd": [0.1, 0.2, 0.3], "simulation": [2, 1, 0]}
    )
    test_dict = utils.df_to_simulation_dict(test_df)

    assert len(test_dict) == test_df.height
    assert (
        test_dict[0]["mean"]
        == test_df.filter(pl.col("simulation") == 0)
        .select(pl.col("mean"))
        .item()
    )


def test_df_to_simulation_dict_indices_failure():
    test_df = pl.DataFrame(
        {"mean": [1, 2, 3], "sd": [0.1, 0.2, 0.3], "simulation": [2, 2, 2]}
    )

    with pytest.raises(
        Exception, match="Values in 'simulation' column are not unique."
    ):
        utils.df_to_simulation_dict(test_df)


def test_default_command():
    default_ixa = "./target --config ./input --prefix ./output/"
    default_gcm = "java -jar target -i input -o output -t 4"

    ixa_cmd = utils.write_default_cmd("input", "output", "ixa", "target")
    assert " ".join(ixa_cmd) == default_ixa
    gcm_cmd = utils.write_default_cmd("input", "output", "gcm", "target")
    assert " ".join(gcm_cmd) == default_gcm

    with pytest.raises(
        Exception, match="Unsupported model type: fake. must be 'gcm' or 'ixa'"
    ):
        utils.write_default_cmd("input", "output", "fake", "target")


def test_default_command_ixa_path():
    default_ixa = "/target --config /input --prefix /output/"
    ixa_cmd = utils.write_default_cmd("/input", "/output", "ixa", "/target")
    assert " ".join(ixa_cmd) == default_ixa
