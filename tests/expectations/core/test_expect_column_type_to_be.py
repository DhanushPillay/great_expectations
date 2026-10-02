import numpy as np
import pandas as pd
import pytest

import great_expectations.expectations as gxe
from great_expectations.expectations.core.expect_column_type_to_be import (
    ExpectColumnTypeToBe,
)
from great_expectations.self_check.util import build_sa_validator_with_data


@pytest.mark.unit
def test_registered_and_uses_table_column_types_metric():
    assert gxe.ExpectColumnTypeToBe is ExpectColumnTypeToBe
    assert ExpectColumnTypeToBe.metric_dependencies == ("table.column_types",)
    assert set(ExpectColumnTypeToBe.success_keys) == {"column", "type_"}
    assert ExpectColumnTypeToBe.args_keys == ("column", "type_")


@pytest.mark.unit
def test_validate_pandas_success_and_observed_value():
    expectation = ExpectColumnTypeToBe(column="a", type_="int64")
    result = expectation._validate_pandas(
        actual_column_type=np.dtype("int64"), expected_type="int64"
    )
    assert result["success"] is True
    assert result["result"] == {"observed_value": "int64"}


@pytest.mark.unit
def test_validate_pandas_failure_reports_observed_value():
    expectation = ExpectColumnTypeToBe(column="a", type_="int64")
    result = expectation._validate_pandas(
        actual_column_type=np.dtype("float64"), expected_type="int64"
    )
    assert result["success"] is False
    assert result["result"] == {"observed_value": "float64"}


@pytest.mark.unit
def test_validate_pandas_native_int_success():
    expectation = ExpectColumnTypeToBe(column="a", type_="int")
    result = expectation._validate_pandas(actual_column_type=np.dtype("int64"), expected_type="int")
    assert result["success"] is True
    assert result["result"] == {"observed_value": "int64"}


@pytest.mark.unit
def test_validate_pandas_object_matches_only_object():
    expectation = ExpectColumnTypeToBe(column="a", type_="object")
    result = expectation._validate_pandas(
        actual_column_type=np.dtype("object"), expected_type="object"
    )
    assert result["success"] is True

    str_expectation = ExpectColumnTypeToBe(column="a", type_="str")
    str_result = str_expectation._validate_pandas(
        actual_column_type=np.dtype("object"), expected_type="str"
    )
    assert str_result["success"] is False
    assert str_result["result"] == {"observed_value": "object"}


@pytest.mark.unit
def test_validate_pandas_extension_types_match_and_mismatch():
    expectation = ExpectColumnTypeToBe(column="a", type_="Int64")
    result = expectation._validate_pandas(actual_column_type=pd.Int64Dtype(), expected_type="Int64")
    assert result["success"] is True

    string_expectation = ExpectColumnTypeToBe(column="a", type_="string")
    string_result = string_expectation._validate_pandas(
        actual_column_type=pd.StringDtype(), expected_type="string"
    )
    assert string_result["success"] is True


@pytest.mark.unit
def test_validate_pandas_known_but_nonmatching_returns_false():
    for known_type in ("float64", "Float64", "string", "boolean"):
        expectation = ExpectColumnTypeToBe(column="a", type_=known_type)
        result = expectation._validate_pandas(
            actual_column_type=np.dtype("int64"), expected_type=known_type
        )
        assert result["success"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_dtype,expected_type",
    [
        pytest.param(np.dtype("int64"), "Int64", id="int64-column-vs-Int64"),
        pytest.param(pd.Int64Dtype(), "int64", id="Int64-column-vs-int64"),
        pytest.param(np.dtype("bool"), "boolean", id="bool-column-vs-boolean"),
        pytest.param(pd.BooleanDtype(), "bool", id="boolean-column-vs-bool"),
        pytest.param(np.dtype("float64"), "Float64", id="float64-column-vs-Float64"),
        pytest.param(pd.Float64Dtype(), "float64", id="Float64-column-vs-float64"),
    ],
)
def test_validate_pandas_nullable_and_numpy_dtypes_are_distinct(column_dtype, expected_type):
    """Nullable and numpy dtypes share a scalar type but are different column types."""
    expectation = ExpectColumnTypeToBe(column="a", type_=expected_type)
    result = expectation._validate_pandas(
        actual_column_type=column_dtype, expected_type=expected_type
    )
    assert result["success"] is False
    assert result["result"] == {"observed_value": str(column_dtype)}


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_dtype,expected_type",
    [
        pytest.param(pd.Int64Dtype(), "Int64", id="Int64"),
        pytest.param(pd.BooleanDtype(), "boolean", id="boolean"),
        pytest.param(pd.Float64Dtype(), "Float64", id="Float64"),
        pytest.param(np.dtype("float64"), "float", id="float-alias"),
        pytest.param(np.dtype("object"), "O", id="O-alias"),
        pytest.param(np.dtype("object"), "object_", id="object_-alias"),
        pytest.param(pd.CategoricalDtype(["x", "y"]), "category", id="category"),
        pytest.param(
            pd.DatetimeTZDtype(unit="ns", tz="UTC"), "datetime64[ns, UTC]", id="tz-datetime"
        ),
        pytest.param(pd.IntervalDtype("int64"), "interval", id="interval-family"),
        pytest.param(pd.StringDtype(), "string", id="string-NA"),
        pytest.param(np.dtype("datetime64[ns]"), "datetime64", id="unitless-datetime-ns"),
        pytest.param(np.dtype("datetime64[us]"), "datetime64", id="unitless-datetime-us"),
        pytest.param(np.dtype("timedelta64[ns]"), "timedelta64", id="unitless-timedelta"),
    ],
)
def test_validate_pandas_matches_exact_dtype(column_dtype, expected_type):
    expectation = ExpectColumnTypeToBe(column="a", type_=expected_type)
    result = expectation._validate_pandas(
        actual_column_type=column_dtype, expected_type=expected_type
    )
    assert result["success"] is True


@pytest.mark.unit
def test_validate_pandas_nan_backed_string_column_does_not_match_string():
    """Every StringDtype compares equal to "string"; only the pd.NA-backed one is that dtype."""
    column_dtype = pd.StringDtype(na_value=np.nan)
    expectation = ExpectColumnTypeToBe(column="a", type_="string")
    result = expectation._validate_pandas(actual_column_type=column_dtype, expected_type="string")
    assert result["success"] is False
    assert result["result"] == {"observed_value": "str"}


@pytest.mark.unit
def test_validate_pandas_observed_value_names_the_nullable_dtype():
    expectation = ExpectColumnTypeToBe(column="a", type_="int64")
    result = expectation._validate_pandas(actual_column_type=pd.Int64Dtype(), expected_type="int64")
    assert result["result"] == {"observed_value": "Int64"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "column_dtype,expected_type",
    [
        pytest.param(pd.DatetimeTZDtype(unit="ns", tz="UTC"), "datetime64[ns]", id="tz-vs-naive"),
        pytest.param(np.dtype("datetime64[ns]"), "datetime64[us]", id="unit-mismatch"),
        pytest.param(pd.CategoricalDtype(["x"]), "object", id="category-vs-object"),
        pytest.param(
            pd.DatetimeTZDtype(unit="ns", tz="UTC"), "datetime64", id="tz-vs-unitless-naive"
        ),
        pytest.param(np.dtype("timedelta64[ns]"), "datetime64", id="timedelta-vs-datetime"),
        pytest.param(np.dtype("int64"), "datetime64", id="int-vs-unitless-datetime"),
    ],
)
def test_validate_pandas_datetime_and_category_mismatches_return_false(column_dtype, expected_type):
    expectation = ExpectColumnTypeToBe(column="a", type_=expected_type)
    result = expectation._validate_pandas(
        actual_column_type=column_dtype, expected_type=expected_type
    )
    assert result["success"] is False


@pytest.mark.unit
def test_validate_pandas_unknown_type_raises():
    expectation = ExpectColumnTypeToBe(column="a", type_="NUMBER")
    with pytest.raises(ValueError, match="Unrecognized pandas type"):
        expectation._validate_pandas(actual_column_type=np.dtype("int64"), expected_type="NUMBER")


@pytest.mark.unit
def test_validate_missing_column_fails_with_null_observed_value():
    expectation = ExpectColumnTypeToBe(column="missing", type_="INTEGER")
    result = expectation._validate(metrics={"table.column_types": []})
    assert result == {"success": False, "result": {"observed_value": None}}


@pytest.mark.unit
def test_result_has_no_row_level_fields():
    expectation = ExpectColumnTypeToBe(column="a", type_="int64")
    result = expectation._validate_pandas(
        actual_column_type=np.dtype("int64"), expected_type="int64"
    )
    assert set(result["result"]) == {"observed_value"}
    assert "mostly" not in expectation.success_keys


@pytest.mark.sqlite
def test_delegates_to_compare_column_type(sa, mocker):
    df = pd.DataFrame({"str_col": ["a", "b", "c"]})
    validator = build_sa_validator_with_data(
        df=df, sa_engine_name="sqlite", table_name="column_type_to_be_wiring"
    )

    mock_compare = mocker.patch(
        "great_expectations.expectations.core.expect_column_type_to_be.compare_column_type",
        return_value=(True, "SENTINEL_TYPE"),
    )

    result = validator.expect_column_type_to_be("str_col", type_="TEXT")

    mock_compare.assert_called_once_with(validator.execution_engine, mocker.ANY, "TEXT")
    assert result.success is True
    assert result.result["observed_value"] == "SENTINEL_TYPE"


@pytest.mark.sqlite
def test_sqlite_end_to_end_success_and_failure(sa):
    df = pd.DataFrame({"col": ["test_val1", "test_val2"]})
    validator = build_sa_validator_with_data(
        df=df,
        sa_engine_name="sqlite",
        table_name="expect_column_type_to_be_sqlite_e2e",
    )

    success_result = validator.expect_column_type_to_be("col", type_="TEXT")
    assert success_result.success is True
    assert success_result.result["observed_value"] == "TEXT"

    failure_result = validator.expect_column_type_to_be("col", type_="INTEGER")
    assert failure_result.success is False
    assert failure_result.exception_info["raised_exception"] is False


@pytest.mark.sqlite
def test_sqlite_unknown_type_reports_exception(sa):
    df = pd.DataFrame({"col": ["test_val1", "test_val2"]})
    validator = build_sa_validator_with_data(
        df=df,
        sa_engine_name="sqlite",
        table_name="expect_column_type_to_be_sqlite_unknown",
    )

    with pytest.raises(ValueError, match="Unrecognized sqlalchemy type"):
        validator.expect_column_type_to_be("col", type_="NUMBER")
