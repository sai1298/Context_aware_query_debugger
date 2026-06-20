from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    # pandas
    WRONG_GROUPBY_KEY = "wrong_groupby_key"
    WRONG_AGGREGATION = "wrong_aggregation"
    WRONG_MERGE_KEY = "wrong_merge_key"
    WRONG_FILTER_CONDITION = "wrong_filter_condition"
    WRONG_COLUMN_SELECTION = "wrong_column_selection"
    MISSING_RESET_INDEX = "missing_reset_index"
    # numpy
    WRONG_AXIS = "wrong_axis"
    BROADCASTING_ERROR = "broadcasting_error"
    WRONG_OPERATION = "wrong_operation"
    # matplotlib
    WRONG_PLOT_TYPE = "wrong_plot_type"
    MISSING_LABELS = "missing_labels"
    WRONG_DATA_MAPPING = "wrong_data_mapping"
    # general
    OFF_BY_ONE = "off_by_one"
    WRONG_VARIABLE = "wrong_variable"
    LOGIC_INVERSION = "logic_inversion"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    UNKNOWN = "unknown"


# Map error types to their human-readable categories
ERROR_CATEGORIES: dict[ErrorType, str] = {
    ErrorType.WRONG_GROUPBY_KEY: "Pandas — incorrect groupby column",
    ErrorType.WRONG_AGGREGATION: "Pandas — incorrect aggregation function",
    ErrorType.WRONG_MERGE_KEY: "Pandas — incorrect merge/join key",
    ErrorType.WRONG_FILTER_CONDITION: "Pandas — incorrect filter condition",
    ErrorType.WRONG_COLUMN_SELECTION: "Pandas — wrong column selected",
    ErrorType.MISSING_RESET_INDEX: "Pandas — missing reset_index() call",
    ErrorType.WRONG_AXIS: "NumPy — incorrect axis argument",
    ErrorType.BROADCASTING_ERROR: "NumPy — shape mismatch / broadcasting error",
    ErrorType.WRONG_OPERATION: "NumPy — wrong mathematical operation",
    ErrorType.WRONG_PLOT_TYPE: "Matplotlib — wrong chart/plot type",
    ErrorType.MISSING_LABELS: "Matplotlib — missing axis labels or title",
    ErrorType.WRONG_DATA_MAPPING: "Matplotlib — incorrect data-to-visual mapping",
    ErrorType.OFF_BY_ONE: "General — off-by-one index error",
    ErrorType.WRONG_VARIABLE: "General — wrong variable referenced",
    ErrorType.LOGIC_INVERSION: "General — inverted boolean or comparison",
    ErrorType.AMBIGUOUS_INTENT: "General — intent is unclear / multiple interpretations",
    ErrorType.UNKNOWN: "General — unclassified error",
}
