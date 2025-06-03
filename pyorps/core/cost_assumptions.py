from typing import Union, Optional, Any, Callable
import os
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd

from geopandas import GeoDataFrame, GeoSeries

from .exceptions import (InvalidSourceError, FileLoadError, FormatError,
                         NoSuitableColumnsError)


class CostAssumptions:
    """
    A class for handling cost assumptions for rasterization.

    This class handles:
    - Loading cost assumptions from files (CSV, Excel, JSON) or generating of cost
    assumptions from a dictionary or a GeoDataFrame.
    - Mapping costs to features in a GeoDataFrame
    - Managing hierarchical cost structures
    """

    def __init__(
            self,
            source: Optional[Union[str, dict]] = None
    ):
        """
        Initialize the CostAssumptions object.

        Parameters:
            source:
                1. Path to a cost assumptions file
                2. A dictionary of cost values
        """
        self.source = source
        self.cost_assumptions = {}
        self.main_feature = None
        self.side_features = []

        if source is not None:
            if isinstance(source, (dict, str)):
                self.load(source)
            else:
                raise InvalidSourceError(
                    f"Parameter 'source' must be either a string, a dictionary or a "
                    f"GeoDataFrame, not {type(source)}"
                )
            if not self.cost_assumptions:
                raise FormatError(f"The format of the cost assumptions file or "
                                  f"dictionary is invalid. Please check "
                                  f"the format of your cost assumptions input: "
                                  f"{self.source}")

    def load(
            self,
            source: Union[str, dict]
    ) -> dict:
        """
        Load cost assumptions from a file or dictionary.

        Parameters:
            source: Path to a file or a dictionary containing cost assumptions

        Returns:
            dictionary of cost assumptions
        """
        if isinstance(source, dict):
            keys, costs = next(iter(source.items()))
            if isinstance(keys, tuple):
                self.main_feature, *self.side_features = keys
            else:
                self.main_feature = keys
            self.cost_assumptions = costs
            return self.cost_assumptions

        if isinstance(source, str) and os.path.isfile(source):
            file_ext = Path(source).suffix.lower()
            loader_map = {
                '.csv': self._load_csv_cost_assumptions,
                '.json': self._load_json_cost_assumptions,
                '.xlsx': self._load_excel_cost_assumptions,
                '.xls': self._load_excel_cost_assumptions,
            }

            loader: Callable[[str], dict] | None = loader_map.get(file_ext)
            if not loader:
                raise InvalidSourceError(f"Unsupported file format: {file_ext}")

            return loader(source)

        raise InvalidSourceError("Source must be a dictionary or a valid file path")

    def _load_csv_cost_assumptions(
            self,
            filepath: str
    ) -> dict:
        """
        Load cost assumptions from a CSV file with auto-detection of encoding,
        delimiter, and decimal separator.

        Parameters:
            filepath: Path to the CSV file

        Returns:
            dictionary of cost assumptions
        """
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        decimal_separators = ['.', ',']
        common_delimiters = [',', ';', '\t', '|']

        # Try using csv.Sniffer to detect the delimiter
        for encoding in encodings:
            try:
                # Read a sample to detect the dialect
                with open(filepath, 'r', encoding=encoding) as f:
                    sample = f.read(4096)

                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter

                # Try with detected delimiter and different decimal separators
                for decimal in decimal_separators:
                    try:
                        df = pd.read_csv(
                            filepath,
                            encoding=encoding,
                            delimiter=delimiter,
                            decimal=decimal
                        )
                        if df.empty:
                            continue
                        df = self._convert_numeric_columns(df)
                        self.cost_assumptions = self.convert_df_to_cost_dict(df)
                        return self.cost_assumptions
                    except (pd.errors.ParserError, ValueError):
                        continue
            except (csv.Error, UnicodeDecodeError, IOError):
                # If auto-detection fails, try common delimiters
                for delimiter in common_delimiters:
                    for decimal in decimal_separators:
                        try:
                            df = pd.read_csv(
                                filepath,
                                encoding=encoding,
                                delimiter=delimiter,
                                decimal=decimal
                            )
                            if df.empty:
                                continue
                            df = self._convert_numeric_columns(df)
                            self.cost_assumptions = self.convert_df_to_cost_dict(df)
                            return self.cost_assumptions
                        except (pd.errors.ParserError, ValueError, UnicodeDecodeError):
                            continue

        raise FileLoadError(f"Could not read CSV file {filepath}. Tried multiple "
                            f"encodings and formats.")

    def _load_json_cost_assumptions(
            self,
            filepath: str
    ) -> dict:
        """
        Load cost assumptions from a JSON file with auto-detection of encoding.

        Parameters:
            filepath: Path to the JSON file

        Returns:
            dictionary of cost assumptions
        """
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        last_error = None

        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    data = json.load(f)

                    # Check if it's the new format with metadata
                    if (isinstance(data, dict) and 'metadata' in data and
                            'cost_assumptions' in data):
                        self.main_feature = data['metadata']['main_feature']
                        self.side_features = data['metadata'].get('side_features', [])

                        # Handle tuple keys if necessary
                        if self.side_features:
                            cost_dict = {}
                            for key_str, value in data['cost_assumptions'].items():
                                if "__" in key_str:
                                    # Convert string representation back to tuple
                                    tuple_key = tuple(key_str.split("__"))
                                    cost_dict[tuple_key] = value
                                else:
                                    cost_dict[key_str] = value
                            self.cost_assumptions = cost_dict
                        else:
                            self.cost_assumptions = data['cost_assumptions']
                    else:
                        # Legacy format - just a plain dictionary
                        self.cost_assumptions = data
                    if len(self.cost_assumptions) == 0:
                        raise FileLoadError(f"Failed to read json file {filepath}. "
                                            f"File contains no data or is not in "
                                            f"the correct format!")
                    return self.cost_assumptions
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                last_error = e
                continue

        raise FileLoadError(f"Could not read JSON file {filepath}: {last_error}")

    def _load_excel_cost_assumptions(
            self,
            filepath: str
    ) -> dict:
        """
        Load cost assumptions from an Excel file, handling different decimal separators.

        Parameters:
            filepath: Path to the Excel file

        Returns:
            dictionary of cost assumptions
        """
        try:
            # First try default settings
            df = pd.read_excel(filepath)
            if df.empty:
                raise FileLoadError(f"Failed to read Excel file {filepath}. File "
                                    f"contains no data or is not in the "
                                    f"correct format!")

            df = self._convert_numeric_columns(df)
            self.cost_assumptions = self.convert_df_to_cost_dict(df)
            return self.cost_assumptions
        except (pd.errors.ParserError, ValueError, IOError) as first_error:
            # If there's an issue, try reading as strings and convert manually
            try:
                df = pd.read_excel(filepath, dtype=str)
                if df.empty:
                    msg = (f"Failed to read Excel file {filepath}. File contains no "
                           f"data or is not in the correct format!")
                    raise FileLoadError(msg)
                df = self._convert_numeric_columns(df)
                self.cost_assumptions = self.convert_df_to_cost_dict(df)
                return self.cost_assumptions
            except (pd.errors.ParserError, ValueError, IOError) as e:
                msg = (f"Failed to read Excel file {filepath}. Original error: "
                       f"{first_error}. Second attempt error: {e}")
                raise FileLoadError(msg)

    def convert_df_to_cost_dict(
            self,
            df: pd.DataFrame
    ) -> dict:
        """
        Convert a DataFrame to a nested dictionary for cost assumptions.

        Parameters:
            df: DataFrame containing cost assumptions with hierarchical structure

        Returns:
            dictionary of cost assumptions with nested structure based on DataFrame
            columns

        Uses one numeric column for costs, and all other columns as a hierarchical
        index:
        - The first column is the 'main_feature'
        - All additional columns are 'side_features'
        """
        # First ensure numeric columns are properly converted
        df = self._convert_numeric_columns(df)

        # Find the numeric column for costs
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            raise FormatError("No numeric column found for cost values")

        # Use the first numeric column as the cost column
        cost_column = numeric_columns[0]

        # All non-numeric columns form the hierarchical index
        index_columns = [col for col in df.columns if col != cost_column]
        if not index_columns:
            raise FormatError("No columns found for feature hierarchy")

        # Fill NaN values and assign features
        for ci, column in enumerate(index_columns):
            df[column] = df[column].fillna('')
            if ci == 0:
                self.main_feature = column
            else:
                self.side_features.append(column)

        # Create a series with a MultiIndex and convert to nested dictionaries
        cost_series = df.set_index(index_columns)[cost_column]
        return cost_series.to_dict()

    @staticmethod
    def _convert_numeric_columns(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert columns to numeric, handling different decimal separators.

        Parameters:
            df: DataFrame with potential numeric columns that might use different
            decimal separators

        Returns:
            DataFrame with properly converted numeric columns
        """
        for col in df.columns:
            # Skip columns that are already numeric or clearly not numeric
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype != object:
                continue

            # Try to convert using various decimal separators
            original_values = df[col].copy()

            # Try standard conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # If conversion successful and no NaN values were introduced, continue
            if df[col].isna().sum() == original_values.isna().sum():
                continue

            # Restore original values for next attempt
            df[col] = original_values

            # Try comma as decimal separator
            try:
                df[col] = df[col].str.replace(',', '.').astype(float)
            except (ValueError, AttributeError):
                # Revert to original if both attempts fail
                df[col] = original_values

        return df

    def apply_to_geodataframe(
            self,
            gdf: GeoDataFrame,
            main_feature: Optional[str] = None,
            side_features: Optional[list[str]] = None
    ):
        """
        Apply cost assumptions to a GeoDataFrame.

        Parameters:
            gdf: GeoDataFrame to apply costs to
            main_feature: Main feature column name
            side_features: list of side feature column names or single side feature name

        Returns:
            GeoDataFrame with 'cost' column added
        """
        main_feature = main_feature or self.main_feature

        if side_features is None:
            side_features = self.side_features
        elif isinstance(side_features, str):
            side_features = [side_features]

        if main_feature is None:
            raise FormatError("Main feature column not specified")

        # Fill NA values
        gdf[main_feature] = gdf[main_feature].fillna('')
        for feat in side_features:
            gdf[feat] = gdf[feat].fillna('')

        # Handle different cost assumption structures
        first_key = next(iter(self.cost_assumptions), None)

        if isinstance(first_key, tuple):
            # Complex tuple keys structure - from multi-index
            self._apply_tuple_costs(gdf, main_feature, side_features)
        elif side_features and isinstance(next(iter(self.cost_assumptions.values()),
                                               None), dict):
            # Nested dictionary structure
            self._apply_nested_costs(gdf, main_feature, side_features)
        else:
            # Simple mapping with numeric values
            gdf['cost'] = gdf[main_feature].map(self.cost_assumptions)

        return gdf

    def _apply_tuple_costs(
            self,
            gdf: GeoDataFrame,
            main_feature: Optional[str] = None,
            side_features: Optional[list[str]] = None
    ):
        """
        Apply costs to the GeoDataFrame based on tuple keys in cost assumptions.

        Parameters:
            gdf: GeoDataFrame to update with cost values
            main_feature: Column name for the primary feature
            side_features: List of column names for secondary features

        Returns:
            None (modifies gdf in-place)
        """
        # Create wildcard dictionary for default values
        iter_items = self.cost_assumptions.items()
        wild_cards = {keys[0]: value for keys, value in iter_items if '' in keys}

        # Apply specific mappings
        for keys, value in self.cost_assumptions.items():
            main_key, *side_keys = keys
            mask = gdf[main_feature] == main_key
            for side_feature, side_key in zip(side_features, side_keys):
                mask &= gdf[side_feature] == side_key
            gdf.loc[mask, 'cost'] = value

        # Apply wildcards for missing values
        cost_nan = gdf['cost'].isna()
        for wild_card_key, wild_card_value in wild_cards.items():
            mask = (gdf[main_feature] == wild_card_key) & cost_nan
            gdf.loc[mask, 'cost'] = wild_card_value

    def _apply_nested_costs(
            self,
            gdf: GeoDataFrame,
            main_feature: Optional[str] = None,
            side_features: Optional[list[str]] = None
    ):
        """
        Apply costs to the GeoDataFrame based on nested dictionary cost assumptions.

        Parameters:
            gdf: GeoDataFrame to update with cost values
            main_feature: Column name for the primary feature
            side_features: List containing a single column name for the
            secondary feature

        Returns:
            None (modifies gdf in-place)
        """
        if len(side_features) != 1:
            msg = "Multiple side features not supported for nested dictionary structure"
            raise FormatError(msg)

        side_feature = side_features[0]

        # Iterate over each main feature value and its inner dictionary
        for main_value, inner_dict in self.cost_assumptions.items():
            # Create mask for the main feature
            main_mask = gdf[main_feature] == main_value

            # Apply costs for each side feature value
            for side_value, cost in inner_dict.items():
                if side_value == "" or pd.isnull(side_value):
                    # Handle wildcard/default values
                    side_mask = (gdf[side_feature].isnull() |
                                 (gdf[side_feature] == side_value) |
                                 ~gdf[side_feature].isin(inner_dict.keys()))
                else:
                    # Standard case - exact match
                    side_mask = gdf[side_feature] == side_value

                # Apply cost where both masks match
                combined_mask = main_mask & side_mask
                gdf.loc[combined_mask, 'cost'] = cost

    def to_csv(
            self,
            filepath: str,
            separator: str = ';',
            decimal: str = '.',
            encoding: str = 'ISO-8859-1'
    ) -> None:
        """
        Save the cost assumptions to a CSV file.

        Parameters:
            filepath: Path where to save the CSV file
            separator: Column separator character (default is ';')
            decimal: Decimal separator character (default is '.')
            encoding: The encoding of the file (default is 'ISO-8859-1')
        """
        # Convert the nested dictionary to DataFrame
        df = self.cost_dict_to_df(self.cost_assumptions)

        # Handle decimal separator conversion if needed
        if decimal == ',':
            # Convert numeric columns to use comma as decimal separator
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(str).str.replace('.', ',')

        # Save DataFrame to CSV
        df.to_csv(filepath, sep=separator, index=False, encoding=encoding)

    def to_json(
            self,
            filepath: str,
            indent: int = 2,
            encoding: str = 'ISO-8859-1'
    ) -> None:
        """
        Save the cost assumptions to a JSON file.

        Parameters:
            filepath: Path where to save the JSON file
            indent: Number of spaces for indentation (default is 2)
            encoding: The encoding of the file (default is 'ISO-8859-1')
        """
        # Create a structure that can be properly serialized to JSON
        output_dict = {
            'metadata': {
                'main_feature': self.main_feature,
                'side_features': self.side_features
            },
            'cost_assumptions': {}
        }

        # Convert the cost assumptions dictionary to a JSON-serializable format
        if self.cost_assumptions:
            first_key = next(iter(self.cost_assumptions))
            if isinstance(first_key, tuple):
                # Handle tuple keys by converting them to string representations
                for key, value in self.cost_assumptions.items():
                    key_str = "__".join(str(k) for k in key)
                    output_dict['cost_assumptions'][key_str] = value
            else:
                # Regular keys can be directly serialized
                output_dict['cost_assumptions'] = self.cost_assumptions

        with open(filepath, mode='w', encoding=encoding) as f:
            json.dump(output_dict, f, indent=indent)

    def to_excel(
            self,
            filepath: str,
            sheet_name: str = 'CostAssumptions',
            index: bool = False
    ) -> None:
        """
        Save the cost assumptions to an Excel file.

        Parameters:
            filepath: Path where to save the Excel file
            sheet_name: Name of the worksheet (default is 'CostAssumptions')
            index: Whether to write row indices (default is False)
        """
        # Convert the nested dictionary to DataFrame
        df = self.cost_dict_to_df(self.cost_assumptions)

        # Save DataFrame to Excel
        df.to_excel(filepath, sheet_name=sheet_name, index=index)

    def cost_dict_to_df(
            self,
            cost_dict: dict
    ) -> pd.DataFrame:
        """
        Convert cost assumptions dictionary to DataFrame.

        Parameters:
            cost_dict: Dictionary of cost assumptions

        Returns:
            DataFrame representation of cost assumptions
        """
        # Check if it's a simple or nested dictionary
        first_key = next(iter(cost_dict), None)

        if isinstance(first_key, tuple):
            # Handle tuple-based structure
            data = []
            for keys, cost in cost_dict.items():
                main_key, *side_keys = keys
                row = {self.main_feature: main_key}

                for side_feature, side_key in zip(self.side_features, side_keys):
                    row[side_feature] = side_key

                row['cost'] = cost
                data.append(row)

            return pd.DataFrame(data)

        elif (self.side_features and
              isinstance(next(iter(cost_dict.values()), None), dict)):
            # Handle nested dictionary structure
            data = []
            for main_value, inner_dict in cost_dict.items():
                for side_value, cost in inner_dict.items():
                    row = {
                        self.main_feature: main_value,
                        self.side_features[0]: side_value,
                        'cost': cost
                    }
                    data.append(row)

            return pd.DataFrame(data)

        else:
            # Simple mapping
            return pd.DataFrame({
                self.main_feature: list(cost_dict.keys()),
                'cost': list(cost_dict.values())
            })


def save_empty_cost_assumptions(
        geo_dataset: Any,
        save_path: Union[str, Path],
        main_feature: Optional[str] = None,
        side_features: Optional[list[str]] = None,
        file_type: str = 'csv',
        **kwargs
) -> dict:
    """
    Generate and save empty cost assumptions with zero values for a geo dataset.

    This function analyzes the given dataset to detect appropriate feature columns,
    creates a CostAssumptions object with zero costs for all feature combinations,
    and saves it to the specified path in the requested format.

    Parameters:
        geo_dataset: GeoDataset object with a 'data' attribute containing a GeoDataFrame
        save_path: File path where the cost assumptions should be saved
        main_feature: Column name for the primary feature
        side_features: List containing a single column name for the secondary feature
        file_type: Output file format - one of 'json', 'csv', or 'excel'
                  (default is 'json')

    Raises:
        TypeError: If file_type is not one of the supported formats
        NoSuitableColumnsError: If no suitable columns can be detected in the dataset

    Returns:
        None: This function saves to a file and doesn't return a value
    """
    if main_feature is None or not side_features:
        # Detect main feature and side features from the GeoDataFrame
        mf, sf = detect_feature_columns(geo_dataset.data)
        main_feature = mf if main_feature is None else main_feature
        side_features = sf if not side_features else side_features

    # Generate cost assumptions with zero costs for all feature combinations
    cost_assumptions = get_zero_cost_assumptions(geo_dataset.data, main_feature,
                                                 side_features)

    # Save the cost assumptions in the appropriate format
    if file_type == 'json':
        cost_assumptions.to_json(save_path, **kwargs)
    elif file_type == 'csv':
        cost_assumptions.to_csv(save_path, **kwargs)
    elif file_type == "excel":
        cost_assumptions.to_excel(save_path, **kwargs)
    else:
        raise TypeError("Parameter file_type must be 'json', 'csv' or 'excel'!")
    return cost_assumptions.cost_assumptions


def detect_feature_columns(
        gdf: GeoDataFrame,
        max_features_per_column: int = 50
) -> tuple[str, list[str]]:
    """
    Analyze columns in a geodataframe to identify the best candidates for
    main_feature and side_features based on statistical metrics.

    Parameters:
        gdf: GeoDataFrame to analyze
        max_features_per_column: Maximum number of unique values allowed in a
        categorical column

    Returns:
        tuple of (main_feature, side_features)

    Raises:
        NoSuitableColumnsError: When no suitable columns are found for feature selection
    """
    # Filter out geometry and standard spatial columns
    ignore_columns = ['geometry', 'id', 'fid', 'gid', 'oid']
    non_spatial_cols = [col for col in gdf.columns if col not in ignore_columns]

    if not non_spatial_cols:
        msg = "No suitable feature columns found in the geodataframe"
        raise NoSuitableColumnsError(msg)

    # Analyze columns by their data characteristics
    col_stats = calculate_column_statistics(gdf, non_spatial_cols,
                                            max_features_per_column)

    # No good candidates found
    if not col_stats:
        msg = "No suitable categorical columns found in the geodataframe"
        raise NoSuitableColumnsError(msg)

    # Select main feature column (nutzart)
    main_feature = select_main_feature(col_stats)

    # Find suitable side features (bez)
    side_features = find_side_features(gdf, main_feature, col_stats)

    return main_feature, side_features


def find_side_features(
        gdf: GeoDataFrame,
        main_feature: str,
        col_stats: dict[str, dict[str, Any]]
) -> list[str]:
    """
    Find suitable side feature columns that refine the main feature.

    Parameters:
        gdf: GeoDataFrame to analyze
        main_feature: Selected main feature column name
        col_stats: dictionary with column statistics

    Returns:
        list of side feature column names
    """
    def is_column_candidate(column: str) -> bool:
        return column != main_feature and col_stats[column]['null_ratio'] <= 0.7

    # For all columns allow up to 70% nulls
    general_candidates = [col for col in col_stats if is_column_candidate(col)]

    side_features = []

    # Then process other candidates with stricter criteria
    for col in general_candidates:
        if column_shows_relationship_to_main_feature(gdf, main_feature, col):
            side_features.append(col)

    # Sort side features by information content
    def get_entropy(col):
        return col_stats.get(col, {}).get('count_entropy', 0)

    side_features.sort(key=get_entropy, reverse=True)

    return side_features if side_features else None


def column_shows_relationship_to_main_feature(
        gdf: GeoDataFrame, main_feature: str,
        side_feature: str
) -> bool:
    """
    Determine if a column adds meaningful information in relation to the main feature.

    Parameters:
        gdf: GeoDataFrame containing the data
        main_feature: Name of the main feature column
        side_feature: Name of the potential side feature column

    Returns:
        True if the column shows a meaningful relationship, False otherwise
    """
    try:
        # Create a cross-tabulation of the two columns
        crosstab = pd.crosstab(gdf[main_feature], gdf[side_feature])

        # Skip columns with too many unique values
        if len(crosstab.columns) > 100:
            return False

        # Check for non-empty cells density
        non_empty_cells = pd.DataFrame(crosstab > 0).sum().sum()
        total_cells = crosstab.size

        # If there's a good density of non-empty combinations, that's a good sign
        if non_empty_cells / total_cells > 0.05:
            return True

        # Even with many nulls, check if there's a pattern to the non-nulls
        for _, row in crosstab.iterrows():
            non_zero_vals = row[row > 0]

            # Skip rows with only one value
            if len(non_zero_vals) <= 1:
                continue

            # Check if there's diversity in the values
            if len(non_zero_vals) >= 2:
                return True

        # Special check for columns with many nulls:
        # If certain main values have side values while others don't, that's meaningful
        null_cols = [col for col in crosstab.columns if pd.isna(col) or col == '']
        if null_cols:
            non_null_main_values = 0
            for main_val, row in crosstab.iterrows():
                if row.drop(null_cols, errors='ignore').sum() > 0:
                    non_null_main_values += 1

            # If some main values have side values and others don't, that's meaningful
            if 0 < non_null_main_values < len(crosstab.index):
                return True

        return False

    except (ValueError, TypeError):
        # If analysis fails, be conservative and return False
        return False


def get_zero_cost_assumptions(
        gdf: GeoDataFrame,
        main_feature: str,
        side_features: list[str]
) -> CostAssumptions:
    """
    Generate cost assumptions with zero values for all feature combinations.

    Creates structures matching format for CostAssumptions:
    - Without side features:
    {main_feature: {val1: 0, val2: 0, ...}}
    - With side features:
    {(main_feature, side_feature1, ...): {(val1, val2, ...): 0, ...}}

    Parameters:
        gdf: GeoDataFrame with feature columns
        main_feature: Primary feature column name
        side_features: List of secondary feature column names

    Returns:
        CostAssumptions: Instacne of zero-cost assumptions
    """
    if not side_features:
        # For simple case with only main feature
        unique_values = gdf[main_feature].unique()
        cost_dict = {main_feature: dict(zip(unique_values, unique_values.size * [0]))}
    else:
        # For complex case with side features
        columns = [main_feature] + side_features
        keys = pd.MultiIndex.from_frame(gdf.loc[:, columns]).values

        def key_valid(key):
            not isinstance(key, str) and np.isnan(key)

        keys = [tuple(['' if key_valid(key) else key for key in row]) for row in keys]
        cost_dict = {tuple(columns): dict(zip(keys, len(keys) * [0]))}
    return CostAssumptions(cost_dict)


def calculate_geometry_area(
        geometries: Union[GeoSeries,]
) -> float:
    """
    Calculate the sum of areas for a collection of geometries.

    Parameters:
        geometries: Collection of geometry objects

    Returns:
        Sum of areas of all geometries with area attribute
    """
    total_area = 0
    for geom in geometries:
        if hasattr(geom, 'area'):
            total_area += geom.area
    return total_area


def calculate_column_statistics(
        gdf: GeoDataFrame,
        columns: list[str],
        max_features_per_column: int = 50
) -> dict[str, dict[str, Any]]:
    """
    Calculate statistical properties of columns for feature selection.

    Parameters:
        gdf: GeoDataFrame to analyze
        columns: list of column names to analyze
        max_features_per_column: Maximum number of unique values for a column to be
        considered categorical

    Returns:
        dictionary with column statistics

    Raises:
        ColumnAnalysisError: When column analysis fails unexpectedly
    """
    col_stats = {}

    # First pass: filter columns and calculate basic stats
    candidate_columns = []
    for col in columns:
        # Skip numeric columns with many unique values
        if pd.api.types.is_numeric_dtype(gdf[col]) and gdf[col].nunique() > 20:
            continue

        # Calculate value counts
        value_counts = gdf[col].value_counts()

        # Skip columns with too many unique values (likely not categorical)
        if len(value_counts) > max_features_per_column:
            continue

        # Calculate basic statistics
        null_ratio = gdf[col].isna().mean()
        is_good_candidate = (
                len(value_counts) > 1 and
                (len(value_counts) < len(gdf) * 0.3) and
                null_ratio < 0.2
        )

        # Calculate entropy of count distribution
        count_fractions = value_counts / len(gdf)
        count_entropy = -sum((count_fractions * np.log2(count_fractions)).dropna())

        # Store basic stats
        col_stats[col] = {
            'unique_values': len(value_counts),
            'max_count': value_counts.max() if len(value_counts) > 0 else 0,
            'min_count': value_counts.min() if len(value_counts) > 0 else 0,
            'count_entropy': count_entropy,
            'null_ratio': null_ratio,
            'is_good': is_good_candidate
        }

        candidate_columns.append(col)

    # Second pass: calculate area-based statistics only for candidates
    for col in candidate_columns:
        # Initialize area-based statistics
        area_entropy = 0
        area_by_value = None
        area_fraction = None

        try:
            # Group by column and calculate total area for each value
            area_by_value = gdf.groupby(col)['geometry'].apply(calculate_geometry_area)
            total_area = area_by_value.sum()

            if total_area > 0:
                area_fraction = area_by_value / total_area
                # Calculate entropy of area distribution
                if not area_fraction.isna().all():
                    entropies = (area_fraction * np.log2(area_fraction)).dropna()
                    area_entropy = -sum(entropies)
        except (AttributeError, ValueError, TypeError):
            # Continue with default values for area statistics
            pass

        # Update with area-based statistics
        col_stats[col].update({
            'area_by_value': area_by_value,
            'area_fraction': area_fraction,
            'area_entropy': area_entropy,
        })

    return col_stats


def calculate_entropy_score(
        column_name: str,
        col_stats: dict[str, dict[str, Any]]
) -> float:
    """
    Calculate combined entropy score for a column, weighing area entropy more heavily.

    Parameters:
        column_name: Name of the column to calculate score for
        col_stats: dictionary with column statistics

    Returns:
        Combined entropy score
    """
    stats = col_stats[column_name]
    return stats['area_entropy'] * 0.7 + stats['count_entropy'] * 0.3


def select_main_feature(
        col_stats: dict[str, dict[str, Any]]
) -> str:
    """
    Select the best main feature column based on statistics.

    Parameters:
        col_stats: dictionary with column statistics

    Returns:
        Name of the best main feature column
    """
    # Select the best main feature column
    main_candidates = [col for col, stats in col_stats.items() if stats['is_good']]

    if not main_candidates:
        # Fall back to any column if no good candidates
        main_candidates = list(col_stats.keys())

    # Sort by entropy score (higher is better)
    sorted_candidates = sorted(
        main_candidates,
        key=lambda c: calculate_entropy_score(c, col_stats),
        reverse=True
    )

    return sorted_candidates[0]
