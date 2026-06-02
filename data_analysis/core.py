import pandas as pd
import numpy as np

from google.colab import files

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.display import HTML


class PlottingMethods:
    """
    Modular plotting utility class.
    Generates Plotly charts and returns HTML-rendered figures.
    """

    @staticmethod
    def bar_chart(df, column):
        counts = df[column].value_counts()

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            labels={"x": column, "y": "Count"},
            title=f"Bar Chart - {column}"
        )

        return HTML(fig.to_html())

    @staticmethod
    def pie_chart(df, column):
        counts = df[column].value_counts()

        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title=f"Pie Chart - {column}"
        )

        return HTML(fig.to_html())

    @staticmethod
    def histogram(df, column):
        fig = px.histogram(
            df,
            x=column,
            title=f"Histogram - {column}"
        )

        return HTML(fig.to_html())


class DataInspector:
    """
    Data Sanitization & Exploration Engine

    Features:
    - Data Upload
    - Missing Value Handling
    - Duplicate Removal
    - Outlier Detection
    - Feature Scaling
    - Categorical Encoding
    - Visualization
    - Association Analysis
    """

    def __init__(self):
        self.df = None

    ##################################################
    # DATA INGESTION
    ##################################################

    def upload_data(self):
        """
        Upload CSV file from Google Colab.
        Automatically sanitizes common garbage strings.
        """

        uploaded = files.upload()

        filename = list(uploaded.keys())[0]

        self.df = pd.read_csv(filename)

        garbage_strings = [
            '?',
            'n/a',
            'N/A',
            'NULL',
            'null',
            ' '
        ]

        self.df.replace(
            garbage_strings,
            np.nan,
            inplace=True
        )

        self.auto_type_correction()

        print("Dataset loaded successfully.")

        return self.df

    def auto_type_correction(self):
        """
        Attempts numeric conversion on columns.
        Conversion is applied if resulting column
        is not entirely null.
        """

        for col in self.df.columns:

            converted = pd.to_numeric(
                self.df[col],
                errors='coerce'
            )

            if converted.notna().sum() > 0:
                self.df[col] = converted

    ##################################################
    # DATA SUMMARY
    ##################################################

    def summary(self):
        """
        Displays dataset structure information.
        """

        if self.df is None:
            print("No dataset loaded.")
            return

        print("=" * 50)
        print(f"Rows: {self.df.shape[0]}")
        print(f"Columns: {self.df.shape[1]}")

        print("\nFirst 20 Rows")
        display(self.df.head(20))

        numeric_cols = self.df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_cols = self.df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        print("\nNumeric Columns")
        print(numeric_cols)

        print("\nCategorical Columns")
        print(categorical_cols)

    ##################################################
    # MISSING VALUES
    ##################################################

    def handle_missing_values(
            self,
            strategy="mean",
            constant_value=None):
        """
        Strategies:
        mean
        median
        mode
        constant
        """

        if self.df is None:
            return

        for col in self.df.columns:

            if self.df[col].isna().sum() == 0:
                continue

            if strategy == "mean":

                if pd.api.types.is_numeric_dtype(
                        self.df[col]):
                    self.df[col].fillna(
                        self.df[col].mean(),
                        inplace=True
                    )

            elif strategy == "median":

                if pd.api.types.is_numeric_dtype(
                        self.df[col]):
                    self.df[col].fillna(
                        self.df[col].median(),
                        inplace=True
                    )

            elif strategy == "mode":

                self.df[col].fillna(
                    self.df[col].mode()[0],
                    inplace=True
                )

            elif strategy == "constant":

                self.df[col].fillna(
                    constant_value,
                    inplace=True
                )

    ##################################################
    # DUPLICATES
    ##################################################

    def remove_duplicates(self):
        """
        Removes duplicate rows.
        """

        before = len(self.df)

        self.df.drop_duplicates(
            inplace=True
        )

        after = len(self.df)

        print(
            f"Removed {before-after} duplicate rows."
        )

    ##################################################
    # OUTLIERS
    ##################################################

    def handle_outliers(
            self,
            column,
            remove=False):
        """
        IQR-based outlier detection.
        """

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        mask = (
            (self.df[column] < lower) |
            (self.df[column] > upper)
        )

        print(
            f"Outliers detected: {mask.sum()}"
        )

        if remove:
            self.df = self.df[~mask]

    ##################################################
    # ROW/COLUMN DELETION
    ##################################################

    def delete_rows(self):
        """
        Delete rows by comma-separated indices.
        """

        rows = input(
            "Enter row indices: "
        )

        rows = [
            int(i.strip())
            for i in rows.split(",")
        ]

        self.df.drop(
            rows,
            inplace=True
        )

    def delete_columns(self):
        """
        Delete columns by comma-separated names.
        """

        cols = input(
            "Enter columns: "
        )

        cols = [
            c.strip()
            for c in cols.split(",")
        ]

        self.df.drop(
            columns=cols,
            inplace=True
        )

    ##################################################
    # NUMERIC NORMALIZATION
    ##################################################

    def extract_normalized_numeric_data(
            self,
            method="minmax"):
        """
        Scaling methods:
        minmax
        standard
        robust
        """

        numeric = self.df.select_dtypes(
            include=np.number
        )

        if numeric.empty:
            return pd.DataFrame()

        if method == "minmax":
            scaler = MinMaxScaler()

        elif method == "standard":
            scaler = StandardScaler()

        elif method == "robust":
            scaler = RobustScaler()

        else:
            raise ValueError(
                "Invalid scaling method."
            )

        scaled = scaler.fit_transform(
            numeric
        )

        return pd.DataFrame(
            scaled,
            columns=numeric.columns
        )

    ##################################################
    # CATEGORICAL ENCODING
    ##################################################

    def extract_normalized_categorical_data(
            self,
            method="onehot"):
        """
        Encoding methods:
        onehot
        ordinal
        uniform
        """

        categorical = self.df.select_dtypes(
            exclude=np.number
        )

        if categorical.empty:
            return pd.DataFrame()

        if method == "onehot":

            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            )

            encoded = encoder.fit_transform(
                categorical
            )

            return pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out()
            )

        elif method == "ordinal":

            encoder = OrdinalEncoder()

            encoded = encoder.fit_transform(
                categorical
            )

            return pd.DataFrame(
                encoded,
                columns=categorical.columns
            )

        elif method == "uniform":

            encoder = OrdinalEncoder()

            temp = encoder.fit_transform(
                categorical
            )

            scaler = MinMaxScaler()

            encoded = scaler.fit_transform(
                temp
            )

            return pd.DataFrame(
                encoded,
                columns=categorical.columns
            )

    ##################################################
    # MERGED DATASET
    ##################################################

    def get_combined_dataset(
            self,
            num_method="minmax",
            cat_method="onehot"):
        """
        Returns normalized numeric +
        encoded categorical data.
        """

        num = self.extract_normalized_numeric_data(
            num_method
        )

        cat = self.extract_normalized_categorical_data(
            cat_method
        )

        return pd.concat(
            [num, cat],
            axis=1
        )

    ##################################################
    # UNIVARIATE VISUALIZATION
    ##################################################

    def univariate_subplot(self, column):

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Violin Plot",
                "Scatter Plot",
                "Histogram"
            ]
        )

        fig.add_trace(
            go.Violin(
                y=self.df[column]
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df[column],
                mode='markers'
            ),
            row=1,
            col=2
        )

        fig.add_trace(
            go.Histogram(
                x=self.df[column]
            ),
            row=1,
            col=3
        )

        fig.show()

    ##################################################
    # RELATIONSHIP PLOTS
    ##################################################

    def plot_relationship(
            self,
            col1,
            col2):

        num1 = pd.api.types.is_numeric_dtype(
            self.df[col1]
        )

        num2 = pd.api.types.is_numeric_dtype(
            self.df[col2]
        )

        if num1 and num2:

            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                trendline="ols"
            )

        elif not num1 and num2:

            fig = px.box(
                self.df,
                x=col1,
                y=col2,
                points="all"
            )

        elif num1 and not num2:

            fig = px.box(
                self.df,
                x=col2,
                y=col1,
                points="all"
            )

        else:

            cross = pd.crosstab(
                self.df[col1],
                self.df[col2]
            )

            fig = px.bar(
                cross,
                barmode="group"
            )

        fig.show()

    ##################################################
    # ASSOCIATION HEATMAP
    ##################################################

    def plot_all_associations_heatmap(self):

        temp = self.df.copy()

        for col in temp.columns:

            if temp[col].dtype == object:

                temp[col] = pd.factorize(
                    temp[col]
                )[0]

        corr = temp.corr(
            numeric_only=False
        )

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Association Heatmap"
        )

        fig.show()
