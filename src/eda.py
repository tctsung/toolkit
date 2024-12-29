import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import src.utils as utils  # assuming script path is tidytool/
from collections import Counter
from pprint import pprint, pformat
from typing import Literal


class Data:
    def __init__(self, file, logging_level="info", display=True):
        """
        TODO: check data quality, understand the data
        Args:
            file (str/pd.DF): The file path/DF to be analyzed. (only support csv for now)
            logging_level (str, optional): The logging level to be used. Defaults to "info".
        """
        # settings:
        utils.set_loggings(level=logging_level, func_name="EDA.Data")

        # load data:
        if isinstance(file, pd.DataFrame):
            self.before = file
        else:
            self.filepath = file
            self.load()
        # display basic info of data
        if display:
            self.info()

    def load(self):
        """TODO: load data from file, will skip bad lines if needed"""
        try:
            self.before = pd.read_csv(self.filepath)
        except:
            logging.warning(
                "Input file ParserError. Bad lines are skipped & saved in .bad_lines"
            )
            bad_lines = []  # buffer to save bad lines

            def bad_line_handler(bad_line):
                bad_lines.append(bad_line)  # save the bad line content
                return None

            # save data as attr
            self.before = pd.read_csv(
                self.filepath, on_bad_lines=bad_line_handler, engine="python"
            )
            self.bad_lines = bad_lines
        # create buffer for processed data
        self.after = self.before.copy()

    def info(
        self, status: Literal["before", "after"] = "before", head=False, max_unique=3
    ):
        """
        TODO: Some summary info of data, including data types, NA count, unique values, etc.
        Args:
            data (Literal['before', 'after']): State of data to be summarized
            head (bool, optional): If True, display first few rows of data
            max_unique (int, optional): Max no. of unique values to display for each feature
        Attrs:
            ov (pd.DataFrame): Each row represents a feature info
                - dtype: Data type of each feature.
                - NA_count: Proportion of missing values in each feature.
                - n_unique: Number of unique values in each feature.
                - examples: Examples of unique values in each feature.
        """
        df = self.before if status == "before" else self.after  # select data
        pd.set_option(
            "display.max_rows", max(df.shape[1], 10)
        )  # to display all features

        top_unique = lambda x, n=max_unique: x.unique()[:n]  # get top n unique values

        info = df.apply(
            lambda x: (x.dtype, x.isna().mean(), x.nunique(), top_unique(x)), axis=0
        ).T
        info.columns = ["dtype", "NA_count", "n_unique", "examples"]
        info_str = pformat(info)

        # collect df.head()
        head_info = df.head() if head else "Skipped"

        # display basic info of data
        logging.critical(
            f"""Status: {status}\n
Table Dimension: {df.shape}\n
Data types summary:\n{df.dtypes.value_counts()}\n
Head of data:\n{head_info}\n
Data info (Data.ov):\n{info_str}
"""
        )
        self.ov = info

    def str_process(self, case: Literal["raw", "upper", "lower"] = "raw"):
        """
        TODO: string processing, including space stripping, case-changing
        """

        def clean_str(x):
            # x: pandas series
            return (
                x.replace(r"['\"]", "", regex=True)
                .str.strip()
                .replace(r"\s+", " ", regex=True)
            )

        # strip space:
        self.after = self.after.apply(
            lambda x: (clean_str(x) if x.dtype == "object" else x)
        )
        # change case:
        if case == "upper":
            self.after = self.after.apply(
                lambda x: x.str.upper() if x.dtype == "object" else x
            )
        elif case == "lower":
            self.after = self.after.apply(
                lambda x: x.str.lower() if x.dtype == "object" else x
            )

    def clean_header(self, keep_space=False):
        """Strip space & single/double quotes for column names."""
        ori_colnames = self.before.columns
        colnames = (
            self.before.columns.str.replace(r"['\"]", "", regex=True)  # rm quotes
            .str.replace(r"\s+", " ", regex=True)  # long space to single space
            .str.strip()  # strip space
        )
        if not keep_space:
            colnames = colnames.str.replace(" ", "_")  # turn space to underscore
        self.after.columns = colnames
        name_log = ""  # buffer to changed names
        cnt = 0  # count changed names
        for ori, new in zip(ori_colnames, colnames):  # display changed names
            if ori != new:
                cnt += 1
                name_log += f"{ori} -> {new}\n"
        logging.info(f"{cnt} colnames were updated:\n{name_log}")

    def replace_with_na(self, na_vals=[" ", "", "?"]):
        """TODO: Replace na_candidates with pd.NA"""
        # merge list into regex pattern:
        na_vals.extend([np.nan, None])  # standardize na values

        # get colnames of each gp:
        float_cols = self.after.select_dtypes(include=["float"]).columns
        date_cols = self.after.select_dtypes(include=["datetime"]).columns
        other_cols = self.after.select_dtypes(exclude=["float", "datetime"]).columns

        # use np.nan for float:
        self.after[float_cols] = self.after[float_cols].replace(na_vals, np.nan)

        # use pd.NaT for datetime:
        self.after[date_cols] = self.after[date_cols].replace(na_vals, pd.NaT)

        # use pd.NA for other types:
        self.after[other_cols] = self.after[other_cols].replace(na_vals, pd.NA)

    def clean(
        self, na_vals=[" ", "", "?", np.nan, None], case="raw", header_keep_space=False
    ):
        self.clean_header(keep_space=header_keep_space)
        self.str_process(case=case)
        self.replace_with_na(na_vals=na_vals)


# class Vis
