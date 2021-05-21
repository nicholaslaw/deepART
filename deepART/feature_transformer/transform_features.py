import numpy as np
import pandas as pd
import warnings
import itertools
import numbers
from typing import Union
from sklearn.preprocessing import MinMaxScaler

class FeatureTransformer:
    def __init__(self):
        self.original_cols = [] # list containing columns to remain as they are
        self.normalize_cols = dict({}) # dictionary containing columns being normalized and corresponding values are the type of normalization
        self.one_hot_cols = [] # list containing columns being one hot encoded
        self.binary_cols = dict({}) # dictionary containing columns being normalized and corresponding values are values deemed as positive
        self.minmax_cols = [] # list containing columns to be minmax scaled
        self.binning_cols = dict({}) # dictionary containing column names as keys and keys are lists with values for binning
        self.normalize_ = dict({}) # contain normalization factors as values
        self.one_hot_ = dict({}) # contain one hot column names with their corresponding unique value columns, e.g. gender_male, gender_female
        self.minmax_ = dict({}) # contain sklearn preprocessing MinMaxScaler objects
        self.fitted_ = False # boolean to check whether object has been fitted

    def normalize_denom(self, arr: np.array, mode: str = 'l2') -> float:
        """
        -----PARAMS-----
        arr: np.array
            1 dimensional numpy array
        mode: string
            Choices are 'l1', 'l2' or 'max'
        
        -----RETURNS-----
        normalized array and denominator (float)
        """
        if mode not in ['l1', 'l2', 'max']:
            raise Exception("mode must be 'l1', 'l2' or 'max'")
        if len(arr.shape) != 1:
            raise Exception("arr must be 1-dimensional")
        if mode == 'l2':
            return sum(arr ** 2) ** 0.5
        elif mode == 'l1':
            return sum(arr)
        else:
            return max(arr)

    def construct_bins(self, arr: list, bins: int) -> list:
        """
        arr: list
            list containing values for a particular column
        bins: list or int
            list would be containing values to aid binning, int would be to self-discover bins

        returns a list containing values to aid binning
        """
        _, bins = pd.cut(arr, bins=bins, labels=list(range(bins)), retbins=True)
        return bins

    def check_repeated_cols(self):
        """
        Checks whether there are repeated columns for normalizing, one hot encoding, binarizing and minmax scaling
        """
        all_combs = list(itertools.combinations([list(self.normalize_cols.keys()), self.one_hot_cols, list(self.binary_cols.keys()), self.minmax_cols, list(self.binning_cols)], 2))
        for i, j in all_combs:
            if len(set(i) & set(j)) != 0:
                overlap = list(set(i) & set(j))
                raise Exception("There are columns which appear in more than one type of preprocessing, they are: {}".format(", ".join(overlap)))
    
    def check_fitted(self):
        """
        Checks whether the object is fitted
        """
        if not self.fitted_:
            raise Exception("Call fit method first")

    def check_preproc_cols(self):
        """
        Checks whether there are any columns allocated to preprocessing
        """
        if not any([len(self.normalize_cols), len(self.one_hot_cols), len(self.binary_cols), len(self.minmax_cols), len(self.binning_cols)]):
            warnings.warn("No columns are listed for preprocessing")

    def check_cols_input(self, cols: Union[list, dict]) -> None:
        if len(cols) == 0:
            warnings.warn("There should be one or more columns listed")
        if len(set(cols)) != len(cols):
            raise Exception("There is a column repeated")

    def set_default_col_dic(self, cols: Union[list, dict], default) -> dict:
        """
        -----PARAMS-----
        cols: list or dictionary
            contains column names as keys and default values for preprocessing as values
        """
        if isinstance(cols, list):
            cols = {c: default for c in cols}
        return cols

    def set_normalizer(self, cols: Union[list, dict]) -> None:
        """
        -----PARAMS-----
        cols: list or dictionary
            list of column names, dictionary would be columns as keys and values would be the norm parameter in sklearn.preprocessing.Normalizer
            norm would be either 'l1', 'l2' or 'max'
        
        -----RETURNS-----
        None
        Sets attribute normalize_cols to cols
        """
        cols = self.set_default_col_dic(cols, 'l2') # if columns are listed as a list, put default value for normalization
        self.check_cols_input(cols) # check whether there is one or more columns listed
        self.normalize_cols = cols
        self.check_repeated_cols() # check for columns being stated in more than one form of preprocessing, e.g. normalized and one hot
    
    def set_one_hot(self, cols: list) -> None:
        """
        -----PARAMS-----
        cols: list
            list of column names for one hot encoding
        
        -----RETURNS-----
        None
        Sets attribute one_hot_cols to cols
        """
        self.check_cols_input(cols) # check whether there is one or more columns listed
        self.one_hot_cols = cols
        self.check_repeated_cols() # check for columns being stated in more than one form of preprocessing, e.g. normalized and one hot

    def set_binary(self, cols: Union[list, dict]) -> None:
        """
        -----PARAMS-----
        cols: list or dict
            list of column names for binary encoding, dictionary would be column names as keys and values would be their corresponding values to be encoded as positive
        
        -----RETURNS-----
        None
        Sets attribute binary_cols to cols
        """
        cols = self.set_default_col_dic(cols, None) # if columns are listed as a list, put default value for binary encoding, cols will be a dictionary with column name as key and corresponding value is value as positive which is 1
        self.check_cols_input(cols) # check whether there is one or more columns listed
        self.binary_cols = cols
        self.check_repeated_cols() # check for columns being stated in more than one form of preprocessing, e.g. normalized and one hot

    def set_minmax(self, cols: list) -> None:
        """
        -----PARAMS-----
        cols: list
            list of column names for minmax scaling
        
        -----RETURNS-----
        None
        Sets attribute one_hot_cols to cols
        """
        self.check_cols_input(cols) # check whether there is one or more columns listed
        self.minmax_cols = cols
        self.check_repeated_cols() # check for columns being stated in more than one form of preprocessing, e.g. normalized and one hot

    def set_binning(self, cols: dict) -> None:
        """
        -----PARAMS-----
        cols: list or dict
            list of column names for binning, dictionary would be column names as keys and values would be their corresponding values to be encoded as positive
        
        -----RETURNS-----
        None
        Sets attribute binning_cols to cols
        """
        self.check_cols_input(cols)
        for c in cols.keys():
            val = cols[c]
            if not isinstance(val, int):
                raise TypeError("Values in Dictionary should be integers")
            if val < 2:
                raise ValueError("Values should be at least 2")
        self.binning_cols = cols
        self.check_repeated_cols() # check for columns being stated in more than one form of preprocessing, e.g. normalized and one hot

    def fit(self, df: pd.DataFrame) -> None:
        """
        -----PARAMS-----
        df: pandas dataframe
            dataframe for preprocessing
        
        -----RETURNS-----
        None
        Obtain relevant fitted sklearn preprocessing objects
        """
        self.check_preproc_cols() # check whether there are any columns listed for preprocessing

        # One Hot
        for col in self.one_hot_cols:
            unique_vals = df[col].unique()
            if len(unique_vals) == 2:
                raise Exception("Column {} should be assigned to binary columns instead".format(col))
            self.one_hot_[col] = [col+"_{}".format(i) for i in df[col].unique()]

        # Binning and then One Hot
        for col in self.binning_cols.keys():
            bins = self.binning_cols[col]
            _, bin_vals = pd.cut(df[col], bins=bins, labels=list(range(bins)), retbins=True) # if want x bins, bin_vals would have x+1 elements
            bin_vals = list(bin_vals)
            mapping = {idx: None for idx in range(bins)}
            self.one_hot_cols.append(col) # we want to one hot encode after binning
            hot = []
            for idx, val in enumerate(bin_vals):
                if idx != (len(bin_vals) - 1):
                    template = "{}-{}".format(val, bin_vals[idx+1]) # accomodate pd.get_dummies
                    encoding = col + "_" + template
                    mapping[idx] = template
                    hot.append(encoding)
            self.one_hot_[col] = hot
            self.binning_cols[col] = mapping

        # Normalize
        for col, mode in self.normalize_cols.items():
            if df[col].min() < 0:
                raise Exception("Should not normalize {} as there are negative values present".format(col))
            self.normalize_[col] = self.normalize_denom(df[col].values, mode)

        # Binary Encoding
        for col in self.binary_cols.keys():
            pos_val = self.binary_cols[col]
            unique_val = df[col].unique()
            if len(unique_val) > 2:
                raise Exception("Column {} has more than 2 unique values and it is chosen for binary encoding".format(col))
            if not self.fitted_:
                if pos_val:
                    if pos_val not in unique_val:
                        raise Exception("User input for positive value {} of column {} is not valid, try {}?".format(pos_val, col, unique_val[0]))
                    self.binary_cols[col] = {pos_val: 1, list(filter(lambda i: i!=pos_val, unique_val))[0]: 0}
                    continue
            self.binary_cols[col] = {unique_val[0]: 0, unique_val[1]: 1}

        # Min Max Scaling
        for col in self.minmax_cols:
            minmax_scaler = MinMaxScaler(feature_range=(0, 1))
            minmax_scaler.fit(df[col].values.reshape(-1,1))
            self.minmax_[col] = minmax_scaler

        self.original_cols = list(set(df.columns) - set(self.one_hot_cols) - set(self.normalize_cols.keys()) - set(self.binary_cols.keys()) - set(self.minmax_cols))
        self.fitted_ = True
    
    def transform(self, df: pd.DataFrame, return_orig: bool = True) -> pd.DataFrame:
        """
        -----PARAMS-----
        df: pandas dataframe
            dataframe for preprocessing
        return_orig: boolean
            True if wants the transformed dataframe to be concatenated with the rest of the columns which did not go through preprocessing
            For one hot, the original column will be removed from original dataframe and swopped with the corresponding value columns
        
        -----RETURNS-----
        returns transformed dataframe
        """
        self.check_fitted() # check whether object is fitted
        self.check_preproc_cols() # check whether there are any columns listed for preprocessing
        one_hot_df = pd.DataFrame()

        # Binning
        for col in self.binning_cols.keys():
            mapping = self.binning_cols[col]
            bins = len(mapping)
            df[col] = pd.cut(df[col], bins=bins, labels=list(range(bins)))
            df[col] = df[col].apply(lambda i: mapping[i])

        # One Hot
        if len(self.one_hot_cols) != 0:
            for col in self.one_hot_cols:
                df[col] = df[col].astype("category")
            one_hot_df = pd.get_dummies(df[self.one_hot_cols])
            one_hot_check = []
            for one_hot_val in self.one_hot_.values():
                one_hot_check.extend(one_hot_val)
            if len(one_hot_df.columns) != len(one_hot_check):
                raise Exception("one_hot_cols attribute not the same as one_hot_ dictionary values")
            del one_hot_check
        
        # Normalize
        normalize_df = dict({})
        for col in self.normalize_cols:
            normalize_df[col] = df[col].values / self.normalize_[col]
        normalize_df = pd.DataFrame(normalize_df)

        # Binary Encoding
        binary_df = dict({})
        for col, mapping in self.binary_cols.items():
            binary_df[col] = df[col].apply(lambda i: mapping[i]) # map positive values to 1 and negative ones to 0
        binary_df = pd.DataFrame(binary_df)

        # MinMax Scaling
        minmax_df = dict({})
        for col in self.minmax_cols:
            minmax_df[col] = self.minmax_[col].transform(df[col].values.reshape(-1,1)).flatten()
        minmax_df = pd.DataFrame(minmax_df)

        if return_orig and len(self.original_cols) != 0:
            return pd.concat([one_hot_df, normalize_df, binary_df, minmax_df, df[self.original_cols]], axis=1)
        else:
            return pd.concat([one_hot_df, normalize_df, binary_df, minmax_df], axis=1)

    def fit_transform(self, df: pd.DataFrame, return_orig: bool = True) -> pd.DataFrame:
        """
        -----PARAMS-----
        df: pandas dataframe
            dataframe for preprocessing
        
        -----RETURNS-----
        returns transformed dataframe
        """
        self.fit(df)
        return self.transform(df, return_orig)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        -----PARAMS-----
        df: pandas dataframe
            dataframe for preprocessing
        
        -----RETURNS-----
        restores art templates to original form
        """
        self.check_fitted() # check whether object is fitted
        self.check_preproc_cols() # check whether there are any columns listed for preprocessing

        # One Hot
        for col, col_val in self.one_hot_.items():
            mapping = {idx: c[len(col)+1:] for idx, c in enumerate(col_val)} # extract names of one hot values and give them indices
            temp_ = df[col_val].values.argmax(axis=1) # extract all the columns corresponding to this one hot variable and then argmax to find activated value
            temp_ = [mapping[idx] for idx in temp_] # extract names of activated one hot values
            df = df.drop(col_val, axis=1)
            df[col] = temp_

        # Normalize
        for col in self.normalize_cols.keys():
            df[col] = df[col].values * self.normalize_[col] # multiply back the normalization factor

        # Binary Encoding
        for col, mapping in self.binary_cols.items():
            reverse_mapping = {idx: val for val, idx in mapping.items()}
            df[col] = df[col].apply(lambda i: reverse_mapping[1] if i >= 0.5 else reverse_mapping[0]) # positive values will be those above 0.5
        
        # Min Max Scaling
        for col in self.minmax_cols:
            df[col] = self.minmax_[col].inverse_transform(df[col].values.reshape(-1,1)).flatten()
        return df