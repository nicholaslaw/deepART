import unittest
import numpy as np
import pandas as pd
from deepART import FeatureTransformer

class featuretransformerTest(unittest.TestCase):

    def test_featuretransformer(self):
        # Dummy Dataset
        raw_df = pd.DataFrame({"CustomerID": np.arange(1000),
                            "gender": np.random.choice(["male", "female"], replace=True, size=1000),
                            "SeniorCitizen": np.random.choice(["yes", "no"], replace=True, size=1000),
                            "Partner": np.random.choice(["yes", "no"], replace=True, size=1000),
                            "Tenure": np.random.randint(low=0, high=73, size=1000),
                            "Monthly_Charges": np.random.uniform(low=30, high=100, size=1000),
                            "PhoneService": np.random.choice(["yes", "no"], replace=True, size=1000),
                            "InternetService": np.random.choice(["DSL", "fiber optic", "no internet service"], replace=True, size=1000),
                            "Contract": np.random.choice(["Month-to-month", "One year", "Two years"], replace=True, size=1000),
                            "PaymentMethod": np.random.choice(["Electronic Check", "Mailed check", "Bank Transfer (automatic)"], replace=True, size=1000)})

        # Initialize Feature Transformer
        feat_transformer = FeatureTransformer()

        # Set Binary Columns
        feat_transformer.set_binary({"gender": "male", "PhoneService": "yes", "Partner": "yes", "SeniorCitizen": "yes"})

        # Set Normalize Columns
        feat_transformer.set_normalizer({"Monthly_Charges": "max"})

        # Set One Hot Columns
        one_hot_cols = [i for i in raw_df.columns if i not in ["gender", "PhoneService", "Partner", "SeniorCitizen", "Monthly_Charges", "Tenure", "CustomerID"]]
        feat_transformer.set_one_hot(one_hot_cols)

        # Set Binning Columns
        binning_cols = {"Tenure": 6}
        feat_transformer.set_binning(binning_cols)

        # Preprocess Data
        feat_transformer.fit(raw_df.copy())
        self.assertTrue(feat_transformer.fitted_)
        preprocessed_df = feat_transformer.transform(raw_df.copy())
        check = True
        for col in preprocessed_df.columns:
            if col not in feat_transformer.original_cols:
                min_ = preprocessed_df[col].min()
                max_ = preprocessed_df[col].max()
                check = check and (0<=min_<=1 and 0<=max_<=1)
        self.assertTrue(check)

        # Check Binning Functionality
        self.assertTrue(len(set(feat_transformer.one_hot_cols) & set(list(feat_transformer.binning_cols.keys()))) > 0)
        self.assertTrue(len(set(list(feat_transformer.one_hot_.keys())) & set(list(feat_transformer.binning_cols.keys()))) > 0)

        # Generate Dummy Templates
        templates = preprocessed_df.drop("CustomerID", axis=1).sample(n=10, replace=False, random_state=0)
        
        # Inverse Transform
        restored_templates = feat_transformer.inverse_transform(templates)
        all_cols = list(raw_df.columns)
        check = True
        for col in restored_templates.columns:
            if col in binning_cols:
                continue
            check = check and col in all_cols
            default_min = raw_df[col].min()
            default_max = raw_df[col].max()
            check = check and (default_min<=restored_templates[col].min()<=default_max and default_min<=restored_templates[col].max()<=default_max)
        self.assertTrue(check)

if __name__ == '__main__':
    unittest.main()