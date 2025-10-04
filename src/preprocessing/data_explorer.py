import numpy as np
import pandas as pd
class DataEDA:
    """
    (This class remains largely the same, but now takes the cleaned data.)
    """

    def __init__(self, df):
        self.df = df
        self.df_clean = df.copy()
        self._feature_engineer_risk_metrics()

    def _feature_engineer_risk_metrics(self):
        """Calculates essential risk metrics like Loss Ratio and Claim Status."""
        if self.df_clean is not None:
            # Loss Ratio: TotalClaims / TotalPremium (handling division by zero)
            self.df_clean['LossRatio'] = np.where(
                self.df_clean['TotalPremium'] > 0,
                self.df_clean['TotalClaims'] / self.df_clean['TotalPremium'],
                0
            )
            # Claim Status
            self.df_clean['HasClaim'] = np.where(self.df_clean['TotalClaims'] > 0, 1, 0)
            print("\n- Engineered 'LossRatio' and 'HasClaim' features.")

    # ... (rest of DataEDA methods remain the same) ...
    def summarize_data_quality(self):
        """Checks data structure, missing values, and descriptive statistics."""
        print("\n--- Data Structure (df.info()) ---")
        self.df_clean.info()

        print("\n--- Missing Values Check (Should be minimal after cleaning) ---")
        missing_report = self.df_clean.isnull().sum()
        print(missing_report[missing_report > 0])

        print("\n--- Descriptive Statistics for Financials ---")
        financial_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate', 'LossRatio']
        print(self.df_clean[financial_cols].describe().T)

    def analyze_univariate(self, column):
        """Displays value counts and descriptive stats for a single column."""
        print(f"\n--- Univariate Analysis for: {column} ---")
        if pd.api.types.is_numeric_dtype(self.df_clean[column]):
            print(self.df_clean[column].describe())
        else:
            print(self.df_clean[column].value_counts(normalize=True).head(10))

    def calculate_risk_by_group(self, group_col):
        """
        Calculates and returns aggregated Total Claims, Total Premium, and Loss Ratio
        for a specified categorical column (e.g., Province, Gender).
        """
        print(f"\n--- Risk Profile by {group_col} ---")
        risk_profile = self.df_clean.groupby(group_col)[['TotalClaims', 'TotalPremium']].sum()
        risk_profile['LossRatio'] = risk_profile['TotalClaims'] / risk_profile['TotalPremium']

        # Calculate Claim Frequency (Number of claims / Total policies)
        policy_counts = self.df_clean.groupby(group_col).size().rename('PolicyCount')
        claim_counts = self.df_clean[self.df_clean['HasClaim'] == 1].groupby(group_col).size().rename('ClaimCount')

        risk_profile = risk_profile.join(policy_counts).join(claim_counts.fillna(0))
        risk_profile['ClaimFrequency'] = risk_profile['ClaimCount'] / risk_profile['PolicyCount']

        return risk_profile.sort_values(by='LossRatio', ascending=False)

    def get_clean_data(self):
        """Returns the DataFrame with engineered features."""
        return self.df_clean
