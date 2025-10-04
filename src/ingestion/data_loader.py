import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# --- 1. Data Loading Class ---

class DataLoader:
    """
    Handles loading the data and performing initial data quality checks
    and essential type conversions.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads the CSV file into a Pandas DataFrame."""
        print(f"Loading data from: {self.file_path}")
        try:
            # Conceptual: Assuming the historical data is a CSV file
            self.df = pd.read_csv(self.file_path,  sep='|')
            print("Data loaded successfully.")
            return self.df
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return None

    def preprocess_initial(self):
        """Performs initial type conversions required for cleaning."""
        if self.df is not None:
            # Convert date columns to datetime objects
            if 'TransactionMonth' in self.df.columns:
                # Assuming the format is simple enough for direct conversion
                self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
                print("- Converted 'TransactionMonth' to datetime.")

            # Ensure financial columns are numeric
            financial_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
            for col in financial_cols:
                if col in self.df.columns:
                    # Coerce non-numeric values to NaN
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            return self.df
        return None