# --- 2. Data Cleaning Class ---

class DataCleaner:
    """
    Handles data quality issues: missing values, incorrect entries,
    and outlier management.
    """

    def __init__(self, df):
        self.df = df.copy()  # Work on a copy

    def handle_missing_values(self):
        """Imputes or removes missing values based on column type and importance."""

        print("\n--- Cleaning Missing Values ---")

        # Strategy 1: Impute categorical/geographic features with 'Unknown'
        categorical_impute_cols = ['Province', 'PostalCode', 'MaritalStatus', 'Gender', 'LegalType']
        for col in categorical_impute_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('UNKNOWN')
        print(f"- Imputed categorical data in {len(categorical_impute_cols)} columns with 'UNKNOWN'.")

        # Strategy 2: Impute numerical features (e.g., car specs) with median/mean or 0
        numerical_zero_impute_cols = ['Kilowatts', 'Cylinders', 'Cubiccapacity']
        for col in numerical_zero_impute_cols:
            if col in self.df.columns:
                # It's safer to assume a missing engine spec means a default/low value or 0
                self.df[col] = self.df[col].fillna(0)
        print(f"- Imputed car specification data with 0.")

        # Strategy 3: Critical financial variables (TotalPremium, TotalClaims, SumInsured)
        # We must ensure these are not missing. A missing premium/claim is critical.
        # If total premium is NaN, we fill it with 0 assuming no payment/policy validity.
        financial_cols = ['TotalPremium', 'TotalClaims', 'SumInsured']
        for col in financial_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # Strategy 4: Drop rows if core identifiers are missing (e.g., PolicyID)
        if 'PolicyID' in self.df.columns:
            self.df.dropna(subset=['PolicyID'], inplace=True)
            print("- Dropped rows with missing PolicyID.")

        return self.df

    def handle_outliers(self, column, method='IQR', threshold=1.5):
        """
        Handles outliers in a numerical column, often by capping (Winsorizing)
        for claims data to avoid skewing summary statistics or modeling.
        """
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            print(f"Warning: Column {column} not found or is not numeric.")
            return self.df

        # Filter to non-zero values for cleaner outlier detection on severity
        data = self.df[self.df[column] > 0][column]

        if data.empty:
            print(f"No non-zero values in {column}. Skipping outlier capping.")
            return self.df

        if method == 'IQR':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + threshold * IQR

            # Cap the values
            original_count = (self.df[column] > upper_bound).sum()
            self.df[column] = np.where(
                self.df[column] > upper_bound,
                upper_bound,
                self.df[column]
            )
            print(f"- Capped {original_count} outliers in {column} above {upper_bound:.2f} using IQR method.")

        return self.df

    def refine_features(self):
        """Standardizes and cleans specific features."""

        print("\n--- Refining Features ---")

        # Clean Gender column (if necessary, assuming values like M/m, F/f)
        if 'Gender' in self.df.columns:
            self.df['Gender'] = self.df['Gender'].str.upper().str.strip()
            # Conceptual: Map non-standard entries if required (e.g., 'U' to 'UNKNOWN')

        # Clean MakeModel and Bodytype (standardize text/remove noise)
        for col in ['MakeModel', 'Bodytype']:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.lower().str.strip()

        return self.df

    def get_cleaned_data(self):
        """Runs the full cleaning pipeline."""
        df_cleaned = self.handle_missing_values()

        # Apply outlier management to key financial severity variables
        df_cleaned = self.handle_outliers('TotalClaims', threshold=3.0)  # Use a higher threshold for claims
        df_cleaned = self.handle_outliers('CustomValueEstimate', threshold=3.0)

        df_cleaned = self.refine_features()
        print("\nCleaning pipeline complete.")
        return df_cleaned
