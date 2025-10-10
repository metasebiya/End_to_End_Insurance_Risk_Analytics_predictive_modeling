# Import the necessary classes from their respective modules
from src.ingestion.data_loader import DataLoader
from data_cleaner import DataCleaner
from data_explorer import DataEDA
from data_visualizer import DataVisualizer
from src.reports.business_reporter import BusinessReporter
from ABTester import ABTester  # Renamed import to lowercase for consistency


class AnalysisPipeline:
    """
    Orchestrates the entire data process: Load -> Clean -> EDA -> Visualize -> Test.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.visualizer = None
        self.data_explorer = None

    def _interpret_ab_results(self, results, alpha=0.05):
        """Interprets and prints the statistical outcomes in business terms."""
        print("\n--- A/B Hypothesis Test Interpretations ---")

        for key, res in results.items():
            # Extract p-value, handling different test result structures
            p_value = res.get('p_value')

            if p_value is not None:
                decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

                # Determine the metric and grouping for business context
                if 'freq' in key:
                    metric = "Claim Frequency (Prop.)"
                    group_type = res.get('group') or res.get('A') + ' vs ' + res.get('B')
                elif 'severity' in key:
                    metric = "Claim Severity (Mean)"
                    group_type = res.get('group') or res.get('A') + ' vs ' + res.get('B')
                elif 'margin' in key:
                    metric = "Margin (Mean Profit)"
                    group_type = res.get('group') or res.get('A') + ' vs ' + res.get('B')
                else:
                    metric = "Metric"
                    group_type = res.get('group', 'Feature')

                # IMPROVEMENT: Use scientific notation for p-values near zero
                if p_value < 0.0001:
                    p_value_str = f"{p_value:.2e} (Scientific)"
                else:
                    p_value_str = f"{p_value:.4f}"

                print(f"[{key.upper()}] Test on {group_type} vs {metric}:")
                print(f"  P-Value: {p_value_str}. Decision: {decision}.")

                if decision == "Reject H₀":
                    print(
                        f"  --> ACTION: {group_type} is a statistically significant risk/profit driver. Consider segmenting or adjusting pricing/underwriting.")
                else:
                    print(
                        f"  --> NO ACTION: We lack statistical evidence that {group_type} significantly impacts {metric}. Current segmentation is likely adequate for this feature.")
            else:
                print(f"[{key.upper()}] Warning: P-value not found in result structure for test.")

    def run_pipeline(self):
        print("======================================================")
        print("  ALPHA CARE INSURANCE ANALYTICS PIPELINE INITIATED")
        print("======================================================")

        # 1. DATA LOADING PHASE
        data_loader = DataLoader(self.file_path)
        raw_df = data_loader.load_data()
        self.df = data_loader.preprocess_initial()

        if self.df is None:
            print("Pipeline aborted due to data loading error.")
            return

        # 2. DATA CLEANING PHASE
        data_cleaner = DataCleaner(self.df)
        self.df = data_cleaner.get_cleaned_data()

        # 3. EDA & FEATURE ENGINEERING PHASE
        self.data_explorer = DataEDA(self.df)

        # Data Summarization (check post-cleaning quality)
        self.data_explorer.summarize_data_quality()

        # Calculate Risk Profiles (needed for visualization and reporting)
        risk_by_province = self.data_explorer.calculate_risk_by_group('Province')
        risk_by_gender = self.data_explorer.calculate_risk_by_group('Gender')

        # Final DataFrame with engineered features
        final_df = self.data_explorer.get_clean_data()

        # 4. DATA VISUALIZATION PHASE
        self.visualizer = DataVisualizer(final_df)

        # Generate the four key plots
        self.visualizer.plot_financial_distributions()
        self.visualizer.plot_geographic_risk_profile(risk_by_province)
        self.visualizer.plot_gender_risk_breakdown(risk_by_gender)
        self.visualizer.plot_premium_vs_value()

        # 5. A/B Testing
        print("\n--- Running A/B Hypothesis Tests ---")
        self.ab_tester = ABTester(final_df)
        results = self.ab_tester.run_all()

        # New: Interpretation of Results
        self._interpret_ab_results(results)

        # 6. Business Reporting
        # The BusinessReporter will use these results to generate the final Markdown audit
        reporter = BusinessReporter(results)
        report_path = reporter.generate_markdown()
        print(f"Report saved: {report_path}")

        print("\n======================================================")
        print("  PIPELINE COMPLETE. Key insights gained for A/B testing.")
        print("======================================================")


# --------------------------------------------------------------------------------------------------
# --- MAIN EXECUTION ---
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define a conceptual file path
    FILE_PATH = '../../data/raw/MachineLearningRating_v3.txt'
    # Instantiate and run the pipeline
    pipeline = AnalysisPipeline(FILE_PATH)
    pipeline.run_pipeline()