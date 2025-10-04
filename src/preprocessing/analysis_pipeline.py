# Import the necessary classes from their respective modules
from src.ingestion.data_loader import DataLoader
from data_cleaner import DataCleaner
from data_explorer import DataEDA
from data_visualizer import DataVisualizer
class AnalysisPipeline:
    """
    Orchestrates the entire data process: Load -> Clean -> EDA -> Visualize.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.visualizer = None
        self.data_explorer = None

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

        # Calculate Risk Profiles (needed for visualization)
        risk_by_province = self.data_explorer.calculate_risk_by_group('Province')
        risk_by_gender = self.data_explorer.calculate_risk_by_group('Gender')

        # Final DataFrame with engineered features
        final_df = self.data_explorer.get_clean_data()

        # 4. DATA VISUALIZATION PHASE
        self.visualizer = DataVisualizer(final_df)

        # Generate the three key plots
        self.visualizer.plot_financial_distributions()
        self.visualizer.plot_geographic_risk_profile(risk_by_province)
        self.visualizer.plot_gender_risk_breakdown(risk_by_gender)
        self.visualizer.plot_premium_vs_value()

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
