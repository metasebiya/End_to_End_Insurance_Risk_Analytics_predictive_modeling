# business_reporter.py
import os
from datetime import datetime

class BusinessReporter:
    """
    Generates a Markdown report summarizing A/B test results and business insights.
    """

    def __init__(self, ab_results, output_dir="reports"):
        self.ab_results = ab_results
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_markdown(self, filename="abtest_report.md"):
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, "w", encoding="utf-8" ) as f:
            f.write(f"# Insurance Risk A/B Testing Report\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n")
            f.write("This report summarizes hypothesis tests on key risk drivers "
                    "including provinces, zip codes, margins, and gender.\n\n")

            f.write("## Hypothesis Testing Results\n")
            f.write("| Hypothesis | Test Used | p-value | Decision | Interpretation |\n")
            f.write("|------------|-----------|---------|----------|----------------|\n")

            for test_name, res in self.ab_results.items():
                p = res.get("p_value", None)
                decision = "Reject H₀" if p and p < 0.05 else "Fail to reject H₀"
                interp = self.interpret(test_name, decision)
                test_used = res.get("test", "N/A")
                f.write(f"| {test_name} | {test_used} | {p:.4f} | {decision} | {interp} |\n")

            f.write("\n## Business Recommendations\n")
            f.write("- **Provinces:** Adjust premiums regionally where risk differences are significant.\n")
            f.write("- **Zip Codes:** Simplify rating if no differences; micro-rate if margins differ.\n")
            f.write("- **Gender:** Avoid gender-based pricing (compliance/fairness).\n")
            f.write("- **Margins:** Investigate low-margin areas for fraud or mispricing.\n")

        print(f"Markdown report generated at: {report_path}")
        return report_path

    def interpret(self, test_name, decision):
        if "province" in test_name and decision == "Reject H₀":
            return "Regional risk differences detected. Adjust premiums by province."
        if "zip" in test_name and decision == "Reject H₀":
            return "Zip-level differences found. Consider micro-rating or fraud checks."
        if "gender" in test_name and decision == "Reject H₀":
            return "Gender differences detected, but use with caution due to compliance."
        return "No significant differences detected."
