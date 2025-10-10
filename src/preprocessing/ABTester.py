import pandas as pd
import numpy as np
from scipy import stats
from src.reports.business_reporter import BusinessReporter
class ABTester:
    """
    Executes hypothesis tests on insurance KPIs:
    - Frequency (binary proportion)
    - Severity (continuous conditional metric)
    - Margin (continuous)
    Provides multi-group and A/B pairwise tests with basic balance checks.
    """

    def __init__(self, df):
        self.df = df.copy()
        # Derived labels
        self.df['HadClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        # If ClaimCount not present, proxy with HadClaim
        if 'ClaimCount' not in self.df.columns:
            self.df['ClaimCount'] = self.df['HadClaim']
        # Severity conditional on ClaimCount > 0
        self.df['AvgClaimSeverity'] = np.where(self.df['ClaimCount'] > 0,
                                               self.df['TotalClaims'] / self.df['ClaimCount'], np.nan)
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

    # ---------- Multi-group tests ----------
    def chi2_frequency_by_group(self, group_col):
        ct = pd.crosstab(self.df[group_col], self.df['HadClaim'])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        return {"group": group_col, "test": "chi2_frequency", "chi2": chi2, "dof": dof, "p_value": p,
                "table": ct}

    def anova_severity_by_group(self, group_col):
        groups = [g.dropna().values for _, g in self.df.groupby(group_col)['AvgClaimSeverity']]
        # Fallback to Kruskal if any group small or non-normal suspected
        if any(len(g) < 20 for g in groups):
            stat, p = stats.kruskal(*groups)
            return {"group": group_col, "test": "kruskal_severity", "stat": stat, "p_value": p}
        f, p = stats.f_oneway(*groups)
        return {"group": group_col, "test": "anova_severity", "F": f, "p_value": p}

    def anova_margin_by_group(self, group_col):
        groups = [g.values for _, g in self.df.groupby(group_col)['Margin']]
        if any(len(g) < 20 for g in groups):
            stat, p = stats.kruskal(*groups)
            return {"group": group_col, "test": "kruskal_margin", "stat": stat, "p_value": p}
        f, p = stats.f_oneway(*groups)
        return {"group": group_col, "test": "anova_margin", "F": f, "p_value": p}

    # ---------- Pairwise A/B tests ----------
    def ttest_metric_between_groups(self, group_col, metric_col, a, b):
        a_vals = self.df[self.df[group_col] == a][metric_col].dropna()
        b_vals = self.df[self.df[group_col] == b][metric_col].dropna()
        # Welch’s t-test by default (unequal variances)
        t, p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        return {"group": group_col, "metric": metric_col, "A": a, "B": b,
                "mean_A": float(a_vals.mean()), "mean_B": float(b_vals.mean()),
                "t_stat": float(t), "p_value": float(p)}

    def ztest_proportions(self, group_col, a, b):
        a_succ = int(self.df[self.df[group_col] == a]['HadClaim'].sum())
        a_total = int(self.df[self.df[group_col] == a]['HadClaim'].count())
        b_succ = int(self.df[self.df[group_col] == b]['HadClaim'].sum())
        b_total = int(self.df[self.df[group_col] == b]['HadClaim'].count())
        p_pool = (a_succ + b_succ) / (a_total + b_total)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/a_total + 1/b_total))
        z = ((a_succ/a_total) - (b_succ/b_total)) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return {"group": group_col, "metric": "HadClaim", "A": a, "B": b,
                "rate_A": a_succ/a_total, "rate_B": b_succ/b_total,
                "z_score": float(z), "p_value": float(p)}

    # ---------- Balance checks ----------
    def balance_check(self, group_col, a, b, covariates_num=None, covariates_cat=None):
        covariates_num = covariates_num or ['CustomValueEstimate', 'SumInsured', 'TotalPremium']
        covariates_cat = covariates_cat or ['CoverCategory', 'Bodytype', 'LegalType']
        report = {"numeric": {}, "categorical": {}}
        a_df = self.df[self.df[group_col] == a]
        b_df = self.df[self.df[group_col] == b]
        for col in covariates_num:
            x = a_df[col].dropna(); y = b_df[col].dropna()
            if len(x) > 5 and len(y) > 5:
                t, p = stats.ttest_ind(x, y, equal_var=False)
                report["numeric"][col] = {"mean_A": float(x.mean()), "mean_B": float(y.mean()),
                                          "t_stat": float(t), "p_value": float(p)}
        for col in covariates_cat:
            ct = pd.crosstab(a_df[col], b_df[col])
            if ct.size > 0:
                chi2, p, _, _ = stats.chi2_contingency(ct)
                report["categorical"][col] = {"chi2": float(chi2), "p_value": float(p)}
        return report

    # ---------- Execution helpers ----------
    def run_all(self):
        results = {}

        # H0-1: Provinces (risk differences)
        results["province_freq"] = self.chi2_frequency_by_group("Province")
        results["province_severity"] = self.anova_severity_by_group("Province")

        # H0-2: Zip codes (risk differences)
        zip_col = "PostalCode" if "PostalCode" in self.df.columns else "ZipCode"
        results["zip_freq"] = self.chi2_frequency_by_group(zip_col)
        results["zip_severity"] = self.anova_severity_by_group(zip_col)

        # H0-3: Zip margins (profit differences)
        results["zip_margin"] = self.anova_margin_by_group(zip_col)

        # H0-4: Gender (risk differences)
        # Frequency (proportion test) and Severity (Welch t-test)
        #results["gender_freq"] = self.ztest_proportions("Gender", "F", "M")
        results["gender_severity"] = self.ttest_metric_between_groups("Gender", "AvgClaimSeverity", "F", "M")

        return results

def interpret_result(test_name, res, alpha=0.05):
    # Basic interpreter returning accept/reject and a short statement
    p = res.get("p_value", None)
    if p is None:
        # For chi2_frequency with detailed table, get its p_value
        p = res.get("p_value", None)
    decision = "Reject H0" if (p is not None and p < alpha) else "Fail to reject H0"
    return {"test": test_name, "p_value": p, "decision": decision}
