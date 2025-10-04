import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class DataVisualizer:
    """
    (This class remains the same, but now uses the cleaned and feature-engineered data.)
    """

    def __init__(self, df):
        self.df = df
        self.overall_loss_ratio = self.df['TotalClaims'].sum() / self.df['TotalPremium'].sum()

    # ... (all plotting methods: plot_financial_distributions, plot_geographic_risk_profile,
    #       plot_gender_risk_breakdown, plot_premium_vs_value remain the same) ...
    def plot_financial_distributions(self):
        """Plots the distributions of TotalPremium and log-transformed TotalClaims."""
        plt.figure(figsize=(14, 6))

        # TotalPremium Distribution
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['TotalPremium'], bins=50, kde=True, color='teal')
        plt.title('Distribution of Total Premium (ZAR)', fontsize=14)
        plt.xlabel('Total Premium')

        # TotalClaims Distribution (Focus on non-zero claims for better visibility)
        plt.subplot(1, 2, 2)
        claims_gt_0 = self.df[self.df['TotalClaims'] > 0]['TotalClaims']
        if not claims_gt_0.empty:
            sns.histplot(claims_gt_0, bins=50, kde=True, log_scale=True, color='salmon')
            plt.title('Distribution of Total Claims (Log Scale, Claims > 0)', fontsize=14)
            plt.xlabel('Total Claims (ZAR)')
        else:
            plt.text(0.5, 0.5, 'No Claims > 0 to plot', ha='center', va='center')

        plt.tight_layout()
        plt.show()

    def plot_geographic_risk_profile(self, risk_profile_df):
        """Plot 1: Geographic Risk Map (Loss Ratio by Province)."""
        # Take the top 10 provinces by total policies for visualization focus
        top_provinces = self.df['Province'].value_counts().head(10).index
        plot_data = risk_profile_df.loc[top_provinces].sort_values(by='LossRatio', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=plot_data.index, y='LossRatio', data=plot_data, palette='viridis')
        plt.title('Loss Ratio by Province (Top 10)', fontsize=16)
        plt.ylabel('Loss Ratio (Total Claims / Total Premium)')
        plt.xlabel('Province')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(self.overall_loss_ratio, color='red', linestyle='--', linewidth=2,
                    label=f'Overall LR: {self.overall_loss_ratio:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_gender_risk_breakdown(self, risk_profile_df):
        """Plot 2: Claims Frequency vs. Severity by Gender."""
        # Calculate Average Claim Severity (Total Claims / Number of Claims)
        plot_data = risk_profile_df.copy()
        plot_data['AvgClaimSeverity'] = np.where(
            plot_data['ClaimCount'] > 0,
            plot_data['TotalClaims'] / plot_data['ClaimCount'],
            0
        )

        # Filter for Men and Women only
        plot_data = plot_data[plot_data.index.isin(['M', 'F'])]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar Chart for Claim Frequency
        sns.barplot(x=plot_data.index, y='ClaimFrequency', data=plot_data, ax=ax1, color='skyblue',
                    label='Claim Frequency')
        ax1.set_ylabel('Claim Frequency (Policies with Claim / Total Policies)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # Line/Point Plot for Average Severity
        ax2 = ax1.twinx()
        sns.pointplot(x=plot_data.index, y='AvgClaimSeverity', data=plot_data, ax=ax2, color='red', markers='D',
                      linestyles='--', label='Avg. Claim Severity')
        ax2.set_ylabel('Average Claim Severity (ZAR)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Claims Frequency vs. Average Severity by Gender', fontsize=16)
        ax1.set_xlabel('Gender')

        # Manually combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_premium_vs_value(self):
        """Plot 3: Vehicle Value vs. Premium Density."""
        # Filter out extreme outliers for better visualization scale
        df_filtered = self.df[(self.df['CustomValueEstimate'] < self.df['CustomValueEstimate'].quantile(0.99)) &
                              (self.df['TotalPremium'] < self.df['TotalPremium'].quantile(0.99))]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x='CustomValueEstimate',
            y='TotalPremium',
            data=df_filtered,
            alpha=0.6,
            hue='CoverCategory',
            size='SumInsured',
            sizes=(20, 300)
        )
        plt.title('Total Premium vs. Vehicle Value (Colored by Cover Type)', fontsize=16)
        plt.xlabel('Custom Value Estimate (ZAR)')
        plt.ylabel('Total Premium (ZAR)')
        plt.legend(title='Cover Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()