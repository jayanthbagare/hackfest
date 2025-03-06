import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Round1Monitor:
    """
    Statistical monitoring for Round 1 of the hackathon (10,000 teams)
    """
    
    def __init__(self, csv_path):
        """Initialize with path to the scoring CSV file"""
        self.data = pd.read_csv(csv_path)
        self.criteria = ['relevance', 'innovation', 'feasibility', 'impact']
        self.jury_pairs = self.data['jury_pair_id'].unique()
        
    def load_data(self):
        """
        Expected CSV format:
        team_id, jury_pair_id, relevance, innovation, feasibility, impact, total_score
        """
        print(f"Loaded data for {len(self.data)} team evaluations")
        print(f"Number of jury pairs: {len(self.jury_pairs)}")
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print("WARNING: Missing values detected:")
            print(missing[missing > 0])
    
    def overall_statistics(self):
        """Generate overall statistics for Round 1 scoring"""
        stats_df = pd.DataFrame({
            'mean': self.data[self.criteria + ['total_score']].mean(),
            'median': self.data[self.criteria + ['total_score']].median(),
            'std': self.data[self.criteria + ['total_score']].std(),
            'min': self.data[self.criteria + ['total_score']].min(),
            'max': self.data[self.criteria + ['total_score']].max()
        })
        
        print("Overall Scoring Statistics:")
        print(stats_df)
        return stats_df
    
    def jury_pair_analysis(self):
        """Analyze scoring patterns by jury pair"""
        jury_stats = self.data.groupby('jury_pair_id')['total_score'].agg(['mean', 'std', 'count'])
        jury_stats = jury_stats.sort_values('mean')
        
        # Identify outlier jury pairs (using z-score for mean scores)
        z_scores = stats.zscore(jury_stats['mean'])
        jury_stats['z_score'] = z_scores
        jury_stats['is_outlier'] = abs(z_scores) > 2.0
        
        outliers = jury_stats[jury_stats['is_outlier']]
        
        if len(outliers) > 0:
            print(f"\nWARNING: Detected {len(outliers)} outlier jury pairs:")
            print(outliers)
        
        return jury_stats
    
    def criteria_consistency(self):
        """Check for consistency in scoring across different criteria"""
        # Correlation matrix between criteria
        corr_matrix = self.data[self.criteria].corr()
        
        print("\nCriteria Correlation Matrix:")
        print(corr_matrix)
        
        # Check for unusual patterns (e.g., criteria that should correlate but don't)
        if (corr_matrix.values < 0.3).any():
            print("\nWARNING: Some criteria show weak correlation, which may indicate inconsistent evaluation")
        
        return corr_matrix
    
    def score_distribution(self):
        """Analyze the distribution of scores"""
        plt.figure(figsize=(15, 10))
        
        # Overall score distribution
        plt.subplot(2, 2, 1)
        sns.histplot(self.data['total_score'], kde=True)
        plt.title('Distribution of Total Scores')
        
        # Distribution by criteria
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.data[self.criteria])
        plt.title('Score Distribution by Criteria')
        
        # Distribution by jury pair
        plt.subplot(2, 2, 3)
        jury_means = self.data.groupby('jury_pair_id')['total_score'].mean()
        sns.histplot(jury_means, kde=True)
        plt.title('Mean Scores by Jury Pair')
        
        # Heatmap of criteria correlations
        plt.subplot(2, 2, 4)
        sns.heatmap(self.data[self.criteria].corr(), annot=True, cmap='coolwarm')
        plt.title('Criteria Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('round1_score_distribution.png')
        print("\nScore distribution plots saved to 'round1_score_distribution.png'")
    
    def generate_alerts(self):
        """Generate alerts for potential scoring issues"""
        alerts = []
        
        # 1. Check for jury pairs with very high/low average scores
        jury_stats = self.jury_pair_analysis()
        high_scorers = jury_stats[jury_stats['mean'] > jury_stats['mean'].mean() + 1.5 * jury_stats['mean'].std()]
        low_scorers = jury_stats[jury_stats['mean'] < jury_stats['mean'].mean() - 1.5 * jury_stats['mean'].std()]
        
        for _, row in high_scorers.iterrows():
            alerts.append(f"Jury pair {row.name} has unusually HIGH average scores: {row['mean']:.2f}")
        
        for _, row in low_scorers.iterrows():
            alerts.append(f"Jury pair {row.name} has unusually LOW average scores: {row['mean']:.2f}")
        
        # 2. Check for jury pairs with very low standard deviation (not differentiating between teams)
        low_variance = jury_stats[jury_stats['std'] < jury_stats['std'].mean() - 1.5 * jury_stats['std'].std()]
        for _, row in low_variance.iterrows():
            alerts.append(f"Jury pair {row.name} shows LOW VARIANCE in scoring (std: {row['std']:.2f})")
        
        # 3. Check for unusual scoring patterns in individual criteria
        for jury_id in self.jury_pairs:
            jury_data = self.data[self.data['jury_pair_id'] == jury_id]
            for criterion in self.criteria:
                mean = jury_data[criterion].mean()
                overall_mean = self.data[criterion].mean()
                if abs(mean - overall_mean) > 0.75:  # If more than 0.75 points different from overall
                    alerts.append(f"Jury pair {jury_id} scores {criterion} {'HIGHER' if mean > overall_mean else 'LOWER'} than average")
        
        # Print all alerts
        if alerts:
            print("\n=== SCORING ALERTS ===")
            for alert in alerts:
                print("⚠️ " + alert)
        else:
            print("\nNo scoring alerts detected.")
        
        return alerts
    
    def run_full_analysis(self):
        """Run all analysis functions and generate a report"""
        print("=== HACKATHON ROUND 1 SCORING ANALYSIS ===")
        print(f"Analyzing scores for {len(self.data)} teams across {len(self.jury_pairs)} jury pairs")
        
        self.load_data()
        self.overall_statistics()
        self.jury_pair_analysis()
        self.criteria_consistency()
        self.score_distribution()
        self.generate_alerts()
        
        print("\nAnalysis complete. See plots and alerts above for potential calibration issues.")

# Example usage
if __name__ == "__main__":
    monitor = Round1Monitor("round1_scores.csv")
    monitor.run_full_analysis()
