import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Round2Monitor:
    """
    Statistical monitoring for Round 2 of the hackathon (2,000 teams)
    Focuses on Design Thinking evaluation with multiple jury members per team
    """
    
    def __init__(self, scores_path):
        """Initialize with path to the scoring CSV file"""
        self.data = pd.read_csv(scores_path)
        self.criteria = ['persona', 'problem_statement', 'user_need', 'solution_alignment']
        
    def load_data(self):
        """
        Expected CSV format:
        team_id, jury_id, persona, problem_statement, user_need, solution_alignment, total_score
        
        Note: Each team should have scores from 2 jury members
        """
        # Check data integrity
        team_counts = self.data.groupby('team_id').size()
        missing_scores = team_counts[team_counts != 2].index.tolist()
        
        if missing_scores:
            print(f"WARNING: {len(missing_scores)} teams don't have exactly 2 jury evaluations")
            if len(missing_scores) < 10:
                print(f"Teams with missing/extra scores: {missing_scores}")
            else:
                print(f"First 10 teams with missing/extra scores: {missing_scores[:10]}...")
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("\nWARNING: Missing values detected:")
            print(missing_values[missing_values > 0])
    
    def jury_agreement_analysis(self):
        """Analyze agreement between jury members scoring the same teams"""
        # Calculate score differences between jury members for the same team
        teams_with_2_jurors = self.data.groupby('team_id').filter(lambda x: len(x) == 2)
        
        # Reshape data to compare scores from different jury members
        jury_comparison = pd.DataFrame()
        
        for team_id in teams_with_2_jurors['team_id'].unique():
            team_scores = self.data[self.data['team_id'] == team_id]
            if len(team_scores) == 2:
                row1, row2 = team_scores.iloc[0], team_scores.iloc[1]
                
                for criterion in self.criteria + ['total_score']:
                    jury_comparison.loc[team_id, f"{criterion}_diff"] = abs(row1[criterion] - row2[criterion])
        
        # Summary statistics of differences
        diff_stats = jury_comparison.describe().transpose()
        print("\nJury Agreement Analysis (absolute differences):")
        print(diff_stats)
        
        # Identify teams with high disagreement
        high_disagreement = jury_comparison[jury_comparison['total_score_diff'] >= 2]
        if not high_disagreement.empty:
            print(f"\nWARNING: {len(high_disagreement)} teams have jury score differences >= 2 points")
        
        # Calculate Intraclass Correlation Coefficient for each criterion
        # (would require reshaping the data - simplified for this example)
        
        return jury_comparison
    
    def calculate_average_scores(self):
        """Calculate average scores for each team across jury members"""
        # Group by team and calculate means for each criterion
        avg_scores = self.data.groupby('team_id')[self.criteria + ['total_score']].mean().reset_index()
        return avg_scores
    
    def jury_bias_analysis(self):
        """Analyze individual jury member scoring patterns for potential bias"""
        jury_stats = self.data.groupby('jury_id')[self.criteria + ['total_score']].agg(['mean', 'std']).reset_index()
        
        # Flatten multi-level columns
        jury_stats.columns = ['_'.join(col).strip('_') for col in jury_stats.columns.values]
        
        # Calculate z-scores for each jury's average scores
        overall_mean = self.data['total_score'].mean()
        overall_std = self.data['total_score'].std()
        
        jury_stats['z_score'] = (jury_stats['total_score_mean'] - overall_mean) / overall_std
        jury_stats['potentially_biased'] = abs(jury_stats['z_score']) > 1.5
        
        # Identify potentially biased jurors
        biased_jurors = jury_stats[jury_stats['potentially_biased']]
        
        if not biased_jurors.empty:
            print(f"\nWARNING: {len(biased_jurors)} jury members show potential scoring bias:")
            print(biased_jurors[['jury_id', 'total_score_mean', 'z_score']])
        
        return jury_stats
    
    def compare_rounds(self, round1_data_path):
        """Compare team performance between Round 1 and Round 2"""
        try:
            round1_data = pd.read_csv(round1_data_path)
            round1_avg = round1_data.groupby('team_id')['total_score'].mean().reset_index()
            round1_avg = round1_avg.rename(columns={'total_score': 'round1_score'})
            
            round2_avg = self.calculate_average_scores()[['team_id', 'total_score']]
            round2_avg = round2_avg.rename(columns={'total_score': 'round2_score'})
            
            # Merge the two rounds
            comparison = pd.merge(round1_avg, round2_avg, on='team_id', how='inner')
            
            # Calculate correlation
            correlation = comparison['round1_score'].corr(comparison['round2_score'])
            print(f"\nCorrelation between Round 1 and Round 2 scores: {correlation:.4f}")
            
            # Identify teams with large score changes
            comparison['score_change'] = comparison['round2_score'] - comparison['round1_score']
            large_increases = comparison[comparison['score_change'] > 2]
            large_decreases = comparison[comparison['score_change'] < -2]
            
            if not large_increases.empty:
                print(f"\n{len(large_increases)} teams had score increases > 2 points from Round 1 to Round 2")
            
            if not large_decreases.empty:
                print(f"\n{len(large_decreases)} teams had score decreases > 2 points from Round 1 to Round 2")
            
            return comparison
        except Exception as e:
            print(f"Error comparing rounds: {e}")
            return None
    
    def visualize_distributions(self):
        """Create visualizations of score distributions"""
        plt.figure(figsize=(15, 10))
        
        # Average scores distribution
        avg_scores = self.calculate_average_scores()
        
        plt.subplot(2, 2, 1)
        sns.histplot(avg_scores['total_score'], kde=True)
        plt.title('Distribution of Average Total Scores')
        plt.axvline(avg_scores['total_score'].mean(), color='r', linestyle='--')
        
        # Distribution by criteria
        plt.subplot(2, 2, 2)
        avg_criteria = avg_scores[self.criteria].melt()
        sns.boxplot(x='variable', y='value', data=avg_criteria)
        plt.title('Score Distribution by Criteria')
        plt.xticks(rotation=45)
        
        # Jury agreement heatmap
        jury_comparison = self.jury_agreement_analysis()
        
        plt.subplot(2, 2, 3)
        diff_cols = [col for col in jury_comparison.columns if col.endswith('_diff')]
        sns.heatmap(jury_comparison[diff_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation of Jury Disagreements')
        
        # Jury bias visualization
        jury_stats = self.jury_bias_analysis()
        
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='total_score_mean', y='total_score_std', data=jury_stats)
        for _, row in jury_stats[abs(jury_stats['z_score']) > 1.5].iterrows():
            plt.annotate(row['jury_id'], 
                         (row['total_score_mean'], row['total_score_std']),
                         xytext=(5, 5), textcoords='offset points')
        plt.title('Jury Scoring Patterns (Mean vs. Std)')
        
        plt.tight_layout()
        plt.savefig('round2_score_analysis.png')
        print("\nScore analysis plots saved to 'round2_score_analysis.png'")
    
    def generate_calibration_recommendations(self):
        """Generate recommendations for jury calibration"""
        # Calculate agreement statistics first
        jury_comparison = self.jury_agreement_analysis()
        jury_stats = self.jury_bias_analysis()
        
        recommendations = []
        
        # 1. Identify criteria with most disagreement
        criteria_diff_means = {col.replace('_diff', ''): jury_comparison[col].mean() 
                              for col in jury_comparison.columns if col.endswith('_diff')}
        
        most_disagreement = max(criteria_diff_means.items(), key=lambda x: x[1])
        recommendations.append(f"Focus calibration on '{most_disagreement[0]}' criterion which shows highest jury disagreement (avg diff: {most_disagreement[1]:.2f} points)")
        
        # 2. Identify jury members needing calibration
        for _, row in jury_stats[abs(jury_stats['z_score']) > 1.5].iterrows():
            direction = "HIGH" if row['z_score'] > 0 else "LOW"
            recommendations.append(f"Calibrate jury member {row['jury_id']} who scores consistently {direction} (z-score: {row['z_score']:.2f})")
        
        # 3. General recommendations based on patterns
        if criteria_diff_means['total_score'] > 1.0:
            recommendations.append("Consider additional clarification on rubric as average jury disagreement exceeds 1 point")
        
        # 4. Teams to review
        high_disagreement = jury_comparison[jury_comparison['total_score_diff'] >= 2]
        if len(high_disagreement) > 0:
            top_review = high_disagreement.nlargest(3, 'total_score_diff').index.tolist()
            recommendations.append(f"Perform manual review of teams with highest scoring disagreement: {top_review}")
        
        print("\n=== CALIBRATION RECOMMENDATIONS ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        return recommendations
    
    def run_full_analysis(self, round1_data_path=None):
        """Run complete analysis with all methods"""
        print("=== HACKATHON ROUND 2 SCORING ANALYSIS ===")
        
        self.load_data()
        self.jury_agreement_analysis()
        self.jury_bias_analysis()
        self.visualize_distributions()
        
        if round1_data_path:
            self.compare_rounds(round1_data_path)
        
        self.generate_calibration_recommendations()
        
        print("\nAnalysis complete. See plots and recommendations above.")

# Example usage
if __name__ == "__main__":
    monitor = Round2Monitor("round2_scores.csv")
    monitor.run_full_analysis(round1_data_path="round1_team_avg_scores.csv")
