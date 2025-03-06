import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import the data generator
from data_generator import generate_test_data

# Import the monitoring classes
sys.path.append('./')  # Add current directory to path
from regional_round import Round1Monitor
from state_hub_round import Round2Monitor

class TestHackathonMonitoring(unittest.TestCase):
    """Test suite for hackathon monitoring classes"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once before running all tests"""
        print("Setting up test data...")
        cls.r1_data, cls.r1_avg, cls.r2_data = generate_test_data()
    
    def test_round1_monitor_initialization(self):
        """Test Round1Monitor initialization and data loading"""
        monitor = Round1Monitor("test_round1_scores.csv")
        self.assertIsNotNone(monitor)
        self.assertIsNotNone(monitor.data)
        self.assertEqual(len(monitor.data), len(self.r1_data))
        
        # Check that criteria and jury pairs are correctly extracted
        self.assertEqual(set(monitor.criteria), set(['relevance', 'innovation', 'feasibility', 'impact']))
        self.assertEqual(len(monitor.jury_pairs), len(self.r1_data['jury_pair_id'].unique()))
    
    def test_round1_overall_statistics(self):
        """Test the overall_statistics method of Round1Monitor"""
        monitor = Round1Monitor("test_round1_scores.csv")
        stats = monitor.overall_statistics()
        
        # Check that stats dataframe has the right shape
        self.assertEqual(stats.shape, (5, 5))  # 5 metrics for 5 columns
        
        # Check that means are within reasonable ranges
        for col in monitor.criteria + ['total_score']:
            self.assertTrue(1 <= stats.loc['mean', col] <= 5)
    
    def test_round1_jury_pair_analysis(self):
        """Test the jury_pair_analysis method of Round1Monitor"""
        monitor = Round1Monitor("test_round1_scores.csv")
        jury_stats = monitor.jury_pair_analysis()
        
        # Check that jury stats has the right shape and entries
        self.assertEqual(len(jury_stats), len(monitor.jury_pairs))
        self.assertTrue('mean' in jury_stats.columns)
        self.assertTrue('std' in jury_stats.columns)
        self.assertTrue('z_score' in jury_stats.columns)
        self.assertTrue('is_outlier' in jury_stats.columns)
        
        # Check that we identify at least one outlier jury pair
        # (synthetic data was designed to include outliers)
        self.assertTrue(jury_stats['is_outlier'].sum() > 0)
    
    def test_round1_alerts(self):
        """Test the generate_alerts method of Round1Monitor"""
        monitor = Round1Monitor("test_round1_scores.csv")
        alerts = monitor.generate_alerts()
        
        # Check that alerts is a list
        self.assertIsInstance(alerts, list)
        
        # Since we included anomalies in the data, we should have alerts
        self.assertTrue(len(alerts) > 0)
        
        # Check that alerts contain expected text patterns
        alert_text = ' '.join(alerts)
        self.assertTrue(any(['HIGH' in alert for alert in alerts]))
        self.assertTrue(any(['LOW' in alert for alert in alerts]))
    
    def test_round1_full_analysis(self):
        """Test the run_full_analysis method of Round1Monitor"""
        monitor = Round1Monitor("test_round1_scores.csv")
        # This should run without errors
        try:
            monitor.run_full_analysis()
            # Check that the plot was created
            self.assertTrue(os.path.exists('round1_score_distribution.png'))
        except Exception as e:
            self.fail(f"run_full_analysis raised {type(e).__name__} unexpectedly!")
    
    def test_round2_monitor_initialization(self):
        """Test Round2Monitor initialization and data loading"""
        monitor = Round2Monitor("test_round2_scores.csv")
        self.assertIsNotNone(monitor)
        self.assertIsNotNone(monitor.data)
        self.assertEqual(len(monitor.data), len(self.r2_data))
        
        # Check that criteria are correctly extracted
        self.assertEqual(set(monitor.criteria), 
                         set(['persona', 'problem_statement', 'user_need', 'solution_alignment']))
    
    def test_round2_jury_agreement(self):
        """Test the jury_agreement_analysis method of Round2Monitor"""
        monitor = Round2Monitor("test_round2_scores.csv")
        jury_comparison = monitor.jury_agreement_analysis()
        
        # Check that we get a dataframe with the expected columns
        self.assertIsNotNone(jury_comparison)
        
        # Check that the dataframe has diff columns
        for col in monitor.criteria + ['total_score']:
            self.assertTrue(f"{col}_diff" in jury_comparison.columns)
        
        # We intentionally added teams with high disagreement,
        # so we should find some with large differences
        self.assertTrue((jury_comparison['total_score_diff'] >= 2).sum() > 0)
    
    def test_round2_jury_bias(self):
        """Test the jury_bias_analysis method of Round2Monitor"""
        monitor = Round2Monitor("test_round2_scores.csv")
        jury_stats = monitor.jury_bias_analysis()
        
        # Check that the dataframe has the expected columns
        self.assertTrue('z_score' in jury_stats.columns)
        self.assertTrue('potentially_biased' in jury_stats.columns)
        
        # We intentionally added biased jurors, so we should find some
        self.assertTrue(jury_stats['potentially_biased'].sum() > 0)
    
    def test_round2_compare_rounds(self):
        """Test the compare_rounds method of Round2Monitor"""
        monitor = Round2Monitor("test_round2_scores.csv")
        comparison = monitor.compare_rounds("test_round1_team_avg_scores.csv")
        
        # Check that we get a dataframe with the expected columns
        self.assertIsNotNone(comparison)
        self.assertTrue('round1_score' in comparison.columns)
        self.assertTrue('round2_score' in comparison.columns)
        self.assertTrue('score_change' in comparison.columns)
        
        # We should have some correlation between rounds
        correlation = comparison['round1_score'].corr(comparison['round2_score'])
        self.assertTrue(correlation > 0.3)  # Expect moderate positive correlation
    
    def test_round2_calibration_recommendations(self):
        """Test the generate_calibration_recommendations method of Round2Monitor"""
        monitor = Round2Monitor("test_round2_scores.csv")
        recommendations = monitor.generate_calibration_recommendations()
        
        # Check that recommendations is a list
        self.assertIsInstance(recommendations, list)
        
        # Since we included anomalies in the data, we should have recommendations
        self.assertTrue(len(recommendations) > 0)
    
    def test_round2_full_analysis(self):
        """Test the run_full_analysis method of Round2Monitor"""
        monitor = Round2Monitor("test_round2_scores.csv")
        # This should run without errors
        try:
            monitor.run_full_analysis(round1_data_path="test_round1_team_avg_scores.csv")
            # Check that the plot was created
            self.assertTrue(os.path.exists('round2_score_analysis.png'))
        except Exception as e:
            self.fail(f"run_full_analysis raised {type(e).__name__} unexpectedly!")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with a non-existent file
        with self.assertRaises(Exception):
            monitor = Round1Monitor("nonexistent_file.csv")
            monitor.load_data()
        
        # Modify our data to create edge cases
        # 1. Create Round1 data with only one jury pair
        single_jury_data = self.r1_data.copy()
        single_jury_data['jury_pair_id'] = 1
        single_jury_data.to_csv("test_single_jury.csv", index=False)
        
        # Test with single jury data
        monitor = Round1Monitor("test_single_jury.csv")
        # This should still work but with limited analysis potential
        stats = monitor.overall_statistics()
        self.assertEqual(len(monitor.jury_pairs), 1)
        
        # 2. Create Round2 data with no teams having exactly 2 jury evaluations
        bad_r2_data = self.r2_data.copy()
        # Add a third juror to each team
        new_rows = []
        for team_id in bad_r2_data['team_id'].unique():
            jury_id = 999  # New jury ID
            new_row = {
                'team_id': team_id,
                'jury_id': jury_id,
                'persona': 3.0,
                'problem_statement': 3.0,
                'user_need': 3.0,
                'solution_alignment': 3.0,
                'total_score': 3.0
            }
            new_rows.append(new_row)
        
        bad_r2_data = pd.concat([bad_r2_data, pd.DataFrame(new_rows)], ignore_index=True)
        bad_r2_data.to_csv("test_bad_round2.csv", index=False)
        
        # Test with bad Round2 data
        monitor = Round2Monitor("test_bad_round2.csv")
        # load_data should show warnings but not crash
        monitor.load_data()
        # Jury agreement analysis should still run but with modified behavior
        jury_comparison = monitor.jury_agreement_analysis()
        
        # Clean up temporary files
        for f in ["test_single_jury.csv", "test_bad_round2.csv"]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    unittest.main()
