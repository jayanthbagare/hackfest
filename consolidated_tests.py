#!/usr/bin/env python3
"""
Hackathon Monitoring Test Runner

This script automates the generation of synthetic test data and runs all test cases
to ensure that the hackathon monitoring system is functioning correctly.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the synthetic data generator
try:
    from synthetic_dataset import generate_test_data
except ImportError:
    logger.error("Could not import generate_test_data from synthetic_dataset.py")
    sys.exit(1)

# Import the monitoring classes
try:
    from regional_round import Round1Monitor
    from state_hub_round import Round2Monitor
except ImportError:
    logger.error("Could not import monitoring classes. Make sure regional_round.py and state_hub_round.py are in the current directory")
    sys.exit(1)

def check_file_exists(filepath):
    """Check if a file exists and log the result"""
    exists = os.path.exists(filepath)
    if exists:
        logger.info(f"File found: {filepath}")
    else:
        logger.warning(f"File not found: {filepath}")
    return exists

def run_data_generation():
    """Generate the synthetic test data and verify output files"""
    logger.info("Starting synthetic data generation...")
    start_time = time.time()
    
    try:
        r1_data, r1_avg, r2_data = generate_test_data()
        logger.info(f"Generated Round 1 data: {len(r1_data)} rows")
        logger.info(f"Generated Round 1 average scores: {len(r1_avg)} teams")
        logger.info(f"Generated Round 2 data: {len(r2_data)} rows")
    except Exception as e:
        logger.error(f"Error during data generation: {str(e)}")
        return False
    
    # Verify that files were created
    files_to_check = [
        "test_round1_scores.csv",
        "test_round1_team_avg_scores.csv",
        "test_round2_scores.csv"
    ]
    
    all_files_exist = all(check_file_exists(file) for file in files_to_check)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Data generation completed in {elapsed_time:.2f} seconds")
    
    return all_files_exist

def verify_data_quality():
    """Verify the generated data has expected properties"""
    logger.info("Verifying data quality...")
    
    try:
        # Load generated files
        r1_data = pd.read_csv("test_round1_scores.csv")
        r1_avg = pd.read_csv("test_round1_team_avg_scores.csv")
        r2_data = pd.read_csv("test_round2_scores.csv")
        
        # Verify Round 1 data structure
        expected_r1_columns = ['team_id', 'jury_pair_id', 'relevance', 'innovation', 
                             'feasibility', 'impact', 'total_score']
        r1_columns_ok = all(col in r1_data.columns for col in expected_r1_columns)
        
        # Verify Round 2 data structure
        expected_r2_columns = ['team_id', 'jury_id', 'persona', 'problem_statement', 
                             'user_need', 'solution_alignment', 'total_score']
        r2_columns_ok = all(col in r2_data.columns for col in expected_r2_columns)
        
        # Verify score ranges (should be between 1-5)
        r1_score_range_ok = (r1_data[['relevance', 'innovation', 'feasibility', 'impact']].min().min() >= 1 and
                           r1_data[['relevance', 'innovation', 'feasibility', 'impact']].max().max() <= 5)
        
        r2_score_range_ok = (r2_data[['persona', 'problem_statement', 'user_need', 'solution_alignment']].min().min() >= 1 and
                           r2_data[['persona', 'problem_statement', 'user_need', 'solution_alignment']].max().max() <= 5)
        
        # Verify some teams have multiple jury evaluations in Round 2
        r2_jury_eval_counts = r2_data.groupby('team_id').size()
        r2_multi_evals_ok = (r2_jury_eval_counts > 1).any()
        
        all_checks_passed = (r1_columns_ok and r2_columns_ok and 
                           r1_score_range_ok and r2_score_range_ok and 
                           r2_multi_evals_ok)
        
        if all_checks_passed:
            logger.info("All data quality checks passed")
        else:
            if not r1_columns_ok:
                logger.warning("Round 1 data is missing expected columns")
            if not r2_columns_ok:
                logger.warning("Round 2 data is missing expected columns")
            if not r1_score_range_ok:
                logger.warning("Round 1 data contains scores outside the valid range (1-5)")
            if not r2_score_range_ok:
                logger.warning("Round 2 data contains scores outside the valid range (1-5)")
            if not r2_multi_evals_ok:
                logger.warning("Round 2 data doesn't contain teams with multiple jury evaluations")
        
        return all_checks_passed
    
    except Exception as e:
        logger.error(f"Error during data quality verification: {str(e)}")
        return False

def run_tests():
    """Run the unit tests and return True if all tests pass"""
    logger.info("Running unit tests...")
    start_time = time.time()
    
    # Import test cases
    try:
        from test_cases import TestHackathonMonitoring
    except ImportError:
        logger.error("Could not import TestHackathonMonitoring from test_cases.py")
        return False
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHackathonMonitoring)
    
    # Run the tests with a custom test result class to capture results
    class TestResultWithCount(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_count = 0
            self.success_count = 0
        
        def startTest(self, test):
            super().startTest(test)
            self.test_count += 1
        
        def addSuccess(self, test):
            super().addSuccess(test)
            self.success_count += 1
    
    # Create a string buffer to capture test output
    from io import StringIO
    test_output = StringIO()
    test_runner = unittest.TextTestRunner(stream=test_output, 
                                          verbosity=2, 
                                          resultclass=TestResultWithCount)
    test_result = test_runner.run(suite)
    
    # Log test output
    logger.info(test_output.getvalue())
    
    # Calculate success rate
    success_rate = test_result.success_count / test_result.test_count * 100 if test_result.test_count > 0 else 0
    
    elapsed_time = time.time() - start_time
    logger.info(f"Testing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Test results: {test_result.success_count}/{test_result.test_count} tests passed ({success_rate:.1f}%)")
    
    return test_result.wasSuccessful()

def run_monitoring_examples():
    """Run examples of the monitoring classes with the test data"""
    logger.info("Running monitoring examples...")
    
    try:
        # Run Round 1 Monitor example
        logger.info("Running Round 1 Monitor example...")
        round1_monitor = Round1Monitor("test_round1_scores.csv")
        round1_monitor.run_full_analysis()
        
        # Run Round 2 Monitor example
        logger.info("Running Round 2 Monitor example...")
        round2_monitor = Round2Monitor("test_round2_scores.csv")
        round2_monitor.run_full_analysis(round1_data_path="test_round1_team_avg_scores.csv")
        
        # Verify that output files were created
        plot_files = [
            "round1_score_distribution.png",
            "round2_score_analysis.png"
        ]
        
        plots_created = all(check_file_exists(file) for file in plot_files)
        
        return plots_created
    
    except Exception as e:
        logger.error(f"Error during monitoring examples: {str(e)}")
        return False

def cleanup():
    """Clean up test files (optional)"""
    files_to_cleanup = []  # Add files to clean up if needed
    
    for file in files_to_cleanup:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"Removed file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file}: {str(e)}")

def main():
    """Main function to run the complete test process"""
    logger.info("=" * 60)
    logger.info("HACKATHON MONITORING SYSTEM TEST RUNNER")
    logger.info("=" * 60)
    
    # Step 1: Generate synthetic data
    logger.info("\n--- STEP 1: GENERATING SYNTHETIC DATA ---")
    data_generated = run_data_generation()
    if not data_generated:
        logger.error("Data generation failed or files not created. Aborting test run.")
        return False
    
    # Step 2: Verify data quality
    logger.info("\n--- STEP 2: VERIFYING DATA QUALITY ---")
    data_quality_ok = verify_data_quality()
    if not data_quality_ok:
        logger.warning("Data quality issues detected. Proceeding with caution.")
    
    # Step 3: Run unit tests
    logger.info("\n--- STEP 3: RUNNING UNIT TESTS ---")
    tests_passed = run_tests()
    if not tests_passed:
        logger.error("Unit tests failed. System may have issues.")
        return False
    
    # Step 4: Run monitoring examples
    logger.info("\n--- STEP 4: RUNNING MONITORING EXAMPLES ---")
    examples_ran = run_monitoring_examples()
    if not examples_ran:
        logger.warning("Monitoring examples had issues or didn't generate expected outputs.")
    
    # Optional: Clean up test files
    # cleanup()
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Data generation: {'SUCCESS' if data_generated else 'FAILED'}")
    logger.info(f"Data quality: {'GOOD' if data_quality_ok else 'ISSUES DETECTED'}")
    logger.info(f"Unit tests: {'PASSED' if tests_passed else 'FAILED'}")
    logger.info(f"Monitoring examples: {'SUCCESS' if examples_ran else 'ISSUES DETECTED'}")
    
    overall_success = data_generated and tests_passed
    logger.info(f"\nOVERALL TEST RESULT: {'SUCCESS' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
