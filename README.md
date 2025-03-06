# Hackathon Monitoring System

A comprehensive system for monitoring and analyzing hackathon scoring data across multiple rounds, designed to ensure fair evaluation and identify potential scoring anomalies.

## Overview

This system provides statistical analysis and monitoring tools for hackathon organizers to:

- Detect scoring anomalies and jury bias
- Analyze scoring consistency across jury members
- Compare performance between hackathon rounds
- Generate automated recommendations for jury calibration
- Visualize scoring distributions and patterns

## Components

The system consists of several Python modules:

- **regional_round.py**: Monitoring for Round 1 (regional level) hackathon evaluations
- **state_hub_round.py**: Monitoring for Round 2 (state hub level) evaluations with design thinking criteria
- **synthetic_dataset.py**: Generates test data with controlled anomalies for system testing
- **test_cases.py**: Comprehensive unit tests for all monitoring functionality
- **consolidated_tests.py**: Test runner that orchestrates the entire testing process

## Data Format

### Round 1 Data

CSV format with columns:

- `team_id`: Unique identifier for each team
- `jury_pair_id`: Identifier for jury pairs evaluating teams
- `relevance`, `innovation`, `feasibility`, `impact`: Scoring criteria (1-5 scale)
- `total_score`: Average of all criteria scores

### Round 2 Data

CSV format with columns:

- `team_id`: Unique identifier for each team
- `jury_id`: Identifier for individual jury members
- `persona`, `problem_statement`, `user_need`, `solution_alignment`: Design thinking criteria (1-5 scale)
- `total_score`: Average of all criteria scores

## Features

### Round 1 Monitoring

- Overall scoring statistics
- Jury pair scoring pattern analysis
- Criteria consistency checking
- Score distribution visualizations
- Anomaly detection and alerts

### Round 2 Monitoring

- Jury agreement analysis
- Scoring bias detection
- Comparison with Round 1 performance
- Detailed calibration recommendations
- Advanced visualizations of scoring patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hackathon-monitoring.git
cd hackathon-monitoring

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy
```

## Usage

### Running the Test Suite

```bash
python consolidated_tests.py
```

This will:

1. Generate synthetic test data
2. Verify data quality
3. Run all unit tests
4. Execute monitoring examples
5. Provide a comprehensive test report

### Using the Monitoring Classes

```python
# For Round 1 monitoring
from regional_round import Round1Monitor

monitor = Round1Monitor("path_to_round1_scores.csv")
monitor.run_full_analysis()

# For Round 2 monitoring
from state_hub_round import Round2Monitor

monitor = Round2Monitor("path_to_round2_scores.csv")
monitor.run_full_analysis(round1_data_path="path_to_round1_avg_scores.csv")
```

### Generating Test Data

```python
from synthetic_dataset import generate_test_data

r1_data, r1_avg, r2_data = generate_test_data()
```

## Analysis Outputs

The system generates the following outputs:

- Detailed console logs of statistical analysis
- Score distribution visualizations
- Jury bias and agreement metrics
- Correlation analysis between rounds
- Specific alerts for scoring anomalies
- Actionable jury calibration recommendations

## Example Visualizations

- **round1_score_distribution.png**: Visualizes Round 1 scoring patterns
- **round2_score_analysis.png**: Analyzes Round 2 scoring consistency and jury agreement

## Testing

The system includes comprehensive unit tests to ensure all functionality works as expected:

```bash
python test_cases.py
```

## License

[MIT License](LICENSE)
