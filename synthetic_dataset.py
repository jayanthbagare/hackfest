import pandas as pd
import numpy as np
import random
import os

def generate_round1_data(num_teams=10000, num_jury_pairs=100, add_anomalies=True):
    """
    Generate synthetic data for Round 1 of the hackathon.
    
    Parameters:
    -----------
    num_teams : int
        Number of teams to generate data for
    num_jury_pairs : int
        Number of jury pairs evaluating teams
    add_anomalies : bool
        Whether to add anomalous scoring patterns for testing
        
    Returns:
    --------
    DataFrame with columns: team_id, jury_pair_id, relevance, innovation, 
                           feasibility, impact, total_score
    """
    team_ids = range(1, num_teams + 1)
    jury_pair_ids = range(1, num_jury_pairs + 1)
    
    # Assign teams to jury pairs (ensure roughly equal distribution)
    teams_per_jury = num_teams // num_jury_pairs
    team_assignments = []
    
    for jury_id in jury_pair_ids:
        # Assign approximately teams_per_jury teams to each jury pair
        start_idx = (jury_id - 1) * teams_per_jury
        end_idx = min(start_idx + teams_per_jury, num_teams)
        for team_id in range(start_idx + 1, end_idx + 1):
            team_assignments.append((team_id, jury_id))
    
    # Create base dataframe
    df = pd.DataFrame(team_assignments, columns=['team_id', 'jury_pair_id'])
    
    # Generate normal scoring distributions (1-5 scale)
    df['relevance'] = np.random.normal(3.5, 0.8, len(df)).clip(1, 5).round(1)
    df['innovation'] = np.random.normal(3.3, 0.9, len(df)).clip(1, 5).round(1)
    df['feasibility'] = np.random.normal(3.2, 0.7, len(df)).clip(1, 5).round(1)
    df['impact'] = np.random.normal(3.4, 0.85, len(df)).clip(1, 5).round(1)
    
    # Calculate total score (average of criteria)
    df['total_score'] = df[['relevance', 'innovation', 'feasibility', 'impact']].mean(axis=1).round(1)
    
    if add_anomalies:
        # Add some jury pair bias (some jury pairs scoring consistently high or low)
        high_scoring_pairs = random.sample(list(jury_pair_ids), 3)
        low_scoring_pairs = random.sample([j for j in jury_pair_ids if j not in high_scoring_pairs], 3)
        
        # High scoring jury pairs: add 0.8-1.0 to all scores
        for jury_id in high_scoring_pairs:
            mask = df['jury_pair_id'] == jury_id
            for col in ['relevance', 'innovation', 'feasibility', 'impact']:
                df.loc[mask, col] = (df.loc[mask, col] + random.uniform(0.8, 1.0)).clip(1, 5).round(1)
            df.loc[mask, 'total_score'] = df.loc[mask, ['relevance', 'innovation', 'feasibility', 'impact']].mean(axis=1).round(1)
        
        # Low scoring jury pairs: subtract 0.8-1.0 from all scores
        for jury_id in low_scoring_pairs:
            mask = df['jury_pair_id'] == jury_id
            for col in ['relevance', 'innovation', 'feasibility', 'impact']:
                df.loc[mask, col] = (df.loc[mask, col] - random.uniform(0.8, 1.0)).clip(1, 5).round(1)
            df.loc[mask, 'total_score'] = df.loc[mask, ['relevance', 'innovation', 'feasibility', 'impact']].mean(axis=1).round(1)
        
        # Jury pair with low variance (not differentiating between teams)
        low_var_pair = random.choice([j for j in jury_pair_ids if j not in high_scoring_pairs and j not in low_scoring_pairs])
        mask = df['jury_pair_id'] == low_var_pair
        mean_value = random.uniform(2.8, 3.8)
        for col in ['relevance', 'innovation', 'feasibility', 'impact']:
            df.loc[mask, col] = np.random.normal(mean_value, 0.2, mask.sum()).clip(1, 5).round(1)
        df.loc[mask, 'total_score'] = df.loc[mask, ['relevance', 'innovation', 'feasibility', 'impact']].mean(axis=1).round(1)
        
        # Add some missing values for testing robustness
        random_indices = np.random.choice(df.index, size=20, replace=False)
        random_cols = np.random.choice(['relevance', 'innovation', 'feasibility', 'impact'], size=20, replace=True)
        for idx, col in zip(random_indices, random_cols):
            df.loc[idx, col] = np.nan
        
        # Add highly correlated criteria for some jury pairs and uncorrelated for others
        correlated_pair = random.choice([j for j in jury_pair_ids if j not in high_scoring_pairs and j not in low_scoring_pairs and j != low_var_pair])
        mask = df['jury_pair_id'] == correlated_pair
        base_scores = np.random.normal(3.5, 0.8, mask.sum()).clip(1, 5)
        df.loc[mask, 'relevance'] = (base_scores + np.random.normal(0, 0.3, mask.sum())).clip(1, 5).round(1)
        df.loc[mask, 'innovation'] = (base_scores + np.random.normal(0, 0.3, mask.sum())).clip(1, 5).round(1)
        df.loc[mask, 'feasibility'] = (base_scores + np.random.normal(0, 0.3, mask.sum())).clip(1, 5).round(1)
        df.loc[mask, 'impact'] = (base_scores + np.random.normal(0, 0.3, mask.sum())).clip(1, 5).round(1)
        df.loc[mask, 'total_score'] = df.loc[mask, ['relevance', 'innovation', 'feasibility', 'impact']].mean(axis=1).round(1)
    
    return df

def generate_round2_data(num_teams=2000, num_jurors=50, previous_round_data=None, add_anomalies=True):
    """
    Generate synthetic data for Round 2 of the hackathon.
    
    Parameters:
    -----------
    num_teams : int
        Number of teams to generate data for
    num_jurors : int
        Number of jury members evaluating teams
    previous_round_data : DataFrame
        Data from Round 1 to maintain some correlation
    add_anomalies : bool
        Whether to add anomalous scoring patterns for testing
        
    Returns:
    --------
    DataFrame with columns: team_id, jury_id, persona, problem_statement, 
                           user_need, solution_alignment, total_score
    """
    # Round 2 has fewer teams (assume teams were selected from Round 1)
    if previous_round_data is not None:
        # Select top teams from Round 1 based on total_score
        avg_scores = previous_round_data.groupby('team_id')['total_score'].mean().reset_index()
        selected_teams = avg_scores.nlargest(num_teams, 'total_score')['team_id'].tolist()
    else:
        # If no previous data, just generate team IDs
        selected_teams = list(range(1, num_teams + 1))
    
    # Each team is evaluated by 2 jurors
    team_assignments = []
    jury_ids = list(range(1, num_jurors + 1))
    
    for team_id in selected_teams:
        # Assign 2 different jurors to each team
        team_jurors = random.sample(jury_ids, 2)
        for jury_id in team_jurors:
            team_assignments.append((team_id, jury_id))
    
    # Create base dataframe
    df = pd.DataFrame(team_assignments, columns=['team_id', 'jury_id'])
    
    # Generate normal scoring distributions (1-5 scale)
    df['persona'] = np.random.normal(3.4, 0.7, len(df)).clip(1, 5).round(1)
    df['problem_statement'] = np.random.normal(3.5, 0.8, len(df)).clip(1, 5).round(1)
    df['user_need'] = np.random.normal(3.3, 0.75, len(df)).clip(1, 5).round(1)
    df['solution_alignment'] = np.random.normal(3.2, 0.85, len(df)).clip(1, 5).round(1)
    
    # Calculate total score (average of criteria)
    df['total_score'] = df[['persona', 'problem_statement', 'user_need', 'solution_alignment']].mean(axis=1).round(1)
    
    if add_anomalies:
        # Add correlation with Round 1 (if provided)
        if previous_round_data is not None:
            r1_avg_scores = previous_round_data.groupby('team_id')['total_score'].mean()
            
            # Adjust scores based on Round 1 performance (with some randomness)
            for team_id in selected_teams:
                if team_id in r1_avg_scores.index:
                    # Get team's Round 1 performance relative to average
                    r1_score = r1_avg_scores[team_id]
                    r1_avg = r1_avg_scores.mean()
                    r1_std = r1_avg_scores.std()
                    
                    # Adjust Round 2 scores with 70% correlation + randomness
                    performance_factor = 0.7 * (r1_score - r1_avg) / r1_std
                    
                    # Apply adjustment (bounded by min/max scores)
                    mask = df['team_id'] == team_id
                    for col in ['persona', 'problem_statement', 'user_need', 'solution_alignment']:
                        df.loc[mask, col] = (df.loc[mask, col] + performance_factor * 0.5).clip(1, 5).round(1)
                    
                    df.loc[mask, 'total_score'] = df.loc[mask, ['persona', 'problem_statement', 'user_need', 'solution_alignment']].mean(axis=1).round(1)
        
        # Add juror bias (some jurors scoring consistently high or low)
        high_scoring_jurors = random.sample(jury_ids, 3)
        low_scoring_jurors = random.sample([j for j in jury_ids if j not in high_scoring_jurors], 3)
        
        # High scoring jurors: add 0.7-0.9 to all scores
        for jury_id in high_scoring_jurors:
            mask = df['jury_id'] == jury_id
            for col in ['persona', 'problem_statement', 'user_need', 'solution_alignment']:
                df.loc[mask, col] = (df.loc[mask, col] + random.uniform(0.7, 0.9)).clip(1, 5).round(1)
            df.loc[mask, 'total_score'] = df.loc[mask, ['persona', 'problem_statement', 'user_need', 'solution_alignment']].mean(axis=1).round(1)
        
        # Low scoring jurors: subtract 0.7-0.9 from all scores
        for jury_id in low_scoring_jurors:
            mask = df['jury_id'] == jury_id
            for col in ['persona', 'problem_statement', 'user_need', 'solution_alignment']:
                df.loc[mask, col] = (df.loc[mask, col] - random.uniform(0.7, 0.9)).clip(1, 5).round(1)
            df.loc[mask, 'total_score'] = df.loc[mask, ['persona', 'problem_statement', 'user_need', 'solution_alignment']].mean(axis=1).round(1)
        
        # Add high disagreement between jurors for some teams
        disagreement_teams = random.sample(selected_teams, 40)
        for team_id in disagreement_teams:
            team_rows = df[df['team_id'] == team_id]
            if len(team_rows) == 2:
                # Get the two jury IDs
                jury1, jury2 = team_rows['jury_id'].values
                
                # Create disagreement in a random criterion
                disagreement_criterion = random.choice(['persona', 'problem_statement', 'user_need', 'solution_alignment'])
                
                # Increase one juror's score and decrease the other's
                df.loc[(df['team_id'] == team_id) & (df['jury_id'] == jury1), disagreement_criterion] = min(5.0, df.loc[(df['team_id'] == team_id) & (df['jury_id'] == jury1), disagreement_criterion].values[0] + random.uniform(1.0, 1.5))
                df.loc[(df['team_id'] == team_id) & (df['jury_id'] == jury2), disagreement_criterion] = max(1.0, df.loc[(df['team_id'] == team_id) & (df['jury_id'] == jury2), disagreement_criterion].values[0] - random.uniform(1.0, 1.5))
                
                # Update total scores
                df.loc[df['team_id'] == team_id, 'total_score'] = df.loc[df['team_id'] == team_id, ['persona', 'problem_statement', 'user_need', 'solution_alignment']].mean(axis=1).round(1)
        
        # Add missing evaluations for a few teams
        teams_missing_eval = random.sample(selected_teams, 5)
        for team_id in teams_missing_eval:
            # Remove one juror's evaluation
            jury_to_remove = df.loc[df['team_id'] == team_id, 'jury_id'].sample(1).values[0]
            df = df.drop(df[(df['team_id'] == team_id) & (df['jury_id'] == jury_to_remove)].index)
        
        # Add missing values for testing robustness
        random_indices = np.random.choice(df.index, size=15, replace=False)
        random_cols = np.random.choice(['persona', 'problem_statement', 'user_need', 'solution_alignment'], size=15, replace=True)
        for idx, col in zip(random_indices, random_cols):
            df.loc[idx, col] = np.nan
    
    return df

def generate_test_data():
    """Generate test datasets for both rounds and save to CSV files"""
    print("Generating synthetic data for testing...")
    
    # Generate smaller datasets for testing
    r1_data = generate_round1_data(num_teams=200, num_jury_pairs=20, add_anomalies=True)
    print(f"Generated Round 1 data: {len(r1_data)} rows")
    
    # Calculate team average scores for Round 1 (needed for Round 2 comparison)
    r1_avg = r1_data.groupby('team_id')['total_score'].mean().reset_index()
    
    # Generate Round 2 data with some correlation to Round 1
    r2_data = generate_round2_data(num_teams=50, num_jurors=15, previous_round_data=r1_data, add_anomalies=True)
    print(f"Generated Round 2 data: {len(r2_data)} rows")
    
    # Save datasets to CSV
    r1_data.to_csv("test_round1_scores.csv", index=False)
    r1_avg.to_csv("test_round1_team_avg_scores.csv", index=False)
    r2_data.to_csv("test_round2_scores.csv", index=False)
    
    print("Test data saved to CSV files:")
    print("- test_round1_scores.csv")
    print("- test_round1_team_avg_scores.csv")
    print("- test_round2_scores.csv")
    
    return r1_data, r1_avg, r2_data

if __name__ == "__main__":
    generate_test_data()
