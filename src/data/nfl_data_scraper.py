import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_nfl_data(season):
    """
    Fetch NFL game data for a given season from a public API
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {
        "limit": 1000,
        "dates": f"{season}0901-{season+1}0228"  # Covers regular season and playoffs
    }
    response = requests.get(url, params=params)
    return response.json()

def process_game_data(game):
    """
    Extract relevant information from a single game
    """
    home_team = game['competitions'][0]['competitors'][0]['team']['abbreviation']
    away_team = game['competitions'][0]['competitors'][1]['team']['abbreviation']
    
    home_score = int(game['competitions'][0]['competitors'][0]['score'])
    away_score = int(game['competitions'][0]['competitors'][1]['score'])
    
    date = datetime.strptime(game['date'], "%Y-%m-%dT%H:%MZ")
    
    # Extract additional stats if available
    home_stats = game['competitions'][0]['competitors'][0].get('statistics', [])
    away_stats = game['competitions'][0]['competitors'][1].get('statistics', [])
    
    # Function to safely extract stat value
    def get_stat_value(stats, stat_name):
        stat = next((s for s in stats if s['name'] == stat_name), None)
        return int(stat['displayValue']) if stat and stat['displayValue'].isdigit() else 0

    home_total_yards = get_stat_value(home_stats, 'totalYards')
    away_total_yards = get_stat_value(away_stats, 'totalYards')
    home_turnovers = get_stat_value(home_stats, 'turnovers')
    away_turnovers = get_stat_value(away_stats, 'turnovers')
    
    return {
        'date': date,
        'home_team': home_team,
        'away_team': away_team,
        'home_score': home_score,
        'away_score': away_score,
        'home_total_yards': home_total_yards,
        'away_total_yards': away_total_yards,
        'home_turnovers': home_turnovers,
        'away_turnovers': away_turnovers
    }

def scrape_nfl_data(start_season, end_season):
    """
    Scrape NFL data for multiple seasons and return a DataFrame
    """
    all_games = []
    
    for season in range(start_season, end_season + 1):
        print(f"Fetching data for {season} season...")
        data = fetch_nfl_data(season)
        
        for event in data['events']:
            all_games.append(process_game_data(event))
    
    df = pd.DataFrame(all_games)
    
    # Sort by date
    df = df.sort_values('date')
    
    # Add derived features
    df['winner'] = (df['home_score'] > df['away_score']).astype(int)
    df['total_points'] = df['home_score'] + df['away_score']
    df['point_difference'] = df['home_score'] - df['away_score']
    
    # Calculate rolling averages
    for team_type in ['home', 'away']:
        for stat in ['score', 'total_yards', 'turnovers']:
            col_name = f'{team_type}_{stat}'
            df[f'{col_name}_rolling_avg'] = df.groupby(f'{team_type}_team')[col_name].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # Calculate win streaks
    df['home_win_streak'] = df.groupby('home_team')['winner'].transform(lambda x: (x == 1).cumsum())
    df['away_win_streak'] = df.groupby('away_team')['winner'].transform(lambda x: (x == 0).cumsum())
    
    return df

def main():
    # Scrape data for the last 5 seasons
    current_year = datetime.now().year
    start_season = current_year - 5
    df = scrape_nfl_data(start_season, current_year)
    
    # Save to CSV
    output_file = f'nfl_data_{start_season}_{current_year}.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
