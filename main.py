import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


def load_season_data(filepath):
    """Load and preprocess season stats from teamStats.csv."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded teamStats.csv with {len(df)} rows")
        df.columns = df.columns.str.strip()
        df['Min'] = df['Min'].str.replace(',', '').astype(float)
        numeric_cols = ['MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 
                        'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        features = ['Player', 'Gls', 'Ast', 'CrdY', 'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', '90s']
        return df[features]
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        raise
    except Exception as e:
        print(f"Error loading teamStats.csv: {str(e)}")
        raise

def load_game_data(filepaths):
    """Load and preprocess game data from multiple CSVs."""
    all_games = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {filepath} with {len(df)} rows")
            df.columns = df.columns.str.strip()
            numeric_cols = ['Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 'SoT', 'CrdY', 'CrdR', 
                            'Touches', 'Tkl', 'Int', 'Blocks', 'xG', 'npxG', 'xAG', 'SCA', 'GCA', 
                            'Cmp', 'Att', 'Cmp%', 'PrgP', 'Carries', 'PrgC', 'Att', 'Succ']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            all_games.append(df)
        except FileNotFoundError:
            print(f"Error: Could not find {filepath}")
            raise
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            raise
    game_df = pd.concat(all_games, ignore_index=True)
    print(f"Combined game data with {len(game_df)} rows")
    return game_df

def prepare_features(season_df, game_df):
    """Engineer features for prediction."""
    recent_stats = game_df.groupby('Player').agg({
        'Gls': 'mean',
        'Ast': 'mean',
        'CrdY': 'mean',
        'Sh': 'mean',
        'SoT': 'mean',
        'xG': 'mean',
        'xAG': 'mean',
        'SCA': 'mean',
        'GCA': 'mean',
        'Tkl': 'mean',
        'Int': 'mean',
        'Min': 'sum'
    }).reset_index()
    recent_stats.columns = ['Player', 'avg_Gls_recent', 'avg_Ast_recent', 'avg_CrdY_recent', 
                           'avg_Sh_recent', 'avg_SoT_recent', 'avg_xG_recent', 'avg_xAG_recent', 
                           'avg_SCA_recent', 'avg_GCA_recent', 'avg_Tkl_recent', 'avg_Int_recent', 
                           'total_Min_recent']
    print(f"Recent stats computed for {len(recent_stats)} players")
    
    merged_df = pd.merge(season_df, recent_stats, on='Player', how='left')
    print(f"Merged dataset with {len(merged_df)} rows")
    
    for col in recent_stats.columns[1:]:
        merged_df[col] = merged_df[col].fillna(0)
    
    merged_df['Gls_90'] = merged_df.apply(lambda x: x['Gls'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    merged_df['Ast_90'] = merged_df.apply(lambda x: x['Ast'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    merged_df['CrdY_90'] = merged_df.apply(lambda x: x['CrdY'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    merged_df['xG_90'] = merged_df.apply(lambda x: x['xG'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    merged_df['npxG_90'] = merged_df.apply(lambda x: x['npxG'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    merged_df['xAG_90'] = merged_df.apply(lambda x: x['xAG'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    merged_df['PrgC_90'] = merged_df.apply(lambda x: x['PrgC'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    merged_df['PrgP_90'] = merged_df.apply(lambda x: x['PrgP'] / x['90s'] if x['90s'] > 0 else 0, axis=1)
    
    nan_cols = merged_df.columns[merged_df.isna().any()].tolist()
    if nan_cols:
        print(f"Warning: NaN values found in columns: {nan_cols}")
        print(merged_df[nan_cols].isna().sum())
    
    feature_cols = {
        'goal': ['Gls_90', 'xG_90', 'npxG_90', 'PrgC_90', 'avg_Gls_recent', 'avg_Sh_recent', 
                 'avg_SoT_recent', 'avg_xG_recent'],
        'assist': ['Ast_90', 'xAG_90', 'PrgC_90', 'PrgP_90', 'avg_Ast_recent', 'avg_xAG_recent', 
                   'avg_SCA_recent', 'avg_GCA_recent'],
        'yellow_card': ['CrdY_90', 'avg_CrdY_recent', 'avg_Tkl_recent', 'avg_Int_recent']
    }
    return merged_df, feature_cols

def create_training_data(game_df, feature_df, feature_cols, target_col, target_name):
    """Create training data for a specific target."""
    training_data = []
    for _, row in game_df.iterrows():
        player_features = feature_df[feature_df['Player'] == row['Player']]
        if player_features.empty:
            continue
        features = player_features[feature_cols].iloc[0].to_dict()
        features[target_name] = 1 if row[target_col] > 0 else 0
        training_data.append(features)
    
    training_df = pd.DataFrame(training_data)
    print(f"Training data for {target_name} created with {len(training_df)} instances")
    return training_df

def predict_contributions(season_filepath, game_filepaths):
    """Predict players most likely to score, assist, or receive a yellow card."""
    season_df = load_season_data(season_filepath)
    game_df = load_game_data(game_filepaths)
    feature_df, feature_cols_dict = prepare_features(season_df, game_df)
    
    predictions = {}
    for target, feature_cols in feature_cols_dict.items():
        target_col = {'goal': 'Gls', 'assist': 'Ast', 'yellow_card': 'CrdY'}[target]
        target_name = f"{target}_outcome"
        
        training_df = create_training_data(game_df, feature_df, feature_cols, target_col, target_name)
        
        if training_df[target_name].sum() == 0:
            print(f"Warning: No {target}s in training data. Falling back to stat-based ranking.")
            stat_col = {'goal': 'xG_90', 'assist': 'xAG_90', 'yellow_card': 'CrdY_90'}[target]
            recent_stat = {'goal': 'avg_xG_recent', 'assist': 'avg_xAG_recent', 'yellow_card': 'avg_CrdY_recent'}[target]
            pred_df = feature_df[['Player', stat_col, recent_stat]].copy()
            pred_df['Probability'] = pred_df[stat_col] / pred_df[stat_col].max()
            pred_df = pred_df.sort_values('Probability', ascending=False)
            predictions[target] = pred_df
            print(f"\nTop Player for {target.capitalize()} (Stat-based):")
            top_player = pred_df.iloc[0]
            print(f"Player: {top_player['Player']}")
            print(f"Normalized Probability: {top_player['Probability']:.3f}")
            print(f"Season {stat_col}: {top_player[stat_col]:.2f}")
            print(f"Recent {recent_stat}: {top_player[recent_stat]:.2f}")
            continue
        
        X = training_df[feature_cols]
        y = training_df[target_name]
        
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        print(f"Imputed {np.isnan(X).sum().sum()} NaN values for {target}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        if len(y.unique()) > 1:
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            print(f"Training AUC for {target}: {auc:.2f}")
        else:
            print(f"Warning: Only one class for {target}. AUC not computed.")
        
        X_pred = imputer.transform(feature_df[feature_cols])
        X_pred_scaled = scaler.transform(X_pred)
        probabilities = model.predict_proba(X_pred_scaled)[:, 1]
        
        pred_df = pd.DataFrame({
            'Player': feature_df['Player'],
            'Probability': probabilities,
            'Season_Stat': feature_df[feature_cols[0]],  # e.g., Gls_90, Ast_90, CrdY_90
            'Recent_Stat': feature_df[f'avg_{target_col}_recent']
        })
        pred_df = pred_df.sort_values('Probability', ascending=False)
        predictions[target] = pred_df
        
        print(f"\nTop Player for {target.capitalize()}:")
        top_player = pred_df.iloc[0]
        print(f"Player: {top_player['Player']}")
        print(f"Probability: {top_player['Probability']:.3f}")
        print(f"Season {feature_cols[0]}: {top_player['Season_Stat']:.2f}")
        print(f"Recent Avg {target_col}: {top_player['Recent_Stat']:.2f}")
        
        print(f"\nTop 5 Players for {target.capitalize()}:")
        print(pred_df[['Player', 'Probability', 'Season_Stat', 'Recent_Stat']].head(5).to_string(index=False))
    
    return predictions

if __name__ == "__main__":
    season_filepath = "data/manUtd/teamStats.csv"
    game_filepaths = [
        "data/manUtd/games/april_13_Newcastle.csv",
        "data/manUtd/games/april1_Nottingham.csv",
        "data/manUtd/games/april6_ManCity.csv",
        "data/manUtd/games/april10_Lyon.csv",
        "data/manUtd/games/march_16_Leicester.csv",
        "data/manUtd/games/march13_RealSoc.csv",
        "data/manUtd/games/march9_arsenal.csv",
        "data/manUtd/games/march6_RealSoc.csv",
       # "data/manUtd/games/march2_Fulham.csv", #Comment out cos of bad data format since this game went to pens
        "data/manUtd/games/feb26_ipswich.csv"
    ]
    

    predict_contributions(season_filepath, game_filepaths)