## MAB training model

# importing packages
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import python_entry


# calculating the rewards of the day (CTR of the day) 

def rewards():

    daily_df = pd.read_csv('mock_data/mock_data.csv') # UPDATE FILE PATH WHEN DEPLOYING

    reward_A = daily_df['clicks A'].sum() / daily_df['impressions A'].sum()
    reward_B = daily_df['clicks B'].sum() / daily_df['impressions B'].sum()
    reward_C = daily_df['clicks C'].sum() / daily_df['impressions C'].sum()
    reward_D = daily_df['clicks D'].sum() / daily_df['impressions D'].sum() 
    reward_E = daily_df['clicks E'].sum() / daily_df['impressions E'].sum()
    reward_F = daily_df['clicks F'].sum() / daily_df['impressions F'].sum()
    
    rewards = pd.Series([reward_A, reward_B, reward_C, reward_D, reward_E, reward_F],index=["rewards A", "rewards B", "rewards C", "rewards D", "rewards E", "rewards F"])

    # updating the dataframe with the results & storing the results in a csv file

    df_rewards = pd.read_csv('dataframes/df_rewards.csv')
    
    concat = pd.concat([df_rewards, rewards.to_frame().T], ignore_index=True)
    
    concat.to_csv('dataframes/df_rewards.csv',index=False)
    

# updating the ratios for the following day

def ratios():
    
    eps = python_entry.eps
    
    ratio_high = (1 - eps) # for the combination with the highest reward 
    ratio_low = eps/5 # for the other combinations

    # administrating the ratio depending on the rewards of the previous line in the dataframe

    df_rewards = pd.read_csv('dataframes/df_rewards.csv')
    
    ratio_A = ratio_high if df_rewards[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[-1].idxmax() == 'rewards A' else ratio_low
    ratio_B = ratio_high if df_rewards[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[-1].idxmax() == 'rewards B' else ratio_low
    ratio_C = ratio_high if df_rewards[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[-1].idxmax() == 'rewards C' else ratio_low
    ratio_D = ratio_high if df_rewards[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[-1].idxmax() == 'rewards D' else ratio_low
    ratio_E = ratio_high if df_rewards[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[-1].idxmax() == 'rewards E' else ratio_low
    ratio_F = ratio_high if df_rewards[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[-1].idxmax() == 'rewards F' else ratio_low

    ratios = pd.Series([ratio_A, ratio_B, ratio_C, ratio_D, ratio_E, ratio_F],index=["ratio A", "ratio B", "ratio C", "ratio D", "ratio E", "ratio F"])
    
    # updating the dataframe with the results & storing the results in a csv file
    
    df_ratios = pd.read_csv('dataframes/df_ratios.csv')
    
    concat = pd.concat([df_ratios, ratios.to_frame().T], ignore_index=True)
    
    concat.to_csv('dataframes/df_ratios.csv',index=False)
    
    
# storing rewards and ratios in a commun file and calculating the average rewards over time

def storing_results():

    df_ratios = pd.read_csv('dataframes/df_ratios.csv')
    df_rewards = pd.read_csv('dataframes/df_rewards.csv')

    df = pd.merge(df_ratios.shift(1), df_rewards, left_index=True, right_index=True)
    
    df['av_reward_A'] = df['rewards A'].expanding().mean()
    df['av_reward_B'] = df['rewards B'].expanding().mean()
    df['av_reward_C'] = df['rewards C'].expanding().mean()
    df['av_reward_D'] = df['rewards D'].expanding().mean()
    df['av_reward_E'] = df['rewards E'].expanding().mean()
    df['av_reward_F'] = df['rewards F'].expanding().mean()
    
    df.to_csv('dataframes/df_results.csv',index=False)
    

# getting the combination ratio for the following day based on the results of the day

def daily_combination_ratio():
    
    df_ratios = pd.read_csv('dataframes/df_ratios.csv')
    
    A = df_ratios['ratio A'].iloc[-1]
    B = df_ratios['ratio B'].iloc[-1]
    C = df_ratios['ratio C'].iloc[-1]
    D = df_ratios['ratio D'].iloc[-1]
    E = df_ratios['ratio E'].iloc[-1]
    F = df_ratios['ratio F'].iloc[-1]
    
    ratios = [A,B,C,D,E,F]

    combinations = [python_entry.combination_A, python_entry.combination_B, python_entry.combination_C, 
                    python_entry.combination_D, python_entry.combination_E, python_entry.combination_F]
    
    r_c = list([key, value] for key, value in zip(ratios,combinations))
    
    return r_c # UPDATE WHERE THE RATIO IS RETURNED