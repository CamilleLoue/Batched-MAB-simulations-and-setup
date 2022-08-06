## Visualizations to capture the development of the model

# import packages
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up the plot to track the history of the process

def plot_history():
    
    df = pd.read_csv('dataframes/df_results.csv')
    
    rewards_A = df["rewards A"]
    rewards_B = df["rewards B"]
    rewards_C = df["rewards C"]
    rewards_D = df["rewards E"]
    rewards_E = df["rewards D"]
    rewards_F = df["rewards F"]
    av_rewards_A = df["av_reward_A"]
    av_rewards_B = df["av_reward_B"]
    av_rewards_C = df["av_reward_C"]
    av_rewards_D = df["av_reward_D"]
    av_rewards_E = df["av_reward_E"]
    av_rewards_F = df["av_reward_F"]

    fig = plt.figure(figsize=[30,8])

    ax = fig.add_subplot(121)
    ax.plot(av_rewards_A,label="avg reward A")
    ax.plot(av_rewards_B,label="avg rewards B")
    ax.plot(av_rewards_C,label="avg rewards C")
    ax.plot(av_rewards_D,label="avg reward D")
    ax.plot(av_rewards_E,label="avg rewards E")
    ax.plot(av_rewards_F,label="avg rewards F")
    ax.legend()

    ax.set_title("Av Rewards")
    
    plt.savefig('visualization/av_rewards_history.png')
