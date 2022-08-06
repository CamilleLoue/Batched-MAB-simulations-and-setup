## Importing packages

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


### BERNOULLI CLICKS AND ENVIRONMENT CREATION ###

## Bernoulli clicks simulations

class BernoulliArm():
    def __init__(self, p):
        self.p = p
    
    # reward system based on Bernoulli: 1 or 0
    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


## Environment of the MAB models

class Env(object):

    def __init__(self, rewards):
        self.rewards = rewards


### MODELS ###

## Random model

class RandomAgent(object):

    def __init__(self, env, means, sigma, sigma_name, horizon, runs, batch_size):
        
        # environment set up
        self.env = env 
        
        # means
        self.means = means
        
        # sigma
        self.sigma = sigma
        
        # sigma name
        self.sigma_name = sigma_name
        
        # number of days of simulation
        self.iterations = horizon
        
        # list of arms to trigger during the period t
        self.arms = list(map(lambda mu: BernoulliArm(mu), self.means))
        
        # number of simulations
        self.runs = runs
        
        # batch size
        self.batch_size = batch_size
        
        # dataframe to gather the data from each run
        self.df_runs = pd.DataFrame(columns=['av_1', 'av_2', 'av_3','av_4','run average rewards',
                                             'cum_1', 'cum_2', 'cum_3','cum_4','run cum rewards'])
                

    def act(self):
        
        for run in range(self.runs):
            run = run+1
            
            # setting a dataframe to update at each iteration (each day)
            df = pd.DataFrame(columns=['rewards A','rewards B','rewards C','rewards D','rewards E', 'rewards F', 
                                        'total impressions','impression ratio','total clicks', 'rewards',
                                        'average rewards', 'cumulative rewards'])
        
            for i in range(self.iterations):

                total_imp = self.batch_size 

                A_ratio = 1/6
                B_ratio = 1/6
                C_ratio = 1/6
                D_ratio = 1/6
                E_ratio = 1/6
                F_ratio = 1/6


                imp_A = int(total_imp * A_ratio)
                imp_B = int(total_imp * B_ratio)
                imp_C = int(total_imp * C_ratio)
                imp_D = int(total_imp * D_ratio)
                imp_E = int(total_imp * E_ratio)
                imp_F = int(total_imp * F_ratio)

                imp = [imp_A,imp_B,imp_C,imp_D,imp_E,imp_F]

        # starting the click simulation of the day using the Bernouilli distributions set up previously

                def click(arm):
                    clicks = []

                    for i in range(imp[arm]):
                        self.means = abs(np.random.normal(self.means,self.sigma))
                        clicks.append(self.arms[arm].draw())

                    return sum(clicks)

        # calculating the rewards and the average rewards of time based on the number of clicks per day devided 
        # by the number of impressions

                click_A = click(0)
                click_B = click(1)
                click_C = click(2)
                click_D = click(3)
                click_E = click(4)
                click_F = click(5)

                reward_A = click_A / imp_A
                reward_B = click_B / imp_B
                reward_C = click_C / imp_C
                reward_D = click_D / imp_D
                reward_E = click_E / imp_E
                reward_F = click_F / imp_F

                total_clicks = click_A + click_B + click_C + click_D + click_E + click_F

                rewards = total_clicks / total_imp

                av_rewards = None if i < 1 else sum(df['total clicks']) / sum(df['total impressions'])

                cum_rewards = None if i < 1 else sum(df['rewards'])

        # updating the dataframe with the results & storing the results in a csv file

                data = {'rewards A':reward_A, 'rewards B':reward_B, 'rewards C':reward_C, 'rewards D':reward_D, 
                        'rewards E':reward_E, 'rewards F':reward_F, 'total impressions':total_imp,
                        'impression ratio':imp,'total clicks':total_clicks, 'rewards':rewards,
                        'average rewards':av_rewards, 'cumulative rewards':cum_rewards}
                df = df.append(data,ignore_index=True)


                self.df_runs = self.df_runs.append({'run average rewards':av_rewards,'run cum rewards':cum_rewards},ignore_index=True)

                
        # gathering and aggregating the data from each run
        
        a = self.df_runs['run average rewards'].to_list()
        chunk_size = len(a)//self.runs
        chunked_list = [a[i:i+chunk_size] for i in range(0, len(a), chunk_size)]
        av = pd.DataFrame(chunked_list).T.add_prefix('av_')
        
        c = self.df_runs['run cum rewards'].to_list()
        chunk_size = len(c)//self.runs
        chunked_list = [c[i:i+chunk_size] for i in range(0, len(c), chunk_size)]
        cum = pd.DataFrame(chunked_list).T.add_prefix('cum_')
        
        av['total runs average rewards'] = av.mean(axis=1)
        cum['total runs cum rewards'] = cum.mean(axis=1)
        total_runs = av.join(cum)
        
        total_runs.to_csv('df_random_{}.csv'.format(self.sigma_name), index=False)
        
        return total_runs



## Baseline model

class BaselineAgent(object):

    def __init__(self, env, means, sigma, sigma_name, horizon, eps, eps_name, runs, batch_size):
        
        # environment set up previously
        self.env = env 
        
        # means
        self.means = means
        
        # sigma
        self.sigma = sigma
        
        # sigma name
        self.sigma_name = sigma_name
        
        # number of days of simulation
        self.iterations = horizon
        
        # list of arms to trigger during the period t
        self.arms = list(map(lambda mu: BernoulliArm(mu), self.means))
        
        # epsilon value
        self.eps = eps
        
        # epsilon name
        self.eps_name = eps_name
        
        # number of simulations
        self.runs = runs
        
        # batch size
        self.batch_size = batch_size
        
        # dataframe to gather the data from each run
        self.df_runs = pd.DataFrame(columns=['av_1', 'av_2', 'av_3','av_4','run average rewards',
                                             'cum_1', 'cum_2', 'cum_3','cum_4','run cum rewards'])
                

    def act(self):
        
        for run in range(self.runs):
            run = run + 1
            
            
            # setting a dataframe to update at each iteration (each day)
            df = pd.DataFrame(columns=['rewards A','rewards B','rewards C','rewards D','rewards E', 'rewards F', 
                                        'total impressions','impression ratio','total clicks', 'rewards',
                                        'average rewards', 'cumulative rewards'])
        
            for i in range(self.iterations):

                total_imp = self.batch_size 

            # for the first iteration

                if i == 0:

            # equal impressions ratio for each combination for the first day

                    A_ratio = 1/6
                    B_ratio = 1/6
                    C_ratio = 1/6
                    D_ratio = 1/6
                    E_ratio = 1/6
                    F_ratio = 1/6

                    imp_A = int(total_imp * A_ratio)
                    imp_B = int(total_imp * B_ratio)
                    imp_C = int(total_imp * C_ratio)
                    imp_D = int(total_imp * D_ratio)
                    imp_E = int(total_imp * E_ratio)
                    imp_F = int(total_imp * F_ratio)

                    imp = [imp_A,imp_B,imp_C,imp_D,imp_E,imp_F]

            # starting the click simulation of the day using the Bernouilli distributions set up previously

                    def click(arm):
                        clicks = []

                        for i in range(imp[arm]):
                            self.means = abs(np.random.normal(self.means,self.sigma))
                            clicks.append(self.arms[arm].draw())

                        return sum(clicks)


                    click_A = click(0)
                    click_B = click(1)
                    click_C = click(2)
                    click_D = click(3)
                    click_E = click(4)
                    click_F = click(5)

            # calculating the rewards based on the number of clicks per day devided by the number of impressions

                    reward_A = click_A / imp_A
                    reward_B = click_B / imp_B
                    reward_C = click_C / imp_C
                    reward_D = click_D / imp_D
                    reward_E = click_E / imp_E
                    reward_F = click_F / imp_F

                    total_clicks = click_A + click_B + click_C + click_D + click_E + click_F

                    rewards = total_clicks / total_imp

            # updating the dataframe with the results & storing the results in a csv file

                    data = {'rewards A':reward_A, 'rewards B':reward_B, 'rewards C':reward_C, 'rewards D':reward_D, 
                            'rewards E':reward_E, 'rewards F':reward_F, 'total impressions':total_imp, 
                            'impression ratio':imp, 'total clicks':total_clicks, 'rewards':rewards}
                    df = df.append(data,ignore_index=True)
                    

            # for the following iterations

                else:

            # setting the impressions ratio depending on the rewards of the previous day (we should play with 
            # the ratios to see their impact on the results)

                    ratio_high = 1 - self.eps # for the combination with the highest reward 
                    ratio_low = self.eps/5 # for the other combinations

                    random_number = random.randint(1,6)

                    # administrating the ratio depending on the rewards of the previous line in the dataframe

                    A_ratio = ratio_high if df[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[i-1].idxmax() == 'rewards A' else ratio_low
                    B_ratio = ratio_high if df[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[i-1].idxmax() == 'rewards B' else ratio_low
                    C_ratio = ratio_high if df[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[i-1].idxmax() == 'rewards C' else ratio_low
                    D_ratio = ratio_high if df[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[i-1].idxmax() == 'rewards D' else ratio_low
                    E_ratio = ratio_high if df[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[i-1].idxmax() == 'rewards E' else ratio_low
                    F_ratio = ratio_high if df[['rewards A','rewards B','rewards C','rewards D','rewards E','rewards F']].iloc[i-1].idxmax() == 'rewards F' else ratio_low


                    imp_A = int(total_imp * A_ratio)
                    imp_B = int(total_imp * B_ratio)
                    imp_C = int(total_imp * C_ratio)
                    imp_D = int(total_imp * D_ratio)
                    imp_E = int(total_imp * E_ratio)
                    imp_F = int(total_imp * F_ratio)

                    imp = [imp_A,imp_B,imp_C,imp_D,imp_E,imp_F]

            # starting the click simulation of the day using the Bernouilli distributions set up previously

                    def click(arm):
                        clicks = []

                        for i in range(imp[arm]):
                            self.means = abs(np.random.normal(self.means,self.sigma))
                            clicks.append(self.arms[arm].draw())

                        return sum(clicks)

            # calculating the rewards and the average rewards of time based on the number of clicks per day devided 
            # by the number of impressions

                    click_A = click(0)
                    click_B = click(1)
                    click_C = click(2)
                    click_D = click(3)
                    click_E = click(4)
                    click_F = click(5)

                    reward_A = click_A / imp_A
                    reward_B = click_B / imp_B
                    reward_C = click_C / imp_C
                    reward_D = click_D / imp_D
                    reward_E = click_E / imp_E
                    reward_F = click_F / imp_F

                    total_clicks = click_A + click_B + click_C + click_D + click_E + click_F

                    rewards = total_clicks / total_imp

                    av_rewards = sum(df['rewards']) if i < 1 else sum(df['total clicks']) / sum(df['total impressions'])

                    cum_rewards = sum(df['rewards']) 


            # updating the dataframe with the results & storing the results in a csv file

                    data = {'rewards A':reward_A, 'rewards B':reward_B, 'rewards C':reward_C, 'rewards D':reward_D, 
                            'rewards E':reward_E, 'rewards F':reward_F, 'total impressions':total_imp,
                            'impression ratio':imp, 'total clicks':total_clicks, 'rewards':rewards,
                            'average rewards':av_rewards,'cumulative rewards':cum_rewards}
                    df = df.append(data,ignore_index=True)
                    
                    self.df_runs = self.df_runs.append({'run average rewards':av_rewards,'run cum rewards':cum_rewards},ignore_index=True)
        
        
        # gathering and aggregating the data from each run
        
        a = self.df_runs['run average rewards'].to_list()
        chunk_size = len(a)//self.runs
        chunked_list = [a[i:i+chunk_size] for i in range(0, len(a), chunk_size)]
        av = pd.DataFrame(chunked_list).T.add_prefix('av_')
        
        c = self.df_runs['run cum rewards'].to_list()
        chunk_size = len(c)//self.runs
        chunked_list = [c[i:i+chunk_size] for i in range(0, len(c), chunk_size)]
        cum = pd.DataFrame(chunked_list).T.add_prefix('cum_')
        
        av['total runs average rewards'] = av.mean(axis=1)
        cum['total runs cum rewards'] = cum.mean(axis=1)
        total_runs = av.join(cum)
        
        total_runs.to_csv('df_baseline_{}_{}.csv'.format(self.sigma_name,self.eps_name), index=False)
        
        return total_runs


## EGreedy model

class EGreedyAgent(object):

    def __init__(self, env, means, sigma, sigma_name, horizon, eps, eps_name, runs, batch_size):
        
        # environment set up previously
        self.env = env 
        
        # means
        self.means = means
        
        # sigma
        self.sigma = sigma
        
        # sigma name
        self.sigma_name = sigma_name
        
        # number of days of simulation
        self.iterations = horizon
        
        # list of arms to trigger during the period t
        self.arms = list(map(lambda mu: BernoulliArm(mu), self.means))
        
        # epsilon value
        self.eps = eps
        
        # epsilon name
        self.eps_name = eps_name
        
        # number of simulations
        self.runs = runs
                 
        # batch size
        self.batch_size = batch_size
        
        # dataframe to gather the data from each run
        self.df_runs = pd.DataFrame(columns=['av_1', 'av_2', 'av_3','av_4','run average rewards',
                                             'cum_1', 'cum_2', 'cum_3','cum_4','run cum rewards'])
                

    def act(self):
        
        for run in range(self.runs):
            run = run + 1
            
            
            # setting a dataframe to update at each iteration (each day)
            df = pd.DataFrame(columns=['rewards A','rewards B','rewards C','rewards D','rewards E', 'rewards F',
                                        'average rewards A','average rewards B','average rewards C','average rewards D','average rewards E', 'average rewards F',
                                        'total impressions','impression ratio','total clicks', 'rewards',
                                        'average rewards', 'cumulative rewards'])
        
            for i in range(self.iterations):

                total_imp = self.batch_size

            # for the first iteration

                if i == 0:

            # equal impressions ratio for each combination for the first day

                    A_ratio = 1/6
                    B_ratio = 1/6
                    C_ratio = 1/6
                    D_ratio = 1/6
                    E_ratio = 1/6
                    F_ratio = 1/6

                    imp_A = int(total_imp * A_ratio)
                    imp_B = int(total_imp * B_ratio)
                    imp_C = int(total_imp * C_ratio)
                    imp_D = int(total_imp * D_ratio)
                    imp_E = int(total_imp * E_ratio)
                    imp_F = int(total_imp * F_ratio)

                    imp = [imp_A,imp_B,imp_C,imp_D,imp_E,imp_F]

            # starting the click simulation of the day using the Bernouilli distributions set up previously

                    def click(arm):
                        clicks = []

                        for i in range(imp[arm]):
                            self.means = abs(np.random.normal(self.means,self.sigma))
                            clicks.append(self.arms[arm].draw())

                        return sum(clicks)


                    click_A = click(0)
                    click_B = click(1)
                    click_C = click(2)
                    click_D = click(3)
                    click_E = click(4)
                    click_F = click(5)

            # calculating the rewards based on the number of clicks per day devided by the number of impressions

                    reward_A = click_A / imp_A
                    reward_B = click_B / imp_B
                    reward_C = click_C / imp_C
                    reward_D = click_D / imp_D
                    reward_E = click_E / imp_E
                    reward_F = click_F / imp_F
                    
                    total_clicks = click_A + click_B + click_C + click_D + click_E + click_F
                
                    rewards = total_clicks / total_imp


            # updating the dataframe with the results & storing the results in a csv file

                    data =  {'rewards A':reward_A, 'rewards B':reward_B, 'rewards C':reward_C, 'rewards D':reward_D, 
                            'rewards E':reward_E, 'rewards F':reward_F, 'average rewards A':reward_A, 'average rewards B':reward_B, 
                            'average rewards C':reward_C, 'average rewards D':reward_D, 
                            'average rewards E':reward_E, 'average rewards F':reward_F,
                            'total impressions':total_imp, 
                            'impression ratio':imp, 'total clicks':total_clicks, 'rewards':rewards}
                    df = df.append(data,ignore_index=True)
                    

            # for the following iterations

                else:

            # setting the impressions ratio depending on the rewards of the previous day (we should play with 
            # the ratios to see their impact on the results)

                    ratio_high = 1 - self.eps # for the combination with the highest reward 
                    ratio_low = self.eps/5 # for the other combinations

                    random_number = random.randint(1,6)

                    # administrating the ratio depending on the rewards of the previous line in the dataframe

                    A_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards A' else ratio_low
                    B_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards B' else ratio_low
                    C_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards C' else ratio_low
                    D_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards D' else ratio_low
                    E_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards E' else ratio_low
                    F_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards F' else ratio_low

                    imp_A = int(total_imp * A_ratio)
                    imp_B = int(total_imp * B_ratio)
                    imp_C = int(total_imp * C_ratio)
                    imp_D = int(total_imp * D_ratio)
                    imp_E = int(total_imp * E_ratio)
                    imp_F = int(total_imp * F_ratio)

                    imp = [imp_A,imp_B,imp_C,imp_D,imp_E,imp_F]

            # starting the click simulation of the day using the Bernouilli distributions set up previously

                    def click(arm):
                        clicks = []

                        for i in range(imp[arm]):
                            self.means = abs(np.random.normal(self.means,self.sigma))
                            clicks.append(self.arms[arm].draw())

                        return sum(clicks)

            # calculating the rewards and the average rewards of time based on the number of clicks per day devided 
            # by the number of impressions

                    click_A = click(0)
                    click_B = click(1)
                    click_C = click(2)
                    click_D = click(3)
                    click_E = click(4)
                    click_F = click(5)

                    reward_A = click_A / imp_A
                    reward_B = click_B / imp_B
                    reward_C = click_C / imp_C
                    reward_D = click_D / imp_D
                    reward_E = click_E / imp_E
                    reward_F = click_F / imp_F
                    
                    av_reward_A = sum(df['rewards A']) / len(df['rewards A'])
                    av_reward_B = sum(df['rewards B']) / len(df['rewards B'])
                    av_reward_C = sum(df['rewards C']) / len(df['rewards C'])
                    av_reward_D = sum(df['rewards D']) / len(df['rewards D'])
                    av_reward_E = sum(df['rewards E']) / len(df['rewards E'])
                    av_reward_F = sum(df['rewards F']) / len(df['rewards F'])
                                
                    total_clicks = click_A + click_B + click_C + click_D + click_E + click_F
                                
                    rewards = total_clicks / total_imp
                
                    av_rewards = sum(df['rewards']) if i < 1 else sum(df['total clicks']) / sum(df['total impressions'])

                    cum_rewards = sum(df['rewards']) 


            # updating the dataframe with the results & storing the results in a csv file

                    data = {'rewards A':reward_A, 'rewards B':reward_B, 'rewards C':reward_C, 'rewards D':reward_D, 
                            'rewards E':reward_E, 'rewards F':reward_F, 'total impressions':total_imp, 'average rewards A':av_reward_A, 'average rewards B':av_reward_B, 
                            'average rewards C':av_reward_C, 'average rewards D':av_reward_D, 
                            'average rewards E':av_reward_E, 'average rewards F':av_reward_F,
                            'impression ratio':imp, 'total clicks':total_clicks, 'rewards':rewards,
                            'average rewards':av_rewards,'cumulative rewards':cum_rewards}
                    df = df.append(data,ignore_index=True)
                    
                    self.df_runs = self.df_runs.append({'run average rewards':av_rewards,'run cum rewards':cum_rewards},ignore_index=True)
        
        
        # gathering and aggregating the data from each run
        
        a = self.df_runs['run average rewards'].to_list()
        chunk_size = len(a)//self.runs
        chunked_list = [a[i:i+chunk_size] for i in range(0, len(a), chunk_size)]
        av = pd.DataFrame(chunked_list).T.add_prefix('av_')
        
        c = self.df_runs['run cum rewards'].to_list()
        chunk_size = len(c)//self.runs
        chunked_list = [c[i:i+chunk_size] for i in range(0, len(c), chunk_size)]
        cum = pd.DataFrame(chunked_list).T.add_prefix('cum_')
        
        av['total runs average rewards'] = av.mean(axis=1)
        cum['total runs cum rewards'] = cum.mean(axis=1)
        total_runs = av.join(cum)
        
        total_runs.to_csv('df_egreedy_{}_{}.csv'.format(self.sigma_name,self.eps_name), index=False)
        
        return total_runs


# Creating the EDecay MAB

class EDecayAgent(object):

    def __init__(self, env, means, sigma, sigma_name, horizon, eps, eps_name, runs, decay, decay_name, batch_size):
        
        # environment set up previously
        self.env = env 
        
        # means
        self.means = means
        
        # sigma
        self.sigma = sigma
        
        # sigma name
        self.sigma_name = sigma_name
        
        # number of days of simulation
        self.iterations = horizon
        
        # list of arms to trigger during the period t
        self.arms = list(map(lambda mu: BernoulliArm(mu), self.means))
        
        # epsilon value
        self.eps = eps
        
        # epsilon name
        self.eps_name = eps_name
        
        # decay
        self.decay = decay
        
        # decay name
        self.decay_name = decay_name
        
        # number of simulations
        self.runs = runs
        
        # batch size
        self.batch_size = batch_size
        
        # dataframe to gather the data from each run
        self.df_runs = pd.DataFrame(columns=['av_1', 'av_2', 'av_3','av_4','run average rewards',
                                             'cum_1', 'cum_2', 'cum_3','cum_4','run cum rewards'])
                

    def act(self):
        
        for run in range(self.runs):
            run = run + 1
            
            
            # setting a dataframe to update at each iteration (each day)
            df = pd.DataFrame(columns=['rewards A','rewards B','rewards C','rewards D','rewards E', 'rewards F',
                                        'average rewards A','average rewards B','average rewards C','average rewards D','average rewards E', 'average rewards F',
                                        'total impressions','impression ratio','total clicks', 'rewards',
                                        'average rewards', 'cumulative rewards'])
        
            for i in range(self.iterations):

                total_imp = self.batch_size 
                
                self.eps *= self.decay

            # for the first iteration

                if i == 0:

            # equal impressions ratio for each combination for the first day

                    A_ratio = 1/6
                    B_ratio = 1/6
                    C_ratio = 1/6
                    D_ratio = 1/6
                    E_ratio = 1/6
                    F_ratio = 1/6

                    imp_A = int(total_imp * A_ratio)
                    imp_B = int(total_imp * B_ratio)
                    imp_C = int(total_imp * C_ratio)
                    imp_D = int(total_imp * D_ratio)
                    imp_E = int(total_imp * E_ratio)
                    imp_F = int(total_imp * F_ratio)

                    imp = [imp_A,imp_B,imp_C,imp_D,imp_E,imp_F]

            # starting the click simulation of the day using the Bernouilli distributions set up previously

                    def click(arm):
                        clicks = []

                        for i in range(imp[arm]):
                            self.means = abs(np.random.normal(self.means,self.sigma))
                            clicks.append(self.arms[arm].draw())

                        return sum(clicks)


                    click_A = click(0)
                    click_B = click(1)
                    click_C = click(2)
                    click_D = click(3)
                    click_E = click(4)
                    click_F = click(5)

            # calculating the rewards based on the number of clicks per day devided by the number of impressions

                    reward_A = click_A / imp_A
                    reward_B = click_B / imp_B
                    reward_C = click_C / imp_C
                    reward_D = click_D / imp_D
                    reward_E = click_E / imp_E
                    reward_F = click_F / imp_F
                    
                    total_clicks = click_A + click_B + click_C + click_D + click_E + click_F
                
                    rewards = total_clicks / total_imp


            # updating the dataframe with the results & storing the results in a csv file

                    data =  {'rewards A':reward_A, 'rewards B':reward_B, 'rewards C':reward_C, 'rewards D':reward_D, 
                            'rewards E':reward_E, 'rewards F':reward_F, 'average rewards A':reward_A, 'average rewards B':reward_B, 
                            'average rewards C':reward_C, 'average rewards D':reward_D, 
                            'average rewards E':reward_E, 'average rewards F':reward_F,
                            'total impressions':total_imp, 
                            'impression ratio':imp, 'total clicks':total_clicks, 'rewards':rewards}
                    df = df.append(data,ignore_index=True)
                    

            # for the following iterations

                else:

            # setting the impressions ratio depending on the rewards of the previous day (we should play with 
            # the ratios to see their impact on the results)

                    ratio_high = 1 - self.eps # for the combination with the highest reward 
                    ratio_low = self.eps/5 # for the other combinations

                    random_number = random.randint(1,6)

                    # administrating the ratio depending on the rewards of the previous line in the dataframe

                    A_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards A' else ratio_low
                    B_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards B' else ratio_low
                    C_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards C' else ratio_low
                    D_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards D' else ratio_low
                    E_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards E' else ratio_low
                    F_ratio = ratio_high if df[['average rewards A','average rewards B','average rewards C',
                                                 'average rewards D','average rewards E','average rewards F']].iloc[i-1].idxmax() == 'average rewards F' else ratio_low

                    imp_A = int(total_imp * A_ratio)
                    imp_B = int(total_imp * B_ratio)
                    imp_C = int(total_imp * C_ratio)
                    imp_D = int(total_imp * D_ratio)
                    imp_E = int(total_imp * E_ratio)
                    imp_F = int(total_imp * F_ratio)

                    imp = [imp_A,imp_B,imp_C,imp_D,imp_E,imp_F]

            # starting the click simulation of the day using the Bernouilli distributions set up previously

                    def click(arm):
                        clicks = []

                        for i in range(imp[arm]):
                            self.means = abs(np.random.normal(self.means,self.sigma))
                            clicks.append(self.arms[arm].draw())

                        return sum(clicks)

            # calculating the rewards and the average rewards of time based on the number of clicks per day devided 
            # by the number of impressions

                    click_A = click(0)
                    click_B = click(1)
                    click_C = click(2)
                    click_D = click(3)
                    click_E = click(4)
                    click_F = click(5)

                    reward_A = 0 if ZeroDivisionError else click_A / imp_A
                    reward_B = 0 if ZeroDivisionError else click_B / imp_B
                    reward_C = 0 if ZeroDivisionError else click_C / imp_C
                    reward_D = 0 if ZeroDivisionError else click_D / imp_D
                    reward_E = 0 if ZeroDivisionError else click_E / imp_E
                    reward_F = 0 if ZeroDivisionError else click_F / imp_F
                    
                    av_reward_A = sum(df['rewards A']) / len(df['rewards A'])
                    av_reward_B = sum(df['rewards B']) / len(df['rewards B'])
                    av_reward_C = sum(df['rewards C']) / len(df['rewards C'])
                    av_reward_D = sum(df['rewards D']) / len(df['rewards D'])
                    av_reward_E = sum(df['rewards E']) / len(df['rewards E'])
                    av_reward_F = sum(df['rewards F']) / len(df['rewards F'])
                                
                    total_clicks = click_A + click_B + click_C + click_D + click_E + click_F
                                
                    rewards = total_clicks / total_imp
                
                    av_rewards = sum(df['rewards']) if i < 1 else sum(df['total clicks']) / sum(df['total impressions'])

                    cum_rewards = sum(df['rewards']) 


            # updating the dataframe with the results & storing the results in a csv file

                    data = {'rewards A':reward_A, 'rewards B':reward_B, 'rewards C':reward_C, 'rewards D':reward_D, 
                            'rewards E':reward_E, 'rewards F':reward_F, 'total impressions':total_imp, 'average rewards A':av_reward_A, 'average rewards B':av_reward_B, 
                            'average rewards C':av_reward_C, 'average rewards D':av_reward_D, 
                            'average rewards E':av_reward_E, 'average rewards F':av_reward_F,
                            'impression ratio':imp, 'total clicks':total_clicks, 'rewards':rewards,
                            'average rewards':av_rewards,'cumulative rewards':cum_rewards}
                    df = df.append(data,ignore_index=True)
                    
                    self.df_runs = self.df_runs.append({'run average rewards':av_rewards,'run cum rewards':cum_rewards},ignore_index=True)
        
        
        # gathering and aggregating the data from each run
        
        a = self.df_runs['run average rewards'].to_list()
        chunk_size = len(a)//self.runs
        chunked_list = [a[i:i+chunk_size] for i in range(0, len(a), chunk_size)]
        av = pd.DataFrame(chunked_list).T.add_prefix('av_')
        
        c = self.df_runs['run cum rewards'].to_list()
        chunk_size = len(c)//self.runs
        chunked_list = [c[i:i+chunk_size] for i in range(0, len(c), chunk_size)]
        cum = pd.DataFrame(chunked_list).T.add_prefix('cum_')
        
        av['total runs average rewards'] = av.mean(axis=1)
        cum['total runs cum rewards'] = cum.mean(axis=1)
        total_runs = av.join(cum)
        
        total_runs.to_csv('df_edecay_{}_{}_{}.csv'.format(self.sigma_name,self.eps_name,self.decay_name), index=False)
        
        return total_runs


