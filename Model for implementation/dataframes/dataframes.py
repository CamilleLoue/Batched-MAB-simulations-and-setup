## creating a file to store rewards & ratios data

# import packages
import numpy as np
import pandas as pd

# rewards df
df_rewards = pd.DataFrame(columns=['rewards A','rewards B','rewards C','rewards D', 
                'rewards E', 'rewards F'])
df_rewards.to_csv('df_rewards.csv',index=False) 

# ratios df
df_ratios = pd.DataFrame(columns=['ratio A','ratio B','ratio C','ratio D', 
                'ratio E', 'ratio F'])
df_ratios.to_csv('df_ratios.csv',index=False)