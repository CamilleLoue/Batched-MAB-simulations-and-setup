## Online training

# import packages
import numpy as np
import pandas as pd
import random
import rec_sys_weights
import sys
sys.path.append( '/home/jupyter/simulation' )
import python_entry
sys.path.append( '/home/jupyter/simulation' )
from training import daily_combination_ratio

# randomly administrating a combination of parameters when the rec system is triggered

def select_combination():
    
    df = pd.DataFrame(daily_combination_ratio())
    
    if random.random() <= python_entry.eps:
        return df[1][df[0].idxmax()]
    else:
        return random.choice(list(df.drop(df[0].idxmax())[1]))
    
# updating the weights of the rec system based on the randomly assigned combination 

def update_weights(combination):
    
    rec_sys_weights.weights.update({
    "popularity": combination[0],
    "nlp": combination[1],
    "random": combination[2]})
    
