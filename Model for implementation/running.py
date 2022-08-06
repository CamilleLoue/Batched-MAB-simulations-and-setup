## Running the MAB model

# import packages
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from training import rewards, ratios, storing_results, daily_combination_ratio
from visualization.plot_script import plot_history

# running the model daily

def run():
    rewards()
    ratios()
    storing_results()
    daily_combination_ratio()
    plot_history()
    
run()