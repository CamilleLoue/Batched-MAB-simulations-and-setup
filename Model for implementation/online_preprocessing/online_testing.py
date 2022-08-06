## Online testing

# import packages
from online_training import select_combination, update_weights

# triggering a random combination of weights for the recommender sytsem to display based on the probabilities set by the ratios 
# of the day

update_weights(select_combination())
