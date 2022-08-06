# Batched-MAB-simulations-and-setup

#### Goal of the model
Finding which combination of parameters generates the highest CTR over time.

#### Naming conventions:
- **combination A** corresponds to the parameters combination {nlp:1,random:0,popular:0}
- **combination B** corresponds to the parameters combination {nlp:0,random:1,popular:0}
- **combination C** corresponds to the parameters combination {nlp:1,random:0,popular:1}
- **combination D** corresponds to the parameters combination {nlp:0.5,random:0.5,popular:0}
- **combination E** corresponds to the parameters combination {nlp:0,random:0.5,popular:0.5}
- **combination F** corresponds to the parameters combination {nlp:0.5,random:0,popular:0.5}

#### General concepts around the process:
During a period of time t, each combination will be triggered x times depending on the ratio attributed to it (the sum of the ratios equals to 1). 

Example: 
*ratio A = 0.5, ratio B = 0.1, ratio C = 0.1, ratio D = 0.1, ratio E = 0.1, ratio F = 0.1.*

This means that during time t, whenever the recommender system is called, there is 50% chance combination A will be triggered, 10% chance B, etc. 

This method enables every combination to be triggered at every period t and evaluated at the end of every period t. 
The performance of each combination is calculated in the form of rewards. The rewards here are the number of clicks for a certain combinaison divided by the number of times it has been triggered during a period t (CTR/period). The ratios are updated for every period t depending on the rewards (or CTR) from the previous period t-1. 
Over time, we will be able to depict the average CTR for each combination and understand which one has the highest.

#### The model in simulation
The aim here is to simulate the environment in which the model will be deployed later on. To do so, the clicks/period t are simulated using a Bernouilli distribution. The rewards are then collected at the end of each period to update the number of impressions for the following time period. In summary, two cycles are working together:
- simulation of the clicks every period
- MAB to attribute the number of impressions for the next period at every iteration.
Here, the MAB loops over the simulation of clicks for a number of iterations.

*NB: the time period t used in the following simulation is 1 day.*
