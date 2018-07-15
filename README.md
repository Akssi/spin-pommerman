# spin-pommerman
Supreme Player Intelligence Network

## Agents ##

Agents are located in the SPINAgents folder. Each agent as its own file. When a new agent is created it must be added into the init file inside the SPINAgents folder and in the train script "competition_team_train.py".

### SPIN_0 ###

Agent *SPIN_0* is a copy of the Pommer exemple agent *Simple Agent*.

### SPIN_1 ###

Agent *SPIN_1* is base on Double dueling DQN. It's input for the Q network is a combination of the board 13x13 surface, the position as a 1 in a 13x13 array and the bomb strength map as a 13x13 array. We use a single layer of convolution and then flatten the input down to a vector for the fully connected layers.

### SPIN_2 ###

Agent *SPIN_2* is base on Double dueling DQN. It's input for the Q network is the board 13x13 surface flatten into a vector, the bomb blast and bomb time map flatten into vectors. We also add the position and attributes (bomb strength, ammo and can kick bomb) of the player.

## Train ##

To train agent on their own use *train_singleAgent.py*. Use -r to always render game, -c *path to checkpoint* to specify saved model to use and -a to specify the version of the agent you wnat to run (0, 1, 2).