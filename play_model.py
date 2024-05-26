from preparation import *
import mario
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
#######################################################################
CHECKPOINT_PATH = 'checkpoints/05-26 18-03-33/mario_net_4.chkpt'
#######################################################################


env = generate_env()
env.reset()
done = False
state, _ = env.reset()
Mario = mario.Mario(save_directory=None)
checkpoint_path = Path(CHECKPOINT_PATH)

Mario.load(checkpoint_path)
Mario.epsilon = 0.0
actions = []
x_poss = []
states = []
while not done:
	chosen_action = Mario.choose_action(state)
	state, reward, done, truncated, info = env.step(chosen_action)
	x_poss.append(info['x_pos'])
	actions.append(chosen_action)
	if info['stage'] == 2:
		print('MARIO PRZESZEDŁ GRĘ !!!!!!!!!!!!!!')
		print()
	states.append(state)

matrix = (np.array(states[60][0]) + np.array(states[60][1])+np.array(states[60][2])+np.array(states[60][3]))/4 * 255
plt.imshow(matrix, cmap='gray', vmin=0, vmax=255)
plt.colorbar()  # Optional: to show a color scale bar
plt.show()