from preparation import *
import mario
from pathlib import Path

#######################################################################
CHECKPOINT_PATH = 'checkpoints/2024-05-22T01-54-15/mario_net_26.chkpt'
#######################################################################




env = generate_env()
env.reset()
next_state, reward, done, truncated, info = env.step(0)
state, _ = env.reset()
Mario = mario.Mario(save_directory=None)
checkpoint_path = Path(CHECKPOINT_PATH)

Mario.load(checkpoint_path)
Mario.epsilon = 0
while not done:
	chosen_action = Mario.choose_action(state)
	state, reward, done, truncated, info = env.step(chosen_action)