import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from preparation import *
import mario
from pathlib import Path
import imageio
from PIL import Image

FILENAME = 'mario_nauczony_w_pelni.gif'
CHECKPOINT_PATH = 'checkpoints/05-29 16-51-08/mario_taught_198200.chkpt'

env = generate_env()
env.reset()
done = False
state, _ = env.reset()
state, reward, done, truncated, info = env.step(0)
Mario = mario.Mario(save_directory=None)
checkpoint_path = Path(CHECKPOINT_PATH)

Mario.load(checkpoint_path)
Mario.epsilon = 0.0

actions = []

while (not done) and (not info['flag_get']):
    chosen_action = Mario.choose_action(state)
    actions.append(chosen_action)
    state, reward, done, truncated, info = env.step(chosen_action)

env = gym.make('SuperMarioBros-v0', render_mode='rgb_array', apply_api_compatibility=True)
env = FrameSkippingWrapper(env, 4)
env = JoypadSpace(env, JSPACE)
state, _ = env.reset()
state, reward, done, truncated, info = env.step(0)
frames = []
print(actions)
for action in actions:
    frames.append(np.array(state))
    state, _, done, _, _ = env.step(action)
print(frames[0].shape)

env.close()
save_path = './' + FILENAME
images = [Image.fromarray(rgb_array) for rgb_array in frames]
imageio.mimsave(save_path, images, format='GIF', duration=0.05, loop=0)