from preparation import *
from smbneuralnetwork import SMBNeuralNetwork
from preparation import NUMBER_OF_EPISODES
import torch
import mario
from charts import learning_outcomes
import datetime
from pathlib import Path

save_directory = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_directory.mkdir(parents=True)

total_rewards = []
losses = []
avg_losses = []
steps = []
x_pos = []

Mario = mario.Mario(save_directory=save_directory)

env = generate_env()
env.reset()
next_state, reward, done, truncated, info = env.step(0)

print("Initial episode...")

state, _ = env.reset()

while not done:
	chosen_action = Mario.choose_action(state)
	new_state, reward, done, truncated, info = env.step(chosen_action)
	Mario.remember_state(state, new_state, chosen_action, reward, done)
	state = new_state
for i in range(NUMBER_OF_EPISODES):
	print("-----Episode:", i)
	done = False
	state, _ = env.reset()
	x_pos_max = 0
	total_reward = 0
	losses = []
	step = 0
	while not done:
		chosen_action = Mario.choose_action(state)
		new_state, reward, done, truncated, info = env.step(chosen_action)
		if reward < 0:
			reward *= 2
		Mario.remember_state(state, new_state, chosen_action, reward, done)
		loss = Mario.learn()
		total_reward += reward
		state = new_state
		step += 1
		if info['x_pos'] > x_pos_max:
			x_pos_max = info['x_pos']
		if loss is not None:
			losses.append(loss)

	print('Max x: ', x_pos_max)
	print(Mario.epsilon)
	avg_losses.append(sum(losses) / len(losses))
	total_rewards.append(total_reward)
	steps.append(step)
	x_pos.append(info['x_pos'])
Mario.save()
state, _ = env.reset()
while not done:
	chosen_action = Mario.choose_action(state)
	new_state, reward, done, truncated, info = env.step(chosen_action)
	print(info['x_pos'])
	state = new_state
learning_outcomes(total_rewards, avg_losses, x_pos, steps, 1)
