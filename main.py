from preparation import *
from preparation import NUMBER_OF_EPISODES, JSPACE
import mario
from charts import learning_outcomes, learning_outcomes_learned
import datetime
from pathlib import Path
from copy import deepcopy


save_directory = Path('checkpoints') / datetime.datetime.now().strftime('%m-%d %H-%M-%S')
save_directory.mkdir(parents=True)

total_rewards = []
losses = []
avg_losses = []
steps = []
x_pos = []
clock = []
left_moves = []

total_rewards_learned = []
losses_learned = []
avg_losses_learned = []
steps_learned = []
x_pos_learned = []
clock_learned = []
left_moves_learned=[]

Mario = mario.Mario(save_directory=save_directory)
parameters = [Mario.epsilon, Mario.lr, Mario.gamma, Mario.epsilon_decay]

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
	left_counter = 0
	info['stage'] = 1
	while (not done) and info['stage'] == 1:

		chosen_action = Mario.choose_action(state)
		new_state, reward, done, truncated, info = env.step(chosen_action)

		# Warunek na ukonczenie poziomu
		if info['flag_get']:
			reward += 500

		if (chosen_action == 0 or chosen_action == 6) and len(JSPACE) > 5:
			left_counter += 1
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
	print('Epsilon: ', Mario.epsilon)


	avg_losses.append(sum(losses) / len(losses))
	total_rewards.append(total_reward)
	steps.append(step)
	clock.append(info['time'])
	x_pos.append(x_pos_max)
	left_moves.append(left_counter/len(losses))

	print('Left moves counter:', left_counter)
	if i % 100 == 0 or i == NUMBER_OF_EPISODES - 1:  # co ile puszczać nauczonego mario
		Mario_temp = deepcopy(Mario)
		done = False
		state, _ = env.reset()
		x_pos_max_learned = 0
		total_reward_learned = 0
		losses_learned = []
		step_learned = 0
		left_counter_learned = 0
		Mario_temp.epsilon = 0.0
		while not done:
			chosen_action = Mario_temp.choose_action(state)
			if chosen_action == 0 or chosen_action == 6:
				left_counter_learned += 1
			new_state, reward_learned, done, truncated, info = env.step(chosen_action)
			loss_learned = Mario_temp.learn()
			if loss_learned is not None:
				losses_learned.append(loss_learned)
			total_reward_learned += reward_learned
			state = new_state
			step_learned += 1
			if info['x_pos'] > x_pos_max_learned:
				x_pos_max_learned = info['x_pos']
		if len(losses_learned) == 0:
			avg_losses_learned.append(0)
		else:
			avg_losses_learned.append(sum(losses_learned) / len(losses_learned))
		total_rewards_learned.append(total_reward_learned)
		steps_learned.append(step_learned)
		clock_learned.append(info['time'])
		x_pos_learned.append(x_pos_max_learned)
		left_moves_learned.append(left_counter_learned / len(losses_learned))
		del Mario_temp.memory_storage
Mario.save()
state, _ = env.reset()

learning_outcomes(total_rewards, avg_losses, x_pos, steps, clock, left_moves, parameters, 100, save_directory)
learning_outcomes_learned(total_rewards_learned, avg_losses_learned, x_pos_learned, steps_learned, clock_learned, left_moves_learned, parameters, 1, save_directory)
