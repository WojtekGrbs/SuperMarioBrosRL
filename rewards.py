def calculate_reward_waste_of_time(info, previous_info):
	reward = 0

	# 1. Nagroda za przemieszczanie się w prawo
	#if 2*(info['x_pos'] - previous_info['x_pos'])>-100:
	#	reward += 2*(info['x_pos'] - previous_info['x_pos'])
	
	# 2. Nagroda za ukończenie poziomu 
	# Uwzgledniona zostala w preparations.py, domyslnie zawsze jest wlaczona w postaci reward += 500

	# 3. Nagroda za unikanie przeszkód (przeskoczenie rury)
	#if info['status'] == 'tall' and previous_info['status'] == 'small':
	#    reward += 50

	# 4. Kara za stanie w miejscu 
	if info['x_pos'] == previous_info['x_pos']:
		reward -= 1

	# 5. kara za czas
	if info['time'] < previous_info['time']:
		reward -= 2

	return reward
