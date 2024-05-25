import numpy as np
import torch.nn

from smbneuralnetwork import *
import random
from collections import deque
from preparation import JSPACE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: Sprawic, zeby uczenie zaczelo sie dopiero po duzej ilosci randomowych ruchow (zebysmy mieli zapelniony
#  troche self.memory) i porownac z uczeniem od poczatku. Czy wtedy epsilon decay aplikowac od poczatku czy od momentu rozpoczecia uczenia? Sprawdzic :)
class Mario:
	def __init__(self, save_directory):

		# Hyperparameters
		self.epsilon = 1.  # Nie ruszac
		self.gamma = 0.9
		self.sync_iteration = 10_000
		self.lr = 0.0003
		self.epsilon_decay = 0.99999925
		self.batch_size = 32

		# Networks
		self.mario = SMBNeuralNetwork()
		self.mario = self.mario.to(device=DEVICE)

		self.teacher = SMBNeuralNetwork(evaluation=True)
		self.teacher = self.teacher.to(device=DEVICE)

		# Loss and optimizer
		self.optimizer = torch.optim.Adam(self.mario.parameters(), lr=self.lr)
		self.loss = torch.nn.SmoothL1Loss()

		# Default misc
		self.iterations = 0

		# storage
		self.memory_storage = deque(maxlen=15_000)
		self.save_increment = 150_000
		self.save_directory = save_directory

	def choose_action(self, state):
		'''
		Epsilon greedy approach that encourages random exploration at the initial phases, focuses more on
		exploitation towards the end of the learning process.

		:param state: state that is to be evaluated by mario
		:return action: index from [0,n], decision made by mario as to which button to press
		'''

		if np.random.rand() < self.epsilon:
			# Random available action
			return np.random.randint(len(JSPACE))
		else:
			state = torch.FloatTensor(np.array(state)).cuda() if DEVICE == 'cuda' else torch.FloatTensor(state)
			state = state.unsqueeze(0)
			scores = self.mario_scores(state)
			# Highest ranked action
			return scores.argmax().item()

	def learn(self):
		'''
		Single iteration of the learning process.

		:param state: Initial state of Mario
		:param action: Action chosen by Mario in retrospect (current choice might be different than the one from the past)
		:param reward: Reward acquired by the action
		:param done: Boolean identifying if the game is done
		:param next_state: Result of doing that action in that state
		:return: loss
		'''

		# Sync'ing the networks after certain period
		if (self.iterations + 1) % self.sync_iteration == 0:
			self.teacher.load_state_dict(self.mario.state_dict())
			print('zmiana :)')

		if (self.iterations + 1) % self.save_increment == 0:
			self.save()

		self.optimizer.zero_grad()

		state, next_state, action, reward, done = self.recall()

		mario_score = self.generate_mario_td(state, action)
		teacher_score = self.generate_teacher_td(next_state, reward, done)

		loss = self.loss(mario_score, teacher_score)
		loss.backward()
		self.optimizer.step()

		self.iterations += 1
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon, 0.05)
		return loss.item()

	def remember_state(self, state, next_state, action, reward, done):
		'''
			Saves state of the game: (state, next_state, action, reward, done) into a bucket
			:return: None
		'''
		if DEVICE == 'cuda':
			state = torch.FloatTensor(np.array(state)).cuda()
			next_state = torch.FloatTensor(np.array(next_state)).cuda()
			action = torch.LongTensor([action]).cuda()
			reward = torch.DoubleTensor([reward]).cuda()
			done = torch.BoolTensor([done]).cuda()
		else:
			state = torch.FloatTensor(state)
			next_state = torch.FloatTensor(next_state)
			action = torch.LongTensor([action])
			reward = torch.DoubleTensor([reward])
			done = torch.BoolTensor([done])

		self.memory_storage.append((state, next_state, action, reward, done))

	def recall(self):
		"""
		Retrieve a batch of experiences from memory
		"""
		batch = random.sample(self.memory_storage, self.batch_size)
		state, next_state, action, reward, done = map(torch.stack, zip(*batch))
		return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

	def generate_mario_td(self, state,action):
		return self.mario_scores(state)[np.arange(self.batch_size), action]
	@torch.no_grad()
	def generate_teacher_td(self, next_state, reward, done):
		mario_proposal_Q = self.mario_scores(next_state)
		best_action = torch.argmax(mario_proposal_Q, axis=1)
		teacher_Q_raw = self.teacher_scores(next_state)[np.arange(self.batch_size), best_action]
		teacher_Q = (reward + (1 - done.float()) * self.gamma * teacher_Q_raw).float()
		return teacher_Q

	# Mario predicting action for a tensor/np.ndarray of states
	def mario_scores(self, state):
		return self.mario(state)

	@torch.no_grad()
	def teacher_scores(self, state):
		return self.teacher(state)

	def save(self):
		save_path = self.save_directory / f"mario_net_{int(self.iterations // self.save_increment)}.chkpt"
		torch.save(
			dict(
				model_mario=self.mario.state_dict(),
				model_teacher=self.teacher.state_dict(),
				epsilon=self.epsilon
			),
			save_path
		)
		print(f"Mario saved to {save_path} at step {self.iterations}")

	def load(self, load_path):
		if not load_path.exists():
			raise ValueError(f"{load_path} does not exist")

		ckp = torch.load(load_path, map_location=('cuda' if DEVICE=='cuda' else 'cpu'))
		epsilon = ckp.get('epsilon')
		state_dict_mario = ckp.get('model_mario')
		state_dict_teacher = ckp.get('model_teacher')

		self.mario.load_state_dict(state_dict_mario)
		self.teacher.load_state_dict(state_dict_teacher)
		self.epsilon = epsilon
