import smbneuralnetwork
import numpy as np
from smbneuralnetwork import *

class Mario():
	def __init__(self):
		'''
		Hiperparametry jako argument (chyba)
		'''

		self.epsilon = 1.
		self.gamma = 0.9

		# Networks
		self.mario = SMBNeuralNetwork()
		self.teacher = SMBNeuralNetwork(evaluation=True)

		# Loss and optimizer
		self.optimizer = torch.optim.Adam(self.mario.parameters(), lr=0.0001)
		self.loss = torch.nn.MSELoss()

		# Progress parameters
		self.iterations = 0
		self.sync_iteration = 1_000


	def choose_action(self, state):
		'''
		Epsilon greedy approach that encourages random exploration at the initial phases, focuses more on
		exploitation towards the end of the learning process.

		:param state: state that is to be evaluated by mario
		:return action: index from [0,4], decision made by mario as to which button to press
		'''

		if np.random.rand() < self.epsilon:
			# Random available action
			return np.random.randint(5)
		else:
			scores = self.mario_scores(state)
			# Highest ranked action
			return scores.argmax().item()

	def learn(self, state, action, reward, next_state, done):
		'''
		Single iteration of the learning process.

		:param state: Initial state of Mario
		:param action: Action chosen by Mario in retrospect (current choice might be different than the one from the past)
		:param reward: Reward acquired by the action
		:param done: Boolean identifying if the game is done
		:param next_state: Result of doing that action in that state
		:return:
		'''

		# Sync'ing the networks after certain period
		if self.iterations + 1 % self.sync_iteration == 0:
			print('###############ZMIENIAM##############')
			self.teacher.load_state_dict(self.mario.state_dict())

		self.optimizer.zero_grad()


		######### Implementacja ponizszego rozni sie w zaleznosci od zrodla :D
		# Q(s, a)
		mario_score = self.mario_scores(state)[0][action]

		# Q(s', a')
		mario_proposition = self.mario_scores(next_state)[0].argmax().item()
		teacher_score = self.teacher_scores(next_state)[0][mario_proposition]

		teacher_score = (reward + (1 - int(done)) * self.gamma * teacher_score).float()
		##########

		loss = self.loss(mario_score, teacher_score)
		loss.backward()
		self.optimizer.step()
		self.iterations += 1
		self.epsilon *= 0.99995

	def mario_scores(self, state):
		state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
		return self.mario(state)

	def teacher_scores(self, state):
		state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
		return self.teacher(state)

