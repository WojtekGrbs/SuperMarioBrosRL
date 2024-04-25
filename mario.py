import smbneuralnetwork
import numpy as np
from smbneuralnetwork import *
from preparation import NUMBER_OF_EPISODES
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: Sprawic, zeby uczenie zaczelo sie dopiero po duzej ilosci randomowych ruchow (zebysmy mieli zapelniony
#  troche self.memory) i porownac z uczeniem od poczatku. Czy wtedy epsilon decay aplikowac od poczatku czy od momentu rozpoczecia uczenia? Sprawdzic :)
class Mario():
	def __init__(self, pretrained_model=None):

		# Hyperparameters
		self.epsilon = 1.
		self.gamma = 0.9
		self.sync_iteration = 5_000
		self.lr = 0.0005
		self.epsilon_decay = 0.99995

		# Networks
		self.mario = SMBNeuralNetwork().cuda()
		self.teacher = SMBNeuralNetwork(evaluation=True).cuda()

		# Loss and optimizer
		self.optimizer = torch.optim.Adam(self.mario.parameters(), lr=self.lr)
		self.loss = torch.nn.MSELoss()

		# Default misc
		self.iterations = 0
		self.training_flag = True

		self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))

		# If pretrained model is being uploaded:
		if pretrained_model is not None:
			self.training_flag = False
			self.mario = pretrained_model

	def choose_action(self, state):
		'''
		Epsilon greedy approach that encourages random exploration at the initial phases, focuses more on
		exploitation towards the end of the learning process.

		:param state: state that is to be evaluated by mario
		:return action: index from [0,4], decision made by mario as to which button to press
		'''

		if (np.random.rand() < self.epsilon) and self.training_flag:
			# Random available action
			return np.random.randint(5)
		else:
			scores = self.mario_scores(state)
			# Highest ranked action
			return scores.argmax().item()


# TODO: Sprawic, zeby uczenie bylo batchowe z rzeczy zapisywanych w self.memory
	def learn(self, state, action, reward, next_state, done):
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
		self.epsilon *= self.epsilon_decay

		return loss.item()

# TODO: Zaimplementowac zapisywanie w trakcie uczenia oraz na koniec uczenia (nie chcemy tracic modeli)
	def save_model(self):
		'''
		:return: current state of the model
		'''
		pass



	def cache(self, state, next_state, action, reward, done):
		'''
			Saves state of the game: (state, next_state, action, reward, done) into a bucket
			:return: None
		'''
		def first_if_tuple(x):
			return x[0] if isinstance(x, tuple) else x

		state = first_if_tuple(state).__array__()
		next_state = first_if_tuple(next_state).__array__()

		state = torch.tensor(state)
		next_state = torch.tensor(next_state)
		action = torch.tensor([action])
		reward = torch.tensor([reward])
		done = torch.tensor([done])

		self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

	def mario_scores(self, state):
		state = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
		return self.mario(state)

	def teacher_scores(self, state):
		state = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
		return self.teacher(state)

