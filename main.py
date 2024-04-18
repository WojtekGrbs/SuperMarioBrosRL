from preparation import *
from smbneuralnetwork import SMBNeuralNetwork
import torch
import mario
network1 = SMBNeuralNetwork().float()
network2 = SMBNeuralNetwork(evaluation=True)


Mario = mario.Mario()

env = generate_env()
env.reset()
next_state, reward, done, truncated, info = env.step(0)


for i in range(5000):
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        a = Mario.choose_action(state)
        new_state, reward, done, truncated, info = env.step(a)
        Mario.learn(state=state, action=a, next_state=new_state, done=done, reward=reward)
        total_reward += reward
        state = new_state