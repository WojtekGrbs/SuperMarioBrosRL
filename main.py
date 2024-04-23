from preparation import *
from smbneuralnetwork import SMBNeuralNetwork
import torch
import mario
from charts import learning_outcomes
network1 = SMBNeuralNetwork().float()
network2 = SMBNeuralNetwork(evaluation=True)

total_rewards = []
losses = []
avg_losses = []
steps = []
x_pos = []

Mario = mario.Mario()

env = generate_env()
env.reset()
next_state, reward, done, truncated, info = env.step(0)


for i in range(5):
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    losses = []
    step = 0
    while not done:
        a = Mario.choose_action(state)
        new_state, reward, done, truncated, info = env.step(a)
        loss = Mario.learn(state=state, action=a, next_state=new_state, done=done, reward=reward)
        total_reward += reward
        state = new_state
        step += 1
        
        if loss is not None:
            losses.append(loss.item())
        
    avg_losses.append(sum(losses) / len(losses))
    total_rewards.append(total_reward)
    steps.append(step)
    x_pos.append(info['x_pos'])
        

learning_outcomes(total_rewards,avg_losses,x_pos,steps,1)