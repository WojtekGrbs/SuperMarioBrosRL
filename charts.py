import matplotlib.pyplot as plt

def divide_into_N_parts(tab,N):
    
    N_parts = []

    for i in range(0,len(tab),N):
        s = 0
        for j in range(min(N,len(tab)-i)):
            s += tab[i + j]
        s = s / min(N, len(tab) - i)
        N_parts.append(s)
     
    return N_parts
    

def learning_outcomes(total_rewards,avg_looses,x,steps,N, save_directory):

    total_rewards_N = divide_into_N_parts(total_rewards,N)
    avg_looses_N = divide_into_N_parts(avg_looses,N)
    x_N = divide_into_N_parts(x,N)
    steps_N = divide_into_N_parts(steps,N)

    fig = plt.figure(figsize=(18,15))
    gs = fig.add_gridspec(2,2)
    gs.update(wspace=0.25,hspace=0.25)
    ax0, ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])

    ax0.plot(x_N)
    ax0.set_xlabel('Episodes')
    ax0.set_ylabel('X')
    ax0.set_title(f'Average max X position per {N} Episode')

    ax1.plot(avg_looses_N)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Avg Loss')
    ax1.set_title(f'Average Avg Loss per {N} Episode')

    ax2.plot(total_rewards_N)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Total Reward')
    ax2.set_title(f'Average Total Reward per {N} Episode')
    
    ax3.plot(steps_N)
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Number of Steps')
    ax3.set_title(f'Average Number of Steps per {N} Episode')
    

    save_path = save_directory / f"plots.png"

    plt.savefig(save_path)
    plt.show()
