import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

epoch=5_000_000

checkpoint_path = f'./checkpoint_data/total_rewards_list_{epoch}.pkl'

file = open(f'./checkpoint_data/total_rewards_list_{epoch}.pkl', 'rb')

# dump information to that file
rewards = pickle.load(file)

# close the file
file.close()

checkpoint_path = f'./checkpoint_data/total_rewards_list_{epoch}.pkl'

file = open(f'./checkpoint_data/total_rewards_list_{epoch}.pkl', 'rb')

# dump information to that file
rewards = pickle.load(file)

# print(len(rewards))
# for i in range(500):
#     print(len(rewards[i]))

new = list(np.concatenate(rewards))

for i in range(len(rewards[499])):
    print(rewards[499][i])

print(len(new))
print(max(new))
print(np.std(new[40000:41000]))
plt.plot(new[40000:41000])
print(np.argmax(new))
plt.title("Average Reward on Breakout")
plt.xlabel("Training Epochs [units of 10,000]")
plt.ylabel("Average Reward per Episode")
plt.show()
plt.close()

# close the file
file.close()
#
#
# # Save smoothed rewards
# with open(f'./checkpoint_data/total_rewards_list_{epoch}.pkl', 'wb') as f:
#     pickle.dump(total_rewards_list, f)
#
# # Save losses
# with open(f'./checkpoint_data/total_loss_list_{epoch}.pkl', 'wb') as f:
#     pickle.dump(total_loss_list, f)