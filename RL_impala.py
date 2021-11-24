import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1), nn.ReLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1), nn.ReLU()
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1), nn.ReLU()
    )
    self.resblock1 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding = 1), nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding = 1), nn.ReLU()
    )
    self.resblock2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding = 1), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding = 1), nn.ReLU()
    )
    self.maxpooling = nn.MaxPool2d(kernel_size = 3, stride = 2)
    self.linear = nn.Sequential(
        nn.Linear(in_features = 32*5*5, out_features = feature_dim)
    )
    self.flat = nn.Flatten()
    self.apply(orthogonal_init)


  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpooling(x)
    x_res = x.clone()
    x = self.resblock1(x) + x_res
    x = self.conv2(x)
    x = self.maxpooling(x)
    x_res = x.clone()
    x = self.resblock2(x) + x_res
    x = self.conv3(x)
    x = self.maxpooling(x)
    x_res = x.clone()
    x = self.resblock2(x) + x_res
    x = self.flat(x)
    x = self.linear(x)
    return x

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

"""Hyperparameters. These values should be a good starting point. You can modify them later once you have a working implementation."""

env_name = 'climber'
#exp_name = input('Insert experiment name: ')
exp_name = 'Test'
# Hyperparameters
total_steps = 25e6
num_envs = 64 # 32
num_levels = 200 #10
num_steps = 256 #256
num_epochs = 3
batch_size = 512 #512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, env_name, num_levels=num_levels, use_backgrounds=False)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
in_channels = env.observation_space.shape[0] # shape of state
feature_dim = 256 # arbitrary chosen
num_actions = env.action_space.n # number of possible actions (for climber: left, right, jump)

encoder = Encoder(in_channels, feature_dim)
policy = Policy(encoder, feature_dim, num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
step = 0
while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps): # for each step we update the baseline (old) policy
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs): # for each epoch we update the new policy

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator: 
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective
      ratio = torch.exp(new_log_prob - b_log_prob) # exp(new policy - old policy)  
      clipped_ratio = ratio.clamp(min=1.0 - eps, max=1.0 + eps)
      policy_reward = -torch.min(ratio * b_advantage, clipped_ratio * b_advantage)
      pi_loss = policy_reward.mean()

      # Clipped value function objective
      clipped_value = b_value + (new_value - b_value).clamp(min=-eps, max=eps)
      vf_loss = torch.max((new_value - b_returns) ** 2, (clipped_value - b_returns) ** 2)
      value_loss = value_coef * vf_loss.mean()

      # Entropy loss
      entropy_loss = new_dist.entropy().mean()

      # Backpropagate losses (see https://www.youtube.com/watch?v=5P7I-xPq8u8 at 15:05)
      # entropy controlls the distrubtion variance for exploitation
      loss = value_loss + pi_loss - entropy_coef * entropy_loss
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')

print('Completed training!')
torch.save(policy.state_dict(), 'checkpoint_impala_'+exp_name+'.pt')

"""# Notes:

PPO (proximal policy optimization) is based on TRPO (trust region policy optimization). TRPO add a KL constraint in order to make sure that the updated policy doesn't move to far away from the old policy (we want to stay in the trusted region, i.e. the region where we know everything works fine). However, the KL constraint increases the complexity of optimization problem and can lead to undesireable training behavior. Instead we wish to implement the constraint directly into our optimization objective function. This is excactly what PPO does. 

First, we define a $r_t(\theta)$ as the probability ratio between the outputs from the new updated policy (being learned) and outputs from the old policy (baseline, from earlier experiece), so given a sequence of sample actions and states, the probability ratio will be: 
- larger than 1 if the action is more likely in the new than old policy (numerator > denominator).
- between 0 and 1 if the action is less likely in the new than old policy (numerator < denominator).

We can then multiply the probability ratio, $r_t(\theta)$, with the advantage function, $A_t$, to get the normal policy gradients (PG) objective.
- The advantage is an estimate for how good an action is compared to the average action for a specific state, i.e. a relative action-value: A=Q-V, where Q is the absolute action-value/quality function and V is the state-value function.

The purpose of the clipping function is to truncate the normal policy gradients objective. 

So the $L_{CLIP}$ compares the normal policy gradients objective and a clipped version of it and takes the minimum (smallest one). Then we apply the expectation operator, i.e. we compute the mean of the batches of trajectories.

The advantage function is noisy and we don't want to change our policy drastically based on a single estimate. This is where the clipping function comes in handy. We remember the adavantage function, $A_t$, can be both postive and negative and this changes the effect of the min-operator. We have 3 cases:
- If action is good (A > 0) and it became more probably than before (r > 1), then we don't want to keep increasing likelihood too much, so we clip it (plateu phase)
- If the action is bad (A < 0) and it became less probable (r < 1), then we don't want to keep reducing likelihood too much, so we clip it.
- If the action is bad (A < 0) and it became more probable (r > 1), then we want to undo our last update. Note: this is the only region where the normal PG objective < clipped PG objective.

The final training objective in PPO (for training an agent) is the sum of $L_{CLIP}$ and two additional terms: $L_{VF}$ and $S$.
- $L_{VF}$ is the value function objective, which is in charge of updating the baseline network: how good is it to be in this state / what is the expected average amount of discounted reward?
- $S$ is the entropy term, which is in charge of making sure the the agent does enough exploration durinng training (we want it to act a bit randomly until the other parts of the objective starts dominating). Entropy is the average amount of bits that is needed to represent its outcome - it is a measure of how unpredictable an outcome of this variable really is. Maximing the entropy will therefore force the distribution to have a wide spread over all possible option resulting in the most unpredictable outcome. The PPO Head outputs the parameters of a Gaussian distribution for each possible action. When running the agent in training mode, the policy samples from these distributions to get continous output value for each action.

Below cell can be used for policy evaluation and saves an episode to mp4 for you to view.
"""

import imageio

# Make evaluation environment
eval_env = make_env(num_envs, env_name, start_level=num_levels, num_levels=num_levels, use_backgrounds=False)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(25*60):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
#norm_reward = ((torch.stack(total_reward)- torch.stack(total_reward).min(0))/(torch.stack(total_reward).min(0)+torch.stack(total_reward).max(0))).mean(0)
print('Average return:', total_reward)
#print('Normalized average return:', norm_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid_impala_'+exp_name+'.mp4', frames, fps=25)
