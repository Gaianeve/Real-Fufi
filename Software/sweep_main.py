
#libraries
import os
import argparse
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
from types import SimpleNamespace

#RL e ML
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# code from git repo to run training
from environment import vectorize_env
from agent_class import Agent
from agent_utils import anneal, collect_data, GAE, PPO_train_agent, evaluate_agent


#default sconfig
def get_default_hyperparameters_and_names():
  default_config = SimpleNamespace(

      #cute names
      exp_name = "Fufi_adventures",
      gym_id = "Fufi-v0",
      torch_deterministic =True,
      cuda =True,

      #W&B setup
      wandb_project_name ="Fufino",
      wandb_entity =None,
      capture_video =False,

      #hyperparameters
      lr = 1.5e-4,
      seed = 1,
      total_timesteps = 1000000,
      num_envs = 8,
      num_steps = 512,
      anneal_lr = True,
      gae =True,
      gamma=0.99,
      gae_lambda=0.95,
      num_minibatches = 4,
      update_epochs = 10,
      norm_adv=True,
      clip_coef =0.2,
      clip_vloss=True,
      ent_coef =0.01,
      vf_coef=0.5,
      max_grad_norm=0.5,
      target_kl=None,
  )
  return default_config

def parse_args(default_config):
    parser = argparse.ArgumentParser()

    # Track esperiment things
    parser.add_argument("--exp-name", type=str, default= default_config.exp_name,
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default= default_config.gym_id,
        help="the id of the gym environment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default= default_config.torch_deterministic,
        nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=default_config.cuda, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    #W&B setup
    #parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        #help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=default_config.wandb_project_name,
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default= default_config.wandb_entity,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=default_config.capture_video,
        nargs="?", const=True, help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments and hyperparamenters
    parser.add_argument("--learning-rate", type=float, default= default_config.lr,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default= default_config.seed,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default= default_config.total_timesteps,
        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default= default_config.num_envs,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=default_config.num_steps,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default = default_config.anneal_lr, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=default_config.gae, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=default_config.gamma,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=default_config.gae_lambda,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=default_config.num_minibatches,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default= default_config.update_epochs,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default= default_config.norm_adv, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default= default_config.clip_coef,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default= default_config.clip_vloss, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default= default_config.ent_coef,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default= default_config.vf_coef,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default= default_config.max_grad_norm,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default= default_config.target_kl,
        help="the target KL divergence threshold") #should be set to 0.015 if wanna use
    args, unknown = parser.parse_known_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    vars(default_config).update(vars(args))
    return


import wandb
def train_main(config):
  run_name = f"{config.gym_id}__{config.exp_name}__{config.seed}__{int(time.time())}"

## -------------------------------------- W&B, TENSORBOARD ----------------------------------------

  # weight and biases project
  wandb.init(
    project=config.wandb_project_name,
    entity=config.wandb_entity,
    config=config,
    name=run_name,
    sync_tensorboard = True,
    save_code=True)
    
  #config sweep
  config = wandb.config
  args = config

  # tensorboard setup
  writer = SummaryWriter(f"runs/{run_name}")
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  ## ------------------------------------- SETTING UP THE GAME -------------------------------------------

  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

  # env setup
  envs = vectorize_env(args.gym_id, args.seed, args.capture_video, run_name, args.num_envs)

  # Agent setup
  agent = Agent(envs).to(device)
  #agent.print_summary(envs)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

  # initializing things
  # ALGO Logic: Storage setup
  obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
  actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values = torch.zeros((args.num_steps, args.num_envs)).to(device)

## ------------------------------------- START THE GAME -------------------------------------------
  # TRY NOT TO MODIFY: start the game
  global_step = 0
  start_time = time.time()
  next_obs = torch.Tensor(envs.reset()).to(device)
  next_done = torch.zeros(args.num_envs).to(device)
  num_updates = args.total_timesteps // args.batch_size

  #sum of episodic returns
  sum_episodes = 0
  for update in range(1, num_updates + 1):
    #print('Starting update {}'.format(update))
    # Annealing the rate if instructed to do so.
    optimizer.param_groups[0]["lr"] = anneal(args.anneal_lr, update, num_updates, \
                                             args.learning_rate)

    for step in range(0, args.num_steps):
      # update global steps
      global_step += 1 * args.num_envs
      #update parameters
      obs, actions, logprobs, rewards, dones, values, next_obs, next_done, info \
      = collect_data(envs, obs, actions, logprobs, rewards, dones, values, next_obs,\
                     next_done, agent, step, device)

      # update tensorboard
      if 'episode' in info.keys():
        for item in info['episode']:
          if item is not None:
            #print(f"global_step={global_step}, episodic_return={item['r']}")
            writer.add_scalar("charts/episodic_return", item["r"], global_step)
            writer.add_scalar("charts/episodic_length", item["l"], global_step)
            #wandb.log({"charts/episodic_return":item["r"]})
            #wandb.log({"charts/episodic_length": item["l"]})

            sum_episodes = sum_episodes + item["r"]
            writer.add_scalar("charts/total_episodic_returns", sum_episodes, global_step)
            wandb.log({"sum_episodes": sum_episodes})

    # general advantages estimation
    returns, advantages = GAE(args.gae, args.gae_lambda, args.gamma, agent,\
        values, dones, rewards, next_obs, next_done,\
        args.num_steps, device)
    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

## ------------------------------------- TRAINING LOOP ----------------------------------------------
    v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs,\
    b_values, b_returns = PPO_train_agent(args.batch_size, args.update_epochs, args.minibatch_size, \
                                      args.clip_coef, args.norm_adv, args.clip_vloss,\
                                      args.ent_coef, args.vf_coef, args.max_grad_norm, args.target_kl,\
                                      agent, optimizer, scheduler, False,\
                                      b_obs, b_actions,b_logprobs,\
                                      b_advantages, b_returns, b_values)

## --------------------------------- UPDATING AND CLOSING UP -----------------------------------------
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    #print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)



## -------------------------------------- Log video to W&B ---------------------------------------------------
  # Explicitly set the metric in the run summary if necessary
  wandb.run.summary["sum_episodes"] = sum_episodes
  if args.capture_video:
    video_files = [file for file in os.listdir(f"./videos/{run_name}") if file.endswith(".mp4")]
    video_files.sort(key=lambda x: os.path.getctime(os.path.join(f"./videos/{run_name}", x)))
    for video_file in video_files:
        #print(video_file)
        video_path = os.path.join(f"./videos/{run_name}", video_file)
        wandb.log({"episode_video": wandb.Video(video_path, fps=4, format="mp4")})
  envs.close()
  writer.close()
  wandb.finish()



