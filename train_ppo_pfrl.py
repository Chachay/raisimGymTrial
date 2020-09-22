"""A training script of PPO on Raisim Gym Torch enviroment [ANYMAL]

Original script of this file is 
   https://github.com/pfnet/pfrl/blob/master/examples/mujoco/reproduction/ppo/train_ppo.py
"""
import argparse
import functools
import os
import platform

from ruamel.yaml import YAML, dump, RoundTripDumper
import gym.spaces

if platform.system() == "Windows":
    # Add path to raisim.dll
    dllpath = os.path.join(os.getcwd(),"raisimlib", "raisim", "win32", "mt_debug", "bin")
    ## This does not work
    #os.add_dll_directory(dllpath)
    ## work around instead of add_dll_directory
    os.environ['PATH'] = dllpath + os.pathsep + os.environ['PATH']

from raisimGymTorch.env.bin import rsg_anymal

import numpy as np
import torch
from torch import nn

import pfrl
from pfrl.agents import PPO
from pfrl import experiments
from pfrl import utils

from gym.core import Env
from gym import spaces

class myEnv(Env):
    def __init__(self, rsc, cfg, visualize):
        self.env = rsg_anymal.RaisimGymEnv(rsc, dump(cfg['environment'], Dumper=RoundTripDumper), visualize)
        self.num_acts = self.env.getActionDim()
        self.num_obs  = self.env.getObDim()

        self.action_space = spaces.Box(np.ones(self.num_acts) * -1.,
                                       np.ones(self.num_acts) *  1.,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(np.ones(self.num_obs) * -100., 
                                            np.ones(self.num_obs) *  100.,
                                            dtype=np.float32)

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': []}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        observation = np.zeros(self.num_obs, dtype=np.float32)
        reward = self.env.step(action)
        done = self.env.isTerminalState(reward)
        self.env.observe(observation)
        return  observation.copy(), reward, done, {}

    def reset(self, **kwargs):
        self.env.reset()
        observation = np.zeros(self.num_obs, dtype=np.float32)
        self.env.observe(observation)
        return observation.copy()

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.setSeed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return None

# [Gym Wrappers \| alexandervandekleut\.github\.io](https://alexandervandekleut.github.io/gym-wrappers/)
class NormalizeObsSpace(gym.ObservationWrapper):
    """Normalize a Box action space to [-1, 1]^n."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=-np.ones_like(env.observation_space.low),
            high=np.ones_like(env.observation_space.low),
        )
    def observation(self, obs):
        n_obs = obs.copy()
        # -> [0, orig_high - orig_low]
        n_obs -= self.env.observation_space.low
        # -> [0, 2]
        n_obs /= (self.env.observation_space.high - self.env.observation_space.low) / 2
        # action is in [-1, 1]
        return n_obs - 1

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2 * 10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per PPO iteration.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--cfg', type=str,
                        default=os.path.abspath(os.path.join(current_dir,
                                                            "raisimlib",
                                                            "raisimGymTorch", "raisimGymTorch",
                                                            "env","envs","rsg_anymal","cfg.yaml")),
                        help='configuration file')
    cfg_abs_path = parser.parse_args().cfg
    cfg = YAML().load(open(cfg_abs_path, 'r'))

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        visualize = True if process_idx==0 and args.render else False
        env = myEnv(os.path.join(current_dir, "raisimlib","rsc"), cfg, visualize)

        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        env = NormalizeObsSpace(env)
        env = pfrl.wrappers.NormalizeActionSpace(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    # Only for getting timesteps, and obs-action spaces
    sample_env = myEnv(os.path.join(current_dir, "raisimlib","rsc"), cfg, False)
    timestep_limit = 100000 #sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    assert isinstance(action_space, gym.spaces.Box)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    )

    obs_size = obs_space.low.size
    action_size = action_space.low.size
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.998,
        lambd=0.95,
    )

    if args.load or args.load_pretrained:
        if args.load_pretrained:
            raise Exception("Pretrained models are currently unsupported.")
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(utils.download_model("PPO", args.env, model_type="final")[0])

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    elif args.num_envs==1:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(0, False),
            eval_env=make_env(0, True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            save_best_so_far_agent=True,
            use_tensorboard=True
        )
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
        )


if __name__ == "__main__":
    main()
