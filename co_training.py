import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import wrappers
import copy
import datetime
import random

import torch
from torch import nn
from torch import distributions as torchd

to_np = lambda x: x.detach().cpu().numpy()



class Dreamer(nn.Module):

    def __init__(self, config=None, logger=None, dataset=None):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_train = tools.Every(config.train_every)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(
            config.expl_until / config.action_repeat))
        self._metrics = {}
        if config.sc_domain:
           #self._step = count_steps(config.sc_traindir)
           self._step = config.step
        else:
           #self._step = count_steps(config.tg_traindir)
           self._step = config.step
        # Schedules.
        config.actor_entropy = (
          lambda x=config.actor_entropy: tools.schedule(x, self._step))
        config.actor_state_entropy = (
          lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
        config.imag_gradient_mix = (
          lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
        self._dataset = dataset
        self._wm = models.WorldModel(self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad)
        reward = lambda f, s, a: self._wm.heads['reward'](f).mean
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]()

    def reset(self, config, logger, dataset):
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_train = tools.Every(config.train_every)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(
            config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = count_steps(config.traindir)
        # Schedules.
        config.actor_entropy = (
            lambda x=config.actor_entropy: tools.schedule(x, self._step))
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
        config.imag_gradient_mix = (
            lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
        self._dataset = dataset
   

    def __call__(self, obs, reset, state=None, reward=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training and self._should_train(step):
            steps = (
                self._config.pretrain if self._should_pretrain()
                else self._config.train_steps)
            for _ in range(steps):
                self._train(next(self._dataset))
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                openl = self._wm.video_pred(next(self._dataset))
                if self._config.video_log:
                    self._logger.video('train_openl', to_np(openl))
                self._logger.write(step=self._step, fps=True, target=not self._config.sc_domain)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._config.step = self._config.action_repeat * self._step
            #self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs['image'])
            latent = self._wm.dynamics.initial(len(obs['image']))
            action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
        else:
            latent, action = state
        embed = self._wm.encoder(self._wm.preprocess(obs))
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, self._config.collect_dyn_sample)
        if self._config.eval_state_mean:
            latent['stoch'] = latent['mean']
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == 'onehot_gumble':
            action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
        action = self._exploration(action, training)
        policy_output = {'action': action, 'logprob': logprob}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if 'onehot' in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._config.action_noise)

    def _train(self, data):
        # World model training
        metrics = {}
        post, context, mets = self._wm._train(data)
        # print(mets)
        metrics.update(mets)
        start = post
        if self._config.pred_discount:  # Last step could be terminal.
            start = {k: v[:, :-1] for k, v in post.items()}
            context = {k: v[:, :-1] for k, v in context.items()}
        reward = lambda f, s, a: self._wm.heads['reward'](
            self._wm.dynamics.get_feat(s)).mode()
        # Task behavior training
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != 'greedy':
            if self._config.pred_discount:
                data = {k: v[:, :-1] for k, v in data.items()}
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({'expl_' + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _train_wm(self, data):
        # Only world model training
        metrics = {}
        post, context, mets = self._wm._train(data)
        # print(mets)
        metrics.update(mets)
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
        for name, values in self._metrics.items():
            self._logger.scalar(name, float(np.mean(values)))
            self._metrics[name] = []


def count_steps(folder):
    return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))

def make_dataset(episodes, config):
    generator = tools.sample_episodes(
        episodes, config.batch_length, config.oversample_ends, config.seed)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, logger, mode, train_eps, eval_eps, source=1):
    suite, task = config.source_task[0].split('_', 1)
    if source == 0:
        suite, task = config.target_task[0].split('_', 1)
    if suite == 'dmc':
        env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)
    elif suite == 'atari':
        env = wrappers.Atari(
            task, config.action_repeat, config.size,
            grayscale=config.grayscale,
            life_done=False and ('train' in mode),
            sticky_actions=True,
            all_actions=True)
        env = wrappers.OneHotAction(env)
    elif suite == 'dmlab':
        env = wrappers.DeepMindLabyrinth(
            task,
            mode if 'train' in mode else 'test',
            config.action_repeat)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":  
        task = "-".join(task.split("_"))
        env = wrappers.MetaWorld(
            task,
            config.seed,
            config.action_repeat,
            config.size,
            config.camera,
            config.device
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "gym":  
        task = "-".join(task.split("_"))
        env = wrappers.GymWrapper(
            task,
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "robodesk":
        env = wrappers.RoboDesk(
            task,
        )
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    if (mode == 'train') or (mode == 'eval'):
        callbacks = [functools.partial(
            process_episode, config, logger, mode, train_eps, eval_eps)]
        env = wrappers.CollectDataset(env, callbacks)
    env = wrappers.RewardObs(env)
    return env


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
    if config.sc_domain:
        directory = dict(train=config.sc_traindir, eval=config.sc_evaldir)[mode]
    else:
        directory = dict(train=config.tg_traindir, eval=config.tg_evaldir)[mode]
    cache = dict(train=train_eps, eval=eval_eps)[mode]
    if config.sc_domain or mode == 'eval':
        filename = tools.save_episodes(directory, [episode])[0]
    length = len(episode['reward']) - 1
    score = float(episode['reward'].astype(np.float64).sum())
    video = episode['image']
    if mode == 'eval':
        cache.clear()
    if mode == 'train' and config.dataset_size:
        total = 0
        for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
            if total <= config.dataset_size - length:
                total += len(ep['reward']) - 1
            else:
                del cache[key]
        logger.scalar('dataset_size', total + length)
    if config.sc_domain or mode == 'eval':
        cache[str(filename)] = episode
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    logger.scalar(f'{mode}_episodes', len(cache))
    if mode == 'eval' or config.expl_gifs and config.video_log:
        logger.video(f'{mode}_policy', video[None])
    logger.write(step=config.step, target=not config.sc_domain)
    print(f'agent episode Step {config.step}.')


def main(config):
    # Set seeds.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = pathlib.Path(config.logdir[0]).expanduser()
    logdir = logdir / config.target_task[0]
    logdir = logdir / 'seed_{}'.format(config.seed)
    logdir = logdir / timestamp
    # source domain configs
    config.sc_traindir = config.sc_traindir or logdir / 'source_domain' / 'train_eps'
    config.sc_evaldir = config.sc_evaldir or logdir / 'source_domain' / 'eval_eps'
    # target domain configs
    config.tg_traindir = config.tg_traindir or logdir / 'target_domain' / 'train_eps'
    config.tg_evaldir = config.tg_evaldir or logdir / 'target_domain' / 'eval_eps'
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)
    config.step = 0

    print('Logdir:', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.sc_traindir.mkdir(parents=True, exist_ok=True)
    config.sc_evaldir.mkdir(parents=True, exist_ok=True)
    config.tg_traindir.mkdir(parents=True, exist_ok=True)
    config.tg_evaldir.mkdir(parents=True, exist_ok=True)
    tg_config = copy.deepcopy(config)  # copy config to target domain
    #sc_step = count_steps(config.sc_traindir)
    #tg_step = count_steps(tg_config.tg_traindir)
    sc_logger = tools.Logger(logdir, config=config)
    tg_logger = sc_logger
    #tg_logger = tools.Logger(logdir / 'target_domain', tg_config.action_repeat * tg_step, config=tg_config, target=True)
    config.sc_domain = True
    tg_config.sc_domain = False

    print('Create source envs.')
    # load source domain data
    sc_directory = config.sc_traindir
    sc_train_eps = tools.load_episodes(sc_directory, limit=config.dataset_size)
    sc_directory = config.sc_evaldir
    sc_eval_eps = tools.load_episodes(sc_directory, limit=1)
    sc_make = lambda mode: make_env(config, sc_logger, mode, sc_train_eps, sc_eval_eps, 1)
    sc_train_envs = [sc_make('train') for _ in range(config.envs)]
    sc_eval_envs = [sc_make('eval') for _ in range(config.envs)]
    sc_acts = sc_train_envs[0].action_space
    config.num_actions = sc_acts.n if hasattr(sc_acts, 'n') else sc_acts.shape[0]

    print('Create target envs.')
    if tg_config.offline_traindir:
        directory = tg_config.offline_traindir.format(**vars(tg_config))
    else:
        print("offline_traindir is None")
        assert False
    # Init load offline dataset
    tg_train_eps = tools.load_episodes(directory, limit=tg_config.dataset_size)
    if tg_config.offline_evaldir:
        directory = tg_config.offline_evaldir.format(**vars(tg_config))
    else:
        print("offline_evaldir is None")
        assert False
    tg_eval_eps = tools.load_episodes(directory, limit=1)
    tg_make = lambda mode: make_env(tg_config, tg_logger, mode, tg_train_eps, tg_eval_eps, 0)
    tg_train_envs = [tg_make('train') for _ in range(tg_config.envs)]
    tg_eval_envs = [tg_make('eval') for _ in range(tg_config.envs)]
    tg_acts = tg_train_envs[0].action_space
    tg_config.num_actions = tg_acts.n if hasattr(tg_acts, 'n') else tg_acts.shape[0]

    # prefill source domain dataset
    prefill = max(0, config.prefill - count_steps(config.sc_traindir))
    print(f'Prefill dataset ({prefill} steps).')
    if hasattr(sc_acts, 'discrete'):
        random_actor = tools.OneHotDist(torch.zeros_like(torch.Tensor(sc_acts.low))[None])
    else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(torch.Tensor(sc_acts.low)[None],
                                   torch.Tensor(sc_acts.high)[None]), 1)

    def random_agent(o, d, s, r):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {'action': action, 'logprob': logprob}, None

    tools.simulate(random_agent, sc_train_envs, prefill)
    tools.simulate(random_agent, sc_eval_envs, episodes=1)
    #sc_logger.step = config.action_repeat * count_steps(config.sc_traindir)
    config.step = config.action_repeat * prefill

    print('Simulate agent.')
    # sample batch dataset
    sc_train_dataset = make_dataset(sc_train_eps, config)
    sc_eval_dataset = make_dataset(sc_eval_eps, config)
    tg_train_dataset = make_dataset(tg_train_eps, tg_config)
    tg_eval_dataset = make_dataset(tg_eval_eps, tg_config)
    sc_agent = Dreamer(config, sc_logger, sc_train_dataset).to(config.device)
    tg_agent = Dreamer(tg_config, tg_logger, tg_train_dataset).to(tg_config.device)
    sc_agent.requires_grad_(requires_grad=False)
    tg_agent.requires_grad_(requires_grad=False)

    # Load the other agent model from save file
    model_dir = pathlib.Path(config.logdir[0]).expanduser()
    print(model_dir / config.load_model_dir)
    if (model_dir / config.load_model_dir).exists() and False:
        tg_agent.load_state_dict(torch.load(model_dir / 'latest_model.pt'))
        tg_agent._should_pretrain._once = False
        tg_agent._wm._tg_wm = tg_agent._wm
        tg_agent._task_behavior._tg_behavior = tg_agent._task_behavior
        print("Successful load tg_model")

    # rollout_dir = config.offline_traindir
    # file_path = os.path.join(rollout_dir, np.random.choice(os.listdir(rollout_dir)))
    # rollout_data = np.load(file_path, allow_pickle=True)

    tg_agent._wm._tg_wm = sc_agent._wm
    tg_agent._task_behavior._tg_behavior = sc_agent._task_behavior
    sc_agent._wm._tg_wm = tg_agent._wm
    sc_state = None
    tg_state = None

    # Step 1: Source domain pre-training
    for _ in range(3):
        # 10k steps for each epoch
        print("start source domain pre-training")
        #print(f"source steps: [{config.step}]")
        sc_logger.write(step=config.step)
        print('Start source evaluation.')
        # video_pred = sc_agent._wm.video_pred(next(sc_eval_dataset))
        # sc_logger.video('source_eval_openl', to_np(video_pred))
        eval_policy = functools.partial(sc_agent, training=False)
        tools.simulate(eval_policy, sc_eval_envs, episodes=1)
        print('Start source training.')
        # sc_agent.reward_predictor_training(sc_train_dataset, tg_train_dataset, config.reward_predictor_train_steps)
        sc_state = tools.simulate(sc_agent, sc_train_envs, steps=config.eval_every, state=sc_state)
    config.pre_train = False

    # Step 2: Target domain RSSM training 
    print("start co-training")
    for _ in range(2):
        # 40 k
        # target: training world model
        print("start target world model training")
        for _ in range(int(tg_config.eval_every)):
            tg_agent._train_wm(next(tg_train_dataset))
        print('Start source evaluation.')
        sc_logger.write(step=config.step)
        eval_policy = functools.partial(sc_agent, training=False)
        tools.simulate(eval_policy, sc_eval_envs, episodes=1)
        print('Start source domain training.')
        sc_agent._wm.set_tg_dataset(tg_train_dataset)
        sc_state = tools.simulate(sc_agent, sc_train_envs, steps=config.eval_every, state=sc_state)

    # Step 3: Co-training
    for i in range(6):
        # Start target training
        for _ in range(config.tg_train_steps):
            tg_logger.write(step=tg_config.step, target=True)
            print('Start target evaluation.')
            eval_policy = functools.partial(tg_agent, training=False)
            tools.simulate(eval_policy, tg_eval_envs, episodes=1)
            tools.evaluate_score(eval_policy, tg_eval_envs, logdir, episodes=10)
            print('Start target training.')
            tg_state = tools.off_simulate(tg_agent, tg_train_envs, tg_config, steps=tg_config.eval_every, state=tg_state)
        # Start source training
        if i == 5: 
            print('Start target evaluation.')
            tg_logger.write(step=tg_config.step, target=True)
            eval_policy = functools.partial(tg_agent, training=False)
            tools.simulate(eval_policy, tg_eval_envs, episodes=1)
            continue
        for _ in range(config.sc_train_steps):
            sc_logger.write(step=config.step)
            print('Start source evaluation.')
            eval_policy = functools.partial(sc_agent, training=False)
            tools.simulate(eval_policy, sc_eval_envs, episodes=1)
            print('Start source training.')
            # reward predictor training
            sc_state = tools.simulate(sc_agent, sc_train_envs, steps=config.eval_every, state=sc_state)
    tg_agent._wm._tg_wm = None
    tg_agent._task_behavior._tg_behavior = None
    torch.save(tg_agent.state_dict(), logdir / 'latest_model.pt')
    for env in sc_train_envs + sc_eval_envs + tg_train_envs + tg_eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    print(remaining)
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / 'co_configs.yaml').read_text())
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
