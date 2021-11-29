from multiprocessing import Process, Pipe
import time
import torch
from torch.autograd import Variable
import numpy as np
import gym
import pybullet_envs
import matplotlib.pyplot as plt
import RobotRL.train as train
import RobotRL.tricks as tricks
from RobotRL.net import StochasticVNet
from RobotRL.utils import RunningManager, load_filter, save_filter
import copy
import os
import pickle
import seaborn as sns

sns.set()
torch.set_default_tensor_type(torch.DoubleTensor)


'''
The implementation of distributed PPO (using (dual) clipped loss and A2C-style update method) with PyTorch.
Algo Reference:
arxiv.org/abs/1707.06347
arxiv.org/abs/1707.02286
arxiv.org/abs/2011.12582
arxiv.org/abs/1811.02553
arxiv.org/abs/2005.12729
openai.com/blog/baselines-acktr-a2c/
Code Reference:
github.com/zhangchuheng123/Reinforcement-Implementation
'''


class Args():
    def __init__(self, env_name):
        self.policy = 'stochastic'
        self.seed = 1234
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        # Hyper parameters
        self.env_name = env_name
        self.horizon = 2048
        self.lr = 3e-4
        self.epochs = 10
        self.minibatch_size = 256
        self.max_step_per_round = 2000
        self.gamma = 0.995
        self.gae_lambda = 0.97
        self.episodes = 2000
        self.eps_clip = 0.2
        self.dual_clip = 3  # Special coefficient in dual-clip PPO
        self.coeff_vf = 0.5
        self.coeff_entropy = 0.01
        self.workers = 8
        self.save_every = 100
        # Tricks
        self.layer_norm = True
        self.advantage_norm = True
        self.state_norm = True
        self.lossvalue_norm = True
        self.reward_norm = False
        self.schedule_adam = 'linear'
        self.schedule_clip = 'linear'
        # Environments
        self.render = False
        self.train = False
        self.pretrained = '../save/RobotRLBike-v0/2021-08-12_21-42-58/RobotRLBike-v0_ppo_1100'
        self.state_shape = 0
        self.action_shape = 0


def update_param(batch_size, batch_states, batch_actions, batch_old_logprobs, batch_returns, batch_advantages, net, optimizer, clip_now, args):
    # shuffling the batch
    batch_ind = np.random.choice(batch_size, int(batch_size / args.minibatch_size) * args.minibatch_size,
                                 replace=False)
    for minibatch in range(int(batch_size / args.minibatch_size)):
    # sample from current batch
        minibatch_ind = batch_ind[minibatch * int(batch_size / args.minibatch_size):(minibatch + 1) * int(batch_size / args.minibatch_size)]
    # update learner
        minibatch_states = batch_states[minibatch_ind]
        minibatch_actions = batch_actions[minibatch_ind]
        minibatch_advantages = batch_advantages[minibatch_ind]
        minibatch_old_logprobs = batch_old_logprobs[minibatch_ind]
        minibatch_returns = batch_returns[minibatch_ind]
        minibatch_new_logprobs = train.get_logproba(net, minibatch_states, minibatch_actions)
        minibatch_values_next = net.forward_critic(minibatch_states).flatten()
        loss = get_loss(clip_now, minibatch_advantages, minibatch_old_logprobs, minibatch_new_logprobs,
                         minibatch_values_next, minibatch_returns, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def get_loss(eps_clip, Advantage, oldlogproba, newlogproba, values_next, return_, args):
    ratio = torch.exp(newlogproba - oldlogproba)
    surr1 = ratio * Advantage
    surr2 = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip) * Advantage
    loss_surr = torch.mean(torch.max(torch.min(surr1, surr2), args.dual_clip * torch.min(Advantage, 0 * Advantage)))
    # loss_surr = torch.mean(torch.min(surr1, surr2))
    if args.lossvalue_norm:
        loss_vf = tricks.lossvalue_norm(values_next, return_)
    else:
        loss_vf = torch.mean((values_next - return_).pow(2))
    loss_entropy = torch.mean(torch.exp(newlogproba) * newlogproba)
    loss = - loss_surr + args.coeff_vf * loss_vf + args.coeff_entropy * loss_entropy
    return loss


def main(env_name='Pendulum-v0'):
    args = Args(env_name)
    env = gym.make(env_name)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    env.close()
    save_dir = '../save/' + args.env_name + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '/'
    manager = RunningManager()
    manager.register('ZFilter', tricks.ZFilter)
    manager.register('Buffer', train.Buffer)
    manager.register('Count', train.Count)
    manager.start()
    rewards_table = []

    if args.seed is not None:
        torch.manual_seed(args.seed)
        npseed = args.seed
    if args.train:
        print('Initializing networks')
        buffer = manager.Buffer(args.device, buffer_size=False)
        count = manager.Count()
        if args.pretrained is None:
            net = StochasticVNet(args).to(args.device)
            running_state = manager.ZFilter((args.state_shape,), clip=5.0)
            running_reward = manager.ZFilter((1,), demean=False, discount=args.gamma, clip=10.0)
        else:
            net = torch.load(args.pretrained + '_net.pkl')
            running_state = load_filter(args.pretrained + '_statefilter.pkl', manager)
            running_reward = load_filter(args.pretrained + '_rewardfilter.pkl', manager)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        clip_now = args.eps_clip

        print('Initializing workers')
        pipe_dict = dict((i, (pipe1, pipe2)) for i in range(args.workers) for pipe1, pipe2 in (Pipe(),))
        child_process_list = []
        for worker in range(args.workers):
            pro = Process(target=train.collect_batch, args=(pipe_dict[worker][1], args, running_state, running_reward, buffer, count, worker))
            child_process_list.append(pro)
        [p.start() for p in child_process_list]

        plt.figure(1)
        plt.xlabel("Time steps")
        plt.ylabel("Rewards")
        plt.title(args.env_name)
        avg_rewards_table = []
        episode_table = []

        print('Start training')
        start_time = time.time()
        for episode in range(args.episodes):
            net_worker = copy.deepcopy(net).to('cpu')
            [pipe_dict[i][0].send(net_worker) for i in range(args.workers)]
            while 1:
                if count.output() == args.workers:
                    break
            count.clean()
            batch = buffer.output()
            buffer.clean()
            batch_size, batch_states, batch_actions, batch_rewards, batch_norm_rewards, batch_masks, batch_steps = train.get_batch(batch, args)
            batch_values = Variable(net.forward_critic(batch_states))
            batch_old_logprobs = Variable(train.get_logproba(net, batch_states, batch_actions))
            batch_returns, batch_advantages = train.get_gaes(batch_size, batch_norm_rewards, batch_values, batch_masks, args)

            avg_rewards = torch.sum(batch_rewards, dim=0) / batch_steps
            avg_rewards = avg_rewards.item()
            episode_table.append(episode)
            avg_rewards_table.append(avg_rewards)
            rewards_table.append(avg_rewards)

            print('Episode:', episode, '. Avg rewards in this episode:', avg_rewards)
            plt.figure(1)
            plt.plot(episode_table, avg_rewards_table, color='#ff7f0e')
            episode_table = [episode]
            avg_rewards_table = [avg_rewards]
            for K in range(args.epochs):
                if args.seed is not None:
                    np.random.seed(npseed)
                    npseed += 1
                update_param(batch_size, batch_states, batch_actions, batch_old_logprobs, batch_returns,
                                   batch_advantages, net, optimizer, clip_now, args)

            if args.schedule_clip == 'linear':
                episode_ratio = 1 - (episode / args.episodes)
                clip_now = args.eps_clip * episode_ratio
            else:
                clip_now = args.eps_clip
            if args.schedule_adam == 'linear':
                episode_ratio = 1 - (episode / args.episodes)
                lr_now = args.lr * episode_ratio
                for g in optimizer.param_groups:
                    g['lr'] = lr_now

            if (episode + 1) % args.save_every == 0:
                save_name = args.env_name + '_' + 'ppo_' + str(episode + 1)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(net, save_dir + save_name + '_net.pkl')
                save_filter(running_state, save_dir + save_name + '_statefilter.pkl')
                save_filter(running_reward, save_dir + save_name + '_rewardfilter.pkl')
                with open(save_dir + save_name + '_reward.pkl', 'wb') as f:
                    pickle.dump(rewards_table, f, pickle.HIGHEST_PROTOCOL)
                print('Model saved successfully!')
                pic_name = save_name + '.jpg'
                plt.savefig(save_dir + pic_name)
                print('Graph saved successfully!')

            plt.pause(0.2)

        [p.terminate() for p in child_process_list]
        end_time = time.time()
        train_time = end_time - start_time
        train_h = train_time // 3600
        train_min = (train_time - train_h * 3600) // 60
        train_sec = int(train_time - train_h * 3600 - train_min * 60)
        print('Training phase ended.', 'Total time used:', train_h, 'h', train_min, 'min', train_sec, 's')

        plt.clf()

    # test
    elif args.pretrained is not None:
        net = torch.load(args.pretrained + '_net.pkl')
        running_state = load_filter(args.pretrained + '_statefilter.pkl', manager)
    else:
        net = StochasticVNet(args)
        running_state = tricks.ZFilter((args.state_shape,), clip=5.0)

    running_state.stop_update()
    args.train = False
    avg_test_rewards = train.test(args, running_state, net.to('cpu'))
    print('Avg rewards in test:', avg_test_rewards)
    args.train = True


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main('Pendulum-v0')
    # main()
