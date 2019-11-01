import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import add
import gym
import pdb
import os

import envs
from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                     # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
OUTPUT_PATH = './Results'
NUM_EPISODES = 5000
outfile = './Results/ddpg_log.txt'

class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(action+self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        self.OUTPUT_PATH = outfile_name
        self.outfile = outfile
        self.action_dim = len(env.action_space.low)
        self.state_dim = len(env.observation_space.low)
        self.burn_in = 5000
        np.random.seed(1337)
        self.env = env
        self.sess = tf.Session()
        self.gamma = GAMMA
        self.tau = TAU
        self.batch_size = BATCH_SIZE
        self.lr = LEARNING_RATE_CRITIC
        self.actor = ActorNetwork(sess = self.sess, state_size =  self.state_dim, action_size = self.action_dim,
                                     batch_size = self.batch_size, tau = self.tau, learning_rate = LEARNING_RATE_ACTOR)
        self.critic = CriticNetwork(sess = self.sess, state_size = self.state_dim, action_size = self.action_dim,
                                     batch_size = self.batch_size,tau = self.tau, learning_rate = LEARNING_RATE_CRITIC)
        self.replay_mem = ReplayBuffer(buffer_size = BUFFER_SIZE)
        self.sample_noise = EpsilonNormalActionNoise(mu = 0,sigma = 0.01, epsilon = 0.1)
        tf.keras.backend.set_session(self.sess)
        # raise NotImplementedError

    def evaluate(self, num_episodes,episode):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.actor.model.predict(s_t[None])[0]
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            if i < 9:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    plt.savefig(os.path.join(self.OUTPUT_PATH,'plots','ep'+str(episode)+'_'+str(num_episodes)+'.png'))
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)
    
    def burn_in_memory(self):
	    # Initialize your replay memory with a burn_in number of episodes / transitions.
        transition_count = 0
        print(self.burn_in)
        while(transition_count <= self.burn_in):
            state = self.env.reset()
            done = False
            if not done and transition_count <= self.burn_in:
                state = np.expand_dims(state,axis=0)
                act = self.env.action_space.sample() 
                action = act + self.sample_noise(act)
                next_state, reward, done, info = self.env.step(action)
                self.replay_mem.add(state = state, action = action, reward = reward, new_state = next_state, done = done)
                state = next_state
                transition_count += 1
        print('Done burn in')
        print(transition_count)
    
    def get_training_data(self,batch):
        states = np.array([x for x in batch[:,0]])
        actions = np.array([x for x in batch[:, 1]])
        rewards = np.array([x for x in batch[:, 2]])
        new_states = np.array([x for x in batch[:, 3]])
        dones = np.array([int(not(x)) for x in batch[:, 4]])
        # pdb.set_trace()
        # states = np.array(states,ndim = 2)
        # actions = np.array(actions,ndim = 2)
        # new_states = np.array(new_states,ndim = 2)
        states = np.squeeze(states,axis = 1)
        dones = np.expand_dims(dones,axis = 1)
        rewards = np.expand_dims(rewards,axis = 1)
        q_values_curr_state =  self.critic.model.predict([states,actions])
        q_values_next_state = self.critic.target_model.predict([new_states,self.actor.target_model.predict(new_states)])
        targets = q_values_next_state
        out = np.multiply(self.gamma,np.multiply(q_values_next_state, dones))
        targets = rewards + out 
        return states,actions,targets,q_values_curr_state

    def update_models(self,states,actions,critic_targets):
        # pdb.set_trace()
        loss = self.critic.model.train_on_batch([states,actions],critic_targets)
        actions1 = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions1)
        self.actor.train(states, grads)
        self.actor.update_target()
        self.critic.update_target()
        return loss

    def plot_graph(self,data, title, xlabel, ylabel):
        plt.figure(figsize=(12,5))
        plt.title(title)
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.OUTPUT_PATH,title+'.png'))

    def plot_errorbar(self,x, y, yerr, title, xlabel, ylabel, label=None):
        plt.figure(figsize=(12,5))
        plt.title(title)
        plt.errorbar(x, y, yerr, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.OUTPUT_PATH, title+'.png'))

    def train(self, num_episodes, hindsight=True):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """
        if hindsight:
            suffix = 'HER'+'_'+str(self.gamma)+'_'+str(self.tau)+'_'+str(self.lr)
        else:
            suffix = 'DDPG'+'_'+str(self.gamma)+'_'+str(self.tau)+'_'+str(self.lr)
        x = []
        train_loss = []
        train_rewards = []
        val_mean_rewards = []
        val_std_rewards = []
        success_eval = []
        self.burn_in_memory()
        for i in range(num_episodes):
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            loss = 0
            store_states = []
            store_actions = []
            while not done:
                state = np.expand_dims(state,axis=0)
                act = self.actor.model.predict(state) 
                action = act + self.sample_noise(act)
                action = np.squeeze(action, axis = 0)
                next_state, reward, done, info = self.env.step(action)
                store_states.append(state)
                store_actions.append(action)
                self.replay_mem.add(state = state, action = action, reward = reward, new_state = next_state, done = done)
                state = next_state
                total_reward += reward
                step += 1  
                # temp = tf.keras.losses.MSE(critic_targets,q_val)
                batch =np.array(self.replay_mem.get_batch(batch_size = BATCH_SIZE))
                states,actions,critic_targets,q_val = self.get_training_data(batch)
                temp = self.update_models(states,actions,critic_targets)
                # temp = self.sess.run(
                #             tf.reduce_sum(tf.pow(critic_targets - q_val, 2)) / (critic_targets.shape[0])) 
                loss += temp
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                # raise NotImplementedError

            if hindsight:
                # For HER, we also want to save the final next_state.
                store_states.append(new_s)
                self.add_hindsight_replay_experience(store_states,
                                                     store_actions)
            
            train_loss.append(loss)
            train_rewards.append(total_reward)
            del store_states, store_actions
            store_states, store_actions = [], []

            # Logging
            print("Episode %d: Total reward = %d" % (i, total_reward))
            print("\tTD loss = %.2f" % (loss / step,))
            print("\tSteps = %d; Info = %s" % (step, info['done']))
            if i % 100 == 0:
                successes, mean_rewards, std_rewards = self.evaluate(10,i)
                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f,\n" % (successes, mean_rewards))
                x.append(i)
                success_eval.append(successes)
                val_mean_rewards.append(mean_rewards)
                val_std_rewards.append(std_rewards)
            if i % 500 == 0:
                self.plot_graph(success_eval, suffix+'_Episode_success', 'Episodes', 'Evaluaton Success')
                self.plot_graph(train_rewards, suffix+'_Episode_rewards', 'Episodes', 'Training Rewards')
                self.plot_graph(train_loss, suffix+'_Training_loss', 'Episodes', 'Training Loss')
                self.plot_errorbar(x, val_mean_rewards, val_std_rewards, suffix+'_mean_val_rewards', 'Episodes', 'Val Rewards', label='std')


        pdb.set_trace()
        self.actor.target_model.save_weights(os.path.join(self.OUTPUT_PATH, suffix+'_actor_model.h5'))
        self.critic.target_model.save_weights(os.path.join(self.OUTPUT_PATH, suffix+'_critic_model.h5'))
        


    def add_hindsight_replay_experience(self, states, actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        # compute the rewards for the current goal
        states_ser, rewards_ser = self.env.apply_hindsight(states)

        # we now have the states,rewards and actions to add to the buffer. I think there will be one action less than the st        # ates and rewards
        self.add_to_buffer(states_ser, rewards_ser, actions)
        
        '''
        # we sample additional goals (-future, episode, random)
        additional_goals = sample_additional_goals(states, actions, strategy='future')


        # get sets of states and rewards for each goal and add to the buffer
        for g in additional_goals:
            states[-1] = g
            states_her, rewards_her = self.env.apply_hindsight(states)

            # add the additional transitions to the buffer
            self.add_to_buffer(states_her, rewards_her, actions)
        '''



if __name__ == '__main__':
    env = gym.make('Pushing2D-v0')
    ddpg = DDPG(env, OUTPUT_PATH)
    ddpg.train(NUM_EPISODES,False)