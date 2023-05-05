from catch import Catch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

class LearningCurvePlot:
    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.fig.savefig(name,dpi=300)

class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, entropy_coefficient):
        self.state_size = state_size
        self.action_size = action_size 
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.optimizer_policy = Adam(learning_rate=self.learning_rate)
        self.optimizer_value = Adam(learning_rate=self.learning_rate)
        self.policy_network, self.value_network = self.init_model()

    def init_model(self):
        '''
            We create our model having to hidden layers of 24 units (neurones)
            The first layer has the same size as a state size
            # The last layer has the size of actions space  '''
       
        policy_network = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(), #synchronized=True
            tf.keras.layers.Dense(64, activation='relu', input_shape=(np.product(self.state_size),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax'),
            tf.keras.layers.Lambda(lambda x: x * 1.0)  # Scale the output by a factor of 1.0
        ])

        value_network = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(), #synchronized=True
            tf.keras.layers.Dense(64, activation='relu', input_shape=(np.product(self.state_size),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        policy_network.compile(optimizer=self.optimizer_policy)
        value_network.compile(optimizer=self.optimizer_value)

        return policy_network, value_network

    def select_action(self, state):
        # Get the action probabilities and choose an action
        probabilities = abs(self.policy_network(state.reshape(1,-1))) #(1, 7x7x2) # get the policy network
        action = tf.random.categorical(probabilities, 1)[0, 0].numpy()  #get the action to perform
        
        return action, probabilities 


    def trace(self, env):

        trace_rewards = []
        trace_actions = []
        trace_states = []
        done = False
        state = env.reset()

        while not done:
            action, probabilities = self.select_action(state)


            try:
              next_state, reward, done, _ = env.step(action)
            except:
              print("except")
              print(action, len(probabilities), done, state)
              
              sta = env.reset()
              act, prob = self.select_action(sta)
              print('different \n', act, prob)
            

              break
            trace_rewards.append(reward)
            trace_actions.append(action)
            trace_states.append(state)
            state = next_state
            
        return trace_rewards, trace_actions, trace_states, done #, trace_probs


    def gradient_policy(self, states, actions, q_hat, learning_rate):  

        epsilon = 1e-8  # Small constant value to avoid NaN

        with tf.GradientTape() as tape:
            loss = 0
            for t in range(len(states)): # t ... T-1
                probabilities = abs(self.policy_network(states[t].reshape(1,-1)))
                log_probability = tf.math.log(probabilities[0, actions[t]] + epsilon)

                # entropy = tf.reduce_sum(probabilities * tf.math.log(probabilities[0]))
                entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + epsilon), axis=1)

                # loss += -q_hat[t] * log_probability - self.entropy_coefficient * entropy
                loss_o = q_hat[t] * log_probability + self.entropy_coefficient * entropy
                loss += loss_o

                if tf.reduce_any(tf.math.is_nan(loss_o)):
                    print("\nThe loss has NaN values!\n", probabilities, entropy, q_hat[t], loss_o)
              
            loss /= len(states)
            # print("Policy loss: ", loss)  

            

            gradients = tape.gradient(loss, self.policy_network.trainable_variables)

            # for grad in gradients:
            #     tf.debugging.check_numerics(grad, "Gradient contains NaN or Inf values")

            
        return gradients

    def gradient_value(self, states, actions, rewards, q, learning_rate):   
        with tf.GradientTape() as tape:
            loss = 0
            for t in range(len(states)):
                value = abs(self.value_network(states[t].reshape(1,-1)))
                loss += (q[t]-value)**2

            loss /= len(states)
            # print("Value loss: ", loss)    
            gradients = tape.gradient(loss, self.value_network.trainable_variables)

        return gradients


def actor_critic(max_epochs, M, learning_rate, gamma, entropy_coefficient, n):
    ''' runs a single repetition of an actor critic agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    rewards = []
    # Initialize environment and Q-array
    env = Catch()
    state_size = env.observation_space.shape # (7,7,2)
    action_size = env.action_space.n

    pi = ActorCriticAgent(state_size, action_size, learning_rate, gamma, entropy_coefficient)

    # Train the agent with entropy regularization
    for episode in range(max_epochs):
        # we will store the traces
        episode_rewards = []

        gradients_policy = []
        gradients_value = []
        
        # DO M TRACES
        for m in range(M):
            # get the whole trace following policy
            trace_rewards, trace_actions, trace_states, done = pi.trace(env)
              
            episode_rewards.append(sum(trace_rewards))
            # print("Episode {} trace nr: {}, avg_ rewards: {}, nr of 1s: {}, episode length: {}".format(episode, m, sum(trace_rewards), trace_rewards.count(1), len(trace_rewards)))
        
            # BOOTSTRAPPING  
            q_hat = []
            for t in range(len(trace_states)):
                if t+n < len(trace_states):
                    cumulative_reward = sum([gamma ** k * trace_rewards[t+k] for k in range(0, n-1)])
                    val = pi.value_network(trace_states[t+n].reshape(1,-1))
                    q = cumulative_reward + gamma **n * val
                    q_hat.append(q)
                else:
                    q_hat.append(0.001)
                
            # compute gradients for every trace
            
            grads_p = pi.gradient_policy(trace_states, trace_actions, q_hat, learning_rate)

            gradients_policy.append(grads_p) 

            grads_v = pi.gradient_value(trace_states, trace_actions, trace_rewards, q_hat, learning_rate)
            gradients_value.append(grads_v)  

            # if done:
            #     break

        
        # for gradient in gradients_policy:
        #     for grad in gradient:
        #         tf.debugging.check_numerics(grad, "Gradient contains NaN or Inf values")
        

        print("Episode {} rewards: {}".format(episode, episode_rewards))
        

        # # averaging
        # grad_policy = [tf.reduce_mean(tensors, axis=0) for tensors in zip(*gradients_policy)]
        grad_policy = [-tf.reduce_mean(tensors, axis=0) for tensors in zip(*gradients_policy)]
        grad_value = [tf.reduce_mean(tensors, axis=0) for tensors in zip(*gradients_value)]


        # clip to avoid large values and unstability
        clip_value = 0.5  # Set the maximum absolute value of the gradients
        grad_policy = [-tf.clip_by_value(tf.reduce_mean(tensors, axis=0), -clip_value, clip_value) for tensors in zip(*gradients_policy)]
        grad_value = [tf.clip_by_value(tf.reduce_mean(tensors, axis=0), -clip_value, clip_value) for tensors in zip(*gradients_value)]

        
        #experiment here with: actor: reduce sum
        # Update the policy and value network with averages
        pi.optimizer_policy.apply_gradients(zip(grad_policy, pi.policy_network.trainable_variables))
        pi.optimizer_value.apply_gradients(zip(grad_value, pi.value_network.trainable_variables))
        
        rewards.append(np.mean(episode_rewards))

    return rewards


def test():
    #parameters
    max_epochs = 600
    M = 3  # number of traces
    learning_rate = 0.01
    gamma = 0.95
    entropy_coefficient = 0.01  
    n = 10 #estimation depth

    results = actor_critic(max_epochs, M, learning_rate, gamma, entropy_coefficient, n)
    # print("Obtained rewards: {}".format(np.unique(rewards)))
    print(results)
    # Plotting parameters
    smoothing_window = 101

    Plot = LearningCurvePlot(title = 'Learning curve')
    smoothres = smooth(results, smoothing_window)
    Plot.add_curve(results, label='aa')
    Plot.save('actor_critic_001.png')
    
if __name__ == '__main__':
    test()