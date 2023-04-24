from catch import Catch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, entropy_coefficient):
        self.state_size = state_size
        self.action_size = action_size 
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.policy_network = self.init_model()

    def init_model(self):
        '''
            We create our model having to hidden layers of 24 units (neurones)
            The first layer has the same size as a state size
            # The last layer has the size of actions space  '''
       
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(np.product(self.state_size),)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])

        model.compile(loss='mse', optimizer=self.optimizer) 

        return model


    def select_action(self, state):
        
        # Get the action probabilities and choose an action
        probabilities = self.policy_network(state.reshape(1,-1)) #(1, 7x7x2) # get the policy network
        
        # action_probabilities = tf.nn.softmax(probabilities)  # normalization
        action = tf.random.categorical(probabilities, 1)[0, 0].numpy()  #get the action to perform

        return action, probabilities   ###### we need to check that the probs are correct


    def trace(self, env):

        trace_rewards = []
        trace_actions = []
        trace_states = []
        trace_probs = []
        done = False
        state = env.reset()

        while not done:   #########should we also check for n_timesteps????
            action, probabilities = self.select_action(state)
            next_state, reward, done, _ = env.step(action) # Take a step in the environment

            trace_rewards.append(reward)
            trace_actions.append(action)
            trace_probs.append(probabilities)
            state = next_state
            trace_states.append(state)
    
        return trace_rewards, trace_actions, trace_states, trace_probs



    def trace_gradients(self, rewards, actions, states):
        
        # Compute the discounted cumulative rewards for each time step of the episode
        cumulative_reward = np.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = self.gamma * G + rewards[t]
            cumulative_reward[t] = G

        # Normalize the cumulative_reward
        cumulative_reward = (cumulative_reward - np.mean(cumulative_reward)) / (np.std(cumulative_reward) + 1e-9)

        # Compute the policy gradient estimate
        with tf.GradientTape() as tape:
            loss = 0
            for t in range(len(states)): # t ... T-1
                probabilities = self.policy_network(states[t].reshape(1,-1))  #should we be using the stored probs?
                log_probability = tf.math.log(probabilities[0,actions[t]])

                entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities))
                loss += -log_probability * cumulative_reward[t] - self.entropy_coefficient * entropy

            gradients = tape.gradient(loss, self.policy_network.trainable_variables)

        return gradients
        # # Update the policy parameters
        # self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))


def REINFORCE(max_epochs, M, learning_rate, gamma, entropy_coefficient):
    ''' runs a single repetition of an REINFORCE agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    rewards = []

    # Initialize environment and Q-array
    env = Catch()
    # s = env.reset()
    # step_pause = 0.3 # the pause between each plot
    # env.render(step_pause)

    state_size = env.observation_space.shape # (7,7,2)
    action_size = env.action_space.n

    pi = REINFORCEAgent(state_size, action_size, learning_rate, gamma, entropy_coefficient)

    # Train the agent using REINFORCE with entropy regularization
    for episode in range(max_epochs):
        # we will store the traces
        episode_rewards = []
        # episode_actions = []
        # episode_states = []
        # episode_probs = []

        gradients_l = []
        
        # DO M TRACES
        for m in range(M):
            # get the whole trace following policy
            trace_rewards, trace_actions, trace_states, trace_probs = pi.trace(env)
            # print('CUMMULATIVE REWARDS OF TRACE', m, sum(trace_rewards))
            # save trace
            episode_rewards.append(sum(trace_rewards))
            # episode_actions.append(trace_actions)
            # episode_probs.append(trace_probs)
            # episode_states.append(trace_states)

            grads = pi.trace_gradients(trace_rewards, trace_actions, trace_states)
            gradients_l.append(grads)   

        ##################### should we averageee????????
        gradients = [tf.reduce_mean(tensors, axis=0) for tensors in zip(*gradients_l)]

        # Update the policy network using REINFORCE with entropy regularization
        pi.optimizer.apply_gradients(zip(gradients, pi.policy_network.trainable_variables))

        if episode%10 == 0:
            print('episode', episode, 'with rewards:', episode_rewards)
        rewards.append(np.mean(episode_rewards))

    return rewards


def test():
    #parameters
    max_epochs = 1000
    M = 2  # number of traces
    learning_rate = 0.001
    gamma = 0.99
    entropy_coefficient = 0.01  #alpha

    results = REINFORCE(max_epochs, M, learning_rate, gamma, entropy_coefficient)
    # print("Obtained rewards: {}".format(np.unique(rewards)))
    print(results)
    # Plotting parameters
    # smoothing_window = 51

    Plot = LearningCurvePlot(title = 'Learning curve')
    smoothres = smooth(results, 3)
    Plot.add_curve(results, label='aa')
    Plot.save('reinforce.png')
    
if __name__ == '__main__':
    test()