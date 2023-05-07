from catch import Catch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

        action_probabilities = tf.nn.softmax(probabilities)  # normalization
        action = tf.random.categorical(probabilities, 1)[0, 0].numpy()  #get the action to perform

        return action, action_probabilities

    def update_policy(self, episode_rewards, episode_actions, episode_states):
        # Compute the discounted cumulative rewards for each time step of the episode
        cumulative_reward = np.zeros_like(episode_rewards)
        
        #this is computing the sum return from the trace: R in the pseudocode
        G = 0
        for t in reversed(range(len(episode_rewards))):
            G = sum([self.gamma ** (k-t) *episode_rewards[k] for k in range(t, len(episode_rewards))])
            cumulative_reward[t] = G

        # Normalize the cumulative_reward (we add small epsilon value to avoid instability when dividing)
        cumulative_reward = (cumulative_reward - np.mean(cumulative_reward)) / (np.std(cumulative_reward) + 1e-9)

        # Compute the policy gradient estimate
        with tf.GradientTape() as tape:
            loss = 0
            
            #we first 
            for t in range(len(episode_states)): # t ... T-1
                state = episode_states[t]
                action = episode_actions[t]
                cumulative_reward_t = cumulative_reward[t]

                probabilities = self.policy_network(state.reshape(1,-1))
                log_probability = tf.math.log(probabilities[0,action] + 1e-8)
                
                entropy = tf.reduce_sum(probabilities * tf.math.log(probabilities[0] + 1e-8))
                loss += -cumulative_reward_t * log_probability 

            loss /= len(episode_states)
            entropy /= len(episode_states)
            loss -= self.entropy_coefficient * entropy
            print("Policy loss: ", loss)   
            # Update the policy parameters
            gradients = tape.gradient(loss, self.policy_network.trainable_variables)
            
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))


def REINFORCE(max_epochs, learning_rate, gamma, entropy_coefficient):
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
        state = env.reset()

        # we will store the trace
        episode_rewards = []
        episode_actions = []
        episode_states = []
        episode_probs = []

        done = False
        
        # get the whole trace following policy
        while not done:   #########should we also check for n_timesteps????
           
            action, probabilities = pi.select_action(state)

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # save trace
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_probs.append(probabilities)
            state = next_state
            episode_states.append(state)

        
        # Update the policy network using REINFORCE with entropy regularization
        pi.update_policy(episode_rewards, episode_actions, episode_states)
        print()
        rewards.append(np.mean(episode_rewards))
        print("Episode {} avg_ rewards: {}, nr of 1s: {}, episode length: {}".format(episode, sum(episode_rewards)/len(episode_rewards), episode_rewards.count(1), len(episode_rewards)))

    return rewards
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

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)
  
def test():
    #parameters
    max_epochs = 200
    learning_rate = 0.001
    gamma = 0.99
    entropy_coefficient = 0.01  #alpha
    smoothing_window = 101
    # Plotting parameters
    # plot = True

    rewards = REINFORCE(max_epochs, learning_rate, gamma, entropy_coefficient)
    print("Obtained rewards: {}".format(rewards))
    
    Plot = LearningCurvePlot(title = 'Learning curve')
    smoothres = smooth(rewards, smoothing_window)
    Plot.add_curve(rewards, label='aa')
    Plot.save('REINFORCE_experiment.png')
    
if __name__ == '__main__':
    test()