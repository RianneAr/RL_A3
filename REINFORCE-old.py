from catch import Catch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
        G = 0
        for t in reversed(range(len(episode_rewards))):
            G = self.gamma * G + episode_rewards[t]
            cumulative_reward[t] = G

        # Normalize the cumulative_reward
        cumulative_reward = (cumulative_reward - np.mean(cumulative_reward)) / (np.std(cumulative_reward) + 1e-9)

        # Compute the policy gradient estimate
        with tf.GradientTape() as tape:
            loss = 0
            for t in range(len(episode_states)): # t ... T-1
                state = episode_states[t]
                action = episode_actions[t]
                cumulative_reward_t = cumulative_reward[t]

                probabilities = self.policy_network(state.reshape(1,-1))  #should we be using the stored probs?
                log_probability = tf.math.log(probabilities[0,action])

                entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities))
                loss += -log_probability * cumulative_reward_t - self.entropy_coefficient * entropy

            gradients = tape.gradient(loss, self.policy_network.trainable_variables)

        # Update the policy parameters
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

        rewards.append(episode_rewards)

    return rewards


def test():
    #parameters
    max_epochs = 10
    learning_rate = 0.001
    gamma = 0.99
    entropy_coefficient = 0.01  #alpha

    # Plotting parameters
    # plot = True

    rewards = REINFORCE(max_epochs, learning_rate, gamma, entropy_coefficient)
    print("Obtained rewards: {}".format(rewards))
    
if __name__ == '__main__':
    test()
