from catch import Catch
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
        
        # model =  tf.keras.Sequential([
        #     tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
        #     tf.keras.layers.Dense(self.action_size, activation='softmax')
        # ])
        # model.compile(loss='mse', optimizer=self.optimizer) 
       
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_size, 2), activation='relu'))  ##### CHECK THISSSS
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=self.optimizer) 
        model.summary()

        return model


    def compute_loss(self, actions, probabilities, rewards):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=probabilities)
        advantages = rewards - tf.reduce_mean(rewards)
        policy_loss = tf.reduce_mean(cross_entropy * tf.stop_gradient(advantages))
        entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probabilities, labels=tf.nn.softmax(probabilities)))
        loss = policy_loss - self.entropy_coefficient * entropy_loss
        return loss


def REINFORCE(max_epochs, learning_rate, gamma, entropy_coefficient):
    ''' runs a single repetition of an REINFORCE agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    rewards = []

    # Initialize environment and Q-array
    env = Catch()
    # s = env.reset()
    # step_pause = 0.3 # the pause between each plot
    # env.render(step_pause)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    pi = REINFORCEAgent(state_size, action_size, learning_rate, gamma, entropy_coefficient)

    # Train the agent using REINFORCE with entropy regularization
    for episode in range(max_epochs):
        state = env.reset()
        # we store the trace
        episode_rewards = []
        episode_actions = []
        episode_probs = []

        done = False
        
        # get the whole trace following policy
        while not done:
            # Get the action probabilities and choose an action
            probabilities = pi.policy_network(tf.convert_to_tensor(state)) # get the policy network
            ####### is this returning correct shape???

            print('HEREEE \n \n', probabilities.shape, probabilities)
            action_probabilities = tf.nn.softmax(probabilities)  # normalization
            action = tf.random.categorical(probabilities, 1)[0, 0].numpy()  #get the action to perform
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_probs.append(probabilities)
            state = next_state
            
        # Update the policy network using REINFORCE with entropy regularization
        with tf.GradientTape() as tape:
            loss = pi.compute_loss(tf.convert_to_tensor(episode_actions), tf.stack(episode_probs), tf.convert_to_tensor(episode_rewards))
        gradients = tape.gradient(loss, pi.policy_network.trainable_variables)
        pi.optimizer.apply_gradients(zip(gradients, pi.policy_network.trainable_variables))

        rewards.append(episode_rewards)

    return rewards


def test():
    #parameters
    max_epochs = 10
    learning_rate = 0.001
    gamma = 0.99
    entropy_coefficient = 0.01

    # Plotting parameters
    # plot = True

    rewards = REINFORCE(max_epochs, learning_rate, gamma, entropy_coefficient)
    print("Obtained rewards: {}".format(rewards))
    
if __name__ == '__main__':
    test()