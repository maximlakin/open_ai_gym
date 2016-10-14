# based on https://github.com/FlankMe/general-gym-player/blob/master/GeneralGymPlayerWithTF.py
import gym, numpy, tensorflow as tf

def main():
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLander-v2')
    # env = gym.make('BipedalWalker-v2')
    # env = gym.make('CarRacing-v0')
    obs_space = env.observation_space.shape[0]
    architecture = [obs_space, 25 * obs_space, 25 * obs_space, env.action_space.n]

    agent = Agent(env.action_space.n,
                  obs_space,
                  architecture)



class Agent():
    def __init__(self,n_actions,obs_space,architecture):
        self.n_actions = n_actions
        self.obs_space = obs_space
        self.architecture = architecture
        self.nn = FeedForwardNeuralNetwork(architecture)

        self.discount = 0.99
        self.training_per_stage = 1
        self.mini_batch_size = 128
        self.replay_memory = 50000
        self.eps = 1.
        self.eps_decay = 0.98
        self.exp_episodes = 25
        self.min_eps = 0.1

        self.total_reward, self.list_rewards = 0.0, []
        self.last_action = np.zeros(self.n_actions)
        self.previous_observations, self.last_state = [], None
        self.last_state_idx=0
        self.action_idx=1
        self.reward_idx=2
        self.cur_state_idx=3
        self.terminal_idx=4

    def act(self, obs, reward, done, episode):
        self.total_reward += reward

        if done:
            self.list_rewards.append(self.total_reward)
            average = np.mean(self.list_rewards[-100:])
            print ('Episode', episode, 'Reward', self.total_reward, 'Average Reward', round(average, 2))
            self.total_reward=0.0
            self.eps = max(self.eps*self.eps_decay,self.min_eps)

        current_state = obs.reshape((1,len(obs)))

        if self.last_state is None:
            self.last_state = current_state
            value_per_action = self.nn.predict(self.last_state)
            chosen_action_index = np.argmax(value_per_action)
            self.last_action = np.zeros(self.n_actions)
            self.last_action[chosen_action_index] = 1
            return chosen_action_index

        new_observation = [0 for _ in range(5)]
        new_observation[self.last_state_idx] = self.last_state.copy()
        new_observation[self.action_idx] = self.last_action.copy()
        new_observation[self.reward] = reward
        new_observation[self.cur_state_idx] = current_state.copy()
        new_observation[self.terminal_idx] = done
        self.previous_observations.append(new_observation)
        self.last_state = current_state.copy()

        while len(self.previous_observations) >= self.replay_memory:
            self.previous_observations.pop(0)

        if episode > exp_episodes:
            for _ in range(self.training_per_stage):
                self.train()

            if np.random.random() > self.eps:
                value_per_action = self.nn.predict(self.last_state)
                chosen_action_index = np.argmax(value_per_action)
            else:
                chosen_action_index = np.random.randint(0, self.n_actions)
        else:
            chosen_action_index = np.random.randint(0, self.n_actions)

        next_action_vector = np.zeros([self.n_actions])
        next_action_vector[chosen_action_index] = 1
        self.last_action = next_action_vector

        return (chosen_action_index)

    def train(self):
        permutations = np.random.permutation(len(self.previous_observations))[:self.mini_batch_size]
        previous_states = np.concatenate([self.previous_observations[i][self.last_state_idx] for i in permutations])
        actions = np.concatenate([self.previous_observations[i][self.action_idx] for i in permutations])
        rewards = np.array([self.previous_observations[i][self.reward_idx] for i in permutations]).astype("float")
        current_states = np.concatenate([self.previous_observations[i][self.cur_state_idx] for i in permutations])
        done = np.array([self.previous_observations[i][self.terminal_idx] for i in permutations]).astype("bool")

        value_current_state = self.nn.predict(current_states)

        value_previous_states = rewards.copy()
        value_previous_states += ((1. - done)*self.discount*value_current_state.max(axis=1))

        self.nn.fit(previous_states, actions, value_previous_states)


class FeedForwardNeuralNetwork():
    def __init__(self, layers):
        layer_array = np.array(layers)
        self._alpha = 1e-3
        self.activation = lambda x : tf.maximum(0.01*x,x)

        self.input_layer = tf.placeholder("float", [None, layer_array[0]])
        self.hidden_activation_layer = [self.input_layer]
        self.feed_forward_weights = []
        self.feed_forward_bias = []

        for i in range(layer_array.size - 1):
            self.feed_forward_weights.append(tf.Variable(tf.truncated_normal([layer_array[i],layer_array[i+1]], mean=0.0, stddev=0.1)))
            self.feed_forward_bias.append(tf.Variable(tf.constant(-0.001, shape=[layer_array[i+1]])))

            if i < layer_array.size-2:
                activation_input=tf.matmul(self.hidden_activation_layer[i],self.feed_forward_weights[i])+self.feed_forward_bias[i]
                self.hidden_activation_layer.append(self.activation(activation_input))

        self.state_value_layer = (tf.matmul(self.hidden_activation_layer[-1],self.feed_forward_weights[-1])+self.feed_forward_bias[-1])
        self.action = tf.placeholder("float", [None, layer_array[-1]])
        self.target = tf.placeholder("float", [None])
        self.action_value_vector = tf.reduce_sum(tf.mul(self.state_value_layer, self.action), reduction_indices=1)
        self.cost = self.reduce_sum(tf.square(self.target - self.action_value_vector))
        self.alpha = tf.placeholder("float")
        self.train_operation = tf.train.AdamOptimizer(self.alpha).minimize(self.cost)
        self.session = tf.Session()
        operation_intizializer = tf.initialize_all_variables()
        self.session.run(operation_intizializer)

        self.forward_pass = lambda state: self.session.run(self.state_value_layer,feed_dict={self.input_layer: state})

        self.back_pass = lambda valueStates, actions, valueTarget: (self.session.run(self.train_operation, feed_dict={self.input_layer: valueStates, self.actions: actions, self.target: valueTarget, self.alpha: self._alpha}))

    def predict(self, state):
        return self.forward_pass(state)

    def fit(self, valueStates, actions, valueTarget):
        self.back_pass(valueStates, actions, valueTarget)

if __name__=="__main__":
   main()
