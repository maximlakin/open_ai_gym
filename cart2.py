# based on https://github.com/FlankMe/general-gym-player/blob/master/GeneralGymPlayerWithTF.py
import gym, numpy, tensorflow as tf

def main():
    env = gym.make('CartPole-v0')
    env.reset()

    agent = Agnet(env)
    reward, done = 0, False
    for i in range(5):
        observation, done = env.reset(), False
        action = agent.act(observation, reward, done, i)
        while not done:
            env.render()
            observation, reward, done, info = env.step(action)
            action = agent.act(observation, reward, done, i)

    env.close()


class Agnet():

    def __init__(self, env):
        obs_space = env.observation_space.shape[0]
        self.total_reward = 0
        self.env = env
        self.nn = FFNN([obs_space, 25*obs_space, 25*obs_space, env.action_space.n])
        self.eps = 1.0
        self.eps_decay = 0.98
        self.eps_min = 0.1
        self.previous_observations = []
        self.last_action = np.zeros(env.action_space.n)
        self.last_state = None

    def act(self, observation, reward, done, i):
        self.total_reward+=reward

        if done:
            print(self.total_reward)
            self.total_reward = 0
            self.eps = max(self.eps*self.eps_decay,self.eps_min)

        new_observation = [self.last_state, self.last_action, reward, done]
        self.previous_observations.append(new_observation)
        if len(self.previous_observations)>50000: self.previous_observations.pop(0)

        if np.random.random() < self.eps:
            next_action = self.env.action_space.sample()
        else:
            next_action = self.nn.predict()




        return next_action

class FFNN():

    def __init__(self, layers):
        self.layers = layers
        self.generate()

    def generate(self):
        self.activation = lambda x : tf.maximum(0.01*x, x)
        self.session = tf.Session()
        self.input_layer = tf.placeholder("float", [None, self.layers[0]])
        self.hidden_layer = []
        self.ff_weights = []
        self.ff_bias = []

        for i in range(len(self.layers[:-1])):
            self.ff_weights.append(tf.Variable(tf.truncated_normal([self.layers[i],self.layers[i+1]], mean=0.0, stddev=0.1)))
            self.ff_bias.append(tf.Variable(tf.constant(-0.01, shape = [self.layers[i+1]])))
            if i==0:
                activation = self.input_layer
            else:
                activation = self.activation(tf.matmul(self.hidden_layer[i-1],self.ff_weights[i-1])+self.ff_bias[i-1])
            self.hidden_layer.append(activation)

        self.state_value_layer = tf.matmul(self.hidden_layer[-1], self.ff_weights[-1])+self.ff_bias[-1]
        self.actions = tf.placeholder("float", [None,self.layers[-1]])
        self.target = tf.placeholder("float", [None])
        self.action_value_vector = tf.reduce_sum(tf.mul(self.state_value_layer,self.actions),1)
        self.cost = tf.reduce_sum(tf.square(self.target - self.action_value_vector))
        self.alpha = tf.constant(1e-3)
        self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.cost)

        self.session.run(tf.initialize_all_variables())
        self.feed_forward = lambda state: self.session.run(self.state_value_layer, feed_dict={self.input_layer: state})
        self.back_prop = lambda states, actions, target: self.session.run(
            self.optimizer,
            feed_dict={
                self.input_layer: states,
                self.actions: actions,
                self.target: target
            })


    def fit(self, states, actions, target):
        self.back_prop(states, actions, target)

    def predict(self, state):
        return self.feed_forward(state)



if __name__=="__main__":
   main()
