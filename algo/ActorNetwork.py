import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import pdb

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_actor_network(state_size, action_size):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    state_input = Input(shape=[state_size])

    x = Dense(HIDDEN1_UNITS, activation = 'relu')(state_input)
    x = Dense(HIDDEN2_UNITS, activation = 'relu')(x)
    state_output = Dense(action_size, activation= 'tanh')(x)
    model = Model(inputs=state_input, outputs=state_output)
    return model,state_input


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.model, self.state_input = create_actor_network(state_size, action_size)
        self.target_model,_ = create_actor_network(state_size, action_size) 
        self.batch_size = batch_size
        self.tau = tau
        self.lr = learning_rate
        self.action_grads = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_grads)
        self.grads = zip(self.params_grad, self.model.trainable_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.grads)
        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        """Updates the actor by applying dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
            action_grads: a batched numpy array storing the
                gradients dQ(s, a) / da.
        """        
        self.sess.run(self.optimize, feed_dict={
            self.state_input: states,
            self.action_grads: action_grads
        })
        # raise NotImplementedError

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        # for i in range(len(self.model.trainable_weights)):
        #     weights = self.model.trainable_weights[i]
        #     target_weights = self.target_model.trainable_weights[i]
        #     # target_weights = self.tau * weights + (1 - self.tau)* target_weights
        #     self.sess.run(tf.assign(target_weights, self.tau * weights + (1 - self.tau)* target_weights))
        # pdb.set_trace()
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model.set_weights(target_weights)
        # raise NotImplementedError
