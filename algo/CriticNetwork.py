import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate,add
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import pdb

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_critic_network(state_size, action_size, learning_rate):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """
    # raise NotImplementedError
    state_input = Input(shape=[state_size])
    action_input = Input(shape=[action_size])

    x1 = Dense(HIDDEN1_UNITS, activation = 'relu')(state_input)
    x2 = Dense(HIDDEN1_UNITS, activation = 'relu')(action_input)
    # y = Dense(HIDDEN2_UNITS, activation = 'linear')(x1)
    # h1 = add([y,x2])
    h1 = concatenate([x1,x2])
    h2 = Dense(HIDDEN2_UNITS, activation = 'relu')(h1)
    value = Dense(1)(h2)
    
    model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model, state_input, action_input


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        # raise NotImplementedError
        self.model, self.state_input, self.action_input = create_critic_network(state_size, action_size,learning_rate)
        self.target_model,_ ,_= create_critic_network(state_size, action_size,learning_rate) 
        self.batch_size = batch_size
        self.tau = tau
        self.sess = sess
        self.action_grads = tf.gradients(self.model.output, self.action_input)
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        
        grads = self.sess.run(self.action_grads, feed_dict={
            self.state_input: states,
            self.action_input: actions
        })[0]
        return grads
        # raise NotImplementedError

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        # for i in range(len(self.model.trainable_weights)):
        #     weights = self.model.trainable_weights[i]
        #     target_weights = self.target_model.trainable_weights[i]
        #     # target_weights = self.tau * weights + (1 - self.tau)* target_weights
        #     self.sess.run(tf.assign(target_weights, self.tau * weights + (1 - self.tau)* target_weights))
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model.set_weights(target_weights)
        # raise NotImplementedError

