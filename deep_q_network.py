#!/usr/bin/env python
import argparse
import cv2
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque

sys.path.append("wrapped_games/")
import pong as game

GAME = "pong"  # The name of the game being played. Used for log files.
CHKPT_DIR = "chkpts-" + GAME
ACTIONS = 3      # number of valid actions
GAMMA = 0.99     # decay rate of past observations
OBSERVE = 100000 # timesteps to observe before training
EXPLORE = 100000 # frames over which to anneal epsilon
LEARNING_RATE = 1e-6 # Learning rate
FINAL_EPSILON = 0.1    # final value of epsilon
INITIAL_EPSILON = 1.0  # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32       # size of minibatch
K = 4            # only select an action every Kth frame, repeat prev for others

def weight_variable(shape, name=None):
    """Return a TensorFlow weight variable with the given shape.

    Parameters
    ----------
    shape : the shape of the weight tensor.
    name : the name for the weight tensor

    Returns
    -------
    out : a TensorFlow variable instance initialized randomly
    """
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    """Return a TensorFlow bias variable with the given shape.

    Parameters
    ----------
    shape : the shape of the bias tensor.

    Returns
    -------
    out : a TensorFlow variable initialized to a constant value.
    """
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, stride, name=None):
    """Return a 2D TensorFlow convolution operation.

    Paramters
    ---------
    x : the input vector
    W : the weight variable matrix for the convolution
    stride : the stride for the convolution
    name : the name for the convolution operation

    Returns
    -------
    out : a TensorFlow 2D convolution operation
    """
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME",
                        name=name)

def max_pool_2x2(x, name=None):
    """Return a TensorFlow 2x2 max pooling operation.

    Parameters
    ----------
    x : the input vector
    name : the name for the operation
    
    Returns
    -------
    out : a TensorFlow 2x2 maxpool operations
    """
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                          padding = "SAME", name=name)

def create_q_fn():
    """Create the action-value function.

    Constructs the operations and placeholders that form the policy
    function estimator. This consists of three convolutional layers
    followed by two fully connected layers. The last layer is the
    action output. Returns a tuple with the state TensorFlow
    placeholder and the readout operation (the result of the network).

    Returns
    -------
    out : a tuple with the state input placeholder and the Q function
          operation
    """
    # Input placeholder
    s = tf.placeholder("float", [None, 84, 84, 4])

    # Network weights
    with tf.name_scope("q_star"):
        with tf.name_scope("variables"):
            W_conv1 = weight_variable([8, 8, 4, 32], name="W_conv1")
            b_conv1 = bias_variable([32], name="b_conv1")
            W_conv2 = weight_variable([4, 4, 32, 64], name="W_conv2")
            b_conv2 = bias_variable([64], name="b_conv2")
            W_conv3 = weight_variable([3, 3, 64, 64], name="W_conv3")
            b_conv3 = bias_variable([64], name="b_conv3")
            W_fc1 = weight_variable([2304, 512], name="W_fc1")
            b_fc1 = bias_variable([512], name="b_fc1")
            W_fc2 = weight_variable([512, ACTIONS], name="W_fc2")
            b_fc2 = bias_variable([ACTIONS], name="b_fc2")

        # Network operations
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1, name="conv1")
        h_pool1 = max_pool_2x2(h_conv1, name="pool1")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2, name="conv2")
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3, name="conv3")
        h_conv3_flat = tf.reshape(h_conv3, [-1, 2304], name="flatten")
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="fc1")
        q = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="fc2")

    return s, q

def create_loss_fn(q_star):
    """Create an operation to compute the loss function.

    Parameters
    ----------
    q_star : estimate of the action-value function

    Returns
    -------
    out : a tuple with an action placeholder, a target placeholder, and
          the loss function operation
    """
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    action = tf.reduce_sum(tf.mul(q_star, a), reduction_indices = 1)
    loss = tf.reduce_mean(tf.square(y - action))

    return (a, y, loss)

def initialize_state(game_state):
    """Create the initial game state.

    This function takes the first frame of the game, and stacks it four
    times to create the initial state.

    Parameters
    ----------
    game_state : the interface to the game

    Returns
    -------
    out : the initial game state.
    """
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
    # _, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    return s_t

def update_state(s_t, x_t1_col):
    """Update the state with the latest frame.

    Parameters
    ----------
    s_t : the current state
    x_t1_col : the new (color) game frame

    Returns
    -------
    out : the updated state
    """
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (84, 84)), cv2.COLOR_BGR2GRAY)
    #_, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (84, 84, 1))
    s_t1 = np.append(x_t1, s_t[:,:,:3], axis = 2)
    return s_t1

def train_network(s, q_star, sess):
    """Train the policy estimator.

    Parameters
    ----------
    s : the TensorFlow placeholder for the state.
    q : the output of the policy estimator.
    sess : the TensorFlow session to use for the computation.
    """
    epsilon = INITIAL_EPSILON
    global_step = tf.Variable(0, name="global_step", trainable=False)
    update_step = global_step.assign(tf.add(global_step, 1))

    # Define the loss function and training operation. The
    # placeholders a and y will be fed from the batch samples we take
    # at each step.
    a, y, loss = create_loss_fn(q_star)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Configure things to save and load trained networks.
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(CHKPT_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        epsilon = FINAL_EPSILON
        print("Loaded %s (step %d)." %
              (checkpoint.model_checkpoint_path, global_step.eval(session=sess)))
    else:
        print("No checkpoints found.")

    # Get the first state by doing nothing, preprocessing the image to 80x80x4,
    # and then stacking up four images.
    game_state = game.GameState()
    s_t = initialize_state(game_state)

    # Create the experience replay buffer.
    D = deque(maxlen=REPLAY_MEMORY)
    t = global_step.eval(session=sess)

    ep_t = 0
    ep = 0
    ep_reward = 0
    ep_q = 0.0
    
    while True:
        # Choose an action epsilon greedily. A random action is chosen
        # with probability epsilon *or* if we are still observing the
        # game.
        q_star_t = q_star.eval(feed_dict = {s : [s_t]}, session=sess)[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or len(D) <= OBSERVE:
            action_index = random.randrange(ACTIONS)
        else:
            action_index = np.argmax(q_star_t)
        a_t[action_index] = 1

        ep_q += np.max(q_star_t)

        # After the observation period, we gradually scale the epsilon
        # down from 1 to the amount set for training.
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # Run the selected action and observe next state, reward, and
            # whether or not the resulting state is terminal.
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)

            # Preprocess the image and build the state stack.
            s_t1 = update_state(s_t, x_t1_col)

            # Store the transition in the experience replay buffer (D).
            D.append((s_t, a_t, r_t, s_t1, terminal))

            ep_reward += r_t

        # Only train if done observing.
        if t > OBSERVE:
            # Sample a minibatch to train on from the experience replay buffer.
            minibatch = random.sample(list(D), BATCH)

            # Get the batch variables: state, action, reward, next state, and
            # whether or not the game terminated.
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            terminal_batch = [d[4] for d in minibatch]

            y_batch = []
            q_star_j1_batch = q_star.eval(feed_dict = {s : s_j1_batch}, session=sess)
            for i in range(0, len(minibatch)):
                if terminal_batch[i]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(q_star_j1_batch[i]))

            # Perform gradient step.
            train_step.run(feed_dict = {y: y_batch, a: a_batch, s: s_j_batch},
                           session=sess)

        # If the state was terminal, print some information.
        if terminal:
            ep += 1
            print("t=%d, episode=%d, epsilon=%f, r_total=%d, Q_avg=%e" %
                  (t, ep, epsilon, ep_reward, ep_q / (t - ep_t)))
            ep_reward = 0
            ep_q = 0
            ep_t = t

        # Update the old values.
        s_t = s_t1
        t = update_step.eval(session=sess)

        # save progress every 10000 iterations
        if t % 100000 == 0:
            saver.save(sess, CHKPT_DIR + os.sep + GAME, global_step=global_step)

def play_game(s, readout, sess):
    # Open up a game state to communicate with emulator.
    game_state = game.GameState()

    # Get the first state by doing nothing and preprocess the image to 84x84x4.
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(CHKPT_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not restore network weights. Can't play.")
        sys.exit(-1)

    t = 0
    while True:
        # choose an action epsilon greedily
        q_star_t = q_star.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = np.argmax(q_star_t)
        a_t[action_index] = 1

        for i in range(0, K):
            # run the selected action and observe next state and reward
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (84, 84)), cv2.COLOR_BGR2GRAY)
            _, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (84, 84, 1))
            s_t1 = np.append(s_t[:,:,1:], x_t1, axis = 2)

        # update the old values
        s_t = s_t1
        t += 1

def main(play_only):
    sess = tf.Session()
    print("Creating network...")
    s, q_star = create_q_fn()
    if not play_only:
        print("Training...")
        train_network(s, q_star, sess)
    else:
        print("Playing using trained network...")
        play_game(s, q_star, sess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--play", help="Run network in play mode only", action="store_true")
    parser.add_argument("-r", "--run-suffix", help="Suffix for checkpoints directory.")
    parser.add_argument("-k", "--skip-act", help="Interval for choosing new action", type=int, default=4)
    args = parser.parse_args()

    # Set up name of directory where we save checkpoints.
    if args.run_suffix is not None:
        CHKPT_DIR += ("-" + args.run_suffix)
    print("Saving checkpoints to %s" % (CHKPT_DIR,))

    K = args.skip_act

    main(args.play)
