# Main source: https://github.com/Zeta36/muzero/blob/master/muzero.ipynb
# Not perfect, might be a good idea to try regularization

import numpy as np
import tensorflow as tf
import itertools

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    if __name__ == '__main__':
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# board is represented by np.zeros([3, 3]), one player plays with 1s the other -1s
board_size = (3, 3)
needed_to_win = 3

def array_str(array_with_minus_zero):
    return np.array_str(np.array(array_with_minus_zero)).replace('-0', ' 0').replace('.', '')

# only works for 3x3 board now
def is_won(board):
    # rows and columns
    if np.any(np.sum(board, axis=0) == 3) or np.any(np.sum(board, axis=0) == -3) or np.any(np.sum(board, axis=1) == 3) or np.any(np.sum(board, axis=1) == -3):
        return True
    # diagonals
    if board[0, 0] + board[1, 1] + board[2, 2] in [-3, 3] or board[0, 2] + board[1, 1] + board[2, 0] in [-3, 3]:
        return True
    return False

def switch_players(board): # kinda useless but looks nicer
    return board * -1

def make_a_move(board, action):
    board[action // board_size[1], action % board_size[1]] = 1
    return board

def create_initial(summarize=False):
    input_board = tf.keras.Input(board_size)
    hidden = tf.keras.layers.Flatten()(input_board)
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    hidden = tf.keras.layers.Dense(np.product(board_size), activation='tanh')(hidden) # also try sin and relu

    model = tf.keras.Model(input_board, hidden, name='Initial')

    if summarize:
        model.summary()

    return model

def create_recurrent(summarize=False):
    input_hidden, input_action = tf.keras.Input((np.product(board_size),)), tf.keras.Input(board_size)
    #hidden = tf.keras.backend.stack([input_hidden, input_action], axis=0)
    input_hidden_reshaped = tf.keras.layers.Reshape((board_size[0], board_size[1], 1))(input_hidden)
    input_action_reshaped = tf.keras.layers.Reshape((board_size[0], board_size[1], 1))(input_action)
    hidden = tf.keras.layers.Concatenate(axis=-1)([input_hidden_reshaped, input_action_reshaped])
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    next_hidden = tf.keras.layers.Dense(np.product(board_size), activation='tanh')(hidden) # also try sin and relu
    reward = tf.keras.layers.Dense(1, activation='tanh')(hidden)

    model = tf.keras.Model([input_hidden, input_action], [next_hidden, reward], name='Recurrent')

    if summarize:
        model.summary()

    return model

def create_prediction(summarize=False):
    input_hidden = tf.keras.Input((np.product(board_size),))
    hidden = tf.keras.layers.Flatten()(input_hidden)
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    policy = tf.keras.layers.Dense(np.product(board_size), activation='softmax')(hidden)
    value = tf.keras.layers.Dense(1, activation='tanh')(hidden)

    model = tf.keras.Model(input_hidden, [policy, value], name='Prediction')

    if summarize:
        model.summary()

    return model

class RecurrentRNNCell(tf.keras.layers.Layer):
    def __init__(self, recurrent_model):
        super(RecurrentRNNCell, self).__init__()
        self.model = recurrent_model
        self.state_size = 9

    def call(self, x, s): # we actaully don't care about the last state, but want to repeat the first one
        next_hidden, reward = self.model([s[0], x])
        return [s[0], reward], [next_hidden] # [next_hidden, reward], [next_hidden]

def build_training_ensemble(initial_model, rnn, prediction_model):
    board_input, action_sequence = tf.keras.Input(board_size), tf.keras.Input([None, board_size[0], board_size[1]])

    initial_hidden = initial_model(board_input)
    hiddens, rewards = rnn(action_sequence, initial_state=initial_hidden)
    policy, value = prediction_model(hiddens[-1])

    #training_model = tf.keras.Model([board_input, action_sequence], [rewards, policy, value], name='Training_Ensemble')
    #training_model.compile(tf.keras.optimizers.Adam(0.001), [tf.keras.losses.MeanSquaredError(), tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()])
    # i decided to ignore rewards for now (note that there are is a missmatch between labels and predictions generated position
    training_model = tf.keras.Model([board_input, action_sequence], [value, policy], name='Training_Ensemble')
    training_model.compile(tf.keras.optimizers.Adam(0.001),
                           [tf.keras.losses.MeanSquaredError(), tf.keras.losses.CategoricalCrossentropy()])

    training_model.summary()

    return training_model

class RecurrentPredictionRNNCell(tf.keras.layers.Layer):
    def __init__(self, recurrent_model, prediction_model):
        super(RecurrentPredictionRNNCell, self).__init__()
        self.recurrent_model = recurrent_model
        self.prediction_model = prediction_model
        self.state_size = 9

    def call(self, x, s): # we actaully don't care about the last state, but want to repeat the first one
        """
        next_hidden, reward = self.recurrent_model([s[0], x])
        policy, value = self.prediction_model(next_hidden)
        return [next_hidden, reward, policy, value], [next_hidden]
        """
        policy, value = self.prediction_model(s[0])
        next_hidden, reward = self.recurrent_model([s[0], x])
        return [s[0], reward, policy, value], [next_hidden]

def build_training_ensemble_with_each_step_prediction(initial_model, rnn):
    board_input, action_sequence = tf.keras.Input(board_size), tf.keras.Input([None, board_size[0], board_size[1]])

    initial_hidden = initial_model(board_input)
    hiddens, rewards, policies, values = rnn(action_sequence, initial_state=initial_hidden)

    #training_model = tf.keras.Model([board_input, action_sequence], [rewards, policies, values], name='Each_Step_Training_Ensemble')
    #training_model.compile(tf.keras.optimizers.Adam(0.001), [tf.keras.losses.MeanSquaredError(), tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()])
    training_model = tf.keras.Model([board_input, action_sequence], [values, policies],  name='Each_Step_Training_Ensemble')
    training_model.compile(tf.keras.optimizers.Adam(0.001),
                           [tf.keras.losses.MeanSquaredError(), tf.keras.losses.CategoricalCrossentropy()])

    training_model.summary()

    return training_model

class Network:
    def __init__(self):
        self.initial = create_initial(True)
        self.recurrent = create_recurrent(True)
        self.prediction = create_prediction(True)

        rnn = tf.keras.layers.RNN(RecurrentRNNCell(self.recurrent), return_sequences=True)
        self.training_ensemble = build_training_ensemble(self.initial, rnn, self.prediction)

        rnn_each_step = tf.keras.layers.RNN(RecurrentPredictionRNNCell(self.recurrent, self.prediction), return_sequences=True)
        self.training_ensemble_each_step = build_training_ensemble_with_each_step_prediction(self.initial, rnn_each_step)

def generate_possible_moves(board): # as indices of flattened board array
    possible_moves = []
    for i in range(np.product(board_size)):
        if board.flatten()[i] == 0:
            possible_moves.append(i)

    return possible_moves

def one_hot_action(action):
    board = np.zeros(board_size)
    return make_a_move(board, action)

# initializing search tree
Q = {}  # state-action values
Nsa = {}  # number of times certain state-action pair has been visited
Ns = {}   # number of times state has been visited
W = {}  # number of total points collected after taking state action pair
P = {}  # initial predicted probabilities of taking certain actions in state

cpuct = 4 # constant weighting factor

def mcts(hidden_state, history, model):
    possible_moves = list(range(np.product(board_size))) # all of the moves

    if len(history) == 9: # this is a slight cheat, an attempt to to train faster, limiting MCTS depth should work also
        policy, v = model.prediction(hidden_state.reshape([1, 9]))
        policy = policy[0, :]
        v = v[0, 0]

        return -v
    else:
        if not array_str(history) in P:

            policy, v = model.prediction(hidden_state.reshape([1, 9]))
            policy = policy[0, :]
            v = v[0, 0]

            P[array_str(history)] = policy
            Ns[array_str(history)] = 1

            for action in possible_moves:
                Q[(array_str(history), action)] = 0
                Nsa[(array_str(history), action)] = 0
                W[(array_str(history), action)] = 0

            return -v

        else:
            biggest_reward = -np.Inf
            best_action = None

            for action in possible_moves:
                reward = Q[(array_str(history), action)] + cpuct * P[array_str(history)][action] * np.sqrt(Ns[array_str(history)]) / (1 + Nsa[(array_str(history), action)])

                if reward > biggest_reward:
                    biggest_reward = reward
                    best_action = action

            next_hidden_state, r = model.recurrent.predict([np.copy(hidden_state).reshape(1, 9), one_hot_action(best_action).reshape(1, 3, 3)])
            next_hidden_state, r = next_hidden_state[0], r[0]

            v = mcts(switch_players(next_hidden_state), history + [best_action], model)

            W[(array_str(history), best_action)] += v
            Ns[array_str(history)] += 1
            Nsa[(array_str(history), best_action)] += 1
            Q[(array_str(history), best_action)] = W[(array_str(history), best_action)] / Nsa[(array_str(history), best_action)]
            return -v

def get_action_probs(board, model, starting_action_history):
    for i in range(25):
        _ = mcts(model.initial.predict(np.copy(board).reshape([1, 3, 3]))[0], starting_action_history, model)

    action_probs = np.zeros(np.product(board_size))

    for action in generate_possible_moves(board):
        action_probs[action] = Nsa[(array_str(starting_action_history), action)] / Ns[array_str(starting_action_history)] + 1e-7

    return action_probs

def ones_where_possible(board):
    mock_policy = np.zeros(9)

    mock_policy[generate_possible_moves(board)] = 1

    return mock_policy

def play_a_game(model):
    finished = False
    move_memory = []
    odd_or_even_counter = 0
    board = np.zeros(board_size)
    starting_action_history = []

    while not finished:
        policy = get_action_probs(np.copy(board), model, starting_action_history)
        policy = policy/np.sum(policy)

        action = np.random.choice(len(policy), p=policy)

        move_memory.append([board, policy, action])

        board = make_a_move(np.copy(board), action)

        starting_action_history.append(action)

        if is_won(board):
            if odd_or_even_counter % 2 == 0:
                move_memory.append([np.copy(board), ones_where_possible(board), 0])
                game_memory = (move_memory, 1)
            else:
                move_memory.append([np.copy(board), ones_where_possible(board), 0])
                game_memory = (move_memory, -1)

            return game_memory

        if not generate_possible_moves(board):
            move_memory.append([np.copy(board), np.zeros(9), 0])
            game_memory = (move_memory, 0)

            return game_memory

        odd_or_even_counter += 1
        board = switch_players(board)

def play_many_games(n, model):
    replay_memory = []

    for game in range(n):
        replay_memory.append(play_a_game(model))

    return replay_memory

# since the game of tic tac toe is symmetrical I have used this little 'trick' which speeds up training
def augmentate(state, value, policy, action):
    augmented_states, augmented_values,  augmented_policies, augmented_actions = [], [], [], []

    policy_rectangle = policy.reshape(board_size)
    augmented_state = state
    augmented_action = action

    for side in range(4): # rotate 4 times to also append the original
        augmented_state = np.rot90(augmented_state)
        policy_rectangle = np.rot90(policy_rectangle)
        augmented_action = np.rot90(augmented_action)

        augmented_states.append(augmented_state)
        augmented_values.append(value)
        augmented_policies.append(policy_rectangle.flatten())
        augmented_actions.append(augmented_action)

    augmented_state = np.flip(augmented_state, axis=0)
    policy_rectangle = np.flip(policy_rectangle, axis=0)
    augmented_action = np.flip(augmented_action, axis=0)
    augmented_states.append(augmented_state)
    augmented_values.append(value)
    augmented_policies.append(policy_rectangle.flatten())
    augmented_actions.append(augmented_action)

    augmented_state = np.flip(augmented_state, axis=1)
    policy_rectangle = np.flip(policy_rectangle, axis=1)
    augmented_action = np.flip(augmented_action, axis=1)
    # duplicate with rotation
    #augmented_states.append(augmented_state)
    #augmented_values.append(value)
    #augmented_policies.append(policy_rectangle.flatten())
    #augmented_actions.append(augmented_action)

    augmented_state = np.flip(augmented_state, axis=0)
    policy_rectangle = np.flip(policy_rectangle, axis=0)
    augmented_action = np.flip(augmented_action, axis=0)
    augmented_states.append(augmented_state)
    augmented_values.append(value)
    augmented_policies.append(policy_rectangle.flatten())
    augmented_actions.append(augmented_action)

    return np.array(augmented_states), np.array(augmented_values), np.array(augmented_policies), np.array(augmented_actions)

def augmentate_sequences(states_batch, values_batch, policies_batch, actions_batch):
    batch_size = actions_batch.shape[0]
    sequence_len = actions_batch.shape[1]

    augmented_states_batch = []
    augmented_values_batch = []
    augmented_policies_batch = []
    augmented_actions_batch = []

    for state, values_sequence, policies_sequence, actions_sequence in zip (states_batch, values_batch, policies_batch, actions_batch):
        augmented_states_sequence = [[] for _ in range(6)]
        augmented_values_sequence = [[] for _ in range(6)]
        augmented_policies_sequence = [[] for _ in range(6)]
        augmented_actions_sequence = [[] for _ in range(6)]

        for value, policy, action in zip(values_sequence, policies_sequence, actions_sequence):
            augmented_states, augmented_values, augmented_policies, augmented_actions = augmentate(state, value, policy, action)

            for i in range(6):
                if not augmented_states_sequence[i]:
                    augmented_states_sequence[i].append(augmented_states[i])
                augmented_values_sequence[i].append(augmented_values[i])
                augmented_policies_sequence[i].append(augmented_policies[i])
                augmented_actions_sequence[i].append(augmented_actions[i])

        augmented_states_batch.append(augmented_states_sequence)
        augmented_values_batch.append(augmented_values_sequence)
        augmented_policies_batch.append(augmented_policies_sequence)
        augmented_actions_batch.append(augmented_actions_sequence)

    augmented_states_batch = np.array(augmented_states_batch).reshape(batch_size*6, 3, 3)
    augmented_values_batch = np.array(augmented_values_batch).reshape(batch_size*6, sequence_len)
    augmented_policies_batch = np.array(augmented_policies_batch).reshape(batch_size*6, sequence_len, 9)
    augmented_actions_batch = np.array(augmented_actions_batch).reshape(batch_size*6, sequence_len, 3, 3)

    #print(augmented_states_batch.shape, augmented_values_batch.shape, augmented_policies_batch.shape, augmented_actions_batch.shape)

    return augmented_states_batch, augmented_values_batch, augmented_policies_batch, augmented_actions_batch




def train_model(model, game_memory):
    # instead of sampling I use all the generated data

    # not the best way to use generators follows
    game_sequences = []
    for game, result in game_memory:
        states = []
        values = []
        policies = []
        actions = []

        for i, (state, policy, action) in enumerate(game):
            states.append(state)
            policies.append(policy)
            actions.append(one_hot_action(action))

            if (i%2 == 0 and result == 1) or (i%2 == 1 and result == -1):
                values.append(1)
            elif (i%2 == 0 and result == -1) or (i%2 == 1 and result == 1):
                values.append(-1)
            else:
                assert  result == 0
                values.append(0)

        game_sequences.append([states, values, policies, actions])

    states_sequences_by_len = [[] for _ in range(10)]
    values_sequences_by_len = [[] for _ in range(10)]
    policies_sequences_by_len = [[] for _ in range(10)]
    actions_sequences_by_len = [[] for _ in range(10)]
    for i in range(10): # 10 == max len(game) + 1
        for states, values, policies, actions in game_sequences:
            assert len(states) == len(values) and len(values) == len(policies) and len(actions) == len(policies) # just some development stuff
            if len(states) > i:
                # assert len(states)-i == len(states[i:]) # just some development stuff
                pos = len(states)-i-1
                states_sequences_by_len[pos].append(np.array(states[i]))
                values_sequences_by_len[pos].append(np.array(values[i:]))
                policies_sequences_by_len[pos].append(np.array(policies[i:]))
                actions_sequences_by_len[pos].append(np.array(actions[i:]))

    for i in range(10):
        states_sequences_by_len[i] = np.array(states_sequences_by_len[i])
        values_sequences_by_len[i] = np.array(values_sequences_by_len[i])
        policies_sequences_by_len[i] = np.array(policies_sequences_by_len[i])
        actions_sequences_by_len[i] = np.array(actions_sequences_by_len[i])

    def generator():
        for state_sequence, value_sequence, policy_sequence, action_sequence in zip(states_sequences_by_len, values_sequences_by_len, policies_sequences_by_len, actions_sequences_by_len):
            if state_sequence.shape[0]:
                state_sequence, value_sequence, policy_sequence, action_sequence = augmentate_sequences(state_sequence, value_sequence, policy_sequence, action_sequence)

                #print(state_sequence.shape, action_sequence.shape, value_sequence.shape, policy_sequence.shape, sep='\n')
                #for state_seq, action_seq, value_seq, policy_seq in zip(state_sequence, action_sequence, value_sequence, policy_sequence):
                #    print(state_seq, action_seq, value_seq, policy_seq, sep='\n')
                #assert action_seq.shape[0] == 1

                yield [state_sequence, action_sequence], [value_sequence, policy_sequence]

    model.training_ensemble_each_step.fit(itertools.cycle(generator()), epochs=1000, steps_per_epoch=10, verbose=2) #2

def train():
    global Q, Nsa, Ns, W, P

    model = Network()

    #model.training_ensemble_each_step.load_weights('model_1.ckpt')

    for episode in range(0, 50): #100
        print('Episode:', episode)

        replay_memory = play_many_games(10, model) #2
        train_model(model, replay_memory)

        Q = {}
        Nsa = {}
        Ns = {}
        W = {}
        P = {}

        model.training_ensemble_each_step.save_weights('model_1.ckpt')
        print('saving model')

def play():
    model = Network()
    model.training_ensemble_each_step.load_weights('model_1.ckpt').expect_partial()

    board = np.zeros(board_size)
    history = []

    while True:
        print(board)

        X, Y = [int(i) for i in input('X Y:').split()]
        board[X, Y] = -1
        history.append(X*3+Y)

        if is_won(board):
            print(board)
            print('game over')
            break

        if True: # use mcts with nn to predict
            policy = get_action_probs(board, model, history)
            policy = policy/np.sum(policy)
        if False: # use pure nn
            policy, _ = model.prediction(model.initial(np.reshape(board, [-1, 3, 3])))
            policy = policy[0]

            #print(policy, _)

            possible_moves = generate_possible_moves(board)

            policy = [policy[i] if i in possible_moves else 0 for i in range(np.product(board_size))]
            policy = policy/np.sum(policy)

        action = np.argmax(policy)
        board = make_a_move(board, action)
        history.append(action)

        if is_won(board):
            print(board)
            print('somebody won')
            break

        if not generate_possible_moves(board):
            print('game over')
            break

train()
#play()
