# Main source: http://karpathy.github.io/2016/05/31/rl/

import numpy as np
import tensorflow as tf

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
    return np.array_str(array_with_minus_zero).replace('-0', ' 0').replace('.', '')

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

def create_model(summarize=False):
    input = tf.keras.Input(board_size)
    hidden = tf.keras.layers.Flatten()(input)
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    #hidden = tf.keras.layers.Dropout(0.2)(hidden)
    hidden = tf.keras.layers.Dense(np.product(board_size)*4, activation='tanh')(hidden) # also try sin and relu
    #hidden = tf.keras.layers.Dropout(0.2)(hidden)
    action = tf.keras.layers.Dense(np.product(board_size), activation='softmax')(hidden)
    reward = tf.keras.layers.Dense(1, activation='tanh')(hidden)

    model = tf.keras.Model(input, [action, reward])
    model.compile(tf.keras.optimizers.Adam(0.000001), [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()]) # , loss_weights=[1, 1]

    if summarize:
        model.summary()

    return model

def generate_possible_moves(board): # as indices of flattened board array
    possible_moves = []
    for i in range(np.product(board_size)):
        if board.flatten()[i] == 0:
            possible_moves.append(i)

    return possible_moves

def get_action_probs(board, model):
    action_probs = np.zeros(np.product(board_size))
    policy, _ = model.predict(board.reshape(-1, 3, 3))
    policy = policy[0]

    for action in generate_possible_moves(board):
        action_probs[action] = policy[action]
    #possible_moves = generate_possible_moves(board)
    #action_probs[possible_moves] = policy[possible_moves]

    return action_probs

def play_a_game(model):
    finished = False
    move_memory = []
    odd_or_even_counter = 0
    board = np.zeros(board_size)

    while not finished:
        policy = get_action_probs(np.copy(board), model)
        policy = policy/np.sum(policy)

        action = np.random.choice(len(policy), p=policy) # could be replaced by: with certain prob make the best move predicted by nn, make a random one otherwise

        one_hot = np.zeros(policy.shape)
        one_hot[action] = 1
        move_memory.append([board, one_hot])

        board = make_a_move(np.copy(board), action)

        if not generate_possible_moves(board):
            game_memory = (move_memory, 0)

            return game_memory

        if is_won(board):
            if odd_or_even_counter % 2 == 0:
                game_memory = (move_memory, 1)
            else:
                game_memory = (move_memory, -1)

            return game_memory

        odd_or_even_counter += 1
        board = switch_players(board)

def play_many_games(n, model):
    replay_memory = []

    for game in range(n):
        replay_memory.append(play_a_game(model))

    return replay_memory

# since the game of tic tac toe is symmetrical I have used this little 'trick' which speeds up training
def augmentate(states, values, policies):
    augmented_states, augmented_values,  augmented_policies = [], [], []

    for state, value, policy in zip(states, values, policies):
        policy_rectangle = policy.reshape(board_size)
        augmented_state = state

        for side in range(4):
            augmented_state = np.rot90(augmented_state)
            policy_rectangle = np.rot90(policy_rectangle)

            augmented_states.append(augmented_state)
            augmented_values.append(value)
            augmented_policies.append(policy_rectangle.flatten())

        augmented_state = np.flip(augmented_state, axis=0)
        policy_rectangle = np.flip(policy_rectangle, axis=0)
        augmented_states.append(augmented_state)
        augmented_values.append(value)
        augmented_policies.append(policy_rectangle.flatten())

        augmented_state = np.flip(augmented_state, axis=1)
        policy_rectangle = np.flip(policy_rectangle, axis=1)
        # duplicate with rotation
        #augmented_states.append(augmented_state)
        #augmented_values.append(value)
        #augmented_policies.append(policy_rectangle.flatten())

        augmented_state = np.flip(augmented_state, axis=0)
        policy_rectangle = np.flip(policy_rectangle, axis=0)
        augmented_states.append(augmented_state)
        augmented_values.append(value)
        augmented_policies.append(policy_rectangle.flatten())

    return augmented_states, augmented_values, augmented_policies

def train_model(model, game_memory, gamma = 1.):
    states = []
    values = []
    policies = []

    for game, result in game_memory:
        for i, (state, policy) in enumerate(game):
            states.append(state)

            if (i%2 == 0 and result == 1) or (i%2 == 1 and result == -1):
                #states.append(state)
                values.append(1)
                policies.append(policy * 1 * gamma ** (len(game) - i))
            elif (i%2 == 0 and result == -1) or (i%2 == 1 and result == 1):
                #pass
                values.append(-1)
                policies.append(policy * -1 * gamma ** (len(game) - i))
            else:
                assert  result == 0
                #states.append(state)
                values.append(0)
                policies.append(policy * 0 * gamma ** (len(game) - i))

    states, values, policies = augmentate(states, values, policies)

    model.fit(np.array(states), [np.array(policies), np.array(values).reshape([-1, 1])], epochs=1, verbose=1) #2

def train():
    model = create_model(True)

    #model.load_weights('model_1.ckpt')

    for episode in range(3000): #100
        print('Episode:', episode)

        replay_memory = play_many_games(10, model) #2
        train_model(model, replay_memory, 0.8)

        model.save_weights('model_1.ckpt')
        print('saving model')

def play():
    model = create_model(True)
    model.load_weights('model_1.ckpt').expect_partial()

    board = np.zeros(board_size)

    while True:
        print(board)

        X, Y = [int(i) for i in input('X Y:').split()]
        board[X, Y] = -1

        if is_won(board):
            print(board)
            print('game over')
            break

        policy, _ = model.predict(np.reshape(board, [-1, 3, 3]))
        policy = policy[0]

        #print(policy, _)

        possible_moves = generate_possible_moves(board)

        policy = [policy[i] if i in possible_moves else 0 for i in range(np.product(board_size))]
        policy = policy/np.sum(policy)

        action = np.argmax(policy)
        board = make_a_move(board, action)

        if is_won(board):
            print(board)
            print('somebody won')
            break

        if not generate_possible_moves(board):
            print('game over')
            break

train()
#play()
