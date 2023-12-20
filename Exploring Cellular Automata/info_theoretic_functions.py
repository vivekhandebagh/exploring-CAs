import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cellpylib as cpl


def get_window_1d(ca_state, window_size, pos):
    ca_size = len(ca_state)

    # Calculate the starting and ending indices for the columns
    start_col = (pos - window_size // 2) % ca_size
    end_col = (pos + window_size // 2 + 1) % ca_size

    # Slice the CA state
    if start_col <= end_col:
        window = ca_state[start_col:end_col]
    else:
        window = np.hstack((ca_state[start_col:], ca_state[:end_col]))

    window = torch.from_numpy(window)
    return window


def get_window_2d(ca_state, size, pos):
    # Get the dimensions of the CA state
    rows, cols = ca_state.shape

    # Calculate the starting and ending indices for the rows
    start_row = (pos[0] - size // 2) % rows
    end_row = (pos[0] + size // 2 + 1) % rows

    # Calculate the starting and ending indices for the columns
    start_col = (pos[1] - size // 2) % cols
    end_col = (pos[1] + size // 2 + 1) % cols

    # Slice the CA state
    if start_row <= end_row:
        window_rows = ca_state[start_row:end_row, :]
    else:
        window_rows = np.vstack((ca_state[start_row:, :], ca_state[:end_row, :]))

    if start_col <= end_col:
        window = window_rows[:, start_col:end_col]
    else:
        window = np.hstack((window_rows[:, start_col:], window_rows[:, :end_col]))

    window = np.array(window).flatten()
    return window


def block_entropy(data, L):
    i = 0
    total_count = len(data)
    counts = {}
    probs = {}

    while i < total_count:

        # If the window goes past the edge, it wraps back around
        if i + L > total_count:
            window = np.hstack((data[i:], data[0:i + L - total_count]))

        else:
            window = data[i:i + L]

        # make window into string
        window = [str(j) for j in window]
        block_str = ''.join(window)

        # update counts
        if block_str not in counts:
            counts[block_str] = 1
            probs[block_str] = 0
        else:
            counts[block_str] += 1

        i += 1

    # get total number of blocks
    sum = 0
    for block in counts:
        sum += counts[block]

    # calculate the probability of each block occuring
    for block in counts:
        probs[block] = float(counts[block]) / float(sum)

    # entropy formula
    H = 0
    for block in probs:
        H -= probs[block] * np.log2(probs[block])

    return H


def get_block_entropy_feature_vector(state, window_size, L):
    entropy_feature_vector = []
    for x in range(len(state)):
        window = get_window_1d(state, window_size, x)
        entropy_feature_vector.append(block_entropy(window, L))

    entropy_feature_vector = np.array(entropy_feature_vector)

    max = np.max(entropy_feature_vector)
    min = np.min(entropy_feature_vector)

    entropy_feature_vector_normalized = (entropy_feature_vector - min) / (max - min)

    return entropy_feature_vector_normalized


def get_block_entropy_feature_matrix(ca_state, window_size):
    rows = len(ca_state)
    columns = len(ca_state[0])
    entropy_feature_matrix = np.zeros((rows, columns))
    for y in range(rows):
        for x in range(columns):
            window = get_window_2d(ca_state, window_size, (x, y))
            entropy_feature_matrix[y][x] = cpl.shannon_entropy(window.tolist())

    max = np.max(entropy_feature_matrix)
    min = np.min(entropy_feature_matrix)

    entropy_feature_matrix_normalized = (entropy_feature_matrix - min) / (max - min)

    return entropy_feature_matrix_normalized


def mutual_information_feature_vector(state, prev_state, window_size):
    I = np.zeros(len(state))
    for x in range(len(state)):
        window_current = ''.join(str(_) for _ in get_window_1d(state, window_size=window_size, pos=x))
        window_prev = ''.join(str(_) for _ in get_window_1d(prev_state, window_size=window_size, pos=x))
        I[x] = cpl.mutual_information(window_current, window_prev)

    max = np.max(I)
    min = np.min(I)

    I_normalized = (I - min) / (max - min)

    return I_normalized


def get_mutual_information_feature_matrix(state, prev_state, window_size):
    rows = len(state)
    columns = len(state[0])
    I = np.zeros((rows, columns))
    for y in range(rows):
        for x in range(columns):
            window_current = ''.join(str(_) for _ in get_window_2d(state, window_size, (x, y)))
            window_prev = ''.join(str(_) for _ in get_window_2d(prev_state, window_size, (x, y)))

            I[y][x] = cpl.mutual_information(window_current, window_prev)

    max = np.max(I)
    min = np.min(I)

    I_normalized = (I - min) / (max - min)

    return I_normalized


def plot_3d_spacetime(ca_states):
    # Concatenate the CA states along the time axis
    ca_3d = np.stack(ca_states)

    # Get the dimensions of the CA states
    num_states, rows, cols = ca_3d.shape

    # Create a grid of coordinates
    X, Y, T = np.meshgrid(np.arange(cols), np.arange(rows), np.arange(num_states))

    # Mask the coordinates to extract the "1s"
    masked_x = X[ca_3d == 1]
    masked_y = Y[ca_3d == 1]
    masked_t = T[ca_3d == 1]

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot of "1s"
    ax.scatter3D(masked_x, masked_y, masked_t, c=masked_t, cmap='jet', marker='o')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    ax.set_title('3D Spacetime of Cellular Automata')

    # Set axis limits
    ax.set_xlim(0, cols - 1)
    ax.set_ylim(0, rows - 1)
    ax.set_zlim(0, num_states - 1)

    # Show the plot
    plt.show()

