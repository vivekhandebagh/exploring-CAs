import os
# import torch
import cellpylib as cpl
import numpy as np
from matplotlib import pyplot as plt
import random as rand


class FutureLightCone:

    def __init__(self, horizon, spacetime, t, x):
        # self.lightcone = torch.zeros((horizon + 1) ** 2)
        self.d_internal = np.zeros((horizon + 1) ** 2 - 1)
        self.lightcone = self.get_lightcone_realization(spacetime=spacetime, x=x, t=t, horizon=horizon)

    def get_lightcone_realization(self, spacetime, x, t, horizon):

        t_length = len(spacetime)
        lc = np.zeros((horizon + 1) ** 2 - 1)

        # this function also calculates internal distance vector as well
        index = 0
        for delta in range(1, horizon + 1):

            if t + delta >= t_length:
                break

            state = spacetime[t + delta]
            window, spatial_distances = self.get_window(state, window_size=1 + 2 * delta, x_pos=x)

            lc[index: index + len(window)] = window

            for sigma in spatial_distances:
                self.d_internal[index] = np.sqrt(sigma ** 2 + delta ** 2)
                index += 1

        return tuple(lc)

    def lightcone_distance(self, lightcone, tau=1):
        # check if lightcones have same length

        distance = 0
        for i in range(len(self.lightcone)):
            distance += np.e ** (-tau * self.d_internal[i]) * (self.lightcone[i] - lightcone[i]) ** 2

        distance = np.sqrt(distance)
        return distance

    def get_window(self, spacetime, window_size, x_pos):

        ca_size = len(spacetime)

        # Calculate the starting and ending indices for the columns

        start_col = (x_pos - window_size // 2) % ca_size
        end_col = (x_pos + window_size // 2 + 1) % ca_size

        # also calculate internal distance:
        spatial_distance = []
        for i in range(x_pos - window_size // 2, x_pos + window_size // 2 + 1):
            sigma = np.abs(x_pos - i)
            spatial_distance.append(sigma)

        # Slice the CA state
        if start_col <= end_col:
            window = spacetime[start_col:end_col]
        else:
            window = np.hstack((spacetime[start_col:], spacetime[:end_col]))

        # window = np.from_numpy(window)
        return window, np.array(spatial_distance)




class PastLightCone:

    def __init__(self, horizon, spacetime, t, x):
        # self.lightcone = torch.zeros((horizon + 1) ** 2)
        self.d_internal = np.zeros((horizon + 1) ** 2)
        self.lightcone = self.get_lightcone_realization(spacetime=spacetime, x=x, t=t, horizon=horizon)

    def get_lightcone_realization(self, spacetime, x, t, horizon):

        lc = np.zeros((horizon + 1) ** 2)

        # this function also calculates internal distance vector as well
        index = 0
        for delta in range(horizon + 1):

            if t - delta < 0:
                break

            state = spacetime[t - delta]
            window, spatial_distances = self.get_window(state, window_size=1 + 2 * delta, x_pos=x)

            lc[index: index + len(window)] = window

            for sigma in spatial_distances:
                self.d_internal[index] = np.sqrt(sigma ** 2 + delta ** 2)
                index += 1

        return tuple(lc)

    def lightcone_distance(self, lightcone, tau=1):
        # check if lightcones have same length

        distance = 0
        for i in range(len(self.lightcone)):
            distance += np.e ** (-tau * self.d_internal[i]) * (self.lightcone[i] - lightcone[i]) ** 2

        distance = np.sqrt(distance)
        return distance

    def get_window(self, spacetime, window_size, x_pos):

        ca_size = len(spacetime)

        # Calculate the starting and ending indices for the columns

        start_col = (x_pos - window_size // 2) % ca_size
        end_col = (x_pos + window_size // 2 + 1) % ca_size

        # also calculate internal distance:
        spatial_distance = []
        for i in range(x_pos - window_size // 2, x_pos + window_size // 2 + 1):
            sigma = np.abs(x_pos - i)
            spatial_distance.append(sigma)

        # Slice the CA state
        if start_col <= end_col:
            window = spacetime[start_col:end_col]
        else:
            window = np.hstack((spacetime[start_col:], spacetime[:end_col]))

        # window = torch.from_numpy(window)
        return window, np.array(spatial_distance)
