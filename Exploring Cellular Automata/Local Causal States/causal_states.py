import os
import torch
import cellpylib as cpl
import numpy as np
from matplotlib import pyplot as plt
from lightcone import PastLightCone, FutureLightCone
import sys
from pgmpy.models import MarkovChain as MC

sys.path.append("D:/Data/Research/CA_Project/Exploring Cellular Automata")
from gen_functions import create_elementary_ca, checkerboard_ca, single_cell_elementary_ca

def get_lightcone_field(spacetime, horizon, want_PLC_coords=True):
    t_length = len(spacetime)
    x_length = len(spacetime[0])
    PLC_Coordinates = {}

    lightcones = np.empty((2, t_length, x_length), dtype=PastLightCone)

    for t in range(horizon, t_length - horizon):  # account for index errors

        for x in range(x_length):

            # calculate past and future lightcones at specified location
            plc = PastLightCone(horizon, spacetime, x=x, t=t)
            flc = FutureLightCone(horizon, spacetime, x=x, t=t)

            # store past and future lightcone objects by spacetime location
            lightcones[0][t][x] = plc
            lightcones[1][t][x] = flc

            # store spacetime locations of each distinct past lightcone

            # plc.lightcone instead of plc as every plc object is different, but we care about actual lightcone realization
            l_minus = plc.lightcone
            if l_minus not in PLC_Coordinates:
                PLC_Coordinates[l_minus] = [(t, x)]
            else:
                PLC_Coordinates[l_minus].append((t, x))

    if want_PLC_coords:
        return lightcones, PLC_Coordinates
    else:
        return lightcones


def get_causal_states(lightcones, PLC_Coordinates, horizon):
    Probability_Distributions = {}

    for l_minus in PLC_Coordinates:

        # print('for l_minus=', l_minus)

        L_plus = {}
        FLC_counts = {}
        seen_flc = []
        # sum = 0

        # get P(L+|l-)
        coordinates = PLC_Coordinates[l_minus]
        total = len(coordinates)
        for coord in coordinates:

            flc = lightcones[1][coord[0]][coord[1]]

            # count occurrences for each distinct l+|l-
            l_plus = tuple(flc.lightcone)

            if l_plus not in seen_flc:
                seen_flc.append(l_plus)
                FLC_counts[l_plus] = 1
            else:
                FLC_counts[l_plus] += 1
            # sum += 1

        # calculate P(L+|l-)
        for l_plus in FLC_counts:
            # normalize to calculate probability
            Probability_l_plus = round((FLC_counts[l_plus]) / total, 4)
            # print('Probability of l_plus given l_minus is:', Probability_l_plus)
            L_plus[l_plus] = Probability_l_plus

        # for each distinct l-, assign P(L+|l-)
        Probability_Distributions[l_minus] = L_plus

    # now we need to find the local causal states
    # we need some sort of mapping from past light cone to local causal state
    pr_to_causal_state = {}
    local_causal_states = {}
    causal_state = 0
    seen = []
    for plc in Probability_Distributions:
        p = frozenset(Probability_Distributions[plc])  # p = probability distribution of future light cones
        if p not in seen:
            seen.append(p)
            causal_state += 1
            pr_to_causal_state[p] = causal_state
            local_causal_states[causal_state] = [plc]
        else:
            local_causal_states[pr_to_causal_state[p]].append(plc)

    return local_causal_states


def causal_state_map(past_lightcone_field, local_causal_states, horizon):
    causal_state_map = np.zeros(past_lightcone_field.shape)
    t_length = len(past_lightcone_field)
    x_length = len(past_lightcone_field[0])

    for t in range(horizon, t_length - horizon):  # account for index errors
        for x in range(x_length):
            plc = past_lightcone_field[t][x]
            l_minus = plc.lightcone
            for causal_state in local_causal_states:
                if l_minus in local_causal_states[causal_state]:
                    causal_state_map[t][x] = causal_state

    # return causal_state_map

    plt.imshow(causal_state_map, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to indicate the values
    plt.show()


def get_state_machine(local_causal_states, past_lightcone_field, causal_state_map, horizon):
    t_length = len(causal_state_map)
    x_length = len(causal_state_map[0])

    transition_counts = {}
    total_counted_transitions = 0
    tm = {}

    for t in range(horizon, t_length - horizon):
        for x in range(x_length):
            state = causal_state_map[t][x]
            next_state = [t + 1][x]
            transition = (state, next_state)
            total_counted_transitions += 1

            if state in tm:
                if next_state in tm[state]:
                    tm[state][next_state] += 1
                else:
                    tm[state][next_state] = 1
            else:
                tm[state] = {}
                tm[state][next_state] = 1

    # need to normalize transition matrix
    for state in tm:
        total = sum(tm[state].values())
        for next_state in tm[state]:
            tm[state][next_state] /= total

    model = MC()
    model.add_variable('causal states', len(local_causal_states))
    model.add_transition_model('causal states', tm)



