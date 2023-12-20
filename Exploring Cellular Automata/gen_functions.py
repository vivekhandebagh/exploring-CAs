import torch
import cellpylib as cpl
import numpy as np
import matplotlib.pyplot as plt
# from causal_states import get_causal_states


def create_elementary_ca(ca_size, t_steps, rule_number, transients=False, ic=None):
    if ic is None:
        ic = cpl.init_random(ca_size)
    if transients:
        ca = cpl.evolve(ic, timesteps=t_steps, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule_number),
                        memoize=True)
    else:
        ca = cpl.evolve(ic, timesteps=t_steps+9, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule_number), memoize=True)
        # remove transients
        ca = ca[9:]
    return ca


def checkerboard_ca(ca_size):
    tile = np.array([0, 1])
    checkerboard_ic = np.tile(tile, ca_size//2)
    checkerboard = cpl.evolve(np.array([(checkerboard_ic)]), timesteps=ca_size, apply_rule=lambda n, c, t: cpl.nks_rule(n, 225))
    return checkerboard


def single_cell_elementary_ca(ca_size, rule_number):
    center = ca_size//2
    checkerboard_ic = np.zeros(ca_size)
    checkerboard_ic[center] = 1
    checkerboard = cpl.evolve(np.array([(checkerboard_ic)]), ca_size, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule_number))
    return checkerboard

