# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:56:14 2023

@author: david.hevisov
"""

import numpy as np
import ctypes

class Photons:

    def __init__(self, N_threads):

        # Define the struct-like Photon class

        photon_type = np.dtype([
        ('dir', np.float32, 4),
        ('zpos', np.float32),
        ("weight", np.float32),
        ("scat_counter", np.uint32)
        ])

        self.photons = np.zeros(N_threads, dtype=photon_type)

class Sim_Parameters:

    def __init__(self,cfg):

        self.mua = float(cfg["sim_parameters"]["mua"])
        self.mus = float(cfg["sim_parameters"]["mus"])
        self.g = float(cfg["sim_parameters"]["g"])

class Tyche_i_state(ctypes.Structure):

    _fields_ = [("a", ctypes.c_uint32),
                ("b", ctypes.c_uint32),
                ("c", ctypes.c_uint32),
                ("d", ctypes.c_uint32)]

