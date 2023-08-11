# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:56:14 2023

@author: david.hevisov
"""

import numpy as np


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

    def __init__(self, cfg):

        self.mua = cfg["sim_parameters"]["mua"]
        self.mus = cfg["sim_parameters"]["mus"]
        self.g_factor = cfg["sim_parameters"]["g"]
        self.n_1 = cfg["sim_parameters"]["n1"]
        self.n_2 = cfg["sim_parameters"]["n2"]
        self.theta_ls = cfg["sim_parameters"]["theta_ls"]
        self.d_slab = cfg["sim_parameters"]["d_slab"]

    def calc_start_dir(self):
        """
        Calculate initial direction after refraction.
        """

        self.cost_start = np.cos(np.deg2rad(self.theta_ls))

        if (self.theta_ls > 0.000001):  # light source is not perpendicular to surface

            n_ratio = self.n_1/self.n_2

            self.cost_start = self.cost_start*n_ratio - (n_ratio*self.cost_start - np.sqrt(1 - n_ratio*n_ratio*(1 - self.cost_start*self.cost_start)))

    def calc_R_fresnel(self):

        n = self.n_2/self.n_1
        mu = self.cost_start
        mu_crit = np.sqrt(n*n-1)/n if n > 1.0 else 0.0

        if (mu > mu_crit):

            mu0 = np.sqrt(1.0-n*n*(1.0-mu*mu))
            f1 = (mu-n*mu0)/(mu+n*mu0)
            f2 = (mu0-n*mu)/(mu0+n*mu)
            self.R = 0.5*f1*f1+0.5*f2*f2

        else:

            self.R = 1  # total reflexion
