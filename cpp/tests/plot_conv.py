#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:41:42 2023

@author: dreitzle
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

data = nc.Dataset("./data/test_conv.nc")

cos_theta = data["cos_theta"][:]
phi = data["phi"][:]

rad_mean_int = data["radiance_mean_int"][:]
rad_var_int = data["radiance_var_int"][:]

rad_mean_lf = data["radiance_mean_lf"][:]
rad_var_lf = data["radiance_var_lf"][:]

data.close()

plt.figure(dpi=300)
plt.loglog(np.abs(cos_theta),rad_var_int[0,:],label="Integral")
plt.loglog(np.abs(cos_theta),rad_var_lf[0,:],label="LF")
plt.xlabel("$|\mu|$")
plt.ylabel("Variance")
plt.legend()
