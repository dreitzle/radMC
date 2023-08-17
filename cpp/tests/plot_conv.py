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

rad_mean_int_r = data["radiance_mean_int_r"][:]
rad_var_int_r = data["radiance_var_int_r"][:]
rad_mean_lf_r = data["radiance_mean_lf_r"][:]
rad_var_lf_r = data["radiance_var_lf_r"][:]

rad_mean_int_t = data["radiance_mean_int_t"][:]
rad_var_int_t = data["radiance_var_int_t"][:]
rad_mean_lf_t = data["radiance_mean_lf_t"][:]
rad_var_lf_t = data["radiance_var_lf_t"][:]

mu_crit = data.getncattr("mu_crit")

data.close()

plt.figure(dpi=300)
plt.plot(np.abs(cos_theta),rad_mean_int_r[0,:],label="Integral R")
plt.plot(np.abs(cos_theta),rad_mean_lf_r[0,:],'--',label="LF R")
plt.plot(np.abs(cos_theta),rad_mean_int_t[0,:],label="Integral T")
plt.plot(np.abs(cos_theta),rad_mean_lf_t[0,:],'--',label="LF T")
plt.xlabel("$|\mu|$")
plt.ylabel("Mean")
plt.yscale('log')
plt.legend()
plt.axvline(x = mu_crit, color = 'b')
plt.xlim(0,1)

plt.figure(dpi=300)
plt.plot(np.abs(cos_theta),rad_var_int_r[0,:],label="Integral R")
plt.plot(np.abs(cos_theta),rad_var_lf_r[0,:],label="LF R")
plt.plot(np.abs(cos_theta),rad_var_int_t[0,:],label="Integral T")
plt.plot(np.abs(cos_theta),rad_var_lf_t[0,:],label="LF T")
plt.xlabel("$|\mu|$")
plt.ylabel("Variance")
plt.yscale('log')
plt.legend()
plt.axvline(x = mu_crit, color = 'b')
plt.xlim(0,1)

# plt.figure(dpi=300)
# plt.loglog(np.abs(cos_theta),np.sqrt(rad_var_int_r[0,:]),label="Integral R")
# plt.loglog(np.abs(cos_theta),np.sqrt(rad_var_lf_r[0,:]),label="LF R")
# plt.loglog(np.abs(cos_theta),np.sqrt(rad_var_int_t[0,:]),label="Integral T")
# plt.loglog(np.abs(cos_theta),np.sqrt(rad_var_lf_t[0,:]),label="LF T")
# plt.xlabel("$|\mu|$")
# plt.ylabel("std-dev")
# plt.legend()
# plt.axvline(x = mu_crit, color = 'b')
# plt.xlim(0,1)