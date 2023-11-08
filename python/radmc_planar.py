# imports
import os
import time
import pyopencl as cl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# set cwd to file location
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(f"Changing directory: {os.path.dirname(os.path.realpath(__file__))}")

# custom imports
from seeds import Seeds
from structs import Photons, Sim_Parameters
from config_loader import config, ocl_config


class Hybrid:
    """
    Calculate radiance for a semi infinite medium
    or a slab using a conventional last flight method
    or an integral last flight implementation.
    """

    def __init__(self, config_path: str = "config.json"):

        self.cwd = os.path.dirname(os.path.realpath(__file__))

        # load config
        self.cfg = config(config_path).data

        self.ocl_cfg = ocl_config()

        # read number of threads
        self.n_threads = int(self.cfg["opencl_config"]["threads"])

        # set MC parameters
        self.n_photons = int(self.cfg["mc_config"]["photons"])

        # use Chebyshev nodes for cost
        self.n_points_costheta = int(self.cfg["mc_config"]["n_costheta"])
        cost_point_idx = np.arange(self.n_points_costheta)+1
        self.cost_points = (0.5*np.cos((2*cost_point_idx-1) /
                            (2*self.n_points_costheta)*np.pi)-0.5).astype(np.float32)

        # Linear phi sampling
        self.n_points_phi = int(self.cfg["mc_config"]["n_phi"])
        self.phi_points = np.linspace(
            0, 2*np.pi, self.n_points_phi, endpoint=False, dtype=np.float32)

        # load optical properties
        parameters = Sim_Parameters(self.cfg)
        mus = parameters.mus
        mua = parameters.mua
        g_factor = parameters.g_factor
        mut = mus + mua
        n_1 = parameters.n_1
        n_2 = parameters.n_2
        d_slab = parameters.d_slab

        # LF?
        # self.ocl_cfg.add_build_define('USE_LF')

        # Include dir for opencl code and set build defines
        self.ocl_cfg.add_path(os.path.join(self.cwd, "../opencl"))  # ./opencl
        # self.ocl_cfg.add_build_define("USE_LF")
        self.ocl_cfg.add_build_define(
            f'N_SCAT_PTHREAD={self.cfg["mc_config"]["n_scat_pthread"]}')
        # Max number of scat event per thread
        self.ocl_cfg.add_build_define(
            f'N_SCAT_MAX={self.cfg["mc_config"]["n_scat_max"]}')
        self.ocl_cfg.add_build_define(
            f'N_PHI={self.n_points_phi}')  # number of phi values
        self.ocl_cfg.add_build_define(
            f'N_COSTHETA={self.n_points_costheta}')  # number of cost values

        self.ocl_cfg.add_build_define(f'C_MUS={mus}f')
        self.ocl_cfg.add_build_define(f'C_MUA={mua}f')

        if abs(g_factor) > 1e-4:  # anisotropic scattering
            self.ocl_cfg.add_build_define(f'C_GF={g_factor}f')

        self.ocl_cfg.add_build_define(f'C_MUT={mut}f')

        # calculate initial direction after refraction
        parameters.calc_start_dir()
        # calculate intial weight accounting for reflections
        parameters.calc_R_fresnel()

        # intial weight considering Fresnel
        self.ocl_cfg.add_build_define(
            f'INIT_WEIGHT={1-parameters.R}f')
        # intial cost direction considering refraction
        self.ocl_cfg.add_build_define(
            f'INIT_COST={parameters.cost_start}f')

        if abs(n_1-n_2) > 1e-6:  # refractive index mismatch

            self.ocl_cfg.add_build_define(
                f'C_N1={n_1}f')  # n outside of medium
            self.ocl_cfg.add_build_define(
                f'C_N2={n_2}f')  # n inside of medium

        if d_slab > 0:  # Kernel for slab

            self.ocl_cfg.add_build_define(
                f'D_SLAB={d_slab}f')

            # Load the OpenCL kernel code and initialize config
            with open(os.path.join(self.cwd, "../opencl", "MC_planar_slab.cl"), 'r') as file:
                self.kernel_code = file.read()
                self.sim_slab = True;

        else:  # Kernel for semi infinite medium

            # Load the OpenCL kernel code and initialize config
            with open(os.path.join(self.cwd, "../opencl", "MC_planar.cl"), 'r') as file:
                self.kernel_code = file.read()
                self.sim_slab = False;

    def run(self):

        platform = cl.get_platforms()[self.cfg["opencl_config"]["platform"]]
        devices = [platform.get_devices()[device_id]
                   for device_id in self.cfg["opencl_config"]["devices"]]

        contexts = [cl.Context([device]) for device in devices]
        queues = [cl.CommandQueue(context) for context in contexts]

        # Build OpenCL program
        programs = [cl.Program(context, self.kernel_code).build(
            options=self.ocl_cfg.build_options) for context in contexts]

        # Prepare input data
        # Count simulated photons per thread
        simulated_photons_pthread = np.zeros(self.n_threads, dtype=np.uint64)
        self.photons = Photons(self.n_threads).photons  # photons struct

        # initialize detector: each thread has its own local detector
        self.detector_loc_r = np.zeros(self.cfg["mc_config"]["n_costheta"] *
                                 self.cfg["mc_config"]["n_phi"]*self.n_threads, dtype=np.float32)
        self.detector_loc_t = np.zeros(self.cfg["mc_config"]["n_costheta"] *
                                 self.cfg["mc_config"]["n_phi"]*self.n_threads, dtype=np.float32)

        # calculate RNG (tyche_i) states in a seperate kernel and create buffer
        rng_states = Seeds(self.n_threads, self.cwd).states
        buffer_rng_states = [cl.Buffer(context, cl.mem_flags.READ_WRITE |
                                       cl.mem_flags.COPY_HOST_PTR, hostbuf=rng_states) for context in contexts]

        # Create buffers for each device
        buffer_photons_pthread = [cl.Buffer(context, cl.mem_flags.READ_WRITE,
                                            size=simulated_photons_pthread.nbytes) for context in contexts]

        cl.enqueue_copy(queues[0], buffer_photons_pthread[0], simulated_photons_pthread).wait()

        buffer_photons = [cl.Buffer(context, cl.mem_flags.READ_WRITE,
                                    size=self.photons.nbytes) for context in contexts]

        cl.enqueue_copy(queues[0], buffer_photons[0], self.photons).wait()

        buffer_detector_t = [cl.Buffer(context, cl.mem_flags.READ_WRITE,
                                       size=self.detector_loc_t.nbytes) for context in contexts]

        cl.enqueue_copy(queues[0], buffer_detector_t[0], self.detector_loc_t).wait()

        buffer_detector_r = [cl.Buffer(context, cl.mem_flags.READ_WRITE,
                                       size=self.detector_loc_r.nbytes) for context in contexts]

        cl.enqueue_copy(queues[0], buffer_detector_r[0], self.detector_loc_r).wait()

        buffer_cost_points = [cl.Buffer(context, cl.mem_flags.READ_ONLY,
                                        size=self.cost_points.nbytes) for context in contexts]

        cl.enqueue_copy(queues[0], buffer_cost_points[0], self.cost_points).wait()

        buffer_phi_points = [cl.Buffer(context, cl.mem_flags.READ_ONLY,
                                       size=self.phi_points.nbytes) for context in contexts]

        cl.enqueue_copy(queues[0], buffer_phi_points[0], self.phi_points).wait()

        # Init photons
        kernel_init = programs[0].init_photons
        kernel_init.set_args(buffer_photons[0])
        gpu_init = cl.enqueue_nd_range_kernel(
            queues[0], kernel_init, (self.n_threads,), (64,))  # Enqeue kernel for execution
        gpu_init.wait()  # Wait for the calculation to finish

        # Create kernel and set kernel args
        kernel = programs[0].run

        if(self.sim_slab):
            kernel.set_args(buffer_rng_states[0], buffer_photons_pthread[0],
                            buffer_photons[0], buffer_detector_r[0], buffer_detector_t[0],
                            buffer_cost_points[0], buffer_phi_points[0])
        else:
            kernel.set_args(buffer_rng_states[0], buffer_photons_pthread[0],
                            buffer_photons[0], buffer_detector_r[0],
                            buffer_cost_points[0], buffer_phi_points[0])

        self.simulated_photons = 0  # initialize photon counter

        # initialize progress bar
        simulated_photons_per_run = 0  # photon counter per GPU run
        simulated_photons_temp = 0  # save last GPU run
        pbar = tqdm(total=self.n_photons,
                    desc="Running Monte Carlo simulation", unit="photons")

        start_timer = time.time()  # time MC simulation

        # run simulation
        while self.simulated_photons < self.n_photons:  # simulate atleast N photons

            gpu_sim = cl.enqueue_nd_range_kernel(
                queues[0], kernel, (self.n_threads,), (64,))  # Enqeue kernel for execution
            gpu_sim.wait()  # Wait for the calculation to finish

            # read photon counting buffer
            cl.enqueue_copy(queues[0], simulated_photons_pthread, buffer_photons_pthread[0]).wait()
            # Sum up all simulated photons per thread
            self.simulated_photons = np.sum(simulated_photons_pthread)

            # progress bar
            simulated_photons_per_run = self.simulated_photons - simulated_photons_temp
            simulated_photons_temp = self.simulated_photons

            pbar.update(simulated_photons_per_run)

        # close progress bar and print time
        pbar.close()

        # finish last photons
        kernel_f = programs[0].finish

        if(self.sim_slab):
            kernel_f.set_args(buffer_rng_states[0], buffer_photons_pthread[0],
                            buffer_photons[0], buffer_detector_r[0], buffer_detector_t[0],
                            buffer_cost_points[0], buffer_phi_points[0])
        else:
            kernel_f.set_args(buffer_rng_states[0], buffer_photons_pthread[0],
                            buffer_photons[0], buffer_detector_r[0],
                            buffer_cost_points[0], buffer_phi_points[0])

        print("Finishing last photons... ", end="")

        gpu_sim = cl.enqueue_nd_range_kernel(
            queues[0], kernel_f, (self.n_threads,), (64,))  # Enqeue kernel for execution
        gpu_sim.wait()  # Wait for the calculation to finish

        print("Done!")

        # read photon counting buffer
        cl.enqueue_copy(queues[0], simulated_photons_pthread, buffer_photons_pthread[0]).wait()
        # Sum up all simulated photons per thread
        self.simulated_photons = np.sum(simulated_photons_pthread)

        cl.enqueue_copy(queues[0], self.detector_loc_r, buffer_detector_r[0]).wait()

        if(self.sim_slab):
            cl.enqueue_copy(queues[0], self.detector_loc_t, buffer_detector_t[0]).wait()

        self.detector_loc_r2 = np.reshape(
            self.detector_loc_r, (self.n_threads, self.cfg["mc_config"]["n_costheta"], self.cfg["mc_config"]["n_phi"]))
        self.detector_r = np.sum(self.detector_loc_r2, axis=0)  # sum over threads
        self.detector_loc_t2 = np.reshape(
           self.detector_loc_t, (self.n_threads, self.cfg["mc_config"]["n_costheta"], self.cfg["mc_config"]["n_phi"]))
        self.detector_t = np.sum(self.detector_loc_t2, axis=0)  # sum over threads

        #cl.enqueue_copy(queues[0], self.photons, buffer_photons[0])

        end_timer = time.time()
        sim_time = end_timer - start_timer

        buffer_rng_states[0].release()
        buffer_photons_pthread[0].release()
        buffer_photons[0].release()
        buffer_detector_t[0].release()
        buffer_detector_r[0].release()
        buffer_cost_points[0].release()
        buffer_phi_points[0].release()

        print(
            f"\n-->{self.simulated_photons:.2E} photons were simulated in {sim_time:.2f} seconds.")


# %%
test = Hybrid()
test.run()
# %%

rad_r = test.detector_r/test.simulated_photons*2*np.pi
rad_t = test.detector_t/test.simulated_photons*2*np.pi
photon = test.photons
cost_point = test.cost_points

# %% Slab

plt.figure(dpi=600)
plt.xlabel(r"$\cos{\theta}$")
plt.ylabel("Radiance")
pn_data_r = np.load("radiance_z0.npz")
pn_data_t = np.load("radiance_zL.npz")

plt.plot(pn_data_r['mu'], pn_data_r['rad'], label="Refl. analytic")
plt.plot(pn_data_t['mu'], pn_data_t['rad'], label="Trans. analytic")

plt.plot(cost_point, rad_r[:, 0], '--', label="Refl. MC")
plt.plot(-cost_point, rad_t[:, 0], '--', label="Trans. MC")
plt.ylim(0,3)

plt.grid()
plt.legend()

# %% Semi

# plt.figure(dpi=600)
# plt.xlabel(r"$\cos{\theta}$")
# plt.ylabel("Radiance")
# pn_data_r = np.load("radiance.npz")

# plt.plot(pn_data_r['mu'], pn_data_r['rad'], label="Refl. analytic")

# plt.plot(cost_point, rad_r[:, 0], '--', label="Refl. MC")

# plt.grid()
# plt.legend()

