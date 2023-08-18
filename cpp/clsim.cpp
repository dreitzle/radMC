#include "clsim.hpp"
#include <iostream>
#include <fstream>
#include <exception>
#include <type_traits>
#include <random>
#include <chrono>
#include <cmath>
#include "netcdf_interface_mc.hpp"
#include "clext.h"

int CLsim::check_platform(const unsigned int platform_num, cl::Platform *_platform)
{
    std::vector<cl::Platform> platforms;

    if(clCheckError(cl::Platform::get(&platforms)))
    {
        std::cerr << "Failed to get platform list." << std::endl;
        return -1;
    }

    if(platform_num >= platforms.size())
    {
        std::cerr << "Requested platform " << platform_num << " does not exits." << std::endl;
        return -1;
    }

    cl::Platform platform = platforms[platform_num];

    std::cout << "================Platform:" << platform_num << "================" << std::endl;
    std::string platformName;
    std::string platformVendor;
    std::string platformVersion;

    if(clCheckError(platform.getInfo(CL_PLATFORM_NAME, &platformName)) ||
       clCheckError(platform.getInfo(CL_PLATFORM_VENDOR, &platformVendor)) ||
       clCheckError(platform.getInfo(CL_PLATFORM_VERSION, &platformVersion)))
    {
        std::cerr << "Failed to aquire info for platform " << platform_num << std::endl;
        return -1;
    }
    
    std::cout << "Platform name: " << platformName << std::endl;
    std::cout << "Platform vendor: " << platformVendor << std::endl;
    std::cout << "Platform version: " << platformVersion << std::endl;

    if(_platform)
        *_platform = platform;

    return 0;
}

int CLsim::check_device(cl::Platform &_platform, const unsigned int device_num, cl::Device *_device)
{
    std::vector<cl::Device> devices;

    if(clCheckError(_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices)))
    {
        std::cerr << "Failed to get device list." << std::endl;
        return -1;
    }

    if(device_num >= devices.size())
    {
        std::cerr << "Requested device " << device_num << " does not exits." << std::endl;
        return -1;
    }

    cl::Device device = devices[device_num];
    
    std::string deviceName;
    std::string deviceVersion;
    std::string opeclcVersion;
    cl_uint maxComputeUnits;
    cl_uint maxClockFrequency;
    cl_uint floatvectorwidth;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    cl_uint adressbits;

    if(clCheckError(device.getInfo(CL_DEVICE_NAME, &deviceName)) ||
       clCheckError(device.getInfo(CL_DEVICE_VERSION, &deviceVersion)) ||
       clCheckError(device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &opeclcVersion)) ||
       clCheckError(device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits)) ||
       clCheckError(device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &maxClockFrequency)) ||
       clCheckError(device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &floatvectorwidth)) ||
       clCheckError(device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size)) ||
       clCheckError(device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &local_mem_size)) ||
       clCheckError(device.getInfo(CL_DEVICE_ADDRESS_BITS, &adressbits)))
    {
        std::cerr << "Failed to aquire info for device " << device_num << "." << std::endl;
        return -1;
    }

    std::cout << "=================Device:" << device_num << "=================" << std::endl;
    std::cout << "Device name: " << deviceName << std::endl;
    std::cout << "Device version: " << deviceVersion << std::endl;
    std::cout << "OpenCL C version: " << opeclcVersion << std::endl;
    std::cout << "Parallel compute cores: " << maxComputeUnits << std::endl;
    std::cout << "Maximum clock frequency: " << maxClockFrequency/1000.0 << "GHz" << std::endl;
    std::cout << "Preferred vector width float: " << floatvectorwidth << std::endl;
    std::cout << "Global memory size: " << global_mem_size/1073741824.0 << "GiB" << std::endl;
    std::cout << "Local memory size: " << local_mem_size/1024.0 << "KiB" << std::endl;
    std::cout << "Address Bits: " << adressbits << std::endl;
    std::cout << "==========================================" << std::endl;

    if(_device)
        *_device = device;

    return 0;
}

void CLsim::create_context()
{
    cl_context_properties properties[] = 
        { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};

    cl_int err = 0;

    context = std::make_unique<cl::Context>(device, properties, nullptr, nullptr, &err);

    if(clCheckError(err))
        throw std::runtime_error("Context creation failed.");
}

void CLsim::create_queue()
{
    cl_int err = 0;

    queue = std::make_unique<cl::CommandQueue>(*context, device, 0, &err);

    if(clCheckError(err))
        throw std::runtime_error("Command queue creation failed.");
}

/* Contructor without platform and device objects */
CLsim::CLsim(const unsigned int platform_num, const unsigned int device_num, const char* add_opts)
{
    if(check_platform(platform_num, &platform))
        throw std::runtime_error("Platform not available.");

    if(check_device(platform, device_num, &device))
        throw std::runtime_error("Device not available.");

    init(add_opts);
}

/* Contructor with platform and device objects */
CLsim::CLsim(cl::Platform &_platform, cl::Device &_device, const char* add_opts)
    :platform(_platform), device(_device)
{
    init(add_opts);
}

void CLsim::reset_build_opts()
{
    /* if sources were already built, delete programs */
    cl_program_seed.reset();
    cl_program_main.reset();

    /* default options */
    default_build_opts.clear();
    default_build_opts.append("-I");
    default_build_opts.append(CL_SOURCE_DIR);
    default_build_opts.append(" -cl-single-precision-constant");
    default_build_opts.append(" -cl-strict-aliasing");
    default_build_opts.append(" -cl-mad-enable");
    default_build_opts.append(" -cl-no-signed-zeros");
}

/* reset and append additional options */
void CLsim::reset_build_opts(const char* add_opts)
{
    reset_build_opts();
    default_build_opts.append(" ");
    default_build_opts.append(add_opts);
}

void CLsim::init(const char* add_opts)
{
    try
    {
        create_context();
        create_queue();
    }
    catch(std::exception &e)
    {
        std::cerr << "Init failed." << std::endl;
        throw;
    }

    if(add_opts)
        reset_build_opts(add_opts);
    else
        reset_build_opts();
}

/* Read OpenCL program from file as string */
std::string CLsim::read_src(std::string &path)
{
    std::ifstream file(path, std::ifstream::in | std::ifstream::binary);

    /* check length */
    file.seekg(0, file.end);
    unsigned int length = file.tellg();
    file.seekg (0, file.beg);

    /* read */
    char *buffer = new char[length+1];
    file.read(buffer,length);
    buffer[length] = '\0';

    file.close();

    std::string src(buffer);
    delete buffer;

    return src;
}

void CLsim::build_program(const std::string &file, const std::string &opts, std::unique_ptr<cl::Program> &program)
{
    cl_int err = 0;;
    
    std::cout << "Building " << file << " ..." << std::endl;
    std::string cl_program_path(CL_SOURCE_DIR);
    cl_program_path.append(file);
    std::string cl_program_src = read_src(cl_program_path);

    program = std::make_unique<cl::Program>(*context, cl_program_src, false, &err);
    if(clCheckError(err))
        throw std::runtime_error("Could not create program from source.");

    /* Build */
    err = program->build(device, opts.c_str());

    if(clCheckError(err))
    {
        /* Try to obtain build log on error */
        std::string build_log;
        if(clCheckError(program->getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log)))
            throw std::runtime_error("Build failed and log could not be obtained.");

        std::cerr << "Build log:" << std::endl << build_log << std::endl;
        throw std::runtime_error("Build failed.");
    }
}

void CLsim::create_buffer(std::unique_ptr<cl::Buffer> &buffer, cl_mem_flags flags, size_t size, void *hostmem)
{
    cl_int err = 0;
    buffer.reset(new cl::Buffer(*context, flags, size, hostmem, &err));
    if(clCheckError(err))
        throw std::runtime_error("Buffer creation failed.");
};

void CLsim::create_buffers(const Config &config)
{
    /* Generate eval points, if none were set */
    if(cost_points.empty())
    {
        const unsigned int n_costheta = config.mc_config.n_costheta;
        cost_points.resize(n_costheta);
        for(unsigned int i = 0; i < n_costheta; i++)
            cost_points[i] = (0.5f*std::cos((2.0f*i-1.0f)/(2.0f*n_costheta)*M_PI)-0.5f);
    }

    if(phi_points.empty())
    {
        const unsigned int n_phi = config.mc_config.n_phi;
        phi_points.resize(n_phi);
        const float phi_step = 2.0f*M_PI/n_phi;
        for(unsigned int i = 0; i < n_phi; i++)
            phi_points[i] = i*phi_step;
    }
    
    const unsigned int threads = config.ocl_config.threads;
    const unsigned int detsize = cost_points.size()*phi_points.size()*threads;

    create_buffer(buffer_rng_states, CL_MEM_READ_WRITE, threads*sizeof(tyche_i_state));

    simulated_photons_pthread.resize(threads);
    std::fill(simulated_photons_pthread.begin(), simulated_photons_pthread.end(), 0ULL);

    create_buffer(buffer_simulated_photons_pthread,CL_MEM_READ_WRITE, vector_bytes(simulated_photons_pthread), nullptr);
    write_buffer(buffer_simulated_photons_pthread, simulated_photons_pthread);

    create_buffer(buffer_photons, CL_MEM_READ_WRITE, threads*sizeof(Photon));

    detector_r.resize(detsize);
    std::fill(detector_r.begin(), detector_r.end(), 0.0F);

    create_buffer(buffer_detector_r, CL_MEM_READ_WRITE, vector_bytes(detector_r), nullptr);
    write_buffer(buffer_detector_r, detector_r);

    sim_slab = false;
    detector_t.resize(detsize);
    std::fill(detector_t.begin(), detector_t.end(), 0.0F);

    if(config.sim_parameters.d_slab > 0.0)
    {
        create_buffer(buffer_detector_t, CL_MEM_READ_WRITE, vector_bytes(detector_t), nullptr);
        write_buffer(buffer_detector_t, detector_t);
        sim_slab = true;
    }

    create_buffer(buffer_cost_points, CL_MEM_READ_ONLY, vector_bytes(cost_points), nullptr);
    write_buffer(buffer_cost_points, cost_points);

    create_buffer(buffer_phi_points, CL_MEM_READ_ONLY, vector_bytes(phi_points), nullptr);
    write_buffer(buffer_phi_points, phi_points);
}

void CLsim::set_args(cl::Kernel &kernel, std::initializer_list<buffer_ref> args)
{
    cl_int err = 0;
    unsigned int argnum = 0U;
    
    for(auto buffer : args)
    {
        err = kernel.setArg(argnum, buffer.get());

        if(clCheckError(err))
            throw std::runtime_error("setArg failed.");

        argnum++;
    }
}

void CLsim::set_points(const std::vector<cl_float>& _cost_points, const std::vector<cl_float>& _phi_points)
{
    cost_points = _cost_points;
    phi_points = _phi_points;

    /* if there are already buffers for these arrays, we need new ones */
    if(buffer_cost_points)
    {
        create_buffer(buffer_cost_points, CL_MEM_READ_ONLY, vector_bytes(cost_points), nullptr);
        write_buffer(buffer_cost_points, cost_points);
    }

    if(buffer_phi_points)
    {
        create_buffer(buffer_phi_points, CL_MEM_READ_ONLY, vector_bytes(phi_points), nullptr);
        write_buffer(buffer_phi_points, phi_points);
    }
}

void CLsim::reset()
{
    /* Zero photon counters and detectors */
    std::fill(simulated_photons_pthread.begin(), simulated_photons_pthread.end(), 0ULL);
    write_buffer(buffer_simulated_photons_pthread, simulated_photons_pthread);
        
    std::fill(detector_r.begin(), detector_r.end(), 0.0F);
    write_buffer(buffer_detector_r, detector_r);

    std::fill(detector_t.begin(), detector_t.end(), 0.0F);

    if(sim_slab)
        write_buffer(buffer_detector_t, detector_t);
}

void CLsim::seed_rng(const Config &config)
{
    const unsigned int threads = config.ocl_config.threads;
    
    if(!cl_program_seed)
        build_program(CL_SEED_RNG_SRC, default_build_opts, cl_program_seed);

    /* Generate 64 bit seeds for RNG */
    std::vector<cl_ulong> seeds(threads);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<cl_ulong> dist(0, 18446744073709551615ULL);
 
    for(unsigned int i = 0; i < threads; i++)
        seeds[i] = dist(gen);

    /* Call kernel to generate initial RNG states from seeds */
    cl_int err = 0;
    cl::Buffer buffer_rng_seeds(*context, CL_MEM_READ_ONLY, vector_bytes(seeds), nullptr, &err);
    
    if(clCheckError(err))
        throw std::runtime_error("RNG seed: copy seeds to device failed.");
    
    write_buffer(buffer_rng_seeds, seeds);

    cl::Kernel cl_kernel_seed(*cl_program_seed, "seeds_generator", &err);

    if(clCheckError(err))
        throw std::runtime_error("RNG seed: loading kernel failed.");

    set_args(cl_kernel_seed,{buffer_rng_seeds, *buffer_rng_states});

    cl::Event event;
    err = queue->enqueueNDRangeKernel(cl_kernel_seed, cl::NullRange, cl::NDRange(threads), cl::NDRange(64), NULL, &event);

    if(clCheckError(err))
        throw std::runtime_error("RNG seed: run kernel failed.");

    event.wait();
    std::cout << "RNG seeding successful." << std::endl;
}

void CLsim::run(const Config &config, double timer, bool pbar)
{
    const unsigned int threads = config.ocl_config.threads;
    
    /* Setup build options */
    std::string build_opts = default_build_opts;

    add_define(build_opts, "N_SCAT_PTHREAD", config.mc_config.n_scat_pthread);
    add_define(build_opts, "N_SCAT_MAX", config.mc_config.n_scat_max);
    add_define(build_opts, "N_PHI", phi_points.size());
    add_define(build_opts, "N_COSTHETA", cost_points.size());

    add_define(build_opts, "C_MUS", config.sim_parameters.mus);
    add_define(build_opts, "C_MUA", config.sim_parameters.mua);
    add_define(build_opts, "C_MUT", config.sim_parameters.mus+config.sim_parameters.mua);

    if(config.sim_parameters.g > 1e-4)
        add_define(build_opts, "C_GF", config.sim_parameters.g);

    if(fabs(config.sim_parameters.n1 - config.sim_parameters.n2) > 1e-6)
    {
        add_define(build_opts, "C_N1", config.sim_parameters.n1);
        add_define(build_opts, "C_N2", config.sim_parameters.n2);
        add_define(build_opts, "INIT_WEIGHT", config.sim_parameters.get_init_weight());
    }
    else
        add_define(build_opts, "INIT_WEIGHT", 1.0);

    add_define(build_opts, "INIT_COST", config.sim_parameters.get_init_cost());

    /* Select kernel set */
    if(sim_slab)
    {
        add_define(build_opts, "D_SLAB", config.sim_parameters.d_slab);
        if(!cl_program_main)
            build_program(CL_SIM_SLAB_SRC, build_opts, cl_program_main);
    }
    else
    {
        if(!cl_program_main)
            build_program(CL_SIM_SEMI_SRC, build_opts, cl_program_main);
    }

    /* Prepare kernels */
    cl_int err = 0;

    cl::Kernel cl_kernel_sim_init(*cl_program_main, "init_photons", &err);

    if(clCheckError(err))
        throw std::runtime_error("sim init: loading kernel failed.");

    set_args(cl_kernel_sim_init, {*buffer_photons});

    cl::Kernel cl_kernel_sim_run(*cl_program_main, "run", &err);

    if(clCheckError(err))
        throw std::runtime_error("sim run: loading kernel failed.");

    if(sim_slab)
        set_args(cl_kernel_sim_run,
                {*buffer_rng_states, *buffer_simulated_photons_pthread, *buffer_photons,
                 *buffer_detector_r, *buffer_detector_t, *buffer_cost_points, *buffer_phi_points} );
    else
        set_args(cl_kernel_sim_run,
                {*buffer_rng_states, *buffer_simulated_photons_pthread, *buffer_photons,
                 *buffer_detector_r, *buffer_cost_points, *buffer_phi_points} );

    cl::Kernel cl_kernel_sim_finish(*cl_program_main, "finish", &err);

    if(clCheckError(err))
        throw std::runtime_error("sim finish: loading kernel failed.");

    if(sim_slab)
        set_args(cl_kernel_sim_finish,
                {*buffer_rng_states, *buffer_simulated_photons_pthread, *buffer_photons,
                 *buffer_detector_r, *buffer_detector_t, *buffer_cost_points, *buffer_phi_points} );
    else
        set_args(cl_kernel_sim_finish,
                {*buffer_rng_states, *buffer_simulated_photons_pthread, *buffer_photons,
                 *buffer_detector_r, *buffer_cost_points, *buffer_phi_points} );

    cl::Event event;

    /* Initialize */
    err = queue->enqueueNDRangeKernel(cl_kernel_sim_init, cl::NullRange, cl::NDRange(threads), cl::NDRange(64), NULL, &event);
    if(clCheckError(err))
        throw std::runtime_error("sim init failed.");

    /* Run main simulation */

    unsigned long long int photons_done_prev = 0ULL;
    unsigned long long int photons_done = 0ULL;
    const unsigned long long int n_photons = config.mc_config.n_photons;
    const unsigned int n_costheta = cost_points.size();
    const unsigned int n_phi = phi_points.size();
    const unsigned int det_length = n_costheta*n_phi;

    tq::progress_bar progress;

    if(pbar)
        progress.update(0.0);

    event.wait();

    if(timer <= 0.0)
    {
        /* No timer, count photons */
        while(photons_done < n_photons)
        {
            auto time_start_kernel = std::chrono::steady_clock::now();
            photons_done_prev = photons_done;

            err = queue->enqueueNDRangeKernel(cl_kernel_sim_run, cl::NullRange, cl::NDRange(threads), cl::NDRange(64), NULL, &event);
            if(clCheckError(err))
                throw std::runtime_error("sim failed.");
            event.wait();

            err = queue->enqueueReadBuffer(*buffer_simulated_photons_pthread, CL_TRUE, 0, vector_bytes(simulated_photons_pthread),
                simulated_photons_pthread.data(), nullptr, &event);
            if(clCheckError(err))
                throw std::runtime_error("sim failed.");
            event.wait();

            photons_done = std::reduce(simulated_photons_pthread.begin(), simulated_photons_pthread.end());

            auto runtime_kernel = tq::elapsed_seconds(time_start_kernel,std::chrono::steady_clock::now());
            double speed_kernel = (double)(photons_done-photons_done_prev) / runtime_kernel;

            if(pbar)
            {
                progress << speed_kernel << " Photons/s";
                progress.update((double)photons_done/(double)n_photons);
            }
        }
    }
    else
    {
        /* Timer set, run for specified time */
        auto time_start_run = std::chrono::steady_clock::now();
        auto time_end_kernel = time_start_run;

        while(tq::elapsed_seconds(time_start_run,time_end_kernel) < timer)
        {
            auto time_start_kernel = std::chrono::steady_clock::now();
            photons_done_prev = photons_done;

            err = queue->enqueueNDRangeKernel(cl_kernel_sim_run, cl::NullRange, cl::NDRange(threads), cl::NDRange(64), NULL, &event);
            if(clCheckError(err))
                throw std::runtime_error("sim failed.");
            event.wait();

            err = queue->enqueueReadBuffer(*buffer_simulated_photons_pthread, CL_TRUE, 0, vector_bytes(simulated_photons_pthread),
                simulated_photons_pthread.data(), nullptr, &event);
            if(clCheckError(err))
                throw std::runtime_error("sim failed.");
            event.wait();

            photons_done = std::reduce(simulated_photons_pthread.begin(), simulated_photons_pthread.end());

            time_end_kernel = std::chrono::steady_clock::now();
            auto runtime_kernel = tq::elapsed_seconds(time_start_kernel,time_end_kernel);
            double speed_kernel = (double)(photons_done-photons_done_prev) / runtime_kernel;

            if(pbar)
            {
                progress << speed_kernel << " Photons/s";
                progress.update((double)(tq::elapsed_seconds(time_start_run,std::chrono::steady_clock::now()) / timer));
            }
        }
    }

    if(pbar)
        std::cerr << std::endl;

    /* Finish remaining photons */
    err = queue->enqueueNDRangeKernel(cl_kernel_sim_finish, cl::NullRange, cl::NDRange(threads), cl::NDRange(64), NULL, &event);
    if(clCheckError(err))
        throw std::runtime_error("sim finish failed.");

    if(pbar)
        std::cout << "Finishing last photons." << std::endl;

    event.wait();

    err = queue->enqueueReadBuffer(*buffer_simulated_photons_pthread, CL_TRUE, 0, vector_bytes(simulated_photons_pthread),
        simulated_photons_pthread.data(), nullptr, &event);
    if(clCheckError(err))
        throw std::runtime_error("sim failed.");
    event.wait();

    photons_done = std::reduce(simulated_photons_pthread.begin(), simulated_photons_pthread.end());

    queue->finish();

    /* Read detector data */
    err = queue->enqueueReadBuffer(*buffer_detector_r, CL_TRUE, 0, vector_bytes(detector_r), detector_r.data(), nullptr, &event);
    if(clCheckError(err))
        throw std::runtime_error("Detector copy failed.");
    event.wait();

    if(sim_slab)
    {
        err = queue->enqueueReadBuffer(*buffer_detector_t, CL_TRUE, 0, vector_bytes(detector_t), detector_t.data(), nullptr, &event);
        if(clCheckError(err))
            throw std::runtime_error("Detector copy failed.");
        event.wait();
    }

    /* Sum over threads */
    detector_r_sum.resize(det_length);
    std::fill(detector_r_sum.begin(), detector_r_sum.end(), 0.0);
    detector_t_sum.resize(det_length);
    std::fill(detector_t_sum.begin(), detector_t_sum.end(), 0.0);

    for(unsigned int i = 0; i < detector_r.size(); i++)
        detector_r_sum[i%det_length] += (double)detector_r[i];

    for(unsigned int i = 0; i < detector_t.size(); i++)
        detector_t_sum[i%det_length] += (double)detector_t[i];

    for(unsigned int i = 0; i < detector_r_sum.size(); i++)
        detector_r_sum[i] /= (double)photons_done;

    for(unsigned int i = 0; i < detector_t_sum.size(); i++)
        detector_t_sum[i] /= (double)photons_done;
}

void CLsim::write_nc(const std::string &filename)
{
    const unsigned int n_costheta = cost_points.size();
    const unsigned int n_phi = phi_points.size();

    std::cout << "Writing NC output file." << std::endl;

    netcdf_interface ncfile(filename);
    ncfile.create_file();

    std::vector<int> dimids = ncfile.def_dimensions({"phi","cos_theta"},{n_phi,n_costheta});

    ncfile.def_variable("radiance_r",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_t",NC_DOUBLE,dimids);
    ncfile.def_variable("cos_theta",NC_FLOAT,dimids[1]);
    ncfile.def_variable("phi",NC_FLOAT,dimids[0]);
    ncfile.enddef();

    ncfile.put_variable("cos_theta",cost_points.data());
    ncfile.put_variable("phi",phi_points.data());
    ncfile.put_variable("radiance_r",detector_r_sum.data());
    ncfile.put_variable("radiance_t",detector_t_sum.data());

    ncfile.close_file();
}

std::pair<std::vector<double>,std::vector<double>> CLsim::get_result() const
{
    return std::make_pair(detector_r_sum, detector_t_sum);
}

std::pair<std::vector<cl_float>, std::vector<cl_float>> CLsim::get_points() const
{
    return std::make_pair(cost_points,phi_points);
}
