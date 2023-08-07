#include "clsim.hpp"
#include <iostream>
#include <fstream>
#include <exception>
#include "clext.h"
#include <type_traits>
#include <random>
#include <chrono>
#include <cmath>
#include "netcdf_interface_mc.hpp"

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

CLsim::CLsim(const unsigned int platform_num, const unsigned int device_num)
{
    if(check_platform(platform_num, &platform))
        throw std::runtime_error("Platform not available.");

    if(check_device(platform, device_num, &device))
        throw std::runtime_error("Device not available.");

    init();
}

CLsim::CLsim(cl::Platform &_platform, cl::Device &_device)
    :platform(_platform), device(_device)
{
    init();
}

void CLsim::init()
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

    default_build_opts.append("-I");
    default_build_opts.append(CL_SOURCE_DIR);
    default_build_opts.append(" -cl-single-precision-constant");
    default_build_opts.append(" -cl-strict-aliasing");
    default_build_opts.append(" -cl-mad-enable");
    default_build_opts.append(" -cl-no-signed-zeros");
}

std::string CLsim::read_src(std::string &path)
{
    std::ifstream file(path, std::ifstream::in | std::ifstream::binary);

    file.seekg(0, file.end);
    unsigned int length = file.tellg();
    file.seekg (0, file.beg);

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

    err = program->build(device, opts.c_str());

    if(clCheckError(err)){
        std::string build_log;
        if(clCheckError(program->getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log)))
            throw std::runtime_error("Build failed and log could not be obtained.");

        std::cerr << "Build log:" << std::endl << build_log << std::endl;
        throw std::runtime_error("Build failed.");
    }
}

void CLsim::create_buffers(const Config &config)
{
    const unsigned int threads = config.ocl_config.threads;
    const unsigned int n_costheta = config.mc_config.n_costheta;
    const unsigned int n_phi = config.mc_config.n_phi;

    cl_int err = 0;
    
    buffer_rng_states = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, threads*sizeof(tyche_i_state), nullptr, &err);
    if(clCheckError(err))
        throw std::runtime_error("RNG state: Buffer creation failed.");

    simulated_photons_pthread.resize(threads);
    std::fill(simulated_photons_pthread.begin(), simulated_photons_pthread.end(), 0ULL);

    buffer_simulated_photons_pthread = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        vector_bytes(simulated_photons_pthread), simulated_photons_pthread.data(), &err);

    if(clCheckError(err))
        throw std::runtime_error("simulated_photons_pthread: Buffer creation failed.");

    buffer_photons = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, threads*sizeof(Photon), nullptr, &err);
    if(clCheckError(err))
        throw std::runtime_error("photons: Buffer creation failed.");

    detector.resize(n_costheta*n_phi*threads);
    std::fill(detector.begin(), detector.end(), 0.0F);

    buffer_detector = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        vector_bytes(detector), detector.data(), &err);

    if(clCheckError(err))
        throw std::runtime_error("detector: Buffer creation failed.");

    cost_points.resize(n_costheta);
    for(unsigned int i = 0; i < n_costheta; i++)
        cost_points[i] = (0.5f*std::cos((2.0f*i-1.0f)/(2.0f*n_costheta)*M_PI)-0.5f);

    buffer_cost_points = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        vector_bytes(cost_points), cost_points.data(), &err);

    if(clCheckError(err))
        throw std::runtime_error("cost_points: Buffer creation failed.");

    phi_points.resize(n_phi);
    const float phi_step = 2.0f*M_PI/n_phi;
    for(unsigned int i = 0; i < n_phi; i++)
        phi_points[i] = i*phi_step;

    buffer_phi_points = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        vector_bytes(phi_points), phi_points.data(), &err);

    if(clCheckError(err))
        throw std::runtime_error("phi_points: Buffer creation failed.");
}

void CLsim::seed_rng(const Config &config)
{
    const unsigned int threads = config.ocl_config.threads;
    
    if(!cl_program_seed)
        build_program(CL_SEED_RNG_SRC, default_build_opts, cl_program_seed);

    std::vector<cl_ulong> seeds(threads);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<cl_ulong> dist(0, 18446744073709551615ULL);
 
    for(unsigned int i = 0; i < threads; i++)
        seeds[i] = dist(gen);

    cl_int err = 0;
    cl::Buffer buffer_rng_seeds(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_bytes(seeds), seeds.data(), &err);

    if(clCheckError(err))
        throw std::runtime_error("RNG seed: copy seeds to device failed.");

    cl::Kernel cl_kernel_seed(*cl_program_seed, "seeds_generator", &err);

    if(clCheckError(err))
        throw std::runtime_error("RNG seed: loading kernel failed.");

    err = cl_kernel_seed.setArg(0, buffer_rng_seeds);
    if(clCheckError(err))
        throw std::runtime_error("RNG seed: setArg 0 failed.");
    err = cl_kernel_seed.setArg(1, *buffer_rng_states);
    if(clCheckError(err))
        throw std::runtime_error("RNG seed: setArg 1 failed.");

    cl::Event event;
    err = queue->enqueueNDRangeKernel(cl_kernel_seed, cl::NullRange, cl::NDRange(threads), cl::NDRange(64), NULL, &event);

    if(clCheckError(err))
        throw std::runtime_error("RNG seed: run kernel failed.");

    event.wait();
    std::cout << "RNG seeding successful." << std::endl;
}

void CLsim::run(const Config &config)
{
    const unsigned int threads = config.ocl_config.threads;
    
    /* Setup build options */
    std::string build_opts = default_build_opts;

    add_define(build_opts, "N_SCAT_PTHREAD", config.mc_config.n_scat_pthread);
    add_define(build_opts, "N_SCAT_MAX", config.mc_config.n_scat_max);
    add_define(build_opts, "N_PHI", config.mc_config.n_phi);
    add_define(build_opts, "N_COSTHETA", config.mc_config.n_costheta);

    add_define(build_opts, "C_MUS", config.sim_parameters.mus);
    add_define(build_opts, "C_MUA", config.sim_parameters.mua);
    add_define(build_opts, "C_GF", config.sim_parameters.g);
    add_define(build_opts, "C_MUT", config.sim_parameters.mus+config.sim_parameters.mua);

    /* Setup kernels */
    if(!cl_program_main)
        build_program(CL_SIM_SRC, build_opts, cl_program_main);

    cl_int err = 0;

    cl::Kernel cl_kernel_sim_init(*cl_program_main, "init_photons", &err);

    if(clCheckError(err))
        throw std::runtime_error("sim init: loading kernel failed.");

    err = cl_kernel_sim_init.setArg(0, *buffer_photons);
    if(clCheckError(err))
        throw std::runtime_error("sim init: setArg 0 failed.");

    cl::Kernel cl_kernel_sim_run(*cl_program_main, "run", &err);

    if(clCheckError(err))
        throw std::runtime_error("sim run: loading kernel failed.");

    err = cl_kernel_sim_run.setArg(0, *buffer_rng_states);
    if(clCheckError(err))
        throw std::runtime_error("sim run: setArg 0 failed.");
    err = cl_kernel_sim_run.setArg(1, *buffer_simulated_photons_pthread);
    if(clCheckError(err))
        throw std::runtime_error("sim run: setArg 1 failed.");
    err = cl_kernel_sim_run.setArg(2, *buffer_photons);
    if(clCheckError(err))
        throw std::runtime_error("sim run: setArg 2 failed.");
    err = cl_kernel_sim_run.setArg(3, *buffer_detector);
    if(clCheckError(err))
        throw std::runtime_error("sim run: setArg 3 failed.");
    err = cl_kernel_sim_run.setArg(4, *buffer_cost_points);
    if(clCheckError(err))
        throw std::runtime_error("sim run: setArg 4 failed.");
    err = cl_kernel_sim_run.setArg(5, *buffer_phi_points);
    if(clCheckError(err))
        throw std::runtime_error("sim run: setArg 5 failed.");

    cl::Kernel cl_kernel_sim_finish(*cl_program_main, "finish", &err);

    if(clCheckError(err))
        throw std::runtime_error("sim finish: loading kernel failed.");

    err = cl_kernel_sim_finish.setArg(0, *buffer_rng_states);
    if(clCheckError(err))
        throw std::runtime_error("sim finish: setArg 0 failed.");
    err = cl_kernel_sim_finish.setArg(1, *buffer_simulated_photons_pthread);
    if(clCheckError(err))
        throw std::runtime_error("sim finish: setArg 1 failed.");
    err = cl_kernel_sim_finish.setArg(2, *buffer_photons);
    if(clCheckError(err))
        throw std::runtime_error("sim finish: setArg 2 failed.");
    err = cl_kernel_sim_finish.setArg(3, *buffer_detector);
    if(clCheckError(err))
        throw std::runtime_error("sim finish: setArg 3 failed.");
    err = cl_kernel_sim_finish.setArg(4, *buffer_cost_points);
    if(clCheckError(err))
        throw std::runtime_error("sim finish: setArg 4 failed.");
    err = cl_kernel_sim_finish.setArg(5, *buffer_phi_points);
    if(clCheckError(err))
        throw std::runtime_error("sim finish: setArg 5 failed.");

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
    progress.update(0.0);

    event.wait();

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

        progress << speed_kernel << " Photons/s";
        progress.update((double)photons_done/(double)n_photons);
    }
    std::cerr << std::endl;

    /* Finish remaining photons */
    err = queue->enqueueNDRangeKernel(cl_kernel_sim_finish, cl::NullRange, cl::NDRange(threads), cl::NDRange(64), NULL, &event);
    if(clCheckError(err))
        throw std::runtime_error("sim finish failed.");
    std::cout << "Finishing last photons." << std::endl;
    event.wait();

    err = queue->enqueueReadBuffer(*buffer_simulated_photons_pthread, CL_TRUE, 0, vector_bytes(simulated_photons_pthread),
        simulated_photons_pthread.data(), nullptr, &event);
    if(clCheckError(err))
        throw std::runtime_error("sim failed.");
    event.wait();

    photons_done = std::reduce(simulated_photons_pthread.begin(), simulated_photons_pthread.end());

    /* Read detector data */
    err = queue->enqueueReadBuffer(*buffer_detector, CL_TRUE, 0, vector_bytes(detector), detector.data(), nullptr, &event);
    if(clCheckError(err))
        throw std::runtime_error("Detector copy failed.");
    event.wait();

    detector_sum.resize(det_length);
    std::fill(detector_sum.begin(), detector_sum.end(), 0.0);

    for(unsigned int i = 0; i < detector.size(); i++)
        detector_sum[i%det_length] += (double)detector[i];

    for(unsigned int i = 0; i < detector_sum.size(); i++)
        detector_sum[i] /= (double)photons_done;
}

void CLsim::write_nc(const std::string &filename)
{
    const unsigned int n_costheta = cost_points.size();
    const unsigned int n_phi = phi_points.size();

    std::cout << "Writing NC output file." << std::endl;

    netcdf_interface ncfile(filename);
    ncfile.create_file();

    std::vector<int> dimids = ncfile.def_dimensions({"phi","cos_theta"},{n_phi,n_costheta});

    ncfile.def_variable("radiance",NC_DOUBLE,dimids);
    ncfile.def_variable("cos_theta",NC_FLOAT,dimids[1]);
    ncfile.def_variable("phi",NC_FLOAT,dimids[0]);
    ncfile.enddef();

    ncfile.put_variable("cos_theta",cost_points.data());
    ncfile.put_variable("phi",phi_points.data());
    ncfile.put_variable("radiance",detector_sum.data());

    ncfile.close_file();
}
