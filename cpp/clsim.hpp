#ifndef _CLSIM_HPP_
#define _CLSIM_HPP_

#define CL_SEED_RNG_SRC "seed.cl"
#define CL_SIM_SEMI_SRC "MC_planar.cl"
#define CL_SIM_SLAB_SRC "MC_planar_slab.cl"

#if OPENCL_CPP_HEADER_TYPE == 1
    #include <CL/cl.hpp>
#elif OPENCL_CPP_HEADER_TYPE == 2
    #include <CL/cl2.hpp>
#elif OPENCL_CPP_HEADER_TYPE == 3
    #include <CL/opencl.hpp>  
#else
    #pragma message "WARNING OPENCL_CPP_HEADER_TYPE not defined"
    #include <CL/cl2.hpp>
#endif

#include "config.hpp"
#include "tqdm.hpp"
#include <memory>
#include "boost/lexical_cast.hpp"

/* clext.h forward */
const char* clGetErrorString(int errorCode);
int clCheckError(int errorCode);

/* OpenCL structs */

typedef struct tyche_i_state{
    cl_uint a,b,c,d;
} tyche_i_state;

typedef struct Dir3d{
    cl_float cos_t;
    cl_float sin_t;
    cl_float phi;
} Dir3d;

typedef struct Photon{
    Dir3d dir;
    cl_float zpos;
    cl_float weight;
    cl_uint scat_counter;
} Photon;

class CLsim
{
    private:

        cl::Platform platform;
        cl::Device device;

        void init(const char* add_opts);
        void create_context();
        void create_queue();

        /* Load file contents as string */
        std::string read_src(std::string &path);

        /* Simulate slab? */
        bool sim_slab;

    protected:

        /* convert values to string literals. Needed to pass parameters to OpenCL kernels as defines */
        template<typename T>
        std::string to_literal(T value)
        {
            std::string str = boost::lexical_cast<std::string>(value);
            if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, float>)
                (str.find(".") == std::string::npos) ? str.append(".F") : str.append ("F");
            else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, long double>)
                (str.find(".") == std::string::npos) ? str.append(".L") : str.append ("L");
            else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, unsigned int>)
                str.append("U");
            else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, long>)
                str.append("L");
            else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, unsigned long>)
                str.append("UL");
            else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, long long int>)
                str.append("LL");
            else if constexpr(std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, unsigned long long>)
                str.append("ULL");
            return str;
        }

        /* get size of vector data in bytes */
        template<typename T>
        inline size_t vector_bytes(const std::vector<T> &vec) const
            { return vec.size()*sizeof(T); }

        /* Append define to string */
        void add_define(std::string &str, char *name)
        {
            str.append(" -D");
            str.append(name);
        }
        /* Append define with value to string*/
        template<typename T>
        void add_define(std::string &str, const char *name, T value)
        {
            str.append(" -D");
            str.append(name);
            str.append("=");
            str.append(to_literal(value));
        }

        std::unique_ptr<cl::Context> context;
        std::unique_ptr<cl::CommandQueue> queue;
        std::string default_build_opts;

        /* CL programs */
        std::unique_ptr<cl::Program> cl_program_seed;
        std::unique_ptr<cl::Program> cl_program_main;

        /* Host memory for buffers */
        std::vector<cl_ulong> simulated_photons_pthread;
        std::vector<cl_float> detector_r;
        std::vector<cl_float> detector_t;
        std::vector<cl_float> cost_points;
        std::vector<cl_float> phi_points;

        /* Device buffers */
        std::unique_ptr<cl::Buffer> buffer_rng_states;
        std::unique_ptr<cl::Buffer> buffer_simulated_photons_pthread;
        std::unique_ptr<cl::Buffer> buffer_photons;
        std::unique_ptr<cl::Buffer> buffer_detector_r;
        std::unique_ptr<cl::Buffer> buffer_detector_t;
        std::unique_ptr<cl::Buffer> buffer_cost_points;
        std::unique_ptr<cl::Buffer> buffer_phi_points;

        /* Detector sums over thread local arrays */
        std::vector<double> detector_r_sum;
        std::vector<double> detector_t_sum;

        /* OpenCL setup stuff */
        void build_program(const std::string &file, const std::string &opts, std::unique_ptr<cl::Program> &program);
        void create_buffer(std::unique_ptr<cl::Buffer> &buffer, cl_mem_flags flags, size_t size, void *hostmem = nullptr);

        using buffer_ref = std::reference_wrapper<cl::Buffer>;
        void set_args(cl::Kernel &kernel, std::initializer_list<buffer_ref> args);

        template<typename T>
        void write_buffer(const std::unique_ptr<cl::Buffer> &buffer, const std::vector<T> &vec) const
        {
            cl_int err = 0;
            cl::Event event;

            err = queue->enqueueWriteBuffer(*buffer, true, 0U, vector_bytes(vec), vec.data(), nullptr, &event);
            if(clCheckError(err))
                throw std::runtime_error("Write to device failed.");
            event.wait();
        }

        template<typename T>
        void write_buffer(const cl::Buffer &buffer, const std::vector<T> &vec) const
        {
            cl_int err = 0;
            cl::Event event;

            err = queue->enqueueWriteBuffer(buffer, true, 0U, vector_bytes(vec), vec.data(), nullptr, &event);
            if(clCheckError(err))
                throw std::runtime_error("Write to device failed.");
            event.wait();
        }

    public:

        /* Check for platform and device */
        static int check_platform(const unsigned int platform_num, cl::Platform *_platform = NULL);
        static int check_device(cl::Platform &_platform, const unsigned int device_num, cl::Device *_device = NULL);

        /* Constructors */
        CLsim(const unsigned int platform_num, const unsigned int device_num, const char* add_opts = nullptr);
        CLsim(cl::Platform &_platform, cl::Device &_device, const char* add_opts = nullptr);

        /* reset build options */
        void reset_build_opts();
        void reset_build_opts(const char* add_opts);

        void set_points(const std::vector<cl_float>& _cost_points, const std::vector<cl_float>& _phi_points);

        /* Sim */
        void create_buffers(const Config &config);
        void seed_rng(const Config &config);
        void run(const Config &config, double timer = 0.0, bool pbar = true);
        void reset();

        /* Output */
        std::pair<std::vector<cl_float>, std::vector<cl_float>> get_points() const; 
        std::pair<std::vector<double>,std::vector<double>> get_result() const;
        void write_nc(const std::string &filename);
};

#endif /* _CLSIM_HPP_ */
