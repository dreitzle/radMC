#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

/* Config struct for MC related parameters */
typedef struct MC_Config {
    unsigned long long int n_photons = 10000000ULL;
    unsigned int n_scat_pthread = 200U;
    unsigned int n_scat_max = 10000000U;
    unsigned int n_costheta = 48U;
    unsigned int n_phi = 1U;
} MC_Config;

/* Config struct for OpenCL */
typedef struct OCl_Config {
    unsigned int platform = 0U;
    std::vector<unsigned int> devices = {0U};
    unsigned int threads = 65536U;
} OCl_Config;

/* Simulation parameters */
typedef struct Sim_Parameter_Config {
    float mua = 0.1;
    float mus = 0.1;
    float g = 0.0;
} Sim_Parameter_Config;

class Config
{
    private:
        static const std::string default_filename;

    public:
        pt::ptree tree;
        MC_Config mc_config;
        OCl_Config ocl_config;
        Sim_Parameter_Config sim_parameters;

        /* Constructors */
        Config(const std::string &filename);
        Config();

        /* Read from file */
        void read(const std::string &filename);
        void read();

        /* Write to file */
        void write(const std::string &filename);
        void write();

        /* Sync tree or structs if one of them was modified */
        void sync_tree();
        void sync_structs();
};

#endif /* _CONFIG_HPP_ */
