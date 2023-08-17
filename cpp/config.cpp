#include "config.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

const std::string Config::default_filename = "config.json";

template<typename T>
std::vector<T> get_vector(const pt::ptree& pt, const pt::ptree::key_type& key)
{
    std::vector<T> vec;
    for(auto& item : pt.get_child(key)) {
        assert(item.first.empty());
        vec.push_back(item.second.get_value<T>());
    }
    return vec;
}

template<typename T>
void put_vector(pt::ptree& pt, const pt::ptree::key_type& key, const std::vector<T>& vec)
{
    pt.erase(key);

    pt::ptree array;
    pt::ptree element;

    for(auto& item : vec) {
        element.put_value<T>(item);
        array.push_back(std::make_pair("",element));
    }

    pt.put_child(key,array);
}

float Sim_Parameter_Config::get_init_cost() const
{
    double cost_start = cos(theta_ls/180.0*M_PI);
    const double n_ratio = n1/n2;

    return static_cast<float>(std::clamp(sqrt(1.0-n_ratio*n_ratio*(1.0-cost_start*cost_start)),0.0,1.0));
}

float Sim_Parameter_Config::get_init_weight() const
{
    const double n = n2/n1;
    const double mu = get_init_cost();
    const double mu_crit = (n > 1.0) ? sqrt(n*n-1.0)/n : 0.0;

    if (mu > mu_crit)
    {
            const double mu0 = sqrt(1.0-n*n*(1.0-mu*mu));
            const double f1 = (mu-n*mu0)/(mu+n*mu0);
            const double f2 = (mu0-n*mu)/(mu0+n*mu);
            return static_cast<float>(1.0-0.5*f1*f1-0.5*f2*f2);
    }
    else //total reflection (should not happen)
        return 0.0; 
}

Config::Config(const std::string &filename)
{
    read(filename);
}

Config::Config()
{
    if(!boost::filesystem::exists(default_filename))
    {
        std::cout <<default_filename <<" does not exists. Creating default file." << std::endl;
        sync_tree();
        write();
    }
    else
    {
        read();
    }
}

void Config::read(const std::string &filename)
{
    pt::read_json(filename, tree);
    sync_structs();
}

void Config::read()
{
    read(default_filename);
}

void Config::write(const std::string &filename)
{
    sync_tree();
    pt::write_json(filename,tree);
}

void Config::write()
{
    write(default_filename);
}

void Config::sync_structs()
{
    mc_config.n_photons = tree.get<unsigned long long int>("mc_config.photons");
    mc_config.n_scat_pthread = tree.get<unsigned int>("mc_config.n_scat_pthread");
    mc_config.n_scat_max = tree.get<unsigned int>("mc_config.n_scat_max");
    mc_config.n_costheta = tree.get<unsigned int>("mc_config.n_costheta");
    mc_config.n_phi = tree.get<unsigned int>("mc_config.n_phi");

    ocl_config.platform = tree.get<unsigned int>("opencl_config.platform");
    ocl_config.threads = tree.get<unsigned int>("opencl_config.threads");
    ocl_config.devices = get_vector<unsigned int>(tree, "opencl_config.devices");

    sim_parameters.mus = tree.get<float>("sim_parameters.mus");
    sim_parameters.mua = tree.get<float>("sim_parameters.mua");
    sim_parameters.g = tree.get<float>("sim_parameters.g");

    try
    {
        sim_parameters.n1 = tree.get<float>("sim_parameters.n1");
        sim_parameters.n2 = tree.get<float>("sim_parameters.n2");
    }
    catch(pt::ptree_bad_path const&)
    {
        sim_parameters.n1 = 1.0;
        sim_parameters.n2 = 1.0;
    }

    try
    {
        sim_parameters.theta_ls = tree.get<float>("sim_parameters.theta_ls");
    }
    catch(pt::ptree_bad_path const&)
    {
        sim_parameters.theta_ls = 0.0;
    }

    try
    {
        sim_parameters.d_slab = tree.get<float>("sim_parameters.d_slab");
    }
    catch(pt::ptree_bad_path const&)
    {
        sim_parameters.d_slab = 0.0;
    }
}

void Config::sync_tree()
{
    tree.put<unsigned long long int>("mc_config.photons", mc_config.n_photons);
    tree.put<unsigned int>("mc_config.n_scat_pthread", mc_config.n_scat_pthread);
    tree.put<unsigned int>("mc_config.n_scat_max", mc_config.n_scat_max);
    tree.put<unsigned int>("mc_config.n_costheta", mc_config.n_costheta);
    tree.put<unsigned int>("mc_config.n_phi", mc_config.n_phi);

    tree.put<unsigned int>("opencl_config.platform", ocl_config.platform);
    put_vector<unsigned int>(tree, "opencl_config.devices", ocl_config.devices);
    tree.put<unsigned int>("opencl_config.threads", ocl_config.threads);

    tree.put<float>("sim_parameters.mus", sim_parameters.mus);
    tree.put<float>("sim_parameters.mua", sim_parameters.mua);
    tree.put<float>("sim_parameters.g", sim_parameters.g);
    tree.put<float>("sim_parameters.n1", sim_parameters.n1);
    tree.put<float>("sim_parameters.n2", sim_parameters.n2);
    tree.put<float>("sim_parameters.theta_ls", sim_parameters.theta_ls);
    tree.put<float>("sim_parameters.d_slab", sim_parameters.d_slab);
}
