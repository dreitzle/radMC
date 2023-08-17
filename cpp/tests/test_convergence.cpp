#include <iostream>
#include "config.hpp"
#include "clsim.hpp"
#include <cmath>
#include "netcdf_interface_mc.hpp"
#include <chrono>
#include "welford.tcc"

int main( int argc, char **argv )
{
    const unsigned int n_runs = 500U;
    const double run_seconds = 0.2;
    
    Config config(std::string(DATA_PATH).append("config_conv3.json"));

    const unsigned int n_costheta = config.mc_config.n_costheta;
    const unsigned int n_phi = config.mc_config.n_phi;
    const unsigned int detsize = n_costheta*n_phi;

    CLsim sim(config.ocl_config.platform, config.ocl_config.devices[0]);

    sim.create_buffers(config);
    sim.seed_rng(config);

    std::cout << "Running simulations (Integral)." << std::endl;

    WelfordMeanVariance<double> welford_int_r(detsize);
    WelfordMeanVariance<double> welford_int_t(detsize);

    tq::progress_bar progress1;

    for(unsigned int i = 0U; i < n_runs; ++i)
    {
        sim.run(config,run_seconds,false);
        const auto& [res_r, res_t] = sim.get_result();
        welford_int_r.update(res_r);
        welford_int_t.update(res_t);
        progress1.update((double)i/(double)n_runs);
        sim.reset();
    }
    progress1.update(1.0);
    std::cerr << std::endl;

    const auto& [cost_points, phi_points] = sim.get_points();
    const auto& [mean_int_r, var_int_r] = welford_int_r.getMeanVariance();
    const auto& [mean_int_t, var_int_t] = welford_int_t.getMeanVariance();

    std::cout << "Running simulations (LF)." << std::endl;

    sim.reset();
    sim.reset_build_opts("-DUSE_LF");

    WelfordMeanVariance<double> welford_lf_r(detsize);
    WelfordMeanVariance<double> welford_lf_t(detsize);

    tq::progress_bar progress2;

    for(unsigned int i = 0U; i < n_runs; ++i)
    {
        sim.run(config,run_seconds,false);
        const auto& [res_r, res_t] = sim.get_result();
        welford_lf_r.update(res_r);
        welford_lf_t.update(res_t);
        progress2.update((double)i/(double)n_runs);
        sim.reset();
    }
    progress2.update(1.0);
    std::cerr << std::endl;

    const auto& [mean_lf_r, var_lf_r] = welford_lf_r.getMeanVariance();
    const auto& [mean_lf_t, var_lf_t] = welford_lf_t.getMeanVariance();

    std::cout << "Writing NC output file." << std::endl;

    netcdf_interface ncfile(std::string(DATA_PATH).append("test_conv.nc"));
    ncfile.create_file();

    std::vector<int> dimids = ncfile.def_dimensions({"phi","cos_theta"},{n_phi,n_costheta});

    ncfile.def_variable("radiance_mean_int_r",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_var_int_r",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_mean_lf_r",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_var_lf_r",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_mean_int_t",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_var_int_t",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_mean_lf_t",NC_DOUBLE,dimids);
    ncfile.def_variable("radiance_var_lf_t",NC_DOUBLE,dimids);
    ncfile.def_variable("cos_theta",NC_FLOAT,dimids[1]);
    ncfile.def_variable("phi",NC_FLOAT,dimids[0]);

    ncfile.enddef();

    ncfile.put_variable("cos_theta",cost_points.data());
    ncfile.put_variable("phi",phi_points.data());
    ncfile.put_variable("radiance_mean_int_r",mean_int_r.data());
    ncfile.put_variable("radiance_var_int_r",var_int_r.data());
    ncfile.put_variable("radiance_mean_lf_r",mean_lf_r.data());
    ncfile.put_variable("radiance_var_lf_r",var_lf_r.data());
    ncfile.put_variable("radiance_mean_int_t",mean_int_t.data());
    ncfile.put_variable("radiance_var_int_t",var_int_t.data());
    ncfile.put_variable("radiance_mean_lf_t",mean_lf_t.data());
    ncfile.put_variable("radiance_var_lf_t",var_lf_t.data());

    ncfile.put_attr("mu_crit", config.sim_parameters.get_mu_crit());

    ncfile.close_file();

    return 0;
}
