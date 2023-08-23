#define BOOST_TEST_MODULE mc_slab_int
#include <boost/test/unit_test.hpp>

#include "config.hpp"
#include "clsim.hpp"
#include "netcdf_interface_mc.hpp"

/* File with config (json) and reference values (netcdf) */
#define REF_FILE "data_test_slab1.nc"
#define CONF_FILE "config_test_slab1.json"

/* Required precision in percent*/
#define PREC 0.5

/* Test Legendre algorithms with double precision */
BOOST_AUTO_TEST_SUITE( test_int )

    /* Test double vector */
    BOOST_AUTO_TEST_CASE( case1_int )
    {
        /* reference data */
        netcdf_interface ncfile(std::string(DATA_PATH).append(REF_FILE));
        ncfile.open_file(NC_NOWRITE);

        std::vector<cl_float> cost_points;
        std::vector<cl_float> phi_points;
        std::vector<cl_float> res_r_ref;
        std::vector<cl_float> res_t_ref;

        ncfile.get_variable_vector("cost", cost_points);
        ncfile.get_variable_vector("phi", phi_points);
        ncfile.get_variable_vector("radiance_top", res_r_ref);
        ncfile.get_variable_vector("radiance_bottom", res_t_ref);

        ncfile.close_file();

        /* MC Simulation */
        Config config(std::string(DATA_PATH).append(CONF_FILE));

        const unsigned int n_costheta = cost_points.size();
        const unsigned int n_phi = phi_points.size();
        const unsigned int detsize = n_costheta*n_phi;

        CLsim sim(config.ocl_config.platform, config.ocl_config.devices[0]);

        sim.set_points(cost_points,phi_points);
        sim.create_buffers(config);
        sim.seed_rng(config);

        sim.run(config);
        const auto& [res_r, res_t] = sim.get_result();

        for(unsigned int i = 0; i < detsize; i++)
            BOOST_CHECK_CLOSE( res_r[i], res_r_ref[i] , PREC);

        for(unsigned int i = 0; i < detsize; i++)
            BOOST_CHECK_CLOSE( res_t[i], res_t_ref[i] , PREC);
    }

BOOST_AUTO_TEST_SUITE_END()
