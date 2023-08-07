#include <iostream>
#include "config.hpp"
#include "clsim.hpp"

int main( int argc, char **argv )
{
    Config config;

    CLsim sim(config.ocl_config.platform, config.ocl_config.devices[0]);

    sim.create_buffers(config);
    sim.seed_rng(config);
    sim.run(config);
    sim.write_nc("out.nc");

    return 0;
}
