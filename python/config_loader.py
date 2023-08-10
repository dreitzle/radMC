import json


class config:

    def __init__(self, file_path):

        try:
            print(f"Trying to open config from path: {file_path}...", end=" ")

            with open(file_path) as file:
                self.data = json.load(file)

            print("Done!")

        except Exception:
            print("No config file found, creating standard config")

            self.data = {
                            "mc_config": {
                                "photons": 1e5,
                                "n_scat_pthread": 100,
                                "n_scat_max": 100000,
                                "n_phi": 1,
                                "n_costheta": 50
                            },
                            "opencl_config": {
                                "platform": 0,
                                "devices": [
                                    0
                                ],
                                "threads": 4096
                            },
                            "sim_parameters": {
                                "mua": 0.01,
                                "mus": 5.0,
                                "g": 0.8,
                                "n1": 1.0,
                                "n2": 1.4,
                                "theta_ls": 0
                            }
                        }

            with open('config.json', 'w') as file:
                json.dump(self.data, file, indent=4)


class ocl_config:

    def __init__(self):

        self.build_options = []

    def add_path(self, include_path):

        self.build_options.append(f"-I {include_path}")

    def add_build_define(self, MACRO):

        self.build_options.append(f"-D {MACRO}")
