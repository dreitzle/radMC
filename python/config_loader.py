import json

class config:   

    def __init__(self, file_path):
        
        try:
            
            print(f"Trying to open config from path: {file_path}...", end=" ")
            
            with open(file_path) as file:  
                self.data = json.load(file)
                
            print("Done!")
                
        
        except:
            
            print("No config file found, creating standard config")
            
            
            self.data =  {
                "mc_config": {
                  "photons": 10000000,
                  "n_scat_pthread": 200,
                  "n_scat_max": 100000,
                  "n_phi": 1,
                  "n_costheta": 180
                  },
                  
                "opencl_config": {
                    "platform": 0,
                    "devices": [0],
                    "threads": 65536
                    },
                    
                "sim_parameters": {
                    "mua": 0.1,
                    "mus": 0.1,
                    "g": 0.0
                }
            }

            with open('config.json', 'w') as file:
                json.dump(self.data, file, indent=4)
  
class ocl_config:   
    
    def __init__(self):
        
        self.build_options = []
        
    def add_path(self, include_path):
        
        self.build_options.append("-I " + include_path)
        
    def add_build_define(self, MACRO):
        
        self.build_options.append("-D " + MACRO)
        
        