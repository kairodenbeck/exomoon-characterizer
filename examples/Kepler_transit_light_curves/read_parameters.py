import numpy as np
import h5py

import sys

if len(sys.argv)>1:
    input_file = sys.argv[1]
else:
    print("Need h5 file as argument.")

hf = h5py.File(input_file, "r")

parameters=[]

print_available_attributes=True
if print_available_attributes:
    print("All stored parameters:")
    print(list(hf[list(hf.keys())[0]].attrs.keys()))

for h in hf:
    period = hf[h].attrs["P_b"]#<-- planet period
    phase = hf[h].attrs["phi_b"]#<-- planet phase angle

    mass_star = hf[h].attrs["M_star"]
    radius_star = hf[h].attrs["R_star"]

    parameters.append((period,phase,mass_star,radius_star))

hf.close()

parameters = np.array(parameters, dtype={"names":("period","phase","mass_star","radius_star"),"formats":("f8","f8","f8","f8")})

print("Periods:", parameters["period"])
print("Phases:", parameters["phase"])
#etc
