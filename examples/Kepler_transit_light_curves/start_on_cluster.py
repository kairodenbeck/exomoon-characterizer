import glob
import os
import sys



temp_file_name = "temp_submit_file.condor"

moonness=str(sys.argv[1])

n_lc=int(sys.argv[2])

sort_mode = sys.argv[3]

n_cores = 24

hostname = os.uname()[1]

if not os.path.isdir("./logs"):
    os.mkdir("./logs")

with open(temp_file_name, "w") as condor:
    condor.write("Executable      = start_on_cluster.sh\n")
    condor.write("Arguments	= "+ moonness +" "+str(n_lc)+" "+sort_mode+"\n")
    condor.write("Universe        = vanilla\n")
    condor.write("machine_count = 1\n")
    condor.write("request_cpus = %i\n" % n_cores)
    condor.write("output        = ./logs/job.$(Cluster).$(Process).out\n")
    condor.write("error         = ./logs/job.$(Cluster).$(Process).err\n")
    condor.write("log           = ./logs/job.$(Cluster).$(Process).log\n")
    if hostname=="seismo16":
        condor.write("Requirements = (Machine == \"seismo18.mps.mpg.de\" || Machine == \"seismo19.mps.mpg.de\" || Machine == \"seismo20.mps.mpg.de\" || Machine == \"seismo23.mps.mpg.de\")\n")
    condor.write("Initialdir      = .\n")
    condor.write("image_size = %i\n"%(50000))
    condor.write("queue\n")
os.system("condor_submit %s" % temp_file_name)


#Example how to set up running on only some nodes:
#Requirements = (Machine == "seismo18.mps.mpg.de" || Machine == "seismo19.mps.mpg.de" || Machine == "seismo20.mps.mpg.de" || Machine == "seismo23.mps.mpg.de" || Machine == "seismo30.mps.mpg.de" )

