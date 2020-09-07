
from setuptools import setup, find_packages

setup(

    name="exomoon_characterizer",
    author="Kai Rodenbeck",
    author_email="rodenbck@mps.mpg.de",
    description="Fitting planet-only and planet-moon transits",
    version="0.1",
    long_description="Fitting planet-only and planet-moon transits",
    license="TBD",
    packages=find_packages(),

    
	install_requires=["h5py","numpy>0.14","emcee>2.1","scipy"]
)
