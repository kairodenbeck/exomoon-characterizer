import exomoon_characterizer as xmc
import numpy as np

import unittest

def test_transit_no_contact():
    time=np.linspace(-10,10,100)
    model=xmc.fitting.model_no_moon(time,0.1,100.,1.2,0.0,300.,0.5,0.5)
    assert np.any(np.abs(model-1.)<1.0e-12)


def test_transit_no_radius():
    time=np.linspace(-10,10,100)
    model=xmc.fitting.model_no_moon(time,0.0,100.,0.2,0.0,300.,0.5,0.5)
    assert np.any(np.abs(model-1.)<1.0e-12)

def test_transit_depth():
    time=np.linspace(-1,1,100)
    model=xmc.fitting.model_no_moon(time,0.05,100.,0.2,0.0,300.,0.0,0.0)
    assert np.abs(np.min(model)-1.+0.05**2)<1.0e-12

def test_transit_midpoint():
    time=np.linspace(-0,5,100)
    model=xmc.fitting.model_no_moon(time,0.05,100.,0.2,0.01,300.,0.5,0.5)
    min_point=time[np.argmin(model)]
    assert np.abs(min_point-3.)<5./100.
    
