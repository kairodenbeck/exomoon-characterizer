"""
-------------------------------------------------------
The Mandel & Agol (2002) transit light curve equations.
-------------------------------------------------------
:FUNCTIONS:
   :func:`occultuniform` -- uniform-disk transit light curve
   :func:`occultquad` -- quadratic limb-darkening
   :func:`occultnonlin` -- full (4-parameter) nonlinear limb-darkening
   :func:`occultnonlin_small` -- small-planet approximation with full
                                 nonlinear limb-darkening.
   :func:`t2z` -- convert dates to transiting z-parameter for circular
                  orbits.
:REQUIREMENTS:
   `numpy <http://www.numpy.org/>`_
   `scipy.special <http://www.scipy.org/>`_
:NOTES:
    Certain values of p (<0.09, >0.5) cause some routines to hang;
    your mileage may vary.  If you find out why, please let me know!
    Cursory testing suggests that the Python routines contained within
     are slower than the corresponding IDL code by a factor of 5-10.
    For :func:`occultquad` I relied heavily on the IDL code of E. Agol
    and J. Eastman.
    Function :func:`appellf1` comes from the mpmath compilation, and
    is adopted (with modification) for use herein in compliance with
    its BSD license (see function documentation for more details).
:REFERENCE:
    The main reference is that seminal work by `Mandel and Agol (2002)
    <http://adsabs.harvard.edu/abs/2002ApJ...580L.171M>`_.
:LICENSE:
    Created by `Ian Crossfield <http://www.astro.ucla.edu/~ianc/>`_ at
    UCLA.  The code contained herein may be reused, adapted, or
    modified so long as proper attribution is made to the original
    authors.
:REVISIONS:
   2011-04-22 11:08 IJMC: Finished, renamed occultation functions.
                          Cleaned up documentation. Published to
                          website.
   2011-04-25 17:32 IJMC: Fixed bug in :func:`ellpic_bulirsch`.
   2012-03-09 08:38 IJMC: Several major bugs fixed, courtesy of
                          S. Aigrain at Oxford U.
   2012-03-20 14:12 IJMC: Fixed modeleclipse_simple based on new
                          format of :func:`occultuniform.  `
"""
from __future__ import print_function, division

import numpy as np
from scipy import special, misc
import pdb

eps = np.finfo(float).eps
zeroval = eps*1e6

def appelf1_ac(a, b1, b2, c, z1, z2, **kwargs):
    """Analytic continuations of the Appell hypergeometric function of 2 variables.
    :REFERENCE:
       Olsson 1964, Colavecchia et al. 2001
    """
    # 2012-03-09 12:05 IJMC: Created



def appellf1(a,b1,b2,c,z1,z2,**kwargs):
    """Give the Appell hypergeometric function of two variables.
    :INPUTS:
       six parameters, all scalars.
    :OPTIONS:
       eps -- scalar, machine tolerance precision.  Defaults to 1e-10.
    :NOTES:
       Adapted from the `mpmath <http://code.google.com/p/mpmath/>`_
       module, but using the scipy (instead of mpmath) Gauss
       hypergeometric function speeds things up.
    :LICENSE:
       MPMATH Copyright (c) 2005-2010 Fredrik Johansson and mpmath
       contributors.  All rights reserved.
       Redistribution and use in source and binary forms, with or
       without modification, are permitted provided that the following
       conditions are met:
       a. Redistributions of source code must retain the above
          copyright notice, this list of conditions and the following
          disclaimer.
       b. Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.
       c. Neither the name of mpmath nor the names of its contributors
          may be used to endorse or promote products derived from this
          software without specific prior written permission.
       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
       CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
       INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
       MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
       DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE
       LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
       EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
       TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
       DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
       ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
       LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
       IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
       THE POSSIBILITY OF SUCH DAMAGE.
    """
    #2011-04-22 10:15 IJMC: Adapted from mpmath, but using scipy Gauss
    #   hypergeo. function
    # 2013-03-11 13:34 IJMC: Added a small error-trap for 'nan' hypgf values

    if kwargs.has_key('eps'):
        eps = kwargs['eps']
    else:
        eps = 1e-9

    # Assume z1 smaller
    # We will use z1 for the outer loop
    if abs(z1) > abs(z2):
        z1, z2 = z2, z1
        b1, b2 = b2, b1
    def ok(x):
        return abs(x) < 0.99
    # IJMC: Ignore the finite cases for now....
    ## Finite cases
    #if ctx.isnpint(a):
    #    pass
    #elif ctx.isnpint(b1):
    #    pass
    #elif ctx.isnpint(b2):
    #    z1, z2, b1, b2 = z2, z1, b2, b1
    #else:
    #    #print z1, z2
    #    # Note: ok if |z2| > 1, because
    #    # 2F1 implements analytic continuation
    if not ok(z1):
        u1 = (z1-z2)/(z1-1)
        if not ok(u1):
            raise ValueError("Analytic continuation not implemented")
        #print "Using analytic continuation"
        return (1-z1)**(-b1)*(1-z2)**(c-a-b2)*\
            appellf1(c-a,b1,c-b1-b2,c,u1,z2,**kwargs)

    #print "inner is", a, b2, c
    ##one = ctx.one
    s = 0
    t = 1
    k = 0

    while 1:
        #h = ctx.hyp2f1(a,b2,c,z2,zeroprec=ctx.prec,**kwargs)
        #print a.__class__, b2.__class__, c.__class__, z2.__class__
        h = special.hyp2f1(float(a), float(b2), float(c), float(z2))
        if not np.isfinite(h):
            break
        term = t * h
        if abs(term) < eps and abs(h) > 10*eps:
            break
        s += term
        k += 1
        t = (t*a*b1*z1) / (c*k)
        c += 1 # one
        a += 1 # one
        b1 += 1 # one
        #print k, h, term, s
        #if (k/200.)==int(k/200.) or k==171: pdb.set_trace()


    return s

def ellke2(k, tol=100*eps, maxiter=100):
    """Compute complete elliptic integrals of the first kind (K) and
    second kind (E) using the series expansions."""
    # 2011-04-24 21:14 IJMC: Created

    k = np.array(k, copy=False)
    ksum = 0*k
    kprevsum = ksum.copy()
    kresidual = ksum + 1
    #esum = 0*k
    #eprevsum = esum.copy()
    #eresidual = esum + 1
    n = 0
    sqrtpi = np.sqrt(np.pi)
    #kpow = k**0
    #ksq = k*k

    while (np.abs(kresidual) > tol).any() and n <= maxiter:
        #kpow *= (ksq)
        #print kpow==(k**(2*n))
        ksum += ((misc.factorial2(2*n - 1)/misc.factorial2(2*n))**2) * k**(2*n)
        #ksum += (special.gamma(n + 0.5)/special.gamma(n + 1) / sqrtpi) * k**(2*n)
        kresidual = ksum - kprevsum
        kprevsum = ksum.copy()
        n += 1
        #print n, kresidual

    return ksum * (np.pi/2.)




def ellke(k):
    """Compute Hasting's polynomial approximation for the complete
    elliptic integral of the first (ek) and second (kk) kind.
    :INPUTS:
       k -- scalar or Numpy array
    :OUTPUTS:
       ek, kk
    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJC: Adapted from J. Eastman's IDL code.

    m1 = 1. - k**2
    logm1 = np.log(m1)

    # First kind:
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    ee1 = 1. + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ee2 = m1 * (b1 + m1*(b2 + m1*(b3 + m1*b4))) * (-logm1)

    # Second kind:
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ek2 = (b0 + m1*(b1 + m1*(b2 + m1*(b3 + m1*b4)))) * logm1

    return ee1 + ee2, ek1 - ek2


def ellpic_bulirsch(n, k, tol=1000*eps, maxiter=1e4):
    """Compute the complete elliptical integral of the third kind
    using the algorithm of Bulirsch (1965).
    :INPUTS:
       n -- scalar or Numpy array
       k-- scalar or Numpy array
    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJMC: Adapted from J. Eastman's IDL code.
    # 2011-04-25 11:40 IJMC: Set a more stringent tolerance (from 1e-8
    #                  to 1e-14), and fixed tolerance flag to the
    #                  maximum of all residuals.
    # 2013-04-13 21:31 IJMC: Changed 'max' call to 'any'; minor speed boost.

    # Make p, k into vectors:
    #if not hasattr(n, '__iter__'):
    #    n = array([n])
    #if not hasattr(k, '__iter__'):
    #    k = array([k])

    if not hasattr(n,'__iter__'):
        n = np.array([n])
    if not hasattr(k,'__iter__'):
        k = np.array([k])

    if len(n)==0 or len(k)==0:
        return np.array([])

    kc = np.sqrt(1. - k**2)
    p = n + 1.

    if min(p) < 0:
        print("Negative p")

    # Initialize:
    m0 = np.array(1.)
    c = np.array(1.)
    p = np.sqrt(p)
    d = 1./p
    e = kc.copy()

    outsideTolerance = True
    iter = 0
    while outsideTolerance and iter<maxiter:
        f = c.copy()
        c = d/p + c
        g = e/p
        d = 2. * (f*g + d)
        p = g + p;
        g = m0.copy()
        m0 = kc + m0
        if ((np.abs(1. - kc/g)) > tol).any():
            kc = 2. * np.sqrt(e)
            e = kc * m0
            iter += 1
        else:
            outsideTolerance = False
        #if (iter/10.) == (iter/10):
        #    print iter, (np.abs(1. - kc/g))
        #pdb.set_trace()
        ## For debugging:
        #print min(np.abs(1. - kc/g)) > tol
        #print 'tolerance>>', tol
        #print 'minimum>>  ', min(np.abs(1. - kc/g))
        #print 'maximum>>  ', max(np.abs(1. - kc/g)) #, (np.abs(1. - kc/g))

    return .5 * np.pi * (c*m0 + d) / (m0 * (m0 + p))

def z2dt_circular(per, inc, ars, z):
    """ Convert transit crossing parameter z to a time offset for circular orbits.
    :INPUTS:
        per --  scalar. planetary orbital period
        inc -- scalar. orbital inclination (in degrees)
        ars -- scalar.  ratio a/Rs,  orbital semimajor axis over stellar radius
        z -- scalar or array; transit crossing parameter z.
    :RETURNS:
        |dt| -- magnitude of the time offset from transit center at
                which specified z occurs.
        """
    # 2011-06-14 11:26 IJMC: Created.

    numer = (z / ars)**2 - 1.
    denom = np.cos(inc*np.pi/180.)**2 - 1.
    dt = (per / (2*np.pi)) * np.arccos(np.sqrt(numer / denom))

    return dt

def uniform(*arg, **kw):
    """Placeholder for my old code; the new function is called
    :func:`occultuniform`.
    """
    # 2011-04-19 15:06 IJMC: Created
    print("The function 'uniform()' is deprecated.")
    print("Please use occultuniform() in the future.")
    return occultuniform(*arg, **kw)


def occultuniform(z, p, complement=False, verbose=False):
    """Uniform-disk transit light curve (i.e., no limb darkening).
    :INPUTS:
       z -- scalar or sequence; positional offset values of planet in
            units of the stellar radius.
       p -- scalar;  planet/star radius ratio.
       complement : bool
         If True, return (1 - occultuniform(z, p))
    :SEE ALSO:  :func:`t2z`, :func:`occultquad`, :func:`occultnonlin_small`
    """
    # 2011-04-15 16:56 IJC: Added a tad of documentation
    # 2011-04-19 15:21 IJMC: Cleaned up documentation.
    # 2011-04-25 11:07 IJMC: Can now handle scalar z input.
    # 2011-05-15 10:20 IJMC: Fixed indexing check (size, not len)
    # 2012-03-09 08:30 IJMC: Added "complement" argument for backwards
    #                        compatibility, and fixed arccos error at
    #                        1st/4th contact point (credit to
    #                        S. Aigrain @ Oxford)
    # 2013-04-13 21:28 IJMC: Some code optimization; ~20% gain.

    z = np.abs(np.array(z,copy=True))
    fsecondary = np.zeros(z.shape,float)
    if p < 0:
        pneg = True
        p = np.abs(p)
    else:
        pneg = False

    p2 = p*p

    if len(z.shape)>0: # array entered
        i1 = (1+p)<z
        i2 = (np.abs(1-p) < z) * (z<= (1+p))
        i3 = z<= (1-p)
        i4 = z<=(p-1)

        any2 = i2.any()
        any3 = i3.any()
        any4 = i4.any()
        #print i1.sum(),i2.sum(),i3.sum(),i4.sum()

        if any2:
            zi2 = z[i2]
            zi2sq = zi2*zi2
            arg1 = 1 - p2 + zi2sq
            acosarg1 = (p2+zi2sq-1)/(2.*p*zi2)
            acosarg2 = arg1/(2*zi2)
            acosarg1[acosarg1 > 1] = 1.  # quick fix for numerical precision errors
            acosarg2[acosarg2 > 1] = 1.  # quick fix for numerical precision errors
            k0 = np.arccos(acosarg1)
            k1 = np.arccos(acosarg2)
            k2 = 0.5*np.sqrt(4*zi2sq-arg1*arg1)
            fsecondary[i2] = (1./np.pi)*(p2*k0 + k1 - k2)

        fsecondary[i1] = 0.
        if any3: fsecondary[i3] = p2
        if any4: fsecondary[i4] = 1.

        if verbose:
            if not (i1+i2+i3+i4).all():
                print("warning -- some input values not indexed!")
            if (i1.sum()+i2.sum()+i3.sum()+i4.sum() != z.size):
                print("warning -- indexing didn't get the right number of values")



    else:  # scalar entered
        if (1+p)<=z:
            fsecondary = 0.
        elif (np.abs(1-p) < z) * (z<= (1+p)):
            z2 = z*z
            k0 = np.arccos((p2+z2-1)/(2.*p*z))
            k1 = np.arccos((1-p2+z2)/(2*z))
            k2 = 0.5*np.sqrt(4*z2-(1+z2-p2)**2)
            fsecondary = (1./np.pi)*(p2*k0 + k1 - k2)
        elif z<= (1-p):
            fsecondary = p2
        elif z<=(p-1):
            fsecondary = 1.

    if pneg:
        fsecondary *= -1

    if complement:
        return fsecondary
    else:
        return 1. - fsecondary


def integral_smallplanet_nonlinear(z, p, cn, lower, upper):
    """Return the integral in I*(z) in Eqn. 8 of Mandel & Agol (2002).
    -- Int[I(r) 2r dr]_{z-p}^{1}, where:
    :INPUTS:
         z = scalar or array.  Distance between center of star &
             planet, normalized by the stellar radius.
         p = scalar.  Planet/star radius ratio.
         cn = 4-sequence.  Nonlinear limb-darkening coefficients,
              e.g. from Claret 2000.
         lower, upper -- floats. Limits of integration in units of mu
    :RETURNS:
         value of the integral at specified z.
         """
    # 2010-11-06 14:12 IJC: Created
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero

    #import pdb

    z = np.array(z, copy=True)
    z[z==0] = zeroval
    lower = np.array(lower, copy=True)
    upper = np.array(upper, copy=True)
    a = (z - p)**2

    def eval_int_at_limit(limit, cn):
        """Evaluate the integral at a specified limit (upper or lower)"""
        term1 = cn[0] * (1. - 0.8 * np.sqrt(limit))
        term2 = cn[1] * (1. - (2./3.) * limit)
        term3 = cn[2] * (1. - (4./7.) * limit**1.5)
        term4 = cn[3] * (1. - 0.5 * limit**2)

        return -(limit**2) * (1. - term1 - term2 - term3 - term4)

    ret = eval_int_at_limit(upper, cn) - eval_int_at_limit(lower, cn)

    return ret


def smallplanet_nonlinear(*arg, **kw):
    """Placeholder for backwards compatibility with my old code.  The
     function is now called :func:`occultnonlin_small`.
    """
    # 2011-04-19 15:10 IJMC: Created

    print("The function 'smallplanet_nonlinear()' is deprecated.")
    print("Please use occultnonlin_small() in the future.")

    return occultnonlin_small(*arg, **kw)

def occultnonlin_small(z,p, cn):
    """Nonlinear limb-darkening light curve in the small-planet
    approximation (section 5 of Mandel & Agol 2002).
    :INPUTS:
        z -- sequence of positional offset values
        p -- planet/star radius ratio
        cn -- four-sequence nonlinear limb darkening coefficients.  If
              a shorter sequence is entered, the later values will be
              set to zero.
    :NOTE:
       I had to divide the effect at the near-edge of the light curve
       by pi for consistency; this factor was not in Mandel & Agol, so
       I may have coded something incorrectly (or there was a typo).
    :EXAMPLE:
       ::
         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         cns = vstack((zeros(4), eye(4)))
         figure()
         for coef in cns:
             f = occultnonlin_small(z, 0.1, coef)
             plot(z, f, '--')
    :SEE ALSO:
       :func:`t2z`
    """
    # 2010-11-06 14:23 IJC: Created
    # 2011-04-19 15:22 IJMC: Updated documentation.  Renamed.
    # 2011-05-24 14:00 IJMC: Now check the size of cn.
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero

    #import pdb

    cn = np.array([cn], copy=True).ravel()
    if cn.size < 4:
        cn = np.concatenate((cn, [0.]*(4-cn.size)))

    z = np.array(z, copy=True)
    F = np.ones(z.shape, float)
    z[z==0] = zeroval # cheat!

    a = (z - p)**2
    b = (z + p)**2
    c0 = 1. - np.sum(cn)
    Omega = 0.25 * c0 + np.sum( cn / np.arange(5., 9.) )

    ind1 = ((1. - p) < z) * ((1. + p) > z)
    ind2 = z <= (1. - p)

    # Need to specify limits of integration in terms of mu (not r)
    Istar_edge = integral_smallplanet_nonlinear(z[ind1], p, cn, \
                                                np.sqrt(1. - a[ind1]), 0.) / \
                                                (1. - a[ind1])
    Istar_inside = integral_smallplanet_nonlinear(z[ind2], p, cn, \
                                              np.sqrt(1. - a[ind2]), \
                                              np.sqrt(1. - b[ind2])) / \
                                              (4. * z[ind2] * p)


    term1 = 0.25 * Istar_edge / (np.pi * Omega)
    term2 = p**2 * np.arccos((z[ind1] - 1.) / p)
    term3 = (z[ind1] - 1) * np.sqrt(p**2 - (z[ind1] - 1)**2)

    term4 = 0.25 * p**2 * Istar_inside / Omega

    F[ind1] = 1. - term1 * (term2 - term3)
    F[ind2] = 1. - term4

    #pdb.set_trace()
    return F

def occultquad(z,p0, gamma, retall=False, verbose=False):
    """Quadratic limb-darkening light curve; cf. Section 4 of Mandel & Agol (2002).
    :INPUTS:
        z -- sequence of positional offset values
        p0 -- planet/star radius ratio
        gamma -- two-sequence.
           quadratic limb darkening coefficients.  (c1=c3=0; c2 =
           gamma[0] + 2*gamma[1], c4 = -gamma[1]).  If only a single
           gamma is used, then you're assuming linear limb-darkening.
    :OPTIONS:
        retall -- bool.
           If True, in addition to the light curve return the
           uniform-disk light curve, lambda^d, and eta^d parameters.
           Using these quantities allows for quicker model generation
           with new limb-darkening coefficients -- the speed boost is
           roughly a factor of 50.  See the second example below.
    :EXAMPLE:
       ::
         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         gammavals = [[0., 0.], [1., 0.], [2., -1.]]
         figure()
         for gammas in gammavals:
             f = occultquad(z, 0.1, gammas)
             plot(z, f)
       ::
         # Calculate the same geometric transit with two different
         #    sets of limb darkening coefficients:
         from pylab import *
         import transit
         p, b = 0.1, 0.5
         x = (arange(300.)/299. - 0.5)*2.
         z = sqrt(x**2 + b**2)
         gammas = [.25, .75]
         F1, Funi, lambdad, etad = occultquad(z, p, gammas, retall=True)
         gammas = [.35, .55]
         F2 = 1. - ((1. - gammas[0] - 2.*gammas[1])*(1. - F1) +
            (gammas[0] + 2.*gammas[1])*(lambdad + 2./3.*(p > z)) + gammas[1]*etad) /
            (1. - gammas[0]/3. - gammas[1]/6.)
         figure()
         plot(x, F1, x, F2)
         legend(['F1', 'F2'])
    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`
    :NOTES:
       In writing this I relied heavily on the occultquad IDL routine
       by E. Agol and J. Eastman, especially for efficient computation
       of elliptical integrals and for identification of several
       apparent typographic errors in the 2002 paper (see comments in
       the source code).
       From some cursory testing, this routine appears about 9 times
       slower than the IDL version.  The difference drops only
       slightly when using precomputed quantities (i.e., retall=True).
       A large portion of time is taken up in :func:`ellpic_bulirsch`
       and :func:`ellke`, but at least as much is taken up by this
       function itself.  More optimization (or a C wrapper) is desired!
    """
    # 2011-04-15 15:58 IJC: Created; forking from smallplanet_nonlinear
    # 2011-05-14 22:03 IJMC: Now linear-limb-darkening is allowed with
    #                        a single parameter passed in.
    # 2013-04-13 21:06 IJMC: Various code tweaks; speed increased by
    #                        ~20% in some cases.
    #import pdb

    # Initialize:
    z = np.array(z, copy=False)
    lambdad = np.zeros(z.shape, float)
    etad = np.zeros(z.shape, float)
    F = np.ones(z.shape, float)

    p = np.abs(p0) # Save the original input

    # Define limb-darkening coefficients:
    if len(gamma) < 2 or not hasattr(gamma, '__iter__'):  # Linear limb-darkening
        gamma = np.concatenate([gamma.ravel(), [0.]])
        c2 = gamma[0]
    else:
        c2 = gamma[0] + 2 * gamma[1]

    c4 = -gamma[1]



    # Test the simplest case (a zero-sized planet):
    if p==0:
        if retall:
            ret = np.ones(z.shape, float), np.ones(z.shape, float), \
                  np.zeros(z.shape, float), np.zeros(z.shape, float)
        else:
            ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    fourOmega = 1. - gamma[0]/3. - gamma[1]/6. # Actually 4*Omega
    a = (z - p)*(z - p)
    b = (z + p)*(z + p)
    k = 0.5 * np.sqrt((1. - a) / (z * p))
    p2 = p*p
    z2 = z*z
    ninePi = 9*np.pi

    # Define the many necessary indices for the different cases:
    pgt0 = p > 0

    i01 = pgt0 * (z >= (1. + p))
    i02 = pgt0 * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = pgt0 * (p < 0.5) * (z > p) * (z < (1. - p))
    i04 = pgt0 * (p < 0.5) * (z == (1. - p))
    i05 = pgt0 * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i09 = pgt0 * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = pgt0 * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))
    #any01 = i01.any()
    #any02 = i02.any()
    #any03 = i03.any()
    any04 = i04.any()
    any05 = i05.any()
    any06 = i06.any()
    any07 = i07.any()
    #any08 = i08.any()
    #any09 = i09.any()
    any10 = i10.any()
    any11 = i11.any()
    #print n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11
    if verbose:
        allind = i01 + i02 + i03 + i04 + i05 + i06 + i07 + i08 + i09 + i10 + i11
        nused = (i01.sum() + i02.sum() + i03.sum() + i04.sum() + \
                     i05.sum() + i06.sum() + i07.sum() + i08.sum() + \
                     i09.sum() + i10.sum() + i11.sum())

        print("%i/%i indices used" % (nused, i01.size))
        if not allind.all():
            print("Some indices not used!")

    #pdb.set_trace()


    # Lambda^e and eta^d are more tricky:
    # Simple cases:
    lambdad[i01] = 0.
    etad[i01] = 0.

    if any06:
        lambdad[i06] = 1./3. - 4./ninePi
        etad[i06] = 0.09375 # = 3./32.

    if any11:
        lambdad[i11] = 1.
        # etad[i11] = 1.  # This is what the paper says
        etad[i11] = 0.5 # Typo in paper (according to J. Eastman)


    # Lambda_1:
    ilam1 = i02 + i08
    q1 = p2 - z2[ilam1]
    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - 1./a[ilam1], k[ilam1])
    # ellipe, ellipk = ellke(k[ilam1])

    # This is what J. Eastman's code has:

    # 2011-04-24 20:32 IJMC: The following codes act funny when
    #                        sqrt((1-a)/(b-a)) approaches unity.
    qq = np.sqrt((1. - a[ilam1]) / (b[ilam1] - a[ilam1]))
    ellippi = ellpic_bulirsch(1./a[ilam1] - 1., qq)
    ellipe, ellipk = ellke(qq)
    lambdad[ilam1] = (1./ (ninePi*np.sqrt(p*z[ilam1]))) * \
        ( ((1. - b[ilam1])*(2*b[ilam1] + a[ilam1] - 3) - \
               3*q1*(b[ilam1] - 2.)) * ellipk + \
              4*p*z[ilam1]*(z2[ilam1] + 7*p2 - 4.) * ellipe - \
              3*(q1/a[ilam1])*ellippi)

    # Lambda_2:
    ilam2 = i03 + i09
    q2 = p2 - z2[ilam2]

    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - b[ilam2]/a[ilam2], 1./k[ilam2])
    # ellipe, ellipk = ellke(1./k[ilam2])

    # This is what J. Eastman's code has:
    ailam2 = a[ilam2] # Pre-cached for speed
    bilam2 = b[ilam2] # Pre-cached for speed
    omailam2 = 1. - ailam2 # Pre-cached for speed
    ellippi = ellpic_bulirsch(bilam2/ailam2 - 1, np.sqrt((bilam2 - ailam2)/(omailam2)))
    ellipe, ellipk = ellke(np.sqrt((bilam2 - ailam2)/(omailam2)))

    lambdad[ilam2] = (2. / (ninePi*np.sqrt(omailam2))) * \
        ((1. - 5*z2[ilam2] + p2 + q2*q2) * ellipk + \
             (omailam2)*(z2[ilam2] + 7*p2 - 4.) * ellipe - \
             3*(q2/ailam2)*ellippi)


    # Lambda_3:
    #ellipe, ellipk = ellke(0.5/ k)  # This is what the paper says
    if any07:
        ellipe, ellipk = ellke(0.5/ p)  # Corrected typo (1/2k -> 1/2p), according to J. Eastman
        lambdad[i07] = 1./3. + (16.*p*(2*p2 - 1.)*ellipe -
                                (1. - 4*p2)*(3. - 8*p2)*ellipk / p) / ninePi


    # Lambda_4
    #ellipe, ellipk = ellke(2. * k)  # This is what the paper says
    if any05:
        ellipe, ellipk = ellke(2. * p)  # Corrected typo (2k -> 2p), according to J. Eastman
        lambdad[i05] = 1./3. + (2./ninePi) * (4*(2*p2 - 1.)*ellipe + (1. - 4*p2)*ellipk)

    # Lambda_5
    ## The following line is what the 2002 paper says:
    #lambdad[i04] = (2./(3*np.pi)) * (np.arccos(1 - 2*p) - (2./3.) * (3. + 2*p - 8*p2))
    # The following line is what J. Eastman's code says:
    if any04:
        lambdad[i04] = (2./3.) * (np.arccos(1. - 2*p)/np.pi - \
                                      (6./ninePi) * np.sqrt(p * (1.-p)) * \
                                      (3. + 2*p - 8*p2) - \
                                      float(p > 0.5))

    # Lambda_6
    if any10:
        lambdad[i10] = -(2./3.) * (1. - p2)**1.5

    # Eta_1:
    ilam3 = ilam1 + i07 # = i02 + i07 + i08
    z2ilam3  = z2[ilam3]    # pre-cache for better speed
    twoZilam3  = 2*z[ilam3] # pre-cache for better speed
    #kappa0 = np.arccos((p2+z2ilam3-1)/(p*twoZilam3))
    #kappa1 = np.arccos((1-p2+z2ilam3)/(twoZilam3))
    #etad[ilam3] = \
    #    (0.5/np.pi) * (kappa1 + kappa0*p2*(p2 + 2*z2ilam3) - \
    #                    0.25*(1. + 5*p2 + z2ilam3) * \
    #                    np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.)))
    etad[ilam3] = \
        (0.5/np.pi) * ((np.arccos((1-p2+z2ilam3)/(twoZilam3))) + (np.arccos((p2+z2ilam3-1)/(p*twoZilam3)))*p2*(p2 + 2*z2ilam3) - \
                        0.25*(1. + 5*p2 + z2ilam3) * \
                        np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.)))


    # Eta_2:
    etad[ilam2 + i04 + i05 + i10] = 0.5 * p2 * (p2 + 2. * z2[ilam2 + i04 + i05 + i10])


    # We're done!


    ## The following are handy for debugging:
    #term1 = (1. - c2) * lambdae
    #term2 = c2*lambdad
    #term3 = c2*(2./3.) * (p>z).astype(float)
    #term4 = c4 * etad
    # Lambda^e is easy:
    lambdae = 1. - occultuniform(z, p)
    F = 1. - ((1. - c2) * lambdae + \
                  c2 * (lambdad + (2./3.) * (p > z).astype(float)) - \
                  c4 * etad) / fourOmega

    #pdb.set_trace()
    if retall:
        ret = F, lambdae, lambdad, etad
    else:
        ret = F

    #pdb.set_trace()
    return ret

def occultnonlin(z,p0, cn):
    """Nonlinear limb-darkening light curve; cf. Section 3 of Mandel & Agol (2002).
    :INPUTS:
        z -- sequence of positional offset values
        p0 -- planet/star radius ratio
        cn -- four-sequence. nonlinear limb darkening coefficients
    :EXAMPLE:
        ::
         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 50)
         cns = vstack((zeros(4), eye(4)))
         figure()
         for coef in cns:
             f = occultnonlin(z, 0.1, coef)
             plot(z, f)
    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`, :func:`occultquad`
    :NOTES:
        Scipy is much faster than mpmath for computing the Beta and
        Gauss hypergeometric functions.  However, Scipy does not have
        the Appell hypergeometric function -- the current version is
        not vectorized.
    """
    # 2011-04-15 15:58 IJC: Created; forking from occultquad
    #import pdb

    # Initialize:
    cn0 = np.array(cn, copy=True)
    z = np.array(z, copy=True)
    F = np.ones(z.shape, float)

    p = np.abs(p0) # Save the original input


    # Test the simplest case (a zero-sized planet):
    if p==0:
        ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    c0 = 1. - np.sum(cn0)
    # Row vectors:
    c = np.concatenate(([c0], cn0))
    n = np.arange(5, dtype=float)
    # Column vectors:
    cc = c.reshape(5, 1)
    nn = n.reshape(5,1)
    np4 = n + 4.
    nd4 = n / 4.
    twoOmega = 0.5*c[0] + 0.4*c[1] + c[2]/3. + 2.*c[3]/7. + 0.25*c[4]

    a = (z - p)**2
    b = (z + p)**2
    am1 = a - 1.
    bma = b - a

    k = 0.5 * np.sqrt(-am1 / (z * p))
    p2 = p**2
    z2 = z**2


    # Define the many necessary indices for the different cases:
    i01 = (p > 0) * (z >= (1. + p))
    i02 = (p > 0) * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = (p > 0) * (p < 0.5) * (z > p) * (z <= (1. - p))  # also contains Case 4
    #i04 = (z==(1. - p))
    i05 = (p > 0) * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i08a = (p == 1) * (z == 0)
    i09 = (p > 0) * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = (p > 0) * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))

    iN = i02 + i08
    iM = i03 + i09

    # Compute N and M for the appropriate indices:
    #  (Use the slow, non-vectorized appellf1 function:)
    myappellf1 = np.frompyfunc(appellf1, 6, 1)
    N = np.zeros((5, z.size), float)
    M = np.zeros((3, z.size), float)
    if iN.any():
        termN = myappellf1(0.5, 1., 0.5, 0.25*nn + 2.5, am1[iN]/a[iN], -am1[iN]/bma[iN])
        N[:, iN] = ((-am1[iN])**(0.25*nn + 1.5)) / np.sqrt(bma[iN]) * \
            special.beta(0.25*nn + 2., 0.5) * \
            (((z2[iN] - p2) / a[iN]) * termN - \
                 special.hyp2f1(0.5, 0.5, 0.25*nn + 2.5, -am1[iN]/bma[iN]))

    if iM.any():
        termM = myappellf1(0.5, -0.25*nn[1:4] - 1., 1., 1., -bma[iM]/am1[iM], -bma[iM]/a[iM])
        M[:, iM] = ((-am1[iM])**(0.25*nn[1:4] + 1.)) * \
            (((z2[iM] - p2)/a[iM]) * termM - \
                 special.hyp2f1(-0.25*nn[1:4] - 1., 0.5, 1., -bma[iM]/am1[iM]))


    # Begin going through all the cases:

    # Case 1:
    F[i01] = 1.

    # Case 2: (Gauss and Appell hypergeometric functions)
    F[i02] = 1. - (1. / (np.pi*twoOmega)) * \
        (N[:, i02] * cc/(nn + 4.) ).sum(0)

    # Case 3 : (Gauss and Appell hypergeometric functions)
    F[i03] = 1. - (0.5/twoOmega) * \
        (c0*p2 + 2*(M[:, i03] * cc[1:4]/(nn[1:4] + 4.)).sum(0) + \
             c[-1]*p2*(1. - 0.5*p2 - z2[i03]))

    #if i04.any():
    #    F[i04] = occultnonlin_small(z[i04], p, cn)
    #    print "Value found for z = 1-p: using small-planet approximation "
    #    print "where Appell F2 function will not otherwise converge."

    #pdb.set_trace()
    #F[i04] = 0.5 * (occultnonlin(z[i04]+p/2., p, cn) + occultnonlin(z[i04]-p/2., p, cn))

    # Case 5: (Gauss hypergeometric function)
    F[i05] = 0.5 + \
        ((c/np4) * special.hyp2f1(0.5, -nd4 - 1., 1., 4*p2)).sum() / twoOmega

    # Case 6:  Gamma function
    F[i06] = 0.5 + (1./(np.sqrt(np.pi) * twoOmega)) * \
        ((c/np4) * special.gamma(1.5 + nd4) / special.gamma(2. + nd4)).sum()

    # Case 7: Gauss hypergeometric function, beta function
    F[i07] = 0.5 + (0.5/(p * np.pi * twoOmega)) * \
        ((c/np4) * special.beta(0.5, nd4 + 2.) * \
             special.hyp2f1(0.5, 0.5, 2.5 + nd4, 0.25/p2)).sum()

    # Case 8: (Gauss and Appell hypergeometric functions)
    F[i08a] = 0.
    F[i08] =  -(1. / (np.pi*twoOmega)) * (N[:, i02] * cc/(nn + 4.) ).sum(0)

    # Case 9: (Gauss and Appell hypergeometric functions)
    F[i09] = (0.5/twoOmega) * \
        (c0 * (1. - p2) + c[-1] * (0.5 - p2*(1. - 0.5*p2 - z2[i09])) - \
             2*(M[:, i09] * cc[1:4] / (nn[1:4] + 4.)).sum(0))

    # Case 10:
    F[i10] = (2. / twoOmega) * ((c/np4) * (1. - p2)**(nd4 + 1.)).sum()

    # Case 11:
    F[i11] = 0.


    # We're done!

    return F

