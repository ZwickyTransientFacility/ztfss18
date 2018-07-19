# https://github.com/astropy/astropy/issues/2244#issuecomment-64900470
from astropy import time
from astropy import constants as const
from astropy import units as u
from astropy.utils.iers import IERS
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
from astropy.io import ascii
from astropy import coordinates as coord
from astropy import _erfa as erfa
import numpy as np
import warnings

# mean sidereal rate (at J2000) in radians per (UT1) second
SR = 7.292115855306589e-5

''' time class: inherits astropy time object, and adds heliocentric, barycentric
    correction utilities.'''
class BaryTime(time.Time):

    def __init__(self,*args,**kwargs):
        super(BaryTime, self).__init__(*args, **kwargs)
#        self.height = kwargs.get('height',0.0)

    def _pvobs(self):
        '''calculates position and velocity of the observatory
           returns position/velocity in AU and AU/d in GCRS reference frame
        '''

        # convert obs position from WGS84 (lat long) to ITRF geocentric coords in AU
        xyz = self.location.to(u.AU).value

        # now we need to convert this position to Celestial Coords
        # specifically, the GCRS coords.
        # conversion from celestial to terrestrial coords given by
        # [TRS] = RPOM * R_3(ERA) * RC2I * [CRS]
        # where:
        # [CRS] is vector in GCRS (geocentric celestial system)
        # [TRS] is vector in ITRS (International Terrestrial Ref System)
        # ERA is earth rotation angle
        # RPOM = polar motion matrix

        tt = self.tt
        mjd = self.utc.mjd

        # we need the IERS values to correct for the precession/nutation of the Earth
        iers_tab = IERS.open()

        # Find UT1, which is needed to calculate ERA
        # uses IERS_B by default , for more recent times use IERS_A download
        try:      
            ut1 = self.ut1 
        except:
            try:
                iers_a_file = download_file(IERS_A_URL, cache=True)
                iers_a = IERS_A.open(iers_a_file)
                self.delta_ut1_utc = self.get_delta_ut1_utc(iers_a)
                ut1 = self.ut1
            except:
                # fall back to UTC with degraded accuracy
                warnings.warn('Cannot calculate UT1: using UTC with degraded accuracy') 
                ut1 = self.utc

        # Gets x,y coords of Celestial Intermediate Pole (CIP) and CIO locator s
        # CIO = Celestial Intermediate Origin
        # Both in GCRS
        X,Y,S = erfa.xys00a(tt.jd1,tt.jd2)

        # Get dX and dY from IERS B
        dX = np.interp(mjd, iers_tab['MJD'], iers_tab['dX_2000A']) * u.arcsec 
        dY = np.interp(mjd, iers_tab['MJD'], iers_tab['dY_2000A']) * u.arcsec

        # Get GCRS to CIRS matrix
        # can be used to convert to Celestial Intermediate Ref Sys
        # from GCRS.
        rc2i = erfa.c2ixys(X+dX.to(u.rad).value, Y+dY.to(u.rad).value, S)

        # Gets the Terrestrial Intermediate Origin (TIO) locator s'
        # Terrestrial Intermediate Ref Sys (TIRS) defined by TIO and CIP.
        # TIRS related to to CIRS by Earth Rotation Angle
        sp = erfa.sp00(tt.jd1,tt.jd2)

        # Get X and Y from IERS B
        # X and Y are
        xp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_x']) * u.arcsec
        yp = np.interp(mjd, iers_tab['MJD'], iers_tab['PM_y']) * u.arcsec 

        # Get the polar motion matrix. Relates ITRF to TIRS.
        rpm = erfa.pom00(xp.to(u.rad).value, yp.to(u.rad).value, sp)

        # multiply ITRF position of obs by transpose of polar motion matrix
        # Gives Intermediate Ref Frame position of obs
        x,y,z = np.array([rpmMat.T.dot(xyz) for rpmMat in rpm]).T

        # Functions of Earth Rotation Angle, theta
        # Theta is angle bewtween TIO and CIO (along CIP)
        # USE UT1 here.
        theta = erfa.era00(ut1.jd1,ut1.jd2)
        S,C = np.sin(theta),np.cos(theta)

        # Position #GOT HERE
        pos = np.asarray([C*x - S*y, S*x + C*y, z]).T

        # multiply by inverse of GCRS to CIRS matrix
        # different methods for scalar times vs arrays
        if pos.ndim > 1:
            pos = np.array([np.dot(rc2i[j].T,pos[j]) for j in range(len(pos))])
        else:   
            pos = np.dot(rc2i.T,pos)

        # Velocity
        vel = np.asarray([SR*(-S*x - C*y), SR*(C*x-S*y), np.zeros_like(x)]).T
        # multiply by inverse of GCRS to CIRS matrix
        if vel.ndim > 1:
            vel = np.array([np.dot(rc2i[j].T,vel[j]) for j in range(len(pos))])
        else:        
            vel = np.dot(rc2i.T,vel)

        #return position and velocity
        return pos,vel

    def _obs_pos(self):
        '''calculates heliocentric and barycentric position of the earth in AU and AU/d'''
        tdb = self.tdb

        # get heliocentric and barycentric position and velocity of Earth
        # BCRS reference frame
        h_pv,b_pv = erfa.epv00(tdb.jd1,tdb.jd2)

        # h_pv etc can be shape (ntimes,2,3) or (2,3) if given a scalar time
        if h_pv.ndim == 2:
            h_pv = h_pv[np.newaxis,:]
        if b_pv.ndim == 2:
            b_pv = b_pv[np.newaxis,:]

        # unpack into position and velocity arrays
        h_pos = h_pv[:,0,:]
        h_vel = h_pv[:,1,:]

        # unpack into position and velocity arrays
        b_pos = b_pv[:,0,:]
        b_vel = b_pv[:,1,:]

        #now need position and velocity of observing station
        pos_obs, vel_obs = self._pvobs()

        #add this to heliocentric and barycentric position of center of Earth
        h_pos += pos_obs
        b_pos += pos_obs
        h_vel += vel_obs        
        b_vel += vel_obs        
        return (h_pos,h_vel,b_pos,b_vel)

    def _vect(self,coord):
        '''get unit vector pointing to star, and modulus of vector, in AU
           coordinate of star supplied as astropy.coordinate object

           assume zero proper motion, parallax and radial velocity'''
        pmra = pmdec = px = rv = 0.0

        rar  = coord.ra.radian
        decr = coord.dec.radian
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore warnings about 0 parallax
            pos,vel = erfa.starpv(rar,decr,pmra,pmdec,px,rv)

        modulus = np.sqrt(pos.dot(pos))
        unit = pos/modulus
        modulus /= const.au.value
        return modulus, unit

    def hcor(self,coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of observatory, in AU, AU/d
        h_pos,h_vel,b_pos,b_vel = self._obs_pos()

        # heliocentric light travel time, s
        tcor_hel = const.au.value * np.array([np.dot(spos,hpos) for hpos in h_pos]) / const.c.value
        #print 'Correction to add to get time at heliocentre = %.7f s' % tcor_hel
        dt = time.TimeDelta(tcor_hel, format='sec', scale='tdb')
        return self.utc + dt

    def bcor(self,coord):
        mod, spos = self._vect(coord)
        # get helio/bary-centric position and velocity of observatory, in AU, AU/d
        h_pos,h_vel,b_pos,b_vel  = self._obs_pos()

        # barycentric light travel time, s
        tcor_bar = const.au.value *  np.array([np.dot(spos,bpos) for bpos in b_pos]) / const.c.value
        #print 'Correction to add to get time at barycentre  = %.7f s' % tcor_bar
        dt = time.TimeDelta(tcor_bar, format='sec', scale='tdb')
        return self.tdb + dt
