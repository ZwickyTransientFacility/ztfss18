"""A collection of utility functions to support the notebooks"""

import numpy as np
import astropy.coordinates as coords
import astropy.units as u
import shelve, pickle
from BaryTime import BaryTime

P48_loc = coords.EarthLocation(lat=coords.Latitude('33d21m26.35s'),
    lon=coords.Longitude('-116d51m32.04s'),
    height=1707.)

def barycenter_times(mjds, ra, dec, loc=P48_loc):
    t = BaryTime(mjds, format='mjd',scale='utc',
        location=loc)
    obj_coords = coords.SkyCoord(ra,dec,frame='icrs', unit=u.deg)

    return t.bcor(obj_coords).mjd



def source_lightcurve(rel_phot_shlv, ra, dec, matchr = 1.0):
    """Crossmatch ra and dec to a PTF shelve file, to return light curve of a given star"""
    shelf = shelve.open(rel_phot_shlv)
    ref_coords = coords.SkyCoord(shelf["ref_coords"].ra, shelf["ref_coords"].dec,frame='icrs',unit='deg')    
    
    source_coords = coords.SkyCoord(ra, dec,frame='icrs',unit='deg')
    idx, sep, dist = coords.match_coordinates_sky(source_coords, ref_coords)        
    
    wmatch = (sep <= matchr*u.arcsec)
    
    if sum(wmatch) == 1:
        mjds = shelf["mjds"]
        mags = shelf["mags"][idx]
        magerrs = shelf["magerrs"][idx]

        # filter so we only return good points
        wgood = (mags.mask == False)

        if (np.sum(wgood) == 0):
            raise ValueError("No good photometry at this position.")
        
        return mjds[wgood], mags[wgood], magerrs[wgood]

    else:
        raise ValueError("There are no matches to the provided coordinates within %.1f arcsec" % (matchr))
