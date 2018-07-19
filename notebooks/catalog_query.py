from astroquery.simbad import Simbad
result_table = Simbad.query_object("m1")
result_table.pprint(show_unit=True)

from astroquery.sdss import SDSS
from astropy import coordinates as coords
#pos = coords.SkyCoord(packet['candidate']['ra'],packet['candidate']['dec'], unit="deg”) #to use info from avro packet
pos = coords.SkyCoord(244.566202,16.138884, unit="deg”) #to test query

xid = SDSS.query_region(pos, spectro=True)
print(xid)
print(xid.columns)
xid['z']

sp = SDSS.get_spectra(matches=xid)
im = SDSS.get_images(matches=xid, band='r')
