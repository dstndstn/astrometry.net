from math import *

# RA, Dec in degrees
def radectoxyz(ra, dec):
    rarad = radians(ra)
    decrad = radians(dec)
    cosd = cos(decrad)
    return (cosd * cos(rarad), cosd * sin(rarad), sin(decrad))
