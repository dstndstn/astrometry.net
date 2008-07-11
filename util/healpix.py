from math import *
from starutil import *

import unittest

# returns (base hp, x, y)
def decompose_xy(hp, nside):
    finehp = hp % (nside**2)
    return (int(hp / (nside**2)), int(finehp / nside), finehp % nside)

def get_base_neighbour(hp, dx, dy):
    if isnorthpolar(hp):
        if (dx ==  1) and (dy ==  0):
            return (hp + 1) % 4
        if (dx ==  0) and (dy ==  1):
            return (hp + 3) % 4
        if (dx ==  1) and (dy ==  1):
            return (hp + 2) % 4
        if (dx == -1) and (dy ==  0):
            return (hp + 4)
        if (dx ==  0) and (dy == -1):
            return 4 + ((hp + 1) % 4)
        if (dx == -1) and (dy == -1):
            return hp + 8
        return -1
    elif issouthpolar(hp):
        if (dx ==  1) and (dy ==  0):
            return 4 + ((hp + 1) % 4)
        if (dx ==  0) and (dy ==  1):
            return hp - 4
        if (dx == -1) and (dy ==  0):
            return 8 + ((hp + 3) % 4)
        if (dx ==  0) and (dy == -1):
            return 8 + ((hp + 1) % 4)
        if (dx == -1) and (dy == -1):
            return 8 + ((hp + 2) % 4)
        if (dx ==  1) and (dy ==  1):
            return hp - 8
        return -1
    else:
        if (dx ==  1) and (dy ==  0):
            return hp - 4
        if (dx ==  0) and (dy ==  1):
            return (hp + 3) % 4
        if (dx == -1) and (dy ==  0):
            return 8 + ((hp + 3) % 4)
        if (dx ==  0) and (dy == -1):
            return hp + 4
        if (dx ==  1) and (dy == -1):
            return 4 + ((hp + 1) % 4)
        if (dx == -1) and (dy ==  1):
            return 4 + ((hp - 1) % 4)
        return -1
    return -1

def get_neighbours(hp, nside):
    neighbours = []
    (base, x, y) = decompose_xy(hp, nside)

    # ( + , 0 )
    nx = (x + 1) % nside
    ny = y
    if x == (nside - 1):
        nbase = get_base_neighbour(base, 1, 0)
        if (isnorthpolar(base)):
            nx = x;
            # swap nx,ny
            (nx,ny) = (ny,nx)
    else:
        nbase = base

    neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    # ( + , + )
    nx = (x + 1) % nside
    ny = (y + 1) % nside
    if (x == nside - 1) and (y == nside - 1):
        if (ispolar(base)):
            nbase = get_base_neighbour(base, 1, 1)
        else:
            nbase = -1
    elif (x == (nside - 1)):
        nbase = get_base_neighbour(base, 1, 0)
    elif (y == (nside - 1)):
        nbase = get_base_neighbour(base, 0, 1)
    else:
        nbase = base;

    if isnorthpolar(base):
        if (x == (nside - 1)):
            nx = nside - 1
        if (y == (nside - 1)):
            ny = nside - 1
        if (x == (nside - 1)) or (y == (nside - 1)):
            # swap nx,ny
            (nx,ny) = (ny,nx)

    if (nbase != -1):
        neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    # ( 0 , + )
    nx = x
    ny = (y + 1) % nside
    if (y == (nside - 1)):
        nbase = get_base_neighbour(base, 0, 1)
        if (isnorthpolar(base)):
            ny = y
            # swap nx,ny
            (nx,ny) = (ny,nx)
    else:
        nbase = base

    neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    # ( - , + )
    nx = (x + nside - 1) % nside
    ny = (y + 1) % nside
    if (x == 0) and (y == (nside - 1)):
        if isequatorial(base):
            nbase = get_base_neighbour(base, -1, 1)
        else:
            nbase = -1
    elif (x == 0):
        nbase = get_base_neighbour(base, -1, 0)
        if issouthpolar(base):
            nx = 0
            # swap nx,ny
            (nx,ny) = (ny,nx)
    elif (y == (nside - 1)):
        nbase = get_base_neighbour(base, 0, 1)
        if isnorthpolar(base):
            ny = y
            # swap nx,ny
            (nx,ny) = (ny,nx)
    else:
        nbase = base

    if (nbase != -1):
        neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    # ( - , 0 )
    nx = (x + nside - 1) % nside
    ny = y
    if (x == 0):
        nbase = get_base_neighbour(base, -1, 0)
        if issouthpolar(base):
            nx = 0
            # swap nx,ny
            (nx,ny) = (ny,nx)
    else:
        nbase = base

    neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    # ( - , - )
    nx = (x + nside - 1) % nside
    ny = (y + nside - 1) % nside
    if (x == 0) and (y == 0):
        if ispolar(base):
            nbase = get_base_neighbour(base, -1, -1)
        else:
            nbase = -1
    elif (x == 0):
        nbase = get_base_neighbour(base, -1, 0)
    elif (y == 0):
        nbase = get_base_neighbour(base, 0, -1)
    else:
        nbase = base;

    if issouthpolar(base):
        if (x == 0):
            nx = 0
        if (y == 0):
            ny = 0
        if (x == 0) or (y == 0):
            # swap nx,ny
            (nx,ny) = (ny,nx)

    if (nbase != -1):
        neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    # ( 0 , - )
    ny = (y + nside - 1) % nside
    nx = x
    if (y == 0):
        nbase = get_base_neighbour(base, 0, -1)
        if issouthpolar(base):
            ny = y
            # swap nx,ny
            (nx,ny) = (ny,nx)
    else:
        nbase = base

    neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    # ( + , - )
    nx = (x + 1) % nside
    ny = (y + nside - 1) % nside
    if (x == (nside - 1)) and (y == 0):
        if isequatorial(base):
            nbase = get_base_neighbour(base, 1, -1)
        else:
            nbase = -1

    elif (x == (nside - 1)):
        nbase = get_base_neighbour(base, 1, 0)
        if isnorthpolar(base):
            nx = x
            # swap nx,ny
            (nx,ny) = (ny,nx)
    elif (y == 0):
        nbase = get_base_neighbour(base, 0, -1)
        if issouthpolar(base):
            ny = y
            # swap nx,ny
            (nx,ny) = (ny,nx)
    else:
        nbase = base

    if (nbase != -1):
        neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

    return neighbours

def ispolar(hp):
    # the north polar healpixes are 0,1,2,3
    # the south polar healpixes are 8,9,10,11
    return (hp <= 3) or (hp >= 8)

def isequatorial(hp):
    # the north polar healpixes are 0,1,2,3
    # the south polar healpixes are 8,9,10,11
    return (hp >= 4) and (hp <= 7)

def isnorthpolar(hp):
    return (hp <= 3)

def issouthpolar(hp):
    return (hp >= 8)


# radius of the field, in degrees.
def get_closest_pow2_nside(radius):
    # 4 pi steradians on the sky, into 12 base-level healpixes
    area = 4. * pi / 12.
    # in radians:
    baselen = sqrt(area)
    n = baselen / (radians(radius*2.))
    p = int(round(log(n, 2.0)))
    return 2**p

def compose_xy(x, y, nside):
    return x*nside + y

##### NOTE NOTE NOTE -- this is out of sync with the C version!
def xyztohealpix(x, y, z, nside):
    twothirds = 2. / 3.
    twopi = 2. * pi

    phi = atan2(y, x)
    if phi < 0:
        phi += twopi
    phioverpi = phi / pi

    # North or south pole
    if (z >= twothirds) or (z <= -twothirds):
        # Which pole?
        if z >= twothirds:
            north = True
            zfactor = 1.
        else:
            north = False
            zfactor = -1.

        phit = fmod(phi, pi / 2.)
        assert(phit >= 0)

        root = (1. - z*zfactor) * 3. * (nside * (2. * phit - pi) / pi)**2
        if root <= 0.0:
            x = 1
        else:
            x = int(ceil(sqrt(root)))

        assert(x >= 1)
        assert(x <= nside)

        root = (1. - z*zfactor) * 3. * (nside * 2. * phit / pi)**2
        if root <= 0:
            y = 1
        else:
            y = int(ceil(sqrt(root)))

        assert(y >= 1)
        assert(y <= nside)

        x = nside - x
        y = nside - y

        if not north:
            # swap x,y
            (x,y) = (y,x)

        pnprime = compose_xy(x, y, nside)

        if not north:
            pnprime = nside * nside - 1 - pnprime

        column = int(phi / (pi / 2.))

        if north:
            basehp = column
        else:
            basehp = 8 + column

        hp = basehp * (nside**2) + pnprime
        return hp

    # could be polar or equatorial.
    phim = fmod(phi, pi / 2.)

    # project into the unit square z=[-2/3, 2/3], phi=[0, pi/2]
    zunits = (z + twothirds) / (4. / 3.)
    phiunits = phim / (pi / 2.)
    u1 = (zunits + phiunits) / 2.
    u2 = (zunits - phiunits) / 2.
    # x is the northeast direction, y is the northwest.
    x = int(floor(u1 * 2.0 * nside))
    y = int(floor(u2 * 2.0 * nside))
    x %= nside
    y %= nside
    if x < 0:
        x += nside
    if y < 0:
        y += nside
    pnprime = compose_xy(x, y, nside)

    # now compute which big healpix it's in.
    offset = int(phioverpi * 2.)
    phimod = phioverpi - offset * 0.5
    offset = ((offset % 4) + 4) % 4
    z1 = twothirds  - (8. / 3.) * phimod
    z2 = -twothirds + (8. / 3.) * phimod
    if (z >= z1) and (z >= z2):
        # north polar
        basehp = offset
    elif (z <= z1) and (z <= z2):
        # south polar
        basehp = 8 + offset
    elif phimod < 0.25:
        # left equatorial
        basehp = offset + 4
    else:
        # right equatorial
        basehp = ((offset + 1) % 4) + 4

    hp = basehp * (nside**2) + pnprime
    return hp

# ra, dec in degrees
def radectohealpix(ra, dec, nside):
    (x,y,z) = radectoxyz(ra, dec)
    return xyztohealpix(x,y,z, nside)




class testhealpix(unittest.TestCase):
    def check_neighbours(self, hp, nside, truen):
        neigh = get_neighbours(hp, nside)
        print 'True(%3i): [ %s ]' % (hp, ', '.join([str(n) for n in truen]))
        print 'Got (%3i): [ %s ]' % (hp, ', '.join([str(n) for n in neigh]))
        self.assertEqual(len(neigh), len(truen))
        for n in truen:
            self.assert_(n in neigh)
        
    def test_neighbours(self):
        # from test_healpix.c's output.
        self.check_neighbours(0, 4, [ 4, 5, 1, 77, 76, 143, 83, 87, ])
        self.check_neighbours(12, 4, [ 19, 23, 13, 9, 8, 91, 95, ])
        self.check_neighbours(14, 4, [ 27, 31, 15, 11, 10, 9, 13, 23, ])
        self.check_neighbours(15, 4, [ 31, 47, 63, 62, 11, 10, 14, 27, ])
        self.check_neighbours(27, 4, [ 31, 15, 14, 13, 23, 22, 26, 30, ])
        self.check_neighbours(108, 4, [ 32, 33, 109, 105, 104, 171, 175, 115, ])
        self.check_neighbours(127, 4, [ 51, 44, 40, 123, 122, 126, 50, ])
        self.check_neighbours(64, 4, [ 68, 69, 65, 189, 188, 131, 135, ])
        self.check_neighbours(140, 4, [ 80, 81, 141, 137, 136, 146, 147, ])
        self.check_neighbours(152, 4, [ 156, 157, 153, 149, 148, 161, 162, 163, ])
        self.check_neighbours(160, 4, [ 164, 165, 161, 148, 144, 128, 176, 177, ])
        self.check_neighbours(18, 4, [ 22, 23, 19, 95, 94, 93, 17, 21, ])
        self.check_neighbours(35, 4, [ 39, 29, 28, 111, 110, 34, 38, ])
        self.check_neighbours(55, 4, [ 59, 46, 45, 44, 51, 50, 54, 58, ])
        self.check_neighbours(191, 4, [ 67, 48, 124, 120, 187, 186, 190, 66, ])
        self.check_neighbours(187, 4, [ 191, 124, 120, 116, 183, 182, 186, 190, ])
        self.check_neighbours(179, 4, [ 183, 116, 112, 172, 168, 178, 182, ])
        self.check_neighbours(178, 4, [ 182, 183, 179, 172, 168, 164, 177, 181, ])


if __name__ == '__main__':
    unittest.main()
