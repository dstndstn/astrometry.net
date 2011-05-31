import math

def merc2ra(mx):
    return (1.0-mx) * 360.0

def merc2dec(my):
	return math.atan(math.sinh((my - 0.5) * (2.0*math.pi))) * 180.0/math.pi;

