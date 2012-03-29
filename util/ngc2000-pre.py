## FIXME -- use ngcic_accurate...

class NgcObject(object):
	pass

def get_ngc(ngcnum):
	for n in ngc2000:
		if n['id'] == ngcnum and n['is_ngc']:
			obj = NgcObject()
			obj.isngc = True
			obj.ngcnum = ngcnum
			obj.ra = n['ra']
			obj.dec = n['dec']
			obj.size = n['size']
			obj.constellation = n['constellation']
			obj.classification = n['classification']
			return obj

class IcObject(object):
	pass

def get_ic(icnum):
	for n in ngc2000:
		if n['id'] == icnum and not n['is_ngc']:
			obj = IcObject()
			obj.isngc = False
			obj.ngcnum = ngcnum
			obj.ra = n['ra']
			obj.dec = n['dec']
			obj.size = n['size']
			obj.constellation = n['constellation']
			obj.classification = n['classification']
			return obj
	
# This format is crazy...
ngc2000 = [
