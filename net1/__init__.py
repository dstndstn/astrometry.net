
# see "astrometry.net.media" in urls.py
def media(req, filename=None):
	import astrometry.net1.portal.views as views
	return views.media(req, filename)

# likewise "logout"
def logout():
    pass

def login():
    pass

def changepassword():
    pass

def changedpassword():
    pass

def resetpassword():
    pass

def setpassword():
    pass

def newaccount():
    pass
