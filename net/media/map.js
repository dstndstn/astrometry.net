/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang and Keir Mierle.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

/*
  This function was borrowed from the file "wms236.js" by John Deck,
  UC Berkeley, which interfaces the Google Maps client to a WMS
  (Web Mapping Service) server.
 */
CustomGetTileUrl = function(a,b,c) {
	var lULP = new GPoint(a.x*256,(a.y+1)*256);
	var lLRP = new GPoint((a.x+1)*256,a.y*256);
	var lUL = G_NORMAL_MAP.getProjection().fromPixelToLatLng(lULP,b,c);
	var lLR = G_NORMAL_MAP.getProjection().fromPixelToLatLng(lLRP,b,c);
	var lBbox=lUL.x+","+lUL.y+","+lLR.x+","+lLR.y;
	var lURL=this.myBaseURL;
	lURL+="&layers=" + this.myLayers;
	if (this.jpeg)
		lURL += "&jpeg";
	lURL+="&bb="+lBbox;
	lURL+="&w=256";
	lURL+="&h=256";
	return lURL;
}

/*
  Pulls entries out of the GET line and puts them in an array.
  ie, if the current URL is:
  .  http://wherever.com/map.html?a=b&x=4&z=9
  then this will return:
  .  ["a"] = "b"
  .  ["x"] = "4"
  .  ["z"] = "9"
*/
function getGetData(){
	GET_DATA=new Object();
	var myurl=new String(window.location);
	// Strip off trailing "#"
	if (myurl.charAt(myurl.length-1) == '#') {
		myurl = myurl.substr(0, myurl.length-1);
	}
	var questionMarkLocation=myurl.search(/\?/);
	if (questionMarkLocation!=-1){
		myurl=myurl.substr(questionMarkLocation+1);
		var getDataArray=myurl.split(/&/g);
		for (var i=0;i<getDataArray.length;i++){
			var nameValuePair=getDataArray[i].split(/=/);
			GET_DATA[unescape(nameValuePair[0])]=unescape(nameValuePair[1]);
		}
	}
	return GET_DATA;
}

var dodebug = false;

/*
  Prints text to the debug form.
*/
function debug(txt) {
	if (dodebug) {
		GLog.write(txt);
	}
}

// The GMap2
var map;

// URLs of tileserver.  These are defined in the HTML (map.php)
var TILE_URL  = CONFIG_TILE_URL;
var IMAGE_URL  = CONFIG_IMAGE_URL;
var IMAGE_LIST_URL  = CONFIG_IMAGE_LIST_URL;
var BLACK_URL = CONFIG_BLACK_URL;

// The arguments in the HTTP request
var getdata;

// args that we pass on.
var passargs = [ 'imagefn', 'wcsfn', 'cc', 'arcsinh', 'arith', //'gain',
    'dashbox', 'cmap',
    'rdlsfn', 'rdlsfield', 'rdlsstyle',
    'rdlsfn2', 'rdlsfield2', 'rdlsstyle2',
				 'density',
				 'submission',
    ];

var gotoform = document.getElementById("gotoform");

// List of "images in this view" names.
var visImages = [];
// List of "images in this view" bounding boxes.
var visBoxes  = [];

var selectedIndex = -1;
var selectedImage = "";
var selectedPoly;

function ra2long(ra) {
	return 360 - ra;
}

function long2ra(lng) {
	ra = -lng;
	if (ra < 0.0) { ra += 360.0; }
	if (ra > 360.0) { ra -= 360.0; }
	return ra;
}

/*
  This function gets called as the user moves the map.
*/
function mapmoved() {
	center = map.getCenter();
	// update the center ra,dec textboxes.
	ra = long2ra(center.lng());
	gotoform.ra_center.value  = "" + ra;
	gotoform.dec_center.value = "" + center.lat();
}

/*
  This function gets called when the user changes the zoom level.
*/
function mapzoomed(oldzoom, newzoom) {
	// update the "zoom" textbox.
	gotoform.zoomlevel.value = "" + newzoom;
	mapmoved();
	selectedIndex = -1;
}

function colorimagelinks(latlng) {
	inset = [];

	var lat = latlng.lat();
	var lng = latlng.lng();
	if (lng < 0)
		lng += 360;

	debug("lat,lng " + lat + ", " + lng);

	for (var i=0; i<visBoxes.length; i++) {
		var link = document.getElementById('imagename-' + visImages[i]);
		if (visImages[i] == selectedImage) {
			link.style.color = '#00FF88';
		} else if (inPoly(visBoxes[i], lng, lat)) {
			inset.push(visImages[i]);
			link.style.color = '#FF0000';
		} else {
			link.style.color = '#666';
		}
	}
	debug("In polies: [ " + inset.join(", ") + " ]");
}

function mouseclicked(overlay, latlng) {
	if (selectedPoly)
		map.removeOverlay(selectedPoly);

	if (!latlng) {
		return;
	}
	var lat = latlng.lat();
	var lng = latlng.lng();
	if (lng < 0)
		lng += 360;

	debug("lat,lng " + lat + ", " + lng);

	for (var off=0; off<visBoxes.length; off++) {
		var i = (selectedIndex - (1 + off) + visBoxes.length) % visBoxes.length;

		if (inPoly(visBoxes[i], lng, lat)) {
			selectedImage = visImages[i];
			selectedIndex = i;

			poly = visBoxes[i];
			gpoly = [];
			for (var j=0; j<poly.length/2; j++) {
				gpoly.push(new GLatLng(poly[j*2+1], poly[j*2]));
			}
			selectedPoly = new GPolyline(gpoly, "#00FF88", 2, 0.8);

			break;
		}
	}
	if (selectedPoly)
		map.addOverlay(selectedPoly);
	colorimagelinks(latlng);
}

var mouseLatLng;

/*
  This function gets called when the mouse is moved.
*/
function mousemoved(latlng) {
	mouseLatLng = latlng;

	var ra = long2ra(latlng.lng());
	gotoform.ra_mouse.value  = "" + ra;
	gotoform.dec_mouse.value = "" + latlng.lat();

	colorimagelinks(latlng);
}

/*
  This function gets called when the "Go" button is pressed, or when Enter
  is hit in one of the ra/dec/zoom textfields.
*/
function moveCenter() {
	var ra   = gotoform.ra_center.value;
	var dec  = gotoform.dec_center.value;
	var zoom = gotoform.zoomlevel.value;
	//debug("Moving map to (" + ra + ", " + dec + ") zoom " + zoom + ", old zoom " + map.getZoom() + "\n");
	map.setCenter(new GLatLng(dec, ra2long(ra)), Number(zoom));
}

/*
  Create a URL with "ra", "dec", and "zoom" GET args, and go there.
*/
function linktohere() {
	var url=new String(window.location);
	if (url.charAt(url.length - 1) == '#')
		url = url.substr(0, url.length - 1);
	var qm = url.search(/\?/);
	if (qm!=-1) {
		url = url.substr(0, qm);
	}
	center = map.getCenter();
	url += "?ra=" + long2ra(center.lng()) + "&dec=" + center.lat()
		+ "&zoom=" + map.getZoom();

	var show = [];
	if (tychoShowing)
		show.push("tycho");
	if (usnobShowing)
		show.push("usnob");
	if (imagesShowing)
		show.push("images");
	if (imageOutlinesShowing)
		show.push("imageOutlines");
	if (gridShowing)
		show.push("grid");
	if (messierShowing)
		show.push("messier");
	if (constellationShowing)
		show.push("constellation");
	if (userImageShowing)
		show.push('userImage');
	if (userOutlineShowing)
		show.push('userOutline');
	if (userRdlsShowing)
		show.push('userRdls');
	url += "&show=" + show.join(",");

	if ('userimage' in getdata) {
		url += "&userimage=" + getdata['userimage'];
	}

	if (selectedImages.length) {
		url += "&selectedImages=" + selectedImages.join(',');
	}

	for (var i=0; i<passargs.length; i++) {
		if (passargs[i] in getdata) {
			url += "&" + passargs[i];
			if (getdata[passargs[i]] != undefined) {
				url += "=" + getdata[passargs[i]];
			}
		}
	}
	window.location = url;
}

function buttonStyleOn(button) {
	button.style.borderTopColor    = "#b0b0b0";
	button.style.borderLeftColor   = "#b0b0b0";
	button.style.borderBottomColor = "white";
	button.style.borderRightColor  = "white";
	button.style.fontWeight = "bold";
}

function buttonStyleOff(button) {
	button.style.borderTopColor    = "white";
	button.style.borderLeftColor   = "white";
	button.style.borderBottomColor = "#b0b0b0";
	button.style.borderRightColor  = "#b0b0b0";
	button.style.fontWeight = "normal";
}

function buttonStyleCommon(button) {
	button.style.color = "white";
	button.style.backgroundColor = "#000000";
	button.style.fontSize = "x-small";
	button.style.fontFamily = "monospace";
	button.style.borderStyle = "solid";
	button.style.borderWidth = "1px";
	button.style.marginBottom = "1px";
	button.style.marginTop = "1px";
	button.style.textAlign = "center";
	button.style.cursor = "pointer";
	button.style.width = "70px";
	buttonStyleOn(button);
}

var lineOverlay;
var tychoOverlay;
var usnobOverlay;
var imagesOverlay;
var userImageOverlay;
var selectedImageOverlay;

var gridShowing = 0;
var messierShowing = 0;
var constellationShowing = 0;
var tychoShowing = 0;
var usnobShowing = 0;
var imagesShowing = 0;
var imageOutlinesShowing = 0;
var userImageShowing = 0;
var userOutlineShowing = 0;
var userRdlsShowing = 0;
var selectedImageShowing = 0;

var selectedImages = [];

function toggleButton(overlayName) {
	button = document.getElementById(overlayName+"ToggleButton");
	if (eval(overlayName+"Showing")) {
		eval(overlayName+"Showing = 0");
		button.style.color = "#666";
	} else {
		eval(overlayName+"Showing = 1");
		button.style.color = "white";
	}
}

function makeOverlay(layers, tag) {
	var newTile = new GTileLayer(new GCopyrightCollection(""), 1, 17);
	newTile.myLayers=layers;
	newTile.myBaseURL=TILE_URL + tag;
	newTile.getTileUrl=CustomGetTileUrl;
	return new GTileLayerOverlay(newTile);
}

function makeMapType(tiles, label) {
	return new GMapType(tiles, G_SATELLITE_MAP.getProjection(), label, G_SATELLITE_MAP);
}

function getBlackUrl(tile, zoom) {
	return BLACK_URL;
}

var tychoGain = 0; // must match HTML
var tychoArcsinh = 1; // must match HTML

var usnobGain = 0; // must match HTML
var usnobArcsinh = 1; // must match HTML

var imagesGain = 0; // must match HTML

function min(a, b) {
	if (a < b) return a;
	return b;
}

var oldOverlays = [];

function restackOverlays() {
	newOverlays = [];
	if (tychoShowing)
		newOverlays.push(tychoOverlay);
	if (usnobShowing)
		newOverlays.push(usnobOverlay);
	if (imagesShowing || imageOutlinesShowing)
		newOverlays.push(imagesOverlay);
	if (userImageShowing || userOutlineShowing || userRdlsShowing)
		newOverlays.push(userImageOverlay);
	if (gridShowing || messierShowing || constellationShowing)
		newOverlays.push(lineOverlay);
	if (selectedImageShowing)
		newOverlays.push(selectedImageOverlay);

	// How many layers stay the same?
	var nsame;
	for (nsame=0; nsame<min(newOverlays.length, oldOverlays.length); nsame++) {
		if (newOverlays[nsame] != oldOverlays[nsame])
			break;
	}
	debug('Old and new arrays are the same up to layer ' + nsame + ', old has ' + oldOverlays.length + ', new has ' + newOverlays.length);

	// Remove old layers...
	for (var i=nsame; i<oldOverlays.length; i++) {
		map.removeOverlay(oldOverlays[i]);
	}

	// Add new layers...
	for (var i=nsame; i<newOverlays.length; i++) {
		map.addOverlay(newOverlays[i]);
	}
	oldOverlays = newOverlays;
}

function toggleOverlayRestack(overlayName) {
	toggleButton(overlayName);
	restackOverlays();
}

function toggleLineOverlay(overlayName) {
	toggleButton(overlayName);
	updateLine();
	restackOverlays();
}

function toggleImageOutlines() {
	toggleButton("imageOutlines");
	updateImages();
	restackOverlays();
}

function toggleImages() {
	toggleButton("images");
	updateImages();
	restackOverlays();
}

function toggleUserOutline() {
	toggleButton("userOutline");
	updateUserImage();
	restackOverlays();
}

function toggleUserRdls() {
	toggleButton("userRdls");
	updateUserImage();
	restackOverlays();
}

function toggleUserImage() {
	toggleButton("userImage");
	updateUserImage();
	restackOverlays();
}

function updateLine() {
	var tag = "&tag=line";
	var layers = [];
	if (constellationShowing)
		layers.push('constellation');
	if (messierShowing)
		layers.push('messier');
	if (gridShowing)
		layers.push('grid');
	layerstr = layers.join(",");
	lineOverlay = makeOverlay(layerstr, tag);
}

function updateImages() {
	var tag = "&tag=images";
	tag += "&gain=" + imagesGain;
	var lay = [];
	if (imagesShowing)
		lay.push('images');
	if (imageOutlinesShowing)
		lay.push('boundaries');
	imagesOverlay = makeOverlay(lay.join(","), tag);
}

function updateUserImage() {
	var jobid = getdata['userimage'];
	var image = jobid + '/fullsize.png';
	var wcs = jobid + '/wcs.fits';
	var rdls = jobid + '/field.rd.fits';
	var tag = "&imagefn=" + image + "&wcsfn=" + wcs + "&rdlsfn=" + rdls;
	var lay = [];
	if (userImageShowing)
		lay.push('userimage');
	if (userOutlineShowing)
		lay.push('userboundary');
	if (userRdlsShowing)
		lay.push('rdls');
	userImageOverlay = makeOverlay(lay.join(","), tag);
}

function updateTycho() {
	var tag = "&tag=tycho";
	tag += "&gain=" + tychoGain;
	if (tychoArcsinh) {
		tag += "&arcsinh";
	}
	tychoOverlay = makeOverlay('tycho', tag);
}

function updateUsnob() {
	var tag = "&tag=usnob";
	tag += "&gain=" + usnobGain;
	tag += "&cmap=rb";
	if (usnobArcsinh) {
		tag += "&arcsinh";
	}
	usnobOverlay = makeOverlay('usnob', tag);
}

function updateSelectedImage() {
	var tag = "&ubstyle=y";
	tag += "&wcsfn=" + selectedImages.join('.wcs,') + '.wcs';
	selectedImageOverlay = makeOverlay('userboundary', tag);
}

function indexOf(arr, element) {
	ind = -1;
	for (var i=0; i<arr.length; i++) {
		if (arr[i] == element) {
			ind = i;
			break;
		}
	}
	return ind;
}

function mymap(f, arr) {
	var res = [];
	for (var i=0; i<arr.length; i++) {
		res.push(f(arr[i]));
		//for (var x in arr) {
		//res.push(f(x));
	}
	return res;
}

function toggleSelectedImage(img) {
	debug('Toggling ' + img);
	debug('Selected images: [' + selectedImages.join(', ') + ']');
	ind = indexOf(selectedImages, img);
	//debug('Ind ' + ind);
	if (ind == -1) {
		selectedImages.push(img);
	} else {
		selectedImages.splice(ind, 1);
	}
	debug('After: [' + selectedImages.join(', ') + ']');
	//debug('Len ' + selectedImages.length);
	if (selectedImages.length > 0) {
		selectedImageShowing = 1;
		updateSelectedImage();
	} else {
		selectedImageShowing = 0;
	}
	restackOverlays();
	showhide = document.getElementById('showhide' + img);
	removeAllChildren(showhide);
	if (ind == -1) {
		txt = '[outline]';
		color = "white";
	} else {
		txt = '[outline]';
		color = "#666";
	}
	debug('Txt ' + txt);
	txtnode = document.createTextNode(txt);
	showhide.style.color = color;
	showhide.appendChild(txtnode);
}

function changeArcsinh() {
	tychoArcsinh = gotoform.arcsinh.checked;
	updateTycho();
	restackOverlays();
}

function changeGain() {
	var gain = Number(gotoform.gain.value);
	tychoGain = gain;
	updateTycho();
	usnobGain = gain;
	updateUsnob();
	imagesGain = gain;
	updateImages();
	restackOverlays();
}

function removeAllChildren(node) {
	while (node.childNodes.length) {
		node.removeChild(node.childNodes[0]);
	}
}

function emptyImageList() {
	imglist = document.getElementById('imagelist');
	removeAllChildren(imglist);
}

function imageListLoaded(txt) {
	debug('image list loaded.');
	emptyImageList();
	//debug("txt: " + txt);
	xml = GXml.parse(txt);
	debug("xml: " + xml);
	imgtags = xml.documentElement.getElementsByTagName("image");
	debug("Found " + imgtags.length + " images.");

	visImages = [];
	visBoxes = [];
	for (var i=0; i<imgtags.length; i++) {
		name = imgtags[i].getAttribute('name');
		visImages.push(name);
		poly = mymap(parseFloat, imgtags[i].getAttribute('poly').split(','));
		visBoxes.push(poly);
		debug("Found " + poly.length + " polygon points.");
		debug("  " + poly.join(","));

	}
	debug('Selected images: [' + selectedImages.join(', ') + ']');
	debug('Visible images: [' + visImages.join(', ') + ']');

	// Remove selected images that are no longer visible.
	for (var i=0; i<selectedImages.length; i++) {
		ind = indexOf(visImages, selectedImages[i]);
		if (ind == -1) {
			selectedImages.splice(ind, 1);
			i--;
		}
	}

	for (var i=0; i<visImages.length; i++) {
		if (i) {
			imglist.appendChild(document.createElement("br"));
		}

		img = visImages[i];

		link2 = document.createElement("a");
		link2.setAttribute('href', '#');
		link2.setAttribute('onclick', 'toggleSelectedImage("' + img + '")');
		link2.setAttribute('id', 'showhide' + img);
		if (indexOf(selectedImages, img) > -1) {
			txt = '[outline]';
			color = "white";
		} else {
			txt = '[outline]';
			color = "#666";
		}
		txtnode = document.createTextNode(txt);
		link2.style.color = color;
		link2.appendChild(txtnode);
		imglist.appendChild(link2);

		imglist.appendChild(document.createTextNode(" "));

		link = document.createElement("a");
		link.setAttribute('href', IMAGE_URL + "?filename=" + img);
		link.setAttribute('id', 'imagename-' + img);
		link.appendChild(document.createTextNode(img));
		imglist.appendChild(link);
	}

	if (mouseLatLng)
		colorimagelinks(mouseLatLng);
}

function movestarted() {
	emptyImageList();
}

var lastListUrl = '';

/*
  This function gets called when the user stops moving the map (mouse drag),
  and also after it's moved programmatically (via setCenter(), etc).
*/
function moveended() {
	mapmoved();

	debug('moveended()');

	visBoxes = [];
	visImages = [];

	if (imagesShowing || imageOutlinesShowing) {
		url = IMAGE_LIST_URL + "?";
		bounds = map.getBounds();
		sw = bounds.getSouthWest();
		ne = bounds.getNorthEast();
		L = sw.lng();
		R = ne.lng();
		if (L > R) {
			if (L > 180) {
				L -= 360;
			} else {
				R += 360;
			}
		}
		url += "bb=" + L + "," + sw.lat() + "," + R + "," + ne.lat();
		if (url == lastListUrl) {
			debug('Not downloading identical image list (' + lastListUrl + ')');
		} else {
			debug("Downloading: " + url);
			lastListUrl = url;
			GDownloadUrl(url, imageListLoaded);
		}
	}
}

/*
  This function gets called when the page loads.
*/
function startup() {
	getdata = getGetData();

	// Create a new Google Maps client in the "map" <div>.
	map = new GMap2(document.getElementById("map"));

	var ra=0;
	var dec=0;
	var zoom=2;

	if ("ra" in getdata) {
		ra = Number(getdata["ra"]);
	}
	if ("dec" in getdata) {
		dec = Number(getdata["dec"]);
	}
	if ("zoom" in getdata) {
		zoom = Number(getdata["zoom"]);
	}
	map.setCenter(new GLatLng(dec, ra2long(ra)), zoom);

	if ('debug' in getdata) {
		dodebug = true;
	}

	// Add pass-thru args
	firstone = true;
	for (var i=0; i<passargs.length; i++) {
		if (passargs[i] in getdata) {
			if (!firstone)
				TILE_URL += "&";
			TILE_URL += passargs[i] + "=" + getdata[passargs[i]];
			firstone = false;
		}
	}

	// Handle user images.
	if ('userimage' in getdata) {
		var holder = document.getElementById("userImageToggleButtonHolder");
		var link = document.createElement("a");
		link.setAttribute('href', '#');
		link.setAttribute('onclick', 'toggleUserImage()');
		link.setAttribute('id', 'userImageToggleButton');
		var button = document.createTextNode("My Image");
		var bar = document.createTextNode(" | ");
		link.appendChild(button);
		holder.appendChild(link);
		holder.appendChild(bar);

		var link2 = document.createElement("a");
		link2.setAttribute('href', '#');
		link2.setAttribute('onclick', 'toggleUserOutline()');
		link2.setAttribute('id', 'userOutlineToggleButton');
		var button2 = document.createTextNode("My Image Outline");
		var bar2 = document.createTextNode(" | ");
		link2.appendChild(button2);
		holder.appendChild(link2);
		holder.appendChild(bar2);

		var link3 = document.createElement("a");
		link3.setAttribute('href', '#');
		link3.setAttribute('onclick', 'toggleUserRdls()');
		link3.setAttribute('id', 'userRdlsToggleButton');
		var button3 = document.createTextNode("My Image Sources");
		var bar3 = document.createTextNode(" | ");
		link3.appendChild(button3);
		holder.appendChild(link3);
		holder.appendChild(bar3);
	}

	if ('selectedImages' in getdata) {
		selectedImages = getdata['selectedImages'].split(',');
		if (selectedImages.length > 0) {
			selectedImageShowing = 1;
			updateSelectedImage();
		}
	}

	// Clear the set of map types.
	map.getMapTypes().length = 0;
	
	var blackTile = new GTileLayer(new GCopyrightCollection(""), 1, 17);
	blackTile.getTileUrl = getBlackUrl;
	var blackMapType = makeMapType([blackTile], "Map");
	map.addMapType(blackMapType);
    map.setMapType(blackMapType);

	updateTycho();
	updateUsnob();

	if ('gain' in getdata) {
		gotoform.gain.value = getdata['gain'];
		changeGain();
	}

	if ('show' in getdata) {
		var showstr = getdata['show'];
		var ss = showstr.split(',');
		var show = [];
		for (var i=0; i<ss.length; i++)
			show[ss[i]] = 1;

		var layers = [ 'tycho', 'usnob', 'images', 'imageOutlines', 'grid', 'constellation', 'messier', 'userImage', 'userOutline', 'userRdls' ];
		for (var i=0; i<layers.length; i++)
			if (layers[i] in show)
				toggleButton(layers[i]);
	} else {
		toggleButton('tycho');
		toggleButton('images');
		if ('userimage' in getdata) {
			toggleButton('userImage');
			toggleButton('userOutline');
		}
	}
	updateLine();
	updateUserImage();
	updateImages();
	restackOverlays();

	// Connect up the event listeners...
	GEvent.addListener(map, "move", mapmoved);
	GEvent.addListener(map, "moveend", moveended);
	GEvent.addListener(map, "movestart", movestarted);
	GEvent.addListener(map, "zoomend", mapzoomed);
	GEvent.addListener(map, "mousemove", mousemoved);
	GEvent.addListener(map, "click", mouseclicked);
	GEvent.bindDom(window, "resize", map, map.onResize);

	map.addControl(new GLargeMapControl());

	moveended();
	mapzoomed(map.getZoom(), map.getZoom());
}

/* inPoly: borrowed from http://www.scottandrew.com/weblog/jsjunk#inpoly
   copyright 2001 scott andrew lepera, damn you!
*/
/* inPoly()
Finds if a given point is within a polygon.

Based on Bob Stein's inpoly() function for C.
http://home.earthlink.net/~bobstein/inpoly/

Modified for JavaScript by Scott Andrew LePera.

Parameters:
poly: array containing x/y coordinate pairs that
  describe the vertices of the polygon. Format is
  indentical to that of HTML image maps, i.e. [x1,y1,x2,y2,...]
  
px: the x-coordinate of the target point.

py: the y-coordinate of the target point.

Return value:
true if the point is within the polygon, false if not.
*/
function inPoly(poly, px, py) {
	var npoints = poly.length; // number of points in polygon
	var xnew,ynew,xold,yold,x1,y1,x2,y2,i;
	var inside=false;

	if (npoints/2 < 3) { // points don't describe a polygon
		return false;
	}
	xold=poly[npoints-2];
	yold=poly[npoints-1];
     
	for (i=0 ; i < npoints ; i=i+2) {
		xnew=poly[i];
		ynew=poly[i+1];
		if (xnew > xold) {
			x1=xold;
			x2=xnew;
			y1=yold;
			y2=ynew;
		} else {
			x1=xnew;
			x2=xold;
			y1=ynew;
			y2=yold;
		}
		if ((xnew < px) == (px <= xold) && ((py-y1)*(x2-x1) < (y2-y1)*(px-x1))) {
			inside=!inside;
		}
		xold=xnew;
		yold=ynew;
	}
	return inside;
}

