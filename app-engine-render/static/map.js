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

var tilesize = 128;

CustomGetTileUrl = function(a,b) {
	var lULP = new GPoint(a.x*tilesize,(a.y+1)*tilesize);
	var lLRP = new GPoint((a.x+1)*tilesize,a.y*tilesize);
	//var lUL = G_NORMAL_MAP.getProjection().fromPixelToLatLng(lULP,b,c);
	//var lLR = G_NORMAL_MAP.getProjection().fromPixelToLatLng(lLRP,b,c);
	//var lBbox=lUL.x+","+lUL.y+","+lLR.x+","+lLR.y;
	var lURL=this.myBaseURL;
	lURL+="&layers=" + this.myLayers;
	//if (this.jpeg)
	//	lURL += "&jpeg";
	lURL += '&ax=' + a.x + '&ay=' + a.y + '&b=' + b;
	//lURL+="&bb="+lBbox;
	lURL+='&w=' + tilesize;
	lURL+='&h=' + tilesize;
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

// The arguments in the HTTP request
var getdata;

// args that we pass on.
var passargs = [ ];

var gotoform = document.getElementById("gotoform");

function long2x(lng) {
    x = lng / 360.0;
    if (x < 0) {
        x += 1.0;
    }
    return x;
}

function x2long(x) {
    return x * 360.0;
}

function asinh(x) {
    return Math.log(x + Math.sqrt(1. + x*x));
}

function sinh(x) {
    return (Math.exp(x) - Math.exp(-x)) / 2.0;
}

function lat2y(lat) {
    return 0.5 + (asinh(Math.tan(lat * Math.PI / 180.0)) / (2.0 * Math.PI));
}

function y2lat(y) {
	return Math.atan(sinh((y - 0.5) * (2.0 * Math.PI))) * (180.0 / Math.PI);
}

/*
  This function gets called as the user moves the map.
*/
function mapmoved() {
	center = map.getCenter();
	// update the center x,y textboxes.
    
	var x = long2x(center.lng());
    var y = lat2y(center.lat());
	gotoform.x_center.value = "" + x;
	gotoform.y_center.value = "" + y;
}

/*
  This function gets called when the user changes the zoom level.
*/
function mapzoomed(oldzoom, newzoom) {
	// update the "zoom" textbox.
	gotoform.zoomlevel.value = "" + newzoom;
	mapmoved();
}

function mouseclicked(overlay, latlng) {
}

var mouseLatLng;

/*
  This function gets called when the mouse is moved.
*/
function mousemoved(latlng) {
	mouseLatLng = latlng;

	var x = long2x(latlng.lng());
    var y = lat2y (latlng.lat());
	gotoform.x_mouse.value = "" + x;
	gotoform.y_mouse.value = "" + y;
}

/*
  This function gets called when the "Go" button is pressed, or when Enter
  is hit in one of the ra/dec/zoom textfields.
*/
function moveCenter() {
    var x = gotoform.x_center.value;
    var y = gotoform.y_center.value;
	var zoom = gotoform.zoomlevel.value;
    var lat = x2long(x);
    var lng = y2lat(y);
	map.setCenter(new GLatLng(lat, lng), Number(zoom));
}

/*
 Create a URL with "x", "y", and "zoom" GET args, and go there.
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
	url += "?x=" + long2x(center.lng()) + "&y=" + lat2y(center.lat())
		+ "&zoom=" + map.getZoom();

    url += '&seedx=' + seedx + '&seedy=' + seedy;

	var show = [];
	if (imageShowing)
		show.push("image");
	if (gridShowing)
		show.push("grid");
	url += "&show=" + show.join(",");

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

var imageShowing = 0;
var gridShowing = 0;

var imageOverlay;
var gridOverlay;

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
	return new GMapType(tiles, G_SATELLITE_MAP.getProjection(), label,
				{tileSize: tilesize});
}

var seedx = 0;
var seedy = 0;

function min(a, b) {
	if (a < b) return a;
	return b;
}

var oldOverlays = [];

function restackOverlays() {
	newOverlays = [];
    if (imageShowing)
		newOverlays.push(imageOverlay);
    if (gridShowing)
		newOverlays.push(gridOverlay);

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

function toggleImage() {
	toggleButton("image");
	updateImage();
	restackOverlays();
}

function toggleGrid() {
	toggleButton("grid");
	updateGrid();
	restackOverlays();
}

/*
var otherMapType;
var imageLayer2;
var gridLayer2;
var image2Showing = 0;
var grid2Showing = 0;

function updateMap2() {
	map.removeMapType(otherMapType);
	layers = []
	if (image2Showing) {
	layers.push(imageLayer2);
    }
	if (grid2Showing) {
	layers.push(gridLayer2);
	}
	otherMapType = new GMapType(layers,
			G_SATELLITE_MAP.getProjection(), "Map2",
			{tileSize: tilesize});
	map.addMapType(otherMapType);
	map.setMapType(otherMapType);
}

function toggleImage2() {
	toggleButton("image2");
	updateMap2();
}

function toggleGrid2() {
	toggleButton("grid2");
	updateMap2();
}

|
<a id="grid2ToggleButton" href="#" onclick="toggleGrid2()"> Grid2</a> | 
<a id="image2ToggleButton" href="#" onclick="toggleImage2()">Image2 </a>

*/

function updateGrid() {
	//var tag = "&tag=grid";
	//var tag = "&rotation=" + rotation;
    var tag = '';
	var layers = [];
	if (gridShowing)
		layers.push('grid');
	gridOverlay = makeOverlay(layers.join(','), tag);
}

function updateImage() {
	//var tag = "&tag=image";
	var tag = "&seedx=" + seedx + "&seedy=" + seedy;
	var lay = [];
	if (imageShowing)
		lay.push('image');
	imageOverlay = makeOverlay(lay.join(','), tag);
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
	}
	return res;
}

function changeSeed() {
	seedx = Number(gotoform.seedx.value);
	seedy = Number(gotoform.seedy.value);
	updateImage();
	updateGrid();
	restackOverlays();
}

function setSeed(x, y) {
    //seedx = x;
    //seedy = y;
    gotoform.seedx.value = "" + x;
    gotoform.seedy.value = "" + y;
    changeSeed();
}

function removeAllChildren(node) {
	while (node.childNodes.length) {
		node.removeChild(node.childNodes[0]);
	}
}

function movestarted() {
}

/*
  This function gets called when the user stops moving the map (mouse drag),
  and also after it's moved programmatically (via setCenter(), etc).
*/
function moveended() {
	mapmoved();
}

function getBlackUrl(tile, zoom) {
    return "black.png";
}

/*
  This function gets called when the page loads.
*/
function startup() {
	getdata = getGetData();

	// Create a new Google Maps client in the "map" <div>.
	map = new GMap2(document.getElementById("map"));

	var x=0;
	var y=0;
	var zoom=2;

	if ("x" in getdata) {
		x = Number(getdata["x"]);
	}
	if ("y" in getdata) {
		y = Number(getdata["y"]);
	}
	if ("zoom" in getdata) {
		zoom = Number(getdata["zoom"]);
	}
	map.setCenter(new GLatLng(y2lat(y), x2long(x)), zoom);

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

	// Clear the set of map types.
	map.getMapTypes().length = 0;
	
    var blackTile = new GTileLayer(new GCopyrightCollection(""), 1, 17);
    blackTile.getTileUrl = getBlackUrl;
    var blackMapType = makeMapType([blackTile], "Map");

    map.addMapType(blackMapType);
    map.setMapType(blackMapType);

    setSeed(-0.726895347709114071439,
            0.188887129043845954792);

	if ('seedx' in getdata) {
		gotoform.seedx.value = getdata['seedx'];
	}
	if ('seedy' in getdata) {
		gotoform.seedy.value = getdata['seedy'];
	}
    changeSeed();

	if ('show' in getdata) {
		var showstr = getdata['show'];
		var ss = showstr.split(',');
		var show = [];
		for (var i=0; i<ss.length; i++)
			show[ss[i]] = 1;

		var layers = [ 'grid', 'image' ];
		for (var i=0; i<layers.length; i++)
			if (layers[i] in show)
				toggleButton(layers[i]);
	} else {
		toggleButton('image');
	}

	updateImage();
	updateGrid();

	restackOverlays();

	/*
	imageLayer2 = new GTileLayer(new GCopyrightCollection(""), 1, 17);
	imageLayer2.myBaseURL = TILE_URL;
	imageLayer2.getTileUrl = CustomGetTileUrl;
	gridLayer2 = new GTileLayer(new GCopyrightCollection(""), 1, 17);
	gridLayer2.myLayers = 'grid';
	gridLayer2.myBaseURL = TILE_URL;
	gridLayer2.getTileUrl = CustomGetTileUrl;
	otherMapType = new GMapType([imageLayer2, gridLayer2],
			G_SATELLITE_MAP.getProjection(), "Map2",
			{tileSize: tilesize});
	map.addMapType(otherMapType);
	*/

	// Connect up the event listeners...
	GEvent.addListener(map, "move", mapmoved);
	GEvent.addListener(map, "moveend", moveended);
	GEvent.addListener(map, "movestart", movestarted);
	GEvent.addListener(map, "zoomend", mapzoomed);
	GEvent.addListener(map, "mousemove", mousemoved);
	GEvent.addListener(map, "click", mouseclicked);
	GEvent.bindDom(window, "resize", map, map.onResize);

	map.addControl(new GLargeMapControl());
	map.addControl(new GMapTypeControl());

	moveended();
	mapzoomed(map.getZoom(), map.getZoom());
}

