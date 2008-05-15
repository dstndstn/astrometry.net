var tileSize = 128;

CustomGetTileUrl = function(pt, zoom) {
	var url = this.myBaseURL;
	url += 'ax=' + pt.x + '&ay=' + pt.y + '&b=' + zoom;
	url+='&w='+tileSize;
	url+='&h='+tileSize;
	return url;
}

// The GMap2
var map;

function startup() {
	// Create a new Google Maps client in the "map" <div>.
	map = new GMap2(document.getElementById("map"));

    var bgtile = new GTileLayer(new GCopyrightCollection(""), 1, 17);
	bgtile.getTileUrl = function(a,b) { return CONFIG_BGTILE_URL; };
	var maptype = new GMapType([bgtile], G_SATELLITE_MAP.getProjection(), "Custom Map B",
							{tileSize: tileSize});

	GEvent.bindDom(window, "resize", map, map.onResize);
	map.addControl(new GLargeMapControl());
	//map.addControl(new GMapTypeControl());
	map.setCenter(new GLatLng(0, 0), 1);

	// Clear the set of map types.
	map.getMapTypes().length = 0;
	map.addMapType(maptype);
    map.setMapType(maptype);

    var tile = new GTileLayer(new GCopyrightCollection(""), 1, 17);
	tile.myBaseURL = CONFIG_TILE_URL;
	// Use a custom tile server.
	tile.getTileUrl = CustomGetTileUrl;
	var overlaytile = new GTileLayerOverlay(tile);
	map.addOverlay(overlaytile);
	
}

