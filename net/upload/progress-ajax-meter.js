var pm_req{{name}};
var pm_reload{{name}} = true;
var pm_period{{name}} = 2000;
var pm_failed{{name}} = false;
var pm_succeeded{{name}} = false;
var pm_id{{name}} = '';

function stopProgressMeter{{name}}() {
	pm_reload{{name}} = false;
}

function startProgressMeter{{name}}(id) {
	//alert('StartProgressMeter(' + id + ')');
	pm_id{{name}} = id;
	pm_reload{{name}} = true;
	pm_failed{{name}} = false;
	pm_succeeded{{name}} = false;
	pm_sendRequest{{name}}();
}

function pm_debug{{name}}(str) {
	txt = document.getElementById('meter_text{{name}}');
	while (txt.childNodes.length) {
		txt.removeChild(txt.childNodes[0]);
	}
	txt.appendChild(document.createTextNode(str));
}

function pm_sendRequest{{name}}() {
	pm_req{{name}} = new XMLHttpRequest();
	pm_req{{name}}.onreadystatechange = pm_contentReady{{name}};
	var url = '{{ xmlurl }}' + pm_id{{name}}
	//alert('Open ' + url);
	pm_req{{name}}.open('GET', url, true);
	pm_req{{name}}.setRequestHeader("Content-Type", "text/plain;charset=UTF-8");
	//alert('send()');
	pm_req{{name}}.send('');
	//pm_req{{name}}.send();
}

function pm_processIt{{name}}() {
	//alert('Processing response.');
	if (!pm_req{{name}}) {
		//alert('no req');
		return false;
	}
	//alert('req state: ' + pm_req{{name}}.readyState);
	if (pm_req{{name}}.readyState != 4) {
		if (pm_req{{name}}.readyState == 1) {
			//alert('send()');
			//pm_req{{name}}.send();
			//pm_req{{name}}.send('');
		}
		if (pm_req{{name}}.readyState == 2) {
			//alert('req stat 2, status: ' + pm_req{{name}}.status + ', ' + pm_req{{name}}.statusText);
		} else {
			//alert('req state: ' + pm_req{{name}}.readyState);
		}
		return false;
	}
	if (pm_req{{name}}.status != 200) {
		//alert('req status ' + pm_req{{name}}.status);
		pm_debug{{name}}('req status ' + pm_req{{name}}.status);
		return true;
	}
	xml = pm_req{{name}}.responseXML;
	if (!xml) {
		//alert('not xml');
		pm_debug{{name}}('not xml');
		return true;
	}
	//alert('got xml');
	prog = xml.getElementsByTagName('progress');
	if (!prog.length) {
		//alert('no progress');
		pm_debug{{name}}('no progress');
		return true;
	}

	// Did the upload finish with an error?
	err = prog[0].getAttribute('error');
	if (err) {
		txt = document.getElementById('meter_text{{name}}');
		while (txt.childNodes.length) {
			txt.removeChild(txt.childNodes[0]);
		}
		txt.appendChild(document.createTextNode(err));
		fore = document.getElementById('meter_fore{{name}}');
		fore.style.width = '0px';
		back = document.getElementById('meter_back{{name}}');
		back.style.background = 'pink';
		pm_reload{{name}} = false;
		pm_failed{{name}} = true;
		//alert('error ' + err);
		return true;
	}

	pct = prog[0].getAttribute('pct');
	txt = document.getElementById('meter_text{{name}}');
	while (txt.childNodes.length) {
		txt.removeChild(txt.childNodes[0]);
	}
	txt.appendChild(document.createTextNode('' + pct + '%'));
	fore = document.getElementById('meter_fore{{name}}');
	fore.style.width = '' + pct + '%';
	if (pct == 100) {
		pm_reload{{name}} = false;
		pm_succeeded{{name}} = false;
	}
	//alert('pct ' + pct);
	return true;
}

function pm_contentReady{{name}}() {
	if (pm_processIt{{name}}()) {
		if (pm_reload{{name}}) {
			// do it again!
			setTimeout('pm_sendRequest{{name}}()', pm_period{{name}});
		}
	}
}

