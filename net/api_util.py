import json

def json2python(json):
    try:
        return json.loads(json)
    except:
        pass
    return None

python2json = json.dumps

