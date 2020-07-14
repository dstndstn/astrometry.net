import json

def json2python(j):
    try:
        return json.loads(j)
    except:
        pass
    return None

python2json = json.dumps

