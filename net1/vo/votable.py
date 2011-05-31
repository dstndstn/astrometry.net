from an.vo.log import log


class xmlelement(object):
    etype = None
    args = {}
    children = []

    def __init__(self, etype=None):
        self.children = []
        self.args = {}
        self.etype = etype

    def __str__(self):
        #log('str(): %s' % self.etype)
        s = ('<%s' % str(self.etype))
        for k,v in self.args.items():
            s += (' %s="%s"' % (k, v))
        children = self.get_children()
        if len(children):
            s += '>' + self.self_child_separator()
            #log('%i children.' % len(self.children))
            for c in children:
                #log('%s child:' % self.etype)
                s += self.get_child_tag(c)
            s += ('</%s>' % str(self.etype))
        else:
            s += ' />'
        return s
    
    def self_child_separator(self):
        return '\n'

    def child_child_separator(self):
        return '\n'
    
    def get_child_tag(self, child):
        return str(child) + self.child_child_separator()

    def get_children(self):
        return self.children

    def add_child(self, x):
        #log('adding child to %s' % self.etype)
        self.children.append(x)

    def add_children(self, x):
        for c in x:
            self.add_child(c)
        #self.children += x

class VOTableDocument(xmlelement):
    xml_header = r'<?xml version="1.0"?>'

    def __init__(self):
        super(VOTableDocument, self).__init__('VOTABLE')
        self.args.update({
            'version' : '1.1',
            'xmlns:xsi' : 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:noNamespaceSchemaLocation' : 'http://www.ivoa.net/xml/VOTable/VOTable/v1.1',
            })

    def __str__(self):
        s = self.xml_header + '\n' + super(VOTableDocument, self).__str__()
        return s

class VOResource(xmlelement):
    def __init__(self, name=None):
        super(VOResource, self).__init__('RESOURCE')
        if name:
            self.args['name'] = name

class VOTable(xmlelement):
    fields = []
    params = []
    rows = []

    def __init__(self, name=None):
        super(VOTable, self).__init__('TABLE')
        if name:
            self.args['name'] = name
        self.fields = []
        self.rows = []

    def get_children(self):
        data = xmlelement('DATA')
        tabledata = xmlelement('TABLEDATA')
        tabledata.children = self.rows
        data.add_child(tabledata)
        return self.fields + self.params + [data,]

    def add_param(self, f):
        self.params.append(f)

    def add_field(self, f):
        self.fields.append(f)

    def add_row(self, f):
        self.rows.append(f)

class VOField(xmlelement):
    def __init__(self, name, datatype, arraysize=None, ucd=None):
        super(VOField, self).__init__('FIELD')
        self.args['name'] = name
        self.args['datatype'] = datatype
        if arraysize:
            self.args['arraysize'] = arraysize
        if ucd:
            self.args['ucd'] = ucd

class VOParam(xmlelement):
    def __init__(self, name, datatype, arraysize=None, ucd=None, value=None):
        super(VOParam, self).__init__('PARAM')
        self.args['name'] = name
        if datatype:
            self.args['datatype'] = datatype
        if arraysize:
            self.args['arraysize'] = arraysize
        if ucd:
            self.args['ucd'] = ucd
        if value:
            self.args['value'] = value

class VOInfo(xmlelement):
    def __init__(self, name=None):
        super(VOInfo, self).__init__('INFO')
        if name:
            self.args['name'] = name

class VORow(xmlelement):
    cols = []
    def __init__(self, cols=None):
        super(VORow, self).__init__('TR')
        if cols:
            self.cols = cols

    def add_data(self, x):
        col = VOColumn(x)
        self.cols.append(col)

    def get_children(self):
        return cols

    #def get_child_tag(self, child):
    #    return str(child)
    #def child_child_separator(self):
    #    return '\n'
    #def self_child_separator(self):
    #    return '\n'

class VOColumn(xmlelement):
    data = None

    def __init__(self, data=None):
        super(VOColumn, self).__init__('TD')
        if data:
            self.data = data

    def __str__(self):
        self.children = [ self.data ]
        return super(VOColumn, self).__str__()

    def get_child_tag(self, child):
        raw = self.get_child_tag_raw(child)
        repls = {
            '<' : '&lt;',
            '>' : '&gt;',
            '&' : '&amp;',
            '\'' : '&apos;',
            '"' : '&quot;',
            }
        for k,v in repls.items():
            raw = raw.replace(k, v)
        return raw

    def get_child_tag_raw(self, child):
        if isinstance(child, list):
            return ' '.join(map(str, child))
        if child is None:
            return ''
        return str(child)
                
    def self_child_separator(self):
        return ''


class VODescription(VOColumn):
    def __init__(self, data=None):
        super(VODescription, self).__init__(data)
        self.etype = 'DESCRIPTION'

class VOValues(xmlelement):
    def __init__(self, vals=None):
        super(VOValues, self).__init__('VALUES')
        if vals:
            self.add_children(vals)

    def get_child_tag(self, child):
        return '<OPTION>' + child + '</OPTION>'
    
