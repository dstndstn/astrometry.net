def shell_escape(s):
    repl = ('\\', '|', '&', ';', '(', ')', '<', '>', ' ', '\t',
            '\n', '$', "'", '"', "`")
    # (note, \\ must be first!)
    for x in repl:
        s = s.replace(x, '\\'+x)
    return s

# escape a string that will appear inside double-quotes.
def shell_escape_inside_quotes(s):
    repl = ('\\', '\t', '`', '"', '$')
    # (note, \\ must be first!)
    for x in repl:
        s = s.replace(x, '\\'+x)
    return s
