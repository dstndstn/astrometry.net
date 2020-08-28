/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <string.h>
#include <libgen.h>

#include "fileutils.h"
#include "ioutils.h"
#include "os-features.h"

char* resolve_path(const char* filename, const char* basedir) {
    // we don't use canonicalize_file_name() because it requires the paths
    // to actually exist, while this function should work for output files
    // that don't already exist.
    char* path;
    char* rtn;
    // absolute path?
    if (filename[0] == '/')
        //return strdup(filename);
        return an_canonicalize_file_name(filename);
    asprintf_safe(&path, "%s/%s", basedir, filename);
    //return path;
    rtn = an_canonicalize_file_name(path);
    free(path);
    return rtn;
}

char* find_executable(const char* progname, const char* sibling) {
    char* sib;
    char* sibdir;
    char* path;
    char* pathenv;

    // If it's an absolute path, just return it.
    if (progname[0] == '/')
        return strdup(progname);

    // If it's a relative path, resolve it.
    if (strchr(progname, '/')) {
        path = an_canonicalize_file_name(progname);
        if (path && file_executable(path))
            return path;
        free(path);
    }

    // If "sibling" contains a "/", then check relative to it.
    if (sibling && strchr(sibling, '/')) {
        // dirname() overwrites its arguments, so make a copy...
        sib = strdup(sibling);
        sibdir = strdup(dirname(sib));
        free(sib);

        asprintf_safe(&path, "%s/%s", sibdir, progname);
        free(sibdir);

        if (file_executable(path))
            return path;

        free(path);
    }

    // Search PATH.
    pathenv = getenv("PATH");
    while (1) {
        char* colon;
        int len;
        if (!strlen(pathenv))
            break;
        colon = strchr(pathenv, ':');
        if (colon)
            len = colon - pathenv;
        else
            len = strlen(pathenv);
        if (pathenv[len - 1] == '/')
            len--;
        asprintf_safe(&path, "%.*s/%s", len, pathenv, progname);
        if (file_executable(path))
            return path;
        free(path);
        if (colon)
            pathenv = colon + 1;
        else
            break;
    }

    // Not found.
    return NULL;
}

char* an_canonicalize_file_name(const char* fn) {
    sl* dirs;
    int i;
    char* result;
    // Ugh, special cases.
    if (streq(fn, ".") || streq(fn, "/"))
        return strdup(fn);

    dirs = sl_split(NULL, fn, "/");
    for (i=0; i<sl_size(dirs); i++) {
        if (streq(sl_get(dirs, i), "")) {
            // don't remove '/' from beginning of path!
            if (i) {
                sl_remove(dirs, i);
                i--;
            }
        } else if (streq(sl_get(dirs, i), ".")) {
            sl_remove(dirs, i);
            i--;
        } else if (streq(sl_get(dirs, i), "..")) {
            // don't remove ".." at start of path.
            if (!i)
                continue;
            // don't remove chains of '../../../' at the start.
            if (streq(sl_get(dirs, i-1), ".."))
                continue;
            // but do collapse '/../' to '/' at the start.
            if (streq(sl_get(dirs, i-1), "")) {
                sl_remove(dirs, i);
                i--;
            } else {
                sl_remove(dirs, i-1);
                sl_remove(dirs, i-1);
                i -= 2;
            }
        }
    }
    result = sl_join(dirs, "/");
    sl_free2(dirs);
    return result;
}

