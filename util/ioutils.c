/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <stdarg.h>
#include <libgen.h>
#include <dirent.h>
#include <time.h>

#include "os-features.h"
#include "ioutils.h"
#include "errors.h"
#include "log.h"

uint32_t ENDIAN_DETECTOR = 0x01020304;

#include "qsort_reentrant.c"

int copy_file(const char* infn, const char* outfn) {
    FILE* fin = fopen(infn, "rb");
    FILE* fout = fopen(outfn, "wb");
    struct stat st;
    off_t len;
    if (!fin) {
        SYSERROR("Failed to open xyls file \"%s\" for copying", infn);
        return -1;
    }
    if (stat(infn, &st)) {
        SYSERROR("Failed to stat file \"%s\"", infn);
        return -1;
    }
    len = st.st_size;
    if (!fout) {
        SYSERROR("Failed to open output xyls file \"%s\" for copying", outfn);
        return -1;
    }
    if (pipe_file_offset(fin, 0, len, fout)) {
        ERROR("Failed to copy xyls file \"%s\" to \"%s\"", infn, outfn);
        return -1;
    }
    if (fclose(fin)) {
        SYSERROR("Failed to close input file \"%s\"", infn);
        return -1;
    }
    if (fclose(fout)) {
        SYSERROR("Failed to close output file \"%s\"", outfn);
        return -1;
    }
    return 0;
}

sl* split_long_string(const char* str, int firstlinew, int linew, sl* lst) {
    const char* s;
    char* added;
    int lw = firstlinew;
    if (!lst)
        lst = sl_new(16);
    assert(linew > 1);
    assert(str);
    s = str;
    while (1) {
        int brk = -1;
        int i, N;
        N = strlen(s);
        if (!N)
            break;
        if (N <= lw) {
            sl_append(lst, s);
            break;
        }
        // scan for last space (' ') before "lw".
        for (i=0; i<MIN(lw+1, N); i++) {
            if (s[i] == ' ')
                brk = i;
        }
        if (brk <= 1) {
            // no space -- hard-break at "lw"; add hyphen.
            added = sl_appendf(lst, "%.*s-", lw-1, s);
            s += strlen(added)-1;
        } else {
            // trim trailing spaces...
            while (brk >= 1 && s[brk-1] == ' ')
                brk--;
            added = sl_appendf(lst, "%.*s", brk, s);
            s += strlen(added);
            // trim spaces.
            while (s && s[0]==' ') s++;
        }
        lw = linew;
    }
    return lst;
}

int split_string_once(const char* str, const char* splitstr,
                      char** first, char** second) {
    char* start = strstr(str, splitstr);
    int n;
    if (!start) {
        if (first) *first = NULL;
        if (second) *second = NULL;
        return 0;
    }
    if (first) {
        n = start - str;
        *first = malloc(1 + n);
        memcpy(*first, str, n);
        (*first)[n] = '\0';
    }
    if (second) {
        char* sec = start + strlen(splitstr);
        n = strlen(sec);
        *second = malloc(1 + n);
        memcpy(*second, sec, n);
        (*second)[n] = '\0';
    }
    return 1;
}

int write_file(const char* fn, const char* data, int len) {
    FILE* fid = fopen(fn, "wb");
    if (!fid) {
        SYSERROR("Failed to open file \"%s\"", fn);
        return -1;
    }
    if (fwrite(data, 1, len, fid) != len) {
        SYSERROR("Failed to write %i bytes to file \"%s\"", len, fn);
        return -1;
    }
    if (fclose(fid)) {
        SYSERROR("Failed to close file \"%s\"", fn);
        return -1;
    }
    return 0;
}

int pad_fid(FILE* fid, size_t len, char pad) {
    off_t offset;
    size_t npad;
    size_t i;
    char buf[1024];
    offset = ftello(fid);
    if (len <= offset)
        return 0;
    npad = len - offset;
    // pad in blocks.
    memset(buf, pad, sizeof(buf));
    for (i=0; i<npad; i+=sizeof(buf)) {
        size_t n = MIN(sizeof(buf), npad-i);
        if (fwrite(buf, 1, n, fid) != n) {
            SYSERROR("Failed to pad file");
            return -1;
        }
    }
    return 0;
}

int pad_file(char* filename, size_t len, char pad) {
    int rtn;
    FILE* fid = fopen(filename, "ab");
    if (!fid) {
        SYSERROR("Failed to open file \"%s\" for padding", filename);
        return -1;
    }
    rtn = pad_fid(fid, len, pad);
    if (!rtn && fclose(fid)) {
        SYSERROR("Failed to close file \"%s\" after padding it", filename);
        return -1;
    }
    return rtn;
}

Malloc
char* dirname_safe(const char* path) {
    char* copy = strdup(path);
    char* res = strdup(dirname(copy));
    free(copy);
    return res;
}

Malloc
char* basename_safe(const char* path) {
    char* copy = strdup(path);
    char* res = strdup(basename(copy));
    free(copy);
    return res;
}

char* find_file_in_dirs(const char** dirs, int ndirs, const char* filename, anbool allow_absolute) {
    int i;
    if (!filename) return NULL;
    if (allow_absolute && filename[0] == '/') {
        if (file_readable(filename))
            return strdup(filename);
    }
    for (i=0; i<ndirs; i++) {
        char* fn;
        asprintf_safe(&fn, "%s/%s", dirs[i], filename);
        if (file_readable(fn))
            return fn;
        free(fn);
    }
    return NULL;
}

float get_cpu_usage() {
    struct rusage r;
    float sofar;
    if (getrusage(RUSAGE_SELF, &r)) {
        SYSERROR("Failed to get resource usage");
        return -1.0;
    }
    sofar = (float)(r.ru_utime.tv_sec + r.ru_stime.tv_sec) +
        (1e-6 * (r.ru_utime.tv_usec + r.ru_stime.tv_usec));
    return sofar;
}

anbool streq(const char* s1, const char* s2) {
    if (s1 == NULL || s2 == NULL)
        return (s1 == s2);
    return (strcmp(s1, s2) == 0) ? TRUE : FALSE;
}

anbool strcaseeq(const char* s1, const char* s2) {
    if (s1 == NULL || s2 == NULL)
        return (s1 == s2);
    return !strcasecmp(s1, s2);
}

int pipe_file_offset(FILE* fin, off_t offset, off_t length, FILE* fout) {
    char buf[1024];
    off_t i;
    if (fseeko(fin, offset, SEEK_SET)) {
        SYSERROR("Failed to seek to offset %zu", (size_t)offset);
        return -1;
    }
    for (i=0; i<length; i+=sizeof(buf)) {
        int n = sizeof(buf);
        if (i + n > length) {
            n = length - i;
        }
        if (fread(buf, 1, n, fin) != n) {
            SYSERROR("Failed to read %i bytes", n);
            return -1;
        }
        if (fwrite(buf, 1, n, fout) != n) {
            SYSERROR("Failed to write %i bytes", n);
            return -1;
        }
    }
    return 0;
}

void asprintf_safe(char** strp, const char* format, ...) {
    va_list lst;
    int rtn;
    va_start(lst, format);
    rtn = vasprintf(strp, format, lst);
    if (rtn == -1) {
        fprintf(stderr, "Error, vasprintf() failed: %s\n", strerror(errno));
        fprintf(stderr, "  (format: \"%s\")\n", format);
        assert(0);
        *strp = NULL;
    }
    va_end(lst);
}

sl* dir_get_contents(const char* path, sl* list, anbool filesonly, anbool recurse) {
    DIR* dir = opendir(path);
    if (!dir) {
        fprintf(stderr, "Failed to open directory \"%s\": %s\n", path, strerror(errno));
        return NULL;
    }
    if (!list)
        list = sl_new(256);
    while (1) {
        struct dirent* de;
        struct stat st;
        char* name;
        char* fullpath;
        anbool freeit = FALSE;
        errno = 0;
        de = readdir(dir);
        if (!de) {
            if (errno)
                fprintf(stderr, "Failed to read entry from directory \"%s\": %s\n", path, strerror(errno));
            break;
        }
        name = de->d_name;
        if (!strcmp(name, ".") || !strcmp(name, ".."))
            continue;
        asprintf_safe(&fullpath, "%s/%s", path, name);
        if (stat(fullpath, &st)) {
            fprintf(stderr, "Failed to stat file %s: %s\n", fullpath, strerror(errno));
            // this can happen when files are deleted, eg
            continue;
            //closedir(dir);
            //sl_free2(list);
            //return NULL;
        }

        if (filesonly) {
            if (S_ISREG(st.st_mode) || S_ISLNK(st.st_mode))
                sl_append_nocopy(list, fullpath);
            else
                freeit = TRUE;
        } else {
            sl_append_nocopy(list, fullpath);
        }
        if (recurse && S_ISDIR(st.st_mode)) {
            dir_get_contents(path, list, filesonly, recurse);
        }
        if (freeit)
            free(fullpath);
    }
    closedir(dir);
    return list;
}

static int readfd(int fd, char* buf, int NB, char** pcursor,
                  sl* lines, anbool* pdone) {
    int nr;
    int i, nleft;
    char* cursor = *pcursor;
    nr = read(fd, cursor, buf + NB - cursor);
    //printf("nr = %i\n", nr);
    if (nr == -1) {
        SYSERROR("Failed to read output fd");
        return -1;
    }
    if (nr == 0) {
        if (cursor != buf) {
            //printf("flushing the last line\n");
            sl_appendf(lines, "%.*s", (int)(cursor - buf), buf);
        }
        *pdone = TRUE;
        return 0;
    }

    // join the newly-read bytes with the carried-over ones.
    nleft = nr + (cursor - buf);
    cursor = buf;
    //printf("nleft = %i\n", nleft);

    for (i=0; i<nleft; i++) {
        if (cursor[i] == '\n' || cursor[i] == '\0') {
            cursor[i] = '\0';
            sl_append(lines, cursor);
            //printf("  found line of length %i: \"%s\"\n", i, cursor);
            cursor += (i+1);
            nleft -= (i+1);
            //printf("  now nleft = %i\n", nleft);
            i = -1;
        }
    }

    if (nleft == NB) {
        //printf("  buffer full with no newline\n");
        sl_appendf(lines, "%.*s", NB, buf);
        cursor = buf;
    } else if (nleft) {
        if (buf == cursor) {
            cursor += nleft;
        } else {
            //printf("  moving %i to the start of the buffer\n", nleft);
            memmove(buf, cursor, nleft);
            cursor = buf + nleft;
        }
    } else {
        cursor = buf;
    }
    *pcursor = cursor;
    return 0;
}

int run_command_get_outputs(const char* cmd, sl** outlines, sl** errlines) {
    int outpipe[2];
    int errpipe[2];
    pid_t pid;

    if (outlines) {
        if (pipe(outpipe) == -1) {
            SYSERROR("Failed to create stdout pipe");
            return -1;
        }
    }
    if (errlines) {
        if (pipe(errpipe) == -1) {
            SYSERROR("Failed to create stderr pipe");
            return -1;
        }
    }

    fflush(stdout);
    pid = fork();
    if (pid == -1) {
        SYSERROR("Failed to fork");
        return -1;
    } else if (pid == 0) {
        // Child process.
        if (outlines) {
            close(outpipe[0]);
            // bind stdout to the pipe.
            if (dup2(outpipe[1], STDOUT_FILENO) == -1) {
                SYSERROR("Failed to dup2 stdout");
                _exit( -1);
            }
        }
        if (errlines) {
            close(errpipe[0]);
            // bind stderr to the pipe.
            if (dup2(errpipe[1], STDERR_FILENO) == -1) {
                SYSERROR("Failed to dup2 stderr");
                _exit( -1);
            }
        }
        // Use a "system"-like command to allow fancier commands.
        if (execlp("/bin/sh", "/bin/sh", "-c", cmd, (char*)NULL)) {
            SYSERROR("Failed to execlp");
            _exit( -1);
        }
        // execlp doesn't return.
    } else {
        char outbuf[1024];
        char errbuf[1024];
        int status;
        anbool outdone=TRUE, errdone=TRUE;
        int outfd = -1;
        int errfd = -1;
        char* outcursor = outbuf;
        char* errcursor = errbuf;
        int rtn = 0;

        // Parent process.
        if (outlines) {
            close(outpipe[1]);
            outdone = FALSE;
            *outlines = sl_new(256);
            outfd = outpipe[0];
            assert(outfd<FD_SETSIZE);
        }
        if (errlines) {
            close(errpipe[1]);
            errdone = FALSE;
            *errlines = sl_new(256);
            errfd = errpipe[0];
            assert(errfd<FD_SETSIZE);
        }

        // Read from child process's streams...
        while (!outdone || !errdone) {
            fd_set readset;
#if !(defined(__CYGWIN__))
            fd_set errset;
#endif
            int ready;
            FD_ZERO(&readset);
#if !(defined(__CYGWIN__))
            FD_ZERO(&errset);
#endif
            //printf("outdone = %i, errdone = %i\n", outdone, errdone);
            if (!outdone) {
                FD_SET(outfd, &readset);
#if !(defined(__CYGWIN__))
                FD_SET(outfd, &errset);
#endif
            }
            if (!errdone) {
                FD_SET(errfd, &readset);
#if !(defined(__CYGWIN__))
                FD_SET(errfd, &errset);
#endif
            }
            ready = select(MAX(outfd, errfd) + 1, &readset, NULL,
#if !(defined(__CYGWIN__))
                           &errset,
#else
                           NULL,
#endif
                           NULL);
            if (ready == -1) {
                SYSERROR("select() failed");
                rtn = -1;
                goto parentreturn;
            }
            if (!outdone) {
                if (FD_ISSET(outfd, &readset)) {
                    // printf("reading 'out' stream\n");
                    if (readfd(outfd, outbuf, sizeof(outbuf), &outcursor,
                               *outlines, &outdone)) {
                        ERROR("Failed to read from child's output stream");
                        rtn = -1;
                        goto parentreturn;
                    }
                }
                // https://groups.google.com/d/msg/astrometry/H0bQBjaoZeo/19pe8DXGoigJ
                // and https://groups.google.com/forum/#!topic/astrometry/quGEbY1CgR8
#if !(defined(__CYGWIN__) || defined(__sun))
                if (FD_ISSET(outfd, &errset)) {
                    SYSERROR("error reading from child output stream");
                    rtn = -1;
                    goto parentreturn;
                }
#endif
            }
            if (!errdone) {
                if (FD_ISSET(errfd, &readset)) {
                    // printf("reading 'err' stream\n");
                    if (readfd(errfd, errbuf, sizeof(errbuf), &errcursor,
                               *errlines, &errdone)) {
                        ERROR("Failed to read from child's error stream");
                        rtn = -1;
                        goto parentreturn;
                    }
					   
                }
#if !(defined(__CYGWIN__))
                if (FD_ISSET(errfd, &errset)) {
                    SYSERROR("error reading from child error stream");
                    rtn = -1;
                    goto parentreturn;
                }
#endif
            }
        }

        //printf("Waiting for command to finish (PID %i).\n", (int)pid);
        do {
            //logverb("Waiting for command to finish...\n");
            int opts = 0; //WNOHANG;
            pid_t wpid = waitpid(pid, &status, opts);
            if (wpid == -1) {
                SYSERROR("Failed to waitpid() for command to finish");
                rtn = -1;
                goto parentreturn;
            }
            //logverb("waitpid() returned\n");
            //if (pid == 0)
            // process has not finished.
            if (WIFSIGNALED(status)) {
                ERROR("Command was killed by signal %i", WTERMSIG(status));
                rtn = -1;
                goto parentreturn;
            } else {
                int exitval = WEXITSTATUS(status);
                if (exitval == 127) {
                    ERROR("Command not found: %s", cmd);
                    rtn = exitval;
                    goto parentreturn;
                } else if (exitval) {
                    ERROR("Command failed: return value %i", exitval);
                    rtn = exitval;
                    goto parentreturn;
                }
            }
        } while (!WIFEXITED(status) && !WIFSIGNALED(status));

    parentreturn:
        if (outlines)
            close(outpipe[0]);
        if (errlines)
            close(errpipe[0]);
        return rtn;
    }
    
    return 0;
}

int mkdir_p(const char* dirpath) {
    sl* tomake = sl_new(4);
    char* path = strdup(dirpath);
    while (!file_exists(path)) {
        char* dir;
        sl_push(tomake, path);
        dir = strdup(dirname(path));
        free(path);
        path = dir;
    }
    free(path);
    while (sl_size(tomake)) {
        char* path = sl_pop(tomake);
        if (mkdir(path, 0777)) {
            SYSERROR("Failed to mkdir(%s)", path);
            sl_free2(tomake);
            free(path);
            return -1;
        }
        free(path);
    }
    sl_free2(tomake);
    return 0;
}

char* shell_escape(const char* str) {
    char* escape = "|&;()<> \t\n\\'\"";
    int nescape = 0;
    int len = strlen(str);
    int i;
    char* result;
    int j;

    for (i=0; i<len; i++) {
        char* cp = strchr(escape, str[i]);
        if (!cp) continue;
        nescape++;
    }
    result = malloc(len + nescape + 1);
    for (i=0, j=0; i<len; i++, j++) {
        char* cp = strchr(escape, str[i]);
        if (!cp) {
            result[j] = str[i];
        } else {
            result[j] = '\\';
            j++;
            result[j] = str[i];
        }
    }
    assert(j == (len + nescape));
    result[j] = '\0';
    return result;
}

static char* get_temp_dir() {
    char* dir = getenv("TMP");
    if (!dir) {
        dir = "/tmp";
    }
    return dir;
}

char* create_temp_file(const char* fn, const char* dir) {
    char* tempfile;
    int fid;
    if (!dir) {
        dir = get_temp_dir();
    }

    asprintf_safe(&tempfile, "%s/tmp.%s.XXXXXX", dir, fn);
    fid = mkstemp(tempfile);
    if (fid == -1) {
        fprintf(stderr, "Failed to create temp file: %s\n", strerror(errno));
        exit(-1);
    }
    close(fid);
    //printf("Created temp file %s\n", tempfile);
    return tempfile;
}

char* create_temp_dir(const char* name, const char* dir) {
    char* tempdir;
    if (!dir) {
        dir = get_temp_dir();
    }
    asprintf_safe(&tempdir, "%s/tmp.%s.XXXXXX", dir, name);
    // no mkdtemp() in some versions of Solaris;
    // https://groups.google.com/forum/#!topic/astrometry/quGEbY1CgR8
#if defined(__sun)
    mktemp(tempdir);
    if (!mkdir(tempdir, 0700)) {
        SYSERROR("Failed to create temp dir");
        return NULL;
    }
#else
    if (!mkdtemp(tempdir)) {
        SYSERROR("Failed to create temp dir");
        return NULL;
    }
#endif
    return tempdir;
}

sl* file_get_lines(const char* fn, anbool include_newlines) {
    FILE* fid;
    sl* list;
    fid = fopen(fn, "r");
    if (!fid) {
        SYSERROR("Failed to open file %s", fn);
        return NULL;
    }
    list = fid_get_lines(fid, include_newlines);
    fclose(fid);
    return list;
}

sl* fid_add_lines(FILE* fid, anbool include_newlines, sl* list) {
    if (!list)
        list = sl_new(256);
    while (1) {
        char* line = read_string_terminated(fid, "\n\r\0", 3, include_newlines);
        if (!line) {
            // error.
            SYSERROR("Failed to read a line");
            sl_free2(list);
            return NULL;
        }
        if (feof(fid) && line[0] == '\0') {
            free(line);
            break;
        }
        sl_append_nocopy(list, line);
        if (feof(fid))
            break;
    }
    return list;
}

sl* fid_get_lines(FILE* fid, anbool include_newlines) {
    return fid_add_lines(fid, include_newlines, NULL);
}

char* file_get_contents_offset(const char* fn, int offset, int size) {
    char* buf = NULL;
    FILE* fid = NULL;
    fid = fopen(fn, "rb");
    if (!fid) {
        SYSERROR("failed to open file \"%s\"", fn);
        goto bailout;
    }
    buf = malloc(size);
    if (!buf) {
        SYSERROR("failed to malloc %i bytes", size);
        goto bailout;
    }
    if (offset) {
        if (fseeko(fid, offset, SEEK_SET)) {
            SYSERROR("failed to fseeko to %i in file \"%s\"", offset, fn);
            goto bailout;
        }
    }
    if (fread(buf, 1, size, fid) != size) {
        SYSERROR("failed to read %i bytes from \"%s\"", size, fn);
        goto bailout;
    }
    fclose(fid);
    return buf;
 bailout:
    if (fid)
        fclose(fid);
    if (buf)
        free(buf);
    return NULL;
}

void* file_get_contents(const char* fn, size_t* len, anbool addzero) {
    struct stat st;
    char* buf;
    FILE* fid;
    off_t size;
    if (stat(fn, &st)) {
        fprintf(stderr, "file_get_contents: failed to stat file \"%s\"", fn);
        return NULL;
    }
    size = st.st_size;
    fid = fopen(fn, "rb");
    if (!fid) {
        fprintf(stderr, "file_get_contents: failed to open file \"%s\": %s\n", fn, strerror(errno));
        return NULL;
    }
    buf = malloc(size + (addzero ? 1 : 0));
    if (!buf) {
        fprintf(stderr, "file_get_contents: couldn't malloc %lu bytes.\n", (long)size);
        return NULL;
    }
    if (fread(buf, 1, size, fid) != size) {
        fprintf(stderr, "file_get_contents: failed to read %lu bytes: %s\n", (long)size, strerror(errno));
        free(buf);
        return NULL;
    }
    fclose(fid);
    if (addzero)
        buf[size] = '\0';
    if (len)
        *len = size;
    return buf;
}
void get_mmap_size(size_t start, size_t size, off_t* mapstart, size_t* mapsize, int* pgap) {
    int ps = getpagesize();
    int gap = start % ps;
    // start must be a multiple of pagesize.
    *mapstart = start - gap;
    *mapsize  = size  + gap;
    *pgap = gap;
}

time_t file_get_last_modified_time(const char* fn) {
    struct stat st;
    if (stat(fn, &st)) {
        SYSERROR("Failed to stat() file \"%s\"", fn);
        return 0;
    }
    return st.st_mtime;
}

int file_get_last_modified_string(const char* fn, const char* timeformat,
                                  anbool utc, char* output, size_t outsize) {
    struct tm tym;
    time_t t;

    t = file_get_last_modified_time(fn);
    if (t == 0) {
        return -1;
    }
    if (utc) {
        if (!gmtime_r(&t, &tym)) {
            SYSERROR("gmtime_r() failed");
            return -1;
        }
    } else {
        if (!localtime_r(&t, &tym)) {
            SYSERROR("localtime_r() failed");
            return -1;
        }
    }
    strftime(output, outsize, timeformat, &tym);
    return 0;
}

anbool file_exists(const char* fn) {
    return fn && (access(fn, F_OK) == 0);
}

anbool file_readable(const char* fn) {
    return fn && (access(fn, R_OK) == 0);
}

anbool file_executable(const char* fn) {
    return fn && (access(fn, X_OK) == 0);
}

anbool path_is_dir(const char* path) {
    struct stat st;
    if (stat(path, &st)) {
        SYSERROR("Couldn't stat path %s", path);
        return FALSE;
    }
    //return st.st_mode & S_IFDIR;
    return S_ISDIR(st.st_mode);
}

int starts_with(const char* str, const char* prefix) {
    int len = strlen(prefix);
    if (strncmp(str, prefix, len))
        return 0;
    return 1;
}

int ends_with(const char* str, const char* suffix) {
    int len = strlen(suffix);
    int len2 = strlen(str);
    if (len > len2)
        return 0;
    if (strncmp(str + len2 - len, suffix, len))
        return 0;
    return 1;
}

char* strdup_safe(const char* str) {
    char* rtn;
    if (!str) return NULL;
    rtn = strdup(str);
    if (!rtn) {
        fprintf(stderr, "Failed to strdup: %s\n", strerror(errno));
        assert(0);
    }
    return rtn;
}

static int oldsigbus_valid = 0;
static struct sigaction oldsigbus;
static void sigbus_handler(int sig) {
    fprintf(stderr, "\n\n"
            "SIGBUS (Bus error) signal received.\n"
            "One reason this can happen is that an I/O error is encountered\n"
            "on a file that we are reading with \"mmap\".\n\n"
            "Bailing out now.\n\n");
    fflush(stderr);
    exit(-1);
}

void add_sigbus_mmap_warning() {
    struct sigaction sigbus;
    memset(&sigbus, 0, sizeof(struct sigaction));
    sigbus.sa_handler = sigbus_handler;
    if (sigaction(SIGBUS, &sigbus, &oldsigbus)) {
        fprintf(stderr, "Failed to change SIGBUS handler: %s\n", strerror(errno));
        return;
    }
    oldsigbus_valid = 1;
}

void reset_sigbus_mmap_warning() {
    if (oldsigbus_valid) {
        if (sigaction(SIGBUS, &oldsigbus, NULL)) {
            fprintf(stderr, "Failed to restore SIGBUS handler: %s\n", strerror(errno));
            return;
        }
    }
}

int is_word(const char* cmdline, const char* keyword, char** cptr) {
    int len = strlen(keyword);
    if (strncmp(cmdline, keyword, len))
        return 0;
    *cptr = (char*)(cmdline + len);
    return 1;
}

void read_complain(FILE* fin, const char* attempted) {
    if (feof(fin)) {
        SYSERROR("Couldn't read %s: end-of-file", attempted);
    } else if (ferror(fin)) {
        SYSERROR("Couldn't read %s", attempted);
    } else {
        SYSERROR("Couldn't read %s", attempted);
    }
}

int read_u8(FILE* fin, unsigned char* val) {
    if (fread(val, 1, 1, fin) == 1) {
        return 0;
    } else {
        read_complain(fin, "u8");
        return 1;
    }
}

int read_u16(FILE* fin, unsigned int* val) {
    uint16_t v;
    if (fread(&v, 2, 1, fin) == 1) {
        *val = v;
        return 0;
    } else {
        read_complain(fin, "u8");
        return 1;
    }
}

int read_u32_portable(FILE* fin, unsigned int* val) {
    uint32_t u;
    if (fread(&u, 4, 1, fin) == 1) {
        *val = ntohl(u);
        return 0;
    } else {
        read_complain(fin, "u32");
        return 1;
    }
}

int read_double(FILE* fin, double* val) {
    if (fread(val, sizeof(double), 1, fin) == 1) {
        return 0;
    } else {
        read_complain(fin, "double");
        return 1;
    }
}

int read_u32(FILE* fin, unsigned int* val) {
    uint32_t u;
    if (fread(&u, 4, 1, fin) == 1) {
        *val = (unsigned int)u;
        return 0;
    } else {
        read_complain(fin, "u32 native");
        return 1;
    }
}

int read_u32s_portable(FILE* fin, unsigned int* val, int n) {
    int i;
    uint32_t* u = malloc(sizeof(uint32_t) * n);
    if (!u) {
        fprintf(stderr, "Couldn't real uint32s: couldn't allocate temp array.\n");
        return 1;
    }
    if (fread(u, sizeof(uint32_t), n, fin) == n) {
        for (i=0; i<n; i++) {
            val[i] = ntohl(u[i]);
        }
        free(u);
        return 0;
    } else {
        read_complain(fin, "uint32s");
        free(u);
        return 1;
    }
}

int read_fixed_length_string(FILE* fin, char* s, int length) {
    if (fread(s, 1, length, fin) != length) {
        read_complain(fin, "fixed-length string");
        return 1;
    }
    return 0;
}

char* read_string(FILE* fin) {
    return read_string_terminated(fin, "\0", 1, FALSE);
}

static char* growable_buffer_add(char* buf, int index, char c, int* size, int* sizestep, int* maxstep) {
    if (index == *size) {
        // expand
        *size += *sizestep;
        buf = realloc(buf, *size);
        if (!buf) {
            fprintf(stderr, "Couldn't allocate buffer: %i.\n", *size);
            return NULL;
        }
        if (*sizestep < *maxstep)
            *sizestep *= 2;
    }
    buf[index] = c;
    return buf;
}

char* read_string_terminated(FILE* fin, const char* terminators, int nterminators,
                             anbool include_terminator) {
    int step = 1024;
    int maxstep = 1024*1024;
    int i = 0;
    int size = 0;
    char* rtn = NULL;
    for (;;) {
        int c = fgetc(fin);
        if (c == EOF)
            break;
        rtn = growable_buffer_add(rtn, i, c, &size, &step, &maxstep);
        if (!rtn)
            return NULL;
        i++;
        if (memchr(terminators, c, nterminators)) {
            if (!include_terminator)
                i--;
            break;
        }
    }
    if (ferror(fin)) {
        read_complain(fin, "string");
        free(rtn);
        return NULL;
    }
    // add \0 if it isn't already there;
    // return "\0" if nothing was read.
    if (i==0 || (rtn[i-1] != '\0')) {
        rtn = growable_buffer_add(rtn, i, '\0', &size, &step, &maxstep);
        if (!rtn)
            return NULL;
        i++;
    }
    if (i < size) {
        rtn = realloc(rtn, i);
        // shouldn't happen - we're shrinking.
        if (!rtn) {
            fprintf(stderr, "Couldn't realloc buffer: %i\n", i);
        }
    }
    return rtn;
}

int write_string(FILE* fout, char* s) {
    int len = strlen(s) + 1;
    if (fwrite(s, 1, len, fout) != len) {
        fprintf(stderr, "Couldn't write string: %s\n", strerror(errno));
        return 1;
    }
    return 0;
}

int write_fixed_length_string(FILE* fout, char* s, int length) {
    char* str;
    int res;
    str = calloc(length, 1);
    if (!str) {
        fprintf(stderr, "Couldn't allocate a temp buffer of size %i.\n", length);
        return 1;
    }
    sprintf(str, "%.*s", length, s);
    res = fwrite(str, 1, length, fout);
    free(str);
    if (res != length) {
        fprintf(stderr, "Couldn't write fixed-length string: %s\n", strerror(errno));
        return 1;
    }
    return 0;
}

int write_double(FILE* fout, double val) {
    if (fwrite(&val, sizeof(double), 1, fout) == 1) {
        return 0;
    } else {
        fprintf(stderr, "Couldn't write double: %s\n", strerror(errno));
        return 1;
    }
}

int write_float(FILE* fout, float val) {
    if (fwrite(&val, sizeof(float), 1, fout) == 1) {
        return 0;
    } else {
        fprintf(stderr, "Couldn't write float: %s\n", strerror(errno));
        return 1;
    }
}

int write_u8(FILE* fout, unsigned char val) {
    if (fwrite(&val, 1, 1, fout) == 1) {
        return 0;
    } else {
        fprintf(stderr, "Couldn't write u8: %s\n", strerror(errno));
        return 1;
    }
}

int write_u32_portable(FILE* fout, unsigned int val) {
    uint32_t v = htonl((uint32_t)val);
    if (fwrite(&v, 4, 1, fout) == 1) {
        return 0;
    } else {
        fprintf(stderr, "Couldn't write u32: %s\n", strerror(errno));
        return 1;
    }
}

int write_u32s_portable(FILE* fout, unsigned int* val, int n) {
    int i;
    uint32_t* v = malloc(sizeof(uint32_t) * n);
    if (!v) {
        fprintf(stderr, "Couldn't write u32s: couldn't allocate temp array.\n");
        return 1;
    }
    for (i=0; i<n; i++) {
        v[i] = htonl((uint32_t)val[i]);
    }
    if (fwrite(v, sizeof(uint32_t), n, fout) == n) {
        free(v);
        return 0;
    } else {
        fprintf(stderr, "Couldn't write u32s: %s\n", strerror(errno));
        free(v);
        return 1;
    }
}

int write_u32(FILE* fout, unsigned int val) {
    uint32_t v = (uint32_t)val;
    if (fwrite(&v, 4, 1, fout) == 1) {
        return 0;
    } else {
        fprintf(stderr, "Couldn't write u32: %s\n", strerror(errno));
        return 1;
    }
}

int write_u16(FILE* fout, unsigned int val) {
    uint16_t v = (uint16_t)val;
    if (fwrite(&v, 2, 1, fout) == 1) {
        return 0;
    } else {
        fprintf(stderr, "Couldn't write u32: %s\n", strerror(errno));
        return 1;
    }
}

int write_uints(FILE* fout, unsigned int* val, int n) {
    if (fwrite(val, sizeof(unsigned int), n, fout) == n) {
        return 0;
    } else {
        fprintf(stderr, "Couldn't write uints: %s\n", strerror(errno));
        return 1;
    }
}

bread_t* buffered_read_new(int elementsize, int Nbuffer, int Ntotal,
                           int (*refill_buffer)(void* userdata, void* buffer, unsigned int offs, unsigned int nelems),
                           void* userdata) {
    bread_t* br;
    br = calloc(1, sizeof(bread_t));
    br->blocksize = Nbuffer;
    br->elementsize = elementsize;
    br->ntotal = Ntotal;
    br->refill_buffer = refill_buffer;
    br->userdata = userdata;
    return br;
}

void* buffered_read(bread_t* br) {
    void* rtn;
    if (!br->buffer) {
        br->buffer = malloc((size_t)br->blocksize * (size_t)br->elementsize);
        br->nbuff = br->off = br->buffind = 0;
    }
    if (br->buffind == br->nbuff) {
        // read a new block!
        int n = br->blocksize;
        // the new block to read starts after the current block...
        br->off += br->nbuff;
        if (n + br->off > br->ntotal)
            n = br->ntotal - br->off;
        if (!n)
            return NULL;
        memset(br->buffer, 0, (size_t)br->blocksize * (size_t)br->elementsize);
        if (br->refill_buffer(br->userdata, br->buffer, br->off, n)) {
            fprintf(stderr, "buffered_read: Error filling buffer.\n");
            return NULL;
        }
        br->nbuff = n;
        br->buffind = 0;
    }
    rtn = (char*)br->buffer + (br->buffind * br->elementsize);
    br->buffind++;
    return rtn;
}

void buffered_read_resize(bread_t* br, int newsize) {
    br->blocksize = newsize;
    if (br->buffer)
        br->buffer = realloc(br->buffer, (size_t)br->blocksize * (size_t)br->elementsize);
}

void buffered_read_reset(bread_t* br) {
    br->nbuff = br->off = br->buffind = 0;
}

void buffered_read_pushback(bread_t* br) {
    if (!br->buffind) {
        fprintf(stderr, "buffered_read_pushback: Can't push back any further!\n");
        return;
    }
    br->buffind--;
}

void buffered_read_free(bread_t* br) {
    free(br->buffer);
}
