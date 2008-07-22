/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <assert.h>

#include "solvedclient.h"
#include "bl.h"

static int serveraddr_initialized = 0;
static struct sockaddr_in serveraddr;
static FILE* fserver = NULL;

int solvedclient_set_server(char* addr) {
	char buf[256];
	char* ind;
	struct hostent* he;
	int len;
	int port;
	if (fserver) {
		if (fflush(fserver) ||
			fclose(fserver)) {
			fprintf(stderr, "Failed to close previous connection to server.\n");
		}
		fserver = NULL;
	}
	if (!addr)
		return -1;
	ind = index(addr, ':');
	if (!ind) {
		fprintf(stderr, "Invalid IP:port address: %s\n", addr);
		return -1;
	}
	len = ind - addr;
	memcpy(buf, addr, len);
	buf[len] = '\0';
	he = gethostbyname(buf);
	if (!he) {
		fprintf(stderr, "Solved server \"%s\" not found: %s.\n", buf, hstrerror(h_errno));
		return -1;
	}
    if (!serveraddr_initialized) {
        memset(&serveraddr, 0, sizeof(serveraddr));
        serveraddr_initialized = 1;
    }
	memcpy(&(serveraddr.sin_addr), he->h_addr, he->h_length);
	port = atoi(ind+1);
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_port = htons(port);

	return 0;
}

static int connect_to_server() {
	int sock;
	if (fserver)
		return 0;
	sock = socket(PF_INET, SOCK_STREAM, 0);
	if (sock == -1) {
		fprintf(stderr, "Couldn't create socket: %s\n", strerror(errno));
		return -1;
	}
	fserver = fdopen(sock, "r+b");
	if (!fserver) {
		fprintf(stderr, "Failed to fdopen socket: %s\n", strerror(errno));
		return -1;
	}
    assert(serveraddr_initialized);
    // gcc with strict-aliasing warns about this cast but it should be okay.
	if (connect(sock, (struct sockaddr*)&serveraddr, sizeof(serveraddr))) {
		fprintf(stderr, "Couldn't connect to server: %s\n", strerror(errno));
		if (fclose(fserver))
			fprintf(stderr, "Failed to close socket: %s\n", strerror(errno));
		fserver = NULL;
		return -1;
	}
	return 0;
}

int solvedclient_get(int filenum, int fieldnum) {
	char buf[256];
	const char* solvedstr = "solved";
	int nchars;
	int solved;

	if (connect_to_server())
		return -1;
	nchars = sprintf(buf, "get %i %i\n", filenum, fieldnum);
	if ((fwrite(buf, 1, nchars, fserver) != nchars) ||
		fflush(fserver)) {
		fprintf(stderr, "Failed to write request to server: %s\n", strerror(errno));
		fclose(fserver);
		fserver = NULL;
		return -1;
	}
	if (!fgets(buf, 256, fserver)) {
		fprintf(stderr, "Couldn't read response: %s\n", strerror(errno));
		fclose(fserver);
		fserver = NULL;
		return -1;
	}
	solved = (strncmp(buf, solvedstr, strlen(solvedstr)) == 0);
	return solved;
}

void solvedclient_set(int filenum, int fieldnum) {
	char buf[256];
	int nchars;
	if (connect_to_server())
		return;
	nchars = sprintf(buf, "set %i %i\n", filenum, fieldnum);
	if ((fwrite(buf, 1, nchars, fserver) != nchars) ||
		fflush(fserver)) {
		fprintf(stderr, "Failed to send command (%s) to solvedserver: %s\n", buf, strerror(errno));
		return;
	}
	// wait for response.
	if (!fgets(buf, 256, fserver)) {
		fprintf(stderr, "Couldn't read response: %s\n", strerror(errno));
		fclose(fserver);
		fserver = NULL;
		return;
	}
}

il* solvedclient_get_fields(int filenum, int firstfield, int lastfield,
							int maxnfields) {
	char* buf;
	int bufsize;
	il* list;
	char* cptr;
	int fld;
	int nchars;

	if (connect_to_server())
		return NULL;
	bufsize = 100 + 10 * (maxnfields ? maxnfields : (1 + lastfield - firstfield));
	buf = malloc(bufsize);
	nchars = sprintf(buf, "getall %i %i %i %i\n", filenum, firstfield,
					 lastfield, maxnfields);
	if ((fwrite(buf, 1, nchars, fserver) != nchars) ||
		fflush(fserver)) {
		fprintf(stderr, "Failed to send command (%s) to solvedserver: %s\n", buf, strerror(errno));
		return NULL;
	}
	// wait for response.
	if (!fgets(buf, bufsize, fserver)) {
		fprintf(stderr, "Couldn't read response: %s\n", strerror(errno));
		fclose(fserver);
		fserver = NULL;
		free(buf);
		return NULL;
	}
	if (sscanf(buf, "unsolved %i%n", &fld, &nchars) != 1) {
		fprintf(stderr, "Couldn't parse response: %s\n", buf);
		free(buf);
		return NULL;
	}
	if (fld != filenum) {
		fprintf(stderr, "Expected file number %i, not %i.\n", filenum, fld);
		free(buf);
		return NULL;
	}
	cptr = buf + nchars;
	list = il_new(256);
	while (*cptr && *cptr != '\n') {
		if (sscanf(cptr, " %i%n", &fld, &nchars) != 1) {
			fprintf(stderr, "Couldn't parse response: %s\n", buf);
			il_free(list);
			free(buf);
			return NULL;
		}
		cptr += nchars;
		il_append(list, fld);
	}
	free(buf);
	return list;
}
