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
#include <signal.h>
#include <sys/select.h>
#include <fcntl.h>

#include "bl.h"
#include "solvedfile.h"
#include "ioutils.h"
#include "boilerplate.h"

const char* OPTIONS = "hp:f:";

static void printHelp(char* progname) {
	boilerplate_help_header(stderr);
	fprintf(stderr, "\nUsage: %s\n"
			"   [-p <port>] (default 6789)\n"
			"   [-f <filename-pattern>]  (default solved.%%02i)\n",
			progname);
}

int bailout = 0;
char* solvedfnpattern = "solved.%02i";

static void sighandler(int sig) {
	bailout = 1;
}

extern char *optarg;
extern int optind, opterr, optopt;

int handle_request(FILE* fid) {
	char buf[256];
	char fn[256];
	int set;
	int get;
	int getall;
	int filenum;
	int fieldnum;
	int lastfieldnum;
	int maxfields;
	char* nextword;

	//printf("Fileno %i:\n", fileno(fid));
	if (!fgets(buf, 256, fid)) {
		fprintf(stderr, "Error: failed to read a line of input.\n");
		fflush(stderr);
		fclose(fid);
		return -1;
	}
	//printf("Got request %s\n", buf);
	get = set = getall = 0;
	if (is_word(buf, "get ", &nextword)) {
		get = 1;
	} else if (is_word(buf, "set ", &nextword)) {
		set = 1;
	} else if (is_word(buf, "getall ", &nextword)) {
		getall = 1;
	}

	if (!(get || set || getall)) {
		fprintf(stderr, "Error: malformed command.\n");
		fclose(fid);
		return -1;
	}

	if (get || set) {
		if (sscanf(nextword, "%i %i", &filenum, &fieldnum) != 2) {
			fprintf(stderr, "Error: malformed request: %s\n", buf);
			fflush(stderr);
			fclose(fid);
			return -1;
		}
	} else if (getall) {
		if (sscanf(nextword, "%i %i %i %i", &filenum, &fieldnum, &lastfieldnum, &maxfields) != 4) {
			fprintf(stderr, "Error: malformed request: %s\n", buf);
			fflush(stderr);
			fclose(fid);
			return -1;
		}
		if (lastfieldnum < fieldnum) {
			fprintf(stderr, "Error: invalid \"getall\" request: lastfieldnum must be >= firstfieldnum.\n");
			fflush(stderr);
			fclose(fid);
			return -1;
		}
	}

	sprintf(fn, solvedfnpattern, filenum);

	if (get) {
		int val;
		printf("Get %s [%i].\n", fn, fieldnum);
		fflush(stdout);
		val = solvedfile_get(fn, fieldnum);
		if (val == -1) {
			fclose(fid);
			return -1;
		} else {
			fprintf(fid, "%s %i %i\n", (val ? "solved" : "unsolved"),
					filenum, fieldnum);
			fflush(fid);
		}
		return 0;
	} else if (set) {
		printf("Set %s [%i].\n", fn, fieldnum);
		fflush(stdout);
		if (solvedfile_set(fn, fieldnum)) {
			fclose(fid);
			return -1;
		}
		fprintf(fid, "ok\n");
		fflush(fid);
		return 0;
	} else if (getall) {
		int i;
		il* list;
		printf("Getall %s [%i : %i], max %i.\n", fn, fieldnum, lastfieldnum, maxfields);
		fflush(stdout);
		fprintf(fid, "unsolved %i", filenum);
		list = solvedfile_getall(fn, fieldnum, lastfieldnum, maxfields);
		if (list) {
			for (i=0; i<il_size(list); i++)
				fprintf(fid, " %i", il_get(list, i));
			il_free(list);
		}
		fprintf(fid, "\n");
		fflush(fid);
		return 0;
	}
	return -1;
}

int main(int argc, char** args) {
    int argchar;
	char* progname = args[0];
	int sock;
	struct sockaddr_in addr;
	int port = 6789;
	unsigned int opt;
	pl* clients;
	int flags;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1) {
		switch (argchar) {
		case 'p':
			port = atoi(optarg);
			break;
		case 'f':
			solvedfnpattern = optarg;
			break;
		case 'h':
		default:
			printHelp(progname);
			exit(-1);
		}
	}

	sock = socket(PF_INET, SOCK_STREAM, 0);
	if (sock == -1) {
 		fprintf(stderr, "Error: couldn't create socket: %s\n", strerror(errno));
		exit(-1);
	}

	opt = 1;
	if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
		fprintf(stderr, "Warning: failed to setsockopt() to reuse address.\n");
	}

	flags = fcntl(sock, F_GETFL, 0);
	if (flags == -1) {
		fprintf(stderr, "Warning: failed to get socket flags: %s\n",
					strerror(errno));
	} else {
		flags |= O_NONBLOCK;
		if (fcntl(sock, F_SETFL, flags) == -1) {
			fprintf(stderr, "Warning: failed to set socket flags: %s\n",
					strerror(errno));
		}
	}

	memset(&addr, 0, sizeof(struct sockaddr_in));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(port);
    // gcc with strict-aliasing warn about this cast but according to "the internet"
    // it's okay because we're not dereferencing the cast pointer.
	if (bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
		fprintf(stderr, "Error: couldn't bind socket: %s\n", strerror(errno));
		exit(-1);
	}

	if (listen(sock, 1000)) {
 		fprintf(stderr, "Error: failed to listen() on socket: %s\n", strerror(errno));
		exit(-1);
	}
	printf("Listening on port %i.\n", port);
	fflush(stdout);

	signal(SIGINT, sighandler);

	clients = pl_new(32);

	// wait for a connection or i/o...
	while (1) {
		struct sockaddr_in clientaddr;
		socklen_t addrsz = sizeof(clientaddr);
		FILE* fid;
		fd_set rset;
		struct timeval timeout;
		int res;
		int maxval = 0;
		int i;

		timeout.tv_sec = 1;
		timeout.tv_usec = 0;

		FD_ZERO(&rset);

		maxval = sock;
		for (i=0; i<pl_size(clients); i++) {
			int val;
			fid = pl_get(clients, i);
			val = fileno(fid);
			FD_SET(val, &rset);
			if (val > maxval)
				maxval = val;
		}
		FD_SET(sock, &rset);
		res = select(maxval+1, &rset, NULL, NULL, &timeout);
		if (res == -1) {
			if (errno != EINTR) {
				fprintf(stderr, "Error: select(): %s\n", strerror(errno));
				exit(-1);
			}
		}
		if (bailout)
			break;
		if (!res)
			continue;

		for (i=0; i<pl_size(clients); i++) {
			fid = pl_get(clients, i);
			if (FD_ISSET(fileno(fid), &rset)) {
				if (handle_request(fid)) {
					fprintf(stderr, "Error from fileno %i\n", fileno(fid));
					pl_remove(clients, i);
					i--;
					continue;
				}
			}
		}
		if (FD_ISSET(sock, &rset)) {
            // See comment about strict aliasing above.  Should be okay, despite gcc warning.
			int s = accept(sock, (struct sockaddr*)&clientaddr, &addrsz);
			if (s == -1) {
				fprintf(stderr, "Error: failed to accept() on socket: %s\n", strerror(errno));
				continue;
			}
			if (addrsz != sizeof(clientaddr)) {
				fprintf(stderr, "Error: client address has size %i, not %i.\n", addrsz, (uint)sizeof(clientaddr));
				continue;
			}
			printf("Connection from %s.\n", inet_ntoa(clientaddr.sin_addr));
			fflush(stdout);
			fid = fdopen(s, "a+b");
			pl_append(clients, fid);
		}
	}

	printf("Closing socket...\n");
	if (close(sock)) {
		fprintf(stderr, "Error: failed to close socket: %s\n", strerror(errno));
	}

	return 0;
}

