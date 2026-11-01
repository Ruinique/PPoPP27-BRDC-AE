/*****************************************************************************
Copyright (c) 2026, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

/* AE: standalone copy of driver/others/openblas_get_num_nodes.c from the OpenBLAS fork (common.h not needed). */

#if defined(OS_LINUX) && defined(SMP)
#include <dirent.h>
#include <string.h>
#include <stdlib.h>

#ifndef OPENBLAS_MAX_NUMA_NODES
#define OPENBLAS_MAX_NUMA_NODES 256
#endif

int openblas_get_num_nodes(void) {
  DIR *dp;
  struct dirent *dir;
  int nodes = 0;

  dp = opendir("/sys/devices/system/node");
  if (dp == NULL) return 1;

  while ((dir = readdir(dp)) != NULL) {
    char *endptr;
    long node;

    if (strncmp(dir->d_name, "node", 4) != 0) continue;
    node = strtol(dir->d_name + 4, &endptr, 10);
    if (*endptr != '\0') continue;
    if (node < 0 || node >= OPENBLAS_MAX_NUMA_NODES) continue;

    nodes++;
  }

  closedir(dp);
  return nodes > 0 ? nodes : 1;
}
#else
int openblas_get_num_nodes(void) {
  return 1;
}
#endif

int openblas_get_num_nodes_(void) {
  return openblas_get_num_nodes();
}
