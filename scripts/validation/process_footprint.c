// Independent macOS process sampler for guarded qualification, including load-time RPC gaps.
// Build: cc -O2 -Wall -Wextra -Werror scripts/validation/process_footprint.c -o build/process-footprint
#include <libproc.h>
#include <sys/resource.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char *end = NULL;
    errno = 0;
    const long pid = strtol(argv[1], &end, 10);
    if (errno || !end || *end || pid <= 0 || pid > INT_MAX) return 2;
    struct rusage_info_v4 value = {0};
    if (proc_pid_rusage((int)pid, RUSAGE_INFO_V4, (rusage_info_t *)&value) != 0) return 1;
    printf("{\"process_id\":%ld,\"rss_bytes\":%" PRIu64 ",\"physical_footprint_bytes\":%" PRIu64
           ",\"physical_footprint_lifetime_peak_bytes\":%" PRIu64 "}\n",
           pid, (uint64_t)value.ri_resident_size, (uint64_t)value.ri_phys_footprint,
           (uint64_t)value.ri_lifetime_max_phys_footprint);
    return 0;
}
