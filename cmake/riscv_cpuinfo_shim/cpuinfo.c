#include "cpuinfo.h"

static const struct cpuinfo_core kDummyCore = {0, 0};
static const struct cpuinfo_processor kDummyProcessor = {
    &kDummyCore,
    {0, 0, 0},
};
static const struct cpuinfo_uarch_info kDummyUarch = {cpuinfo_uarch_unknown};

bool cpuinfo_initialize(void) { return false; }

void cpuinfo_deinitialize(void) {}

uint32_t cpuinfo_get_processors_count(void) { return 0; }

const struct cpuinfo_processor* cpuinfo_get_processor(uint32_t index) {
  (void)index;
  return &kDummyProcessor;
}

bool cpuinfo_has_arm_neon_dot(void) { return false; }
bool cpuinfo_has_x86_sse4_2(void) { return false; }
bool cpuinfo_has_x86_avx(void) { return false; }
bool cpuinfo_has_x86_avx2(void) { return false; }
bool cpuinfo_has_x86_fma3(void) { return false; }
bool cpuinfo_has_x86_avx512f(void) { return false; }
bool cpuinfo_has_x86_avx512dq(void) { return false; }
bool cpuinfo_has_x86_avx512cd(void) { return false; }
bool cpuinfo_has_x86_avx512bw(void) { return false; }
bool cpuinfo_has_x86_avx512vl(void) { return false; }
bool cpuinfo_has_x86_avx512vnni(void) { return false; }

uint32_t cpuinfo_get_current_uarch_index(void) { return 0; }

const struct cpuinfo_uarch_info* cpuinfo_get_uarch(uint32_t index) {
  (void)index;
  return &kDummyUarch;
}
