#ifndef LCE_RISCV_CPUINFO_SHIM_H_
#define LCE_RISCV_CPUINFO_SHIM_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cpuinfo_uarch {
  cpuinfo_uarch_unknown = 0,
};

struct cpuinfo_core {
  uint32_t processor_start;
  uint32_t processor_count;
};

struct cpuinfo_cache {
  uint32_t size;
  uint32_t processor_start;
  uint32_t processor_count;
};

struct cpuinfo_processor_cache {
  const struct cpuinfo_cache* l1d;
  const struct cpuinfo_cache* l2;
  const struct cpuinfo_cache* l3;
};

struct cpuinfo_processor {
  const struct cpuinfo_core* core;
  struct cpuinfo_processor_cache cache;
};

struct cpuinfo_uarch_info {
  enum cpuinfo_uarch uarch;
};

bool cpuinfo_initialize(void);
void cpuinfo_deinitialize(void);
uint32_t cpuinfo_get_processors_count(void);
const struct cpuinfo_processor* cpuinfo_get_processor(uint32_t index);
bool cpuinfo_has_arm_neon_dot(void);
bool cpuinfo_has_x86_sse4_2(void);
bool cpuinfo_has_x86_avx(void);
bool cpuinfo_has_x86_avx2(void);
bool cpuinfo_has_x86_fma3(void);
bool cpuinfo_has_x86_avx512f(void);
bool cpuinfo_has_x86_avx512dq(void);
bool cpuinfo_has_x86_avx512cd(void);
bool cpuinfo_has_x86_avx512bw(void);
bool cpuinfo_has_x86_avx512vl(void);
bool cpuinfo_has_x86_avx512vnni(void);
uint32_t cpuinfo_get_current_uarch_index(void);
const struct cpuinfo_uarch_info* cpuinfo_get_uarch(uint32_t index);

#ifdef __cplusplus
}
#endif

#endif
