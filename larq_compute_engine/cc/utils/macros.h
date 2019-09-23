#ifndef COMPUTE_ENGINE_CORE_MACROS_H_
#define COMPUTE_ENGINE_CORE_MACROS_H_

#ifdef __unix__
#include <arpa/inet.h>
#elif defined(_WIN32) || defined(WIN32)
#include <winsock.h>
#endif

#define CE_IS_BIG_ENDIAN (htonl(47) == 47)

// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define CE_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define CE_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define CE_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define CE_ATTRIBUTE_UNUSED __attribute__((unused))
#define CE_ATTRIBUTE_COLD __attribute__((cold))
#define CE_ATTRIBUTE_WEAK __attribute__((weak))
#define CE_PACKED __attribute__((packed))
#define CE_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
// Non-GCC equivalents
// TODO
#endif

#endif  // COMPUTE_ENGINE_CORE_MACROS_H_
