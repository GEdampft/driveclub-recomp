// VR fork (build): forced into every C++ file on MinGW (gpu/CMakeLists.txt). Clang 23's libc++ no longer includes
// these transitively, and the vendored sources (sirit, the FidelityFX SDK, ...) relied on it.
#pragma once
#ifdef __cplusplus
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <new>
#endif
