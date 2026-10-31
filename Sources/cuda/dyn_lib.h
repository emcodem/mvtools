// Run-time library loading for Windows and POSIX: open a shared library, look up a symbol, find the folder of the module
// that contains a given address. Used by GpuBackend (mvtools2), cudafilters' CudaLib and mvtools_cuda's NVOFA loader, so
// none of them needs <windows.h> or <dlfcn.h> itself.

#ifndef MV_DYN_LIB_H
#define MV_DYN_LIB_H

#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace dynlib
{
#ifdef _WIN32
  constexpr const char *cuda_lib_name = "mvtools_cuda.dll";
  constexpr const char *nvofa_lib_name = "nvofapi64.dll";
#else
  constexpr const char *cuda_lib_name = "libmvtools_cuda.so";
  constexpr const char *nvofa_lib_name = "libnvidia-opticalflow.so.1";
#endif

  // nullptr when the library cannot be loaded. A bare file name searches the system's library path. Libraries are never
  // unloaded: their functions stay in use until the process ends.
  //
  // Contract for every lazily loaded dependency (mvtools_cuda, NVOFA, NVENC, ...): a missing or broken library must never
  // hang or crash the host, only switch the caller to its fallback. Hence:
  // - Windows: no modal "system error" box for a missing / broken DLL or dependency (it would block a headless encode);
  // - POSIX: RTLD_NOW, so unresolved symbols fail here instead of aborting the process at the first call;
  // - callers check every symbol for nullptr and the library's API version before using any of them;
  // - never call this from a global constructor / DllMain (loader lock); load lazily from filter code.
  inline void *open(const std::filesystem::path &path)
  {
#ifdef _WIN32
    DWORD old_mode = 0;
    const BOOL mode_set = SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &old_mode);
    HMODULE m = LoadLibraryW(path.c_str());
    const DWORD err = GetLastError();
    if (mode_set)
      SetThreadErrorMode(old_mode, nullptr);
    SetLastError(err);
    return reinterpret_cast<void *>(m);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  }

  // the last loader error as text, for log messages
  inline std::string last_error()
  {
#ifdef _WIN32
    return "error " + std::to_string(GetLastError());
#else
    const char *e = dlerror();
    return e ? e : "unknown error";
#endif
  }

  inline void *symbol(void *lib, const char *name)
  {
    if (lib == nullptr)
      return nullptr;
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
  }

  template <typename Fn>
  inline void load(void *lib, Fn &fn, const char *name)
  {
    fn = reinterpret_cast<Fn>(symbol(lib, name));
  }

  // folder of the module (exe, DLL or .so) that contains addr; empty on failure. Pass the address of a function or
  // static object of the calling module.
  inline std::filesystem::path module_dir(const void *addr)
  {
#ifdef _WIN32
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(addr), &m))
      return {};
    wchar_t path[32768];
    const DWORD n = GetModuleFileNameW(m, path, static_cast<DWORD>(sizeof(path) / sizeof(path[0])));
    if (n == 0 || n >= sizeof(path) / sizeof(path[0]))
      return {};
    return std::filesystem::path(path).parent_path();
#else
    Dl_info info;
    if (dladdr(addr, &info) == 0 || info.dli_fname == nullptr)
      return {};
    // dli_fname is the path the library was opened with; make it absolute while the working directory is unchanged
    std::error_code ec;
    std::filesystem::path p = std::filesystem::absolute(info.dli_fname, ec);
    return ec ? std::filesystem::path() : p.parent_path();
#endif
  }
} // namespace dynlib

#endif
