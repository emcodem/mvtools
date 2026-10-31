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
#ifdef __linux__
#include <elf.h>
#include <cstdio>
#include <cstring>
#endif

namespace dynlib
{
#ifdef __linux__
  namespace detail
  {
    // ld.so maps the PT_LOAD segments of a library from the file; when the file is shorter than its program headers
    // say (truncated download, interrupted copy), touching the missing pages raises SIGBUS inside dlopen. So: true only
    // when every byte the headers promise is in the file. Windows' LoadLibrary rejects a truncated PE itself.
    template <typename Ehdr, typename Phdr>
    inline bool elf_segments_in_file(std::FILE *f, unsigned long long size)
    {
      Ehdr eh;
      if (std::fseek(f, 0, SEEK_SET) != 0 || std::fread(&eh, sizeof(eh), 1, f) != 1)
        return false;
      if (eh.e_phentsize != sizeof(Phdr) || (unsigned long long)eh.e_phoff + (unsigned long long)eh.e_phnum * sizeof(Phdr) > size)
        return false;
      for (unsigned i = 0; i < eh.e_phnum; i++)
      {
        Phdr ph;
        if (std::fseek(f, (long)(eh.e_phoff + (unsigned long long)i * sizeof(Phdr)), SEEK_SET) != 0 || std::fread(&ph, sizeof(ph), 1, f) != 1)
          return false;
        if ((ph.p_type == PT_LOAD || ph.p_type == PT_DYNAMIC) && (unsigned long long)ph.p_offset + ph.p_filesz > size)
          return false;
      }
      return true;
    }

    // false (and why) for a file that would crash dlopen; true for anything else, including files dlopen rejects itself
    inline bool elf_complete(const std::filesystem::path &path, std::string &why)
    {
      std::error_code ec;
      const unsigned long long size = std::filesystem::file_size(path, ec);
      if (ec)
        return true; // missing: dlopen reports it
      std::FILE *f = std::fopen(path.c_str(), "rb");
      if (!f)
        return true;
      unsigned char ident[EI_NIDENT] = {};
      bool ok = true;
      if (std::fread(ident, 1, EI_NIDENT, f) == EI_NIDENT && std::memcmp(ident, ELFMAG, SELFMAG) == 0)
        ok = ident[EI_CLASS] == ELFCLASS64 ? elf_segments_in_file<Elf64_Ehdr, Elf64_Phdr>(f, size)
           : ident[EI_CLASS] == ELFCLASS32 ? elf_segments_in_file<Elf32_Ehdr, Elf32_Phdr>(f, size) : true;
      std::fclose(f);
      if (!ok)
        why = path.string() + ": truncated or damaged ELF file (shorter than its segments)";
      return ok;
    }

    inline std::string &pending_error()
    {
      static thread_local std::string e;
      return e;
    }
  } // namespace detail
#endif

#ifdef _WIN32
  constexpr const char *cuda_lib_name = "mvtools_cuda.dll";
  constexpr const char *nvofa_lib_name = "nvofapi64.dll";
  constexpr const char *vsbridge_lib_name = "mvtools_vsbridge.dll";
#else
  constexpr const char *cuda_lib_name = "libmvtools_cuda.so";
  constexpr const char *nvofa_lib_name = "libnvidia-opticalflow.so.1";
  constexpr const char *vsbridge_lib_name = "libmvtools_vsbridge.so";
#endif

  // nullptr when the library cannot be loaded. A bare file name searches the system's library path. Libraries are never
  // unloaded: their functions stay in use until the process ends.
  //
  // Contract for every lazily loaded dependency (mvtools_cuda, NVOFA, NVENC, ...): a missing or broken library must never
  // hang or crash the host, only switch the caller to its fallback. Hence:
  // - Windows: no modal "system error" box for a missing / broken DLL or dependency (it would block a headless encode);
  // - POSIX: RTLD_NOW, so unresolved symbols fail here instead of aborting the process at the first call;
  // - Linux: a library file shorter than its ELF segments is refused before dlopen, which would die of SIGBUS on it;
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
#ifdef __linux__
    // only a path names a file to check; a bare name is searched by ld.so (system libraries)
    detail::pending_error().clear();
    if (path.has_parent_path() && !detail::elf_complete(path, detail::pending_error()))
      return nullptr;
#endif
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  }

  // the last loader error as text, for log messages
  inline std::string last_error()
  {
#ifdef _WIN32
    return "error " + std::to_string(GetLastError());
#else
#ifdef __linux__
    if (!detail::pending_error().empty())
    {
      std::string e;
      e.swap(detail::pending_error());
      return e;
    }
#endif
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
