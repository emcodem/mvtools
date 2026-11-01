// mvtools_vulkan: call counters of the cudafilters entry points (vk_resample.cpp, vk_temporal.cpp, vk_kobs.cpp). With
// MVVK_LOG=1 every failed call is printed at once (the caller then uses its CPU filter) and the counts at unload, so a
// test can verify that the Vulkan path really ran.
#ifndef MVTOOLS_VK_CF_LOG_H
#define MVTOOLS_VK_CF_LOG_H

#include "vkrt.h"

#include <atomic>
#include <cstdio>

namespace mvk
{
  struct CallLog
  {
    const char *name;
    std::atomic<unsigned long long> ok{ 0 }, failed{ 0 };
    explicit CallLog(const char *n) : name(n) {}
    ~CallLog()
    {
      if ((ok || failed) && vkrt::log_enabled())
        std::fprintf(stderr, "mvtools_vulkan: %s: %llu calls on the GPU, %llu failed\n", name, ok.load(), failed.load());
    }
    // records the result of one call and passes it on
    int operator()(int rc)
    {
      if (rc == 0)
        ok++;
      else
      {
        failed++;
        if (vkrt::log_enabled())
          std::fprintf(stderr, "mvtools_vulkan: %s failed (%d)\n", name, rc);
      }
      return rc;
    }
  };
}

#endif
