// Profiling aid (not for production use): MVTOOLS_SKIP=degrain,compensate,mask,flowblur makes the named filters return
// without doing their pixel processing, after all their input frames were fetched. The picture is wrong; the point is to
// measure the CPU time a filter's own work takes in a real script by comparing total CPU seconds with and without.
#ifndef __MV_PROF_SKIP__
#define __MV_PROF_SKIP__

#include <cstdlib>
#include <cstring>

inline bool mv_prof_skip(const char *name)
{
  static const char *list = std::getenv("MVTOOLS_SKIP");
  return list && std::strstr(list, name) != nullptr;
}

#endif
