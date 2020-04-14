#ifndef WINDOWS_HEADER_H_
#define WINDOWS_HEADER_H_

#include "IsaacPlatform.h"

#if ISAAC_PLATFORM_WINDOWS
# ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN 1
# endif
# include <Windows.h>
# ifdef min
#  undef min
# endif
# ifdef max
#  undef max
# endif
#endif

#endif
