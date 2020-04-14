#ifndef ISAAC_PLATFORM_H_
#define ISAAC_PLATFORM_H_

#include <stdint.h>

#define ISAAC_PLATFORM_WINDOWS          0
#define ISAAC_PLATFORM_WINDOWS32        0
#define ISAAC_PLATFORM_WINDOWS64        0
#define ISAAC_PLATFORM_LINUX            0
#define ISAAC_PLATFORM_UNIX             0

#if defined(_WIN32)
#  undef  ISAAC_PLATFORM_WINDOWS
#  define ISAAC_PLATFORM_WINDOWS        1
#  if defined(_WIN64)
#    undef  ISAAC_PLATFORM_WINDOWS64
#    define ISAAC_PLATFORM_WINDOWS64    1
#  else
#    undef  ISAAC_PLATFORM_WINDOWS32
#    define ISAAC_PLATFORM_WINDOWS32    1
#  endif
#elif defined(__unix__)
#  undef  ISAAC_PLATFORM_UNIX
#  define ISAAC_PLATFORM_UNIX           1
#  if defined(__linux__)
#    undef  ISAAC_PLATFORM_LINUX
#    define ISAAC_PLATFORM_LINUX        1
#  endif
#endif

#if ISAAC_PLATFORM_WINDOWS
#  ifdef ISAAC_DLL_BUILD
#    define ISAAC_API __declspec(dllexport)
#  else
#    define ISAAC_API __declspec(dllimport)
#  endif
#  define ISAAC_CALL __cdecl
#else // not windows
#  define ISAAC_API
#  define ISAAC_CALL
#endif

// sized numeric types
typedef int8_t      IsaacInt8;
typedef uint8_t     IsaacUint8;
typedef uint8_t     IsaacByte;
typedef int16_t     IsaacInt16;
typedef uint16_t    IsaacUint16;
typedef int32_t     IsaacInt32;
typedef uint32_t    IsaacUint32;
typedef float       IsaacFloat32;
typedef double      IsaacFloat64;

// C API for binary compatibility with Python etc.
typedef int32_t     IsaacCInt;
typedef int32_t     IsaacCBool;
typedef uint8_t     IsaacCByte;
typedef float       IsaacCFloat;

#define ISAAC_FALSE  0
#define ISAAC_TRUE   1

#include <string>

#if ISAAC_PLATFORM_WINDOWS && defined(UNICODE)
#include <tchar.h>
typedef wchar_t         IsaacChar;
typedef std::wstring    IsaacString;
# define ISAAC_TEXT(s)  _T(s)
#else
typedef char            IsaacChar;
typedef std::string     IsaacString;
# define ISAAC_TEXT(s)  s
#endif

#endif
