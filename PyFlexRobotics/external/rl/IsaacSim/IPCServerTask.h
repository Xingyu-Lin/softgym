#ifndef IPC_SERVER_TASK_H_
#define IPC_SERVER_TASK_H_

#include "../IsaacCommon/IsaacPlatform.h"

#if ISAAC_PLATFORM_WINDOWS
# include "IPCServerTask_Windows.h"
# elif ISAAC_PLATFORM_LINUX
# include "IPCServerTask_Linux.h"
#else
# error Unsupported platform
#endif

#endif
