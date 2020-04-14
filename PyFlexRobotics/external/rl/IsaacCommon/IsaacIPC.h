#ifndef ISAAC_IPC_H_
#define ISAAC_IPC_H_

#include "IsaacPlatform.h"

#if ISAAC_PLATFORM_WINDOWS
# include "WindowsHeader.h"
#elif ISAAC_PLATFORM_LINUX
# include <sys/types.h>
# include <signal.h>
#endif

#define SHM_IDENT       (ISAAC_TEXT("IsaacIPC"))

#define MAX_MSG_LEN     (256 * 1024 * 1024)

namespace IsaacIPC
{
    enum IsaacMessageID
    {
        MSG_NULL,

        MSG_INIT_REQUEST,           // sets number of actors
        MSG_INIT_RESPONSE,          // sets number of observations and actions

        MSG_RESET_REQUEST,
        MSG_RESET_RESPONSE,

        MSG_STEP_REQUEST,
        MSG_STEP_RESPONSE,

        MSG_QUIT,
    };

    struct ShmBlock
    {
#if ISAAC_PLATFORM_WINDOWS
        volatile DWORD          mServerTID          = 0;
        volatile DWORD          mClientTID          = 0;
#elif ISAAC_PLATFORM_LINUX
        volatile pid_t          mServerPID          = 0;
        volatile pid_t          mServerTID          = 0;
        volatile pid_t          mClientPID          = 0;
        volatile pid_t          mClientTID          = 0;
#endif

        volatile IsaacUint32    mNumActors          = 0;
        volatile IsaacUint32    mNumObservations    = 0;
        volatile IsaacUint32    mNumActions         = 0;
		volatile IsaacUint32	mNumExtras			= 0;
        volatile IsaacUint32    mRequestId          = MSG_NULL;
        volatile IsaacUint32    mResponseId         = MSG_NULL;

        IsaacByte               mServerData[MAX_MSG_LEN];   // data written by server for client
        IsaacByte               mClientData[MAX_MSG_LEN];   // data written by client for server

                                ShmBlock();
    };

#if ISAAC_PLATFORM_WINDOWS

    constexpr UINT      ISAAC_WM_MESSAGE_ALERT  = WM_APP;

    UINT                WaitForThreadMessage(int timeoutMS);

#elif ISAAC_PLATFORM_LINUX

    constexpr int       ISAAC_SIG_MESSAGE_ALERT = SIGUSR1;

    pid_t               GetLinuxThreadId();

    bool                SendLinuxThreadSignal(pid_t pid, pid_t tid, int sig);

#endif
}

#endif
