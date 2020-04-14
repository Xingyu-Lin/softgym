#include "IsaacIPC.h"
#include "IsaacUtil.h"

#include <thread>
#include <cstring>

#if ISAAC_PLATFORM_LINUX
# ifndef _GNU_SOURCE
#  define _GNU_SOURCE
# endif
# include <unistd.h>
# include <sys/syscall.h>
#endif

namespace IsaacIPC
{
    ShmBlock::ShmBlock()
    {
        std::memset(mServerData, 0, sizeof(mServerData));
        std::memset(mClientData, 0, sizeof(mClientData));
    }

#if ISAAC_PLATFORM_WINDOWS

    UINT WaitForThreadMessage(int timeoutMS)
    {
        Timer timer;
        MSG msg;
        while (true) {
            if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
                return msg.message;
            }
            if (timeoutMS >= 0 && timer.GetMilliseconds() >= timeoutMS) {
                return WM_NULL;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // shouldn't get here
        return WM_NULL;
    }

#elif ISAAC_PLATFORM_LINUX

    pid_t GetLinuxThreadId()
    {
        return (pid_t)syscall(SYS_gettid);
    }

    bool SendLinuxThreadSignal(pid_t pid, pid_t tid, int sig)
    {
        return syscall(SYS_tgkill, pid, tid, sig) != -1;
    }
#endif

}
