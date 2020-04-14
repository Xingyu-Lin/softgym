#ifndef IPC_SERVER_TASK_LINUX_H_
#define IPC_SERVER_TASK_LINUX_H_

#include "../IsaacCommon/IsaacIPC.h"

#if ISAAC_PLATFORM_LINUX

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <sys/types.h>

#define ISAAC_SERVER_IPC_LOCKLESS   1

namespace IsaacIPC
{
    struct ClientDescriptor
    {
        std::string             mExePath;
        std::string             mCmdLine;
        std::string             mWorkDir;
    };

    class IPCServerTask
    {
        std::thread             mCommThread;
        pid_t                   mCommTID            = 0;

        std::mutex              mCommMutex;
        std::condition_variable mCommCond;

#if ISAAC_SERVER_IPC_LOCKLESS
        std::atomic<IsaacUint32> mRequestId         { 0 };
#else
        IsaacUint32             mRequestId          = 0;
#endif

        int                     mShmFd              = -1;
        ShmBlock*               mShmBlock           = nullptr;

        const float*            mActionsBuffer      = nullptr;      // where we read actions

        float*                  mObservationsBuffer = nullptr;      // where we write observations
        float*                  mRewardsBuffer      = nullptr;      // where we write rewards
        std::uint8_t*           mDeathsBuffer       = nullptr;      // where we write deaths
		float*					mExtrasBuffer		= nullptr;      // where we write extra info

        void                    CommThreadFunc();

    public:
        bool                    LaunchClient(const ClientDescriptor& cldesc);
        void                    ShutdownClient();

        int                     NumActors() const                   { return mShmBlock ? (int)mShmBlock->mNumActors: 0; }
        int                     NumObservations() const             { return mShmBlock ? (int)mShmBlock->mNumObservations : 0; }
        int                     NumActions() const                  { return mShmBlock ? (int)mShmBlock->mNumActions : 0; }
		int						NumExtras() const					{ return mShmBlock ? (int)mShmBlock->mNumExtras : 0; }

        const float*            GetActions() const                  { return mActionsBuffer; }

        IsaacUint32             PollRequest();

		bool                    PostInitResponse(int numObservations, int numActions, int numExtras = 0);
		bool                    PostResetResponse(const float* observations, const float* extras = 0);
		bool                    PostStepResponse(const float* observations, const float* rewards, const std::uint8_t* deaths, const float* extras = 0);
	};
}

#endif

#endif
