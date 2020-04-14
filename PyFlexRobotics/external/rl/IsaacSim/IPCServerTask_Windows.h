#ifndef IPC_SERVER_TASK_WINDOWS_H_
#define IPC_SERVER_TASK_WINDOWS_H_

#include "../IsaacCommon/IsaacPlatform.h"
#if ISAAC_PLATFORM_WINDOWS

#include "../IsaacCommon/IsaacIPC.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include <string>

#define ISAAC_SERVER_IPC_LOCKLESS   1

namespace IsaacIPC
{
    struct ClientDescriptor
    {
        IsaacString             mExePath;
        IsaacString             mCmdLine;
        IsaacString             mWorkDir;
    };

    class IPCServerTask
    {
        std::thread             mCommThread;
        DWORD                   mCommTID            = 0;

        std::mutex              mCommMutex;
        std::condition_variable mCommCond;

#if ISAAC_SERVER_IPC_LOCKLESS
        std::atomic<IsaacUint32> mRequestId         { 0 };
#else
        IsaacUint32             mRequestId          = 0;
#endif

        HANDLE                  mShmHandle          = nullptr;
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

        IsaacUint32             PollRequest();

		bool                    PostInitResponse(int numObservations, int numActions, int numExtras = 0);
		bool                    PostResetResponse(const float* observations, const float* extras = 0);
		bool                    PostStepResponse(const float* observations, const float* rewards, const std::uint8_t* deaths, const float* extras = 0);

        const float*            GetActions() const                  { return mActionsBuffer; }
    };
}
#endif

#endif
