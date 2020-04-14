#include "IPCServerTask_Linux.h"

#if ISAAC_PLATFORM_LINUX

#include "../IsaacCommon/IsaacUtil.h"

#include <cstring>

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>

namespace IsaacIPC
{
    bool IPCServerTask::LaunchClient(const ClientDescriptor& cldesc)
    {
        printf("Launching client\n");

        //
        // set up shared memory block
        //

        mShmFd = shm_open(SHM_IDENT, O_RDWR | O_CREAT, 00666);
        if (mShmFd == -1)
		{
            perror("*** Failed to create shared memory block");
            return false;
        }
        printf("Created shared memory block '%s'\n", SHM_IDENT);

        // resize shmem block
        if (ftruncate(mShmFd, sizeof(ShmBlock)) == -1)
		{
            perror("*** Failed to resize shared memory block");
            shm_unlink(SHM_IDENT);
            return false;
        }
        printf("Allocated %d bytes of shared memory\n", (int)sizeof(ShmBlock));

        void* shmem = mmap(NULL, sizeof(ShmBlock), PROT_READ | PROT_WRITE, MAP_SHARED, mShmFd, 0);
        if (shmem == MAP_FAILED)
		{
            perror("*** Failed to map shared memory block");
            shm_unlink(SHM_IDENT);
            return false;
        }
        printf("Shared memory mapped at address %p\n", shmem);

        // initialize shared memory block
        mShmBlock = new (shmem) ShmBlock;

        // launch comm thread
        mCommThread = std::thread(&IPCServerTask::CommThreadFunc, this);

        // get the comm thread id
        {
            std::unique_lock<std::mutex> lock(mCommMutex);
            while (!mCommTID)
			{
                // TODO: use wait_until
                mCommCond.wait(lock);
            }
        }

        // tell the client where to reach us
        mShmBlock->mServerPID = getpid();
        mShmBlock->mServerTID = mCommTID;
        printf("Server pid is %u and tid is %u\n", mShmBlock->mServerPID, mShmBlock->mServerTID);

        //
        // launch client process
        //

        pid_t clpid = fork();
        if (clpid < 0)
		{
            perror("*** Failed to fork client process");
            munmap(mShmBlock, sizeof(ShmBlock));
            shm_unlink(SHM_IDENT);
            mShmBlock = nullptr;
            return false;
        } 
		else if (clpid == 0)
		{
            // child process (run client)

            // change working directory
            if (chdir(cldesc.mWorkDir.c_str()) == -1)
			{
                printf("*** Warning: failed to change into client working directory '%s'\n", cldesc.mWorkDir.c_str());
            }

            // construct command line argument vector
            std::string cmdline = cldesc.mCmdLine;
            std::vector<char*> args = TokenizeInPlace(cmdline);
            printf("Client command line arguments:\n");
            for (auto arg : args) {
                printf("  %s\n", arg);
            }
            args.push_back(nullptr);    // required by execv

            if (execv(cldesc.mExePath.c_str(), args.data()) == -1)
			{
                perror("*** Failed to execute client process");
                munmap(mShmBlock, sizeof(ShmBlock));
                shm_unlink(SHM_IDENT);
                mShmBlock = nullptr;
                return false;
            } 
			else
			{
                // shouldn't get here
                printf("*** Error: This isn't happening\n");
                munmap(mShmBlock, sizeof(ShmBlock));
                shm_unlink(SHM_IDENT);
                mShmBlock = nullptr;
                return false;
            }
        }

        printf("Launched client process with pid %d\n", clpid);

        sleep(1);

        return true;
    }

    void IPCServerTask::ShutdownClient()
    {
        // FIXME: finish

        if (mCommThread.joinable())
		{
            mCommThread.join();
        }

#if 0
        PostThreadMessage(mCommTID, WM_QUIT, 0, 0);             // hmmm, send Isaac MSG_QUIT or just kill the thread?

        if (mCommThread.joinable())
		{
            mCommThread.join();
        }

        // FIXME: finish
#endif
    }

    IsaacUint32 IPCServerTask::PollRequest()
    {
#if ISAAC_SERVER_IPC_LOCKLESS
        // atomically fetch and clear the request id
        return mRequestId.fetch_and(0);
#else
        // use mutex to fetch and clear the request id
        std::lock_guard<std::mutex> lock(mCommMutex);
        IsaacUint32 req = mRequestId;
        mRequestId = MSG_NULL;
        return req;
#endif
    }

    bool IPCServerTask::PostInitResponse(int numObservations, int numActions, int numExtras)
    {
        printf("Server sending init response\n");

        if (!mShmBlock)
		{
            return false;
        }

        // write payload to shared memory
        mShmBlock->mNumObservations = (IsaacUint32)numObservations;
        mShmBlock->mNumActions = (IsaacUint32)numActions;
		mShmBlock->mNumExtras = (IsaacUint32)numExtras;

        // initialize buffer pointers
        mActionsBuffer = (const float*)mShmBlock->mClientData;

        int numActors = mShmBlock->mNumActors;
        int obsOffset = 0;
        int rewOffset = numActors * mShmBlock->mNumObservations * sizeof(float);
        int dieOffset = rewOffset + numActors * sizeof(float);
		int extraOffset = dieOffset + numActors * mShmBlock->mNumExtras * sizeof(float);

        mObservationsBuffer = (float*)(mShmBlock->mServerData + obsOffset);
        mRewardsBuffer = (float*)(mShmBlock->mServerData + rewOffset);
        mDeathsBuffer = (std::uint8_t*)(mShmBlock->mServerData + dieOffset);
		mExtrasBuffer = (float*)(mShmBlock->mServerData + extraOffset);

        // notify client
        mShmBlock->mResponseId = MSG_INIT_RESPONSE;
        if (!SendLinuxThreadSignal(mShmBlock->mClientPID, mShmBlock->mClientTID, ISAAC_SIG_MESSAGE_ALERT))
		{
            printf("*** Failed to send init response\n");
            return false;
        }

        return true;
    }

    bool IPCServerTask::PostResetResponse(const float* observations, const float* extras)
    {
        //printf("Server sending reset response\n");

        if (!mShmBlock)
		{
            return false;
        }

        // write payload to shared memory
        memcpy(mObservationsBuffer, observations, NumActors() * NumObservations() * sizeof(float));
		if (extras && (NumExtras() > 0))
		{
			memcpy(mExtrasBuffer, extras, NumActors() * NumExtras() * sizeof(float));
		}
        // notify client
        mShmBlock->mResponseId = MSG_RESET_RESPONSE;
        if (!SendLinuxThreadSignal(mShmBlock->mClientPID, mShmBlock->mClientTID, ISAAC_SIG_MESSAGE_ALERT))
		{
            printf("*** Failed to send reset response\n");
            return false;
        }

        return true;
    }

    bool IPCServerTask::PostStepResponse(const float* observations, const float* rewards, const std::uint8_t* deaths, const float* extras)
    {
        //printf("Server sending step response\n");

        if (!mShmBlock)
		{
            return false;
        }

        // write payload to shared memory
        memcpy(mObservationsBuffer, observations, NumActors() * NumObservations() * sizeof(float));
        memcpy(mRewardsBuffer, rewards, NumActors() * sizeof(float));
        memcpy(mDeathsBuffer, deaths, NumActors());
		if (extras && (NumExtras() > 0))
		{
			memcpy(mExtrasBuffer, extras, NumActors() * NumExtras() * sizeof(float));
		}

        // notify client
        mShmBlock->mResponseId = MSG_STEP_RESPONSE;
        if (!SendLinuxThreadSignal(mShmBlock->mClientPID, mShmBlock->mClientTID, ISAAC_SIG_MESSAGE_ALERT))
		{
            printf("*** Failed to send step response\n");
            return false;
        }

        return true;
    }


    // local signal handling stuff
    namespace
    {
        volatile sig_atomic_t g_HaveMessage = 0;

        void MessageSignalHandler(int sig, siginfo_t* siginfo, void* ucontext)
        {
            if (sig == ISAAC_SIG_MESSAGE_ALERT)
			{
                g_HaveMessage = 1;
            }
        }
    }

    void IPCServerTask::CommThreadFunc()
    {
        pid_t threadId = GetLinuxThreadId();

        printf("Server comm thread id is %u\n", (unsigned)threadId);

        //
        // set up signal handler for message alerts
        //

        struct sigaction sigAction;
        struct sigaction oldAction;
        memset(&sigAction, 0, sizeof(sigAction));
        memset(&oldAction, 0, sizeof(oldAction));
        
        sigset_t sigMask;
        sigemptyset(&sigMask);

        sigAction.sa_sigaction = MessageSignalHandler;
        sigAction.sa_mask = sigMask;
        sigAction.sa_flags = SA_SIGINFO | SA_RESTART;

        if (sigaction(ISAAC_SIG_MESSAGE_ALERT, &sigAction, &oldAction) == -1)
		{
            perror("*** Failed to install comm signal handler");
            return;
        }

        // notify parent that we're ready
        {
            std::lock_guard<std::mutex> lock(mCommMutex);
            mCommTID = threadId;
            mCommCond.notify_all();
        }

        //
        // main loop
        //

        bool shouldQuit = false;
        while (!shouldQuit)
		{
            // don't block any signals
            sigset_t sigmask;
            sigemptyset(&sigmask);

            //printf("+++ Waiting for signal\n");

            // wait until a signal arrives
            sigsuspend(&sigmask);

            bool haveMessage = g_HaveMessage != 0;
            g_HaveMessage = 0;

            if (haveMessage)
			{
                //printf("+++ Server comm thread got message signal\n");

                // pass message to main sim thread
                auto m = mShmBlock->mRequestId;
                {
#if !ISAAC_SERVER_IPC_LOCKLESS
                    std::lock_guard<std::mutex> lock(mCommMutex);
#endif
                    mRequestId = m;
                }
                if (m == MSG_QUIT)
				{
                    printf("Server comm thread got client quit message\n");
                    shouldQuit = true;
                }
            }
        }
    }
}

#endif
