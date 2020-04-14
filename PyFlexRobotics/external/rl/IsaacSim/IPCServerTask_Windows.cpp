#include "IPCServerTask_Windows.h"

#if ISAAC_PLATFORM_WINDOWS

namespace IsaacIPC
{
    bool IPCServerTask::LaunchClient(const ClientDescriptor& cldesc)
    {
        printf("Launching client\n");

        //
        // set up shared memory block
        //

        mShmHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(ShmBlock), SHM_IDENT);
        if (!mShmHandle)
		{
            DWORD e = GetLastError();
            // hmmm, what if we need to resize?
            if (e != ERROR_ALREADY_EXISTS)
			{
                printf("*** Failed to create shared memory block: error %u\n", e);
                return false;
            }
        }

        void* shmem = MapViewOfFile(mShmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(ShmBlock));
        if (!shmem)
		{
            printf("*** Failed to map view of shared memory block\n");
            CloseHandle(mShmHandle);
            mShmHandle = nullptr;
            return false;
        }

        //memset(mShmBlock, 0, sizeof(ShmBlock));

        // construct shared memory block in mapped memory
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
        mShmBlock->mServerTID = mCommTID;

        //
        // launch client process
        //

        //
        // TODO: this should happen in comm thread to enable parallel initialization
        //

        //const wchar_t* exePath = L"C:\\lukasz\\isaac\\CommTest\\Debug\\TestPeer.exe";
        //wchar_t cmdLine[1024];
        //wsprintf(cmdLine, L"%s", exePath);

        PROCESS_INFORMATION processInfo;
        ZeroMemory(&processInfo, sizeof(processInfo));

        STARTUPINFO startupInfo;
        ZeroMemory(&startupInfo, sizeof(startupInfo));
        startupInfo.cb = sizeof(startupInfo);

        // cmdline passed to CreateProcess must be writable
		#ifdef UNICODE
			std::unique_ptr<wchar_t[]> cmdline(new wchar_t[cldesc.mCmdLine.size() + 1]);
			lstrcpyW(cmdline.get(), cldesc.mCmdLine.c_str());
		#else
			std::unique_ptr<char> cmdline(new char[cldesc.mCmdLine.size() + 1]);
			strcpy(cmdline.get(), cldesc.mCmdLine.c_str());
		#endif	

        if (!CreateProcess(cldesc.mExePath.c_str(), cmdline.get(), NULL, NULL, TRUE, 0, NULL, cldesc.mWorkDir.c_str(), &startupInfo, &processInfo))
		{
            printf("*** Failed to create process: error %u\n", GetLastError());
            CloseHandle(mShmHandle);
            mShmHandle = nullptr;
            mShmBlock = nullptr;
            // kill the comm thread
            PostThreadMessage(mCommTID, WM_QUIT, 0, 0);
            return false;
        }

        printf("Created client process with pid %u and thread id %u\n", processInfo.dwProcessId, processInfo.dwThreadId);

        return true;
    }

    void IPCServerTask::ShutdownClient()
    {
        PostThreadMessage(mCommTID, WM_QUIT, 0, 0);             // hmmm, send Isaac MSG_QUIT or just kill the thread?

        if (mCommThread.joinable())
		{
            mCommThread.join();
        }

        // FIXME: finish
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
        //printf("Server sending init response\n");

        if (!mShmBlock)
		{
            return false;
        }

        // write payload to shared memory
        mShmBlock->mNumObservations = (DWORD)numObservations;
        mShmBlock->mNumActions = (DWORD)numActions;
		mShmBlock->mNumExtras = (DWORD)numExtras;

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
        if (!PostThreadMessage(mShmBlock->mClientTID, ISAAC_WM_MESSAGE_ALERT, 0, 0))
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
        if (!PostThreadMessage(mShmBlock->mClientTID, ISAAC_WM_MESSAGE_ALERT, 0, 0))
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
        if (!PostThreadMessage(mShmBlock->mClientTID, ISAAC_WM_MESSAGE_ALERT, 0, 0)) {
            printf("*** Failed to send step response\n");
            return false;
        }

        return true;
    }


    void IPCServerTask::CommThreadFunc()
    {
        DWORD threadId = GetCurrentThreadId();

        printf("Server comm thread id is %u\n", threadId);

        // trigger creation of thread message queue before notifying parent
        MSG msg;
        if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
            TranslateMessage(&msg);
            DispatchMessage(&msg);
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

        while (true)
		{
            if (GetMessage(&msg, 0, 0, 0))
			{
            
                //printf("++ Server comm thread got message %u\n", msg.message);
            
                if (msg.message == ISAAC_WM_MESSAGE_ALERT)
				{
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
				else
				{
                    printf("*** Server comm thread got unexpected windows thread message %u\n", msg.message);
                    // hmmm, quit?
                    break;
                }

                // hmmm, needed?
                TranslateMessage(&msg);
                DispatchMessage(&msg);

                if (shouldQuit)
				{
                    break;
                }

            }
			else
			{
                // told to quit
                printf("Server comm thread told to quit\n");
                break;
            }
        }

        printf("Server comm thread exiting\n");
    }
}

#endif
