#include "RLFlexEnv.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cmath>

using namespace std;

extern json g_sceneJson;

#ifdef NV_FLEX_GYM
extern RLFlexEnv* g_rlflexenv;
#endif

RLFlexEnv::RLFlexEnv()
#if USE_ISAAC_COMM
#else
	: mContext(1), mSocket(mContext, ZMQ_REP)
#endif
{
	mObsBuf = nullptr;
	mRewBuf = nullptr;
	mDieBuf = nullptr;
	mExtraBuf = nullptr;
    mPyToCBuf = nullptr;
	mNumExtras = 0;
    mNumPyToC = 0;
    
#ifdef NV_FLEX_GYM
	g_rlflexenv = this;
#endif
}

void RLFlexEnv::ParseJsonParams(const json& sceneJson)
{
	if (sceneJson.is_null())
	{
		return;
	}
	// Parsing of common JSON parameters
	mNumAgents = sceneJson.value(RL_JSON_NUM_AGENTS, mNumAgents);
	mNumObservations = sceneJson.value(RL_JSON_NUM_OBSERVATIONS, mNumObservations);
	mNumActions = sceneJson.value(RL_JSON_NUM_ACTIONS, mNumActions);
	mMaxEpisodeLength = sceneJson.value(RL_JSON_MAX_EPISODE_LENGTH, mMaxEpisodeLength);

	powerScale = sceneJson.value(RL_JSON_POWER_SCALE, powerScale);
}

void RLFlexEnv::LaunchPythonProcess(const char* python_exe, const char* working_dir, const char* logdir, PPOLearningParams& ppo_params, const json& params_json)
#ifdef NV_FLEX_GYM
{}
#else
{
	if (!mObsBuf)
	{
		cout << "InitRLInfo must be called first" << endl;
		return;
	}

	if(params_json.is_null())
	{
		mLearningStep = ppo_params.resume;
		mPPOParams = ppo_params;

		LaunchPythonProcess_Deprecated(python_exe, working_dir, logdir, ppo_params);
		return;
	}

	json modifiedParams(params_json);

	// setting current "portnum"
	{
		hash<string> hhh;
		mPortNum = hhh(logdir) % 60000;
		// Trying to read value from script if it's present there
		mPortNum = modifiedParams.value(RL_JSON_SCRIPT_PORTNUM, mPortNum);
		// Making sure that mPortNum and json params are in sync
		modifiedParams[RL_JSON_SCRIPT_PORTNUM] = mPortNum;
	}

	// setting correct "optim_batchsize" for the script
	{
		const int optim_batchsize_per_agent = modifiedParams.value(RL_JSON_SCRIPT_OPTIM_BATCH_SIZE_PER_AGENT, ppo_params.optim_batchsize_per_agent);
		modifiedParams[RL_JSON_SCRIPT_OPTIM_BATCH_SIZE] = mNumAgents * optim_batchsize_per_agent;
	}

	// setting correct "num parralel"
	{
		modifiedParams[RL_JSON_SCRIPT_NUM_PARALLEL] = mNumAgents;
	}

	// setting current log directory if it's not set yet
	{
		modifiedParams[RL_JSON_SCRIPT_LOG_DIR] = ppo_params.relativeLogDir;
	}

	// creating script launch string
	string launchStr;
	launchStr.reserve(1000);

	// getting the script filename
	launchStr += JsonGetOrExit<const string>(modifiedParams, RL_JSON_SCRIPT_FILE, "The python script file wasn't specified!");

	// collecting all parameters for the Python script
	const string PythonPrefix = "-";
	for (json::iterator it = modifiedParams.begin(); it != modifiedParams.end(); ++it)
	{
		const string& key = it.key();

		if (!key.compare(0, PythonPrefix.size(), PythonPrefix))
		{
			launchStr += " -";
			launchStr += key;
			launchStr += "=";

			if (it.value().is_string())
			{
				launchStr += it.value().get<string>();
			}
			else
			{
				launchStr += it.value().dump();
			}
		}
	}

	cout << "Run : " << launchStr << endl;

#if USE_ISAAC_COMM

#if ISAAC_PLATFORM_WINDOWS
	constexpr char dirsep = '\\';
#else
	constexpr char dirsep = '/';
#endif

	ClientDescriptor cldesc;
	cldesc.mExePath = python_exe;
	cldesc.mCmdLine = cldesc.mExePath + ' ' + ppo_params.workingDir + dirsep + launchStr;
	cldesc.mWorkDir = ppo_params.workingDir;

	printf("Using client directory '%s'\n", cldesc.mWorkDir.c_str());

	if (!mServerTask.LaunchClient(cldesc))
	{
		printf("*** Failed to launch client\n");
		exit(1);	// eek?
	}

#else
#if RANDOM_CONTROL
#else
	STARTUPINFO info = { sizeof(info) };
	char ccc[5000];
	sprintf(ccc, "python %s", launchStr.c_str());
	CreateProcess(python_exe, ccc, NULL, NULL, TRUE, 0, NULL, workingDir, &info, &mProcessInfo);
	sprintf(ccc, "tcp://*:%d", mPortNum);
	mSocket.bind(ccc);
#endif
#endif
	mLearningStep = ppo_params.resume;
	mPPOParams = ppo_params;

}
#endif

void RLFlexEnv::LaunchPythonProcess_Deprecated(const char* python_exe, const char* working_dir, const char* logdir, PPOLearningParams& ppo_params)
{
	hash<string> hhh;
	mPortNum = hhh(logdir) % 60000;
	char command[10000];
	if (!ppo_params.useGAN)
	{
		sprintf(command, "run_ppo.py --agentName=%s --optim_batchsize=%d --num_timesteps=%d --optim_epochs=%d --optim_stepsize=%f --optim_schedule=%s --desired_kl=%f  --timesteps_per_batch=%d --num-cpu=1 --env-id=Custom1-v0 --hid_size=%d --resume=%d --logdir=%s --portnum=%d --num_parallel=%d --num_hid_layers=%d --gamma=%f --lam=%f --seed=%d --clip_param=%f --entcoeff=%f",
				ppo_params.agent_name.c_str(), ppo_params.optim_batchsize_per_agent * mNumAgents, ppo_params.num_timesteps, ppo_params.optim_epochs, ppo_params.optim_stepsize, ppo_params.optim_schedule.c_str(), ppo_params.desired_kl, ppo_params.timesteps_per_batch, ppo_params.hid_size, ppo_params.resume, logdir, mPortNum, mNumAgents, ppo_params.num_hid_layers, ppo_params.gamma, ppo_params.lam, ppo_params.seed, ppo_params.clip_param, ppo_params.entcoeff);
		//	sprintf(command, "run_ppo2.py --agentName=%s --optim_batchsize=%d --num_timesteps=%d --optim_epochs=%d --optim_stepsize=%f --optim_schedule=%s --desired_kl=%f  --timesteps_per_batch=%d --num-cpu=1 --env-id=Custom1-v0 --hid_size=%d --resume=%d --logdir=%s --portnum=%d --num_parallel=%d --num_hid_layers=%d --gamma=%f --lam=%f --seed=%d --clip_param=%f --entcoeff=%f",
		//		ppo_params.agent_name.c_str(), ppo_params.optim_batchsize_per_agent * mNumAgents, ppo_params.num_timesteps, ppo_params.optim_epochs, ppo_params.optim_stepsize, ppo_params.optim_schedule.c_str(), ppo_params.desired_kl, ppo_params.timesteps_per_batch, ppo_params.hid_size, ppo_params.resume, logdir, mPortNum, mNumAgents, ppo_params.num_hid_layers, ppo_params.gamma, ppo_params.lam, ppo_params.seed, ppo_params.clip_param, ppo_params.entcoeff);
	}
	else
	{
		sprintf(command, "run_ppo_gan.py --agentName=%s --optim_batchsize=%d --num_timesteps=%d --optim_epochs=%d --timesteps_per_batch=%d --num-cpu=1 --env-id=Custom1-v0 --hid_size=%d --resume=%d --logdir=%s --portnum=%d --num_parallel=%d --optim_stepsize=%g --num_hid_layers=%d --gamma=%g --lam=%g --seed=%d --clip_param=%g --entcoeff=%g --gan_hid_size=%d --gan_num_hid_layers=%d --gan_reward_scale=%g --gan_learning_rate=%g --resume_disc=%d --resume_non_disc=%d --mocap_path=%s --gan_batch_size=%d --gan_num_epochs=%d --gan_replay_buffer_size=%d --gan_prob_to_put_in_replay=%g --gan_reward_to_retrain_discriminator=%g --use_distance=%d --use_blend=%d --gan_queue_size=%d",
				ppo_params.agent_name.c_str(), ppo_params.optim_batchsize_per_agent*mNumAgents, ppo_params.num_timesteps, ppo_params.optim_epochs, ppo_params.timesteps_per_batch, ppo_params.hid_size, ppo_params.resume,
				logdir, mPortNum, mNumAgents, ppo_params.optim_stepsize, ppo_params.num_hid_layers, ppo_params.gamma, ppo_params.lam, ppo_params.seed, ppo_params.clip_param, ppo_params.entcoeff, ppo_params.gan_hid_size, ppo_params.gan_num_hid_layers,
				ppo_params.gan_reward_scale, ppo_params.gan_learning_rate, ppo_params.resume_disc, ppo_params.resume_non_disc, ppo_params.mocapPath.c_str(),
				ppo_params.gan_batch_size,
				ppo_params.gan_num_epochs,
				ppo_params.gan_replay_buffer_size,
				ppo_params.gan_prob_to_put_in_replay,
				ppo_params.gan_reward_to_retrain_discriminator,
				ppo_params.useDistance?1:0,
				ppo_params.useBlend?1:0,
				ppo_params.gan_queue_size
			   );
	}
	cout << "Run : " << command << endl;

#if USE_ISAAC_COMM

	/*
		std::wstring cwd = _wgetcwd(nullptr, 0);
		char g_PythonPath[5000];
		char g_ClientDir[5000];
		char g_ScriptName[5000];
		strcpy(g_PythonPath, python_exe);
		strcpy(g_ClientDir, working_dir);
		strcpy(g_ScriptName, command);

		ClientDescriptor cldesc;
		cldesc.mExePath = g_PythonPath;
		cldesc.mCmdLine = cldesc.mExePath + ' ' + g_ClientDir + '\\' + g_ScriptName;
		cldesc.mWorkDir = g_ClientDir;
	*/

#if ISAAC_PLATFORM_WINDOWS
	constexpr char dirsep = '\\';
#else
	constexpr char dirsep = '/';
#endif

	ClientDescriptor cldesc;
	cldesc.mExePath = python_exe;
	cldesc.mCmdLine = cldesc.mExePath + ' ' + ppo_params.workingDir + dirsep + command;
	cldesc.mWorkDir = ppo_params.workingDir;

	printf("Using client directory '%s'\n", cldesc.mWorkDir.c_str());

#ifndef NV_FLEX_GYM
	if (!mServerTask.LaunchClient(cldesc))
	{
		printf("*** Failed to launch client\n");
		exit(1);	// eek?
	}
#endif

#else
#if RANDOM_CONTROL
#else
	STARTUPINFO info = { sizeof(info) };
	char ccc[5000];
	sprintf(ccc, "python %s", command);
	CreateProcess(python_exe, ccc, NULL, NULL, TRUE, 0, NULL, workingDir, &info, &mProcessInfo);
	sprintf(ccc, "tcp://*:%d", mPortNum);
	mSocket.bind(ccc);
#endif
#endif
}

void RLFlexEnv::InitRLInfo()
{
	if (mObsBuf)
	{
		delete[] mObsBuf;
	}
	if (mRewBuf)
	{
		delete[] mRewBuf;
	}
	if (mDieBuf)
	{
		delete[] mDieBuf;
	}
	if (mExtraBuf)
	{
		delete[] mExtraBuf;
	}
    if (mPyToCBuf)
    {
        delete[] mPyToCBuf;
    }

	mAgentDie.resize(mNumAgents, false);
	mFrames.resize(mNumAgents, 0);
	mObsBuf = new float[mNumAgents * mNumObservations]();
	mRewBuf = new float[mNumAgents]();
	mDieBuf = new std::uint8_t[mNumAgents]();
	mCtls.resize(mNumAgents);
	mPrevCtls.resize(mNumAgents);
	for (int a = 0; a < mNumAgents; a++)
	{
		mCtls[a].resize(mNumActions, 0.0f);
		mPrevCtls[a].resize(mNumActions, 0.0f);
	}

	if (mNumExtras > 0)
	{
		mExtraBuf = new float[mNumAgents * mNumExtras]();
	}
    if (mNumPyToC > 0)
    {
		mPyToCBuf = new float[mNumAgents * mNumPyToC]();
    }
}

void RLFlexEnv::HandleCommunication()
{
	// Call this function for handling all communication
	PreHandleCommunication();
	int cmd = -1;

	mCtls.resize(mNumAgents);
	for (int a = 0; a < mNumAgents; a++)
	{
		mCtls[a].resize(mNumActions, 0.0f);
	}

#if RANDOM_CONTROL
	istringstream is("");
	cmd = 1;
#else
#if USE_ISAAC_COMM
	IsaacUint32 req = 0;
#ifndef NV_FLEX_GYM
	while ((req = mServerTask.PollRequest()) == 0);
#endif
	if (req == MSG_INIT_REQUEST)
	{
		cmd = 99;
	}
	if (req == MSG_RESET_REQUEST)
	{
		cmd = 0;
	}
	if (req == MSG_STEP_REQUEST)
	{
		cmd = 1;
	}
	if (req == MSG_QUIT)
	{
#ifndef NV_FLEX_GYM
		mServerTask.ShutdownClient();
#endif
		exit(0);  // eek?
	}
	if (cmd == -1)
	{
		return;
	}
#else
	string str;
	zmq::message_t request;

	//	cout<<"Wait for connection"<<endl;

	//		cout << "** Recv" << endl;
	char* cmdT;
	static char tmpS[300000];
	static char tmpR[300000];
	mSocket.recv(&request);
	cmdT = (char*)request.data();
	int l = (int)request.size();
	memcpy(tmpS, cmdT, l);
	tmpS[l] = '\0';
	//cout<<"Receive: "<<tmpS<<endl;
	//cout << "Got message" << tmpS<<endl;
	istringstream is(tmpS);
	is >> cmd;
#endif
#endif
	if (cmd == 99)
	{
		// Ask for dim
#if USE_ISAAC_COMM
#ifndef NV_FLEX_GYM
		mServerTask.PostInitResponse(mNumObservations, mNumActions, mNumExtras);
#endif
#else
		int mNumAgentsExpected = 0;
		is >> mNumAgentsExpected;
		if (mNumAgents != mNumAgentsExpected)
		{
			printf("Num agents don't match!\n");
		}

		sprintf(tmpR, "%d %d", mNumObservations, mNumActions);
		str = tmpR;
#endif
		//cout<<"no_new = "<<g_coarsePar.size()*3<<" na_new = "<<g_connections.size()/2<<endl;
	}
	else if (cmd == 0)
	{
		// Reset
		/*
		g_thisRunResetCount++;
		if (g_thisRunResetCount > 10000)
		{
			TerminateProcess(g_processInfo.hProcess, 0);
			exit(0);
		}
		g_resetCount++;
		cout << "Reset " << g_resetCount << " times" << endl;
		if (g_resetCount % CAPTURE_INTERVAL == 1)
		{
			gRender = 1;
			ToggleVideoCapture();
		}
		if (g_resetCount % CAPTURE_INTERVAL == 2)
		{
			gRender = 0;
			ToggleVideoCapture();
		}
		gShouldReset = true;
		*/

		cout << "Reset!" << endl;

		LockWrite();
		ResetAllAgents();
		UnlockWrite();
		for (int a = 0; a < mNumAgents; a++)
		{
			for (int i = 0; i < mNumActions; i++)
			{
				mCtls[a][i] = 0.0f;
			}
		}

		for (int a = 0; a < mNumAgents; a++)
		{
			mAgentDie[a] = 0;
		}
	}
	else
	{
		// Actions!
#if RANDOM_CONTROL
		for (int a = 0; a < mNumAgents; a++)
			for (int i = 0; i < mNumActions; i++)
			{
				ctls[a][i] = (rand() % 100000)*0.00001f;
			}
#else
#if USE_ISAAC_COMM
#ifndef NV_FLEX_GYM
		const float* acts = mServerTask.GetActions();
		for (int a = 0; a < mNumAgents; a++)
		{
			std::memcpy(&mCtls[a][0], &acts[a * mNumActions], sizeof(float) * mNumActions);
		}
#endif
#else
		for (int a = 0; a < mNumAgents; a++)
			for (int i = 0; i < mNumActions; i++)
			{
				is >> mCtls[a][i];
			}
#endif
#endif
	}
	bool ppoStep = false;
	if ((cmd != 99) && (cmd != 0) && (cmd != 98))
	{
		if (mCtls[0][0] >= 2.9f)
		{
			mCtls[0][0] -= 4.0f;
			cout << "PPO step!" << endl;
			ppoStep = true;
		}
		Simulate();
	}

	if (cmd != 99)
	{
		LockWrite();
		ostringstream os;

		for (int a = 0; a < mNumAgents; a++)
		{
			if (a != 0)
			{
				os << " ";
			}
			PopulateState(a, &mObsBuf[a * mNumObservations]);
			if (mNumExtras > 0)
			{
				PopulateExtra(a, &mExtraBuf[a * mNumExtras]);
			}

			float rew = 0.0f;
			bool dead = false;
			if (cmd != 99)
			{
				ComputeRewardAndDead(a, GetAction(a), &mObsBuf[a * mNumObservations], rew, dead);
			}

			mFrames[a]++;
			if (mFrames[a] >= mMaxEpisodeLength)
			{
				dead = 1;
				//printf("Done = %d because frane >= 1000\n", done);
			}

			if (!std::isfinite(rew))
			{
				printf("Infinite reward!!!!\n");
				rew = 0.0f;
				dead = 1;
			}
			if (mAgentDie[a])
			{
				dead = 1;
				rew = 0.0f;
			}

			if (dead == 1)
			{
				ResetAgent(a);
				PopulateState(a, &mObsBuf[a * mNumObservations]);
				if (mNumExtras > 0)
				{
					PopulateExtra(a, &mExtraBuf[a * mNumExtras]);
				}
			}

#if RANDOM_CONTROL
#else
#if USE_ISAAC_COMM
			mRewBuf[a] = rew;
			mDieBuf[a] = dead ? 1 : 0;
#else
			for (int i = 0; i < mNumObservations; i++)
			{
				if (i > 0)
				{
					os << " ";
				}
				os << mObsBuf[a*mNumObservations + i];
			}
			os << " " << rew << " " << dead;
#endif
#endif
		}
#if USE_ISAAC_COMM
#else
		str = os.str();
#endif
		if (ppoStep)
		{
			mLearningStep++;
			SaveState();
		}
		UnlockWrite();
	}

	// -----------------------------------------
#if RANDOM_CONTROL
#else
#if USE_ISAAC_COMM
	if (cmd != 99)
	{
#ifndef NV_FLEX_GYM
		if (cmd == 0)
		{
			mServerTask.PostResetResponse(mObsBuf, mExtraBuf);
		}
		else
		{
			mServerTask.PostStepResponse(mObsBuf, mRewBuf, mDieBuf, mExtraBuf);
		}
#endif
	}
#else
	zmq::message_t reply(str.size());
	memcpy(reply.data(), &str[0], str.size());
	//cout << "** Send" << endl;
	mSocket.send(reply);
#endif
#endif
}

RLFlexEnv::~RLFlexEnv()
{
	if (mObsBuf)
	{
		delete[] mObsBuf;
	}
	if (mRewBuf)
	{
		delete[] mRewBuf;
	}
	if (mDieBuf)
	{
		delete[] mDieBuf;
	}
	if (mExtraBuf)
	{
		delete[] mExtraBuf;
	}

#ifdef NV_FLEX_GYM
	g_rlflexenv = nullptr;
#endif
}

