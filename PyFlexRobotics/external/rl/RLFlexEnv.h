#pragma once
#define USE_ISAAC_COMM 1
#define RANDOM_CONTROL 0

#if USE_ISAAC_COMM
#include "IsaacSim/IPCServerTask.h"
using namespace IsaacIPC;
#else
#include "../../zmq/zmq.hpp"
#endif

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>
#include "../json/json_utils.h"

using namespace std;

using json = nlohmann::json;

#define RL_JSON_NUM_AGENTS "NumAgents"
#define RL_JSON_NUM_OBSERVATIONS "NumObservations"
#define RL_JSON_NUM_ACTIONS "NumActions"
#define RL_JSON_MAX_EPISODE_LENGTH "MaxEpisodeLength"
#define RL_JSON_POWER_SCALE "PowerScale"

// Note: JSON can have any combination of thise parameters or none at all
// All parameters (even those that are not parsed by PPOLearningParams) that start with "-" will be trasfered to Python's script command-line with by adding additional "-"
// Now most of the parameters pased only to maintain old code

#define RL_JSON_PYTHON_PATH "PythonPath"
#define RL_JSON_SCRIPT_DIR "ScriptDir"
#define RL_JSON_SCRIPT_FILE "ScriptFile"

#define RL_JSON_SCRIPT_AGENT_NAME "-agentName"
#define RL_JSON_SCRIPT_OPTIM_BATCH_SIZE "-optim_batchsize"
#define RL_JSON_SCRIPT_OPTIM_BATCH_SIZE_PER_AGENT "OptimBatchSizePerAgent"
#define RL_JSON_SCRIPT_LOG_DIR "-logdir"
#define RL_JSON_SCRIPT_NUM_TIMESTEPS "-num_timesteps"
#define RL_JSON_SCRIPT_OPTIM_EPOCHS "-optim_epochs"
#define RL_JSON_SCRIPT_TIMESTEPS_PER_BATCH "-timesteps_per_batch"
#define RL_JSON_SCRIPT_HID_SIZE "-hid_size"
#define RL_JSON_SCRIPT_RESUME "-resume"
#define RL_JSON_SCRIPT_PORTNUM "-portnum"
#define RL_JSON_SCRIPT_NUM_PARALLEL "-num_parallel"
#define RL_JSON_SCRIPT_OPTIM_STEPSIZE "-optim_stepsize"
#define RL_JSON_SCRIPT_NUM_HID_LAYERS "-num_hid_layers"
#define RL_JSON_SCRIPT_GAMMA "-gamma"
#define RL_JSON_SCRIPT_LAM "-lam"
#define RL_JSON_SCRIPT_SEED "-seed"
#define RL_JSON_SCRIPT_CLIP_PARAM "-clip_param"
#define RL_JSON_SCRIPT_ENTCOEFF "-entcoeff"
#define RL_JSON_SCRIPT_MOCAP_PATH "mocap_path"
#define RL_JSON_SCRIPT_RESUME_DISC "-resume_disc"
#define RL_JSON_SCRIPT_RESUME_NON_DISC "-resume_non_disc"
#define RL_JSON_SCRIPT_USE_DISTANCE "-use_distance"
#define RL_JSON_SCRIPT_USE_BLEND "-use_blend"
#define RL_JSON_SCRIPT_OPTIM_SCHEDULE "-optim_schedule"
#define RL_JSON_SCRIPT_DESIRED_KL "-desired_kl"

#define RL_JSON_SCRIPT_GAN_HID_SIZE "-gan_hid_size"
#define RL_JSON_SCRIPT_GAN_NUM_HID_LAYERS "-gan_num_hid_layers"
#define RL_JSON_SCRIPT_GAN_REWARD_SCALE "-gan_reward_scale"
#define RL_JSON_SCRIPT_GAN_LEARNING_RATE "-gan_learning_rate"
#define RL_JSON_SCRIPT_GAN_BATCH_SIZE "-gan_batch_size"
#define RL_JSON_SCRIPT_GAN_NUM_EPOCHS "-gan_num_epochs"
#define RL_JSON_SCRIPT_GAN_REPLAY_BUFFER_SIZE "-gan_replay_buffer_size"
#define RL_JSON_SCRIPT_GAN_PROB_TO_PUT_IN_REPLAY "-gan_prob_to_put_in_replay"
#define RL_JSON_SCRIPT_GAN_REWARD_TO_RETRAIN_DISCRIMINATOR "-gan_reward_to_retrain_discriminator"
#define RL_JSON_SCRIPT_GAN_QUEUE_SIZE "-gan_queue_size"

// Note: mostly deprecated as most of the parameters can be set for a Python script just by setting them in the scene's json file
class PPOLearningParams
{
public:
    PPOLearningParams()
    {
        agent_name = "Humanoid";
        timesteps_per_batch = 128;
        optim_batchsize_per_agent = 32;
        num_timesteps = 1000000000;
        optim_epochs = 10;
        optim_stepsize = 1e-3f;
        optim_schedule = "constant";
        desired_kl = 0.02f;

		hid_size = 128;
		num_hid_layers = 2;

		gamma = 0.99f;
		lam = 0.95f;

		seed = 7;
		resume = 0;

		clip_param = 0.2f;
		entcoeff = 0.0f;
		useGAN = false;

		// Viktor
		// Linux
		//    workingDir = "/home/viktor/DeepRL/NvBaselines/baselines/ppo1";
		//    pythonPath = "/home/viktor/anaconda3/envs/mujoco/bin/python";
		//    mocapPath = "/home/Isaac/Flex/flex/dev/rbd/data/bvh/motion";

		// Wndows
		workingDir = "C:/Deep_RL/baselines/baselines/ppo1";
		pythonPath = "C:/Users/vmakoviychuk/AppData/Local/Continuum/Anaconda3/python.exe";
		//mocapPath = "F:\NVIDIA\sw\devrel\libdev\flex\dev\rbd\data\bvh";

		// Nuttapong
		//workingDir = "c:/baselines_ex";
		//workingDir = "C:/safe/nv/new_baselines/NVBaselines";
		workingDir = "C:/new_baselines";
		//workingDir = "C:/x_b/NvBaselines";
		pythonPath = "c:/python/python.exe";
		//    mocapPath = "D:/p4sw/devrel/libdev/flex/dev/rbd/data/bvh/motion_simple";

		relativeLogDir = "log";

		gan_hid_size = 256;
		gan_num_hid_layers = 2;
		gan_learning_rate = 0.0001f;
		gan_reward_scale = 1.0f;

		resume_disc = 0;
		resume_non_disc = 0;

		gan_queue_size = 1;
		gan_batch_size = 128;
		gan_num_epochs = 1;
		gan_replay_buffer_size = 1000000;
		gan_prob_to_put_in_replay = 0.01f;
		gan_reward_to_retrain_discriminator = 5.0f;
		useDistance = false;
		useBlend = false;
	}

	static void FixPathSeparators(string& path)
	{
#if ISAAC_PLATFORM_WINDOWS
		constexpr char ReplacedSepartor = '/';
		constexpr char ReplacingSepartor = '\\';
#else
		constexpr char ReplacedSepartor = '\\';
		constexpr char ReplacingSepartor = '/';
#endif
		for (size_t ind = 0; ind < path.length(); ++ind)
		{
			if (path[ind] == ReplacedSepartor)
			{
				path[ind] = ReplacingSepartor;
			}
		}
	}

	void TryParseJson(const json& jsonParams)
	{
		if (jsonParams.is_null())
		{
			return;
		}

		pythonPath = jsonParams.value(RL_JSON_PYTHON_PATH, pythonPath);
		FixPathSeparators(pythonPath);

		workingDir = jsonParams.value(RL_JSON_SCRIPT_DIR, workingDir);
		FixPathSeparators(workingDir);

		mocapPath = jsonParams.value(RL_JSON_SCRIPT_MOCAP_PATH, mocapPath);
		FixPathSeparators(mocapPath);

		relativeLogDir = jsonParams.value(RL_JSON_SCRIPT_LOG_DIR, relativeLogDir);
		FixPathSeparators(relativeLogDir);

		agent_name = jsonParams.value(RL_JSON_SCRIPT_AGENT_NAME, agent_name);
		optim_batchsize_per_agent = jsonParams.value(RL_JSON_SCRIPT_OPTIM_BATCH_SIZE_PER_AGENT, optim_batchsize_per_agent);

		num_timesteps = jsonParams.value(RL_JSON_SCRIPT_NUM_TIMESTEPS, num_timesteps);
		optim_epochs = jsonParams.value(RL_JSON_SCRIPT_OPTIM_EPOCHS, optim_epochs);
		timesteps_per_batch = jsonParams.value(RL_JSON_SCRIPT_TIMESTEPS_PER_BATCH, timesteps_per_batch);
		hid_size = jsonParams.value(RL_JSON_SCRIPT_HID_SIZE, hid_size);
		optim_schedule = jsonParams.value(RL_JSON_SCRIPT_OPTIM_SCHEDULE, optim_schedule);
		optim_stepsize = jsonParams.value(RL_JSON_SCRIPT_OPTIM_STEPSIZE, optim_stepsize);
		desired_kl = jsonParams.value(RL_JSON_SCRIPT_DESIRED_KL, desired_kl);
		num_hid_layers = jsonParams.value(RL_JSON_SCRIPT_NUM_HID_LAYERS, num_hid_layers);
		resume = jsonParams.value(RL_JSON_SCRIPT_RESUME, resume);
		useDistance = jsonParams.value(RL_JSON_SCRIPT_USE_DISTANCE, useDistance);
		useBlend = jsonParams.value(RL_JSON_SCRIPT_USE_BLEND, useBlend);
		gamma = jsonParams.value(RL_JSON_SCRIPT_GAMMA, gamma);
		lam = jsonParams.value(RL_JSON_SCRIPT_LAM, lam);
		seed = jsonParams.value(RL_JSON_SCRIPT_SEED, seed);
		clip_param = jsonParams.value(RL_JSON_SCRIPT_CLIP_PARAM, clip_param);
		entcoeff = jsonParams.value(RL_JSON_SCRIPT_ENTCOEFF, entcoeff);
		resume_disc = jsonParams.value(RL_JSON_SCRIPT_RESUME_DISC, resume_disc);
		resume_non_disc = jsonParams.value(RL_JSON_SCRIPT_RESUME_NON_DISC, resume_non_disc);

		gan_hid_size = jsonParams.value(RL_JSON_SCRIPT_GAN_HID_SIZE, gan_hid_size);
		gan_num_hid_layers = jsonParams.value(RL_JSON_SCRIPT_GAN_NUM_HID_LAYERS, gan_num_hid_layers);
		gan_reward_scale = jsonParams.value(RL_JSON_SCRIPT_GAN_REWARD_SCALE, gan_reward_scale);
		gan_learning_rate = jsonParams.value(RL_JSON_SCRIPT_GAN_LEARNING_RATE, gan_learning_rate);
		gan_batch_size = jsonParams.value(RL_JSON_SCRIPT_GAN_BATCH_SIZE, gan_batch_size);
		gan_num_epochs = jsonParams.value(RL_JSON_SCRIPT_GAN_NUM_EPOCHS, gan_num_epochs);
		gan_replay_buffer_size = jsonParams.value(RL_JSON_SCRIPT_GAN_REPLAY_BUFFER_SIZE, gan_replay_buffer_size);
		gan_queue_size = jsonParams.value(RL_JSON_SCRIPT_GAN_QUEUE_SIZE, gan_queue_size);
		gan_prob_to_put_in_replay = jsonParams.value(RL_JSON_SCRIPT_GAN_PROB_TO_PUT_IN_REPLAY, gan_prob_to_put_in_replay);
		gan_reward_to_retrain_discriminator = jsonParams.value(RL_JSON_SCRIPT_GAN_REWARD_TO_RETRAIN_DISCRIMINATOR, gan_reward_to_retrain_discriminator);
	}

	bool useDistance;
	bool useGAN;
	bool useBlend;

	string agent_name;

	int timesteps_per_batch;
	int optim_batchsize_per_agent;
	int num_timesteps;
	int optim_epochs;
	float optim_stepsize;
	string optim_schedule;
	float desired_kl;

	int hid_size;
	int num_hid_layers;
	float gamma;
	float lam;

	int seed;
	int resume;

	float clip_param;
	float entcoeff;

	string workingDir;
	string pythonPath;
	string mocapPath;
	string relativeLogDir; // specified relative to the "workingDir"

	// GAN
	int gan_hid_size;
	int gan_num_hid_layers;
	float gan_learning_rate;
	float gan_reward_scale;
	int gan_batch_size;
	int gan_num_epochs;
	int gan_replay_buffer_size;
	float gan_prob_to_put_in_replay;
	float gan_reward_to_retrain_discriminator;
	int gan_queue_size;

	int resume_disc;   // Frame to resume discreminant related vars from
	int resume_non_disc;  // Frame to resume non-discreminant related vars from
};

class RLFlexEnv
{
public:
	RLFlexEnv();
	~RLFlexEnv();

	void LaunchPythonProcess(const char* python_exe, const char* working_dir, const char* logdir, PPOLearningParams& ppo_params, const json& params_json);
	void LaunchPythonProcess_Deprecated(const char* python_exe, const char* working_dir, const char* logdir, PPOLearningParams& ppo_params);
	void InitRLInfo();
	void HandleCommunication(); // Call this function for handling all communication
	virtual void ParseJsonParams(const json& sceneJson);

	// This should be called by Simulate function only
	float* GetAction(int a)
	{
		// Return ptr to action of agent a
		return &mCtls[a][0];
	}

	float* GetPrevAction(int a)
	{
		// Return ptr to previous action of agent a
		return &mPrevCtls[a][0];
	}

	// These functions will be called by the HandleCommunication
	virtual void ResetAllAgents()
	{
		for (int i = 0; i < mNumAgents; i++)
		{
			mAgentDie[i] = false;
			mFrames[i] = 0;
		}
	}
	virtual void ResetAgent(int a)
	{
		mAgentDie[a] = false;
		mFrames[a] = 0;
	}

	virtual void PreHandleCommunication() {} // Do whatever needed to be done before handling communication
	virtual void Simulate() = 0; // Do whatever needed with the action to transition to the next state

	virtual void PopulateState(int a, float* state) = 0; // Populate state for agent a
	virtual void PopulateExtra(int a, float* extra) {}; // Optional populate extra information
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) = 0; // Compute the reward for agent a, after taking the action and ends up in state state and also specify if it dies

	virtual void ClearContactInfo() = 0;

	virtual void LockWrite() = 0; // Do whatever needed to lock write to simulation
	virtual void UnlockWrite() = 0; // Do whatever needed to unlock write to simulation
	virtual void LoadState() {}; // Do whatever needed to restore state
	virtual void SaveState() {}; // Do whatever needed to save state

#ifdef NV_FLEX_GYM
	inline void SetActions(const float* actions, int firstAction, int numActions)
	{
		mCtls.swap(mPrevCtls);
		mCtls.resize(mNumAgents);
		for (int agent = 0; agent < mNumAgents; ++agent)
		{
			mCtls[agent].resize(mNumActions, 0.0f);
			for (int action = 0; action < mNumActions; ++action)
			{
				int index = agent * mNumActions + action;
				if (index >= firstAction && index < numActions) mCtls[agent][action] = actions[index - firstAction];
			}
		}
	}

	inline void SetPyToC(const float* actions)
	{
        for (int agent = 0; agent < mNumAgents; agent++)
        {
            for(int i = 0; i < mNumPyToC; i++){
                mPyToCBuf[agent * mNumPyToC + i] = actions[i];
            }
        }
	}

    virtual void PrepareHER(int x) {};
    
	inline int GetNumAgents() const { return mNumAgents; }
	inline int GetNumActions() const { return mNumActions; }
	inline int GetNumObservations() const { return mNumObservations; }
	inline int GetNumExtras() const { return mNumExtras; }
	inline int GetNumPyToC() const { return mNumPyToC; }    
	inline float* GetObservations() const { return mObsBuf; }
	inline float* GetRewards() const { return mRewBuf; }
	inline float* GetExtras() const { return mExtraBuf; }
    inline float* GetPyToC() const {return mPyToCBuf; }
    
	inline const std::uint8_t* GetDeaths() const { return mDieBuf; }
#endif

	virtual int GetRigidBodyIndex(int agentIndex, const wchar_t* bodyName) const { return -1; }
	virtual int GetRigidJointIndex(int agentIndex, const wchar_t* jointName) const { return -1; }

protected:
	int mNumAgents;
	int mNumActions;
	int mNumObservations;
	int mNumExtras;
    int mNumPyToC;
	int mMaxEpisodeLength;
	int mLearningStep;
	PPOLearningParams mPPOParams;

	std::vector<int> mAgentDie;
	std::vector<int> mFrames;
	vector<vector<float>> mCtls;
	vector<vector<float>> mPrevCtls;

	float powerScale; // Power scale

	int mPortNum;
	float* mObsBuf;
	float* mRewBuf;
	float* mExtraBuf;
    float* mPyToCBuf;
	std::uint8_t* mDieBuf;

#if USE_ISAAC_COMM
	IPCServerTask mServerTask;
#else
	// ZMQ
	zmq::context_t mContext;
	zmq::socket_t mSocket;
	PROCESS_INFORMATION mProcessInfo;
#endif
};



#pragma once
