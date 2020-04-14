#pragma once
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include "../../core/maths.h"
#include "../../external/tinyxml2/tinyxml2.h"
#include "../../external/rl/RLFlexEnv.h"
#include "../../external/json/json.hpp"

#include "../mjcf.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
#include <unistd.h>
#include <sys/stat.h>

#endif

#define PxPi kPi
#define GEN_INIT 0

using namespace std;
using namespace tinyxml2;
using json = nlohmann::json;

// List of JSON keys
#define RL_JSON_PARENT_RELATIVE_PATH "ParentRelativePath"
#define RL_JSON_SCENE_NAME "SceneName"

#define RL_JSON_LOAD_PATH "LoadPath"
#define RL_JSON_SOLVER_TYPE "SolverType"
#define RL_JSON_NUM_SUBSTEPS "NumSubsteps"
#define RL_JSON_NUM_ITERATIONS "NumIterations"
#define RL_JSON_NUM_INNER_ITERATIONS "NumInnerIterations"
#define RL_JSON_WARMSTART "WarmStart"
#define RL_JSON_NUM_POST_COLLISION_ITERATIONS "NumPostCollisionIterations"
#define RL_JSON_PAUSE "Pause"
#define RL_JSON_DO_LEARNING "DoLearning"
#define RL_JSON_DO_FLAGRUN "DoFlagrun"
#define RL_JSON_NUM_RENDER_STEPS "NumRenderSteps"
#define RL_JSON_NUM_PER_ROW "NumPerRow"
#define RL_JSON_SPACING "Spacing"
#define RL_JSON_ANGLE_RESET_NOISE "AngleResetNoise"
#define RL_JSON_ANGLE_VELOCITY_RESET_NOISE "AngleVelocityResetNoise"
#define RL_JSON_VELOCITY_RESET_NOISE "VelocityResetNoise"
#define RL_JSON_PUSH_FREQUENCY "PushFrequency"
#define RL_JSON_FORCE_MAGNITUDE "ForceMagnitude"
#define RL_JSON_USE_STATIC_REWS "UseStaticRewards"
#define RL_JSON_TERMINAION_Z "TerminationZ"

extern json g_sceneJson;

bool EnsureDirExists(const std::string& dirName_in)
{
	string dirName = dirName_in;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
	if (ftyp != INVALID_FILE_ATTRIBUTES)
	{
		if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		{
			cout << "Dir exists!" << endl;
			return true;   // this is a directory!
		}
	}
	for (size_t i = 0; i < dirName.size(); i++)
	{
		if (dirName[i] == '/')
		{
			dirName[i] = '\\';
		}
	}
#else
	struct stat sb;

	if (stat(dirName.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
	{
		return true;
	}
	for (size_t i = 0; i < dirName.size(); i++)
	{
		if (dirName[i] == '\\')
		{
			dirName[i] = '/';
		}
		}

#endif
	char cmd[1000];
	sprintf(cmd, "mkdir %s", dirName.c_str());
	system(cmd);
	return false;
	}

void GetCurrentDir(const int len, char* buf)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	GetCurrentDirectoryA(len, buf);
#else
	getcwd(buf, len);
#endif
}

template <class T>
T GetJsonVal(json jsonObj, string key, T defaultVal)
{
	if (jsonObj.find(key) != jsonObj.end())
	{
		return jsonObj.value(key, defaultVal);
	}
	return defaultVal;
}

class CommonFlexGymBase : public Scene, public RLFlexEnv
{
public:
	template <class T>
	void SaveNvFlexVector(FILE* f, NvFlexVector<T>& vec)
	{
		bool um = false;
		if (!vec.mappedPtr)
		{
			um = true;
			vec.map();
		}
		int num = (int)vec.size();
		fwrite(&num, sizeof(int), 1, f);
		if (num > 0)
		{
			fwrite(&vec[0], sizeof(T), num, f);
		}
		if (um)
		{
			vec.unmap();
		}
	}

	template <class T>
	void LoadNvFlexVector(FILE* f, NvFlexVector<T>& vec)
	{
		bool um = false;
		if (!vec.mappedPtr)
		{
			um = true;
			vec.map();
		}

		int num = 0;
		fread(&num, sizeof(int), 1, f);
		vec.resize(num);

		if (num > 0)
		{
			fread(&vec[0], sizeof(T), num, f);
		}
		if (um)
		{
			vec.unmap();
		}
	}
	template <class T>
	void SaveVec(FILE* f, vector<T>& vec)
	{
		int num = (int)vec.size();
		fwrite(&num, sizeof(int), 1, f);
		if (num > 0)
		{
			fwrite(&vec[0], sizeof(T), num, f);
		}
	}

	template <class T>
	void LoadVec(FILE* f, vector<T>& vec)
	{
		int num = 0;
		fread(&num, sizeof(int), 1, f);
		vec.resize(num);
		if (num > 0)
		{
			fread(&vec[0], sizeof(T), num, f);
		}
	}

	template <class T>
	void SaveVecVec(FILE* f, vector<vector<T> > & vec)
	{
		int numV = (int)vec.size();
		fwrite(&numV, sizeof(int), 1, f);
		if (numV > 0)
		{
			for (int i = 0; i < numV; i++)
			{
				int num = (int)vec[i].size();
				fwrite(&num, sizeof(int), 1, f);
				if (num > 0)
				{
					fwrite(&vec[i][0], sizeof(T), num, f);
				}
			}
		}
	}

	template <class T>
	void LoadVecVec(FILE* f, vector<vector<T> > & vec)
	{
		int numV = 0;
		fread(&numV, sizeof(int), 1, f);
		vec.resize(numV);
		if (numV > 0)
		{
			for (int i = 0; i < numV; i++)
			{
				int num = 0;
				fread(&num, sizeof(int), 1, f);
				vec[i].resize(num);
				if (num > 0)
				{
					fread(&vec[i][0], sizeof(T), num, f);
				}
			}
		}
	}

	void SaveFlexRigidBodyState(FILE* f)
	{
		// rigid bodies
		SaveNvFlexVector(f, g_buffers->rigidBodies);
		SaveNvFlexVector(f, g_buffers->rigidJoints);
	}

	void LoadFlexRigidBodyState(FILE* f)
	{
		LoadNvFlexVector(f, g_buffers->rigidBodies);
		LoadNvFlexVector(f, g_buffers->rigidJoints);
	}

	virtual void LoadState()
	{
		char fname[500];
		sprintf(fname, "%s/%s/%s_%08d.dat", mPPOParams.workingDir.c_str(), mPPOParams.relativeLogDir.c_str(), mPPOParams.agent_name.c_str(), mPPOParams.resume);
		printf("Load from %s\n", fname);
		FILE* f = fopen(fname, "rb");
		if (!f) return;
		printf("Load rb\n");
		LoadFlexRigidBodyState(f);
		printf("Load rl\n");
		LoadRLState(f);
		fclose(f);
	}
	virtual void SaveState()
	{
		// Only save every 10 frames
		if (mLearningStep % 10 == 0)
		{
			char fname[500];
			int of = mLearningStep - 50;
			if ((of % 500 != 0) && (of > mPPOParams.resume))
			{
				sprintf(fname, "%s/%s/%s_%08d.dat", mPPOParams.workingDir.c_str(), mPPOParams.relativeLogDir.c_str(), mPPOParams.agent_name.c_str(), of);
				remove(fname);
			}
			sprintf(fname, "%s/%s/%s_%08d.dat", mPPOParams.workingDir.c_str(), mPPOParams.relativeLogDir.c_str(), mPPOParams.agent_name.c_str(), mLearningStep);
			printf("Save to %s\n", fname);

			FILE* f = fopen(fname, "wb");
			SaveFlexRigidBodyState(f);
			SaveRLState(f);
			fclose(f);
		}
	}

	virtual void LoadRLState(FILE* f)
	{
		LoadVec(f, mAgentDie);
		LoadVec(f, mFrames);
		LoadVecVec(f, mCtls);
	}

	virtual void SaveRLState(FILE* f)
	{
		SaveVec(f, mAgentDie);
		SaveVec(f, mFrames);
		SaveVecVec(f, mCtls);
	}
};

class FlexGymBase : public CommonFlexGymBase
{
public:

	FlexGymBase() : rigidContacts(g_flexLib, g_solverDesc.maxRigidBodyContacts), rigidContactCount(g_flexLib, 1)
	{
		loadPath = "../../data/humanoid_symmetric.xml";
		mNumAgents = 500;
		mNumObservations = 44;
		mNumActions = 17;
		mMaxEpisodeLength = 1000;

		g_numSubsteps = 2;
		g_params.numIterations = 30;

		g_sceneLower = Vec3(-50.f, 0.f, -50.f);
		g_sceneUpper = Vec3(150.f, 50.f, 100.f);

		g_pause = true;
		mDoLearning = g_doLearning;

		numRenderSteps = 1;

		maxContactsPerAgent = 64;
		g_solverDesc.maxRigidBodyContacts = maxContactsPerAgent * mNumAgents;

		numPerRow = 24;
		spacing = 20.f;

		powerScale = 0.4f;
		maxPower = 300.f;

		angleResetNoise = 0.f;
		angleVelResetNoise = 0.f;
		velResetNoise = 0.f;

		pushFrequency = 500;	// How much steps in average per 1 kick
		forceMag = 0.0f;
		yOffset = 0.0f;
		preTransform = Transform(Vec3(), Quat());
		rotCreateNoise = 0.05f;
	}

	Transform preTransform;
	string loadPath;
	vector<shared_ptr<MJCFImporter>> mjcfs;

	vector<pair<int, int> > agentBodies;
	vector<NvFlexRigidBody> initBodies;
	vector<NvFlexRigidJoint> initJoints;
	vector<float> masses;

	// Transforms from initial
	vector<Transform> agentOffsetInv;
	vector<Transform> agentOffset;

	// Start transforms with randomization
	vector<Transform> agentStartOffset;

	vector<vector<pair<int, NvFlexRigidJointAxis>>> ctrls;
	vector<vector<float>> motorPower;
	float powerScale;
	float maxPower;

	vector<int> torso;

	NvFlexVector<NvFlexRigidContact> rigidContacts;
	NvFlexVector<int> rigidContactCount;

	vector<vector<Transform>> initTrans; // Do we need them?

	vector<pair<int, int>> robotJoints;
	vector<pair<int, int>> robotShapes;

	bool mDoLearning;
	int numRenderSteps;

	int maxContactsPerAgent;

	// Randomization
	float angleResetNoise;
	float angleVelResetNoise;
	float velResetNoise;
	float rotCreateNoise;

	int pushFrequency;	// How much steps in average per 1 kick
	float forceMag;

	int numPerRow;
	float spacing;
	float yOffset;

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower) = 0;

	void ParseJsonParams(const json& sceneJson) override
	{
		if (sceneJson.is_null())
		{
			return;
		}
		RLFlexEnv::ParseJsonParams(sceneJson);

		// Parsing of common JSON parameters
		loadPath = sceneJson.value(RL_JSON_LOAD_PATH, loadPath);

		g_params.solverType = sceneJson.value(RL_JSON_SOLVER_TYPE, g_params.solverType);
		g_numSubsteps = sceneJson.value(RL_JSON_NUM_SUBSTEPS, g_numSubsteps);
		g_params.numIterations = sceneJson.value(RL_JSON_NUM_ITERATIONS, g_params.numIterations);
		g_params.numInnerIterations = sceneJson.value(RL_JSON_NUM_INNER_ITERATIONS, g_params.numInnerIterations);
		g_params.warmStart = sceneJson.value(RL_JSON_WARMSTART, g_params.warmStart);
		// SOR
	//	g_params.relaxationFactor = sceneJson.value(RL_JSON_RELAXATION_FACTOR, g_params.relaxationFactor);
		g_params.numPostCollisionIterations = sceneJson.value(RL_JSON_NUM_POST_COLLISION_ITERATIONS, g_params.numPostCollisionIterations);
		g_pause = sceneJson.value(RL_JSON_PAUSE, g_pause);

		mDoLearning = sceneJson.value(RL_JSON_DO_LEARNING, mDoLearning);
		numRenderSteps = sceneJson.value(RL_JSON_NUM_RENDER_STEPS, numRenderSteps);

		numPerRow = sceneJson.value(RL_JSON_NUM_PER_ROW, numPerRow);
		spacing = sceneJson.value(RL_JSON_SPACING, spacing);

		angleResetNoise = sceneJson.value(RL_JSON_ANGLE_RESET_NOISE, angleResetNoise);
		angleVelResetNoise = sceneJson.value(RL_JSON_ANGLE_VELOCITY_RESET_NOISE, angleVelResetNoise);
		velResetNoise = sceneJson.value(RL_JSON_VELOCITY_RESET_NOISE, velResetNoise);

		pushFrequency = sceneJson.value(RL_JSON_PUSH_FREQUENCY, pushFrequency);
		forceMag = sceneJson.value(RL_JSON_FORCE_MAGNITUDE, forceMag);
	}

	void PrepareScene() override
	{	
		ParseJsonParams(g_sceneJson);
	}

	// Todo abstract?
	virtual void LoadEnv()
	{
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		torso.clear();
		torso.resize(mNumAgents, -1);

		masses.clear();

		for (int i = 0; i < mNumAgents; i++)
		{
			Vec3 pos = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f);

			Vec3 posStart = Vec3((i % numPerRow) * spacing, fabs(yOffset + Randf(-0.02f, 0.02f)), (i / numPerRow) * spacing);
			Quat rotStart = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));

			rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * rot;
			rotStart = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise))
				* rotStart;

			Transform gtStart(posStart, rotStart);
			gtStart = gtStart * preTransform;

			Transform gt(pos, rot);
			gt = gt * preTransform;

			int begin = g_buffers->rigidBodies.size();

			AddAgentBodiesJointsCtlsPowers(i, gtStart, ctrls[i], motorPower[i]);

			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
			agentStartOffset.push_back(gtStart);

			int end = g_buffers->rigidBodies.size();
			agentBodies.push_back(make_pair(begin, end));
		}

		for (int bi = agentBodies[0].first; bi < agentBodies[0].second; ++bi)
		{
			masses.push_back(g_buffers->rigidBodies[bi].mass);
		}

		maxPower = *max_element(std::begin(motorPower[0]), std::end(motorPower[0]));

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());
	}

	virtual void PreSimulation()
	{		
		if (!mDoLearning)
		{
			if (!g_pause || g_step)
			{
				for (int s = 0; s < numRenderSteps; s++)
				{
					// tick solver
					NvFlexSetParams(g_solver, &g_params);
					NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
				}

				g_frame++;
				g_step = false;
			}
		}
		else
		{
			NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
			g_buffers->rigidBodies.map();
			NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
			g_buffers->rigidJoints.map();

			for (int s = 0; s < numRenderSteps; s++)
			{
#ifdef NV_FLEX_GYM
				Simulate();
				FinalizeContactInfo();
				for (int a = 0; a < mNumAgents; ++a)
				{
					PopulateState(a, &mObsBuf[a * mNumObservations]);
					if (mNumExtras > 0) PopulateExtra(a, &mExtraBuf[a * mNumExtras]);
					ComputeRewardAndDead(a, GetAction(a), &mObsBuf[a * mNumObservations], mRewBuf[a], (bool&)mDieBuf[a]);
				}
#else
				HandleCommunication();
#endif
				ClearContactInfo();
			}

			g_buffers->rigidJoints.unmap();
			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!		
		}
	}

	virtual void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
	}

	virtual void ApplyTorqueControl()
	{
		// Random push to torso during training
		int push_ai = Rand(0, pushFrequency - 1);

		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second; i++)
			{
				g_buffers->rigidBodies[i].force[0] = 0.0f;
				g_buffers->rigidBodies[i].force[1] = 0.0f;
				g_buffers->rigidBodies[i].force[2] = 0.0f;
				g_buffers->rigidBodies[i].torque[0] = 0.0f;
				g_buffers->rigidBodies[i].torque[1] = 0.0f;
				g_buffers->rigidBodies[i].torque[2] = 0.0f;
			}

			float* actions = GetAction(ai);
			for (int i = 0; (unsigned int)i < ctrls[ai].size(); i++)
			{
				float cc = Clamp(actions[i], -1.f, 1.f);

				NvFlexRigidJoint& j = initJoints[ctrls[ai][i].first];
				NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
				NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
				Transform& pose0 = *((Transform*)&j.pose0);
				Transform gpose;
				NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
				Transform tran = gpose * pose0;

				Vec3 axis;
				if (ctrls[ai][i].second == eNvFlexRigidJointAxisTwist)
				{
					axis = GetBasisVector0(tran.q);
				} 
				else
				if (ctrls[ai][i].second == eNvFlexRigidJointAxisSwing1)
				{
					axis = GetBasisVector1(tran.q);
				} 
				else
				if (ctrls[ai][i].second == eNvFlexRigidJointAxisSwing2)
				{
					axis = GetBasisVector2(tran.q);
				} 
				else {
					printf("Invalid axis, probably bad code migration?\n");
					exit(0);
				}

				Vec3 torque = axis * motorPower[ai][i] * cc * powerScale;
				a0.torque[0] += torque.x;
				a0.torque[1] += torque.y;
				a0.torque[2] += torque.z;
				a1.torque[0] -= torque.x;
				a1.torque[1] -= torque.y;
				a1.torque[2] -= torque.z;
			}
		}
	}

	virtual void ApplyRandomPertubations()
	{
		// Random push to torso during training
		int push_ai = Rand(0, pushFrequency - 1);

		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			if (ai % pushFrequency == push_ai && torso[ai] != -1)
			{
				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[ai]], (NvFlexRigidPose*)&torsoPose);
				float z = torsoPose.p.y;
				Vec3 pushForce = forceMag * RandomUnitVector();
				if (z > 1.f)
				{
					pushForce.z *= 0.2f;
				}
				else
				{
					pushForce.x *= 0.2f;
					pushForce.y *= 0.2f;
					pushForce.z *= 0.2f;
				}
				g_buffers->rigidBodies[torso[ai]].force[0] += pushForce.x;
				g_buffers->rigidBodies[torso[ai]].force[1] += pushForce.y;
				g_buffers->rigidBodies[torso[ai]].force[2] += pushForce.z;
			}
		}
	}

	virtual void UpdateSolver()
	{
		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		g_frame++;
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
		g_buffers->rigidBodies.map();
	}

	virtual void Simulate()
	{
		ApplyTorqueControl();
		ApplyRandomPertubations();
		UpdateSolver();
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual bool IsSkipSimulation()
	{
		return true;
	}

	void GetAngles(int a, float* angles, float* lows, float* highs)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = 0.f;
			float low = 0.f;
			float high = 0.f;
			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
				Transform body1Pose;
				NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

				Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
				Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
				Transform relPose = Inverse(pose0) * pose1;

				prevPos = relPose.p;
				Quat qd = relPose.q;
				if (qd.w < 0)
				{
					qd *= -1.f;
				}

				Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
				Quat qswing = qd*Inverse(qtwist);
				prevTwist = asin(qtwist.x) * 2.f;
				prevSwing1 = asin(qswing.y) * 2.f;
				prevSwing2 = asin(qswing.z) * 2.f;
				prevIdx = ctrls[a][qq].first;

				// If same, no need to recompute
			}

			NvFlexRigidJointAxis idx = ctrls[a][qq].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];
			if (idx == eNvFlexRigidJointAxisTwist)
			{
				pos = prevTwist;
			}
			else if (idx == eNvFlexRigidJointAxisSwing1)
			{
				pos = prevSwing1;
			}
			else if (idx == eNvFlexRigidJointAxisSwing2)
			{
				pos = prevSwing2;
			}
			else if (idx == eNvFlexRigidJointAxisX)
			{
				pos = prevPos.x;
			}
			else if (idx == eNvFlexRigidJointAxisY)
			{
				pos = prevPos.y;
			}
			else if (idx == eNvFlexRigidJointAxisZ)
			{
				pos = prevPos.z;
			}

			angles[i] = pos;
			lows[i] = low;
			highs[i] = high;
		}
	}

	virtual void GetGlobalPose(int a, Transform* trans)
	{
		Transform& inv = agentOffsetInv[a];
		pair<int, int> p = agentBodies[a];

		int ind = 0;
		for (int i = p.first; i < p.second; i++)
		{
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&pose);
			trans[ind++] = inv * pose;
		}
	}

	virtual Vec3 GetLinearVel(int a, int index)
	{
		Transform& inv = agentOffsetInv[a];
		pair<int, int> p = agentBodies[a];

		return Rotate(inv.q, Vec3(g_buffers->rigidBodies[p.first + index].linearVel));
	}

	virtual Vec3 GetAngularVel(int a, int index)
	{
		Transform& inv = agentOffsetInv[a];
		pair<int, int> p = agentBodies[a];

		return Rotate(inv.q, Vec3(g_buffers->rigidBodies[p.first + index].angularVel));
	}

	int GetNumBodies()
	{
		return agentBodies[0].second - agentBodies[0].first;
	}

	int GetNumControls()
	{
		return ctrls[0].size();
	}

	virtual void FinalizeContactInfo() = 0;
	virtual void ClearContactInfo() = 0;
};


template <class CTransform, class CVec3, class CQuat, class CMat33>
class RLWalkerEnv : public FlexGymBase
{
public:
	static const int MaxActions = 100; // Required for populate state, increase if number of actions >= 100

	int numFeet;
	float electricityCostScale;
	float electricityCost;		//    # cost for using motors-- this parameter should be carefully tuned against reward for making progress, other values less improtant
	float stallTorqueCostScale;
	float stallTorqueCost;		//    # cost for running electric current through a motor even at zero rotational speed, small
	float footCollisionCost;	//    # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself

	float jointsAtLimitCost;	// # discourage stuck joints
	float initialZ;
	float terminationZ;

	float pitch, r, yaw; // Should be local in ExtractState()!

	float maxX, maxY; // Max size for randomply generated flag

	int maxFlagResetSteps;
	
	bool useStaticRews;

	vector<vector<float>> bPrevAngleValue;

	vector<float> potentialsOld;
	vector<float> potentials;
	
	vector<int> pelvis;
	vector<int> head;
	vector<int> hands;
	vector<int> feet;
	
	vector<int> footFlag;
	vector<int> handFlag;
	
	vector<float> handsContact;
	vector<float> feetContact;
	vector<int> numCollideOther;

	vector<float> ps;
	vector<float> walkTargetDists;

	vector<vector<float>> jointSpeeds;
	vector<int> jointsAtLimits;
	
	vector<float> headings;
	vector<float> upVecs;
	
	vector<float> walkTargetX;
	vector<float> walkTargetY;

	vector<CVec3> bodyXYZ;
	vector<Transform> bodyPoses;
	
	vector<int> flagRunSteps;
	vector<int> aliveLengths;

	int aliveLengthThreshold, colorChangeThreshold; // used for changing color of agents as alive length increase
	Vec3 redColor = Vec3(1.0f, 0.04f, 0.07f);
	Vec3 originalColor;

	bool doFlagRun;
	float dt;
	float upVecWeight;

	RLWalkerEnv() : numFeet(2),
		electricityCostScale(2.f), 
		electricityCost(-2.f),
		stallTorqueCostScale(5.f), 
		stallTorqueCost(-0.1f),
		footCollisionCost(-1.f), 
		jointsAtLimitCost(-0.2f),
		initialZ(0.8f), 
		terminationZ(0.79f),
		maxX(200.f), maxY(200.f),
		maxFlagResetSteps(200),
		doFlagRun(false),
		dt(1.f / 60.f), 
		upVecWeight(0.05f)
	{
		powerScale = 0.405f;
	};

	// Need to implement these functions
	virtual float AliveBonus(float z, float pitch) = 0;

	virtual void SaveRLState(FILE* f)
	{
		FlexGymBase::SaveRLState(f);

		SaveVecVec(f, bPrevAngleValue);

		SaveVec(f, potentialsOld);
		SaveVec(f, potentials);

		SaveVec(f, pelvis);
		SaveVec(f, head);
		SaveVec(f, hands);
		SaveVec(f, feet);

		SaveVec(f, footFlag);
		SaveVec(f, handFlag);

		SaveVec(f, handsContact);
		SaveVec(f, feetContact);
		SaveVec(f, numCollideOther);

		SaveVec(f, ps);
		SaveVec(f, walkTargetDists);

		SaveVecVec(f, jointSpeeds);
		SaveVec(f, jointsAtLimits);

		SaveVec(f, headings);
		SaveVec(f, upVecs);

		SaveVec(f, walkTargetX);
		SaveVec(f, walkTargetY);

		SaveVec(f, bodyXYZ);

		SaveVec(f, flagRunSteps);
	}

	virtual void LoadRLState(FILE* f)
	{
		FlexGymBase::LoadRLState(f);

		LoadVecVec(f, bPrevAngleValue);

		LoadVec(f, potentialsOld);
		LoadVec(f, potentials);

		LoadVec(f, pelvis);
		LoadVec(f, head);
		LoadVec(f, hands);
		LoadVec(f, feet);

		LoadVec(f, footFlag);
		LoadVec(f, handFlag);

		LoadVec(f, handsContact);
		LoadVec(f, feetContact);
		LoadVec(f, numCollideOther);

		LoadVec(f, ps);
		LoadVec(f, walkTargetDists);
	
		LoadVecVec(f, jointSpeeds);
		LoadVec(f, jointsAtLimits);
		LoadVec(f, headings);
		LoadVec(f, upVecs);
		LoadVec(f, walkTargetX);
		LoadVec(f, walkTargetY);
		LoadVec(f, bodyXYZ);
		LoadVec(f, flagRunSteps);
	}

	void ParseJsonParams(const json& sceneJson) override
	{
		if (sceneJson.is_null())
		{
			return;
		}
		FlexGymBase::ParseJsonParams(sceneJson);
		doFlagRun = g_sceneJson.value(RL_JSON_DO_FLAGRUN, doFlagRun);
		useStaticRews = g_sceneJson.value(RL_JSON_USE_STATIC_REWS, true);
		terminationZ = g_sceneJson.value(RL_JSON_TERMINAION_Z, 0.5f);
		aliveLengthThreshold = g_sceneJson.value("AliveLengthThreshold", 0);
		colorChangeThreshold = int(0.8f * aliveLengthThreshold);
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		mjcfs.push_back(make_shared<MJCFImporter>(loadPath.c_str()));
		int startJoint = g_buffers->rigidJoints.size();
		int startShape = g_buffers->rigidShapes.size();
		mjcfs.back()->AddPhysicsEntities(gt, ctrl, mpower);
		int endJoint = g_buffers->rigidJoints.size();
		int endShape = g_buffers->rigidShapes.size();
		robotJoints.push_back({ startJoint, endJoint });
		robotShapes.push_back({ startShape, endShape });
	
		torso.push_back(mjcfs[i]->bmap["torso"]);
		pelvis.push_back(mjcfs[i]->bmap["pelvis"]);
		head.push_back(mjcfs[i]->bmap["head"]);
	
		feet.push_back(mjcfs[i]->bmap["right_foot"]);
		feet.push_back(mjcfs[i]->bmap["left_foot"]);
	}

	virtual void resetTarget(int a, bool firstTime = true)
	{
		if (doFlagRun)
		{
			if (firstTime && (a % 5))
			{
				walkTargetX[a] = 1000.f;
				walkTargetY[a] = 0.f;
			}
			else
			{
				walkTargetX[a] = Randf(-maxX, maxX) + bodyXYZ[a].x;
				walkTargetY[a] = Randf(-maxY, maxY) + bodyXYZ[a].y;
			}

			flagRunSteps[a] = 0;
		}
		else
		{
			walkTargetX[a] = 1000.f;
			walkTargetY[a] = 0.f;
		}
	}

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		// Prepare state
		//--------------------
		numJointsAtLimit = 0;

		float joints[MaxActions * 2];
		float angles[MaxActions];
		float lows[MaxActions];
		float highs[MaxActions];

		int numBodies = GetNumBodies();

		GetAngles(a, angles, lows, highs);
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = angles[i], vel;
			float low = lows[i], high = highs[i];

			if (bPrevAngleValue[a][qq] > 1e8)
			{
				bPrevAngleValue[a][qq] = pos;
			}

			float posDiff = pos - bPrevAngleValue[a][qq];
			if (fabs(posDiff) < kPi)
			{
				vel = posDiff / dt;
			}
			else
			{
				vel = (-(Sign(pos) * 2.f * kPi) + posDiff) / dt;
			}
			bPrevAngleValue[a][qq] = pos;

			float posMid = 0.5f * (low + high);
			pos = 2.f * (pos - posMid) / (high - low);

			joints[2 * i] = pos;
			joints[2 * i + 1] = vel * 0.1f;

			jointSpeeds[i] = joints[2 * i + 1];
			if (fabs(joints[2 * i]) > 0.99f)
			{
				numJointsAtLimit++;
			}
		}
		Transform bodies[100];

		GetGlobalPose(a, bodies);
		Transform bodyPose = bodies[0];

		upVec = GetBasisVector2(bodyPose.q).z;
		Vec3 pos(0.f);
		float totalMass = 0.f;
		for (int i = 0; i < numBodies; i++)
		{
			float bodyMass = masses[i];

			pos += bodies[i].p * bodyMass;
			totalMass += bodyMass;
		}
		bodyXYZ[a] = Vec3(pos.x / totalMass, pos.y / totalMass, bodyPose.p.z);

		getEulerZYX(bodyPose.q, yaw, p, r);
		float z = bodyXYZ[a].z;

		if (initialZ > 1e9)
		{
			initialZ = z;
		}

		Vec3 toTarget = Vec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
		walkTargetDist = Length(toTarget);
		if (doFlagRun)
		{
			if (flagRunSteps[a] > maxFlagResetSteps || walkTargetDist < 1.f)
			{
				resetTarget(a, false);
				toTarget = Vec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
				walkTargetDist = Length(toTarget);

				potentialsOld[a] = -walkTargetDist / dt;
				potentials[a] = -walkTargetDist / dt;
			}
			flagRunSteps[a]++;
		}

		float walkTargetTheta = atan2(walkTargetY[a] - bodyXYZ[a].y, walkTargetX[a] - bodyXYZ[a].x);

		toTarget = Normalize(toTarget);
		heading = Dot(GetBasisVector0(bodyPose.q), toTarget);
		float angleToTarget = walkTargetTheta - yaw;

		Matrix33 mat = Matrix33(
						   Vec3(cos(-yaw), sin(-yaw), 0.0f),
						   Vec3(-sin(-yaw), cos(-yaw), 0.0f),
						   Vec3(0.0f, 0.0f, 1.0f));

		Vec3 vel = GetLinearVel(a, 0);
		Vec3 bvel = mat * vel;
		float vx = bvel.x;
		float vy = bvel.y;
		float vz = bvel.z;

		Vec3 angVel = GetAngularVel(a, 0);
		Vec3 bangVel = mat * angVel;
		float avx = bangVel.x;
		float avy = bangVel.y;
		float avz = bangVel.z;

		float more[11] = { z - initialZ,
						   sin(angleToTarget), cos(angleToTarget),
						   0.25f * vx, 0.25f * vy, 0.25f * vz,
						   0.25f * avx, 0.25f * avy, 0.25f * avz,
						   r, p
						 };

		int ct = 0;
		for (int i = 0; i < 11; ++i)
		{
			state[ct++] = more[i];
		}

		for (int i = 0; (unsigned int)i < ctrls[a].size() * 2; ++i)
		{
			state[ct++] = joints[i];
		}

		for (int i = 0; i < numFeet; i++)
		{
			state[ct++] = feetContact[numFeet * a + i];
		}

		float* prevActions = GetAction(a);
		for (int i = 0; i < mNumActions; ++i)
		{
			state[ct++] = prevActions[i];
		}

		for (int i = 0; i < ct; ++i)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}
	}

	void init(PPOLearningParams& ppoParams,
			  const char* pythonFile,
			  const char* workingDir,
			  const char* logDir,
			  float deltaT = 1.f / 60.f)
	{
		if (logDir)
		{
			EnsureDirExists(workingDir + string("/") + logDir);
		}

		dt = deltaT;

		InitRLInfo();
		LaunchPythonProcess(pythonFile, workingDir, logDir, ppoParams, g_sceneJson);

		potentialsOld.resize(mNumAgents);
		potentials.resize(mNumAgents);

		ps.resize(mNumAgents);
		walkTargetDists.resize(mNumAgents);
		jointSpeeds.resize(mNumAgents);
		jointsAtLimits.resize(mNumAgents);
		headings.resize(mNumAgents);
		upVecs.resize(mNumAgents);
		walkTargetX.resize(mNumAgents);
		walkTargetY.resize(mNumAgents);
		bodyXYZ.resize(mNumAgents);
		bodyPoses.resize(mNumAgents);

		feetContact.resize(numFeet * mNumAgents);
		numCollideOther.resize(mNumAgents);
		bPrevAngleValue.resize(mNumAgents);

		flagRunSteps.clear();
		flagRunSteps.resize(mNumAgents, 0);

		aliveLengths.resize(mNumAgents, 0);
		
		// TODO(jaliang): Get initial color from mjcf instead of hardcoding!
		originalColor = Vec3(0.97f, 0.38f, 0.06);

		for (int a = 0; a < mNumAgents; a++)
		{
			jointSpeeds[a].resize(mNumActions);
			potentialsOld[a] = 0.0f;
			potentials[a] = 0.0f;
			bodyXYZ[a] = CVec3(0.f, 0.f, 0.f);
			resetTarget(a, true);
			for (int fi = 0; fi < numFeet; ++fi)
			{
				feetContact[numFeet * a + fi] = 0.f;
			}

			numCollideOther[a] = 0;
			bPrevAngleValue[a].resize(mNumActions);
		}
	}

	void getEulerZYX(CQuat& q, float& yawZ, float& pitchY, float& rollX)
	{
		float squ;
		float sqx;
		float sqy;
		float sqz;
		float sarg;
		sqx = q.x * q.x;
		sqy = q.y * q.y;
		sqz = q.z * q.z;
		squ = q.w * q.w;

		rollX = atan2(2 * (q.y * q.z + q.w * q.x), squ - sqx - sqy + sqz);
		sarg = (-2.0f) * (q.x * q.z - q.w * q.y);
		pitchY = sarg <= (-1.0f) ? (-0.5f) * PxPi : (sarg >= (1.0f) ? (0.5f) * PxPi : asinf(sarg));
		yawZ = atan2(2 * (q.x * q.y + q.w * q.z), squ + sqx - sqy - sqz);
	}

	virtual void PopulateState(int a, float* state)
	{
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& jointsAtLimitA = jointsAtLimits[a];
		float& heading = headings[a];
		float& upVec = upVecs[a];

		ExtractState(a, state, p, walkTargetDist, jointSpeedsA, jointsAtLimitA, heading, upVec);
	}

	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float& potential = potentials[a];
		float& potentialOld = potentialsOld[a];
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& numJointsAtLimitA = jointsAtLimits[a];
		float& heading = headings[a];
		float& upVec = upVecs[a];

		float electrCost = electricityCostScale * electricityCost;
		float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

		float alive = AliveBonus(state[0] + initialZ, p); //  # state[0] is body height above ground, body_rpy[1] is pitch
		dead = alive < 0.f;

		potentialOld = potential;
		potential = -walkTargetDist / dt;
		if (potentialOld > 1e9)
		{
			potentialOld = potential;
		}

		float progress = potential - potentialOld;
		if (progress > 100.f)
		{
			printf("progress is infinite %d %f %f %f \n", a, progress, potential, potentialOld);
			progress = 0.f;
		}

		float electricityCostCurrent = 0.0f;
		float sum = 0.0f;
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			float vv = abs(action[i] * jointSpeedsA[i]) * motorPower[a][i] / maxPower; // Take motor power into account
			if (!isfinite(vv))
			{
				printf("vv at %d is infinite, vv = %lf, ctl = %lf, js = %lf reset agent\n", i, vv, action[i], jointSpeedsA[i]);
				alive = -1.f;
			}

			if (!isfinite(action[i]))
			{
				printf("action at %d is infinite, reset agent\n", i);
				alive = -1.f;
			}

			if (!isfinite(jointSpeedsA[i]))
			{
				printf("jointSpeeds at %d is infinite, reset agent\n", i);
				alive = -1.f;
			}

			sum += vv;
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl*joint_speed is infinite, reset agent!\n");
			alive = -1.f;
		}

		electricityCostCurrent += electrCost * sum / (float)ctrls[a].size();

		sum = 0.0f;
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			sum += action[i] * action[i];
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");		
			alive = -1.f;		
		}		

		//electricity_costCurrent += stall_torque_cost * float(np.square(a).mean())
		electricityCostCurrent += stallTorqCost * sum / (float)ctrls[a].size();

		float jointsAtLimitCostCurrent = jointsAtLimitCost * (float)numJointsAtLimitA;

		float feetCollisionCostCurrent = 0.0f;
		if (numCollideOther[a] > 0)
		{
			feetCollisionCostCurrent += footCollisionCost;
		}

		//	cout << "heading = " << heading << endl;
		float headingRew = 0.5f * ((heading > 0.8f) ? 1.f : heading / 0.8f) + upVecWeight * ((upVec > 0.93f) ? 1.f : 0.f); // MJCF4
		if (!useStaticRews)
		{
			headingRew = 0.f;
			alive = 0.f;
		}

		float rewards[6] =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent,
			headingRew
		};
		// cout <<"rewards: "<< rewards[0]<<" "<< rewards[1]<<" "<< rewards[2]<<" "<< rewards[3]<<" "<<rewards[4]<<" "<<rewards[5]<<" "<<rewards[6]<<endl;

		rew = 0.f;
		for (int i = 0; i < 6; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}
	}

	virtual void ResetAllAgents()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			ResetAgent(a);
		}
		RLFlexEnv::ResetAllAgents(); // Duplicate, fix!

		// Load state
		if (mPPOParams.resume != 0)
		{
			LoadState();
		}
	}

	virtual void ResetAgent(int a)
	{
		potentialsOld[a] = 1e10f;
		potentials[a] = 1e10f;

		for (int i = 0; i < (int)bPrevAngleValue[a].size(); i++)
		{
			bPrevAngleValue[a][i] = 1e9f;
		}

		resetTarget(a, true);

		for (int fi = 0; fi < numFeet; ++fi)
		{
			feetContact[numFeet * a + fi] = 0.f;
		}

		numCollideOther[a] = 0;

		aliveLengths[a] = 0;

		RLFlexEnv::ResetAgent(a);
	}

	virtual void ClearContactInfo()
	{
		for (auto &fc : feetContact)
		{
			fc = 0.f;
		}

		for (auto &nc : numCollideOther)
		{
			nc = 0;
		}
	}

	virtual void PreHandleCommunication()
	{
		FinalizeContactInfo();
	}

	virtual void UpdateCOM()
	{
		// Update robot COM
		Transform bodies[50];
		int numBodies = GetNumBodies();
		for (int a = 0; a < mNumAgents; a++)
		{
			GetGlobalPose(a, bodies);
			bodyPoses[a] = bodies[0];
			Vec3 pos(0.f);
			float totalMass = 0.f;
			for (int i = 0; i < numBodies; i++)
			{
				float bodyMass = masses[i];

				pos += bodies[i].p * bodyMass;
				totalMass += bodyMass;
			}
			bodyXYZ[a] = Vec3(pos.x / totalMass, pos.y / totalMass, bodyPoses[a].p.z);
		}
	}

	void SetAgentColor(int a, Vec3 color)
	{
		int renderMaterial = AddRenderMaterial(color, 0.3f, 0.4f);
		for (int i = robotShapes[a].first; i < robotShapes[a].second; i++)
		{
			g_buffers->rigidShapes[i].user = UnionCast<void*>(renderMaterial);
		}
	}

	virtual void DoGui()
	{
		if (mDoLearning && aliveLengthThreshold > 0)
		{
			for (int a = 0; a < mNumAgents; a++)
			{
				if (aliveLengths[a] == 0)
				{
					SetAgentColor(a, originalColor);
				}
				else if (aliveLengths[a] > colorChangeThreshold) 
				{
					float x = float(aliveLengths[a] - colorChangeThreshold) / float(aliveLengthThreshold - colorChangeThreshold);
					Vec3 color = x * redColor + (1.f - x) * originalColor;
					SetAgentColor(a, color);
				}
				aliveLengths[a]++;
			}
		}
		
		FlexGymBase::DoGui();
	}
};


template <class CTransform, class CVec3, class CQuat, class CMat33>
class RLWalkerHardEnv : public FlexGymBase
{
protected:
	static const int MaxActions = 100; // Required for populate state, increase if number of actions >= 100

	enum Task
	{
		ePlain,
		eWalk,
		eStandUp,
		eRollOver,
		eJump,
		eTasks = 5
	};

	enum StartDistribution
	{
		eStanding,
		eHalfHalf,
		eHard,
		eGround,
		eStartDistribution = 3
	};

public:

	// Need to implement these functions
	StartDistribution startDistribution;
	vector<Task> tasks;
	map<Task, Quat> taskRotationMap;

	vector<float> electricityCostScale;
	float masterElectricityCostScale;
	float electricityCost;		//	# cost for using motors-- this parameter should be carefully tuned against reward for making progress, other values less improtant
	float stallTorqueCostScale;
	float stallTorqueCost;		//	# cost for running electric current through a motor even at zero rotational speed, small
	float footCollisionCost;	//	# touches another leg, or other objects, that cost makes robot avoid smashing feet into itself

	float jointsAtLimitCost;	//	# discourage stuck joints

	float upVecWeight;
	float initialZ;
	float terminationZ;
	float pitch, r, yaw;
	float maxX, maxY; // Max size for randomply generated flag

	vector<int> feet;
	vector<int> footFlag;
	vector<float> feetContact;

	vector<int> knees;
	vector<int> kneeFlag;
	vector<float> kneesContact;

	vector<int> torsoFlag;
	vector<float> torsoContact;

	vector<int> hands;
	vector<int> handFlag;
	vector<float> handsContact;

	vector<int> heads;
	vector<int> headFlag;

	vector<int> pelvis;

	vector<int> numCollideOther;

	vector<vector<float>> bPrevAngleValue;

	vector<float> potentials;
	vector<float> potentialsOld;

	vector<float> ps;
	vector<float> walkTargetDists;
	vector<float> potentialLeaksOld;

	vector<vector<float>> jointSpeeds;
	vector<int> jointsAtLimits;

	vector<float> headings;
	vector<float> upVecs;
	vector<Transform> bodyPoses;
	vector<CVec3> bodyXYZ;
	vector<float> pelvisZ;

	vector<float> walkTargetX;
	vector<float> walkTargetY;

	// Running target vis
	vector<int> targetSphere;
	vector<int> targetRenderMaterials;
	float radius;
	Vec3 redColor;
	Vec3 greenColor;

	bool useStaticRews;
	vector<int> flagRunSteps;
	vector<int> onGroundFrameCounter;
	vector<int> walkingNormally;

	vector<int> aliveCounter;
	bool doRoboschool;

	// roboschool constants
	float roboschoolElectricityCost = 4.25f * -2.f;
	float roboschoolStalTorqueCost = 4.25f * -0.1f;
	float roboschoolJointsAtLimistCost = -0.2f;
	float roboschoolFeetCollisionCost = -1.f;

	bool doManualControl;
	float targetAngle; // For humnoid control from menu

	bool doFlagRun;
	int maxFlagResetSteps;
	int maxStepsOnGround;
	bool throwBox;
	float dt;

	RLWalkerHardEnv() :
		masterElectricityCostScale(2.f), 
		electricityCost(-2.f),
		stallTorqueCostScale(5.f), 
		stallTorqueCost(-0.1f),
		footCollisionCost(-0.5f), 
		jointsAtLimitCost(-0.2f),
		upVecWeight(0.05f),
		initialZ(0.8f), 
		terminationZ(0.79f),
		maxX(25.f), maxY(25.f),
		doFlagRun(true),
		maxFlagResetSteps(200),
		maxStepsOnGround(170),
		doManualControl(false),
		targetAngle(0.f),
		throwBox(false),
		dt(1.f / 60.f)
	{
		aliveCounter.resize(mNumAgents, 0);
	};

	virtual void SaveRLState(FILE* f)
	{
		FlexGymBase::SaveRLState(f);

		SaveVec(f, tasks);

		SaveVec(f, electricityCostScale);
		SaveVec(f, pelvis);
		SaveVec(f, heads);
		SaveVec(f, hands);
		SaveVec(f, feet);

		SaveVec(f, footFlag);
		SaveVec(f, handFlag);

		SaveVec(f, handsContact);
		SaveVec(f, feetContact);
		SaveVec(f, numCollideOther);

		SaveVecVec(f, bPrevAngleValue);

		SaveVec(f, potentials);
		SaveVec(f, potentialsOld);

		SaveVec(f, ps);
		SaveVec(f, walkTargetDists);
		SaveVec(f, potentialLeaksOld);

		SaveVecVec(f, jointSpeeds);
		SaveVec(f, jointsAtLimits);

		SaveVec(f, headings);
		SaveVec(f, upVecs);
		SaveVec(f, bodyXYZ);
		SaveVec(f, pelvisZ);

		SaveVec(f, walkTargetX);
		SaveVec(f, walkTargetY);

		SaveVec(f, flagRunSteps);
		SaveVec(f, onGroundFrameCounter);
		SaveVec(f, walkingNormally);
	}

	virtual void LoadRLState(FILE* f)
	{
		FlexGymBase::LoadRLState(f);

		LoadVec(f, tasks);

		LoadVec(f, electricityCostScale);
		LoadVec(f, pelvis);
		LoadVec(f, heads);
		LoadVec(f, hands);
		LoadVec(f, feet);

		LoadVec(f, footFlag);
		LoadVec(f, handFlag);

		LoadVec(f, handsContact);
		LoadVec(f, feetContact);
		LoadVec(f, numCollideOther);

		LoadVecVec(f, bPrevAngleValue);

		LoadVec(f, potentials);
		LoadVec(f, potentialsOld);

		LoadVec(f, ps);
		LoadVec(f, walkTargetDists);
		LoadVec(f, potentialLeaksOld);

		LoadVecVec(f, jointSpeeds);
		LoadVec(f, jointsAtLimits);

		LoadVec(f, headings);
		LoadVec(f, upVecs);
		LoadVec(f, bodyXYZ);
		LoadVec(f, pelvisZ);

		LoadVec(f, walkTargetX);
		LoadVec(f, walkTargetY);

		LoadVec(f, flagRunSteps);
		LoadVec(f, onGroundFrameCounter);
		LoadVec(f, walkingNormally);
	}

	void ParseJsonParams(const json& sceneJson) override
	{
		if (sceneJson.is_null())
		{
			return;
		}
		FlexGymBase::ParseJsonParams(sceneJson);
		doFlagRun = g_sceneJson.value(RL_JSON_DO_FLAGRUN, doFlagRun);
		maxStepsOnGround = GetJsonVal(g_sceneJson, "MaxStepsOnGround", 170);
		useStaticRews = g_sceneJson.value(RL_JSON_USE_STATIC_REWS, true);
		terminationZ = g_sceneJson.value(RL_JSON_TERMINAION_Z, 0.78f);
		rotCreateNoise = GetJsonVal(g_sceneJson, "RotStartNoise", 0.05f);
		startDistribution = GetJsonVal(g_sceneJson, "StartDistribution", StartDistribution::eHard);
		doRoboschool = GetJsonVal(g_sceneJson, "DoRoboschool", false);
	}

	void createTasks()
	{
		tasks.resize(mNumAgents);

		for (int a = 0; a < mNumAgents; ++a)
		{
			if (startDistribution == StartDistribution::eStanding)
			{
				if (a % 4)
				{
					tasks[a] = Task::ePlain;
				}
				else
				{
					tasks[a] = Task::eWalk;
				}
			}
			else if (startDistribution == StartDistribution::eHalfHalf)
			{

				if (a % 9 == 1 || a % 9 == 4 || a % 9 == 6)
				{
					tasks[a] = Task::eStandUp;
				}
				else if (a % 9 == 2 || a % 9 == 7)
				{
					tasks[a] = Task::eRollOver;
				}
				else if (a % 9 == 5 || a % 9 == 8)
				{
					tasks[a] = Task::eWalk;
				}
				else
				{
					tasks[a] = Task::ePlain;
				}
			}
			else if (startDistribution == StartDistribution::eHard)
			{

				if (a % 9 == 1 || a % 9 == 4)
				{
					tasks[a] = Task::eStandUp;
				}
				else if (a % 9 == 2 || a % 9 == 3 || a % 9 == 6 || a % 9 == 7)
				{
					tasks[a] = Task::eRollOver;
				}
				else if (a % 9 == 5 || a % 9 == 8)
				{
					tasks[a] = Task::eWalk;
				}
				else
				{
					tasks[a] = Task::ePlain;
				}
			}
			else if (startDistribution == StartDistribution::eGround)
			{
				if (a % 6)
				{
					tasks[a] = Task::eRollOver;
				}
				else
				{
					tasks[a] = Task::eStandUp;
				}
			}
			else
			{
				if (a % 6)
				{
					tasks[a] = Task::eRollOver;
				}
				else
				{
					tasks[a] = Task::eStandUp;
				}
			}
		}
	}

	virtual void AddTargetSpheres(int ai, Transform gt)
	{
		NvFlexRigidShape targetShape;
		NvFlexMakeRigidSphereShape(&targetShape, -1, radius, NvFlexMakeRigidPose(0, 0));

		int renderMaterial = AddRenderMaterial(redColor);
		targetRenderMaterials[ai] = renderMaterial;
		targetShape.user = UnionCast<void*>(renderMaterial);
		targetShape.group = 1;
		targetSphere[ai] = g_buffers->rigidShapes.size();
		g_buffers->rigidShapes.push_back(targetShape);

		NvFlexRigidPose pose = NvFlexMakeRigidPose(Vec3(20.f, 0.f, 0.f) + gt.p, Quat());
		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;

	//	g_buffers->rigidShapes.unmap();
	//	cout << "Add unmapped" << endl;
	//	NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
	//	cout << "Add set shapes" << endl;
	//
	//	g_buffers->rigidShapes.map();
	//	cout << "Add mapped" << endl;
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		mjcfs.push_back(make_shared<MJCFImporter>(loadPath.c_str()));

		int startJoint = g_buffers->rigidJoints.size();
		int startShape = g_buffers->rigidShapes.size();
		mjcfs.back()->AddPhysicsEntities(gt, ctrl, mpower, 20.f * kPi, true, true);
		int endJoint = g_buffers->rigidJoints.size();
		int endShape = g_buffers->rigidShapes.size();
		robotJoints.push_back({ startJoint, endJoint });
		robotShapes.push_back({ startShape, endShape });

	//	AddTargetSpheres(i, gt);

		torso.push_back(mjcfs[i]->bmap["lwaist"]);
		pelvis.push_back(mjcfs[i]->bmap["pelvis"]);
		// heads.push_back(mjcfs[i]->bmap["head"]);

		hands.push_back(mjcfs[i]->bmap["right_lower_arm"]);
		hands.push_back(mjcfs[i]->bmap["left_lower_arm"]);

		knees.push_back(mjcfs[i]->bmap["right_shin"]);
		knees.push_back(mjcfs[i]->bmap["left_shin"]);

		feet.push_back(mjcfs[i]->bmap["right_foot"]);
		feet.push_back(mjcfs[i]->bmap["left_foot"]);
	}

	virtual void LoadEnv()
	{
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

	//	radius = 0.4f;
	//	redColor = Vec3(1.0f, 0.04f, 0.07f);
	//	greenColor = Vec3(0.06f, 0.92f, 0.13f);

		targetSphere.resize(mNumAgents, -1);
		targetRenderMaterials.resize(mNumAgents);

		heads.clear();
		torso.clear();
		pelvis.clear();
		hands.clear();
		knees.clear();
		feet.clear();

		createTasks();

		// WIP
		Quat baseRot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f); // Or should be a transform?
		taskRotationMap[Task::ePlain] = baseRot;
		taskRotationMap[Task::eStandUp] = baseRot;
		taskRotationMap[Task::eRollOver] = baseRot;

		for (int i = 0; i < mNumAgents; i++)
		{
			Vec3 pos = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f);

			Vec3 posStart = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rotStart = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));

			if (tasks[i] == Task::eStandUp)
			{
				posStart.y += Randf(0.5f, 0.55f);
				rotStart = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));
			}
			else if (tasks[i] == Task::eRollOver)
			{
				posStart.y += Randf(0.25f, 0.3f);
				rotStart = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));
			}
			else
			{
				posStart.y += Randf(-0.025f, 0.025f);
			}

			rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * rot;
			rotStart = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise))
					   * rotStart;

			Transform gtStart(posStart, rotStart);
			gtStart = gtStart * preTransform;

			Transform gt(pos, rot);
			gt = gt * preTransform;

			int begin = g_buffers->rigidBodies.size();

			AddAgentBodiesJointsCtlsPowers(i, gtStart, ctrls[i], motorPower[i]);

			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
			agentStartOffset.push_back(gtStart);

			int end = g_buffers->rigidBodies.size();
			agentBodies.push_back(make_pair(begin, end));
		}

		maxPower = *max_element(std::begin(motorPower[0]), std::end(motorPower[0]));
		masses.clear();
		for (int bi = agentBodies[0].first; bi < agentBodies[0].second; ++bi)
		{
			masses.push_back(g_buffers->rigidBodies[bi].mass);
		}

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());
	}

	void setTask(int a, Task newTask = Task::ePlain)
	{
		tasks[a] = newTask;
	}

	virtual void resetTarget(int a, bool firstTime = true)
	{
		if (doFlagRun)
		{
			if (firstTime && (a % 3))
			{
				walkTargetX[a] = maxX;
				walkTargetY[a] = 0.f;
			}
			else
			{
				walkTargetX[a] = Randf(-maxX, maxX) + bodyXYZ[a].x;
				walkTargetY[a] = Randf(-maxY, maxY) + bodyXYZ[a].y;
			}

			flagRunSteps[a] = 0;

		//	g_buffers->rigidShapes.map();
		//
		//	NvFlexRigidPose pose = NvFlexMakeRigidPose(Vec3(walkTargetX[a], 0.f, walkTargetY[a]), Quat());
		//	g_buffers->rigidShapes[targetSphere[a]].pose = pose;
		//
		//	g_buffers->rigidShapes.unmap();
		//	NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
		//	const Vec3 target = Vec3(walkTargetX[a], 1.f, walkTargetY[a]);
		//	cout << "Starting reseting target sphere " << a << endl;
		//	resetTargetSphere(a, target);
		//	cout << "Reset target sphere done!" << endl;
		}
		else
		{
			walkTargetX[a] = 1000.f;
			walkTargetY[a] = 0.f;
		}
	}

	void resetTargetSphere(int a, const Vec3& target)
	{
	//	cout << "premap" << endl;
	//	g_buffers->rigidShapes.map();
	//	cout << "map" << endl;

	//	cout << "Spheres size: "<< targetSphere.size() << endl;
		NvFlexRigidPose pose = NvFlexMakeRigidPose(target, Quat());
	//	g_buffers->rigidShapes[targetSphere[a]].pose = pose;
	//	cout << "Pose" << endl;

	//	cout << "unmap" << endl;
	//	NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
	//	g_buffers->rigidShapes.map();
	//	cout << "map" << endl;
	}

	float potentialLeakRoboschool(int a)
	{
		float z1 = 0.f;
		if (torso[a] != -1)
		{
			Transform torsoPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[a]], (NvFlexRigidPose*)&torsoPose);

			z1 = torsoPose.p.y;
		}

		float z2 = 0.f;
		if (pelvis[a] != -1)
		{
			Transform pelvisPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[pelvis[a]], (NvFlexRigidPose*)&pelvisPose);

			z2 = pelvisPose.p.y;
		}

		float z = 0.3f * z2 + 0.7f * z1;
		z = min(0.7f, max(0.f, z));

		float potenLeak = 0.5f + 1.5f * (z / 0.7f);

		//	cout << "z1 = " << z1 << " z2 = " << z2 << " z = " << z << endl;
		//	cout << "Potential leak for agent "<< a <<" = " << potenLeak << endl;

		return potenLeak;
	}

	float potentialLeak(int a)
	{
		float z1 = 0.f;
		if (torso[a] != -1)
		{
			Transform torsoPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[a]], (NvFlexRigidPose*)&torsoPose);

			z1 = torsoPose.p.y;
		}

		float z2 = 0.f;
		if (pelvis[a] != -1)
		{
			Transform pelvisPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[pelvis[a]], (NvFlexRigidPose*)&pelvisPose);

			z2 = pelvisPose.p.y;
		}

		float z = 0.3f * z2 + 0.7f * z1;
		z = min(0.9f, max(0.f, z));

		float potenLeak = 2.5f * z / 0.9f;

		//	cout << "z1 = " << z1 << " z2 = " << z2 << " z = " << z << endl;
		//	cout << "Potential leak for agent "<< a <<" = " << potenLeak << endl;

		return potenLeak;
	}

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		// Prepare state
		//--------------------
		numJointsAtLimit = 0;

		float joints[MaxActions * 2];
		float angles[MaxActions];
		float lows[MaxActions];
		float highs[MaxActions];

		int numBodies = GetNumBodies();

		GetAngles(a, angles, lows, highs);
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = angles[i], vel;
			float low = lows[i], high = highs[i];

			if (bPrevAngleValue[a][qq] > 1e8)
			{
				bPrevAngleValue[a][qq] = pos;
			}

			float posDiff = pos - bPrevAngleValue[a][qq];
			if (fabs(posDiff) < kPi)
			{
				vel = posDiff / dt;
			}
			else
			{
				vel = (-(Sign(pos) * 2.f * kPi) + posDiff) / dt;
			}
			bPrevAngleValue[a][qq] = pos;

			float posMid = 0.5f * (low + high);
			pos = 2.f * (pos - posMid) / (high - low);

			joints[2 * i] = pos;
			joints[2 * i + 1] = vel * 0.1f;

			jointSpeeds[i] = joints[2 * i + 1];
			if (fabs(joints[2 * i]) > 0.99f)
			{
				numJointsAtLimit++;
			}
		}
		CTransform bodies[100];

		GetGlobalPose(a, bodies);
		CTransform bodyPose = bodies[0];

		upVec = GetBasisVector2(bodyPose.q).z;
		float sumX = 0.0f;
		float sumY = 0.0f;

		float totalMass = 0.f;
		for (int i = 0; i < numBodies; i++)
		{
			float bodyMass = masses[i];

			Vec3 pos = bodies[i].p * bodyMass;
			sumX += pos.x;
			sumY += pos.y;
			totalMass += bodyMass;
		}
		bodyXYZ[a] = Vec3(sumX / totalMass, sumY / totalMass, bodyPose.p.z);

		getEulerZYX(bodyPose.q, yaw, p, r);
		float z = bodyXYZ[a].z;

		if (initialZ > 1e9)
		{
			initialZ = z;
		}

		CVec3 toTarget = CVec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
		walkTargetDist = Length(toTarget);
		if (doFlagRun)
		{
			if (flagRunSteps[a] > maxFlagResetSteps || walkTargetDist < 1.f)
			{
				resetTarget(a, false);
				toTarget = CVec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
				walkTargetDist = Length(toTarget);

				potentials[a] = -walkTargetDist / dt + 100.f * potentialLeak(a);
				potentialsOld[a] = potentials[a];

				const Vec3 target = Vec3(walkTargetX[a], 1.f, walkTargetY[a]);
			//	cout << "Reset ES" << endl;
				resetTargetSphere(a, target);
			//	cout << "End of the reset" << endl;
			}
			flagRunSteps[a]++;
		}

		float x = max(walkTargetDist / maxX, 1.f);
		g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;

		float walkTargetTheta = atan2(walkTargetY[a] - bodyXYZ[a].y, walkTargetX[a] - bodyXYZ[a].x);

		toTarget = Normalize(toTarget);
		heading = Dot(GetBasisVector0(bodyPose.q), toTarget);
		float angleToTarget = walkTargetTheta - yaw;

		Matrix33 mat = Matrix33(
						 Vec3(cos(-yaw), sin(-yaw), 0.0f),
						 Vec3(-sin(-yaw), cos(-yaw), 0.0f),
						 Vec3(0.0f, 0.0f, 1.0f));

		Vec3 vel = GetLinearVel(a, 0);
		Vec3 bvel = mat * vel;
		float vx = bvel.x;
		float vy = bvel.y;
		float vz = bvel.z;

		Vec3 angVel = GetAngularVel(a, 0);
		Vec3 bangVel = mat * angVel; // ?
		float avx = bangVel.x;
		float avy = bangVel.y;
		float avz = bangVel.z;

		float more[11] = { z - initialZ,
						   sin(angleToTarget), cos(angleToTarget),
						   0.25f * vx, 0.25f * vy, 0.25f * vz,
						   0.25f * avx, 0.25f * avy, 0.25f * avz,
						   r, p
						 };

		int ct = 0;
		for (int i = 0; i < 11; ++i)
		{
			state[ct++] = more[i];
		}

		for (int i = 0; (unsigned int)i < ctrls[a].size() * 2; ++i)
		{
			state[ct++] = joints[i];
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = feetContact[2 * a + i];
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = kneesContact[2 * a + i];
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = handsContact[2 * a + i];
		}

	//	state[ct++] = torsoContact[a];

		// 21 Joint forces
		// float jointReadingScale = 4e-3f;
		// for (int i = 0; i < ctrls[a].size(); i++)
		// {
		// 	int jointIdx = ctrls[a][i].first;
		// 	int jointDof = ctrls[a][i].second + 3;
		// 	float force = g_buffers->rigidJoints[jointIdx].lambda[jointDof];
		// //	if (force > 200.f)
		// //		cout << force << endl;
		// 	state[ct++] = force * jointReadingScale;
		// }

		float* prevActions = GetAction(a);
		if (prevActions) // could be null if this is called for the first time before agent acts
		{
			for (int i = 0; i < mNumActions; ++i)
			{
				state[ct++] = prevActions[i];
			}
		}

		for (int i = 0; i < ct; ++i)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}

		if (ct != mNumObservations)
		{
			printf("NumObservation is %d, but ct got %d!\n", mNumObservations, ct);
		}
	}

	virtual void ExtractStateRoboschool(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		// Prepare state
		//--------------------
		numJointsAtLimit = 0;

		float joints[MaxActions * 2];
		float angles[MaxActions];
		float lows[MaxActions];
		float highs[MaxActions];

		int numBodies = GetNumBodies();

		GetAngles(a, angles, lows, highs);
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = angles[i], vel;
			float low = lows[i], high = highs[i];

			if (bPrevAngleValue[a][qq] > 1e8)
			{
				bPrevAngleValue[a][qq] = pos;
			}

			float posDiff = pos - bPrevAngleValue[a][qq];
			if (fabs(posDiff) < kPi)
			{
				vel = posDiff / dt;
			}
			else
			{
				vel = (-(Sign(pos) * 2.f * kPi) + posDiff) / dt;
			}
			bPrevAngleValue[a][qq] = pos;

			float posMid = 0.5f * (low + high);
			pos = 2.f * (pos - posMid) / (high - low);

			joints[2 * i] = pos;
			joints[2 * i + 1] = vel * 0.1f;

			jointSpeeds[i] = joints[2 * i + 1];
			if (fabs(joints[2 * i]) > 0.99f)
			{
				numJointsAtLimit++;
			}
		}
		CTransform bodies[100];

		GetGlobalPose(a, bodies);
		CTransform bodyPose = bodies[0];
		
		float sumX = 0.0f;
		float sumY = 0.0f;

		float totalMass = 0.f;
		for (int i = 0; i < numBodies; i++)
		{
			float bodyMass = masses[i];

			Vec3 pos = bodies[i].p * bodyMass;
			sumX += pos.x;
			sumY += pos.y;
			totalMass += bodyMass;
		}

		bodyXYZ[a] = Vec3(sumX / totalMass, sumY / totalMass, bodyPose.p.z);

		getEulerZYX(bodyPose.q, yaw, p, r);
		float z = bodyXYZ[a].z;
		if (initialZ > 1e9)
		{
			initialZ = z;
		}

		CVec3 toTarget = CVec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
		walkTargetDist = Length(toTarget);
		if (doFlagRun)
		{
			if (flagRunSteps[a] > maxFlagResetSteps || walkTargetDist < 1.f)
			{
				resetTarget(a, false);
				toTarget = CVec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
				walkTargetDist = Length(toTarget);

				potentials[a] = -walkTargetDist / dt + 100.f * potentialLeakRoboschool(a);
				potentialsOld[a] = potentials[a];
			}
			flagRunSteps[a]++;
		}

		float walkTargetTheta = atan2(walkTargetY[a] - bodyXYZ[a].y, walkTargetX[a] - bodyXYZ[a].x);

		toTarget = Normalize(toTarget);
		heading = Dot(GetBasisVector0(bodyPose.q), toTarget);
		float angleToTarget = walkTargetTheta - yaw;

		CMat33 mat = CMat33(
						 CVec3(cos(-yaw), sin(-yaw), 0.0f),
						 CVec3(-sin(-yaw), cos(-yaw), 0.0f),
						 CVec3(0.0f, 0.0f, 1.0f));
		CVec3 vel = GetLinearVel(a, 0);
		CVec3 bvel = mat * vel;
		float vx = bvel.x;
		float vy = bvel.y;
		float vz = bvel.z;

		float more[8] = { z - initialZ,
						   sin(angleToTarget), cos(angleToTarget),
						   0.3f * vx, 0.3f * vy, 0.3f * vz,
						   r, p
						 };

		int ct = 0;
		for (int i = 0; i < 8; ++i)
		{
			state[ct++] = more[i];
		}

		for (int i = 0; (unsigned int)i < ctrls[a].size() * 2; ++i)
		{
			state[ct++] = joints[i];
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = feetContact[2 * a + i];
		}

		for (int i = 0; i < ct; ++i)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}
	}

	void init(PPOLearningParams& ppoParams, const char* pythonFile, const char* workingDir, const char* logDir, float deltaT = 1.f / 60.f)
	{
		if (logDir)
		{
			EnsureDirExists(workingDir + string("/") + logDir);
		}

		dt = deltaT;

		InitRLInfo();
		LaunchPythonProcess(pythonFile, workingDir, logDir, ppoParams, g_sceneJson);

		potentialsOld.resize(mNumAgents);
		potentials.resize(mNumAgents);

		ps.resize(mNumAgents);
		walkTargetDists.resize(mNumAgents, 1000.f);
		potentialLeaksOld.resize(mNumAgents, 0.f);
		jointSpeeds.resize(mNumAgents);
		jointsAtLimits.resize(mNumAgents);
		headings.resize(mNumAgents);
		upVecs.resize(mNumAgents);
		walkTargetX.resize(mNumAgents);
		walkTargetY.resize(mNumAgents);
		bodyXYZ.resize(mNumAgents);

		feetContact.resize(2 * mNumAgents, 0.f);
		kneesContact.resize(2 * mNumAgents, 0.f);
		handsContact.resize(2 * mNumAgents, 0.f);
		torsoContact.resize(mNumAgents, 0.f);
		numCollideOther.resize(mNumAgents, 0);
		bPrevAngleValue.resize(mNumAgents);

		flagRunSteps.clear();
		flagRunSteps.resize(mNumAgents, 0);
		electricityCostScale.clear();
		electricityCostScale.resize(mNumAgents, masterElectricityCostScale);

		onGroundFrameCounter.clear();
		onGroundFrameCounter.resize(mNumAgents, 0);
		walkingNormally.clear();
		walkingNormally.resize(mNumAgents, 0);

		bodyPoses.resize(mNumAgents);

		aliveCounter.resize(mNumAgents, 0);

		for (int a = 0; a < mNumAgents; a++)
		{
			jointSpeeds[a].resize(mNumActions);
			potentialsOld[a] = 0.0f;
			potentials[a] = 0.0f;
			bodyXYZ[a] = CVec3(0.f, 0.f, 0.f);
			resetTarget(a, true);

			for (int i = 0; i < 2; ++i)
			{
				feetContact[2 * a + i] = 0.f;
				kneesContact[2 * a + i] = 0.f;
				handsContact[2 * a + i] = 0.f;
			}
			torsoContact[a] = 0.f;
			numCollideOther[a] = 0;
			bPrevAngleValue[a].resize(mNumActions);
		}

		createTasks();
	}

	void getEulerZYX(CQuat& q, float& yawZ, float& pitchY, float& rollX)
	{
		float squ;
		float sqx;
		float sqy;
		float sqz;
		float sarg;
		sqx = q.x * q.x;
		sqy = q.y * q.y;
		sqz = q.z * q.z;
		squ = q.w * q.w;

		rollX = atan2(2 * (q.y * q.z + q.w * q.x), squ - sqx - sqy + sqz);
		sarg = (-2.0f) * (q.x * q.z - q.w * q.y);
		pitchY = sarg <= (-1.0f) ? (-0.5f) * PxPi : (sarg >= (1.0f) ? (0.5f) * PxPi : asinf(sarg));
		yawZ = atan2(2 * (q.x * q.y + q.w * q.z), squ + sqx - sqy - sqz);
	}

	virtual float AliveBonus(int a, float z, float pitch)
	{
		//	if z < 0.8:
		//		if self.task == self.TASK_WALK :
		//			self.on_ground_frame_counter = 10000  # This ends episode immediately
		//		self.on_ground_frame_counter += 1
		//		self.electricity_cost = RoboschoolHumanoid.electricity_cost / 5.0   # Don't care about electricity, just stand up!
		//	elif self.on_ground_frame_counter > 0:
		//		self.on_ground_frame_counter -= 1
		//		self.electricity_cost = RoboschoolHumanoid.electricity_cost / 2.0    # Same as in Flagrun
		//	else:
		//		self.walking_normally += 1
		//			self.electricity_cost = RoboschoolHumanoid.electricity_cost / 2.0
		//			# End episode if the robot can't get up in 170 frames, to save computation and decorrelate observations.
		//	return self.potential_leak() if self.on_ground_frame_counter<170 else - 1

		if (z < terminationZ) // Annealing?!
		{
			if (tasks[a] == Task::ePlain)
			{
				onGroundFrameCounter[a] = 10000; // This ends episode immediately
			}
			onGroundFrameCounter[a]++;
			electricityCostScale[a] = 0.2f * masterElectricityCostScale;
		}
		else if (onGroundFrameCounter[a] > 0)
		{
			onGroundFrameCounter[a]--;
			electricityCostScale[a] = masterElectricityCostScale;
		}
		else
		{
			walkingNormally[a]++;
		}

		if (onGroundFrameCounter[a] < maxStepsOnGround)
		{
			return potentialLeak(a);
		}
		else
		{
			return -1.f;
		}
	}

	virtual float AliveBonusRoboschool(int a, float z, float pitch)
	{
		int groundCounter = onGroundFrameCounter[a];

        if (z < 0.8)
		{
			if (tasks[a] == Task::ePlain)
			{
				onGroundFrameCounter[a] = 10000; // This ends episode immediately
			}
			onGroundFrameCounter[a]++;
            electricityCostScale[a] =  roboschoolElectricityCost / 5.f;
		}
		else if (groundCounter > 0)
		{
			onGroundFrameCounter[a]--;
			electricityCostScale[a] = roboschoolElectricityCost / 2.f;
		}
		else
		{
			walkingNormally[a]++;
			electricityCostScale[a] = roboschoolElectricityCost / 2.f;
		}

		if (groundCounter < 170)    
		{
			return potentialLeakRoboschool(a);
		}
		return -1;
	}

	virtual void PopulateState(int a, float* state)
	{
		float& p = ps[a];

		float& walkTargetDist = walkTargetDists[a];
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& jointsAtLimitA = jointsAtLimits[a];
		float& heading = headings[a];
		float& upVec = upVecs[a];

		if (doRoboschool)
		{
			potentialLeaksOld[a] = potentialLeakRoboschool(a);
			ExtractStateRoboschool(a, state, p, walkTargetDist, jointSpeedsA, jointsAtLimitA, heading, upVec);
		}
		else
		{
			potentialLeaksOld[a] = potentialLeak(a);
			ExtractState(a, state, p, walkTargetDist, jointSpeedsA, jointsAtLimitA, heading, upVec);
		}
	}

	virtual void ComputeRewardAndDeadRoboschool(int a, float* action, float* state, float& rew, bool& dead)
	{

		float& potential = potentials[a];
		float& potentialOld = potentialsOld[a];
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& numJointsAtLimitA = jointsAtLimits[a];

		float currentZ = state[0] + initialZ; // # state[0] is body height above ground, body_rpy[1] is pitch
		float alive = AliveBonusRoboschool(a, currentZ, p);
		dead = alive < 0.f;

		potentialOld = potential;
		potential = -walkTargetDist / dt;
		if (potentialOld > 1e9)
		{
			potentialOld = potential;
		}
		float progress = potential - potentialOld;
		if (progress > 100.f)
		{
			printf("progress is infinite %f %f %f \n", progress, potential, potentialOld);
		}

		float sum = 0.f;
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			float vv = abs(action[i] * jointSpeedsA[i]);
			if (!isfinite(vv))
			{
				printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv, action[i], jointSpeedsA[i]);
			}
			if (!isfinite(action[i]))
			{
				printf("action at %d is infinite\n", i);
			}
			if (!isfinite(jointSpeedsA[i]))
			{
				printf("jointSpeeds at %d is infinite\n", i);
			}

			sum += vv;
		}
		if (!isfinite(sum))
		{
			printf("Sum of ctl*joint_speed is infinite!\n");
		}
		float electricityCostCurrent = electricityCostScale[a] * sum / (float)ctrls[a].size();

		sum = 0.f;
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			sum += action[i] * action[i];
		}
		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
		}
		electricityCostCurrent += roboschoolStalTorqueCost * sum / (float)ctrls[a].size();

		float jointsAtLimitCostCurrent = roboschoolJointsAtLimistCost * (float)numJointsAtLimitA;

		float feetCollisionCostCurrent = 0.0f;
		if (numCollideOther[a] > 0)
		{
			feetCollisionCostCurrent += roboschoolFeetCollisionCost;
		}

		float rewards[5] =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent,
		};
		
		rew = 0.f;
		for (int i = 0; i < 5; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}

		// printf("%f %f %f %f %f = %f\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4], rew);
	}

	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		if (doRoboschool)
		{
			ComputeRewardAndDeadRoboschool(a, action, state, rew, dead);
		}
		else
		{
			float& potential = potentials[a];
			float& potentialOld = potentialsOld[a];
			float& p = ps[a];
			float& potentialLeakOld = potentialLeaksOld[a];
			float potLeak = potentialLeak(a);
			float& walkTargetDist = walkTargetDists[a];
			float* jointSpeedsA = &jointSpeeds[a][0];
			int& numJointsAtLimitA = jointsAtLimits[a];
			float& heading = headings[a];
			float& upVec = upVecs[a];

			float electrCost = electricityCostScale[a] * electricityCost;
			float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

			float currentZ = state[0] + initialZ; // # state[0] is body height above ground, body_rpy[1] is pitch
			float alive = AliveBonus(a, currentZ, p);
			dead = alive < 0.f;

			potentialOld = potential;
			potential = -walkTargetDist / dt + 100.f * potLeak;
			if (potentialOld > 1e9)
			{
				potentialOld = potential;
			}

			float progress = potential - potentialOld;
			if (alive < 2.499f) // 2.5f max value of the leak potential
			{
				progress = 100.f * (potLeak - potentialLeakOld);
			}

			if (progress > 100.f)
			{
				printf("progress is infinite %f %f %f \n", progress, potential, potentialOld);
			}

			float electricityCostCurrent = 0.0f;
			float sum = 0.0f;
			for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
			{
				float vv = abs(action[i] * jointSpeedsA[i]) * motorPower[a][i] / maxPower; // Take motor power into account
				if (!isfinite(vv))
				{
					printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv, action[i], jointSpeedsA[i]);
				}

				if (!isfinite(action[i]))
				{
					printf("action at %d is infinite\n", i);
				}

				if (!isfinite(jointSpeedsA[i]))
				{
					printf("jointSpeeds at %d is infinite\n", i);
				}

				sum += vv;
			}

			if (!isfinite(sum))
			{
				printf("Sum of ctl*joint_speed is infinite!\n");
			}

			electricityCostCurrent += electrCost * sum / (float)ctrls[a].size();

			sum = 0.0f;
			for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
			{
				sum += action[i] * action[i];
			}

			if (!isfinite(sum))
			{
				printf("Sum of ctl^2 is infinite!\n");
			}

			electricityCostCurrent += stallTorqCost * sum / (float)ctrls[a].size();

			float jointsAtLimitCostCurrent = jointsAtLimitCost * (float)numJointsAtLimitA;

			float feetCollisionCostCurrent = 0.0f;
			if (numCollideOther[a] > 0)
			{
				feetCollisionCostCurrent += footCollisionCost;
			}

			//cout << "heading = " << heading.x << " " << heading.y << " " << heading.z << endl;
			float headingRew = 0.5f * ((heading > 0.8f) ? 1.f : heading / 0.8f) + upVecWeight * ((upVec > 0.93f) ? 1.f : 0.f); // MJCF4
			
			if (!useStaticRews)
			{
				headingRew = 0.f;
				alive = 0.f;
			}
			
			float rewards[6] =
			{
				alive,
				progress,
				electricityCostCurrent,
				jointsAtLimitCostCurrent,
				feetCollisionCostCurrent,
				headingRew
			};
			//printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

			rew = 0.f;
			for (int i = 0; i < 6; i++)
			{
				if (!isfinite(rewards[i]))
				{
					printf("Reward %d is infinite\n", i);
				}
				rew += rewards[i];
			}
		}
	}

	virtual void ResetAllAgents()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			ResetAgent(a);
		}
		RLFlexEnv::ResetAllAgents();
	}

	virtual void ResetAgent(int a)
	{
		potentialsOld[a] = 1e10f;
		potentials[a] = 1e10f;

		onGroundFrameCounter[a] = 0;
		walkingNormally[a] = 0;
		numCollideOther[a] = 0;

		for (int i = 0; i < (int)bPrevAngleValue[a].size(); i++)
		{
			bPrevAngleValue[a][i] = 1e9f;
		}

		resetTarget(a, true);
		aliveCounter[a] = 0;
		RLFlexEnv::ResetAgent(a);
	}

	virtual void ClearContactInfo() = 0;

	virtual void PreHandleCommunication()
	{
		FinalizeContactInfo();
	}

	virtual void KeyDown(int key)
	{
		if (key == 'x')
		{
			throwBox = true;
		}
	}

	virtual void DoGui()
	{
		if (imguiCheck("Flagrun", doFlagRun))
		{
			doFlagRun = !doFlagRun;
		}

		if (imguiCheck("Manual Humanoid Control", doManualControl))
		{
			doManualControl = !doManualControl;
		}

		imguiSlider("Target Angle", &targetAngle, -kPi, kPi, 0.1f);
		if (doManualControl)
		{
			for (int a = 0; a < mNumAgents; ++a)
			{
				walkTargetX[a] = maxX * cos(targetAngle);
				walkTargetY[a] = maxX * sin(targetAngle);
			}
		}
	}

	virtual void Update()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			aliveCounter[a] += 1;
		}

		//	cout << "Base update" << endl;
		if (throwBox)
		{
			float bscale = 0.07f + Randf() * 0.01f;

			Vec3 origin, dir;
			GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

			NvFlexRigidShape box;
			NvFlexMakeRigidBoxShape(&box, g_buffers->rigidBodies.size(), bscale, bscale, bscale,
				NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));
			box.filter = 0;
			box.group = -1;
			box.material.friction = 1.0f;
			box.material.torsionFriction = 0.01;
			box.material.rollingFriction = 0.01f;
			box.material.restitution = 0.4f;
			box.thickness = 0.0125f;

			NvFlexRigidBody body;
			float boxDensity = 2500.0f;
			NvFlexMakeRigidBody(g_flexLib, &body, origin + dir, Quat(), &box, &boxDensity, 1);

			// set initial angular velocity
			body.angularVel[0] = 0.f;
			body.angularVel[1] = 0.01f;
			body.angularVel[2] = 0.01f;
			body.angularDamping = 0.0f;
			(Vec3&)body.linearVel = dir * 16.0f;

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(box);

			g_buffers->rigidShapes.unmap();
			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
			g_buffers->rigidShapes.map();
			g_buffers->rigidBodies.map();

			throwBox = false;

			//	footFlag.resize(g_buffers->rigidBodies.size(), -1);
			//	kneeFlag.resize(g_buffers->rigidBodies.size(), -1);
			//	torsoFlag.resize(g_buffers->rigidBodies.size(), -1);
			//	handFlag.resize(g_buffers->rigidBodies.size(), -1);
		}
		FlexGymBase::Update();
	}
};


// Should all this stuff go to the already existingbase FlexEnv class
class FlexGymBase2 : public CommonFlexGymBase
{
public:
	enum FileType
	{
		eMujoco,
		eURDF,
		eSDF,
	};
	FileType fileType;

	enum Control
	{
		eTorque,
		ePosition,
		eVelocity,
		eInverseDynamics,
	};
	Control controlType;

	FlexGymBase2() : rigidContacts(g_flexLib, g_solverDesc.maxRigidBodyContacts), rigidContactCount(g_flexLib, 1)
	{
		controlType = Control::eInverseDynamics;

		mNumAgents = 500;
		mNumObservations = 44;
		mNumActions = 17;
		mMaxEpisodeLength = 1000;

		g_numSubsteps = 2;
		g_params.numIterations = 30;

		g_sceneLower = Vec3(-50.f, 0.f, -50.f);
		g_sceneUpper = Vec3(120.f, 10.f, 100.f);

		g_pause = true;
		mDoLearning = g_doLearning;

		numRenderSteps = 1;

		numPerRow = 25;
		spacing = 20.f;

		powerScale = 1.f;
		maxPower = 100.f;

		angleResetNoise = 0.f;
		angleVelResetNoise = 0.f;
		velResetNoise = 0.f;

		pushFrequency = 500;	// How much steps in average per 1 kick
		forceMag = 0.0f;		// Remove from base class ?
		yOffset = 0.0f;

		preTransform = Transform(Vec3(), Quat());
		rotCreateNoise = 0.05f;

		dt = 1.f / 60.f;
	}

	// Need to implement these functions
	virtual void FinalizeContactInfo() = 0;

	Transform preTransform;
	string loadPath;

	vector<pair<int, int> > agentBodies;
	vector<NvFlexRigidBody> initBodies;
	vector<NvFlexRigidJoint> initJoints;
	vector<vector<float>> prevAngleValue;
	vector<vector<float>> joinVelocities;
	vector<vector<float>> jointSpeeds;
	vector<int> jointsAtLimits;

	vector<float> bodyMasses;

	// Transforms from initial
	vector<Transform> agentOffsetInv;
	vector<Transform> agentOffset;

	// Start transforms with randomization
	vector<Transform> agentStartOffset;

	vector<vector<pair<int, NvFlexRigidJointAxis>>> ctrls;
	vector<vector<float>> motorPower;
	float powerScale;
	float maxPower;

	vector<int> torso; // Move torso to the ansestor too?

	NvFlexVector<NvFlexRigidContact> rigidContacts;
	NvFlexVector<int> rigidContactCount;

	vector<vector<Transform>> initTrans; // Do we need them?
	bool mDoLearning; // Rename
	int numRenderSteps;

	// Randomization
	float angleResetNoise;
	float angleVelResetNoise;
	float velResetNoise;
	float rotCreateNoise;

	// How much steps in average per 1 kick
	int pushFrequency; // Move to ancestor?
	float forceMag;

	int numPerRow;
	float spacing;
	float yOffset;

	float dt;
	virtual void LoadRLState(FILE* f)
	{
		CommonFlexGymBase::LoadRLState(f);
		LoadVecVec(f, prevAngleValue);
		LoadVecVec(f, joinVelocities);
		LoadVecVec(f, jointSpeeds);
		LoadVec(f, jointsAtLimits);
		LoadVec(f, bodyMasses);
	}

	virtual void SaveRLState(FILE* f)
	{
		CommonFlexGymBase::SaveRLState(f);
		SaveVecVec(f, prevAngleValue);
		SaveVecVec(f, joinVelocities);
		SaveVecVec(f, jointSpeeds);
		SaveVec(f, jointsAtLimits);
		SaveVec(f, bodyMasses);
	}

	void ParseJsonParams(const json& sceneJson) override
	{
		if (sceneJson.is_null())
		{
			return;
		}
		RLFlexEnv::ParseJsonParams(sceneJson);

		// Parsing of common JSON parameters
		loadPath = sceneJson.value(RL_JSON_LOAD_PATH, loadPath);

		g_params.solverType = sceneJson.value(RL_JSON_SOLVER_TYPE, g_params.solverType);
		g_numSubsteps = sceneJson.value(RL_JSON_NUM_SUBSTEPS, g_numSubsteps);
		g_params.numIterations = sceneJson.value(RL_JSON_NUM_ITERATIONS, g_params.numIterations);
		g_params.numInnerIterations = sceneJson.value(RL_JSON_NUM_INNER_ITERATIONS, g_params.numInnerIterations);
		g_params.numPostCollisionIterations = sceneJson.value(RL_JSON_NUM_POST_COLLISION_ITERATIONS, g_params.numPostCollisionIterations);

		g_params.warmStart = GetJsonVal(g_sceneJson, RL_JSON_WARMSTART, 0.0f);
		// SOR
		g_params.relaxationFactor = GetJsonVal(g_sceneJson, "RelaxationFactor", 0.75f);
		g_params.systemRegularization = GetJsonVal(g_sceneJson, "SystemRegularization", 1e-6f);

		g_pause = sceneJson.value(RL_JSON_PAUSE, g_pause);

		mDoLearning = sceneJson.value(RL_JSON_DO_LEARNING, mDoLearning);
		numRenderSteps = sceneJson.value(RL_JSON_NUM_RENDER_STEPS, numRenderSteps);

		numPerRow = sceneJson.value(RL_JSON_NUM_PER_ROW, numPerRow);
		spacing = sceneJson.value(RL_JSON_SPACING, spacing);

		angleResetNoise = sceneJson.value(RL_JSON_ANGLE_RESET_NOISE, angleResetNoise);
		angleVelResetNoise = sceneJson.value(RL_JSON_ANGLE_VELOCITY_RESET_NOISE, angleVelResetNoise);
		velResetNoise = sceneJson.value(RL_JSON_VELOCITY_RESET_NOISE, velResetNoise);

		pushFrequency = sceneJson.value(RL_JSON_PUSH_FREQUENCY, pushFrequency);
		forceMag = sceneJson.value(RL_JSON_FORCE_MAGNITUDE, forceMag);
	}

	void init(PPOLearningParams& ppoParams, const char* pythonFile, const char* workingDir, const char* logDir, float deltat = 1.f / 60.f)
	{
		dt = deltat;

		if (logDir)
		{
			EnsureDirExists(workingDir + string("/") + logDir);
		}

		InitRLInfo();
		LaunchPythonProcess(pythonFile, workingDir, logDir, ppoParams, g_sceneJson);

		prevAngleValue.resize(mNumAgents);
		joinVelocities.resize(mNumAgents);
		jointSpeeds.resize(mNumAgents);

		jointsAtLimits.resize(mNumAgents);

		for (int a = 0; a < mNumAgents; a++)
		{
			prevAngleValue[a].resize(mNumActions);
			joinVelocities[a].resize(mNumActions);
			jointSpeeds[a].resize(mNumActions);
		}
	}

	void init(float deltat = 1.f / 60.f)
	{
		dt = deltat;

		InitRLInfo();

		prevAngleValue.resize(mNumAgents);
		joinVelocities.resize(mNumAgents);
		jointSpeeds.resize(mNumAgents);

		jointsAtLimits.resize(mNumAgents);

		for (int a = 0; a < mNumAgents; a++)
		{
			prevAngleValue[a].resize(mNumActions);
			joinVelocities[a].resize(mNumActions);
			jointSpeeds[a].resize(mNumActions);
		}
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpowers) = 0;

	virtual void LoadEnv() = 0;

	virtual void PreSimulation()
	{
		if (!mDoLearning)
		{
			if (!g_pause || g_step)
			{
				for (int s = 0; s < numRenderSteps; s++)
				{
					// tick solver
					NvFlexSetParams(g_solver, &g_params);
					NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
				}

				g_frame++;
				g_step = false;
			}
		}
		else
		{
			g_buffers->rigidShapes.map();

			NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
			g_buffers->rigidBodies.map();

			NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
			g_buffers->rigidJoints.map();

			for (int s = 0; s < numRenderSteps; s++)
			{
#ifdef NV_FLEX_GYM
				Simulate();
				FinalizeContactInfo();
				for (int a = 0; a < mNumAgents; ++a)
				{
					PopulateState(a, &mObsBuf[a * mNumObservations]);
					if (mNumExtras > 0) PopulateExtra(a, &mExtraBuf[a * mNumExtras]);
					ComputeRewardAndDead(a, GetAction(a), &mObsBuf[a * mNumObservations], mRewBuf[a], (bool&)mDieBuf[a]);
				}
#else
				HandleCommunication();
#endif
				ClearContactInfo();
			}

			g_buffers->rigidShapes.unmap();
			NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

			g_buffers->rigidJoints.unmap();
			NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

		}
	}

	virtual void ExtractState(int a, float* state, float* jointSpeeds, int& numJointsAtLimit)
	{
		// Prepare state
		//--------------------
		numJointsAtLimit = 0;

		// Perf?
		vector<float> joints(mNumActions * 2, 0.f);
		vector<float> angles(mNumActions, 0.f);
		vector<float> lows(mNumActions, 0.f);
		vector<float> highs(mNumActions, 0.f);

		GetAngles(a, angles, lows, highs);
		for (int i = 0; i < (int)ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = angles[i];
			float low = lows[i], high = highs[i];

			if (prevAngleValue[a][qq] > 1e8)
			{
				prevAngleValue[a][qq] = pos;
			}

			joinVelocities[a][qq] = (pos - prevAngleValue[a][qq]) / dt;

			prevAngleValue[a][qq] = pos;

			float posMid = 0.5f * (low + high);
			pos = 2.f * (pos - posMid) / (high - low);

			joints[2 * i] = pos;
			joints[2 * i + 1] = joinVelocities[a][qq] * 0.1f;

			jointSpeeds[i] = joints[2 * i + 1];
			if (fabs(joints[2 * i]) > 0.99f)
			{
				numJointsAtLimit++;
			}
		}

		int ct = 0;
		for (int i = 0; (unsigned int)i < ctrls[a].size() * 2; i++)
		{
			state[ct++] = joints[i];
		}

		for (int i = 0; i < ct; i++)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}
	}

	virtual void PopulateState(int a, float* state)
	{
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& jointsAtLimitA = jointsAtLimits[a];

		ExtractState(a, state, jointSpeedsA, jointsAtLimitA);
	}

	virtual void PopulateExtra(int a, float* extra) {}

	// Move to ancestor? Make abstract to handle Mujoco, URDF, SDF?
	virtual void ResetAgent(int a) = 0;

	virtual void ResetAllAgents()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			ResetAgent(a);
		}
	}

	// Can be random pushes or constant, wind type force
	virtual void ApplyForces() {}

	virtual void SimulateMapUnmap()
	{
		g_buffers->rigidJoints.unmap();
		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		g_frame++;
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
		g_buffers->rigidBodies.map();
		g_buffers->rigidJoints.map();
	}

	virtual void ApplyTorqueControl(int agentIndex)
	{
		for (int i = agentBodies[agentIndex].first; i < (int)agentBodies[agentIndex].second; i++)
		{
			g_buffers->rigidBodies[i].force[0] = 0.0f;
			g_buffers->rigidBodies[i].force[1] = 0.0f;
			g_buffers->rigidBodies[i].force[2] = 0.0f;
			g_buffers->rigidBodies[i].torque[0] = 0.0f;
			g_buffers->rigidBodies[i].torque[1] = 0.0f;
			g_buffers->rigidBodies[i].torque[2] = 0.0f;
		}

		float* actions = GetAction(agentIndex);
		for (int i = 0; (unsigned int)i < ctrls[agentIndex].size(); i++)
		{
			float cc = Clamp(actions[i], -1.f, 1.f);

			NvFlexRigidJoint& j = initJoints[ctrls[agentIndex][i].first];
			NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
			NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
			Transform& pose0 = *((Transform*)&j.pose0);
			Transform gpose;
			NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
			Transform tran = gpose * pose0;

			Vec3 axis;
			if (ctrls[agentIndex][i].second == eNvFlexRigidJointAxisTwist)
			{
				axis = GetBasisVector0(tran.q);
			}
			else
			if (ctrls[agentIndex][i].second == eNvFlexRigidJointAxisSwing1)
			{
				axis = GetBasisVector1(tran.q);
			}
			else
			if (ctrls[agentIndex][i].second == eNvFlexRigidJointAxisSwing2)
			{
				axis = GetBasisVector2(tran.q);
			}
			else {
				printf("Invalid axis, probably bad code migration?\n");
				exit(0);
			}

			Vec3 torque = axis * motorPower[agentIndex][i] * cc * powerScale;
			a0.torque[0] += torque.x;
			a0.torque[1] += torque.y;
			a0.torque[2] += torque.z;
			a1.torque[0] -= torque.x;
			a1.torque[1] -= torque.y;
			a1.torque[2] -= torque.z;
		}
	}

	virtual void ApplyTargetControl(int agentIndex)
	{
		for (int i = agentBodies[agentIndex].first; i < (int)agentBodies[agentIndex].second; i++)
		{
			g_buffers->rigidBodies[i].force[0] = 0.0f;
			g_buffers->rigidBodies[i].force[1] = 0.0f;
			g_buffers->rigidBodies[i].force[2] = 0.0f;
			g_buffers->rigidBodies[i].torque[0] = 0.0f;
			g_buffers->rigidBodies[i].torque[1] = 0.0f;
			g_buffers->rigidBodies[i].torque[2] = 0.0f;
		}

		vector<float> angles(mNumActions, 0.f);
		vector<float> lows(mNumActions, 0.f);
		vector<float> highs(mNumActions, 0.f);
		GetAngles(agentIndex, angles, lows, highs);

		float* actions = GetAction(agentIndex);
		for (int i = 0; (unsigned int)i < ctrls[agentIndex].size(); i++)
		{
			//	cout << "action " << i << " = " << actions[i] << endl;
			float cc = Clamp(actions[i], -1.f, 1.f);
			//	cout << cc << endl;

			if ((highs[i] - lows[i]) < 0.001f)
			{
				float maxVelocity = 1.6f;

				float targetVel = (cc + 1.f) * maxVelocity - maxVelocity;
				g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] += targetVel;
			}
			else
			{
				float targetPos = 0.5f * (cc + 1.f) * (highs[i] - lows[i]) + lows[i];
				//	cout << "targetPos = " << targetPos << endl;
				//	cout << "low = " << lows[i] << " " << "high = " << highs[i] << endl;

				float smoothness = 0.02f;
				float pos = Lerp(angles[i], targetPos, smoothness);
				//	cout << "pos = " << pos << endl;
				g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] = pos;
			}
		}
	}

	virtual void ApplyVelocityControl(int agentIndex, float df = 1.f) {}

	virtual void ApplyInverseDynamicsControl(int agentIndex) {}

	virtual void ApplyActions()
	{
		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			if (controlType == Control::eTorque)
			{
				ApplyTorqueControl(ai);
			}
			else if (controlType == Control::ePosition)
			{
				ApplyTargetControl(ai);
			}
			else if (controlType == Control::eVelocity)
			{
				ApplyVelocityControl(ai);
			}
			else if (controlType == Control::eInverseDynamics)
			{
				ApplyInverseDynamicsControl(ai);
			}
			else
			{
				cout << "Unknown control type" << endl;
			}
		}
	}

	// What if control not torque, but velocity, position or IK/ID or MTU?
	// Add enum of control types and abstract function?
	virtual void Simulate()
	{
		ApplyActions();

		ApplyForces();

		SimulateMapUnmap();
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual bool IsSkipSimulation()
	{
		return true;
	}

	void getEulerZYX(Quat& q, float& yawZ, float& pitchY, float& rollX)
	{
		float squ;
		float sqx;
		float sqy;
		float sqz;
		float sarg;
		sqx = q.x * q.x;
		sqy = q.y * q.y;
		sqz = q.z * q.z;
		squ = q.w * q.w;

		rollX = atan2(2 * (q.y * q.z + q.w * q.x), squ - sqx - sqy + sqz);
		sarg = (-2.0f) * (q.x * q.z - q.w * q.y);
		pitchY = sarg <= (-1.0f) ? (-0.5f) * PxPi : (sarg >= (1.0f) ? (0.5f) * PxPi : asinf(sarg));
		yawZ = atan2(2 * (q.x * q.y + q.w * q.z), squ + sqx - sqy - sqz);
	}

	void GetAngles(int a, float* angles, float* lows, float* highs)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;

		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = 0.f;
			float low = 0.f;
			float high = 0.f;

			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
				Transform body1Pose;
				NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

				Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
				Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
				Transform relPose = Inverse(pose0) * pose1;

				prevPos = relPose.p;
				Quat qd = relPose.q;
				if (qd.w < 0)
				{
					qd *= -1.f;
				}

				Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
				Quat qswing = qd * Inverse(qtwist);
				prevTwist = asin(qtwist.x) * 2.f;
				prevSwing1 = asin(qswing.y) * 2.f;
				prevSwing2 = asin(qswing.z) * 2.f;
				prevIdx = ctrls[a][qq].first;

				// If the same, no need to recompute
			}

			NvFlexRigidJointAxis idx = ctrls[a][qq].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];
			if (idx == eNvFlexRigidJointAxisTwist)
			{
				pos = prevTwist;
			}
			else if (idx == eNvFlexRigidJointAxisSwing1)
			{
				pos = prevSwing1;
			}
			else if (idx == eNvFlexRigidJointAxisSwing2)
			{
				pos = prevSwing2;
			}
			else if (idx == eNvFlexRigidJointAxisX)
			{
				pos = prevPos.x;
			}
			else if (idx == eNvFlexRigidJointAxisY)
			{
				pos = prevPos.y;
			}
			else if (idx == eNvFlexRigidJointAxisZ)
			{
				pos = prevPos.z;
			}

			angles[i] = pos;
			lows[i] = low;
			highs[i] = high;
		}
	}

	void GetAngles(int a, vector<float>& angles, vector<float>& lows, vector<float>& highs)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;

		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = 0.f;
			float low = 0.f;
			float high = 0.f;

			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
				Transform body1Pose;
				NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

				Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
				Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
				Transform relPose = Inverse(pose0) * pose1;

				prevPos = relPose.p;
				Quat qd = relPose.q;
				if (qd.w < 0)
				{
					qd *= -1.f;
				}

				Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
				Quat qswing = qd*Inverse(qtwist);
				prevTwist = asin(qtwist.x) * 2.f;
				prevSwing1 = asin(qswing.y) * 2.f;
				prevSwing2 = asin(qswing.z) * 2.f;
				prevIdx = ctrls[a][qq].first;

				// If the same, no need to recompute
			}

			NvFlexRigidJointAxis idx = ctrls[a][qq].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];
			if (idx == eNvFlexRigidJointAxisTwist)
			{
				pos = prevTwist;
			}
			else if (idx == eNvFlexRigidJointAxisSwing1)
			{
				pos = prevSwing1;
			}
			else if (idx == eNvFlexRigidJointAxisSwing2)
			{
				pos = prevSwing2;
			}
			else if (idx == eNvFlexRigidJointAxisX)
			{
				pos = prevPos.x;
			}
			else if (idx == eNvFlexRigidJointAxisY)
			{
				pos = prevPos.y;
			}
			else if (idx == eNvFlexRigidJointAxisZ)
			{
				pos = prevPos.z;
			}

			angles[i] = pos;
			lows[i] = low;
			highs[i] = high;
		}
	}

	void GetAngles(int a, vector<float>& angles)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;

		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = 0.f;

			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
				Transform body1Pose;
				NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

				Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
				Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
				Transform relPose = Inverse(pose0) * pose1;

				prevPos = relPose.p;
				Quat qd = relPose.q;
				if (qd.w < 0)
				{
					qd *= -1.f;
				}

				Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
				Quat qswing = qd * Inverse(qtwist);
				prevTwist = asin(qtwist.x) * 2.f;
				prevSwing1 = asin(qswing.y) * 2.f;
				prevSwing2 = asin(qswing.z) * 2.f;
				prevIdx = ctrls[a][qq].first;

				// If the same, no need to recompute
			}

			NvFlexRigidJointAxis idx = ctrls[a][qq].second;
			if (idx == eNvFlexRigidJointAxisTwist)
			{
				pos = prevTwist;
			}
			else if (idx == eNvFlexRigidJointAxisSwing1)
			{
				pos = prevSwing1;
			}
			else if (idx == eNvFlexRigidJointAxisSwing2)
			{
				pos = prevSwing2;
			}
			else if (idx == eNvFlexRigidJointAxisX)
			{
				pos = prevPos.x;
			}
			else if (idx == eNvFlexRigidJointAxisY)
			{
				pos = prevPos.y;
			}
			else if (idx == eNvFlexRigidJointAxisZ)
			{
				pos = prevPos.z;
			}

			angles[i] = pos;
		}
	}

	void GetJointLimits(vector<float>& lows, vector<float>& highs)
	{
		Vec3 prevPos;

		float low, high;
		for (int i = 0; (unsigned int)i < ctrls[0].size(); i++)
		{
			NvFlexRigidJoint& joint = initJoints[ctrls[0][i].first];

			NvFlexRigidJointAxis idx = ctrls[0][i].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];

			lows[i] = low;
			highs[i] = high;
		}
	}

	void GetJointLimits(vector<float>& jLimits)
	{
		Vec3 prevPos;

		float low, high;
		for (int i = 0; (unsigned int)i < ctrls[0].size(); i++)
		{
			NvFlexRigidJoint& joint = initJoints[ctrls[0][i].first];

			NvFlexRigidJointAxis idx = ctrls[0][i].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];

			jLimits[2 * i] = low;
			jLimits[2 * i + 1] = high;
		}
	}

	virtual void GetGlobalPose(int a, Transform* trans)
	{
		pair<int, int> p = agentBodies[a];
		Transform& inv = agentOffsetInv[a];

		int ind = 0;
		for (int i = p.first; i < p.second; i++)
		{
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&pose);
			trans[ind++] = inv * pose;
		}
	}

	virtual void GetGlobalPose(int a, vector<Transform>& trans)
	{
		pair<int, int> p = agentBodies[a];
		Transform& inv = agentOffsetInv[a];

		int ind = 0;
		for (int i = p.first; i < p.second; i++)
		{
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&pose);
			trans[ind++] = inv * pose;
		}
	}

	virtual Vec3 GetLinearVel(int a, int index)
	{
		Transform& inv = agentOffsetInv[a];
		pair<int, int> p = agentBodies[a];
		return Rotate(inv.q, Vec3(g_buffers->rigidBodies[p.first + index].linearVel));
	}

	// ?
	virtual Vec3 GetAngularVel(int a, int index)
	{
		pair<int, int> p = agentBodies[a];
		return Vec3(g_buffers->rigidBodies[p.first + index].angularVel);
	}

	int GetNumBodies()
	{
		return agentBodies[0].second - agentBodies[0].first;
	}

	int GetNumControls()
	{
		return ctrls[0].size();
	}

	virtual void ClearContactInfo() = 0;
};

class RLMultiGoalEnv
{
public:
	float goalThreshold;
	RLMultiGoalEnv()
	{
		goalThreshold = 0.05f;
	}
	virtual int GetNumGoals() = 0; // return the dimension of a goal
	virtual void GetDesiredGoal(int ai, float* goal) = 0;
	virtual void GetAchievedGoal(int ai, float* goal) = 0;
	// Default way to evaluate goal dist is l2 dist of Vec3
	virtual float GetGoalDist(float* desiredGoal, float* achievedGoal)
	{
		return Length(Vec3(desiredGoal[0] - achievedGoal[0], desiredGoal[1] - achievedGoal[1], desiredGoal[2] - achievedGoal[2]));
	}

	virtual bool IsSuccess(float* desiredGoal, float* achievedGoal)
	{
		return GetGoalDist(desiredGoal, achievedGoal) < goalThreshold;
	}
	int _GetNumExtras()
	{
		return 2 * GetNumGoals() + 1;
	}
	void _PopulateExtra(int ai, float* extra)
	{
		float* desiredGoal = extra;
		float* achievedGoal = &extra[GetNumGoals()];
		GetDesiredGoal(ai, desiredGoal);
		GetAchievedGoal(ai, achievedGoal);
		bool success = IsSuccess(desiredGoal, achievedGoal);
		extra[GetNumGoals() * 2] = success ? 1.f : 0.f;
	}
};


class HumanoidBase : public FlexGymBase2
{
public:
		
	int numFeet;
	bool doFlagRun;
	bool variableVelocity;
	float maxTargetVelocity;

	// Cost coefficients, used in reward calculation, penalize some kind of undesired activities
	float electricityCostScale;
	float electricityCost;		//    # cost for using motors-- this parameter should be carefully tuned against reward for making progress, other values less improtant
	float stallTorqueCostScale;
	float stallTorqueCost;		//    # cost for running electric current through a motor even at zero rotational speed, small
	float footCollisionCost;	//    # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
	// foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
	float jointsAtLimitCost;	// # discourage stuck joints

	//	float initialZ;
	float terminationZ;

	float pitch, r, yaw;

	float maxX, maxY, maxDist; // Max size for randomply generated flag

	vector<MJCFImporter*> mjcfs;

	vector<float> potentialsOld;
	vector<float> potentials;

	vector<int> pelvis;
	vector<int> head;
	vector<int> hands;
	vector<int> feet;
	vector<int> knees;
	vector<int> toes;

	vector<int> handFlag;
	vector<int> kneeFlag;
	vector<int> footFlag;
	vector<int> toeFlag;

	vector<Vec3> handsContacts;
	vector<Vec3> kneesContacts;
	vector<Vec3> feetContacts;
	vector<Vec3> toesContacts;

	vector<int> numCollideOther;

	vector<float> ps;
	vector<float> walkTargetDists;
	vector<float> velocityTargets;

	vector<float> headings;
	vector<float> upVecs;

	vector<float> walkTargetX;
	vector<float> walkTargetY;

	vector<float> masses;

	vector<Vec2> walkTarget;

	vector<Vec3> bodyXYZ;

	vector<int> flagRunSteps;

	int maxFlagResetSteps;
	float upVecWeight;

	HumanoidBase() : numFeet(2), doFlagRun(false), variableVelocity(false),
		maxTargetVelocity(15.f),
		electricityCostScale(2.f), electricityCost(-2.f),
		stallTorqueCostScale(5.f), stallTorqueCost(-0.1f),
		footCollisionCost(-1.f), jointsAtLimitCost(-0.2f),
		/*initialZ(0.8f),*/ terminationZ(0.79f),
		maxX(50.f), maxY(50.f), maxDist(50.f),
		maxFlagResetSteps(200),
		upVecWeight(0.05f)
	{
	};

	// Need to implement these functions
	virtual void FinalizeContactInfo() = 0;
	virtual float AliveBonus(float z, float pitch) = 0;

	virtual void LoadRLState(FILE* f)
	{
		FlexGymBase2::LoadRLState(f);
		LoadVec(f, potentialsOld);
		LoadVec(f, potentials);

		LoadVec(f, pelvis);
		LoadVec(f, head);
		LoadVec(f, hands);
		LoadVec(f, feet);
		LoadVec(f, knees);
		LoadVec(f, toes);

		LoadVec(f, handFlag);
		LoadVec(f, kneeFlag);
		LoadVec(f, footFlag);
		LoadVec(f, toeFlag);

		LoadVec(f, handsContacts);
		LoadVec(f, kneesContacts);
		LoadVec(f, feetContacts);
		LoadVec(f, toesContacts);

		LoadVec(f, numCollideOther);

		LoadVec(f, ps);
		LoadVec(f, walkTargetDists);
		LoadVec(f, velocityTargets);

		LoadVec(f, headings);
		LoadVec(f, upVecs);

		LoadVec(f, walkTargetX);
		LoadVec(f, walkTargetY);

		LoadVec(f, walkTarget);

		LoadVec(f, bodyXYZ);

		LoadVec(f, flagRunSteps);
	}

	virtual void SaveRLState(FILE* f)
	{
		FlexGymBase2::SaveRLState(f);
		SaveVec(f, potentialsOld);
		SaveVec(f, potentials);

		SaveVec(f, pelvis);
		SaveVec(f, head);
		SaveVec(f, hands);
		SaveVec(f, feet);
		SaveVec(f, knees);
		SaveVec(f, toes);

		SaveVec(f, handFlag);
		SaveVec(f, kneeFlag);
		SaveVec(f, footFlag);
		SaveVec(f, toeFlag);

		SaveVec(f, handsContacts);
		SaveVec(f, kneesContacts);
		SaveVec(f, feetContacts);
		SaveVec(f, toesContacts);

		SaveVec(f, numCollideOther);

		SaveVec(f, ps);
		SaveVec(f, walkTargetDists);
		SaveVec(f, velocityTargets);

		SaveVec(f, headings);
		SaveVec(f, upVecs);

		SaveVec(f, walkTargetX);
		SaveVec(f, walkTargetY);

		SaveVec(f, walkTarget);

		SaveVec(f, bodyXYZ);

		SaveVec(f, flagRunSteps);
	}

	virtual void LoadEnv()
	{
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		head.clear();
		torso.clear();
		pelvis.clear();
		hands.clear();
		knees.clear();
		feet.clear();
		toes.clear();

		for (int i = 0; i < mNumAgents; i++)
		{
			Vec3 pos = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f);

			Vec3 posStart = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rotStart = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));

			posStart.y += Randf(-0.05f, 0.0f);

			rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * rot;
			rotStart = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise))
					   * rotStart;

			Transform gtStart(posStart, rotStart);
			gtStart = gtStart * preTransform;

			Transform gt(pos, rot);
			gt = gt * preTransform;

			int begin = g_buffers->rigidBodies.size();

			AddAgentBodiesJointsCtlsPowers(i, gtStart, ctrls[i], motorPower[i]);

			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
			agentStartOffset.push_back(gtStart);

			// Todo - extract masses here

			int end = g_buffers->rigidBodies.size();
			agentBodies.push_back(make_pair(begin, end));
		}

		maxPower = *max_element(std::begin(motorPower[0]), std::end(motorPower[0]));

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());
	}

	virtual void RandomPush(const vector<int>& bodyToPush, const int ai, const float upScale = 0.2f)
	{
		if (bodyToPush[ai] != -1)
		{
			Transform torsoPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[bodyToPush[ai]], (NvFlexRigidPose*)&torsoPose);

			Vec3 pushForce = forceMag * RandomUnitVector();

			pushForce.z *= upScale;

			g_buffers->rigidBodies[bodyToPush[ai]].force[0] += pushForce.x;
			g_buffers->rigidBodies[bodyToPush[ai]].force[1] += pushForce.y;
			g_buffers->rigidBodies[bodyToPush[ai]].force[2] += pushForce.z;
		}
		else
		{
			cout << "Body the push should be applying is missing" << endl;
		}
	}

	virtual void ApplyForces()
	{
		// Random push to some body during training
		int pushNum = Rand(0, pushFrequency - 1);
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			if (ai % pushFrequency == pushNum)
			{
				RandomPush(torso, ai);
			}
		}
	}

	virtual void resetTarget(int a, bool firstTime = true)
	{
		if (doFlagRun)
		{
			if (firstTime && (a % 5))
			{
				walkTarget[a] = Vec2(1000.f, 0.f);

				// Obsolete, will remove them
				walkTargetX[a] = 1000.f;
				walkTargetY[a] = 0.f;
			}
			else
			{
				float alpha = Randf(0.f, 2.f * kPi);
				float rad = Randf(0.f, maxDist);

				walkTarget[a] = Vec2(rad * cos(alpha) + bodyXYZ[a].x, rad * sin(alpha) + bodyXYZ[a].y);

				// Obsolete, will remove them
				walkTargetX[a] = rad * cos(alpha) + bodyXYZ[a].x;
				walkTargetY[a] = rad * sin(alpha) + bodyXYZ[a].y;
			}

			flagRunSteps[a] = 0;
		}
		else
		{
			walkTarget[a] = Vec2(1000.f, 0.f);

			// Obsolete, will remove them
			walkTargetX[a] = 1000.f;
			walkTargetY[a] = 0.f;
		}
	}

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		// Prepare state
		//--------------------
		numJointsAtLimit = 0;

		// Perf?
		vector<float> joints(mNumActions * 2, 0.f);
		vector<float> angles(mNumActions, 0.f);
		vector<float> lows(mNumActions, 0.f);
		vector<float> highs(mNumActions, 0.f);

		int numBodies = GetNumBodies();

		GetAngles(a, angles, lows, highs);
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = angles[i];
			float low = lows[i], high = highs[i];

			if (prevAngleValue[a][qq] > 1e8)
			{
				prevAngleValue[a][qq] = pos;
			}

			joinVelocities[a][qq] = (pos - prevAngleValue[a][qq]) / dt;

			prevAngleValue[a][qq] = pos;

			float posMid = 0.5f * (low + high);
			pos = 2.f * (pos - posMid) / (high - low);

			joints[2 * i] = pos;
			joints[2 * i + 1] = joinVelocities[a][qq] * 0.2f;

			jointSpeeds[i] = joints[2 * i + 1];
			if (fabs(joints[2 * i]) > 0.99f)
			{
				numJointsAtLimit++;
			}
		}

		vector<Transform> bodies(numBodies);

		GetGlobalPose(a, bodies);
		Transform bodyPose = bodies[0];

		upVec = GetBasisVector2(bodyPose.q).z;

		Vec3 sum(0.f);
		float totalMass = 0.f;
		for (int i = 0; i < numBodies; i++)
		{
			float bodyMass = masses[i];

			Vec3 pos = bodies[i].p * bodyMass;
			sum += pos;

			totalMass += bodyMass;
		}
		bodyXYZ[a] = Vec3(sum.x / totalMass, sum.y / totalMass, bodyPose.p.z);

		getEulerZYX(bodyPose.q, yaw, p, r);
		float z = bodyXYZ[a].z;

		Vec3 toTarget = Vec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
		walkTargetDist = Length(toTarget);

		if (doFlagRun)
		{
			if (flagRunSteps[a] > maxFlagResetSteps || walkTargetDist < 1.f)
			{
				resetTarget(a, false);
				toTarget = Vec3(walkTargetX[a] - bodyXYZ[a].x, walkTargetY[a] - bodyXYZ[a].y, 0.0f);
				walkTargetDist = Length(toTarget);

				potentialsOld[a] = -walkTargetDist / dt;
				potentials[a] = -walkTargetDist / dt;
			}
			flagRunSteps[a]++;
		}

		float walkTargetTheta = atan2(walkTargetY[a] - bodyXYZ[a].y, walkTargetX[a] - bodyXYZ[a].x);

		toTarget = Normalize(toTarget);
		heading = Dot(GetBasisVector0(bodyPose.q), toTarget);
		float angleToTarget = walkTargetTheta - yaw;

		Matrix33 mat = Matrix33(
						   Vec3(cos(-yaw), sin(-yaw), 0.0f),
						   Vec3(-sin(-yaw), cos(-yaw), 0.0f),
						   Vec3(0.0f, 0.0f, 1.0f));

		Vec3 vel = GetLinearVel(a, 0);
		Vec3 bvel = mat * vel;
		float vx = bvel.x;
		float vy = bvel.y;
		float vz = bvel.z;

		float avx = bvel.x;
		float avy = bvel.y;
		float avz = bvel.z;

		vector<float> more = { z,
							   sin(angleToTarget), cos(angleToTarget),
							   0.25f * vx, 0.25f * vy, 0.25f * vz,
							   0.25f * avx, 0.25f * avy, 0.25f * avz,
							   r, p
							 };

		int ct = 0;
		for (int i = 0; i < 8; i++)
		{
			state[ct++] = more[i];
		}

		for (int i = 0; (unsigned int)i < ctrls[a].size() * 2; i++)
		{
			state[ct++] = joints[i];
		}

		float* prevActions = GetAction(a);
		if (prevActions) // could be null if this is called for the first time before agent acts
		{
			for (int i = 0; i < mNumActions; ++i)
			{
				state[ct++] = prevActions[i];
			}
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = feetContacts[2 * a + i].x;
			state[ct++] = feetContacts[2 * a + i].y;
			state[ct++] = feetContacts[2 * a + i].z;
		}

		for (int i = 0; i < ct; i++)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}
	}

	void init(PPOLearningParams& ppoParams, const char* pythonFile, const char* workingDir, const char* folder, float deltat = 1.f / 60.f)
	{
		FlexGymBase2::init(ppoParams, pythonFile, workingDir, folder, deltat);

		potentialsOld.resize(mNumAgents);
		potentials.resize(mNumAgents);

		ps.resize(mNumAgents);
		walkTargetDists.resize(mNumAgents);

		headings.resize(mNumAgents);
		upVecs.resize(mNumAgents);
		walkTargetX.resize(mNumAgents);
		walkTargetY.resize(mNumAgents);
		bodyXYZ.resize(mNumAgents);

		feetContacts.resize(2 * mNumAgents);
		// Add others

		numCollideOther.resize(mNumAgents);
		prevAngleValue.resize(mNumAgents);

		flagRunSteps.clear();
		flagRunSteps.resize(mNumAgents, 0);

		for (int a = 0; a < mNumAgents; a++)
		{
			potentialsOld[a] = 0.0f;
			potentials[a] = 0.0f;
			bodyXYZ[a] = Vec3(0.f, 0.f, 0.f);
			resetTarget(a, true);
			//	for (int fi = 0; fi < numFeet; ++fi)
			//	{
			//		feetContact[numFeet * a + fi] = 0.f;
			//	}

			numCollideOther[a] = 0;
		}
	}

	virtual void PopulateState(int a, float* state)
	{
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& jointsAtLimitA = jointsAtLimits[a];
		float& heading = headings[a];
		float& upVec = upVecs[a];

		ExtractState(a, state, p, walkTargetDist, jointSpeedsA, jointsAtLimitA, heading, upVec);
	}

	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float& potential = potentials[a];
		float& potentialOld = potentialsOld[a];
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& numJointsAtLimitA = jointsAtLimits[a];
		float& heading = headings[a];
		float& upVec = upVecs[a];

		float electrCost = electricityCostScale * electricityCost;
		float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

		float alive = AliveBonus(state[0], p); //  # state[0] is body height above ground, body_rpy[1] is pitch
		dead = alive < 0.f;

		potentialOld = potential;
		potential = -walkTargetDist / dt;
		if (potentialOld > 1e9)
		{
			potentialOld = potential;
		}

		float progress = potential - potentialOld;
		if (progress > 100.f)
		{
			printf("progress is infinite %f %f %f \n", progress, potential, potentialOld);
		}

		float electricityCostCurrent = 0.0f;
		float sum = 0.0f;
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			float vv = abs(action[i] * jointSpeedsA[i]) * motorPower[a][i] / maxPower; // Take motor power into account
			if (!isfinite(vv))
			{
				printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv, action[i], jointSpeedsA[i]);
			}

			if (!isfinite(action[i]))
			{
				printf("action at %d is infinite\n", i);
			}

			if (!isfinite(jointSpeedsA[i]))
			{
				printf("jointSpeeds at %d is infinite\n", i);
			}

			sum += vv;
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl*joint_speed is infinite!\n");
		}

		electricityCostCurrent += electrCost * sum / (float)ctrls[a].size();

		sum = 0.0f;
		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			sum += action[i] * action[i];
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
		}

		//electricity_costCurrent += stall_torque_cost * float(np.square(a).mean())
		electricityCostCurrent += stallTorqCost * sum / (float)ctrls[a].size();

		float jointsAtLimitCostCurrent = jointsAtLimitCost * (float)numJointsAtLimitA;

		float feetCollisionCostCurrent = 0.0f;
		if (numCollideOther[a] > 0)
		{
			feetCollisionCostCurrent += footCollisionCost;
		}

		//	cout << "heading = " << heading << endl;
		float headingRew = 0.5f * ((heading > 0.8f) ? 1.f : heading / 0.8f) + upVecWeight * ((upVec > 0.93f) ? 1.f : 0.f); // MJCF4
		float rewards[6] =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent,
			headingRew
		};
		//cout <<"rewards: "<< rewards[0]<<" "<< rewards[1]<<" "<< rewards[2]<<" "<< rewards[3]<<" "<<rewards[4]<<" "<<rewards[5]<<" "<<rewards[5]<<endl;

		rew = 0.f;
		for (int i = 0; i < 6; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}
	}

	virtual void ResetAllAgents()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			ResetAgent(a);
		}
		RLFlexEnv::ResetAllAgents(); // Duplicate, fix!
	}

	virtual void ResetAgent(int a)
	{
	//	mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);

		potentialsOld[a] = 1e10f;
		potentials[a] = 1e10f;

		for (int i = 0; i < (int)prevAngleValue[a].size(); i++)
		{
			prevAngleValue[a][i] = 1e9f;
		}

		resetTarget(a, true);

		numCollideOther[a] = 0;
		RLFlexEnv::ResetAgent(a);
	}

	// Remove from here
	virtual void ClearContactInfo()
	{
		for (int ai = 0; ai < mNumAgents; ++ai)
		{
			for (int i = 0; i < 2; ++i)
			{
				handsContacts[2 * ai + i] = 0.f;
				kneesContacts[2 * ai + i] = 0.f;
				feetContacts[2 * ai + i] = 0.f;
				toesContacts[2 * ai + i] = 0.f;
			}
			numCollideOther[ai] = 0;
		}
	}

	virtual void PreHandleCommunication()
	{
		FinalizeContactInfo();
	}
};

// Should all this stuff go to the already existingbase FlexEnv class
class CarterBase : public CommonFlexGymBase
{
public:
	enum FileType
	{
		eMujoco,
		eURDF,
		eSDF,
	};
	FileType fileType;

	enum Control
	{
		eTorque,
		ePosition,
		eVelocity,
		eInverseDynamics,
	};
	Control controlType;

	CarterBase() : rigidContacts(g_flexLib, g_solverDesc.maxRigidBodyContacts), rigidContactCount(g_flexLib, 1)
	{
		controlType = Control::eInverseDynamics;

		mNumAgents = 500;
		mNumObservations = 44;
		mNumActions = 17;
		mMaxEpisodeLength = 1000;

		g_numSubsteps = 2;
		g_params.numIterations = 30;

		g_sceneLower = Vec3(-50.f, 0.f, -50.f);
		g_sceneUpper = Vec3(120.f, 10.f, 100.f);

		g_pause = true;
		mDoLearning = g_doLearning;

		numRenderSteps = 1;

		numPerRow = 25;
		spacing = 20.f;

		powerScale = 1.f;
		maxPower = 100.f;

		angleResetNoise = 0.f;
		angleVelResetNoise = 0.f;
		velResetNoise = 0.f;

		pushFrequency = 500;	// How much steps in average per 1 kick
		forceMag = 0.0f;		// Remove from base class ?
		yOffset = 0.0f;

		preTransform = Transform(Vec3(), Quat());
		rotCreateNoise = 0.05f;

		dt = 1.f / 60.f;
	}

	// Need to implement these functions
	virtual void FinalizeContactInfo() = 0;

	Transform preTransform;
	string loadPath;

	vector<pair<int, int> > agentBodies;
	vector<NvFlexRigidBody> initBodies;
	vector<NvFlexRigidJoint> initJoints;
	vector<vector<float>> prevAngleValue;
	vector<vector<float>> joinVelocities;
	vector<vector<float>> jointSpeeds;
	vector<int> jointsAtLimits;

	vector<float> bodyMasses;

	// Transforms from initial
	vector<Transform> agentOffsetInv;
	vector<Transform> agentOffset;

	// Start transforms with randomization
	vector<Transform> agentStartOffset;

	vector<vector<pair<int, NvFlexRigidJointAxis>>> ctrls;
	vector<vector<float>> motorPower;
	float powerScale;
	float maxPower;

	vector<int> torso; // Move torso to the ansestor too?

	NvFlexVector<NvFlexRigidContact> rigidContacts;
	NvFlexVector<int> rigidContactCount;

	vector<vector<Transform>> initTrans; // Do we need them?

	bool mDoLearning; // Rename
	int numRenderSteps;

	// Randomization
	float angleResetNoise;
	float angleVelResetNoise;
	float velResetNoise;
	float rotCreateNoise;

	// How much steps in average per 1 kick
	int pushFrequency; // Move to ancestor?
	float forceMag;

	int numPerRow;
	float spacing;
	float yOffset;

	float dt;
	virtual void LoadRLState(FILE* f)
	{
		CommonFlexGymBase::LoadRLState(f);
		LoadVecVec(f, prevAngleValue);
		LoadVecVec(f, joinVelocities);
		LoadVecVec(f, jointSpeeds);
		LoadVec(f, jointsAtLimits);
		LoadVec(f, bodyMasses);
	}

	virtual void SaveRLState(FILE* f)
	{
		CommonFlexGymBase::SaveRLState(f);
		SaveVecVec(f, prevAngleValue);
		SaveVecVec(f, joinVelocities);
		SaveVecVec(f, jointSpeeds);
		SaveVec(f, jointsAtLimits);
		SaveVec(f, bodyMasses);
	}

	void ParseJsonParams(const json& sceneJson) override
	{
		if (sceneJson.is_null())
		{
			return;
		}
		RLFlexEnv::ParseJsonParams(sceneJson);

		// Parsing of common JSON parameters
		loadPath = sceneJson.value(RL_JSON_LOAD_PATH, loadPath);

		g_params.solverType = sceneJson.value(RL_JSON_SOLVER_TYPE, g_params.solverType);
		g_numSubsteps = sceneJson.value(RL_JSON_NUM_SUBSTEPS, g_numSubsteps);
		g_params.numIterations = sceneJson.value(RL_JSON_NUM_ITERATIONS, g_params.numIterations);
		g_params.numInnerIterations = sceneJson.value(RL_JSON_NUM_INNER_ITERATIONS, g_params.numInnerIterations);
		g_params.numPostCollisionIterations = sceneJson.value(RL_JSON_NUM_POST_COLLISION_ITERATIONS, g_params.numPostCollisionIterations);

		g_params.warmStart = GetJsonVal(g_sceneJson, RL_JSON_WARMSTART, 0.0f);
		// SOR
		g_params.relaxationFactor = GetJsonVal(g_sceneJson, "RelaxationFactor", 0.75f);
		g_params.systemRegularization = GetJsonVal(g_sceneJson, "SystemRegularization", 1e-6f);

		g_pause = sceneJson.value(RL_JSON_PAUSE, g_pause);

		mDoLearning = sceneJson.value(RL_JSON_DO_LEARNING, mDoLearning);
		numRenderSteps = sceneJson.value(RL_JSON_NUM_RENDER_STEPS, numRenderSteps);

		numPerRow = sceneJson.value(RL_JSON_NUM_PER_ROW, numPerRow);
		spacing = sceneJson.value(RL_JSON_SPACING, spacing);

		angleResetNoise = sceneJson.value(RL_JSON_ANGLE_RESET_NOISE, angleResetNoise);
		angleVelResetNoise = sceneJson.value(RL_JSON_ANGLE_VELOCITY_RESET_NOISE, angleVelResetNoise);
		velResetNoise = sceneJson.value(RL_JSON_VELOCITY_RESET_NOISE, velResetNoise);

		pushFrequency = sceneJson.value(RL_JSON_PUSH_FREQUENCY, pushFrequency);
		forceMag = sceneJson.value(RL_JSON_FORCE_MAGNITUDE, forceMag);
	}

	void init(PPOLearningParams& ppoParams, const char* pythonFile, const char* workingDir, const char* logDir, float deltat = 1.f / 60.f)
	{
		dt = deltat;

		if (logDir)
		{
			EnsureDirExists(workingDir + string("/") + logDir);
		}

		InitRLInfo();
		LaunchPythonProcess(pythonFile, workingDir, logDir, ppoParams, g_sceneJson);

		prevAngleValue.resize(mNumAgents);
		joinVelocities.resize(mNumAgents);
		jointSpeeds.resize(mNumAgents);

		jointsAtLimits.resize(mNumAgents);

		for (int a = 0; a < mNumAgents; a++)
		{
			prevAngleValue[a].resize(mNumActions);
			joinVelocities[a].resize(mNumActions);
			jointSpeeds[a].resize(mNumActions);
		}
	}

	void init(float deltat = 1.f / 60.f)
	{
		dt = deltat;

		InitRLInfo();

		// prevAngleValue.resize(mNumAgents);
		// joinVelocities.resize(mNumAgents);
		// jointSpeeds.resize(mNumAgents);

		// jointsAtLimits.resize(mNumAgents);

		// for (int a = 0; a < mNumAgents; a++)
		// {
		// 	prevAngleValue[a].resize(mNumActions);
		// 	joinVelocities[a].resize(mNumActions);
		// 	jointSpeeds[a].resize(mNumActions);
		// }
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpowers) = 0;

	virtual void LoadEnv() = 0;

	virtual void PreSimulation()
	{
		if (!mDoLearning)
		{
			if (!g_pause || g_step)
			{
				for (int s = 0; s < numRenderSteps; s++)
				{
					// tick solver
					NvFlexSetParams(g_solver, &g_params);
					NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
				}

				g_frame++;
				g_step = false;
			}
		}
		else
		{
			cout<<"HERE : mDoLearning"<<endl;
			g_buffers->rigidShapes.map();

			NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
			g_buffers->rigidBodies.map();

			NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
			g_buffers->rigidJoints.map();

			for (int s = 0; s < numRenderSteps; s++)
			{
#ifdef NV_FLEX_GYM
				cout<<"HERE :  calling Simulate()"<<endl;
				Simulate();
				FinalizeContactInfo();
				/*for (int a = 0; a < mNumAgents; ++a)
				{
					PopulateState(a, &mObsBuf[a * mNumObservations]);
					if (mNumExtras > 0) PopulateExtra(a, &mExtraBuf[a * mNumExtras]);
					ComputeRewardAndDead(a, GetAction(a), &mObsBuf[a * mNumObservations], mRewBuf[a], (bool&)mDieBuf[a]);
				}*/
#else
				HandleCommunication();
#endif
				ClearContactInfo();
			}

			g_buffers->rigidShapes.unmap();
			NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

			g_buffers->rigidJoints.unmap();
			NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

		}
	}

	virtual void ExtractState(int a, float* state, float* jointSpeeds, int& numJointsAtLimit)
	{
		// Prepare state
		//--------------------
		numJointsAtLimit = 0;

		// Perf?
		vector<float> joints(mNumActions * 2, 0.f);
		vector<float> angles(mNumActions, 0.f);
		vector<float> lows(mNumActions, 0.f);
		vector<float> highs(mNumActions, 0.f);

		GetAngles(a, angles, lows, highs);
		for (int i = 0; i < (int)ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = angles[i];
			float low = lows[i], high = highs[i];

			if (prevAngleValue[a][qq] > 1e8)
			{
				prevAngleValue[a][qq] = pos;
			}

			joinVelocities[a][qq] = (pos - prevAngleValue[a][qq]) / dt;

			prevAngleValue[a][qq] = pos;

			float posMid = 0.5f * (low + high);
			pos = 2.f * (pos - posMid) / (high - low);

			joints[2 * i] = pos;
			joints[2 * i + 1] = joinVelocities[a][qq] * 0.1f;

			jointSpeeds[i] = joints[2 * i + 1];
			if (fabs(joints[2 * i]) > 0.99f)
			{
				numJointsAtLimit++;
			}
		}

		int ct = 0;
		for (int i = 0; (unsigned int)i < ctrls[a].size() * 2; i++)
		{
			state[ct++] = joints[i];
		}

		for (int i = 0; i < ct; i++)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}
	}

	virtual void PopulateState(int a, float* state)
	{
		float* jointSpeedsA = &jointSpeeds[a][0];
		int& jointsAtLimitA = jointsAtLimits[a];

		ExtractState(a, state, jointSpeedsA, jointsAtLimitA);
	}

	virtual void PopulateExtra(int a, float* extra) {}

	// Move to ancestor? Make abstract to handle Mujoco, URDF, SDF?
	virtual void ResetAgent(int a) = 0;

	virtual void ResetAllAgents()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			ResetAgent(a);
		}
	}

	// Can be random pushes or constant, wind type force
	virtual void ApplyForces() {}

	virtual void SimulateMapUnmap()
	{
		cout<<"ENTERED SIMULATE MAP UNMAP"<<endl;
		g_buffers->rigidJoints.unmap();
		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		g_frame++;
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
		g_buffers->rigidBodies.map();
		g_buffers->rigidJoints.map();
	}

	virtual void ApplyTorqueControl(int agentIndex)
	{
		for (int i = agentBodies[agentIndex].first; i < (int)agentBodies[agentIndex].second; i++)
		{
			g_buffers->rigidBodies[i].force[0] = 0.0f;
			g_buffers->rigidBodies[i].force[1] = 0.0f;
			g_buffers->rigidBodies[i].force[2] = 0.0f;
			g_buffers->rigidBodies[i].torque[0] = 0.0f;
			g_buffers->rigidBodies[i].torque[1] = 0.0f;
			g_buffers->rigidBodies[i].torque[2] = 0.0f;
		}

		float* actions = GetAction(agentIndex);
		for (int i = 0; (unsigned int)i < ctrls[agentIndex].size(); i++)
		{
			float cc = Clamp(actions[i], -1.f, 1.f);

			NvFlexRigidJoint& j = initJoints[ctrls[agentIndex][i].first];
			NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
			NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
			Transform& pose0 = *((Transform*)&j.pose0);
			Transform gpose;
			NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
			Transform tran = gpose * pose0;

			Vec3 axis;
			if (ctrls[agentIndex][i].second == eNvFlexRigidJointAxisTwist)
			{
				axis = GetBasisVector0(tran.q);
			}
			else
			if (ctrls[agentIndex][i].second == eNvFlexRigidJointAxisSwing1)
			{
				axis = GetBasisVector1(tran.q);
			}
			else
			if (ctrls[agentIndex][i].second == eNvFlexRigidJointAxisSwing2)
			{
				axis = GetBasisVector2(tran.q);
			}
			else {
				printf("Invalid axis, probably bad code migration?\n");
				exit(0);
			}

			Vec3 torque = axis * motorPower[agentIndex][i] * cc * powerScale;
			a0.torque[0] += torque.x;
			a0.torque[1] += torque.y;
			a0.torque[2] += torque.z;
			a1.torque[0] -= torque.x;
			a1.torque[1] -= torque.y;
			a1.torque[2] -= torque.z;
		}
	}

	virtual void ApplyTargetControl(int agentIndex)
	{
		for (int i = agentBodies[agentIndex].first; i < (int)agentBodies[agentIndex].second; i++)
		{
			g_buffers->rigidBodies[i].force[0] = 0.0f;
			g_buffers->rigidBodies[i].force[1] = 0.0f;
			g_buffers->rigidBodies[i].force[2] = 0.0f;
			g_buffers->rigidBodies[i].torque[0] = 0.0f;
			g_buffers->rigidBodies[i].torque[1] = 0.0f;
			g_buffers->rigidBodies[i].torque[2] = 0.0f;
		}

		vector<float> angles(mNumActions, 0.f);
		vector<float> lows(mNumActions, 0.f);
		vector<float> highs(mNumActions, 0.f);
		GetAngles(agentIndex, angles, lows, highs);

		float* actions = GetAction(agentIndex);
		for (int i = 0; (unsigned int)i < ctrls[agentIndex].size(); i++)
		{
			//	cout << "action " << i << " = " << actions[i] << endl;
			float cc = Clamp(actions[i], -1.f, 1.f);
			//	cout << cc << endl;

			if ((highs[i] - lows[i]) < 0.001f)
			{
				float maxVelocity = 1.6f;

				float targetVel = (cc + 1.f) * maxVelocity - maxVelocity;
				g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] += targetVel;
			}
			else
			{
				float targetPos = 0.5f * (cc + 1.f) * (highs[i] - lows[i]) + lows[i];
				//	cout << "targetPos = " << targetPos << endl;
				//	cout << "low = " << lows[i] << " " << "high = " << highs[i] << endl;

				float smoothness = 0.02f;
				float pos = Lerp(angles[i], targetPos, smoothness);
				//	cout << "pos = " << pos << endl;
				g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] = pos;
			}
		}
	}

	virtual void ApplyVelocityControl(int agentIndex, float df = 1.f) {}

	virtual void ApplyInverseDynamicsControl(int agentIndex) {}

	virtual void ApplyActions()
	{
		cout<<"ENTERED APPLY ACTIONS!"<<endl;
		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			if (controlType == Control::eTorque)
			{
				cout<<"ENTERED TRQ!"<<endl;
				ApplyTorqueControl(ai);
			}
			else if (controlType == Control::ePosition)
			{
				ApplyTargetControl(ai);
			}
			else if (controlType == Control::eVelocity)
			{
				cout<<"ENTERED VEL!"<<endl;
				ApplyVelocityControl(ai);
			}
			else if (controlType == Control::eInverseDynamics)
			{
				ApplyInverseDynamicsControl(ai);
			}
			else
			{
				cout << "Unknown control type" << endl;
			}
		}
	}

	// What if control not torque, but velocity, position or IK/ID or MTU?
	// Add enum of control types and abstract function?
	virtual void Simulate()
	{
		cout<<"ENTERED SIMULATE!"<<endl;
		ApplyActions();

		ApplyForces();

		SimulateMapUnmap();
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual bool IsSkipSimulation()
	{
		return true;
	}

	void getEulerZYX(Quat& q, float& yawZ, float& pitchY, float& rollX)
	{
		float squ;
		float sqx;
		float sqy;
		float sqz;
		float sarg;
		sqx = q.x * q.x;
		sqy = q.y * q.y;
		sqz = q.z * q.z;
		squ = q.w * q.w;

		rollX = atan2(2 * (q.y * q.z + q.w * q.x), squ - sqx - sqy + sqz);
		sarg = (-2.0f) * (q.x * q.z - q.w * q.y);
		pitchY = sarg <= (-1.0f) ? (-0.5f) * PxPi : (sarg >= (1.0f) ? (0.5f) * PxPi : asinf(sarg));
		yawZ = atan2(2 * (q.x * q.y + q.w * q.z), squ + sqx - sqy - sqz);
	}

	void GetAngles(int a, float* angles, float* lows, float* highs)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;

		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = 0.f;
			float low = 0.f;
			float high = 0.f;

			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
				Transform body1Pose;
				NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

				Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
				Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
				Transform relPose = Inverse(pose0) * pose1;

				prevPos = relPose.p;
				Quat qd = relPose.q;
				if (qd.w < 0)
				{
					qd *= -1.f;
				}

				Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
				Quat qswing = qd * Inverse(qtwist);
				prevTwist = asin(qtwist.x) * 2.f;
				prevSwing1 = asin(qswing.y) * 2.f;
				prevSwing2 = asin(qswing.z) * 2.f;
				prevIdx = ctrls[a][qq].first;

				// If the same, no need to recompute
			}

			NvFlexRigidJointAxis idx = ctrls[a][qq].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];
			if (idx == eNvFlexRigidJointAxisTwist)
			{
				pos = prevTwist;
			}
			else if (idx == eNvFlexRigidJointAxisSwing1)
			{
				pos = prevSwing1;
			}
			else if (idx == eNvFlexRigidJointAxisSwing2)
			{
				pos = prevSwing2;
			}
			else if (idx == eNvFlexRigidJointAxisX)
			{
				pos = prevPos.x;
			}
			else if (idx == eNvFlexRigidJointAxisY)
			{
				pos = prevPos.y;
			}
			else if (idx == eNvFlexRigidJointAxisZ)
			{
				pos = prevPos.z;
			}

			angles[i] = pos;
			lows[i] = low;
			highs[i] = high;
		}
	}

	void GetAngles(int a, vector<float>& angles, vector<float>& lows, vector<float>& highs)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;

		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = 0.f;
			float low = 0.f;
			float high = 0.f;

			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
				Transform body1Pose;
				NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

				Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
				Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
				Transform relPose = Inverse(pose0) * pose1;

				prevPos = relPose.p;
				Quat qd = relPose.q;
				if (qd.w < 0)
				{
					qd *= -1.f;
				}

				Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
				Quat qswing = qd*Inverse(qtwist);
				prevTwist = asin(qtwist.x) * 2.f;
				prevSwing1 = asin(qswing.y) * 2.f;
				prevSwing2 = asin(qswing.z) * 2.f;
				prevIdx = ctrls[a][qq].first;

				// If the same, no need to recompute
			}

			NvFlexRigidJointAxis idx = ctrls[a][qq].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];
			if (idx == eNvFlexRigidJointAxisTwist)
			{
				pos = prevTwist;
			}
			else if (idx == eNvFlexRigidJointAxisSwing1)
			{
				pos = prevSwing1;
			}
			else if (idx == eNvFlexRigidJointAxisSwing2)
			{
				pos = prevSwing2;
			}
			else if (idx == eNvFlexRigidJointAxisX)
			{
				pos = prevPos.x;
			}
			else if (idx == eNvFlexRigidJointAxisY)
			{
				pos = prevPos.y;
			}
			else if (idx == eNvFlexRigidJointAxisZ)
			{
				pos = prevPos.z;
			}

			angles[i] = pos;
			lows[i] = low;
			highs[i] = high;
		}
	}

	void GetAngles(int a, vector<float>& angles)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;

		for (int i = 0; (unsigned int)i < ctrls[a].size(); i++)
		{
			int qq = i;

			float pos = 0.f;

			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
				Transform body1Pose;
				NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

				Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
				Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
				Transform relPose = Inverse(pose0) * pose1;

				prevPos = relPose.p;
				Quat qd = relPose.q;
				if (qd.w < 0)
				{
					qd *= -1.f;
				}

				Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
				Quat qswing = qd * Inverse(qtwist);
				prevTwist = asin(qtwist.x) * 2.f;
				prevSwing1 = asin(qswing.y) * 2.f;
				prevSwing2 = asin(qswing.z) * 2.f;
				prevIdx = ctrls[a][qq].first;

				// If the same, no need to recompute
			}

			NvFlexRigidJointAxis idx = ctrls[a][qq].second;
			if (idx == eNvFlexRigidJointAxisTwist)
			{
				pos = prevTwist;
			}
			else if (idx == eNvFlexRigidJointAxisSwing1)
			{
				pos = prevSwing1;
			}
			else if (idx == eNvFlexRigidJointAxisSwing2)
			{
				pos = prevSwing2;
			}
			else if (idx == eNvFlexRigidJointAxisX)
			{
				pos = prevPos.x;
			}
			else if (idx == eNvFlexRigidJointAxisY)
			{
				pos = prevPos.y;
			}
			else if (idx == eNvFlexRigidJointAxisZ)
			{
				pos = prevPos.z;
			}

			angles[i] = pos;
		}
	}

	void GetJointLimits(vector<float>& lows, vector<float>& highs)
	{
		Vec3 prevPos;

		float low, high;
		for (int i = 0; (unsigned int)i < ctrls[0].size(); i++)
		{
			NvFlexRigidJoint& joint = initJoints[ctrls[0][i].first];

			NvFlexRigidJointAxis idx = ctrls[0][i].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];

			lows[i] = low;
			highs[i] = high;
		}
	}

	void GetJointLimits(vector<float>& jLimits)
	{
		Vec3 prevPos;

		float low, high;
		for (int i = 0; (unsigned int)i < ctrls[0].size(); i++)
		{
			NvFlexRigidJoint& joint = initJoints[ctrls[0][i].first];

			NvFlexRigidJointAxis idx = ctrls[0][i].second;
			low = joint.lowerLimits[idx];
			high = joint.upperLimits[idx];

			jLimits[2 * i] = low;
			jLimits[2 * i + 1] = high;
		}
	}

	virtual void GetGlobalPose(int a, Transform* trans)
	{
		pair<int, int> p = agentBodies[a];
		Transform& inv = agentOffsetInv[a];

		int ind = 0;
		for (int i = p.first; i < p.second; i++)
		{
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&pose);
			trans[ind++] = inv * pose;
		}
	}

	virtual void GetGlobalPose(int a, vector<Transform>& trans)
	{
		pair<int, int> p = agentBodies[a];
		Transform& inv = agentOffsetInv[a];

		int ind = 0;
		for (int i = p.first; i < p.second; i++)
		{
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&pose);
			trans[ind++] = inv * pose;
		}
	}

	virtual Vec3 GetLinearVel(int a, int index)
	{
		Transform& inv = agentOffsetInv[a];
		pair<int, int> p = agentBodies[a];
		return Rotate(inv.q, Vec3(g_buffers->rigidBodies[p.first + index].linearVel));
	}

	// ?
	virtual Vec3 GetAngularVel(int a, int index)
	{
		pair<int, int> p = agentBodies[a];
		return Vec3(g_buffers->rigidBodies[p.first + index].angularVel);
	}

	int GetNumBodies()
	{
		return agentBodies[0].second - agentBodies[0].first;
	}

	int GetNumControls()
	{
		return ctrls[0].size();
	}

	virtual void ClearContactInfo() = 0;
};