#pragma once

#include "rlbase.h"

using namespace std;
using namespace tinyxml2;


class RLAnt : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

	vector<int> frontLeftFoot;
	vector<int> frontRightFoot;
	vector<int> backLeftFoot;
	vector<int> backRightFoot;

	vector<int> footFlag;

	float actionScale;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, frontLeftFoot);
		LoadVec(f, frontRightFoot);
		LoadVec(f, backLeftFoot);
		LoadVec(f, backRightFoot);
		LoadVec(f, footFlag);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, frontLeftFoot);
		SaveVec(f, frontRightFoot);
		SaveVec(f, backLeftFoot);
		SaveVec(f, backRightFoot);
		SaveVec(f, footFlag);
	}

	RLAnt()
	{
		loadPath = "../../data/ant.xml";
		mNumAgents = 500;
		mNumActions = 8;
		mNumObservations = 2 * mNumActions + 11 + 4 + mNumActions; // 39, was 28
		mMaxEpisodeLength = 1000;

		g_numSubsteps = 4;
		g_params.numIterations = 25;

		g_sceneLower = Vec3(-50.f, -1.f, -50.f);
		g_sceneUpper = Vec3(100.f, 4.f, 80.f);

		numFeet = 4;

		powerScale = 0.04f;
		initialZ = 0.25f;
		terminationZ = 0.295f; // 0.26f original

		electricityCostScale = 1.f;
		stallTorqueCostScale = 1.f;
		footCollisionCost = -1.f;
		jointsAtLimitCost = -0.2f;

		angleResetNoise = 0.02f;
		angleVelResetNoise = 0.02f;
		velResetNoise = 0.02f;

		pushFrequency = 250;	// How much steps in average per 1 kick
		forceMag = 0.005f;

		actionScale = 0.f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);
		cout << "Power scale = " << powerScale << endl;

	//	g_sceneLower = Vec3(-4.f, -1.f, -4.f);
	//	g_sceneUpper = Vec3(8.f, 2.f, 8.f);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		LoadEnv();

		footFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
		}

		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[frontLeftFoot[i]] = numFeet * i;
			footFlag[frontRightFoot[i]] = numFeet * i + 1;
			footFlag[backLeftFoot[i]] = numFeet * i + 2;
			footFlag[backRightFoot[i]] = numFeet * i + 3;
		}

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			//------------- Viktor --------------------
			ppo_params.agent_name = "Ant_96";
			ppo_params.resume = 0;
			ppo_params.timesteps_per_batch = 256;
			ppo_params.hid_size = 96;
			ppo_params.num_hid_layers = 2;
			ppo_params.optim_epochs = 10;
			ppo_params.optim_stepsize = 5e-4f;
			ppo_params.optim_schedule = "adaptive";
			ppo_params.desired_kl = 0.02f;
			ppo_params.optim_batchsize_per_agent = 32;
			ppo_params.clip_param = 0.2f;

			ppo_params.relativeLogDir = "Ant";

			ppo_params.TryParseJson(g_sceneJson);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		mjcfs.push_back(make_shared<MJCFImporter>(loadPath.c_str()));
		mjcfs.back()->AddPhysicsEntities(gt, ctrl, mpower);

		torso.push_back(mjcfs[i]->bmap["torso"]);

		frontLeftFoot.push_back(mjcfs[i]->bmap["front_left_foot"]);
		frontRightFoot.push_back(mjcfs[i]->bmap["front_right_foot"]);
		backLeftFoot.push_back(mjcfs[i]->bmap["left_back_foot"]);
		backRightFoot.push_back(mjcfs[i]->bmap["right_back_foot"]);
	}

	void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		RLWalkerEnv::ResetAgent(a);
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

	virtual void FinalizeContactInfo()
	{
		rigidContacts.map();
		rigidContactCount.map();
		int numContacts = rigidContactCount[0];

		// check if we overflowed the contact buffers
		if (numContacts > g_solverDesc.maxRigidBodyContacts)
		{
			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
		}

		float lambdaScale = 5e-2f;

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body0];
					feetContact[ff] += lambdaScale * ct[i].lambda;

					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body0];
					numCollideOther[ff / numFeet]++;
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body1];
					feetContact[ff] += lambdaScale * ct[i].lambda;

					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body1];
					numCollideOther[ff / numFeet]++;
				}
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	virtual void DoGui()
	{
		if (!mDoLearning)
		{
			imguiSlider("Actions scale", &actionScale, 0.f, 5.f, 0.0001f);

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

				for (int i = 0; i < mNumActions; i++)
				{
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
					else 
					{
						printf("Invalid axis index, probably bad migration\n");
						exit(0);
					}

					float action = motorPower[ai][i] * powerScale * actionScale * (float)Rand(-2, 3);

					Vec3 torque = axis * action;
					a0.torque[0] += torque.x;
					a0.torque[1] += torque.y;
					a0.torque[2] += torque.z;
					a1.torque[0] -= torque.x;
					a1.torque[1] -= torque.y;
					a1.torque[2] -= torque.z;
				}
			}

			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

			NvFlexSetParams(g_solver, &g_params);
			NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
			g_frame++;
			NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
			NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
			g_buffers->rigidBodies.map();
		}
	}

	float AliveBonus(float z, float pitch)
	{
		if (z > terminationZ)
		{
			return 0.5f;
		}
		else
		{
			return -1.f;
		}
	}
};


class RLSimpleHumanoid : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

	vector<int> rightFoot;
	vector<int> leftFoot;

	vector<int> footFlag;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);
		LoadVec(f, footFlag);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);
		SaveVec(f, footFlag);
	}

	RLSimpleHumanoid()
	{
		loadPath = "../../data/humanoid_symmetric.xml";

		mNumAgents = 1024;
		numPerRow = 32;
		mNumActions = 17;
		mNumObservations = 2 * mNumActions + 11 + 2 + mNumActions; // was 44
		mMaxEpisodeLength = 1000;

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 4;
		g_params.numInnerIterations = 6;
		g_params.relaxationFactor = 0.75f;
		powerScale = 0.5f;

		g_sceneLower = Vec3(-25.f, 0.f, -25.f);
		g_sceneUpper = Vec3(120.f, 5.f, 100.f);

		g_pause = true;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;

		spacing = 10.f;

		numFeet = 2;

		initialZ = 0.8f;
		terminationZ = 0.85f;

		electricityCostScale = 1.5f;
		stallTorqueCostScale = 2.f;

		angleResetNoise = 0.02f;
		angleVelResetNoise = 0.01f;
		velResetNoise = 0.01f;

		pushFrequency = 250;	// How much steps in average per 1 kick
		forceMag = 1.f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		maxContactsPerAgent = 48;
		g_solverDesc.maxRigidBodyContacts = maxContactsPerAgent * mNumAgents;

		LoadEnv();

		for (int i = 0; i < mNumAgents; i++)
		{
			rightFoot.push_back(mjcfs[i]->bmap["right_foot"]);
			leftFoot.push_back(mjcfs[i]->bmap["left_foot"]);
		}

		footFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
		}

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint)*g_buffers->rigidJoints.size());
		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[rightFoot[i]] = numFeet * i;
			footFlag[leftFoot[i]] = numFeet * i + 1;
		}

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;
			ppo_params.TryParseJson(g_sceneJson);

			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
	}

	virtual void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		RLWalkerEnv::ResetAgent(a);
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual void FinalizeContactInfo()
	{
		rigidContacts.map();
		rigidContactCount.map();
		int numContacts = rigidContactCount[0];

		// check if we overflowed the contact buffers
		if (numContacts > g_solverDesc.maxRigidBodyContacts)
		{
			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
		}

		float lambdaScale = 4e-3f;

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body0];
					feetContact[ff] += lambdaScale * ct[i].lambda;

					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body0];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body1];
					feetContact[ff] += lambdaScale * ct[i].lambda;

					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body1];
					numCollideOther[ff / 2]++;
				}
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	float AliveBonus(float z, float pitch)
	{
		if (z > terminationZ)
		{
			return 2.f;
		}
		else
		{
			return -1.f;
		}
	}
};


class RigidSimpleHumanoidHard : public RLWalkerHardEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	vector<int> rightFoot;
	vector<int> leftFoot;
	vector<int> footFlag;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerHardEnv::LoadRLState(f);
		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);
		LoadVec(f, footFlag);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerHardEnv::SaveRLState(f);
		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);
		SaveVec(f, footFlag);
	}

	//    int maxStepsOnGround;

	RigidSimpleHumanoidHard()
	{
		loadPath = "../../data/humanoid_symmetric.xml";

		mNumAgents = 500;
		mNumActions = 17;
		mNumObservations = 2 * mNumActions + 11 + 2 + mNumActions; // was 44
		mMaxEpisodeLength = 1000;

		g_numSubsteps = 4;
		g_params.numIterations = 25;

		g_sceneLower = Vec3(-50.f, 0.f, -40.f);
		g_sceneUpper = Vec3(150.f, 50.f, 100.f);

		g_pause = true;
		mDoLearning = g_doLearning;
		numRenderSteps = 1000;

		numPerRow = 20;

		powerScale = 0.5f;
		initialZ = 0.8f;
		terminationZ = 0.79f;

		masterElectricityCostScale = 1.8f;
		stallTorqueCostScale = 5.f;

		maxX = 40.f;
		maxY = 40.f;
		maxFlagResetSteps = Rand(170, 200);
		maxStepsOnGround = 180;

		angleResetNoise = 0.1f;
		angleVelResetNoise = 0.1f;
		velResetNoise = 0.1f;

		pushFrequency = 270;	// How much steps in average per 1 kick
		forceMag = 3.f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		LoadEnv();

		footFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
		}

		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[feet[2 * i]] = 2 * i;
			footFlag[feet[2 * i + 1]] = 2 * i + 1;

			handFlag[hands[2 * i]] = 2 * i;
			handFlag[hands[2 * i + 1]] = 2 * i + 1;
		}

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint)*g_buffers->rigidJoints.size());

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;
			ppo_params.TryParseJson(g_sceneJson);

			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
	}

	virtual void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		RLWalkerHardEnv::ResetAgent(a);
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual void FinalizeContactInfo()
	{
		//Ask Miles about ground contact
		rigidContacts.map();
		rigidContactCount.map();
		int numContacts = rigidContactCount[0];

		// check if we overflowed the contact buffers
		if (numContacts > g_solverDesc.maxRigidBodyContacts)
		{
			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
		}

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body0];
					feetContact[ff] = 1;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body0];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body1];
					feetContact[ff] = 1;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body1];
					numCollideOther[ff / 2]++;
				}
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}
};


class RigidHumanoid : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	vector<int> rightFoot;
	vector<int> leftFoot;
	vector<int> footFlag;

	float actionScale;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);
		LoadVec(f, footFlag);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);
		SaveVec(f, footFlag);
	}

	RigidHumanoid()
	{
		loadPath = "../../data/humanoid_20_5.xml";

		mNumAgents = 200;
		mNumActions = 21;
		mNumObservations = 2 * mNumActions + 11 + 2 + mNumActions; // 76, was 52
		mMaxEpisodeLength = 1000;

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 4;
		g_params.numInnerIterations = 10;
		g_params.relaxationFactor = 0.75f;
		powerScale = 0.25f;

		g_sceneLower = Vec3(-50.f, -1.f, -50.f);
	//	g_sceneUpper = Vec3(150.f, 4.f, 100.f);
		g_sceneUpper = Vec3(120.f, 4.f, 90.f);

		g_pause = true;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;
		doFlagRun = false;

		numPerRow = 20;
		spacing = 12.f;

		numFeet = 2;

		initialZ = 0.9f;
		terminationZ = 0.795f;

		actionScale = 0.f;

		electricityCostScale = 1.5f;
		stallTorqueCostScale = 2.f;

		maxX = 25.f;
		maxY = 25.f;
		maxFlagResetSteps = 170;

		angleResetNoise = 0.02f;
		angleVelResetNoise = 0.02f;
		velResetNoise = 0.02f;

		pushFrequency = 260;	// How much steps in average per 1 kick
		forceMag = 2.5f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		LoadEnv();

		for (int i = 0; i < mNumAgents; i++)
		{
			rightFoot.push_back(mjcfs[i]->bmap["right_foot"]);
			leftFoot.push_back(mjcfs[i]->bmap["left_foot"]);
		}

		footFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
		}

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint)*g_buffers->rigidJoints.size());
		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[rightFoot[i]] = numFeet * i;
			footFlag[leftFoot[i]] = numFeet * i + 1;
		}

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.TryParseJson(g_sceneJson);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
	}

	virtual void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		RLWalkerEnv::ResetAgent(a);
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual void FinalizeContactInfo()
	{
		//Ask Miles about ground contact
		rigidContacts.map();
		rigidContactCount.map();
		int numContacts = rigidContactCount[0];

		// check if we overflowed the contact buffers
		if (numContacts > g_solverDesc.maxRigidBodyContacts)
		{
			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
		}

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body0];
					feetContact[ff] = 1;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body0];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body1];
					feetContact[ff] = 1;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body1];
					numCollideOther[ff / 2]++;
				}
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	float AliveBonus(float z, float pitch)
	{
		// Viktor: modified original one to enforce standing and walking high, not on knees
		// Also due to reduced electric cost bonus for living has been decreased
		if (g_frame >= 256 * 500)
		{
			// Very simple curriculum learning
			maxFlagResetSteps = 150;

			angleResetNoise = 0.1f;
			angleVelResetNoise = 0.1f;
			velResetNoise = 0.1f;

			pushFrequency = 240;
			forceMag = 2.5f;
		}

		if (z > terminationZ)
		{
			return 2.f;
		}
		else
		{
			return -1.f;
		}
	}

	virtual void DoGui()
	{
		if (!mDoLearning)
		{
			imguiSlider("Actions scale", &actionScale, 0.f, 10.f, 0.01f);

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

				for (int i = 0; i < mNumActions; i++)
				{
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
					else 
					{
						printf("Invalid axis index, probably bad migration\n");
						exit(0);
					}

					float action = motorPower[ai][i] * powerScale * actionScale * 2.f * ((float)Rand(0, 2) - 0.5f);

					Vec3 torque = axis * action;
					a0.torque[0] += torque.x;
					a0.torque[1] += torque.y;
					a0.torque[2] += torque.z;
					a1.torque[0] -= torque.x;
					a1.torque[1] -= torque.y;
					a1.torque[2] -= torque.z;
				}
			}

			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

			NvFlexSetParams(g_solver, &g_params);
			NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
			g_frame++;
			NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
			NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
			g_buffers->rigidBodies.map();
		}
	}
};


class RLHumanoidPool : public RigidHumanoid
{
public:
	// pool wall params (widths, heights, lengths are halves)
	float poolWidth, poolHeight, poolLength, poolEdge, goalDist;
	// pool fluid params
	float particleRadius, particleDensity;
	int particleWidth, particleHeight, particleLength;

	Vec3 poolOrigin;
	vector<pair<int, int>> humanoidJoints;

	// if true, will give a small individual pool to each agent. otherwise make one big pool
	// make sure to tune spacing and numPerRow accordingly no pools collide with each other!
	bool smallPools;

	int baseNumObservations;

	RLHumanoidPool()
	{
		mNumAgents = 6;
		spacing = 1.f;
		numPerRow = 3;
		smallPools = false;
		g_sceneUpper = Vec3(50.f, 4.f, 4.f);

		// pool wall params for both small and big pools
		poolEdge = 0.05f;
		poolHeight = 1.25f;
		goalDist = 7.f;

		// pool fluid params for both small and big pools
		particleRadius = 0.08f; //0.07f;
		particleDensity = 9.5e2f;

		baseNumObservations = mNumObservations;
		mNumObservations = baseNumObservations + 24; // add force reading for each joint

		SetSimParams();
	}

	void LoadEnv() 
	{
		// TODO(jaliang): read smallPools from external cfg
		if (smallPools)
		{
			// pool wall params
			poolWidth = 1.5f;
			poolLength = 3.f;
			poolOrigin = Vec3(0.f, 0.f, 2.f);
		}
		else
		{
			// pool wall params
			int numRows = int(ceil(mNumAgents / 1.f / numPerRow));
			poolWidth = spacing * (float)(min(mNumAgents, numPerRow) + 1.f) / 2.f;
			poolLength = (spacing * float(numRows)) / 2.f + goalDist + 0.5f; 
			poolOrigin = Vec3(poolWidth - spacing, 0.f, poolLength - spacing);
		}

		// set pool fluid params
		particleWidth = int(2.f * poolWidth / particleRadius);
		particleHeight = int(2.f * poolHeight / particleRadius);
		particleLength = int(2.f * poolLength / particleRadius);

		if (!smallPools)
		{
			AddPoolWalls(Transform());
			AddPool(Transform());
		}

		humanoidJoints.resize(mNumAgents);

		g_sceneUpper = Vec3(50.f, 4.f, 2.f);
		yOffset = 0.5f;
/*
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		torso.clear();
		torso.resize(mNumAgents, -1);

	
		masses.clear();

		for (int i = 0; i < mNumAgents; i++)
		{
			Vec3 pos = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing + 0.25f);
			Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));

			Vec3 posStart = Vec3((i % numPerRow) * spacing, yOffset + Randf(-0.02f, 0.02f), (i / numPerRow) * spacing + 0.25f);
			Quat rotStart = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), Randf(-0.5f, 0.5f));

			rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise))
				* rot;

			rotStart = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), Randf(-0.5f, 0.5f)) * rotStart;

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
		*/
/*
		RLWalkerEnv::LoadEnv();

		// Make fingers heavier when ID control
		float mul = 1.45f;
		float imul = 1.0f / mul;

		for (int i = 0; i < g_buffers->rigidBodies.size(); ++i)
		{
			g_buffers->rigidBodies[i].mass *= mul;
			g_buffers->rigidBodies[i].invMass *= imul;
			for (int k = 0; k < 9; k++)
			{
				g_buffers->rigidBodies[i].inertia[k] *= mul;
				g_buffers->rigidBodies[i].invInertia[k] *= imul;
			}

			g_buffers->rigidBodies[i].mass *= mul;
			g_buffers->rigidBodies[i].invMass *= imul;
			for (int k = 0; k < 9; k++)
			{
				g_buffers->rigidBodies[i].inertia[k] *= mul;
				g_buffers->rigidBodies[i].invInertia[k] *= imul;
			}
		}

		powerScale *= (0.965f * mul); */
	}

	void SetSimParams()
	{
		g_params.radius = 0.14f;
		g_params.fluidRestDistance = particleRadius;
		g_params.numIterations = 20;
		g_params.viscosity = 0.1f;
		g_params.dynamicFriction = 0.1f;
		g_params.staticFriction = 0.4f;
		g_params.particleCollisionMargin = 0.0f;
		g_params.collisionDistance = g_params.fluidRestDistance * 0.5f;
		
		g_params.vorticityConfinement = 120.0f;
		g_params.cohesion = 0.0001f;
		g_params.drag = 0.f;
		g_params.lift = 0.f;
		g_params.solidPressure = 0.0f;
		g_params.smoothing = 1.0f;
		g_params.relaxationFactor = 1.0f;

		g_maxDiffuseParticles = 64 * 1024;
		g_diffuseScale = 0.25f;
		g_diffuseShadow = false;
		g_diffuseColor = 2.5f;
		g_diffuseMotionScale = 1.5f;
		g_params.diffuseThreshold *= 0.01f;
		g_params.diffuseBallistic = 35;
		g_numSubsteps = 2;

		// draw options		
		g_drawEllipsoids = true;
		g_drawPoints = false;
		g_drawDiffuse = true;
		g_ropeScale = 0.2f;
		g_warmup = false;
		g_pause = false;
	}

	void ExtractState(int a, float* state, float& p, float& walkTargetDist, float* jointSpeeds, int& numJointsAtLimit, float& heading, float& upVec)
	{
		RLWalkerEnv::ExtractState(a, state, p, walkTargetDist, jointSpeeds, numJointsAtLimit, heading, upVec);
		g_buffers->rigidJoints.map();
		int ct = baseNumObservations;
		for (int i = humanoidJoints[a].first; i < humanoidJoints[a].second; i++)
		{
			state[ct++] = g_buffers->rigidJoints[i].lambda[eNvFlexRigidJointAxisTwist];
		}

		g_buffers->rigidJoints.unmap();
	}

	void AddAgentBodiesJointsCtlsPowers(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		int startShape = g_buffers->rigidShapes.size();
		int startJoint = g_buffers->rigidJoints.size();
		RLWalkerEnv::AddAgentBodiesJointsCtlsPowers(ai, gt, ctrl, mpower);
		int endJoint = g_buffers->rigidJoints.size();
		int endShape = g_buffers->rigidShapes.size();

		humanoidJoints[ai] = { startJoint, endJoint };
		for (int i = startShape; i < endShape; i++)
		{
			g_buffers->rigidShapes[i].group = ai;
		}
		
		if (smallPools)
		{
			AddPoolWalls(gt);
			AddPool(gt);
		}
	}

	void AddPoolWalls(Transform gt)
	{
		vector<NvFlexRigidShape> poolWalls;
		poolWalls.resize(0);

		Vec3 poolOriginGlobal = poolOrigin + gt.p;
		
		NvFlexRigidShape leftWall;
		NvFlexMakeRigidBoxShape(&leftWall, -1, poolEdge, poolHeight, poolLength + poolEdge * 2.f,
			NvFlexMakeRigidPose(poolOriginGlobal + Vec3(-poolWidth - poolEdge, poolHeight, 0.f), Quat()));
		poolWalls.push_back(leftWall);

		NvFlexRigidShape rightWall;
		NvFlexMakeRigidBoxShape(&rightWall, -1, poolEdge, poolHeight, poolLength + poolEdge * 2.f,
			NvFlexMakeRigidPose(poolOriginGlobal + Vec3(poolWidth + poolEdge, poolHeight, 0.f), Quat()));
		poolWalls.push_back(rightWall);

		NvFlexRigidShape frontWall;
		NvFlexMakeRigidBoxShape(&frontWall, -1, poolWidth + 2.f * poolEdge, poolHeight, poolEdge,
			NvFlexMakeRigidPose(poolOriginGlobal + Vec3(0.f, poolHeight, -poolLength - poolEdge), Quat()));
		poolWalls.push_back(frontWall);

		NvFlexRigidShape backWall;
		NvFlexMakeRigidBoxShape(&backWall, -1, poolWidth + 2.f * poolEdge, poolHeight, poolEdge,
			NvFlexMakeRigidPose(poolOriginGlobal + Vec3(0.f, poolHeight, poolLength + poolEdge), Quat()));
		poolWalls.push_back(backWall);

		for (int i = 0; (unsigned int)i < poolWalls.size(); i++)
		{
			poolWalls[i].filter = 0;
			poolWalls[i].group = -1;
			poolWalls[i].material.friction = 0.7f;
			poolWalls[i].user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));
			g_buffers->rigidShapes.push_back(poolWalls[i]);
		}
	}

	void AddPool(Transform gt)
	{
		float mass = 4.f / 3.f * kPi * pow(particleRadius, 3.f)  * particleDensity;
		CreateParticleGrid(poolOrigin + gt.p - Vec3(poolWidth, 0.0f, poolLength), particleWidth, particleHeight, particleLength,
			particleRadius, Vec3(0.f), 1.f / mass, false, 0.0f, 
			NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid));
	}

	virtual void resetTarget(int a, bool firstTime = true)
	{
		walkTargetX[a] = poolLength * 2.f;
		walkTargetY[a] = agentOffset[a].p[0];
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

		float progressScale = 50.f; // Swimming is much slower than running
		float progress = progressScale * (potential - potentialOld);
		if (progress > 1000.f)
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

		if (walkTargetDists[a] < 0.75f)
		{
			rew += 2.f;
			dead = true;
		}

		for (int i = 0; i < 6; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}
	}

	float AliveBonus(float z, float pitch)
	{		
		float heightBonus = z > terminationZ ? 0.25f + 0.5f * (z - terminationZ) : -1.f;

		// 0 is upright, pi/2 is flat w/ face down
		float pitchBonus = 2.f * sin(pitch);
		if (heightBonus > 0.f)
		{
			heightBonus += pitchBonus;
		}

		return heightBonus;
	}

	void PostUpdate()
	{
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
	}
};


class RigidHumanoidSpeed : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	vector<int> hands;
	vector<float> targetVelocities;
	float targetVelocity;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, hands);
		LoadVec(f, targetVelocities);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, hands);
		SaveVec(f, targetVelocities);
	}

	RigidHumanoidSpeed()
	{
		loadPath = "../../data/humanoid_20_5.xml";

		mNumAgents = 500;
		mNumActions = 21;
		mMaxEpisodeLength = 1000;

		g_numSubsteps = 4;
		g_params.numIterations = 30;

		g_sceneLower = Vec3(-50.f, -1.f, -50.f);
		g_sceneUpper = Vec3(120.f, 3.f, 80.f);

		g_pause = true;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;
		doFlagRun = false;

		targetVelocity = 5.f;

		numPerRow = 20;
		spacing = 12.f;

		numFeet = 2;

		powerScale = 0.25f;
		initialZ = 0.9f;
		terminationZ = 0.9f;

		electricityCostScale = 1.5f;
		stallTorqueCostScale = 2.f;

		maxX = 25.f;
		maxY = 25.f;
		maxFlagResetSteps = 300;

		angleResetNoise = 0.1f;
		angleVelResetNoise = 0.1f;
		velResetNoise = 0.1f;
		rotCreateNoise = 0.05f;

		pushFrequency = 260;	// How much steps in average per 1 kick
		forceMag = 2.5f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		mNumObservations = 2 * mNumActions + 11 + 2 + mNumActions + 1; // 77

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		targetVelocities.resize(mNumAgents, 0.f);

		LoadEnv();

		footFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
		}

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());
		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[feet[2 * i]] = 2 * i;
			footFlag[feet[2 * i + 1]] = 2 * i + 1;
		}

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.TryParseJson(g_sceneJson);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}

		cout << "Num agents = " << mNumAgents << endl;
		cout << "Termination z = " << terminationZ << endl;
	}

//	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
//	{
//		mjcfs.push_back(new MJCFImporter(loadPath.c_str(), gt, ctrl, mpower));
//
//		torso.push_back(mjcfs[i]->bmap["torso"]);
//		pelvis.push_back(mjcfs[i]->bmap["pelvis"]);
//		head.push_back(mjcfs[i]->bmap["head"]);
//
//		feet.push_back(mjcfs[i]->bmap["right_foot"]);
//		feet.push_back(mjcfs[i]->bmap["left_foot"]);
//	}

	virtual void LoadEnv()
	{
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		head.clear();
		torso.clear();
		pelvis.clear();
		feet.clear();

		for (int i = 0; i < mNumAgents; i++)
		{
			Vec3 pos = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f);

			Vec3 posStart = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rotStart = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));

			posStart.y += Randf(-0.1f, -0.05f);

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

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());
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
		Transform bodies[50];

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

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = feetContact[2 * a + i];
		}

		float* prevActions = GetAction(a);
		if (prevActions) // could be null if this is called for the first time before agent acts
		{
			for (int i = 0; i < mNumActions; ++i)
			{
				state[ct++] = prevActions[i];
			}
		}

		state[ct++] = 0.1f * targetVelocity; //targetVelocities[a];

		for (int i = 0; i < ct; ++i)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}
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

		// float maxTargetVelocity = 10.f;
	//	float progress = maxTargetVelocity * (potential - potentialOld) / targetVelocity;
		float progress = Sign(targetVelocity) * (potential - potentialOld);

		if (progress > 100.f)
		{
			printf("progress is infinite %f %f %f \n", progress, potential, potentialOld);
		}

		// Vec3 vel = GetLinearVel(a, 0);
		//float velMagnitude = Length(vel);
		float bias = 0.1f;
		float sqrDiff = sqr(targetVelocity - bias - progress);
		// float sqrDiffNorm = sqrDiff / (targetVelocity * targetVelocity + 0.1f);

		// float prScale = abs(targetVelocity) > 0.1f ? min(maxTargetVelocity / abs(targetVelocity), 1.f) : 0.f;
		// float normProgress = progress * prScale;
	//	progress = 0.5f * ( 1.f + 1.5f * exp(-4.f * sqrDiffNorm)) * normProgress;

		float backScale = targetVelocity > 0.f ? 1.f : 2.f;

		progress *= backScale;
		progress += 0.75f * exp(-(0.075f + 0.025f * Sign(targetVelocity)) * sqrDiff);
		if (abs(targetVelocity) <= 1.f)
		{
			progress = 2.5f * exp(-sqrDiff);
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
				cout << "Reward "<< i << " is infinite" << endl;
			}
			rew += rewards[i];
		}

		// Reset of the target velocity
		float res = Randf(0.f, 1000.f * (float)mNumAgents);
		if (res < 5.f)
		{
			//	targetVelocity = sqrt(Randf(0.f, 170.f)) + Randf(-2.f, 0.5f);
			targetVelocity = Randf(-10.f, 12.f);
			if (targetVelocity <= 2.f && targetVelocity >= -1.f)
			{
				targetVelocity = 0.f;
			}

			//	cout << "New target velocity = " << targetVelocity << endl;
		}
	}

	virtual void DoGui()
	{
		imguiSlider("Target velocity", &targetVelocity, -8.0f, 12.0f, 0.1f);
		if (imguiCheck("Flagrun", doFlagRun))
		{
			doFlagRun = !doFlagRun;
		}
	}

	virtual void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		RLWalkerEnv::ResetAgent(a);
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual void ClearContactInfo()
	{
		for (int ai = 0; ai < mNumAgents; ++ai)
		{
			for (int i = 0; i < 2; ++i)
			{
				feetContact[2 * ai + i] = 0.f;
			}
			numCollideOther[ai] = 0;
		}
	}

	virtual void FinalizeContactInfo()
	{
		rigidContacts.map();
		rigidContactCount.map();
		int numContacts = rigidContactCount[0];

		// check if we overflowed the contact buffers
		if (numContacts > g_solverDesc.maxRigidBodyContacts)
		{
			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
		}

		float lambdaScale = 4e-3f;

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body0];
					feetContact[ff] += lambdaScale * ct[i].lambda;

					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body0];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body1];
					feetContact[ff] += lambdaScale * ct[i].lambda;

					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body1];
					numCollideOther[ff / 2]++;
				}
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	// Not used, implemented as it's an abstract function
	float AliveBonus(float z, float pitch)
	{
		if (z > terminationZ)
		{
			return 1.f; // Lower value due to the additional reward contribution from following the target velocity
		}
		else
		{
			return -1.f;
		}
	}

	float AliveBonus(float z, float pitch, int a)
	{
		float defRew = 1.f;

		// Vec3 vel = GetLinearVel(a, 0);
		// float velMagnitude = Length(vel);
		// float sqrDiff = sqr(targetVelocity - velMagnitude);

		float speedRew = 0.f; //2.f * exp(-0.25f * sqrDiff);

		if (z > terminationZ)
		{
			return defRew + speedRew;
		}
		else
		{
			return -1.f;
		}
	}
};


class RigidHumanoidHard : public RLWalkerHardEnv<Transform, Vec3, Quat, Matrix33>
{
public:

	RigidHumanoidHard()
	{
		loadPath = "../../data/humanoid_20_5.xml";

		mNumAgents = 400;
		mNumActions = 21;
		mMaxEpisodeLength = 1000;

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 4;
		g_params.numInnerIterations = 10;
		g_params.relaxationFactor = 0.75f;
		powerScale = 0.24f;

		g_sceneLower = Vec3(-50.f, 0.f, -50.f);
		g_sceneUpper = Vec3(120.f, 4.f, 75.f);

		g_pause = true;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;

		doFlagRun = true;

		numPerRow = 20;
		spacing = 7.5f;

		initialZ = 0.9f;
		terminationZ = 0.79;

		masterElectricityCostScale = 2.f;
		stallTorqueCostScale = 2.f;

		maxX = 40.f;
		maxY = 40.f;
		maxFlagResetSteps = 180;
		maxStepsOnGround = 170;

		angleResetNoise = 0.15f;
		angleVelResetNoise = 0.15f;
		velResetNoise = 0.15f;
		rotCreateNoise = 0.055f;

		pushFrequency = 300;	// How much steps in average per 1 kick
		forceMag = 6.f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		if (doRoboschool)
		{
			mNumObservations = 52;
		}
		else
		{
			mNumObservations = 2 * mNumActions + 11 + 3 * 2 + mNumActions; // + mNumActions; // 78, old - 52;
		}

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		LoadEnv();

		footFlag.resize(g_buffers->rigidBodies.size(), -1);
		kneeFlag.resize(g_buffers->rigidBodies.size(), -1);

		torsoFlag.resize(g_buffers->rigidBodies.size(), -1);
		handFlag.resize(g_buffers->rigidBodies.size(), -1);
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
		}

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());

		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[feet[2 * i]] = 2 * i;
			footFlag[feet[2 * i + 1]] = 2 * i + 1;

			kneeFlag[knees[2 * i]] = 2 * i;
			kneeFlag[knees[2 * i + 1]] = 2 * i + 1;

			handFlag[hands[2 * i]] = 2 * i;
			handFlag[hands[2 * i + 1]] = 2 * i + 1;

			torsoFlag[torso[i]] = i;
		}

		for (int i = 0; i < (int)g_buffers->rigidShapes.size(); ++i)
		{
			g_buffers->rigidShapes[i].filter = 1 << 5;
			g_buffers->rigidShapes[i].group = -1;
		}

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.TryParseJson(g_sceneJson);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
	}

	virtual void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		RLWalkerHardEnv::ResetAgent(a);
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual void ClearContactInfo()
	{
		for (auto &ff : feetContact)	ff = 0.f;
		for (auto &kf : kneesContact)	kf = 0.f;
		for (auto &hf : handsContact)	hf = 0.f;
		for (auto &tf : torsoContact)	tf = 0.f;
		for (auto &nc : numCollideOther) nc = 0;
	}

	virtual void Update()
	{
		RLWalkerHardEnv::Update();
	}

	virtual void FinalizeContactInfo()
	{
		//Ask Miles about ground contact
		rigidContacts.map();
		rigidContactCount.map();
		int numContacts = rigidContactCount[0];

		// check if we overflowed the contact buffers
		if (numContacts > g_solverDesc.maxRigidBodyContacts)
		{
			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
		}

		float lambdaScale = 4e-3f;

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if (ct[i].body0 >= 0 && ct[i].lambda > 0.f)
			{
				if (footFlag[ct[i].body0] >= 0)
				{
					int ff = footFlag[ct[i].body0];
					feetContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else if (kneeFlag[ct[i].body0] >= 0)
				{
					int ff = kneeFlag[ct[i].body0];
					kneesContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Knee lambda = " << ct[i].lambda << endl;
				}
				else if (handFlag[ct[i].body0] >= 0)
				{
					int ff = handFlag[ct[i].body0];
					handsContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Hand lambda = " << ct[i].lambda << endl;
				}
				else if (torsoFlag[ct[i].body0] >= 0)
				{
					int ff = torsoFlag[ct[i].body0];
					torsoContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Torso lambda = " << ct[i].lambda << endl;
				}
			}

			if (ct[i].body1 >= 0 && ct[i].lambda > 0.f)
			{
				if (footFlag[ct[i].body1] >= 0)
				{
					int ff = footFlag[ct[i].body1];
					feetContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Foot lambda = " << ct[i].lambda << endl;
				}
				else if (kneeFlag[ct[i].body1] >= 0)
				{
					int ff = kneeFlag[ct[i].body1];
					kneesContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Hand lambda = " << ct[i].lambda << endl;
				}
				else if (handFlag[ct[i].body1] >= 1)
				{
					int ff = handFlag[ct[i].body1];
					handsContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Hand lambda = " << ct[i].lambda << endl;
				}
				else if (torsoFlag[ct[i].body1] >= 1)
				{
					int ff = torsoFlag[ct[i].body1];
					torsoContact[ff] += lambdaScale * ct[i].lambda;
					//	cout << "Hand lambda = " << ct[i].lambda << endl;
				}
			}
		}
		/*
			for (int i = 0; i < numContacts; ++i)
			{
				if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
				{
					if (ct[i].body1 < 0)
					{
						// foot contact with ground
						int ff = footFlag[ct[i].body0];
						feetContact[ff] += lambdaScale * ct[i].lambda;

						//	cout << "Foot lambda = " << ct[i].lambda << endl;
					}
					else
					{
						// foot contact with something other than ground
						int ff = footFlag[ct[i].body0];
						numCollideOther[ff / 2]++;
					}
				}

				if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
				{
					if (ct[i].body0 < 0)
					{
						// foot contact with ground
						int ff = footFlag[ct[i].body1];
						feetContact[ff] += lambdaScale * ct[i].lambda;

						//	cout << "Foot lambda = " << ct[i].lambda << endl;
					}
					else
					{
						// foot contact with something other than ground
						int ff = footFlag[ct[i].body1];
						numCollideOther[ff / 2]++;
					}
				}

				if ((ct[i].body0 >= 0) && (handFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
				{
					if (ct[i].body1 < 0)
					{
						// foot contact with ground
						int ff = handFlag[ct[i].body0];
						handsContact[ff] += lambdaScale * ct[i].lambda;

						//	cout << "Hand lambda = " << ct[i].lambda << endl;
					}
				}

				if ((ct[i].body1 >= 0) && (handFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
				{
					if (ct[i].body0 < 0)
					{
						// hand contact with ground
						int ff = handFlag[ct[i].body1];
						handsContact[ff] += ct[i].lambda;

						//	cout << "Hand lambda = " << ct[i].lambda << endl;
					}
				}
			}
		*/
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}
};


class CMUHumanoid : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	vector<int> toes;
	vector<int> toeFlag;
	vector<float> toeContacts;

	vector<int> knees;
	vector<int> kneeFlag;
	vector<float> kneeContacts;

	vector<int> elbows;
	vector<int> elbowFlag;
	vector<float> elbowContacts;

	float actionScale;

//	vector<int> head;

	CMUHumanoid()
	{
		loadPath = "../../data/humanoid_CMU_hands_new.xml";

		mNumAgents = 64;
		numPerRow = 8;
		mNumActions = 56;
		mNumObservations = 2 * mNumActions + 11 + 4 * 2 + mNumActions; // 187
		mMaxEpisodeLength = 1000;

		spacing = 10.f;

		g_sceneLower = Vec3(-40.f, -1.f, -40.f);
		g_sceneUpper = Vec3(72.f, 2.f, 50.f);

		g_pause = true;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;
		doFlagRun = true;

		numFeet = 2;

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 5;
		g_params.numInnerIterations = 40;
		g_params.relaxationFactor = 0.5f;
		powerScale = 1.2f;

		terminationZ = 0.65f;

		electricityCostScale = 1.5f;
		stallTorqueCostScale = 2.f;

		maxX = 40.f;
		maxY = 40.f;
		maxFlagResetSteps = 200;

		angleResetNoise = 0.01f;
		angleVelResetNoise = 0.01f;
		velResetNoise = 0.01f;

		pushFrequency = 200;	// How much steps in average per 1 kick
		forceMag = 2.f;

		actionScale = 0.0f;
	}

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, toes);
		LoadVec(f, toeFlag);
		LoadVec(f, toeContacts);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, toes);
		SaveVec(f, toeFlag);
		SaveVec(f, toeContacts);
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		mjcfs.push_back(make_shared<MJCFImporter>(loadPath.c_str()));
		mjcfs.back()->AddPhysicsEntities(gt, ctrl, mpower, 50);

//		torso.push_back(mjcfs[i]->bmap["torso"]);
		pelvis.push_back(mjcfs[i]->bmap["pelvis"]);
		head.push_back(mjcfs[i]->bmap["head"]);

		hands.push_back(mjcfs[i]->bmap["lhand"]);
		hands.push_back(mjcfs[i]->bmap["rhand"]);

		feet.push_back(mjcfs[i]->bmap["lfoot"]);
		feet.push_back(mjcfs[i]->bmap["rfoot"]);

		knees.push_back(mjcfs[i]->bmap["ltibia"]);
		knees.push_back(mjcfs[i]->bmap["rtibia"]);

		toes.push_back(mjcfs[i]->bmap["ltoes"]);
		toes.push_back(mjcfs[i]->bmap["rtoes"]);
	}

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
			Quat rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f);

			Vec3 posStart = Vec3((i % numPerRow) * spacing, yOffset + Randf(-0.02f, 0.02f), (i / numPerRow) * spacing);
			Quat rotStart = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));

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

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
		head.resize(mNumAgents);

		toeContacts.resize(2 * mNumAgents, 0.f);
		feetContact.resize(2 * mNumAgents, 0.f);
		kneeContacts.resize(2 * mNumAgents, 0.f);
		handsContact.resize(2 * mNumAgents, 0.f);

		LoadEnv();

		toeFlag.resize(g_buffers->rigidBodies.size(), -1);
		footFlag.resize(g_buffers->rigidBodies.size(), -1);
		kneeFlag.resize(g_buffers->rigidBodies.size(), -1);
		handFlag.resize(g_buffers->rigidBodies.size(), -1);
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
		}

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());

		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[feet[2 * i]] = 2 * i;
			footFlag[feet[2 * i + 1]] = 2 * i + 1;

			kneeFlag[feet[2 * i]] = 2 * i;
			kneeFlag[feet[2 * i + 1]] = 2 * i + 1;

			toeFlag[toes[2 * i]] = 2 * i;
			toeFlag[toes[2 * i + 1]] = 2 * i + 1;

			handFlag[hands[2 * i]] = 2 * i;
			handFlag[hands[2 * i + 1]] = 2 * i + 1;
		}

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.TryParseJson(g_sceneJson);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
	}

	void ExtractState(int a, float* state,
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
		Transform bodies[200];

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

		float lambdaScale = 2e-3f;
		for (int i = 0; i < 2; i++)
		{
			state[ct++] = lambdaScale * toeContacts[2 * a + i];
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = lambdaScale * feetContact[2 * a + i];
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = lambdaScale * kneeContacts[2 * a + i];
		}

		for (int i = 0; i < 2; i++)
		{
			state[ct++] = lambdaScale * handsContact[2 * a + i];
		}

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
			cout << "Num of observations is wrong: " << mNumObservations << " != " << ct << endl;
			cout << "Num of actions: " << mNumActions << endl;
		}
	}

	virtual void ResetAgent(int a)
	{
		mjcfs[a]->reset(agentStartOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		RLWalkerEnv::ResetAgent(a);
	}

	virtual void LockWrite()
	{
		// Do whatever needed to lock write to simulation
	}

	virtual void UnlockWrite()
	{
		// Do whatever needed to unlock write to simulation
	}

	virtual void FinalizeContactInfo()
	{
		//Ask Miles about ground contact
		rigidContacts.map();
		rigidContactCount.map();
		int numContacts = rigidContactCount[0];

		// check if we overflowed the contact buffers
		if (numContacts > g_solverDesc.maxRigidBodyContacts)
		{
			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
		}

		// Or move to clear contacts function?
		for (int ai = 0; ai < 2 * mNumAgents; ++ai)
		{
			toeContacts[ai] = 0.f;
			feetContact[ai] = 0.f;
			kneeContacts[ai] = 0.f;
			handsContact[ai] = 0.f;
		}

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body0];
					feetContact[ff] += ct[i].lambda;

				//	cout << "Foot1 lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body0];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// foot contact with ground
					int ff = footFlag[ct[i].body1];
					feetContact[ff] += ct[i].lambda;

				//	cout << "Foot2 lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// foot contact with something other than ground
					int ff = footFlag[ct[i].body1];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body0 >= 0) && (toeFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// toe contact with ground
					int ff = toeFlag[ct[i].body0];
					toeContacts[ff] += ct[i].lambda;

				//	cout << "Knee1 lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// knee contact with something other than ground
					int ff = toeFlag[ct[i].body0];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body1 >= 0) && (toeFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// toe contact with ground
					int ff = toeFlag[ct[i].body1];
					toeContacts[ff] += ct[i].lambda;

					//	cout << "Knee1 lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// knee contact with something other than ground
					int ff = toeFlag[ct[i].body1];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body0 >= 0) && (kneeFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// knee contact with ground
					int ff = kneeFlag[ct[i].body0];
					kneeContacts[ff] += ct[i].lambda;

					//	cout << "Knee2 lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// knee contact with something other than ground
					int ff = kneeFlag[ct[i].body0];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body1 >= 0) && (kneeFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// knee contact with ground
					int ff = kneeFlag[ct[i].body1];
					kneeContacts[ff] += ct[i].lambda;

				//	cout << "Knee2 lambda = " << ct[i].lambda << endl;
				}
				else
				{
					// knee contact with something other than ground
					int ff = kneeFlag[ct[i].body1];
					numCollideOther[ff / 2]++;
				}
			}

			if ((ct[i].body0 >= 0) && (handFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body1 < 0)
				{
					// hand contact with ground
					int ff = handFlag[ct[i].body0];
					handsContact[ff] += ct[i].lambda;

					//	cout << "Hand lambda = " << ct[i].lambda << endl;
				}
			}

			if ((ct[i].body1 >= 0) && (handFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				if (ct[i].body0 < 0)
				{
					// hand contact with ground
					int ff = handFlag[ct[i].body1];
					handsContact[ff] += ct[i].lambda;

					//	cout << "Hand lambda = " << ct[i].lambda << endl;
				}
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	float AliveBonus(float z, float pitch)
	{
		if (g_frame >= 128 * 500)
		{
			maxFlagResetSteps = 180;

			maxX = 25.f;
			maxY = 25.f;

			angleResetNoise = 0.05f;
			angleVelResetNoise = 0.02f;
			velResetNoise = 0.02f;

			pushFrequency = 240;	// How much steps in average per 1 kick
			forceMag = 2.5f;
		}

		if (z > terminationZ)
		{
			return 2.5f;
		}
		else
		{
			return -1.f;
		}
	}
	/*
	virtual void DoStats()
	{
		BeginLines(3.0f, true);
		for (size_t i = 0; i < mNumAgents; i++)
		{
			Transform trans;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[agentBodies[i].first], (NvFlexRigidPose*)&trans);
			trans.p.y += 0.5f;
			Vec3 dir(1.0f, 0.0f, 0.0f);
			dir = Rotate(agentOffset[i].q, dir);
			
			DrawLine(trans.p, trans.p + dir*0.5f, Vec4(1.0f, 0.0f, 0.0f));
		}
		EndLines();
	}
	*/
	virtual void DoGui()
	{
		imguiSlider("Actions scale", &actionScale, 0.f, 10.f, 0.01f);
		if (!mDoLearning && !g_pause)
		{
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

				for (int i = 0; i < mNumActions; i++)
				{
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
					else
					{
						printf("Invalid axis, probably bad code migration?\n");
						exit(0);
					}

					float action = motorPower[ai][i] * powerScale * actionScale * 2.f * ((float)Rand(0, 2) - 0.5f);

					Vec3 torque = axis * action;
					a0.torque[0] += torque.x;
					a0.torque[1] += torque.y;
					a0.torque[2] += torque.z;
					a1.torque[0] -= torque.x;
					a1.torque[1] -= torque.y;
					a1.torque[2] -= torque.z;
				}
			}

			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());

			NvFlexSetParams(g_solver, &g_params);
			NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
			g_frame++;
			NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
			NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
			g_buffers->rigidBodies.map();
		}
	}
};