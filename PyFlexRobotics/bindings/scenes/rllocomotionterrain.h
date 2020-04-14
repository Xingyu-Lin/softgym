#pragma once

#include "rlbase.h"
#include "spawnableobstacles.h"
#include "rllocomotion.h"

using namespace std;
using namespace tinyxml2;


class RLAntHF : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

	vector<int> frontLeftFoot;
	vector<int> frontRightFoot;
	vector<int> backLeftFoot;
	vector<int> backRightFoot;
	vector<int> torso;

	vector<int> footFlag;
	vector<int> torsoFlag;

	vector<int> stepsOnGround;
	int maxStepsOnGround;

	vector<float> feetContactForces;
	vector<float> torsoContactForces;

	vector<float> grid;

	int terrainInd;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, frontLeftFoot);
		LoadVec(f, frontRightFoot);
		LoadVec(f, backLeftFoot);
		LoadVec(f, backRightFoot);
		LoadVec(f, footFlag);
		LoadVec(f, torsoFlag);
		LoadVec(f, feetContactForces);
		LoadVec(f, torsoContactForces);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, frontLeftFoot);
		SaveVec(f, frontRightFoot);
		SaveVec(f, backLeftFoot);
		SaveVec(f, backRightFoot);
		SaveVec(f, footFlag);
		SaveVec(f, torsoFlag);
		SaveVec(f, feetContactForces);
		SaveVec(f, torsoContactForces);
	}

	RLAntHF()
	{
		loadPath = "../../data/ant.xml";
		mNumAgents = 500;
		mNumActions = 8;
		mNumObservations = 2 * mNumActions + 11 + 15 + mNumActions; // 50, was 39, was 28
		mMaxEpisodeLength = 1000;

		g_numSubsteps = 4;
		g_params.numIterations = 25;

		g_sceneLower = Vec3(-1.f);
		g_sceneUpper = Vec3(180.f, 5.f, 150.f);

		g_lightDistance *= 0.25f;
	//	g_diffuseColor = Vec4(0.5f);
	//	g_fogDistance = 0.001f;

		terrainInd = -1;

		maxStepsOnGround = 25;

		numFeet = 4;

		powerScale = 0.04f;
		yOffset = 1.4f;
		terminationZ = - yOffset; // 0.28f

		electricityCostScale = 1.f;
		stallTorqueCostScale = 1.f;
		footCollisionCost = -0.8f;
		jointsAtLimitCost = -0.2f;

		angleResetNoise = 0.02f;
		angleVelResetNoise = 0.02f;
		velResetNoise = 0.02f;

		pushFrequency = 250;	// How much steps in average per 1 kick
		forceMag = 0.005f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
		stepsOnGround.resize(mNumAgents, 0);

		feetContactForces.resize(3 * numFeet * mNumAgents, 0.f);
		torsoContactForces.resize(3 * mNumAgents, 0.f);

		LoadEnv();

		footFlag.resize(g_buffers->rigidBodies.size());
		torsoFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
			torsoFlag[i] = -1;
		}

		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[frontLeftFoot[i]] = numFeet * i;
			footFlag[frontRightFoot[i]] = numFeet * i + 1;
			footFlag[backLeftFoot[i]] = numFeet * i + 2;
			footFlag[backRightFoot[i]] = numFeet * i + 3;
			torsoFlag[torso[i]] = i;
		}

		Transform terrainTrans = Transform(Vec3(90.f, 0.98f, 100.f), Quat());
		terrainInd = createTerrain(220.f, 250.f, 120, 130, RandVec3() * 10.f, terrainTrans, Vec3(25.f, 1.3f, 25.f), 5, 0.41f);

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

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
		for (int i = 0; i < mNumActions; i++)
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
		Transform bodies[20];

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

		for (int i = 0; i < mNumActions * 2; ++i)
		{
			state[ct++] = joints[i];
		}

		for (int i = 0; i < numFeet; i++)
		{
			int footInd = numFeet * a + i;
			for (int j = 0; j < 3; ++j)
			{
				state[ct++] = feetContactForces[3 * footInd + j];
			}
		}

		for (int j = 0; j < 3; ++j)
		{
			state[ct++] = torsoContactForces[3 * a + j];
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

		float alive = AliveBonus(state[0] + initialZ, p, a); //  # state[0] is body height above ground, body_rpy[1] is pitch
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
		for (int i = 0; i < mNumActions; i++)
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

		electricityCostCurrent += electrCost * sum / (float)mNumActions;

		sum = 0.0f;
		for (int i = 0; i < mNumActions; i++)
		{
			sum += action[i] * action[i];
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
		}

		//electricity_costCurrent += stall_torque_cost * float(np.square(a).mean())
		electricityCostCurrent += stallTorqCost * sum / (float)mNumActions;

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

	void ResetAgent(int a)
	{
		stepsOnGround[a] = 0;
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

	virtual void ClearContactInfo()
	{
		for (auto &ff : feetContactForces)
		{
			ff = 0.f;
		}

		for (auto &tf : torsoContactForces)
		{
			tf = 0.f;
		}

		for (auto &nc : numCollideOther)
		{
			nc = 0;
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

		float lambdaScale = 5e-2f;
		
		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				int ff = footFlag[ct[i].body0];

				// Todo - remove copy-paste!
				int a = ff / numFeet;
				Transform& inv = agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));
				
				bool success;
				auto invR = Inverse(R, success);

				Vec3 transForce = lambdaScale * ct[i].lambda * invR * Vec3(ct[i].normal);

				for (int j = 0; j < 3; ++j)
				{
					feetContactForces[3 * ff + j] -= transForce[j];
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				int ff = footFlag[ct[i].body1];

				int a = ff / numFeet;
				Transform& inv = agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));
				
				bool success;
				auto invR = Inverse(R, success);

				Vec3 transForce = lambdaScale * ct[i].lambda * invR * Vec3(ct[i].normal);
				for (int j = 0; j < 3; ++j)
				{
					feetContactForces[3 * ff + j] += transForce[j];
				}
			}

			if ((ct[i].body0 >= 0) && (torsoFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				int ff = torsoFlag[ct[i].body0];
				int a = ff;
				Transform& inv = agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));
				
				bool success;
				auto invR = Inverse(R, success);

				Vec3 transForce = lambdaScale * ct[i].lambda * invR * Vec3(ct[i].normal);

				for (int j = 0; j < 3; ++j)
				{
					torsoContactForces[3 * ff + j] -= transForce[j];
				}
				numCollideOther[ff]++;
				stepsOnGround[ff]++;
			}

			if ((ct[i].body1 >= 0) && (torsoFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				int ff = torsoFlag[ct[i].body1];
				int a = ff;
				Transform& inv = agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));

				bool success;
				auto invR = Inverse(R, success);

				Vec3 transForce = lambdaScale * ct[i].lambda * invR * Vec3(ct[i].normal);

				for (int j = 0; j < 3; ++j)
				{
					torsoContactForces[3 * ff + j] += transForce[j];
				}
				numCollideOther[ff]++;
				stepsOnGround[ff]++;
			}
		}

		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	float AliveBonus(float z, float pitch, int a)
	{
		if (z > terminationZ && stepsOnGround[a] < maxStepsOnGround)
		{
			return 0.5f;
		}
		else
		{
			return -1.f;
		}
	}

	float AliveBonus(float z, float pitch)
	{
		return 1.f;
	}
};


class RLAntParkour : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

	vector<int> frontLeftFoot;
	vector<int> frontRightFoot;
	vector<int> backLeftFoot;
	vector<int> backRightFoot;
	vector<int> torso;

	vector<int> footFlag;
	vector<int> torsoFlag;

	vector<int> stepsOnGround;
	int maxCollisionsWithGround;

	vector<float> feetContactForces;
	vector<float> torsoContactForces;

	NvFlexVector<NvFlexRay>* rays;
	NvFlexVector<NvFlexRayHit>* hits;
	vector<float> hitHeight;

	float gridCellSize;
	float gridUpDisplacement;
	float terrainUpOffset;
	vector<Vec3> grid;
	
	int numRaysPerSide;
	int numRaysPerAgent;
	int numRays;

	int terrainInd;

	// Debug and test
	bool renderRaysNormals;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, frontLeftFoot);
		LoadVec(f, frontRightFoot);
		LoadVec(f, backLeftFoot);
		LoadVec(f, backRightFoot);
		LoadVec(f, footFlag);
		LoadVec(f, torsoFlag);
		LoadVec(f, feetContactForces);
		LoadVec(f, torsoContactForces);
	}

	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, frontLeftFoot);
		SaveVec(f, frontRightFoot);
		SaveVec(f, backLeftFoot);
		SaveVec(f, backRightFoot);
		SaveVec(f, footFlag);
		SaveVec(f, torsoFlag);
		SaveVec(f, feetContactForces);
		SaveVec(f, torsoContactForces);
	}

	RLAntParkour()
	{
		loadPath = "../../data/ant.xml";
		mNumAgents = 100;
		numPerRow = 10;
		spacing = 12.f;

		mNumActions = 8;
		mMaxEpisodeLength = 1000;

		numRaysPerSide = 15;
		numRaysPerAgent = numRaysPerSide * numRaysPerSide;
		numRays = mNumAgents * numRaysPerSide * numRaysPerSide;
		mNumObservations = 2 * mNumActions + 11 + 15 + mNumActions + numRaysPerAgent; // 50, was 39, was 28
		
		gridCellSize = 0.4f;
		gridUpDisplacement = 1.5f;

		terrainInd = -1;
		renderRaysNormals = false;

		maxCollisionsWithGround = 10;

		g_numSubsteps = 4;
		g_params.numIterations = 30;

		g_sceneLower = Vec3(-1.f);
		g_sceneUpper = Vec3(80.f, 4.f, 75.f);

		g_lightDistance *= 2.f;

		numFeet = 4;

		powerScale = 0.045f;
		terrainUpOffset = 0.85f;
		yOffset = terrainUpOffset + 0.6f;
		terminationZ = -0.29f; // Or when collide ground?

		electricityCostScale = 1.f;
		stallTorqueCostScale = 1.f;
		footCollisionCost = -0.8f;
		jointsAtLimitCost = -0.2f;

		angleResetNoise = 0.02f;
		angleVelResetNoise = 0.02f;
		velResetNoise = 0.02f;

		pushFrequency = 250;	// How much steps in average per 1 kick
		forceMag = 0.005f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
		stepsOnGround.resize(mNumAgents, 0);

		feetContactForces.resize(3 * numFeet * mNumAgents, 0.f);
		torsoContactForces.resize(3 * mNumAgents, 0.f);

		LoadEnv();

		footFlag.resize(g_buffers->rigidBodies.size());
		torsoFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
			torsoFlag[i] = -1;
		}

		for (int i = 0; i < mNumAgents; i++)
		{
			footFlag[frontLeftFoot[i]] = numFeet * i;
			footFlag[frontRightFoot[i]] = numFeet * i + 1;
			footFlag[backLeftFoot[i]] = numFeet * i + 2;
			footFlag[backRightFoot[i]] = numFeet * i + 3;
			torsoFlag[torso[i]] = i;
		}

		for (int i = 0; i < (int)g_buffers->rigidShapes.size(); ++i)
		{
			g_buffers->rigidShapes[i].filter = 1<<5;
			g_buffers->rigidShapes[i].group = -1;
			g_buffers->rigidShapes[i].material.friction = 1.0f;
			g_buffers->rigidShapes[i].material.rollingFriction = 0.05f;
			g_buffers->rigidShapes[i].material.torsionFriction = 0.05f;
		}
	
		rays = new NvFlexVector<NvFlexRay>(g_flexLib, numRays);
		hits = new NvFlexVector<NvFlexRayHit>(g_flexLib, numRays);
		grid.resize(numRaysPerAgent, Vec3());
		hitHeight.resize(numRays, -10.f);

		float translate = float(numRaysPerSide / 2) * gridCellSize;
		for (int i = 0; i < numRaysPerSide; ++i)
		{
			for (int j = 0; j < numRaysPerSide; ++j)
			{
				grid[i * numRaysPerSide + j] = Vec3(float(i) * gridCellSize - translate, gridUpDisplacement, float(j) * gridCellSize - translate);
			}
		}

		Transform terrainTrans = Transform(Vec3(80.f, terrainUpOffset, 100.f), Quat());
		terrainInd = createTerrain(210.f, 240.f, 85, 100, RandVec3() * 8.f, terrainTrans, Vec3(25.f, 1.2f, 25.f), 5, 0.41f);

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

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

	virtual void Update()
	{
		rays->map();
		for (int a = 0; a < mNumAgents; ++a)
		{
			pair<int, int> p = agentBodies[a];

			Transform robotPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&robotPose);

			for (int i = 0; i < numRaysPerAgent; ++i)
			{
				Vec3 origin = Vec3(robotPose.p.x, robotPose.p.y, robotPose.p.z) + grid[i];

				Vec3 dir = Vec3(0.f, -1.f, 0.f);

				NvFlexRay ray;
				(Vec3&)ray.start = origin;
				(Vec3&)ray.dir = dir;
				ray.filter = 1<<5;
				ray.group = -1;
				ray.maxT = 4.f;

				(*rays)[a * numRaysPerAgent + i] = ray;
			}
		}
		rays->unmap();

		NvFlexRayCast(g_solver, rays->buffer, hits->buffer, rays->size());
		rays->map();
		hits->map();
		for (int i = 0; i < hits->size(); ++i)
		{
			NvFlexRay ray = (*rays)[i];
			NvFlexRayHit hit = (*hits)[i];

			if (hit.t < ray.maxT)
			{
				hitHeight[i] = (Vec3(ray.start) + Vec3(ray.dir) * hit.t).y;
			}
			else
			{
				hitHeight[i] = -10.f;
			}
		}
		rays->unmap();
		hits->unmap();
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
		for (int i = 0; i < mNumActions; i++)
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
		Transform bodies[25];

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
		// 11
		for (int i = 0; i < 11; ++i) 
		{
			state[ct++] = more[i]; // 0 - 10
		}
		// 16
		for (int i = 0; i < mNumActions * 2; ++i)
		{
			state[ct++] = joints[i]; // 11 - 18
		}
		// 12
		float lambdaScale = 1e-2f;
		for (int i = 0; i < numFeet; i++)
		{
			int footInd = numFeet * a + i;
			for (int j = 0; j < 3; ++j)
			{
				state[ct++] = lambdaScale * feetContactForces[3 * footInd + j]; // 18 - 29
			}
		}
		// 3
		for (int j = 0; j < 3; ++j)
		{
			state[ct++] = lambdaScale * torsoContactForces[3 * a + j]; // 30 - 32
		}

		// 8
		float* prevActions = GetAction(a);
		if (prevActions) // could be null if this is called for the first time before agent acts
		{
			for (int i = 0; i < mNumActions; ++i)
			{
				state[ct++] = prevActions[i]; // 33 - 40
			}
		}

		// 15 x 15 = 225
		for (int j = 0; j < numRaysPerAgent; ++j)
		{
			state[ct++] = hitHeight[numRaysPerAgent * a + j] - z; // 0 - 224
		}

		for (int i = 0; i < ct; ++i)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}

		if (ct != mNumObservations)
		{
			cout << "Num of observations is wrong: " << mNumObservations << " != " << ct << endl;
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

		float heightAboveGround = state[0] - hitHeight[a * numRaysPerAgent + numRaysPerAgent / 2];
	//	float alive = AliveBonus(state[0] + initialZ, p, a); //  # state[0] is body height above ground, body_rpy[1] is pitch
		float alive = AliveBonus(heightAboveGround + initialZ, p, a);
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
		for (int i = 0; i < mNumActions; i++)
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

		electricityCostCurrent += electrCost * sum / (float)mNumActions;

		sum = 0.0f;
		for (int i = 0; i < mNumActions; i++)
		{
			sum += action[i] * action[i];
		}
		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
		}

		electricityCostCurrent += stallTorqCost * sum / (float)mNumActions;

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

	void ResetAgent(int a)
	{
		stepsOnGround[a] = 0;
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

	virtual void ClearContactInfo()
	{
		for (auto &ff : feetContactForces)
		{
			ff = 0.f;
		}

		for (auto &tf : torsoContactForces)
		{
			tf = 0.f;
		}

		for (auto &nc : numCollideOther)
		{
			nc = 0;
		}
	}

	virtual void FinalizeContactInfo() override
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

		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				int ff = footFlag[ct[i].body0];

				// Todo - remove copy-paste!
				int a = ff / numFeet;
				Transform& inv = agentOffsetInv[a]; // ? agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[ct[i].body0], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));
				
				bool success;
				auto invR = Inverse(R, success);

				Vec3 transForce = ct[i].lambda * invR * Vec3(ct[i].normal);

				for (int j = 0; j < 3; ++j)
				{
					feetContactForces[3 * ff + j] -= transForce[j];
				}
			}

			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				int ff = footFlag[ct[i].body1];

				int a = ff / numFeet;
				Transform& inv = agentOffsetInv[a]; // ? agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[ct[i].body1], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));

				bool success;
				auto invR = Inverse(R, success);


				Vec3 transForce = ct[i].lambda * invR * Vec3(ct[i].normal);
				for (int j = 0; j < 3; ++j)
				{
					feetContactForces[3 * ff + j] += transForce[j];
				}
			}

			if ((ct[i].body0 >= 0) && (torsoFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
			{
				int a = torsoFlag[ct[i].body0];

				Transform& inv = agentOffsetInv[a]; // ? agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[ct[i].body0], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));

				bool success;
				auto invR = Inverse(R, success);

				Vec3 transForce = ct[i].lambda * invR * Vec3(ct[i].normal);

				for (int j = 0; j < 3; ++j)
				{
					torsoContactForces[3 * a + j] -= transForce[j];
				}
				numCollideOther[a]++;

				if (ct[i].body1 < 0)
				{
					stepsOnGround[a]++;
				}
			}

			if ((ct[i].body1 >= 0) && (torsoFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
			{
				int a = torsoFlag[ct[i].body1];

				Transform& inv = agentOffsetInv[a]; // ? agentOffsetInv[torso[a]];

				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[ct[i].body1], (NvFlexRigidPose*)&torsoPose);
				torsoPose = inv * torsoPose;

				Vec3 headinVec = GetBasisVector0(torsoPose.q);
				Vec3 xv = Normalize(headinVec);
				Vec3 yv = Cross(Vec3(0.f, 0.f, 1.f), xv);
				auto R = Matrix33(xv, yv, Vec3(0.f, 0.f, 1.f));

				bool success;
				auto invR = Inverse(R, success);

				Vec3 transForce = ct[i].lambda * invR * Vec3(ct[i].normal);

				for (int j = 0; j < 3; ++j)
				{
					torsoContactForces[3 * a + j] += transForce[j];
				}
				numCollideOther[a]++;

				if (ct[i].body0 < 0)
				{
					stepsOnGround[a]++;
				}
			}
		}

		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	virtual void Draw(int pass)
	{
		if (renderRaysNormals)
		{
			if (pass == 0)
			{
				hits->map();
				rays->map();

				BeginLines(0.7f);

				for (int i = 0; i < hits->size(); ++i)
				{
					NvFlexRay ray = (*rays)[i];
					NvFlexRayHit hit = (*hits)[i];

					if (hit.t < ray.maxT)
					{
						DrawLine(Vec3(ray.start), Vec3(ray.start) + Vec3(ray.dir) * hit.t, Vec4(0.06f, 0.8f, 0.1f));
					}
				}

				EndLines();

				BeginLines(2.f);

				for (int i = 0; i < hits->size(); ++i)
				{
					NvFlexRay ray = (*rays)[i];
					NvFlexRayHit hit = (*hits)[i];

					if (hit.t < ray.maxT)
					{
						Vec3 hitPoint = Vec3(ray.start) + Vec3(ray.dir) * hit.t;
						DrawLine(hitPoint, hitPoint + 0.3f * Vec3(hit.n), Vec4(0.96f, 0.08f, 0.1f));
					}
				}

				EndLines();

				hits->unmap();
				rays->unmap();
			}
		}
	}
	
	virtual void DoGui()
	{
		if (imguiCheck("Render rays and normals", renderRaysNormals))
		{
			renderRaysNormals = !renderRaysNormals;
		}

		if (imguiCheck("Flagrun", doFlagRun))
		{
			doFlagRun = !doFlagRun;
		}
	}

	float AliveBonus(float z, float pitch, int a)
	{
		if (stepsOnGround[a] < maxCollisionsWithGround)
		{
			return 0.5f;
		}
		else
		{
			return -1.f;
		}
	}

	float AliveBonus(float z, float pitch)
	{
		return 1.f;
	}
};


class RLHumanoidHardParkour : public RigidHumanoidHard
{
public:
	
	bool firstFrame = true; 

	// Terrain variables
	SpawnableBigObstacles obstacles, topObstacles;
	SpawnableGapObstacles gapObstacles;
	SpawnableTerrain terrain;
	SpawnableStairs stairs, pyramids;
	bool spawnGroundObstacles, spawnTopObstacles, spawnTerrain, spawnStairs, spawnPyramids, spawnGaps;
	// planar rotation angle for blocks
	float minAngle, maxAngle;
	float topMinAngle, topMaxAngle;

	// Raycast variables
	vector<float> groundHits;
	NvFlexVector<NvFlexRay>* rays;
	NvFlexVector<NvFlexRayHit>* hits;
	vector<float> hitHeight;

	float gridCellSize;
	float gridUpDisplacement;
	vector<Vec3> grid;

	vector<float> smoothedHeadingAngles;
	vector<float> headingAngleDeltas;

	float downVelocityThreashold;

	int numForwardRays, numSideRays;
	int raysOffsetFront, raysOffsetSide;
	int numRaysPerAgent; // rays only used for observations
	int numTotalRaysPerAgent; // includes rays for measuring ground height
	int numRays;
	bool singlePointRays;
	float maxRayHitDist, singlePointRayHeight;
	float forwardRayOffsetScale, raycastHeightOffset;
	int forwardRayOffsetStart;

	// used for height measurement
	int numRaysHeight; 
	float gridCellSizeHeight;
	bool useRaycastHeight; // if true, agent height is relative to the surface beneath it. otherwise use agent height directly.
	vector<float> groundHeights;

	int parentNumObservations;
	bool renderRaysNormals;
	int MaxActions = 100;

	RLHumanoidHardParkour()
	{
		initialZ = 0.f;
		renderRaysNormals = false;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		mNumObservations = 2 * mNumActions + 11 + 3 * 2 + 1 + mNumActions; // + mNumActions; // 78, old - 52;

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

		PopulateTerrain();
		PrepareRays();
		mNumObservations += numRaysPerAgent;

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.TryParseJson(g_sceneJson);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
	}

	void PopulateTerrain()
	{
		downVelocityThreashold = GetJsonVal(g_sceneJson, "DownVelocityThreashold", -100.f);

		// Gap env. All other spawn variables should be false if spawnGaps is true.
		spawnGaps = GetJsonVal(g_sceneJson, "SpawnGaps", false);
		if (spawnGaps)
		{
			gapObstacles = SpawnableGapObstacles(
				spacing,
				GetJsonVal(g_sceneJson, "MinGap", 0.5f),
				GetJsonVal(g_sceneJson, "MaxGap", 1.5f),
				GetJsonVal(g_sceneJson, "GapsMinHeight", 2.95f),
				GetJsonVal(g_sceneJson, "GapsMaxHeight", 3.05f)
			);

			gapObstacles.SpawnGrid(Vec3(-2.f * spacing, 0.f, -2.f * spacing), 0.f, numPerRow + 4, spacing,
				mNumAgents / numPerRow + 4, spacing,
				GetJsonVal(g_sceneJson, "GapsMinAngle", -0.01f * kPi),
				GetJsonVal(g_sceneJson, "GapsMaxAngle", 0.01f * kPi));
		}

		spawnGroundObstacles = GetJsonVal(g_sceneJson, "SpawnGroundObstacles", false);
		if (spawnGroundObstacles)
		{
			minAngle = GetJsonVal(g_sceneJson, "MinAngle", -0.5f * kPi);
			maxAngle = GetJsonVal(g_sceneJson, "MaxAngle", 0.5f * kPi);
			obstacles = SpawnableBigObstacles(
				numPerRow * spacing, int(mNumAgents / numPerRow) * spacing,
				// the length, height, and width params of blocks are all *half* values
				GetJsonVal(g_sceneJson, "Density", 0.05f),
				GetJsonVal(g_sceneJson, "MinWidth", 0.5f),
				GetJsonVal(g_sceneJson, "MaxWidth", 2.5f),
				GetJsonVal(g_sceneJson, "MinHeight", 0.05f),
				GetJsonVal(g_sceneJson, "MaxHeight", 0.6f),
				GetJsonVal(g_sceneJson, "MinLength", 0.05f),
				GetJsonVal(g_sceneJson, "MaxLength", 1.5f),
				0.f, 0.f
			);
		}

		// Top layer
		spawnTopObstacles = GetJsonVal(g_sceneJson, "SpawnTopObstacles", false);
		if (spawnTopObstacles)
		{
			topMinAngle = GetJsonVal(g_sceneJson, "TopMinAngle", -0.5f * kPi);
			topMaxAngle = GetJsonVal(g_sceneJson, "TopMaxAngle", 0.5f * kPi);
			topObstacles = SpawnableBigObstacles(
				numPerRow * spacing, int(mNumAgents / numPerRow) * spacing,
				GetJsonVal(g_sceneJson, "TopDensity", 0.02f),
				GetJsonVal(g_sceneJson, "TopMinWidth", 0.5f),
				GetJsonVal(g_sceneJson, "TopMaxWidth", 2.f),
				GetJsonVal(g_sceneJson, "TopMinHeight", 0.05f),
				GetJsonVal(g_sceneJson, "TopMaxHeight", 0.1f),
				GetJsonVal(g_sceneJson, "TopMinLength", 0.05f),
				GetJsonVal(g_sceneJson, "TopMaxLength", 1.25f),
				GetJsonVal(g_sceneJson, "TopMinHeightOffset", 1.25f),
				GetJsonVal(g_sceneJson, "TopMaxHeightOffset", 1.75f),
				Vec3(0.9f, 0.1f, 0.05f)
			);
		}

		// Terrain
		spawnTerrain = GetJsonVal(g_sceneJson, "SpawnTerrain", true);
		if (spawnTerrain)
		{
			terrain = SpawnableTerrain(
				GetJsonVal(g_sceneJson, "SizeX", 300.f),
				GetJsonVal(g_sceneJson, "SizeZ", 300.f),
				GetJsonVal(g_sceneJson, "NumSubdiv", 20),
				GetJsonVal(g_sceneJson, "TerrainUpOffset", 0.4f),
				GetJsonVal(g_sceneJson, "TerrainUpOffsetRange", 0.02f),
				GetJsonVal(g_sceneJson, "OctavesMin", 5),
				GetJsonVal(g_sceneJson, "OctavesMax", 7),
				GetJsonVal(g_sceneJson, "Persistance", 0.4f),
				GetJsonVal(g_sceneJson, "PersistanceRange", 0.02f),
				GetJsonVal(g_sceneJson, "XScale", 25.f),
				GetJsonVal(g_sceneJson, "ZScale", 25.f),
				GetJsonVal(g_sceneJson, "YScaleMin", 0.7f),
				GetJsonVal(g_sceneJson, "YScaleMax", 0.9f)
			);
		}

		// Stairs
		spawnStairs = GetJsonVal(g_sceneJson, "SpawnStairs", false);
		if (spawnStairs)
		{
			stairs = SpawnableStairs(
				GetJsonVal(g_sceneJson, "StairsMinWidth", 7.f),
				GetJsonVal(g_sceneJson, "StairsMaxWidth", 7.f),
				GetJsonVal(g_sceneJson, "StairsMinLength", 9.f),
				GetJsonVal(g_sceneJson, "StairsMaxLength", 9.3f),
				GetJsonVal(g_sceneJson, "StairsMinStepLength", .35f),
				GetJsonVal(g_sceneJson, "StairsMaxStepLength", .5f),
				GetJsonVal(g_sceneJson, "StairsMinStepHeight", .16f),
				GetJsonVal(g_sceneJson, "StairsMaxStepHeight", .24f),
				GetJsonVal(g_sceneJson, "StairsMinNumSteps", 5),
				GetJsonVal(g_sceneJson, "StairsMaxNumSteps", 15)
			);
		}

		// Pyramids
		spawnPyramids = GetJsonVal(g_sceneJson, "SpawnPyramids", false);
		if (spawnPyramids)
		{
			pyramids = SpawnableStairs(
				GetJsonVal(g_sceneJson, "StairsMinWidth", 8.f),
				GetJsonVal(g_sceneJson, "StairsMaxWidth", 8.f),
				GetJsonVal(g_sceneJson, "StairsMinLength", 8.f),
				GetJsonVal(g_sceneJson, "StairsMaxLength", 8.f),
				GetJsonVal(g_sceneJson, "StairsMinStepLength", .35f),
				GetJsonVal(g_sceneJson, "StairsMaxStepLength", .5f),
				GetJsonVal(g_sceneJson, "StairsMinStepHeight", .16f),
				GetJsonVal(g_sceneJson, "StairsMaxStepHeight", .24f),
				GetJsonVal(g_sceneJson, "StairsMinNumSteps", 5),
				GetJsonVal(g_sceneJson, "StairsMaxNumSteps", 15)
			);
		}

		if (doFlagRun)
		{
			if (spawnGroundObstacles)
			{
				obstacles.SpawnGrid(Vec3(-2.f * spacing, 0.f, -2.f * spacing), spacing / 2.f, numPerRow + 3,
					spacing / 2.f, int(mNumAgents / numPerRow) + 3, spacing, minAngle, maxAngle);
			}
			if (spawnTopObstacles)
			{
				topObstacles.SpawnGrid(Vec3(-2.f * spacing, 0.f, -2.f * spacing), spacing / 2.f, numPerRow + 3,
					spacing / 2.f, int(mNumAgents / numPerRow) + 3, spacing, topMinAngle, topMaxAngle);
			}
		}
		else
		{
			if (spawnGroundObstacles)
			{
				obstacles.SpawnGrid(Vec3(), 0.f, numPerRow, spacing / 2.f, int(mNumAgents / numPerRow), spacing, minAngle, maxAngle, false);
			}
			if (spawnTopObstacles)
			{
				topObstacles.SpawnGrid(Vec3(), 0.f, numPerRow, spacing / 2.f, int(mNumAgents / numPerRow), spacing, topMinAngle, topMaxAngle, false);
			}
		}

		if (spawnTerrain)
		{
			terrain.Spawn(Vec3(Vec3(-2.f * spacing, 0.f, -2.f * spacing)));
		}

		// Stairs
		if (spawnStairs)
		{
			stairs.SpawnGrid(Vec3(), 0.f, max(numPerRow/4, 1), spacing * 4.f, 
										  max(mNumAgents / numPerRow / 4, 1), spacing * 4.f, 
				GetJsonVal(g_sceneJson, "StairsMinAngle", 0.f),
				GetJsonVal(g_sceneJson, "StairsMaxAngle", 0.f)
			);
		}

		// Pyramids
		if (spawnPyramids)
		{
			pyramids.SpawnGrid(Vec3(), 0.f, max(numPerRow / 4, 1), spacing * 4.f,
				max(mNumAgents / numPerRow / 4, 1), spacing * 4.f,
				GetJsonVal(g_sceneJson, "StairsMinAngle", 0.f),
				GetJsonVal(g_sceneJson, "StairsMaxAngle", 0.f)
			);
		}
	}

	void PrepareRays()
	{
		smoothedHeadingAngles.resize(mNumAgents, 0.f);
		headingAngleDeltas.resize(mNumAgents, 0.f);

		// Reading config
		numForwardRays = GetJsonVal(g_sceneJson, "NumForwardRays", 13);
		numSideRays = GetJsonVal(g_sceneJson, "NumSideRays", 13);
		raysOffsetFront = GetJsonVal(g_sceneJson, "RaysOffsetFront", 0);
		raysOffsetSide = GetJsonVal(g_sceneJson, "RaysOffsetSide", 0);
		singlePointRays = GetJsonVal(g_sceneJson, "SinglePointRays", false);
		maxRayHitDist = GetJsonVal(g_sceneJson, "MaxRayHitDist", 4.f);
		singlePointRayHeight = GetJsonVal(g_sceneJson, "SinglePointRayHeight", 1.f);
		forwardRayOffsetScale = GetJsonVal(g_sceneJson, "ForwardRayOffsetScale", 0.f);
		forwardRayOffsetStart = GetJsonVal(g_sceneJson, "ForwardRayOffsetStart", 0);
		useRaycastHeight = GetJsonVal(g_sceneJson, "UseRaycastHeight", true);
		gridCellSize = GetJsonVal(g_sceneJson, "GridCellSize", 0.2f);

		numRaysPerAgent = numForwardRays * numSideRays;
		gridUpDisplacement = 0.f;

		// For measuring ground heights
		numRaysHeight = 2;
		gridCellSizeHeight = 0.1f;
		
		// raycasts for observation + current ground height
		numTotalRaysPerAgent = numRaysPerAgent + numRaysHeight * numRaysHeight;
		numRays = mNumAgents * numTotalRaysPerAgent;

		groundHits.resize(numRaysHeight * numRaysHeight, 0.f);

		rays = new NvFlexVector<NvFlexRay>(g_flexLib, numRays);
		hits = new NvFlexVector<NvFlexRayHit>(g_flexLib, numRays);
		
		// first section for raycast observations, second section for current ground height measurement
		grid.resize(numTotalRaysPerAgent, Vec3());
		hitHeight.resize(numRays, -10.f);
		groundHeights.resize(mNumAgents);

		float translateX = (float(numSideRays / 2) - raysOffsetSide) * gridCellSize;
		float translateZ = (float(numForwardRays / 2) - raysOffsetFront) * gridCellSize;
		float forwardOffset = 0;
		for (int j = 0; j < numForwardRays; ++j)
		{
			float variableFowardRayOffset = gridCellSize + float(max(j - forwardRayOffsetStart, 0)) * forwardRayOffsetScale;
			forwardOffset += variableFowardRayOffset;
			for (int i = 0; i < numSideRays; ++i)
			{

				grid[i * numForwardRays + j] = Vec3(float(i) * gridCellSize - translateX,
					gridUpDisplacement,
					forwardOffset - translateZ);
			}
		}
		float translateHeight = float(numRaysHeight / 2) * gridCellSizeHeight;
		for (int i = 0; i < numRaysHeight; ++i)
		{
			for (int j = 0; j < numRaysHeight; j++)
			{
				grid[numRaysPerAgent + i * numRaysHeight + j] = Vec3(
					float(i) * gridCellSizeHeight - translateHeight,
					gridUpDisplacement,
					float(j) * gridCellSizeHeight - translateHeight);
			}
		}

		// mNumObservations = parentNumObservations + numRaysPerAgent;
	}

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		// Prepare state
		//--------------------
		numJointsAtLimit = 0;
		
		float joints[25 * 2];
		float angles[25];
		float lows[25];
		float highs[25];

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

		getEulerZYX(bodyPose.q, yaw, p, r);
		
		Transform torsoPose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[a]], (NvFlexRigidPose*)&torsoPose);
		float bodyHeight = torsoPose.p.y;
		float agentHeight = bodyHeight;
		if (useRaycastHeight)
		{
			agentHeight -= groundHeights[a];
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

				potentials[a] = -walkTargetDist / dt + 100.f * potentialLeak(a);
				potentialsOld[a] = potentials[a];
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
		Vec3 bangVel = mat * angVel; // ?
		float avx = bangVel.x;
		float avy = bangVel.y;
		float avz = bangVel.z;

		float more[11] = { agentHeight,
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

		state[ct++] = headingAngleDeltas[a];

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

		// Raycast
		for (int j = 0; j < numRaysPerAgent; ++j)
		{
			state[ct++] = 0.2f * (hitHeight[a * numTotalRaysPerAgent + j] - groundHeights[a]); 
		}

		for (int i = 0; i < mNumObservations; ++i)
		{
			state[i] = Clamp(state[i], -5.f, 5.f);
		}

		if (ct != mNumObservations)
		{
			printf("Num of observations is wrong. Got %d expected %d\n", ct, mNumObservations);
		}
	}

	void ResetAgent(int a)
	{
		smoothedHeadingAngles[a] = 0.f;
		RigidHumanoidHard::ResetAgent(a);
	}

	void UpdateAgentStartOffset()
	{
		NvFlexVector<NvFlexRay>* raysSpawn = new NvFlexVector<NvFlexRay>(g_flexLib, mNumAgents);
		NvFlexVector<NvFlexRayHit>* hitsSpawn = new NvFlexVector<NvFlexRayHit>(g_flexLib, mNumAgents);

		raysSpawn->map();
		hitsSpawn->map();

		float rayHeight = 100.f; // number large enough to be higher than anything in the terrain

		// create current height raycasts
		for (int a = 0; a < mNumAgents; a++)
		{
			NvFlexRay ray;
			(Vec3&)ray.start = agentStartOffset[a].p + Vec3(0.f, rayHeight, 0.f);
			(Vec3&)ray.dir = Vec3(0.f, -1.f, 0.f);
			ray.filter = 1 << 5;
			ray.group = -1;
			ray.maxT = fabs(agentStartOffset[a].p.y) * 2.f + rayHeight;
			(*raysSpawn)[a] = ray;
		}
		raysSpawn->unmap();
		hitsSpawn->unmap();

		NvFlexRayCast(g_solver, raysSpawn->buffer, hitsSpawn->buffer, raysSpawn->size());

		raysSpawn->map();
		hitsSpawn->map();

		// read ground height vals
		for (int a = 0; a < mNumAgents; a++)
		{
			NvFlexRay ray = (*raysSpawn)[a];
			NvFlexRayHit hit = (*hitsSpawn)[a];

			float height = (Vec3(ray.start) + Vec3(ray.dir) * hit.t).y + 0.02f;

			// increase agent offset
			Transform heightT(Vec3(0.f, height, 0.f), Quat());
			agentStartOffset[a] = heightT * agentStartOffset[a];
			agentOffset[a] = heightT * agentOffset[a];
			agentOffsetInv[a] = Inverse(agentOffset[a]);

			ResetAgent(a);
		}
	}

	virtual void Update()
	{
		if (firstFrame)
		{
			if (mDoLearning)
			{
				UpdateAgentStartOffset();
			}
			firstFrame = false;
		}

		RigidHumanoidHard::Update();
	}

	virtual void FinalizeContactInfo()
	{
		RigidHumanoidHard::FinalizeContactInfo();
		
		UpdateCOM();
	
		rays->map();
		hits->map();
		CastRays();

		for (int i = 0; i < hits->size(); ++i)
		{
			NvFlexRay ray = (*rays)[i];
			NvFlexRayHit hit = (*hits)[i];

			if (hit.t < ray.maxT)
			{
				hitHeight[i] = (Vec3(ray.start) + Vec3(ray.dir) * hit.t).y;
			}
			else
			{
				hitHeight[i] = -10.f;
			}
		}

		rays->unmap();
		hits->unmap();

		NvFlexRayCast(g_solver, rays->buffer, hits->buffer, rays->size());

		// Getting ground height
		for (int a = 0; a < mNumAgents; a++)
		{
			for (int i = 0; i < numRaysHeight * numRaysHeight; i++)
			{
				groundHits[i] = hitHeight[numTotalRaysPerAgent * a + numRaysPerAgent + i];
			}
			groundHeights[a] = getMedian(groundHits);
		}
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

	float getMedian(vector<float> &scores)
	{
		size_t size = scores.size();

		if (size == 0)
		{
			return 0;  // Undefined, really.
		}
		else
		{
			sort(scores.begin(), scores.end());
			if (size % 2 == 0)
			{
				return (scores[size / 2 - 1] + scores[size / 2]) / 2;
			}
			else 
			{
				return scores[size / 2];
			}
		}
	}

	virtual void CastRays()
	{
		for (int a = 0; a < mNumAgents; ++a)
		{
			// getting the robot heading angle
			Transform torsoPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[a]], (NvFlexRigidPose*)&torsoPose);
			Vec3 normalDir = GetBasisVector1(torsoPose.q);
			float headingAngle = ASin(Dot(normalDir, Vec3(0.f, 0.f, 1.f)));
			if (Dot(normalDir, Vec3(1.f, 0.f, 0.f)) < 0.f)
			{
				headingAngle = kPi - headingAngle;
			}
			headingAngleDeltas[a] = headingAngle - smoothedHeadingAngles[a];
			smoothedHeadingAngles[a] = headingAngle;

			// setting rays to track the robot's current translation and rotation
			Vec3 robotGlobalTra = Vec3(bodyXYZ[a].y, bodyXYZ[a].z, bodyXYZ[a].x) + agentOffset[a].p;
			Transform rotT(Vec3(), QuatFromAxisAngle(Vec3(0.f, 1.f, 0.f), -smoothedHeadingAngles[a]));
			for (int i = 0; i < numRaysPerAgent + numRaysHeight * numRaysHeight; ++i)
			{
				Vec3 rayPosGlobal = robotGlobalTra + TransformVector(rotT, grid[i]);
				Vec3 rayOrigin = rayPosGlobal;
				Vec3 rayDir = Vec3(0.f, -1.f, 0.f);
				if (singlePointRays && i < numRaysPerAgent) // for single point rays
				{
					rayOrigin = robotGlobalTra;
					rayDir = Normalize(Vec3(
						rayPosGlobal.x,
						groundHeights[a],
						rayPosGlobal.z
					) - rayOrigin);
				}

				NvFlexRay ray;
				(Vec3&)ray.start = rayOrigin;
				(Vec3&)ray.dir = rayDir;
				ray.filter = 1 << 5;
				ray.group = -1;
				ray.maxT = maxRayHitDist;

				(*rays)[a * numTotalRaysPerAgent + i] = ray;
			}
		}
	}

	virtual void DoGui()
	{
		if (imguiCheck("Render rays and normals", renderRaysNormals))
		{
			renderRaysNormals = !renderRaysNormals;
		}

		RigidHumanoidHard::DoGui();
	}

	virtual void Draw(int pass)
	{
		if (pass == 0)
		{
			if (renderRaysNormals)
			{
				// Draw heading direction arrow
				BeginLines(9.f);
				for (int a = 0; a < mNumAgents; ++a)
				{
					pair<int, int> p = agentBodies[a];

					Transform robotPose;
					NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&robotPose);

					float headingAngle = smoothedHeadingAngles[a];
					Transform rotT(Vec3(), QuatFromAxisAngle(Vec3(0.f, 1.f, 0.f), -headingAngle));

					for (int i = 0; i < numRaysPerAgent; ++i)
					{
						Vec3 dir = TransformVector(rotT, Vec3(0.f, 0.f, 1.f));
						Vec3 robotPos = Vec3(robotPose.p.x, robotPose.p.y, robotPose.p.z);
						DrawLine(robotPos, robotPos + 0.75f * dir, Vec4(0.12f, 0.7f, 0.1f));
					}
				}
				EndLines();

				hits->map();
				rays->map();
				
				// Draw rays
				BeginLines(0.7f);
				for (int i = 0; i < hits->size(); ++i)
				{
					NvFlexRay ray = (*rays)[i];
					NvFlexRayHit hit = (*hits)[i];

					if (true || hit.t < ray.maxT)
					{
						DrawLine(Vec3(ray.start), Vec3(ray.start) + Vec3(ray.dir) * hit.t, Vec4(0.06f, 0.8f, 0.1f));
					}
				}
				EndLines();

				// Draw ray hits
				BeginLines(2.5f);
				for (int i = 0; i < hits->size(); ++i)
				{
					NvFlexRay ray = (*rays)[i];
					NvFlexRayHit hit = (*hits)[i];
					
					if (hit.t < ray.maxT)
					{
						Vec3 hitPoint = Vec3(ray.start) + Vec3(ray.dir) * hit.t;
						DrawLine(hitPoint, hitPoint + 0.1f * Vec3(hit.n), Vec4(0.96f, 0.08f, 0.1f));
					}
				}
				EndLines();

				hits->unmap();
				rays->unmap();
			}
		}
	}

	virtual float AliveBonus(int a, float z, float pitch)
	{
		if (spawnGaps)
		{
			pair<int, int> p = agentBodies[a];
			// -2 because falling starts with delay => dead bonus is smaller than regular.
			if (g_buffers->rigidBodies[p.first].linearVel[1] < downVelocityThreashold)
			{
				return -1.25f;
			}
		}

		if (z < terminationZ)
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
};