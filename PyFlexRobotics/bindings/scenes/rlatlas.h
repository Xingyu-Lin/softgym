#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"
#include "rlbase.h"


class RLAtlas : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
    URDFImporter* urdf;

    vector<Transform> rigidTrans;
    map<string, int> jointMap;
    map<string, int> activeJointMap;

	vector<string> motors;
	vector<float> powers;

	vector<string> kneesNames;
	vector<int> knees;
	vector<string> feetNames;
	vector<int> feet;
	vector<int> footFlag;

	Mesh mesh;

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, true, 1000.0f, 0.0f, 25.f, 0.01f, 10.f, 12.f, false, 1e10f);

		for (size_t j = 0; j < motors.size(); j++)
        {
            ctrl.push_back(make_pair(urdf->jointNameMap[motors[j]], eNvFlexRigidJointAxisTwist));
            mpower.push_back(powers[j]);
        }

		for (auto rj : urdf->joints)
		{
			NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[rj->name]];
			if (rj->type == URDFJoint::REVOLUTE)
			{
				joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-3f;
				joint.damping[eNvFlexRigidJointAxisTwist] = 50.f;
			}
		}

		for (auto fn : feetNames)
		{
		//	int find = urdf->rigidNameMap[fn];
		//	cout << "Foot " << fn << " index = " << find << endl;
			feet.push_back(urdf->rigidNameMap[fn]);
		}

		for (auto kn : kneesNames)
		{
			//	int find = urdf->rigidNameMap[kn];
			//	cout << "Knee " << kn << " index = " << find << endl;
			knees.push_back(urdf->rigidNameMap[kn]);
		}

	//	pelvis[i] = urdf->rigidNameMap["pelvis"];
		torso[i] = urdf->rigidNameMap["utorso"];
    }

	RLAtlas()
	{
		mNumAgents = 200;
		numPerRow = 20;
		spacing = 11.f;
		yOffset = 0.95f;

		mMaxEpisodeLength = 1000;

		g_params.shapeCollisionMargin = 0.01f;
		g_params.numPostCollisionIterations = 0;
		g_numSubsteps = 4;
		g_params.numIterations = 50;

		g_sceneLower = Vec3(-40.f, -1.f, -40.f);
		g_sceneUpper = Vec3(100.f, 2.5f, 60.f);

		g_pause = true;

		numRenderSteps = 1;

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		powerScale = 2.f;
		initialZ = 0.0f;
		terminationZ = 0.82f;

		electricityCostScale = 2.f;
		stallTorqueCostScale = 0.5f;
		footCollisionCost = -1.f;
		jointsAtLimitCost = -1.f;

		angleResetNoise = 0.2f;
		angleVelResetNoise = 0.1f;
		velResetNoise = 0.1f;

		maxFlagResetSteps = 200;
		pushFrequency = 400;	// How much steps in average per 1 kick
		forceMag = 1.f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		rigidTrans.clear();

		urdf = new URDFImporter("../../data", "atlas_description/urdf/atlas_v5_damp_simple_shapes.urdf");
		kneesNames = { "l_leg_kny", "r_leg_kny" };
		feetNames = { "l_foot", "r_foot" };
		motors = { "back_bkx", "back_bky", "back_bkz", "l_arm_elx",	"l_arm_ely", "l_arm_shx",
			"l_arm_shz", "l_arm_wrx", "l_arm_wry", "l_arm_wry2", "l_leg_akx", "l_leg_aky",
			"l_leg_hpx", "l_leg_hpy", "l_leg_hpz", "l_leg_kny",	"neck_ry", "r_arm_elx" ,
			"r_arm_ely", "r_arm_shx", "r_arm_shz", "r_arm_wrx",	"r_arm_wry", "r_arm_wry2",
			"r_leg_akx", "r_leg_aky", "r_leg_hpx", "r_leg_hpy",	"r_leg_hpz", "r_leg_kny"
		};

		numFeet = feetNames.size();

		mNumActions = motors.size();
		mNumObservations = 3 * mNumActions + 11 + 2; // 103, was 70
		powers.clear();

		for (int i = 0; i < mNumActions; ++i)
		{
			float power = urdf->joints[i]->effort;
			cout << motors[i] << " power = " << power << endl;

			powers.push_back(power);
		}
		LoadEnv();

		footFlag.resize(g_buffers->rigidBodies.size());
		for (int i = 0; i < (int)g_buffers->rigidBodies.size(); i++)
		{
			initBodies.push_back(g_buffers->rigidBodies[i]);
			footFlag[i] = -1;
		}
		for (int i = 0; i < mNumAgents; i++)
		{
			for (int j = 0; j < numFeet; j++)
			{
				footFlag[feet[numFeet * i + j]] = numFeet * i + j;
			}
		}

        g_params.dynamicFriction = 1.0f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
		
        g_params.relaxationFactor = 0.25f;
        g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.005f;

        g_drawPoints = false;
        g_drawCloth = false;

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.agent_name = "AtlasFlagrun_256";
			ppo_params.resume = 8421;
			ppo_params.num_timesteps = 1280000000;
			ppo_params.hid_size = 256;
			ppo_params.num_hid_layers = 2;

			ppo_params.timesteps_per_batch = 25600;
			ppo_params.optim_batchsize_per_agent = 32;
			ppo_params.optim_epochs = 10;
			ppo_params.optim_stepsize = 5e-4f;
			ppo_params.optim_schedule = "adaptive";
			ppo_params.desired_kl = 0.02;

			ppo_params.clip_param = 0.2f;
			ppo_params.gamma = 0.99f;
			ppo_params.entcoeff = 0.f;

			ppo_params.relativeLogDir = "Atlas";

			ppo_params.TryParseJson(g_sceneJson);

			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
    }

	float footHeightVelPenalty(float height, float vel, float desiredHeight = 0.1f)
	{
		return (desiredHeight - height) * vel;
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

		float feetHeightVelRew = 0.f;
		float penaltyScale = -0.25f;

		for (int fi = 0; fi < numFeet; ++fi)
		{
			float height = 0.f;

			Transform footPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[feet[numFeet * a + fi]], (NvFlexRigidPose*)&footPose);
			height = footPose.p.y;

			Vec3 fVel = Vec3(g_buffers->rigidBodies[feet[numFeet * a + fi]].linearVel);
			fVel.y = 0.f;

			feetHeightVelRew += penaltyScale * footHeightVelPenalty(height, Length(fVel), 0.15f);
		}

		vector<float> rewards =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent,
			headingRew,
			feetHeightVelRew
		};

		rew = 0.f;
		for (auto rw : rewards)
		{
			rew += rw;
		}
	}

	~RLAtlas()
	{
		if (urdf)
		{
			delete urdf;
		}
	}
/*
	virtual void LoadEnv()
	{
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		torso.clear();
		torso.resize(mNumAgents, -1);

		pelvis.clear();
		pelvis.resize(mNumAgents, -1);

		for (int i = 0; i < mNumAgents; i++)
		{
			Vec3 pos = Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);
			Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise));
			rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(0.5f - rotCreateNoise, 0.5f + rotCreateNoise)) * rot;

			Transform gt(pos, rot);
			gt = gt * preTransform;

			int begin = g_buffers->rigidBodies.size();

			AddAgentBodiesJointsCtlsPowers(i, gt, ctrls[i], motorPower[i]);

			int end = g_buffers->rigidBodies.size();

			gt = gt * Inverse(preTransform);
			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
			agentBodies.push_back(make_pair(begin, end));
		}

		maxPower = *max_element(std::begin(motorPower[0]), std::end(motorPower[0]));

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint)*g_buffers->rigidJoints.size());
	}
*/
    void ResetAgent(int a)
    {
        for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
        {
            g_buffers->rigidBodies[i] = initBodies[i];
        }
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
            if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0.f))
            {
				//cout << "lambda1 = " << ct[i].lambda << endl;
                if (ct[i].body1 < 0)
                {
                    // foot contact with ground
                    int ff = footFlag[ct[i].body0];
                    feetContact[ff] += lambdaScale * ct[i].lambda;
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
				//cout << "lambda2 = " << ct[i].lambda << endl;
                if (ct[i].body0 < 0)
                {
                    // foot contact with ground
                    int ff = footFlag[ct[i].body1];
                    feetContact[ff] += lambdaScale * ct[i].lambda;
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

	virtual void resetTarget(int a, bool firstTime = true)
	{
		if (doFlagRun)
		{
			if (firstTime && (a % 3))
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

    virtual void DoGui()
    {
    }

    virtual void DoStats()
    {
    }

    virtual void Update()
    {
    }

    virtual void PostUpdate()
    {
        // joints are not read back by default
        //NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }

	virtual void Draw(int pass)
	{
	/*	if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i = 0; i < (int)g_buffers->triangles.size(); ++i)
			{
				mesh.m_indices[i] = g_buffers->triangles[i];
			}

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i = 0; i < (int)g_buffers->tetraIndices.size(); i += 4)
			{
				float vonMises = fabsf(g_buffers->tetraStress[i/4]);

				//printf("%f\n", vonMises);

				averageStress[g_buffers->tetraIndices[i + 0]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i + 1]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i + 2]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i + 3]] += Vec2(vonMises, 1.0f);

				rangeMin = Min(rangeMin, vonMises);
				rangeMax = Max(rangeMax, vonMises);
			}

			//printf("%f %f\n", rangeMin,rangeMax);

			rangeMin = 0.0f; //Min(rangeMin, vonMises);
			rangeMax = 0.5f; //Max(rangeMax, vonMises);

			for (int i=0; i < (int)g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x / averageStress[i].y);
			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);*/
	}

	virtual float AliveBonus(float z, float pitch)
	{
		if (z > terminationZ)
		{
			return 2.5f;
		}
		else
		{
			return -1.f;
		}
	}
};


