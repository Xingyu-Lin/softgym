#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"
#include "rlbase.h"


class RLANYmal : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
    URDFImporter* urdf;

    vector<Transform> rigidTrans;
    map<string, int> jointMap;
    map<string, int> activeJointMap;

	vector<string> motors;
	vector<float> powers;

	Mesh mesh;

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, true, 1000.0f, 0.0f, 25.f, 0.01f, 10.f, 12.f, false, 1e10f);

		for (size_t j = 0; j < motors.size(); j++)
        {
            ctrl.push_back(make_pair(urdf->jointNameMap[motors[j]], eNvFlexRigidJointAxisTwist)); //TODO: Only support twist for now
            mpower.push_back(powers[j]);
        }

	//	torso[i] = urdf->rigidNameMap["utorso"];
    }

	RLANYmal(bool flagRun = false)
    {
		mNumAgents = 10;
		numPerRow = 10;
		spacing = 6.f;
		yOffset = 0.95f;

		// Set-up correct values
		mNumObservations = 70;
		mMaxEpisodeLength = 1000;

		g_params.shapeCollisionMargin = 0.01f;
		g_params.numPostCollisionIterations = 0;
		g_numSubsteps = 4;
		g_params.numIterations = 40;

		g_sceneLower = Vec3(-1.f);
		g_sceneUpper = Vec3(12.f, 2.f, 3.f);

		g_pause = true;

		doFlagRun = flagRun;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;

		powerScale = 1.f;
		initialZ = 0.0f;
		terminationZ = 0.5f;

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

		rigidTrans.clear();

		urdf = new URDFImporter("../../data", "atlas_description/urdf/atlas_v5_damp_simple_shapes.urdf");
	
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		// Todo for ANYmal - add correct feet and motor names from URDF
		//	feet = { "l_foot", "r_foot"	};
		//	motors = { "back_bkx", "back_bky", "back_bkz", "l_arm_elx",	"l_arm_ely", "l_arm_shx",
		//		"l_arm_shz", "l_arm_wrx", "l_arm_wry", "l_arm_wry2", "l_leg_akx", "l_leg_aky",
		//		"l_leg_hpx", "l_leg_hpy", "l_leg_hpz", "l_leg_kny",	"neck_ry", "r_arm_elx" ,
		//		"r_arm_ely", "r_arm_shx", "r_arm_shz", "r_arm_wrx",	"r_arm_wry", "r_arm_wry2",
		//		"r_leg_akx", "r_leg_aky", "r_leg_hpx", "r_leg_hpy",	"r_leg_hpz", "r_leg_kny"
		//	};

		numFeet = feet.size();

		mNumActions = motors.size();
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
		//		footFlag[i * numFeet + j] = i * numFeet + j;
			}
		}

        g_params.dynamicFriction = 1.0f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
		
        g_params.relaxationFactor = 0.5f;
        g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.005f;

        g_drawPoints = false;
        g_drawCloth = false;

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.agent_name = "ANYmal_128";
			ppo_params.resume = 0;
			ppo_params.num_timesteps = 1280000000;
			ppo_params.hid_size = 128;
			ppo_params.num_hid_layers = 2;

			ppo_params.timesteps_per_batch = 256;
			ppo_params.optim_batchsize_per_agent = 32;
			ppo_params.optim_epochs = 10;
			ppo_params.optim_stepsize = 5e-4f;
			ppo_params.optim_schedule = "adaptive";
			ppo_params.desired_kl = 0.02f;

			ppo_params.clip_param = 0.2f;
			ppo_params.gamma = 0.99f;
			ppo_params.entcoeff = 0.f;

			ppo_params.relativeLogDir = "Anymal";

			ppo_params.TryParseJson(g_sceneJson);

			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}
    }

	~RLANYmal()
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

		for (auto fc : feetContact)
		{
			fc = 0.f;
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
                    feetContact[ff] = lambdaScale * ct[i].lambda;
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
			return 1.;
		}
		else
		{
			return -1.f;
		}
	}
};


