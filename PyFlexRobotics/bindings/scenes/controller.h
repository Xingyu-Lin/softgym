#pragma once
#include "rlbase.h"

using namespace std;
using namespace tinyxml2;

#include "../urdf.h"


class Controller : public FlexGymBase
{
public:
	std::default_random_engine generator = std::default_random_engine();
	
	bool save_state = false;
	bool load_state = false;
	
	// MPPI things
	int H = 10; // needs this here to do gui things
	int MAX_H = 100;
	float sigma = 0.3f;
	float lambda = 0.25f; // 0.25 for franka
	float alpha = 0.8125f;
	
	float u_cost = 1.0f;
	float u_cost_exp = -3.0f;
	float targetheight = 1.2f;
	
	vector<vector<float>> theta; // H x nu
	vector<vector<vector<float>>> noisyActions; // K x H x nu
	vector<float> costs; // K x H
	vector<float> weights; // K x H 
	
	//NvFlexSolver* opt_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc); // TODO!!!
	
	// management code
	vector<vector<float>> jointAngles;
	vector<vector<float>> jointVelocities;
	vector<float> jointLower;
	vector<float> jointUpper;
	
	// PLOTTING CODE
	vector<vector<float>> plot_controls;
	int plot_idx = 0;

	Controller()
	{
		loadPath = "../../data/humanoid_20_5.xml";
		
		mNumAgents = 128; //*2;
		mNumActions = 0;
		//mNumObservations = 2 * mNumActions + 11 + 2 + mNumActions; // 76, was 52
		mMaxEpisodeLength = 1000;
		
		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 4;
		g_params.numInnerIterations = 10;
		g_params.relaxationFactor = 0.75f;
		powerScale = 0.24f;
		
		//g_sceneLower = Vec3(-50.f, -1.f, -50.f);
		//	g_sceneUpper = Vec3(150.f, 4.f, 100.f);
		//g_sceneUpper = Vec3(120.f, 4.f, 90.f);
		g_sceneLower = Vec3(-1.0f); // look at primary agent
		g_sceneUpper = Vec3(1.0f);
		
		g_pause = true;
		numRenderSteps = 1;
		//doFlagRun = false;
		
		numPerRow = 11;
		spacing = 8.f;
		
		//numFeet = 2;
		
		//initialZ = 0.9f;
		//terminationZ = 0.795f;
		
		//electricityCostScale = 1.5f;
		//stallTorqueCostScale = 2.f;
		
		//maxX = 25.f;
		//maxY = 25.f;
		//maxFlagResetSteps = 170;
		
		g_params.dynamicFriction = 1.0f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.staticFriction = 0.6f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		
		g_params.gravity[0] = 0.0f;
		g_params.gravity[1] = 0.0f;
		g_params.gravity[2] = 0.0f;
		
		// No noise reset
		angleResetNoise = 0.0f;
		velResetNoise = 0.0f;
		angleVelResetNoise = 0.0f;
		rotCreateNoise = 0.0f;
		
		pushFrequency = 260;	// How much steps in average per 1 kick
		forceMag = 0.0f;
	}

    void PrepareScene() override
	{
		LoadEnv();

		// MPPI
		noisyActions.resize(mNumAgents);
		theta.resize(MAX_H);

		costs.resize(mNumAgents);
		weights.resize(mNumAgents);
		for (int ai=0; ai<mNumAgents; ai++)
		{
			noisyActions[ai].resize(MAX_H);
			for (int i = 0; i < MAX_H; i++)
			{
				noisyActions[ai][i].resize(GetNumControls());
				theta[i].resize(GetNumControls());
				for (int a=0; a<GetNumControls(); a++)
				{
					theta[i][a] = 0.0f;
				}
			}
			costs[ai] = 0.0f;
			weights[ai] = 0.0f;
		}

		// Common datas for rigid body robots
		jointAngles.resize(mNumAgents);
		jointVelocities.resize(mNumAgents);
		printf("num controls: %d\n", GetNumControls());
		jointLower.resize(GetNumControls());
		jointUpper.resize(GetNumControls());
		vector<float> angles;
		angles.resize(GetNumControls());
		GetAngles(0, angles, jointLower, jointUpper);

		for (int a = 0; a < GetNumControls(); a++)
			printf("%1.4f < --- > %1.4f\n", jointLower[a], jointUpper[a]);

		for (int i = 0; i < mNumAgents; i++)
		{
			jointAngles[i].resize(GetNumControls());
			jointVelocities[i].resize(GetNumControls());
		}

		plot_controls.resize(60);
		for (int i = 0; i < 60; i++)
		{
			plot_controls[i].resize(GetNumControls());
			for (int a=0; a<GetNumControls(); a++)
			{
				plot_controls[i][a] = 0.0f;
			}
		}
	}

	virtual void EnvCommonSetup() = 0;
	virtual void AddChildEnvBodies(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower) = 0;
	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower) = 0;
	
	void ParseJsonParams(const json& sceneJson) override
	{}
	virtual void PopulateState(int a, float* state) {}
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) {}
	
	virtual void LoadEnv()
	{
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
		masses.clear();
		
		EnvCommonSetup();
		
		for (int i = 0; i < mNumAgents; i++)
		{
			Vec3 pos = Vec3(float(i % numPerRow) * spacing, yOffset, float(i / numPerRow) * spacing);
			Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f);
			rot      = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * rot;
		
			Transform gt(pos, rot);
			gt = gt * preTransform;
			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
			agentStartOffset.push_back(gt); // should be same as agentOffset
		
			int begin = g_buffers->rigidBodies.size();
		
			AddAgentBodiesJointsCtlsPowers(i, gt, ctrls[i], motorPower[i]);
			AddChildEnvBodies(i, gt, ctrls[i], motorPower[i]);
		
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
	
	virtual void ResetAgent(int a) { }
	virtual void FinalizeContactInfo() { }
	virtual void ClearContactInfo() { }
	//virtual void GetBodyTransformAndAngles() { }
	//float AliveBonus(float z, float pitch) { return 0.0f; }
	
	void step()
	{
		// tick solver
		NvFlexSetParams(g_solver, &g_params);
		for (int s = 0; s < numRenderSteps; s++)
		{
			NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		}
		g_frame++;
		g_step = false;
	}
	
	// Reward function helper functions
	float euclidean(Transform t1, Transform t2)
	{
		return sqrt(sqeuclidean(t1, t2));
	}
	
	float sqeuclidean(Transform t1, Transform t2)
	{
		return pow(t1.p.x - t2.p.x, 2) + pow(t1.p.y - t2.p.y, 2) + pow(t1.p.z - t2.p.z, 2);
	}
	
	void saveAgent(int ai, Transform* bodyPose, Vec3* fullVels, Vec3* fullAVels)
	{
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		g_buffers->rigidBodies.map();
		
		GetGlobalPose(ai, bodyPose);
		
		Transform& inv = agentOffsetInv[ai];
		pair<int, int> p = agentBodies[ai];
		
		for (int i = p.first; i < p.second; i++)
		{
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&pose);
			
			Vec3 vel  = g_buffers->rigidBodies[i].linearVel;
			Vec3 avel = g_buffers->rigidBodies[i].angularVel;       
			
			int ind = i - p.first;
			fullVels[ind]  = Rotate(inv.q, vel);
			fullAVels[ind] = Rotate(inv.q, avel);
		}
		g_buffers->rigidBodies.unmap();
	}
	
	void setAgents(int src_ai, Transform* bodyPose, Vec3* fullVels, Vec3* fullAVels)
	{
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		g_buffers->rigidBodies.map();
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second; i++)
			{
				int bi = i - agentBodies[ai].first;
				Transform gt = agentOffset[ai] * bodyPose[bi];
	
				NvFlexSetRigidPose(&(g_buffers->rigidBodies[i]), (NvFlexRigidPose*)&gt);
				(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[ai].q, fullVels[bi]);
				(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[ai].q, fullAVels[bi]);
			}
		}
		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
	}
	
	virtual void costFunction(int i) = 0;
	
	void mppi(float* controls)
	{
		auto distribution = std::normal_distribution<float>(0.0f, 1.0f); //sigma);
		auto start = std::chrono::high_resolution_clock::now();
	
		NvFlexSetParams(g_solver, &g_params);
	
		//float mydt = g_dt; //*2;
		int mysubsteps = g_numSubsteps;
	
		if (H > 0)
		{  // set action samples to be theta + noise
			for (int i=0; i<H; i++)
				for (int ai = 0; ai < mNumAgents; ai++)
					for (int a=0; a < GetNumControls(); a++)
						noisyActions[ai][i][a] = distribution(generator) * sigma + theta[i][a];
		}
	
		for (int i = 0; i < H; i++)
		{
			{ // set random actuator torques for all agents
				NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
				g_buffers->rigidBodies.map();
				for (int ai = 0; ai < mNumAgents; ai++)
				{
					ApplyTorqueControl(ai, noisyActions[ai][i].data());
				}
				g_buffers->rigidBodies.unmap();
				NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
			}
	
			// STEP
			NvFlexUpdateSolver(g_solver, g_dt, mysubsteps, false);
	
			// "REWARD FUNCTION" HERE
			{
				NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
				NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
				g_buffers->rigidBodies.map();
				g_buffers->rigidJoints.map();
	
				costFunction(i); // calls passed in reward function
	
				g_buffers->rigidBodies.unmap();
				g_buffers->rigidJoints.unmap();
			}
		}
	
		// process costs into weights
		{
			float baseline = costs[0];
			for (int ai = 1; ai < mNumAgents; ai++)
			{ // find min cost
				if (costs[ai] < baseline)
					baseline = costs[ai];
			}

			float sum = 0.0f;
			for (int ai = 0; ai < mNumAgents; ai++)
			{
				weights[ai] = exp(-(costs[ai] - baseline) / lambda);
				sum += weights[ai];
			}

			for (int ai = 0; ai < mNumAgents; ai++)
			{
				weights[ai] = weights[ai] / sum;
			}

			// apply weights to each agent's controls
			for (int i = 0; i < H; i++) // clear theta
				for (int a = 0; a < GetNumControls(); a++)
					theta[i][a] = 0.0f;

			for (int i = 0; i < H; i++)
			{
				for (int ai = 0; ai < mNumAgents; ai++)
				{
					for (int a = 0; a < GetNumControls(); a++)
					{
						theta[i][a] += noisyActions[ai][i][a] * weights[ai];
					}
				}
			}

			for (int i = 1; i < H; i++) // low pass filter theta
			{
				for (int a = 0; a < GetNumControls(); a++)
				{
					theta[i][a] = theta[i - 1][a] + alpha*(theta[i][a] - theta[i - 1][a]); // do some smoothing
					theta[i][a] = Clamp(theta[i][a], -1.f, 1.f);
				}
				printf("min cost: %1.4f\n", baseline);
			}

			for (int a = 0; a < GetNumControls(); a++)
			{
				controls[a] = theta[0][a]; // save initial theta to output controls
			}

			for (int i = 0; i < H - 1; i++)
			{
				for (int a = 0; a < GetNumControls(); a++)
				{
					theta[i][a] = theta[i + 1][a];
				}
			}

			for (int a = 0; a < GetNumControls(); a++)
			{
				theta[H][a] = 0.0f;
			}
		}
	
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		printf("dt: %1.4f opt_t: %1.4f for %1.4fs horizon, %1.4f per step\n",
		g_dt, elapsed.count(), g_dt*H, elapsed.count()/H); // dont print in tight loops
	}

    //struct agentState // TODO
    Transform bodyPose[100];
    Vec3 fullVels[100];
    Vec3 fullAVels[100];

	virtual void PreSimulation()
	{
		if (!g_pause || g_step)
		{
			// save agent states to s0
			//int agent = 0;
			saveAgent(0, bodyPose, fullVels, fullAVels); // maps and unmaps
			//setAgents(0, bodyPose, fullVels, fullAVels); // eventually set everything to start from some s0's

			// MPPI does rollout, returns theta
			float controls[100];
			mppi(controls);

			// reset all agents to s0 state
			setAgents(0, bodyPose, fullVels, fullAVels);

			// MPC: take first controls from theta and set in agents
			{
				//static auto generator = std::default_random_engine();
				//auto distribution = std::normal_distribution<float>(0.0f,sigma);
				//for (int a=0; a<GetNumControls(); a++) controls[a] = 0.0f;
				NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
				g_buffers->rigidBodies.map();
				for (int ai = 0; ai < mNumAgents; ai++)
				{
					ApplyTorqueControl(ai, controls);
				}
				g_buffers->rigidBodies.unmap();
				NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
			}

			step(); // THE ONLY REAL STEP

			appendPlotData(plot_controls, &plot_idx, controls);
		}
	}

	virtual void DoGui()
	{
		float horizon = float(H);
		if (imguiSlider("H", &horizon, 0.f, (float)MAX_H, 1.f)) { H = int(horizon); }
		imguiSlider("Sigma", &sigma, 0.0f, 3.0f, 0.01f);
		imguiSlider("Lambda", &lambda, 0.01f, 1.0f, 0.01f);
		imguiSlider("Control Cost Scl", &u_cost, 0.0f, 9.0f, 0.5f);
		imguiSlider("Control Cost Exp", &u_cost_exp, -6.f, 6.0f, 1.0f);
		imguiSlider("Alpha", &alpha, 0.001f, 1.0f, 0.001f);
	}

	virtual void ApplyTorqueControl(int ai, float* actions)
	{
		for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second; i++)
		{
			g_buffers->rigidBodies[i].force[0] = 0.0f; g_buffers->rigidBodies[i].torque[0] = 0.0f;
			g_buffers->rigidBodies[i].force[1] = 0.0f; g_buffers->rigidBodies[i].torque[1] = 0.0f;
			g_buffers->rigidBodies[i].force[2] = 0.0f; g_buffers->rigidBodies[i].torque[2] = 0.0f;
		}

		Transform gpose;
		Transform tran;
		for (int i = 0; i < (int)ctrls[ai].size(); i++)
		{
			NvFlexRigidJoint& j = initJoints[ ctrls[ai][i].first ];
			NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
			NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
			Transform& pose0 = *((Transform*)&j.pose0);
			NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
			tran = gpose * pose0;

			Vec3 axis;
			// HERP DERP TODO HACK
			if (ctrls[ai][i].second == eNvFlexRigidJointAxisX)
			{
				axis = GetBasisVector0(tran.q);
			}
			else if (ctrls[ai][i].second == eNvFlexRigidJointAxisY)
			{
				axis = GetBasisVector1(tran.q);
			}
			else if (ctrls[ai][i].second == eNvFlexRigidJointAxisZ)
			{
				axis = GetBasisVector2(tran.q);
			}
			else if (ctrls[ai][i].second == eNvFlexRigidJointAxisTwist)
			{
				axis = GetBasisVector0(tran.q);
			}
			else if (ctrls[ai][i].second == eNvFlexRigidJointAxisSwing1)
			{
				axis = GetBasisVector1(tran.q);
			}
			else if (ctrls[ai][i].second == eNvFlexRigidJointAxisSwing2)
			{
				axis = GetBasisVector2(tran.q);
			}
			else 
			{
				printf("Invalid axis, probably bad code migration?\n");
				exit(0);
			}

			Vec3 torque = axis * motorPower[ai][i] * actions[i] ;//* powerScale;
			if (ctrls[ai][i].second <= eNvFlexRigidJointAxisZ)
			{
				//a0.force[0] += torque.x; a1.force[0] -= torque.x;
				//a0.force[1] += torque.y; a1.force[1] -= torque.y;
				//a0.force[2] += torque.z; a1.force[2] -= torque.z;
				a0.torque[0] += torque.x; a1.torque[0] -= torque.x;
				a0.torque[1] += torque.y; a1.torque[1] -= torque.y;
				a0.torque[2] += torque.z; a1.torque[2] -= torque.z;
			}
			else
			{
				a0.torque[0] += torque.x; a1.torque[0] -= torque.x;
				a0.torque[1] += torque.y; a1.torque[1] -= torque.y;
				a0.torque[2] += torque.z; a1.torque[2] -= torque.z;
			}
		}
	}

    void GetBodyPose(int ai, int body, Transform* trans)
	{
		Transform& inv = agentOffsetInv[ai];
		pair<int, int> p = agentBodies[ai];

		Transform pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first+body], (NvFlexRigidPose*)&pose);
		trans[0] = inv * pose;
    }

    void GetBodyVels(int ai, int body, Vec3* vels)
	{
		pair<int, int> p = agentBodies[ai];
		Vec3 vel = g_buffers->rigidBodies[p.first+body].linearVel;
		Transform& inv = agentOffsetInv[ai];
		vels[0] = Rotate(inv.q, vel);
    }

    int GetNumControls()
	{
		return mNumActions;
    }

    // TODO change these to take arrays instead of vectors
    void GetAngles(int a, vector<float>& angles)
	{
		vector<float> tmp1, tmp2;
		GetAngles(a, angles, tmp1, tmp2);
	}

	void GetAngles(int a, vector<float>& angles, vector<float>& lows, vector<float>& highs)
	{
		float prevTwist = 0.f, prevSwing1 = 0.f, prevSwing2 = 0.f;
		Vec3 prevPos;
		int prevIdx = -1;

		for (int i = 0; i < (int)ctrls[a].size(); i++)
		{
			int qq = i;
			float pos = 0.f;
			NvFlexRigidJoint& joint = initJoints[ctrls[a][qq].first];
			if (ctrls[a][qq].first != prevIdx)
			{
				NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
				NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

				Transform body0Pose;
				Transform body1Pose;
				NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
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

			int idx = ctrls[a][qq].second;
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
			// JOINT UPPER AND LOWER LIMITS BROKEN!!!!!!!!!!!
			if (highs.size())
				highs[i] = joint.upperLimits[3 + idx];
			if (lows.size())
				lows[i] = joint.lowerLimits[3 + idx];
		}
	}

    virtual void DoStats()
	{
//		plotData(plot_controls, plot_idx, 1.5f, g_screenWidth - 20.0f, 80.0f, "Franka Controls, 1 second");
    }

	void plotData(vector<vector<float>> p, int p_idx, float maxV, float x, float y, string title)
	{
		int numSamples = p.size();

		int start = 0;
		int end = p.size();

		// convert from position changes to forces
		float units = -1.0f / Sqr(g_dt/g_numSubsteps);

		float height = 50.0f;

		float dx = 5.0f;
		float sy = height/maxV;

		float lineHeight = 10.0f;

		float rectMargin = 10.0f;
		float rectWidth = dx * float(numSamples) + rectMargin * 4.0f;

		x = x - rectWidth;

		DrawRect(x, y - height - rectMargin, rectWidth, 2.0f * height + rectMargin * 3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));
		x += rectMargin * 3.0f;
		DrawImguiString(int(x) + int(dx) * numSamples, int(y) + 55, Vec3(1.0f), IMGUI_ALIGN_RIGHT, title.c_str());

		DrawLine(x, y, x + float(numSamples) * dx, y, 1.0f, Vec3(1.0f));
		DrawLine(x, y -50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

		float margin = 5.0f;
		DrawImguiString(int(x - margin), int(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
		DrawImguiString(int(x - margin), int(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxV);
		DrawImguiString(int(x - margin), int(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxV);

		for (int i = start; i < end - 1; ++i)
		{
			int idx = (i + p_idx) % (end - 1);
			for (int a = 0; a < GetNumControls(); a++)
			{
				float fl0 = Clamp(p[idx][a],   -maxV, maxV) * sy;
				float fl1 = Clamp(p[idx+1][a], -maxV, maxV) * sy;

				DrawLine(x, y + fl0, x + dx, y + fl1, 1.0f, Vec3(1.0f, 0.0f, 0.0f));
			}
			x += dx;
		}
	}

    //// PLOTTING CODE
	void appendPlotData(vector<vector<float>> &p, int* p_idx, float* d)
	{
		int idx = *p_idx;
		for (int a=0; a<p[0].size(); a++) p[idx][a] = d[a];

		idx += 1;
		if (idx >= p.size()) idx = 0;

		*p_idx = idx;
	}
};


// sub task should: LoadEnv, costfunction, initial MPPI params, DoStats
class MPC_Franka : public Controller
{
public:
    URDFImporter* urdf; // for franka
    string modelfile;
    vector<string> motors = { "panda_joint1", "panda_joint2",  "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2"};
    vector<float> powers;
    vector<float> velLimits;

    float d_weight = 10.0f;

	MPC_Franka()
	{
		mNumAgents = 100; // K
		numPerRow = 10;
		spacing = 5.f;

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 4;
		g_params.numInnerIterations = 15;
		g_params.relaxationFactor = 0.75f;

		H = 10; // needs this here to do gui things
		MAX_H = 100;
		sigma = 0.3f;
		lambda = 0.25f;
		alpha = 0.8125f;

		u_cost = 1.0f;
		u_cost_exp = -3.0f;
		targetheight = 1.2f;

		g_params.numPostCollisionIterations = 0;
		g_params.shapeCollisionMargin = 0.005f;
		g_params.collisionDistance = 0.005f;

		modelfile = "franka_description/robots/franka_panda.urdf";
	}

    virtual void EnvCommonSetup()
	{
		g_solverDesc.maxRigidBodyContacts = mNumAgents * 256 * 4;
		rigidContacts.map();
		rigidContacts.resize(g_solverDesc.maxRigidBodyContacts);
		rigidContacts.unmap();

		urdf = new URDFImporter("../../data", modelfile.c_str(), true);
		powers.clear();
		velLimits.clear();
		mNumActions = motors.size();
		printf("\n\n%d actions\n", mNumActions);
		int motoridx = 0;
		for (int a = 0; a < urdf->joints.size(); ++a)
		{
			if (urdf->joints[a]->type != URDFJoint::FIXED)
			{
				float power = urdf->joints[a]->effort;
				powers.push_back(power);
				float vellimit = urdf->joints[a]->velocity;
				velLimits.push_back(vellimit);
				cout << motors[motoridx++] << " power = " << power << " vellimit = " << vellimit << endl;
			}
		}
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower) override
	{
		int startJoint = g_buffers->rigidJoints.size();
		int startShape = g_buffers->rigidShapes.size();
		int b_bodies   = g_buffers->rigidBodies.size();

		float density = 500.0f; 
		float bodyDamping = 1.0f;
		float armature = 0.1f; // what is this value

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, true, density, 0.0f, 25.f, armature, 10.f, bodyDamping, false, 1e10f);

		g_buffers->rigidBodies[b_bodies].invMass = 0.0f; // fixed base
		(Matrix33&)g_buffers->rigidBodies[b_bodies].invInertia = Matrix33();

		int endJoint = g_buffers->rigidJoints.size();
		int endShape = g_buffers->rigidShapes.size();
		for (int i=startShape; i<endShape; i++)
		{
			g_buffers->rigidShapes[i].filter = 0x1;
		}

		robotJoints.push_back({ startJoint, endJoint });
		robotShapes.push_back({ startShape, endShape });

		for (size_t i = 0; i < motors.size(); i++)
		{
			auto jointType = urdf->joints[urdf->urdfJointNameMap[motors[i]]]->type;
			if (jointType == URDFJoint::Type::REVOLUTE)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[motors[i]], eNvFlexRigidJointAxisTwist));
			}
			else if (jointType == URDFJoint::Type::PRISMATIC)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[motors[i]], eNvFlexRigidJointAxisX));
			}
			else
			{
				cout << "Error! Motor can't be a fixed joint" << endl;
			}
			mpower.push_back(powers[i]);
		}

		for (int i = startShape; i < endShape; i++)
		{
			g_buffers->rigidShapes[i].filter = 0x1;
		}

		//// DAMPING and COMPLIANCE
		/*
		for (auto rj : urdf->joints)
		{
		NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[rj->name]];
		if (rj->type == URDFJoint::REVOLUTE)
		{
			joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-3f;
			joint.damping[eNvFlexRigidJointAxisTwist] = 1.0f;
		}
		}
		*/
    }

    virtual void AddChildEnvBodies(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
    }

    void CommonCostFunction(int ai, int i)
	{
      // Everything should have control cost
      float control_cost = 0.0f;
      for (int a=0; a<GetNumControls(); a++)
	  {
        //motorPower[ai][a] weights?
        control_cost += (noisyActions[ai][i][a] * noisyActions[ai][i][a]);
      }
      costs[ai] += control_cost*(u_cost*pow(10.0f,u_cost_exp));
    }

    virtual void costFunction(int i)
	{
      Transform goal;
      goal.p.x = 0.5f;
      goal.p.y = 0.0f;
      goal.p.z = 0.7f; //1.2f; //0.9f;
      vector<float> angles;
      angles.resize(GetNumControls());
      for (int ai = 0; ai < mNumAgents; ai++)
	  {
        Transform pose;
        GetBodyPose(ai, 8, &pose);

        costs[ai] = 0.0f;
        costs[ai] += d_weight*sqeuclidean(goal, pose);
        //costs[ai] += d_weight*euclidean(goal, pose);

        // velocity / momentum costs
        GetAngles(ai, angles); // get agent's angles
        for (int a=0; a<GetNumControls(); a++)
		{
          //jointVelocities[ai][a] = (angles[a]-jointAngles[ai][a])/g_dt;
          jointVelocities[ai][a] = (angles[a]-jointAngles[ai][a]); // rads per sec
          jointAngles[ai][a] = angles[a];

          // big margin on vel limit
          if (fabs(jointVelocities[ai][a]*3.0f) > velLimits[a])
            costs[ai] += 10000.0; // Penalize heavily any faults

          // Stay in the middle cost, but should more likely be stay
          // away from joint limits TODO
          //costs[ai] += 10.0f*fabs(fabs(jointUpper[a]-angles[a]) - fabs(jointLower[a]-angles[a]));
        }

        CommonCostFunction(ai, i);

        /// LETS PRINTF DEBUG!
        pair<int, int> pb = agentBodies[ai];
        pair<int, int> pj = robotJoints[ai];
        if (i == 0 && ai == 0) {
          printf("%d \n", GetNumControls());
          printf("ang: ");
          for (int a=0; a<GetNumControls(); a++){
            printf("%1.4f ", jointAngles[ai][a]);
          }
          printf("\nlim: ");
          for (int a=0; a<GetNumControls(); a++) {
            printf("%1.4f ", velLimits[a]);
          }
          printf("\ncur: ");
          for (int a=0; a<GetNumControls(); a++) {
            printf("%1.4f ", jointVelocities[ai][a]);
          }
          printf("\n");
        }
        /*
           int bidx = pb.first+3;
           int jidx = pj.first+3;
           int nbody = pb.second-pb.first;
           printf("mass: ");
           for (int b=0; b<nbody; b++) {
           printf("%1.4f ", g_buffers->rigidBodies[b].mass);
           }
           printf("\n");
           printf("%1.4f %1.4f %1.4f\n%1.4f %1.4f %1.4f\n",
           g_buffers->rigidJoints[jidx].damping[0],
           g_buffers->rigidJoints[jidx].damping[1],
           g_buffers->rigidJoints[jidx].damping[2],
           g_buffers->rigidJoints[jidx].damping[3],
           g_buffers->rigidJoints[jidx].damping[4],
           g_buffers->rigidJoints[jidx].damping[5]);
           printf("%1.4f\n%1.4f %1.4f %1.4f\n%1.4f %1.4f %1.4f\n%1.4f %1.4f %1.4f\n", 
           g_buffers->rigidBodies[bidx].mass,
           g_buffers->rigidBodies[bidx].inertia[0],
           g_buffers->rigidBodies[bidx].inertia[1],
           g_buffers->rigidBodies[bidx].inertia[2],
           g_buffers->rigidBodies[bidx].inertia[3],
           g_buffers->rigidBodies[bidx].inertia[4],
           g_buffers->rigidBodies[bidx].inertia[5],
           g_buffers->rigidBodies[bidx].inertia[6],
           g_buffers->rigidBodies[bidx].inertia[7],
           g_buffers->rigidBodies[bidx].inertia[8]);
           */
        //printf("body 8 x: %1.4f\n", pose.p.z);
      }
    }

    virtual void DoGui()
	{
      Controller::DoGui();
      imguiSlider("D weight", &d_weight, 1.0f, 100.0f, 1.0f);
    }
};

class MPC_FrankaCabinet : public MPC_Franka{
  public:
    vector<int> targetRenderMaterials;
    vector<int> targetSphere;
    Vec3 redColor;
    Vec3 greenColor;
    vector<Vec3> targetPoses;
    float radius;

    URDFImporter* cabinet_urdf; // for franka
    vector<int> topDrawerHandleBodyIds;
    vector<int> topDrawerHandleShapeIds;
    vector<int> topDrawerJoints;

    vector<Vec3> drawerPose;
    vector<int> cabinetStartShape;
    vector<int> cabinetEndShape;
    int cabinetMaterial;

    MPC_FrankaCabinet()
	{
      mNumAgents = 64;
	  numPerRow = 8;

      targetSphere.resize(mNumAgents, -1);
      radius = 0.04f;
      redColor = Vec3(1.0f, 0.04f, 0.07f);
      greenColor = Vec3(0.06f, 0.92f, 0.13f);
      targetRenderMaterials.resize(mNumAgents);

      topDrawerHandleBodyIds.resize(mNumAgents, 0);
      topDrawerHandleShapeIds.resize(mNumAgents, 0);
      topDrawerJoints.resize(mNumAgents, 0);
      cabinetStartShape.resize(mNumAgents, 0);
      cabinetEndShape.resize(mNumAgents, 0);
      drawerPose.resize(mNumAgents, Vec3(0.0f, 0.0f, 0.0f));

      bool useObjForCollision = true;
      float dilation = 0.005f;
      float thickness = 0.005f;
      bool replaceCylinderWithCapsule=true;
      int slicesPerCylinder = 20;
      bool useSphereIfNoCollision = true;

      cabinet_urdf = new URDFImporter("../../data/", "sektion_cabinet_model/urdf/sektion_cabinet.urdf",
          useObjForCollision,
          dilation,
          thickness,
          replaceCylinderWithCapsule,
          slicesPerCylinder,
          useSphereIfNoCollision);

      cabinetMaterial = AddRenderMaterial(Vec3(0.4f, 0.5f, 0.8f));
    }

    virtual void AddChildEnvBodies(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{

      Transform cabinet_transform(Vec3(0.0f, 0.03f,0.23f) + gt.p,
          QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * QuatFromAxisAngle(
            Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));

      cabinetStartShape[ai] = g_buffers->rigidShapes.size();
      int startBody = g_buffers->rigidBodies.size();
      cabinet_urdf->AddPhysicsEntities(cabinet_transform, cabinetMaterial, false, 1000.0f, 0.0f, 1e1f, 0.1f, 20.7f, 7.0f, false);
      cabinetEndShape[ai] = g_buffers->rigidShapes.size();
      int endBody = g_buffers->rigidBodies.size();

      for (int i = cabinetStartShape[ai]; i < cabinetEndShape[ai]; i++)
	  {
        if (i == cabinet_urdf->rigidShapeNameMap["drawer_handle_top"].first)
		{
          topDrawerHandleShapeIds[ai] = i;
        }
        else if (i == cabinet_urdf->rigidShapeNameMap["drawer_top"].first)
		{
          //g_buffers->rigidShapes[i].thickness += 0.015f;
        }

        g_buffers->rigidShapes[i].filter = 0x2;
      }

      for (int i = 0; i < (int)cabinet_urdf->joints.size(); i++)
	  {
        URDFJoint* j = cabinet_urdf->joints[i];
        NvFlexRigidJoint& joint = g_buffers->rigidJoints[cabinet_urdf->jointNameMap[j->name]];
        if (j->type == URDFJoint::REVOLUTE)
		{
          joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeLimit;	// 10^6 N/m
          joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-8f;	// 10^6 N/m
          joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+3f;	// 5*10^5 N/m/s
          joint.targets[eNvFlexRigidJointAxisTwist] = 0.0f;
        }
        else if (j->type == URDFJoint::PRISMATIC)
		{
          joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeLimit;
          joint.targets[eNvFlexRigidJointAxisX] = 0.0f;
          joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f;
          joint.damping[eNvFlexRigidJointAxisX] = 3e+1f;

          if (j->name == "drawer_top_joint") {
            topDrawerJoints[ai] = cabinet_urdf->jointNameMap[j->name];
          }
        }
      }
      topDrawerHandleBodyIds[ai] = cabinet_urdf->rigidNameMap["drawer_handle_top"];

      // fix the cabinet
      g_buffers->rigidBodies[startBody].invMass = 0.0f;
      (Matrix33&)g_buffers->rigidBodies[startBody].invInertia = Matrix33();

      Transform pose;
      NvFlexGetRigidPose(&g_buffers->rigidBodies[topDrawerHandleBodyIds[ai]],
          (NvFlexRigidPose*)&pose);
      drawerPose[ai] = pose.p - gt.p;
    }

    virtual void costFunction(int i)
	{
      Transform handle;
      Transform target;
      target.p.x = 0.5;
      target.p.y = 0.0f;
      target.p.z = 0.7f;
      Vec3 handleOffset(-0.25, 0.0, 0.0);
      static vector<float> angles;
      angles.resize(GetNumControls());
      for (int ai = 0; ai < mNumAgents; ai++)
	  {
        Transform pose;
        GetBodyPose(ai, 9, &pose);
        GetBodyPose(ai, topDrawerHandleBodyIds[ai]-agentBodies[ai].first, &handle); // Cabinet
        handle.p = handle.p + handleOffset;
        //handle = drawerPose[ai];

        costs[ai] = 0.0f;
        //costs[ai] += d_weight*sqeuclidean(handle, pose);
        float hand = euclidean(handle, pose);
        float open = euclidean(handle, target);
        if (hand > 0.05f)
		{
          costs[ai] += d_weight * hand;
		}
        else
		{
          costs[ai] -= 1.0f; // reward bonus
          costs[ai] += 10.f * d_weight * open;
        }

        // velocity / momentum costs
        GetAngles(ai, angles); // get agent's angles
        for (int a=0; a<GetNumControls(); a++)
		{
          // calc velocities, then update angles to current angles
          //jointVelocities[ai][a] = (angles[a]-jointAngles[ai][a])/g_dt;
          jointVelocities[ai][a] = (angles[a]-jointAngles[ai][a]); // rads per sec
          jointAngles[ai][a] = angles[a];

          // big margin on vel limit
          if (fabs(jointVelocities[ai][a]*10.0f) > velLimits[a])
            costs[ai] += 10000.0; // Penalize heavily any faults

          // Stay in the middle cost, but should more likely be stay
          // away from joint limits TODO
          //costs[ai] += 10.0f * fabs(fabs(jointUpper[a]-angles[a]) - fabs(jointLower[a] - angles[a]));
        }

        CommonCostFunction(ai, i);

        /// LETS PRINTF DEBUG!
        if (i == 0 && ai == 0)
		{
          printf("handle: %1.4f %1.4f %1.4f\n", handle.p.x, handle.p.y, handle.p.z);
        }
      }
    }
};

class MPC_FrankaAllegro : public MPC_Franka
{
  public:
    URDFImporter* allegro_urdf; // for franka
    int numFingers;
    int numJointsPerFinger;

	MPC_FrankaAllegro()
	{
		mNumAgents = 16;

		numFingers = 4;
		numJointsPerFinger = 4;

		g_params.numIterations = 5;
		g_params.numInnerIterations = 40;

		bool useObjForCollision = true;
		float dilation = 0.005f;
		float thickness = 0.005f;
		bool replaceCylinderWithCapsule = false;
		int slicesPerCylinder = 20;
		bool useSphereIfNoCollision = true;

		allegro_urdf = new URDFImporter("../../data", "allegro_hand_description/allegro_hand_description_right.urdf");
	}

    virtual void AddChildEnvBodies(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		const float density         = 500.0f;
		const float jointCompliance = 1.e-6f;
		const float jointDamping    = 1.e+1f;
		const float bodyArmature    = 1.e-5f;
		const float bodyDamping     = 5.0f;
		const int hiddenMaterial    = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

		Transform wrist;
		int end = g_buffers->rigidBodies.size();
		NvFlexGetRigidPose(&g_buffers->rigidBodies[end-1], (NvFlexRigidPose*)&wrist);
		// TODO approximate
		wrist.p.x += 0.0f;
		wrist.p.y += -0.09f;
		wrist.p.z += 0.015f;

		int startShape = g_buffers->rigidShapes.size();
		allegro_urdf->AddPhysicsEntities(wrist, hiddenMaterial, true, false, density, jointCompliance, jointDamping, bodyArmature, bodyDamping, kPi * 8.0f, false);
		int endShape = g_buffers->rigidShapes.size();

		NvFlexRigidJoint joint;
		NvFlexRigidPose wristpose;
		memcpy(&wristpose, &wrist, sizeof(NvFlexRigidPose));
		NvFlexMakeFixedJoint(&joint, end-1, end, wristpose, wristpose);
		g_buffers->rigidJoints.push_back(joint);

		for (int i = startShape; i < endShape; i++)
		{
			g_buffers->rigidShapes[i].filter = 0x0;
			g_buffers->rigidShapes[i].group = 0;
		}

		// Configure finger joints
		ostringstream jointName;
		for (int i = 0; i < numFingers; i++)
		{
			for (int j = 0; j < numJointsPerFinger; j++)
			{
				jointName.str("");
				jointName << "joint_" << i * numFingers + j << ".0";
				auto jointID = allegro_urdf->urdfJointNameMap[jointName.str()];
				auto jointType = allegro_urdf->joints[jointID]->type;

				ctrl.push_back(make_pair(jointID, eNvFlexRigidJointAxisTwist));
				float power = allegro_urdf->joints[jointID]->effort;
				power = 12.0f;
				mpower.push_back(power);

				float vellimit = allegro_urdf->joints[jointID]->velocity;
				vellimit = 2.175f;
				velLimits.push_back(vellimit);
				if (ai == 0)
					printf("%s: %d power = %1.4f vellimit = %1.4f\n", jointName.str().c_str(), jointID, power, vellimit);
			}
		}

		for (int i = 0; i < (int)allegro_urdf->joints.size(); i++)
		{
			URDFJoint* j = allegro_urdf->joints[i];
			NvFlexRigidJoint& joint = g_buffers->rigidJoints[allegro_urdf->jointNameMap[j->name]];
			if (j->type == URDFJoint::REVOLUTE)
			{
				joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeLimit;	// 10^6 N/m
				joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-8f;	// 10^6 N/m
				joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+3f;	// 5*10^5 N/m/s
				joint.targets[eNvFlexRigidJointAxisTwist] = 0.0f;
			}
			else if (j->type == URDFJoint::PRISMATIC)
			{
				joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeLimit;
				joint.targets[eNvFlexRigidJointAxisX] = 0.0f;
				joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f;
				joint.damping[eNvFlexRigidJointAxisX] = 3e+1f;
			}
		}
	}

	virtual void costFunction(int i)
	{
		Transform target;
		target.p.x = 0.5;
		target.p.y = 0.0f;
		target.p.z = 0.7f;
		Vec3 handleOffset(-0.25, 0.0, 0.0);

		static vector<float> angles;
		angles.resize(GetNumControls());
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			Transform pose;
			GetBodyPose(ai, 9, &pose);
			//handle = drawerPose[ai];

			costs[ai] = 0.0f;
			costs[ai] += d_weight*sqeuclidean(target, pose);

			// velocity / momentum costs
			GetAngles(ai, angles); // get agent's angles
			for (int a=0; a<GetNumControls(); a++)
			{
				// calc velocities, then update angles to current angles
				//jointVelocities[ai][a] = (angles[a]-jointAngles[ai][a])/g_dt;
				jointVelocities[ai][a] = (angles[a]-jointAngles[ai][a]); // rads per sec
				jointAngles[ai][a] = angles[a];

				// big margin on vel limit
				if (fabs(jointVelocities[ai][a] * 10.0f) > velLimits[a])
					costs[ai] += 10000.0; // Penalize heavily any faults

				// Stay in the middle cost, but should more likely be stay
				// away from joint limits TODO
				//costs[ai] += 10.0f*fabs(fabs(jointUpper[a]-angles[a]) - fabs(jointLower[a]-angles[a]));
			}

			CommonCostFunction(ai, i);
		}
	}
};

/*
// HUMANOID STANDUP
void costFunction(int i) {
float xspeed = 1.0f;
for (int ai = 0; ai < mNumAgents; ai++) {
Transform root;
Vec3 vel;
GetBodyPose(ai, 0, &root);
GetBodyVels(ai, 0, &vel);

//float yaxis = root.com[0];
float height = root.p.z;
if (i == 0 && ai == 0)
printf("height: %f\n", root.p.z);

//float xaxis = root.com[2];
costs[ai] = 0.0f;
//costs[ai] += rootbody.com[0] + rootbody.com[1]; // stay close to center
//if (height < targetheight) {
//  float diff = targetheight-height;
//  costs[ai] += 3.0f*(diff);//*diff);
//}
//else {
costs[ai] += fabs(xspeed - vel.x); // first linvel is x speed?
//}
}
}
*/
