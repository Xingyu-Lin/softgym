#pragma once
#include <iostream>
#include <vector>
#include "rlbase.h"
#include "../urdf.h"
#include <stdexcept>
#include <cmath>

// Collision settings: robot, table have group 0 and filter 0
class RLFetchBase : public FlexGymBase2
{
public:
	URDFImporter* urdf;
	bool sampleInitStates;
	float tableHeight, tableLength, tableWidth;
	Vec3 tableOrigin;
	vector<int> fingersLeft;
	vector<int> fingersRight;
	vector<int> effector;
	vector<int> effectorJoints;
	vector<NvFlexRigidJoint> effectorJoints0;

	float speedTra, speedRot, speedGrip; // in meters or radians per timestep
	vector<Vec2> limitsTra;
	Vec2 limitsRot, limitsGrip;
	bool doDeltaPlanarControl;
	bool doGripperControl;
	bool doWristRollControl;
	bool doStats;
   	bool sparseRewards;
    bool renderTarget;
    bool relativeTarget;
    bool controlOnPlane;
	int hiddenMaterial;
    string rewardScheme;
    string viewName;
    string urdfName;
    
	vector<pair<float, float>> effectorLimits;
	vector<Vec3> robotPoses;

	float fingerWidthMin, fingerWidthMax;
	vector<float> fingerWidths;
	vector<float> rolls, pitches, yaws;
	vector<Vec3> targetEffectorTranslations;
	Vec3 initEffectorTranslation;

	int maxContactsPerAgent;

	vector<string> motors;
	vector<float> powers;

	vector<float> forceLeft;
	vector<float> forceRight;

	RLFetchBase()
	{
		mNumAgents = 100;
		numPerRow = 10;
		mMaxEpisodeLength = 100;

        mNumObservations = 0;
        mNumActions = 0;
        mNumPyToC = 0;
        
		controlType = eInverseDynamics;
		tableHeight = 0.39f;
		tableLength = 0.55f;
		tableWidth = 0.27f;
		tableOrigin = Vec3(0.0f, 0.0f, 0.6f);
		fingerWidthMin = 0.002f;
		fingerWidthMax = 0.05;

		spacing = 3.5f;

		speedTra = 5.f * g_dt;
		speedRot = 2 * kPi * g_dt;
		speedGrip = 0.1f * g_dt;
		// This is a useful parameter to change to decrease the amount of exploration required by the robot
		limitsTra = {
			Vec2(-0.5f, 0.5f),
			Vec2(tableHeight, 0.8f),
			Vec2(0.f, 1.0f)
		};
		limitsRot = Vec2(-kPi, kPi);
		limitsGrip = Vec2(0.002f, 0.05f);
		initEffectorTranslation = Vec3(0.f, 0.6f, 0.5f);

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(8.6f, 0.9f, 3.5f);

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 6;
		g_params.numInnerIterations = 20;
		g_params.warmStart = 0.f;

		g_params.dynamicFriction = 1.25f;	// yes, this is a physically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;
		g_params.relaxationFactor = 0.75f;
		g_params.shapeCollisionMargin = 0.001f;

		mDoLearning = g_doLearning;
		doStats = true;
		g_pause = true;
		g_drawPoints = false;
		g_drawCloth = true;

		effectorLimits = {
			{ -.5f, .5f },
			{ 0.4f, 1.5f },
			{ 0.f, 1.2f }
		};
	}

    virtual void ParseJson() 
    {
		ParseJsonParams(g_sceneJson);

		sampleInitStates = GetJsonVal(g_sceneJson, "SampleInitStates", true);

		g_sceneLower = Vec3(-0.5f);
		g_sceneUpper = Vec3(0.4f, 0.8f, 0.4f);

		doGripperControl =  GetJsonVal(g_sceneJson, "DoGripperControl", false);
		doDeltaPlanarControl =  GetJsonVal(g_sceneJson, "DoDeltaPlanarControl", false);
		doWristRollControl =  GetJsonVal(g_sceneJson, "DoWristRollControl", false);
		sparseRewards =  GetJsonVal(g_sceneJson, "SparseRewards", false);
		renderTarget = GetJsonVal(g_sceneJson, "RenderTarget", true);
		relativeTarget = GetJsonVal(g_sceneJson, "RelativeTarget", true);
        controlOnPlane = GetJsonVal(g_sceneJson, "ControlOnPlane", false);
        rewardScheme = GetJsonVal(g_sceneJson, "RewardScheme", string("default"));
        viewName = GetJsonVal(g_sceneJson, "ViewName", string("default"));
        urdfName = GetJsonVal(g_sceneJson, "urdfName", string("fetch_description/robots/fetch.urdf"));
    }
    
    virtual void PrepareScene() override
    {
        ParseJson();
        
        mNumActions = computeNumActions();
        mNumObservations = computeNumObservations();
        printf("actions and observations are %d and %d\n", mNumActions, mNumObservations);
        if(mNumActions > 100000 || mNumObservations > 100000){
            throw std::logic_error("???");
        }

		LoadEnv();
		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());

		if (mDoLearning)
		{
			init();
		}
	}

    virtual int computeNumActions()
	{
        int a;
        
        if (!doDeltaPlanarControl) a = 7;
        else if(doDeltaPlanarControl && doWristRollControl) a = 5;
    	else a = 4;

        if (!doGripperControl) a -= 1;

        return a;
    }

    virtual int computeNumObservations()
	{
        int o;
        
        if (!doDeltaPlanarControl) o = 7; // xyz + gripper
        else if(doDeltaPlanarControl && doWristRollControl) o = 5; // +1 for r
        else o = 4; // +3 for rpy

        if (!doGripperControl) o -= 1;

        return o;
    }

	// To be overwritten by child env
	virtual void LoadChildEnv() {}

	void LoadEnv() override
	{
		LoadChildEnv();

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		fingerWidths.resize(mNumAgents, 0.03f);
		rolls.resize(mNumAgents, 0.f);
		pitches.resize(mNumAgents, 0.f);
		if (mDoLearning)
		{
			yaws.resize(mNumAgents, -kPi/2);
		}
		else
		{
			yaws.resize(mNumAgents, -90.f);
		}

		effectorJoints0.clear();
		effector.resize(mNumAgents);
		fingersLeft.resize(mNumAgents);
		fingersRight.resize(mNumAgents);
		effectorJoints.resize(mNumAgents);
		robotPoses.resize(mNumAgents);
		effectorJoints0.resize(mNumAgents);
		targetEffectorTranslations.resize(mNumAgents);

		motors = {
			//	"r_wheel_joint",
			//	"l_wheel_joint",
			//	"torso_lift_joint",
			//	"head_pan_joint",
			//	"head_tilt_joint",
			"shoulder_pan_joint",
			"shoulder_lift_joint",
			"upperarm_roll_joint",
			"elbow_flex_joint",
			"forearm_roll_joint",
			"wrist_flex_joint",
			"wrist_roll_joint",
			"r_gripper_finger_joint",
			"l_gripper_finger_joint",
			//	"bellows_joint"
		};

		int rbIt = 0;
		int jointsIt = 0;

		// hide collision shapes
		hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
        urdf = new URDFImporter("../../data/fetch_ros-indigo-devel", urdfName);

		powers.clear();

		for (int ai = 0; ai < mNumAgents; ++ai)
		{
			Vec3 robotPos = Vec3((ai % numPerRow) * spacing, 0.05f, (ai / numPerRow) * spacing - 0.25f);
			Transform gt(robotPos, QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));
			robotPoses[ai] = robotPos;

			int begin = g_buffers->rigidBodies.size();
			AddAgentBodiesJointsCtlsPowers(ai, gt, ctrls[ai], motorPower[ai], rbIt, jointsIt);
			AddChildEnvBodies(ai, gt);
			int end = g_buffers->rigidBodies.size();
			agentBodies.push_back(make_pair(begin, end));

			rbIt = g_buffers->rigidBodies.size();
			jointsIt = g_buffers->rigidJoints.size();

			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
		}

		forceLeft.resize(0);
		forceRight.resize(0);

		maxPower = *max_element(std::begin(motorPower[0]), std::end(motorPower[0]));

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());

		initBodies.resize(g_buffers->rigidBodies.size());
		memcpy(&initBodies[0], &g_buffers->rigidBodies[0], sizeof(NvFlexRigidBody) * g_buffers->rigidBodies.size());
	}

	// To be overwritten by child envs
	virtual void AddChildEnvBodies(int ai, Transform gt) {}

	void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpowers) override {}
	void AddAgentBodiesJointsCtlsPowers(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower,
												int rbIt, int jointsIt)
	{
		int startShape = g_buffers->rigidShapes.size();
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 10000.0f, 0.0f, 10.f, 0.01f, 20.f, 7.f, false);
		int endShape = g_buffers->rigidShapes.size();
		for (int i = startShape; i < endShape; i++)
		{
			//g_buffers->rigidShapes[i].filter = 0x0;
			//g_buffers->rigidShapes[i].group = 0;
            g_buffers->rigidShapes[i].thickness = 0.001f;
		}

		for (auto m : motors)
		{	
			auto jointType = urdf->joints[urdf->urdfJointNameMap[m]]->type;
			if (jointType == URDFJoint::Type::CONTINUOUS || jointType == URDFJoint::Type::REVOLUTE)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[m], eNvFlexRigidJointAxisTwist)); // CHANGE (romurthy): Changed from 3 to 0, because 3 doesn't mean anything
			}
			else if (jointType == URDFJoint::Type::PRISMATIC)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[m], eNvFlexRigidJointAxisX)); // Is it correct?
			}
			else
			{
				cout << "Error! Motor can't be a fixed joint" << endl;
			}
					
			float effort = urdf->joints[urdf->urdfJointNameMap[m]]->effort;
			// cout << m << " power = " << effort << endl;
			mpower.push_back(effort);
		}

		for (int i = 0; i < (int)urdf->joints.size(); i++)
		{
			URDFJoint* j = urdf->joints[i];
			NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[j->name]];
			if (j->type == URDFJoint::REVOLUTE)
			{
				joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-8f;	// 10^6 N/m
				joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+3f;	// 5*10^5 N/m/s
				joint.motorLimit[eNvFlexRigidJointAxisTwist] = 100.f;			
			}
			else if (j->type == URDFJoint::PRISMATIC)
			{
                joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
                joint.targets[eNvFlexRigidJointAxisX] = joint.lowerLimits[eNvFlexRigidJointAxisX];
                joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f;
                joint.damping[eNvFlexRigidJointAxisX] = 0.0f;				
			}
		}

		effector[ai] = urdf->rigidNameMap["l_gripper_finger_link"];

		// fix base in place, todo: add a kinematic body flag?
		g_buffers->rigidBodies[rbIt].invMass = 0.0f;
		(Matrix33&)g_buffers->rigidBodies[rbIt].invInertia = Matrix33();

		fingersLeft[ai] = urdf->jointNameMap["l_gripper_finger_joint"];
		fingersRight[ai] = urdf->jointNameMap["r_gripper_finger_joint"];

        NvFlexRigidJoint* fingers[2] = { &g_buffers->rigidJoints[fingersLeft[ai]], &g_buffers->rigidJoints[fingersRight[ai]] };
        for (int i=0; i < 2; ++i)
        {
            fingers[i]->modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
            fingers[i]->targets[eNvFlexRigidJointAxisX] = 0.02f;
            fingers[i]->compliance[eNvFlexRigidJointAxisX] = 1.e-6f;
            fingers[i]->damping[eNvFlexRigidJointAxisX] = 0.0f;
			fingers[i]->motorLimit[eNvFlexRigidJointAxisX] = 40.0f;
        }    

		NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["l_gripper_finger_joint"]];

		if (!mDoLearning || controlType == Control::eInverseDynamics || controlType == Control::eVelocity)
		{
			// set up end effector targets
			NvFlexMakeFixedJoint(&effectorJoints0[ai], -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(Randf(0.0f, 0.25f), Randf(0.6f, 0.7f), Randf(0.45f, 0.5f)) + robotPoses[ai],
				QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), -kPi * 0.5f)), NvFlexMakeRigidPose(0, 0));
			for (int i = 0; i < 6; ++i)
			{
				effectorJoints0[ai].compliance[i] = 1.e-4f;
				effectorJoints0[ai].damping[i] = 1.e+3f;
			}

			effectorJoints[ai] = g_buffers->rigidJoints.size();

			targetEffectorTranslations[ai] = initEffectorTranslation;
			Vec3 effectorTranslation = targetEffectorTranslations[ai] + robotPoses[ai];
			effectorJoints0[ai].pose0.p[0] = effectorTranslation[0];
			effectorJoints0[ai].pose0.p[1] = effectorTranslation[1];
			effectorJoints0[ai].pose0.p[2] = effectorTranslation[2];
			g_buffers->rigidJoints.push_back(effectorJoints0[ai]);
		}

		NvFlexRigidShape table;
		NvFlexMakeRigidBoxShape(&table, -1, tableLength, tableHeight, tableWidth, NvFlexMakeRigidPose(tableOrigin + robotPoses[ai], Quat()));
		table.filter = 0x0;
		table.group = 0;
		table.material.friction = 0.7f;
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.17f, 0.24f, 0.4f)));

		g_buffers->rigidShapes.push_back(table);
	}

    virtual int computeNumJoints()
    {
        return mNumActions;
    }
    
	virtual int ExtractState(int a, float* state, float* jointSpeeds) 
	{
		// Prepare state
		//--------------------
		int numJoints = motors.size();
		vector<float> joints(numJoints * 2, 0.f);
		vector<float> angles(numJoints, 0.f);
		vector<float> lows(numJoints, 0.f);
		vector<float> highs(numJoints, 0.f);

		// auto bd = initJoints[0].body0;
		GetAngles(a, angles, lows, highs);
		for (int i = 0; i < computeNumJoints(); i++)
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

			jointSpeeds[i] = joinVelocities[a][qq];
		}

		int ct = 0;

		Transform pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[a]], (NvFlexRigidPose*)&pose);

		Vec3 effectorPose1 = pose.p - robotPoses[a];

		// 0-2 end-effector translation
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = effectorPose1[i];
		}

        if (!doDeltaPlanarControl && !doWristRollControl)
		{
            // 3-5 end-effector rpy
            state[ct++] = rolls[a];
            state[ct++] = pitches[a];
            state[ct++] = yaws[a];
        }
        else if (doDeltaPlanarControl && doWristRollControl)
    	{
            // 3 end-effector roll
            state[ct++] = rolls[a];
        }
        else if (!doDeltaPlanarControl && doWristRollControl)
        {
            throw std::logic_error("Incomplatible options selected for DoDeltaPlanarControl and DoWristRollControl");
        }
        else {}

        if (doGripperControl)
		{
            // 6 - end-effector finger width
            state[ct++] = fingerWidths[a];
        }
        
		return ct;
	}

	// To be overwritten by child env
	virtual void ExtractChildState(int ai, float* state, int& ct) {}
    virtual void ExtractSensorState(int ai, float* state, int& ct) {}

	void PopulateState(int ai, float* state) override
	{	
		float* jointSpeedsA = &jointSpeeds[ai][0];
		int ct = ExtractState(ai, state, jointSpeedsA);
		ExtractChildState(ai, state, ct);
        //printf("cam starts at %d\n", ct);
        ExtractSensorState(ai, state, ct);

        
        if (ct != mNumObservations)
		{
            printf("ct: %d, mNumObservations: %d\n", ct, mNumObservations);
            throw std::logic_error("state not populated correctly");
        }

        for(int j = 0; j < ct; j++)
        {
            if (std::isnan(state[j]))
            {
                printf("nan at %d\n", j);
                throw std::logic_error("state has nans");
            }
        }
        
	}

	// To be overwritten by child env
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) override = 0;

	virtual void ApplyTargetControl(int agentIndex)
	{
		vector<float> angles(mNumActions, 0.f);
		vector<float> lows(mNumActions, 0.f);
		vector<float> highs(mNumActions, 0.f);
		GetAngles(agentIndex, angles, lows, highs);

		float* actions = GetAction(agentIndex);
		for (int i = 0; i < mNumActions; i++)
		{
			float cc = Clamp(actions[i], -1.f, 1.f);

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

				float smoothness = 0.01f;
				float pos = Lerp(angles[i], targetPos, smoothness);
				//	cout << "pos = " << pos << endl;
			//	g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] = pos;

				if (i < (mNumActions - 1))
				{
					g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] = pos;
				}
				else
				{
				//	g_buffers->rigidJoints[fingersLeft[agentIndex]].targets[eNvFlexRigidJointAxisX] = 0.f; //pos;
				//	g_buffers->rigidJoints[fingersRight[agentIndex]].targets[eNvFlexRigidJointAxisX] = 0.f; //pos;
				}
			}
		}
	}

	void ApplyVelocityControl(int agentIndex, float df) override
	{
		float* actions = GetAction(agentIndex);
		for (int i = 0; i < mNumActions; i++)
		{
			actions[i] = Clamp(actions[i], -1.f, 1.f);
		}

		float x = targetEffectorTranslations[agentIndex].x + actions[0] * speedTra;
		float y = targetEffectorTranslations[agentIndex].y + actions[1] * speedTra;
		float z = targetEffectorTranslations[agentIndex].z + actions[2] * speedTra;

		float roll = rolls[agentIndex] + actions[3] * speedRot;
		float pitch = pitches[agentIndex] + actions[4] * speedRot;
		float yaw = yaws[agentIndex] + actions[5] * speedRot;

		float grip = fingerWidths[agentIndex] + actions[6] * speedGrip;
		
		PerformIDStep(agentIndex, x, y, z, roll, pitch, yaw, grip, false);
	}

	virtual void ApplyInverseDynamicsControl(int agentIndex) override
	{
		float* actions = GetAction(agentIndex);
		for (int i = 0; i < mNumActions; i++)
		{
			actions[i] = Clamp(actions[i], -1.f, 1.f);
		}

		// PerformIDStep(agentIndex, actions[0], actions[1], actions[2], actions[3], actions[4], actions[5], actions[6], true);
		PerformIDStep(agentIndex, actions, true);
	}

	virtual void PerformIDStep(int ai, float* actions, bool scale=true)
	{
		float targetx, targety, targetz, oroll, opitch, oyaw, newWidth;

		if (scale)
		{	
			if(doDeltaPlanarControl)
			{
				targetx = scaleActions(limitsTra[0][0], limitsTra[0][1], actions[0]);
				targety = scaleActions(limitsTra[1][0], limitsTra[1][1], actions[1]);
				targetz = scaleActions(limitsTra[2][0], limitsTra[2][1], actions[2]);

                if (controlOnPlane)
                {
                    targety = 0.55;
                }

				if(doGripperControl)
				{
					newWidth = scaleActions(limitsGrip[0], limitsGrip[1], actions[3]);
				}
				else
				{
					newWidth = fingerWidthMin;
				}

				float asmoothing = 0.04f;
				if(doWristRollControl)
				{
					oroll = scaleActions(limitsRot[0], limitsRot[1], actions[4]);
					rolls[ai] = Lerp(rolls[ai], oroll, asmoothing);
				}
				else
				{
					rolls[ai] = 0.0f;
				}

				pitches[ai] = 0.0f;
				yaws[ai] = DegToRad(-90.f);
			}
			else
			{
				targetx = scaleActions(limitsTra[0][0], limitsTra[0][1], actions[0]);
				targety = scaleActions(limitsTra[1][0], limitsTra[1][1], actions[1]);
				targetz = scaleActions(limitsTra[2][0], limitsTra[2][1], actions[2]);
				oroll = scaleActions(limitsRot[0], limitsRot[1], actions[3]);
				opitch = scaleActions(limitsRot[0], limitsRot[1], actions[4]);
				oyaw = scaleActions(limitsRot[0], limitsRot[1], actions[5]);

				if(doGripperControl) newWidth = scaleActions(limitsGrip[0], limitsGrip[1], actions[6]);
				else newWidth = fingerWidthMin;

				float asmoothing = 0.04f;

				rolls[ai] = Lerp(rolls[ai], oroll, asmoothing);
				pitches[ai] = Lerp(pitches[ai], opitch, asmoothing);
				yaws[ai] = Lerp(yaws[ai], oyaw, asmoothing);
			}
		}
		else
		{
			if (doDeltaPlanarControl)
			{	
				float maxSpeed = 5.f * g_dt; // 5 m/s
				targetx = Clamp(targetEffectorTranslations[ai].x + scaleActions(-maxSpeed, maxSpeed, actions[0]), limitsTra[0][0], limitsTra[0][1]);
				targety = Clamp(targetEffectorTranslations[ai].y + scaleActions(-maxSpeed, maxSpeed, actions[1]), limitsTra[1][0], limitsTra[1][1]);
				targetz = Clamp(targetEffectorTranslations[ai].z + scaleActions(-maxSpeed, maxSpeed, actions[2]), limitsTra[2][0], limitsTra[2][1]);
				
				float maxGripperSpeed = .1f * g_dt;
				if(doGripperControl)
				{
					newWidth = Clamp(fingerWidths[ai] + scaleActions(-maxGripperSpeed, maxGripperSpeed, actions[3]), fingerWidthMin, fingerWidthMax);
				}
				else
				{
					newWidth = fingerWidthMin;
				}
				rolls[ai] = 0.f;
				pitches[ai] = 0.f;
				yaws[ai] = DegToRad(-90.f);
			}
			else
			{	
				targetx = Clamp(actions[0], limitsTra[0][0], limitsTra[0][1]);
				targety = Clamp(actions[1], limitsTra[1][0], limitsTra[1][1]);
				targetz = Clamp(actions[2], limitsTra[2][0], limitsTra[2][1]);
				oroll = Clamp(actions[3], limitsRot[0], limitsRot[1]);
				opitch = Clamp(actions[4], limitsRot[0], limitsRot[1]);
				oyaw = Clamp(actions[5], limitsRot[0], limitsRot[1]);
				newWidth = Clamp(actions[6], limitsGrip[0], limitsGrip[1]);

				float asmoothing = 0.04f;

				rolls[ai] = Lerp(rolls[ai], oroll, asmoothing);
				pitches[ai] = Lerp(pitches[ai], opitch, asmoothing);
				yaws[ai] = Lerp(yaws[ai], oyaw, asmoothing);
			}
		}

		NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

		const float smoothing = 0.05f;

		// low-pass filter controls otherwise it is too jerky
		targetEffectorTranslations[ai].x = Lerp(targetEffectorTranslations[ai].x, targetx, smoothing);
		targetEffectorTranslations[ai].y = Lerp(targetEffectorTranslations[ai].y, targety, smoothing);
		targetEffectorTranslations[ai].z = Lerp(targetEffectorTranslations[ai].z, targetz, smoothing);

		Vec3 targetEffectorTranslation = targetEffectorTranslations[ai] + robotPoses[ai];
		effector0.pose0.p[0] = targetEffectorTranslation.x;
		effector0.pose0.p[1] = targetEffectorTranslation.y;
		effector0.pose0.p[2] = targetEffectorTranslation.z;

		Quat q = rpy2quat(rolls[ai], pitches[ai], yaws[ai]);
		effector0.pose0.q[0] = q.x;
		effector0.pose0.q[1] = q.y;
		effector0.pose0.q[2] = q.z;
		effector0.pose0.q[3] = q.w;

		g_buffers->rigidJoints[effectorJoints[ai]] = effector0;

		const float fsmoothing = 0.05f;

		// float force = g_buffers->rigidJoints[fingersLeft[ai]].lambda[eNvFlexRigidJointAxisX];
		fingerWidths[ai] = Lerp(fingerWidths[ai], newWidth, fsmoothing);

		g_buffers->rigidJoints[fingersLeft[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
		g_buffers->rigidJoints[fingersRight[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
	}

	void PerformIDStep(int ai, float targetx, float targety, float targetz, float oroll, float opitch, float oyaw, float newWidth, bool scale=true)
	{
		if (scale)
		{	
			targetx = scaleActions(limitsTra[0][0], limitsTra[0][1], targetx);
			targety = scaleActions(limitsTra[1][0], limitsTra[1][1], targety);
			targetz = scaleActions(limitsTra[2][0], limitsTra[2][1], targetz);
			oroll = scaleActions(limitsRot[0], limitsRot[1], oroll);
			opitch = scaleActions(limitsRot[0], limitsRot[1], opitch);
			oyaw = scaleActions(limitsRot[0], limitsRot[1], oyaw);

			if(doGripperControl)
			{
				newWidth = scaleActions(limitsGrip[0], limitsGrip[1], newWidth);
			}
			else
			{
				newWidth = fingerWidthMin;
			}

			float asmoothing = 0.04f;

			rolls[ai] = Lerp(rolls[ai], oroll,asmoothing);
			pitches[ai] = Lerp(pitches[ai], opitch, asmoothing);
			yaws[ai] = Lerp(yaws[ai], oyaw, asmoothing);
		}
		else
		{
			if (doDeltaPlanarControl)
			{	
				float maxSpeed = 5.f * g_dt; // 5 m/s
				targetx = Clamp(targetEffectorTranslations[ai].x + scaleActions(-maxSpeed, maxSpeed, targetx), limitsTra[0][0], limitsTra[0][1]);
				targety = Clamp(targetEffectorTranslations[ai].y + scaleActions(-maxSpeed, maxSpeed, targety), limitsTra[1][0], limitsTra[1][1]);
				targetz = Clamp(targetEffectorTranslations[ai].z + scaleActions(-maxSpeed, maxSpeed, targetz), limitsTra[2][0], limitsTra[2][1]);
				
				float maxGripperSpeed = .1f * g_dt;
				if(doGripperControl)
				{
					newWidth = Clamp(fingerWidths[ai] + scaleActions(-maxGripperSpeed, maxGripperSpeed, newWidth), fingerWidthMin, fingerWidthMax);
				}
				else
				{
					newWidth = fingerWidthMin;
				}
				rolls[ai] = 0.f;
				pitches[ai] = 0.f;
				yaws[ai] = DegToRad(-90.f);
			}
			else
			{	
				targetx = Clamp(targetx, limitsTra[0][0], limitsTra[0][1]);
				targety = Clamp(targety, limitsTra[1][0], limitsTra[1][1]);
				targetz = Clamp(targetz, limitsTra[2][0], limitsTra[2][1]);
				oroll = Clamp(oroll, limitsRot[0], limitsRot[1]);
				opitch = Clamp(opitch, limitsRot[0], limitsRot[1]);
				oyaw = Clamp(oyaw, limitsRot[0], limitsRot[1]);
				newWidth = Clamp(newWidth, limitsGrip[0], limitsGrip[1]);
			}
		}

		NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

		float asmoothing = 0.04f;

		rolls[ai] = Lerp(rolls[ai], oroll, asmoothing);
		pitches[ai] = Lerp(pitches[ai], opitch, asmoothing);
		yaws[ai] = Lerp(yaws[ai], oyaw, asmoothing);

		const float smoothing = 0.05f;

		// low-pass filter controls otherwise it is too jerky
		targetEffectorTranslations[ai].x = Lerp(targetEffectorTranslations[ai].x, targetx, smoothing);
		targetEffectorTranslations[ai].y = Lerp(targetEffectorTranslations[ai].y, targety, smoothing);
		targetEffectorTranslations[ai].z = Lerp(targetEffectorTranslations[ai].z, targetz, smoothing);

		Vec3 targetEffectorTranslation = targetEffectorTranslations[ai] + robotPoses[ai];
		effector0.pose0.p[0] = targetEffectorTranslation.x;
		effector0.pose0.p[1] = targetEffectorTranslation.y;
		effector0.pose0.p[2] = targetEffectorTranslation.z;

		Quat q = rpy2quat(rolls[ai], pitches[ai], yaws[ai]);
		effector0.pose0.q[0] = q.x;
		effector0.pose0.q[1] = q.y;
		effector0.pose0.q[2] = q.z;
		effector0.pose0.q[3] = q.w;

		g_buffers->rigidJoints[effectorJoints[ai]] = effector0;

		const float fsmoothing = 0.04f;
		fingerWidths[ai] = Lerp(fingerWidths[ai], newWidth, fsmoothing);

		g_buffers->rigidJoints[fingersLeft[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
		g_buffers->rigidJoints[fingersRight[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
	}

	// To be overwritten by child env
	virtual void ResetAgent(int a) override {}
	
	~RLFetchBase()
	{
		if (urdf)
		{
			delete urdf;
		}
	}

	virtual void PreHandleCommunication() override {} // Do whatever needed to be done before handling communication
	virtual void ClearContactInfo() override {}
	virtual void FinalizeContactInfo() override {}
	virtual void LockWrite() override {} // Do whatever needed to lock write to simulation
	virtual void UnlockWrite() override {} // Do whatever needed to unlock write to simulation

	float scaleActions(float minx, float maxx, float x)
	{
		x = 0.5f * (x + 1.f) * (maxx - minx) + minx;
		return x;
	}

	virtual void DoGui() override
	{
		if (!mDoLearning)
		{
			NvFlexRigidJoint effector0_0 = g_buffers->rigidJoints[effectorJoints[0]];

			float targetx = effector0_0.pose0.p[0] - robotPoses[0].x;
			float targety = effector0_0.pose0.p[1] - robotPoses[0].y;
			float targetz = effector0_0.pose0.p[2] - robotPoses[0].z;

			float oroll = rolls[0];
			float opitch = pitches[0];
			float oyaw = yaws[0];
			imguiSlider("Gripper X", &targetx, -0.5f, 0.5f, 0.0001f);
			imguiSlider("Gripper Y", &targety, 0.0f, 1.6f, 0.0001f);
			imguiSlider("Gripper Z", &targetz, 0.0f, 1.2f, 0.0001f);
			imguiSlider("Roll", &rolls[0], -180.0f, 180.0f, 0.01f);
			imguiSlider("Pitch", &pitches[0], -180.0f, 180.0f, 0.01f);
			imguiSlider("Yaw", &yaws[0], -180.0f, 180.0f, 0.01f);

			float newWidth = fingerWidths[0];
			imguiSlider("Finger Width", &newWidth, limitsGrip[0], limitsGrip[1], 0.001f);

			for (int ai = 0; ai < mNumAgents; ++ai)
			{
				NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

				float f = 0.1f;

				rolls[ai] = Lerp(oroll, rolls[ai], f);
				pitches[ai] = Lerp(opitch, pitches[ai], f);
				yaws[ai] = Lerp(oyaw, yaws[ai], f);

				const float smoothing = 0.05f;

				// low-pass filter controls otherwise it is too jerky
				float newx = Lerp(effector0.pose0.p[0] - robotPoses[ai].x, targetx, smoothing);
				float newy = Lerp(effector0.pose0.p[1] - robotPoses[ai].y, targety, smoothing);
				float newz = Lerp(effector0.pose0.p[2] - robotPoses[ai].z, targetz, smoothing);

				effector0.pose0.p[0] = newx + robotPoses[ai].x;
				effector0.pose0.p[1] = newy + robotPoses[ai].y;
				effector0.pose0.p[2] = newz + robotPoses[ai].z;

				Quat q = rpy2quat(rolls[ai] * kPi / 180.0f, pitches[ai] * kPi / 180.0f, yaws[ai] * kPi / 180.0f);
				effector0.pose0.q[0] = q.x;
				effector0.pose0.q[1] = q.y;
				effector0.pose0.q[2] = q.z;
				effector0.pose0.q[3] = q.w;

				//     g_buffers->rigidJoints[effectorJoint] = effector0;
				g_buffers->rigidJoints[effectorJoints[ai]] = effector0;

				fingerWidths[ai] = Lerp(fingerWidths[ai], newWidth, smoothing);

				g_buffers->rigidJoints[fingersLeft[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
				g_buffers->rigidJoints[fingersRight[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
			}
		}
	}

	virtual void DoStats() override
	{
		if (doStats)
		{
			int numSamples = 200;

			int start = Max(int(forceLeft.size()) - numSamples, 0);
			int end = Min(start + numSamples, int(forceLeft.size()));

			// convert from position changes to forces
			float units = -1.0f / Sqr(g_dt / g_numSubsteps);

			float height = 50.0f;
			float maxForce = 20.0f;  // What is maxForce?

			float dx = 1.0f;
			float sy = height / maxForce;

			float lineHeight = 10.0f;

			float rectMargin = 10.0f;
			float rectWidth = dx * numSamples + rectMargin * 4.0f;

			float x = float(g_screenWidth) - rectWidth - 20.0f;
			float y = 300.0f;

			DrawRect(x, y - height - rectMargin, rectWidth, 2.0f * height + rectMargin * 3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));

			x += rectMargin * 3.0f;

			DrawImguiString(int(x + dx * float(numSamples)), int(y + 55.0f), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

			DrawLine(x, y, x + float(numSamples) * dx, y, 1.0f, Vec3(1.0f));
			DrawLine(x, y - 50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

			float margin = 5.0f;

			DrawImguiString(int(x - margin), int(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
			DrawImguiString(int(x - margin), int(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
			DrawImguiString(int(x - margin), int(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

			for (int i = start; i < end - 1; ++i)
			{
				float fl0 = Clamp(forceLeft[i] * units, -maxForce, maxForce)*sy;
				float fr0 = Clamp(forceRight[i] * units, -maxForce, maxForce)*sy;

				float fl1 = Clamp(forceLeft[i + 1] * units, -maxForce, maxForce)*sy;
				float fr1 = Clamp(forceRight[i + 1] * units, -maxForce, maxForce)*sy;

				DrawLine(x, y + fl0, x + dx, y + fl1, 1.0f, Vec3(1.0f, 0.0f, 0.0f));
				DrawLine(x, y + fr0, x + dx, y + fr1, 1.0f, Vec3(0.0f, 1.0f, 0.0f));

				x += dx;
			}
		}
	}

	virtual void Update() override
	{
		if (doStats)
		{
			for (int ai = 0; ai < 1; ++ai)
			{
				// record force on the 1st robot finger joints
				forceLeft.push_back(g_buffers->rigidJoints[fingersLeft[ai]].lambda[eNvFlexRigidJointAxisX]);
				forceRight.push_back(g_buffers->rigidJoints[fingersRight[ai]].lambda[eNvFlexRigidJointAxisX]);
			}
		}
	}

	virtual void PostUpdate() override
	{
		// joints are not read back by default
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
	}

	void printState(const float* state)
	{
		std::cout << "------------------------\n";
		for(int i=0; i < mNumObservations; ++i)
		{
			std::cout << "state[" << i << "]: " << state[i] << std::endl;
		}
	}

	void printInfo(const float* state, const float distance, const float reward)
	{
		cout << "mNumObservations: " << mNumObservations << endl;
        cout << "mNumActions: " << mNumActions << endl;
        cout << "distance: " << distance << endl;
        cout << "reward: " << reward << endl;
        printState(state);
	}

	void printVec3(Vec3 v)
	{
		cout << "Vec3: " << v[0] << ", " << v[1] << ", " << v[2] << endl;
	}
	
    virtual Vec3 constructVec3(float* addr)
    {
        return Vec3(addr[0], addr[1], addr[2]);
    }
    
    virtual Vec3 getEndEffectorLocation(int a, float* state)
    {
        return constructVec3(state);
    }

    virtual int getTargetOffset()
    {
        //the target xyz comes right after the parent state
        return RLFetchBase::computeNumObservations();
    }

    virtual int getCubeOffset()
    {
    	return getTargetOffset() + 3;  // 3 = xyz of target
    }
    
    virtual Vec3 getTargetLocation(int a, float* state)
    {	
        return constructVec3(state+getTargetOffset());
    }

    virtual Vec3 getCubeLocation(int a, float* state)
    {	
    	return constructVec3(state+getCubeOffset());
    }

    virtual float distanceToGoal(int a, float* state)
    {
        Vec3 EE = getEndEffectorLocation(a, state);
        Vec3 Target = getTargetLocation(a, state);

        if(relativeTarget)
        {
	        return Length(Target);
        }
        else
        {
        	return Length(EE - Target);
        }
    }

    virtual float distanceOfCubeToGoal(int a, float* state)
    {	
    	Vec3 Cube = getCubeLocation(a, state); 
        Vec3 Target = getTargetLocation(a, state);

        if(relativeTarget)
        {
	        return Length(Target);
        }
        else
        {
        	return Length(Cube - Target);
        }
    }
};

class RLFetchReach : public RLFetchBase
{
public:

	vector<int> targetSphere;
	vector<int> targetRenderMaterials;
	Vec3 redColor;
	Vec3 greenColor;
	vector<Vec3> targetPoses;
    vector<vector<float>> targetImages;
    bool initOnPlane;
    
	float radius;

	RLFetchReach()
	{
		doStats = false;

		radius = 0.04f;
		redColor = Vec3(1.0f, 0.04f, 0.07f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);

		effectorLimits = {
			{ tableOrigin[0] - tableLength / 2.f, tableOrigin[0] + tableLength / 2.f },
			{ tableHeight, 1.f },
			{ tableOrigin[2] - tableWidth / 2.f, tableOrigin[2] + tableWidth / 2.f }
		};
	}

    virtual void ParseJson() override
    {
        RLFetchBase::ParseJson();
        initOnPlane = GetJsonVal(g_sceneJson, "InitOnPlane", false);
    }

    virtual int computeNumObservations() override
    {
        return RLFetchBase::computeNumObservations() + 3; //target
    }

	virtual void LoadChildEnv() override
	{
		targetPoses.resize(mNumAgents, Vec3(0.4f, 1.f, 0.8f));
		targetSphere.resize(mNumAgents, -1);
		targetRenderMaterials.resize(mNumAgents);
	}

	virtual void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
            if (initOnPlane)
            {
                targetPoses[ai] = Vec3(
                    Randf(limitsTra[0][0], limitsTra[0][1]),
                    0.55, 
                    Randf(limitsTra[2][0], limitsTra[2][1]));
            }
            else
            {
                targetPoses[ai] = Vec3(
                    Randf(limitsTra[0][0], limitsTra[0][1]),
                    Randf(limitsTra[1][0], limitsTra[1][1]),
                    Randf(limitsTra[2][0], limitsTra[2][1]));
            }
		}
		else
		{
			targetPoses[ai] = Vec3(-0.3f, 0.5f, 0.5f);
		}

		if (renderTarget)
		{
			NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
			g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
		}
	}

	virtual void AddChildEnvBodies(int ai, Transform gt) override
	{
		if (renderTarget)
		{
			NvFlexRigidShape targetShape;
			NvFlexMakeRigidSphereShape(&targetShape, -1, radius, NvFlexMakeRigidPose(0,0));

			int renderMaterial = AddRenderMaterial(redColor);
			targetRenderMaterials[ai] = renderMaterial;
			targetShape.user = UnionCast<void*>(renderMaterial);
			targetShape.group = 1;
			targetSphere[ai] = g_buffers->rigidShapes.size();
			g_buffers->rigidShapes.push_back(targetShape);
		}
		
		SampleTarget(ai);
	}

	virtual void ExtractChildState(int ai, float* state, int& ct) override
	{   
		// 7-9 target xyz
		for (int i = 0; i < 3; ++i)
		{
			if (relativeTarget)
			{
				state[ct++] = targetPoses[ai][i] - state[i];
			}
			else
			{
				state[ct++] = targetPoses[ai][i];
			}
		}
	}
   
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) override
	{
        float distance = distanceToGoal(a, state);

        if (!sparseRewards)
		{
            float distReward = 0.f;
            if (rewardScheme == "default")
            {
                float reg = 0.f;
                float* prevAction = GetPrevAction(a);
                for (int i = 0; i < mNumActions; i++)
                {
                    reg += Pow((action[i] - prevAction[i]), 2) / (float)mNumActions;
                }
                distReward = 2.f * exp(-5.f * distance) - 0.4f * reg;

                if(renderTarget)
                {
                    ColorTargetForReward(distReward, a);
                }
       
                rew = distReward;
            }
            else if (rewardScheme == "linear")
            {
            	rew = -distance;
            }
            else
            {
                throw std::logic_error("invalid reward scheme");
            }
        } 
		else
		{
            float distance_threshold = 0.05f;
            rew = (distance > distance_threshold) ? -1.0f : 0.0f;

            if(renderTarget)
	        {	// TODO (romurthy): Shift to ColotTargetForReward?
		        float x = max(exp(-10.0f*distance), 0.f);
				g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;	
            }
            
        }

        // printInfo(state, distance, rew);
	}

   	virtual void ColorTargetForReward(float distReward, int a)
   	{
		float x = max((distReward - 0.2f) / 1.79f, 0.f); 
		g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;
	}

	void SetInitArmPose(int ai)
	{	
		fingerWidths[ai] = fingerWidthMin;
	}
   	 
	virtual void ResetAgent(int ai) override
	{
		for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second; i++)
		{
			g_buffers->rigidBodies[i] = initBodies[i];
		}

		g_buffers->rigidShapes.map();
		SampleTarget(ai);
		g_buffers->rigidShapes.unmap();
		NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

		SetInitArmPose(ai);

		RLFlexEnv::ResetAgent(ai);
	}
};


class RLFetchCube : public RLFetchBase
{
public:
	vector<pair<int, int>> cubeShapeBodyIds;
	vector<bool> cubeGrasped;
	vector<pair<bool, bool>> contactsLeftRight;
	unordered_map<int, int> mapFingerToAgent;
	unordered_map<int, bool> mapFingerToSide;
	
	vector<float> forceFingerLeft;
	vector<float> forceFingerRight;
	float graspForceThresh;

	vector<float> cubeContactForces;

	Mesh* cubeMesh;
	vector<float> cubeScales;
	vector<float> cubeMasses;

	vector<int> targetSphere;
	vector<int> targetRenderMaterials;
	Vec3 redColor;
	Vec3 greenColor;
	vector<Vec3> targetPoses;

	float radius;
	bool sampleTableStates;
	int sampleTargetSpace;

	float distance_threshold;

	RLFetchCube()
	{
		controlType = eInverseDynamics;

		graspForceThresh = 20.f;
		radius = 0.022f;
		doStats = true;

		redColor = Vec3(1.0f, 0.02f, 0.06f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);

		distance_threshold = 0.1f;

		sampleTableStates =  GetJsonVal(g_sceneJson, "SampleTableStates", false);
		sampleTargetSpace =  GetJsonVal(g_sceneJson, "SampleTargetSpace", 2);
	}
    
    int computeNumObservations()
	{
        // + 3 for cube, +3 for target, +2 finger contacts
        return RLFetchBase::computeNumObservations() + 8;
    }

	void LoadChildEnv()
	{
		initEffectorTranslation = Vec3(0.0f, tableHeight + 0.2f, 0.6f);

		forceFingerLeft.resize(mNumAgents);
		forceFingerRight.resize(mNumAgents);
		forceLeft.resize(mNumAgents);
		forceRight.resize(mNumAgents);
		cubeContactForces.resize(mNumAgents);
		cubeMasses.resize(mNumAgents);
		cubeScales.resize(mNumAgents);
		targetSphere.resize(mNumAgents);
		targetRenderMaterials.resize(mNumAgents);
		targetPoses.resize(mNumAgents);
		contactsLeftRight.resize(mNumAgents);
		cubeShapeBodyIds.resize(mNumAgents);
		cubeGrasped.resize(mNumAgents);
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
			// Randomly generate a target either on the table or in the air (Look at Footnote 3, Pg. 6. Hindsight Experience Replay)
			if (sampleTableStates)
			{
				if (Randf(0.0f, 1.0f) < 0.5f)
				{
					if(sampleTargetSpace == 0)	targetPoses[ai] = Vec3(Randf(-0.3f, 0.3f), tableHeight + Randf(0.15f, 0.75f), Randf(0.45f, 0.75f));
					else if(sampleTargetSpace == 1)	targetPoses[ai] = Vec3(Randf(-0.15f, 0.15f), tableHeight + Randf(0.10f, 0.35f), Randf(0.45f, 0.75f));  // OpenAI Goal Space
					else	targetPoses[ai] = Vec3(Randf(-0.05f, 0.05f), tableHeight + Randf(0.10f, 0.15f), Randf(0.55f, 0.65f));  // Small Workspace
				}
				else
				{
					if(sampleTargetSpace == 0)	targetPoses[ai] = Vec3(Randf(-0.3f, 0.3f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.45f, 0.75f));
					else if(sampleTargetSpace == 1)	targetPoses[ai] = Vec3(Randf(-0.15f, 0.15f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.45f, 0.75f));  // OpenAI Goal Space
					else	targetPoses[ai] = Vec3(Randf(-0.05f, 0.05f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.55f, 0.65f));  // Small Workspace
				}
			}
			else
			{
				if(sampleTargetSpace == 0)	targetPoses[ai] = Vec3(Randf(-0.3f, 0.3f), tableHeight + Randf(0.15f, 0.75f), Randf(0.45f, 0.75f));
				else if(sampleTargetSpace == 1)	targetPoses[ai] = Vec3(Randf(-0.15f, 0.15f), tableHeight + Randf(0.10f, 0.35f), Randf(0.45f, 0.75f));  // OpenAI Goal Space
				else	targetPoses[ai] = Vec3(Randf(-0.05f, 0.05f), tableHeight + Randf(0.10f, 0.15f), Randf(0.55f, 0.65f));  // Small Workspace
			}
		}
		else
		{
			targetPoses[ai] = Vec3(0.3f, 0.6f, 0.6f);
		}

		if(renderTarget)
		{
			NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
			g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
		}
	}


	void SampleCube(int ai)
	{	
		Vec3 pos; Quat quat; float cubeMass;
		if (sampleInitStates)
		{
			// pos = Vec3(Randf(-0.2f, 0.2f), 0.4f + cubeScales[ai] * 0.5f + 0.02f, Randf(0.55f, 0.75f));  //OpenAI Workspace
			pos = Vec3(Randf(-0.1f, 0.1f), 0.4f + cubeScales[ai] * 0.5f + 0.02f, Randf(0.5f, 0.7f));  // Small Workspace
			if (doDeltaPlanarControl)
			{
				quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), 0.f);
			}
			else
			{
				quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(-0.2f, 0.2f));
			}
			cubeMass = cubeMasses[ai] * Randf(0.95f, 1.05f);
		}
		else
		{
			pos = Vec3(0.f, 0.4f + cubeScales[ai] * 0.5f + 0.02f, 0.7f);
			quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), 0.f);
			cubeMass = cubeMasses[ai];
		}

		// If the cube is close to the target then sample again
		if(Length(pos - targetPoses[ai]) < distance_threshold)
		{
			SampleCube(ai);
		}

		int	bodyId = cubeShapeBodyIds[ai].second;
		g_buffers->rigidBodies[bodyId].mass = cubeMass;
		NvFlexRigidPose pose = NvFlexMakeRigidPose(pos + robotPoses[ai], quat);
		NvFlexSetRigidPose(&g_buffers->rigidBodies[bodyId], &pose);
	}

	void AddChildEnvBodies(int ai, Transform gt)
	{
		// initialize maps for robot fingers
		mapFingerToAgent.emplace(urdf->rigidNameMap["l_gripper_finger_link"], ai);
		mapFingerToAgent.emplace(urdf->rigidNameMap["r_gripper_finger_link"], ai);
		mapFingerToSide.emplace(urdf->rigidNameMap["l_gripper_finger_link"], true);
		mapFingerToSide.emplace(urdf->rigidNameMap["r_gripper_finger_link"], false);

		// Create target sphere
		if(renderTarget)
		{
			NvFlexRigidShape targetShape;
			NvFlexMakeRigidSphereShape(&targetShape, -1, radius, NvFlexMakeRigidPose(0,0));
			int renderMaterial = AddRenderMaterial(redColor);
			targetRenderMaterials[ai] = renderMaterial;
			targetShape.user = UnionCast<void*>(renderMaterial);
			targetShape.group = 1;
			targetSphere[ai] = g_buffers->rigidShapes.size();
			g_buffers->rigidShapes.push_back(targetShape);
		}

		SampleTarget(ai);

		// Create cube
		cubeShapeBodyIds[ai] = pair<int, int>(g_buffers->rigidShapes.size(), g_buffers->rigidBodies.size());

		Mesh* cubeMesh = ImportMesh("../../data/box.ply");
		NvFlexTriangleMeshId cubeMeshId = CreateTriangleMesh(cubeMesh, 0.00125f);

		float friction, density;
		if (sampleInitStates)
		{
			cubeScales[ai] = Randf(0.038f, 0.042f);
			density = Randf(480.f, 520.f);
			friction = Randf(0.95f, 1.15f);
		}
		else
		{
			cubeScales[ai] = 0.04f;
			density = 500.f;
			friction = 1.0f;
		}

		NvFlexRigidShape cubeShape;
		NvFlexMakeRigidTriangleMeshShape(&cubeShape, g_buffers->rigidBodies.size(), cubeMeshId,
			NvFlexMakeRigidPose(0, 0), cubeScales[ai], cubeScales[ai], cubeScales[ai]);
		cubeShape.material.friction = friction;
		cubeShape.material.rollingFriction = 0.0f;
		cubeShape.material.torsionFriction = 0.1f;
		cubeShape.thickness = 0.00125f;
		cubeShape.filter = 0x0;
		cubeShape.group = 0;
		cubeShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.97f, 0.34f, 0.04f)));
		g_buffers->rigidShapes.push_back(cubeShape);

		NvFlexRigidBody cubeBody;
		NvFlexMakeRigidBody(g_flexLib, &cubeBody, Vec3(), Quat(), &cubeShape, &density, 1);
		cubeMasses[ai] = cubeBody.mass;
		g_buffers->rigidBodies.push_back(cubeBody);

		SampleCube(ai);
	}

	void ExtractChildState(int ai, float* state, int& ct)
	{   

		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];
		
		// 10-12 target xyz
		for (int i = 0; i < 3; ++i)
		{
			if(relativeTarget)
			{
				state[ct++] = targetPoses[ai][i] - cubeTraLocal[i];  // For relative target wrt object
			}
			else
			{
				state[ct++] = targetPoses[ai][i];
			}
		}

		// 13-15 cube xyz
		for (int i = 0; i < 3; ++i)
		{
			if(relativeTarget)
			{
				state[ct++] = cubeTraLocal[i] - state[i];  // For relative target wrt object
			}
			else
			{
				state[ct++] = cubeTraLocal[i];
			}
		}

		// 16-17 finger contacts
		state[ct++] = 0.1f * forceFingerLeft[ai];
		state[ct++] = 0.1f * forceFingerRight[ai];
	}

	void SetInitArmPose(int ai)
	{	
		fingerWidths[ai] = fingerWidthMax;
		rolls[ai] = 0.f;
		pitches[ai] = 0.f;
		yaws[ai] = -kPi/2.f;

		// if (sampleInitStates)
		// {	
		// 	targetEffectorTranslations[ai] = Vec3(Randf(-0.2f, 0.2f), tableHeight + Randf(0.2f, 0.3f), Randf(0.4f, 0.6f));
		// }
		// else
		// {
			targetEffectorTranslations[ai] = initEffectorTranslation;
		// }
	}

	void ResetAgent(int a)
	{
		// This needs to be called first to avoid overwriting rigid bodies of updated cubes
		// for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
		// 	g_buffers->rigidBodies[i] = initBodies[i];

		g_buffers->rigidShapes.map();
		SampleTarget(a);
		SampleCube(a);
		g_buffers->rigidShapes.unmap();
		NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
		SetInitArmPose(a);

		RLFlexEnv::ResetAgent(a);
	}

	void ColorTargetForReward(float distReward, int a)
   	{
		float x = max(exp(-12.0f * distReward), 0.f);
		g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;
	}

	void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float distance = distanceOfCubeToGoal(a, state);
        if (!sparseRewards) 
        {	
        	// TODO: Not confirmed to work
			// Shaped Dense Reward
			float rewGrasp = 0.f, rewTarget = 0.f, rewCube = 0.f, rewAction = 0.f, rewSolved = 0.f;
			float distTarget = 1000.f;
            		float distCube;

			float* prevAction = GetPrevAction(a);
			for (int i = 0; i < mNumActions; i++)
			{
				rewAction += -0.25f * Pow(action[i] - prevAction[i], 2) / (float)mNumActions;
			}

			float cubeReachWeight = 0.8f;
			float touchWeight = 0.5f;
			float maxForceWeight = 2.f;
			float cubeTargetWeight = 2.f;
			float solvedWeight = 5.f;

			// reward for getting close to cube
			Vec3 cubeTraLocal = getCubeLocation(a, state);
			distCube = Length(getEndEffectorLocation(a, state) - cubeTraLocal);
			rewCube = cubeReachWeight * exp(-4.f * distCube);
			float x = max((rewCube - 0.7f) / cubeReachWeight, 0.f);

			if (cubeGrasped[a])
			{
				// Reward for touching cube
				rewGrasp = touchWeight;

				// reward for applying fingers force to a cube
				float touchForce = 0.5f * (forceFingerLeft[a] + forceFingerRight[a]);
				rewGrasp += min(touchForce, maxForceWeight);

				// reward for getting cube to target
				distTarget = distance;
				rewTarget = cubeTargetWeight * exp(-5.f * distTarget);

				// update target sphere color
				// TODO(jaliang): change this to something easier to tune
				x += max((rewGrasp - 2.1f) / (touchWeight + maxForceWeight), 0.f);
				x += max((rewTarget - 0.65f) / cubeTargetWeight, 0.f);
			}

			x = Clamp(x, 0.f, 1.f);
			g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;

			bool cubeOffTable = cubeTraLocal[1] < tableHeight ? true : false;
			bool solved = cubeGrasped[a] && distTarget < 0.01f ? true : false;

			dead = cubeOffTable || solved;

			if (solved)
			{
				rewSolved = solvedWeight;
			}
			else if (cubeOffTable)
			{
				rewSolved = -1.f;
			}

			rew = rewGrasp + rewCube + rewTarget + rewAction + rewSolved;

            // L2 Dense reward
            // rew = -distance;
        } 
        else 
        {
            float distance_threshold = 0.1f;
            rew = (distance > distance_threshold) ? -1.0f : 0.0f;
        }

        if(renderTarget)
        {
        	ColorTargetForReward(distance, a);
        }
	}

	void ClearContactInfo()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			cubeGrasped[a] = false;
			contactsLeftRight[a] = (pair<bool, bool>(false, false));
			forceFingerLeft[a] = 0.f;
			forceFingerRight[a] = 0.f;
		}
	}

	void CheckFingerCubeContact(int body0, int body1, int& ai, bool& isLeft)
	{
		// check if body0 is a finger 
		ai = -1;
		if (mapFingerToAgent.find(body0) != mapFingerToAgent.end())
		{
			ai = mapFingerToAgent.at(body0);
			// check if body1 is the cube corresponding to agent id. 
			if (body1 == cubeShapeBodyIds[ai].second)
			{
				isLeft = mapFingerToSide[body0]; // indicate which finger. left is true, right is false
			}
			else
			{
				ai = -1;
			}
		}
	}

	void FinalizeContactInfo()
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

		float forceScale = 0.1f;

		NvFlexRigidContact* ct = &(rigidContacts[0]);

		int ai = 0;
		bool isLeft;
		for (int i = 0; i < numContacts; ++i)
		{
			ai = -1;

			CheckFingerCubeContact(ct[i].body0, ct[i].body1, ai, isLeft);
			if (ai == -1)
			{
				CheckFingerCubeContact(ct[i].body1, ct[i].body0, ai, isLeft);
			}

			if (ai != -1)
			{
				if (isLeft)
				{
					contactsLeftRight[ai].first = true;
					forceFingerLeft[ai] += forceScale * ct[i].lambda;
				}
				else
				{
					contactsLeftRight[ai].second = true;
					forceFingerRight[ai] += forceScale * ct[i].lambda;
				}
			}
		}
		forceLeft.push_back(forceFingerLeft[ai]);
		forceRight.push_back(forceFingerRight[ai]);
		rigidContacts.unmap();
		rigidContactCount.unmap();

		for (int ai = 0; ai < mNumAgents; ai++)
		{
			if (contactsLeftRight[ai].first && contactsLeftRight[ai].second
				&& forceFingerLeft[ai] > 0.01f // forceScale * graspForceThresh 
				&& forceFingerRight[ai] > 0.01f) // forceScale * graspForceThresh)
			{
				if (!sampleInitStates)
				{
					cout << "Left finger - cube contact force = " << forceFingerLeft[ai] / forceScale << endl;
					cout << "Right finger - cube contact force = " << forceFingerRight[ai] / forceScale << endl;
				}
				cubeGrasped[ai] = true;
			}
		}
	}

	void DoStats()
	{
		if (doStats)
		{
			int numSamples = 200;

			int start = Max(int(forceLeft.size()) - numSamples, 0);
			int end = Min(start + numSamples, int(forceLeft.size()));

			// convert from position changes to forces
			float units = -1.0f / Sqr(g_dt / g_numSubsteps);

			float height = 50.0f;
			float maxForce = 50.0f;  // What is maxForce?

			float dx = 1.0f;
			float sy = height / maxForce;

			float lineHeight = 10.0f;

			float rectMargin = 10.0f;
			float rectWidth = dx * numSamples + rectMargin * 4.0f;

			float x = float(g_screenWidth) - rectWidth - 20.0f;
			float y = 300.0f;

			DrawRect(x, y - height - rectMargin, rectWidth, 2.0f * height + rectMargin * 3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));

			x += rectMargin * 3.0f;

			DrawImguiString(int(x + dx * float(numSamples)), int(y + 55.0f), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

			DrawLine(x, y, x + float(numSamples) * dx, y, 1.0f, Vec3(1.0f));
			DrawLine(x, y - 50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

			float margin = 5.0f;

			DrawImguiString(int(x - margin), int(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
			DrawImguiString(int(x - margin), int(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
			DrawImguiString(int(x - margin), int(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

			for (int i = start; i < end - 1; ++i)
			{
				float fl0 = Clamp(forceLeft[i] * units, -maxForce, maxForce) * sy;
				float fr0 = Clamp(forceRight[i] * units, -maxForce, maxForce) * sy;

				float fl1 = Clamp(forceLeft[i + 1] * units, -maxForce, maxForce) * sy;
				float fr1 = Clamp(forceRight[i + 1] * units, -maxForce, maxForce) * sy;

				DrawLine(x, y + fl0, x + dx, y + fl1, 1.0f, Vec3(1.0f, 0.0f, 0.0f));
				DrawLine(x, y + fr0, x + dx, y + fr1, 1.0f, Vec3(0.0f, 1.0f, 0.0f));

				x += dx;
			}
		}
	}

};

class RLFetchPush : public RLFetchBase
{
public:
	vector<pair<int, int>> cubeShapeBodyIds;
	vector<bool> cubeGrasped;
	vector<pair<bool, bool>> contactsLeftRight;
	unordered_map<int, int> mapFingerToAgent;
	unordered_map<int, bool> mapFingerToSide;
	
	vector<float> forceFingerLeft;
	vector<float> forceFingerRight;
	float graspForceThresh;

	vector<float> cubeContactForces;

	Mesh* cubeMesh;
	vector<float> cubeScales;
	vector<float> cubeMasses;

	vector<int> targetSphere;
	vector<int> targetRenderMaterials;
	Vec3 redColor;
	Vec3 greenColor;
	vector<Vec3> targetPoses;
	float distance_threshold;

	float radius;

	RLFetchPush()
	{
		controlType = eInverseDynamics;

		graspForceThresh = 20.f;
		radius = 0.022f;
		doStats = true;

		redColor = Vec3(1.0f, 0.04f, 0.07f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);

		distance_threshold = 0.1f;

	}

    int computeNumObservations()
    {
        // +3 for target, +3 for box location
        return RLFetchBase::computeNumObservations() + 6;
    }


	void LoadChildEnv()
	{
		initEffectorTranslation = Vec3(0.f, tableHeight + 0.15f, 0.5f);

		forceFingerLeft.resize(mNumAgents);
		forceFingerLeft.resize(mNumAgents);
		forceFingerRight.resize(mNumAgents);
		forceLeft.resize(mNumAgents);
		forceRight.resize(mNumAgents);
		cubeContactForces.resize(mNumAgents);
		cubeMasses.resize(mNumAgents);
		cubeScales.resize(mNumAgents);
		targetSphere.resize(mNumAgents);
		targetRenderMaterials.resize(mNumAgents);
		targetPoses.resize(mNumAgents);
		contactsLeftRight.resize(mNumAgents);
		cubeShapeBodyIds.resize(mNumAgents);
		cubeGrasped.resize(mNumAgents);
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{	//The poses have been chosen conservatively for now; (romurthy) Change back to (-0.3, 0.3)
			// targetPoses[ai] = Vec3(Randf(limitsTra[0][0], limitsTra[0][1]), Randf(limitsTra[1][0], limitsTra[1][1]), Randf(limitsTra[2][0], limitsTra[2][1]));
			// targetPoses[ai] = Vec3(Randf(-0.3f, 0.3f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.45f, 0.75f));
			targetPoses[ai] = Vec3(Randf(-0.15f, 0.15f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.45f, 0.75f));
			// targetPoses[ai] = Vec3(Randf(-0.1f, 0.1f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.5f, 0.7f));
		}
		else
		{
			targetPoses[ai] = Vec3(-0.3f, 0.5f, 0.5f);
		}

		if(renderTarget)
		{
			NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
			g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
		}
	}

	bool SampleCube(int ai)
	{	
		Vec3 pos; Quat quat; float cubeMass;
		if (sampleInitStates)
		{	// TODO (romurthy): Why does this seg fault when changed to (-0.3, 0.0)
			// pos = Vec3(Randf(-0.2f, 0.2f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.45f, 0.75f));
			pos = Vec3(Randf(-0.15f, 0.15f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.45f, 0.75f));
			// pos = Vec3(Randf(-0.10f, 0.10f), tableHeight + cubeScales[ai] * 0.5f + 0.02f, Randf(0.5f, 0.7f));
			quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(-0.2f, 0.2f));
			cubeMass = cubeMasses[ai] * Randf(0.95f, 1.05f);
		}
		else
		{
			pos = Vec3(0.f, tableHeight + cubeScales[ai] * 0.5f + 0.02f, 0.6f);
			quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), 0.f);
			cubeMass = cubeMasses[ai];
		}

		// If the cube is close to the target then sample again
		if(Length(pos - targetPoses[ai]) < distance_threshold)
		{
			return false;
		}

		int	bodyId = cubeShapeBodyIds[ai].second;
		g_buffers->rigidBodies[bodyId].mass = cubeMass;
		NvFlexRigidPose pose = NvFlexMakeRigidPose(pos + robotPoses[ai], quat);
		NvFlexSetRigidPose(&g_buffers->rigidBodies[bodyId], &pose);

		return true;
	}

	void AddChildEnvBodies(int ai, Transform gt)
	{
		// initialize maps for robot fingers
		mapFingerToAgent.emplace(urdf->rigidNameMap["l_gripper_finger_link"], ai);
		mapFingerToAgent.emplace(urdf->rigidNameMap["r_gripper_finger_link"], ai);
		mapFingerToSide.emplace(urdf->rigidNameMap["l_gripper_finger_link"], true);
		mapFingerToSide.emplace(urdf->rigidNameMap["r_gripper_finger_link"], false);

		// Create target sphere

		if(renderTarget)
		{
			NvFlexRigidShape targetShape;
			NvFlexMakeRigidSphereShape(&targetShape, -1, radius, NvFlexMakeRigidPose(0, 0));
			int renderMaterial = AddRenderMaterial(redColor);
			targetRenderMaterials[ai] = renderMaterial;
			targetShape.user = UnionCast<void*>(renderMaterial);
			targetShape.group = 1;
			targetSphere[ai] = g_buffers->rigidShapes.size();
			g_buffers->rigidShapes.push_back(targetShape);
		}
		
		SampleTarget(ai);

		// Create cube
		cubeShapeBodyIds[ai] = pair<int, int>(g_buffers->rigidShapes.size(), g_buffers->rigidBodies.size());

		Mesh* cubeMesh = ImportMesh("../../data/box.ply");
		NvFlexTriangleMeshId cubeMeshId = CreateTriangleMesh(cubeMesh, 0.00125f);

		float friction, density;
		if (sampleInitStates)
		{
			cubeScales[ai] = Randf(0.048f, 0.052f);  //TODO (romurthy): Change this back to (0.038f, 0.042f)
			density = Randf(480.f, 520.f);
			friction = Randf(0.65f, 0.85f);  // TODO (romurthy): Change these back to (0.95f, 1.15f)
		}
		else
		{
			cubeScales[ai] = 0.05f;  //TODO (romurthy): Change this back to 0.04f
			density = 500.f;
			friction = 0.75f;  // TODO (romurthy): Change this back to 1.0f
		}

		NvFlexRigidShape cubeShape;
		NvFlexMakeRigidTriangleMeshShape(&cubeShape, g_buffers->rigidBodies.size(), cubeMeshId,
			NvFlexMakeRigidPose(0, 0), cubeScales[ai], cubeScales[ai], cubeScales[ai]);
		cubeShape.material.friction = friction;
		cubeShape.material.rollingFriction = 0.0f;
		cubeShape.material.torsionFriction = 0.1f;
		cubeShape.thickness = 0.00125f;
		cubeShape.filter = 0x0;
		cubeShape.group = 0;
		cubeShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.97f, 0.34f, 0.04f)));
		g_buffers->rigidShapes.push_back(cubeShape);

		NvFlexRigidBody cubeBody;
		NvFlexMakeRigidBody(g_flexLib, &cubeBody, Vec3(), Quat(), &cubeShape, &density, 1);
		cubeMasses[ai] = cubeBody.mass;
		g_buffers->rigidBodies.push_back(cubeBody);

		bool goodSample =false;
		while(goodSample != true)
		{
			goodSample = SampleCube(ai);
		}
	}

	virtual void ExtractChildState(int ai, float* state, int& ct)
	{	
		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];

		// 10-12 target xyz -> 7-9 -> 6-8
		for (int i = 0; i < 3; ++i)
		{
			if(relativeTarget)
			{
				state[ct++] = targetPoses[ai][i] - cubeTraLocal[i];
			}
			else
			{
				state[ct++] = targetPoses[ai][i];	
			}
		}

		// 13-15 cube xyz -> 10-12 -> 9-11
		for (int i = 0; i < 3; ++i)
		{
			if(relativeTarget)
			{
				state[ct++] = cubeTraLocal[i] - state[i];
			}
			else
			{
				state[ct++] = cubeTraLocal[i];  
			}
		}
	}

	void SetInitArmPose(int ai)
	{	//Hack to set the initial gripper to be closed
		fingerWidths[ai] = fingerWidthMin;
		rolls[ai] = 0.f;
		pitches[ai] = 0.f;
		yaws[ai] = -kPi/2;

		// if (sampleInitStates)
		// {	
		// 	targetEffectorTranslations[ai] = Vec3(Randf(-0.2f, 0.2f), tableHeight + Randf(0.2f, 0.3f), Randf(0.4f, 0.6f));
		// }
		// else
		// {
			targetEffectorTranslations[ai] = initEffectorTranslation;
		// }
	}

	void ResetAgent(int a)
	{
		// This needs to be called first to avoid overwriting rigid bodies of updated cubes
		for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
			g_buffers->rigidBodies[i] = initBodies[i];

		g_buffers->rigidShapes.map();
		SampleTarget(a);
		SampleCube(a);
		g_buffers->rigidShapes.unmap();
		NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

		SetInitArmPose(a);		

		RLFlexEnv::ResetAgent(a);
	}

	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float distance = distanceOfCubeToGoal(a, state);
        if (!sparseRewards)
        {
            // L2 Dense reward
            rew = -distance;

        }
        else
        {
            float distance_threshold = 0.1f;
            rew = (distance > distance_threshold) ? -1.0f : 0.0f;
        }

        if(renderTarget)
        {
        	ColorTargetForReward(distance, a);
        }

	}

	void ColorTargetForReward(float distReward, int a)
   	{
		float x = max(exp(-10.0f*distReward), 0.f);
		g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;
	}

};

class RLFetchRopePeg : public RLFetchBase
{
public:
	vector<int> targetPegHolders;
	vector<float> pegMasses;
	vector<int> pegBodyIds;
	vector<int> pegHolderBodyIds;
	unordered_map<int, int> mapPegToAgent;
	vector<bool> pegContacts;
	vector<float> pegForces;
	vector<Vec3> targetPoses;

	// rope
	vector<vector<int>> ropeBodyIds;
	int ropeSegments = 12;
	const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
	const float linkLength = 0.01f;
	const float linkWidth = 0.005f;
	const float ropeDensity = 1.f;
	const bool connectRopeToRobot = true;

	RLFetchRopePeg()
	{
		tableHeight = .27f;
		controlType = eVelocity;
	}

    int computeNumObservations()
    {
        // 3 target xyz + 6 peg xyz, rot + 6 peg tra, rot vels + 2 peg/pegholder contact and force + rope segments
        return RLFetchBase::computeNumObservations() + 17 + 9 * ropeSegments;
    }

	void LoadChildEnv()
	{
		initEffectorTranslation = Vec3(0.f, 0.8, 0.5f);
		targetPegHolders.resize(mNumAgents);
		pegMasses.resize(mNumAgents);
		pegBodyIds.resize(mNumAgents);
		pegHolderBodyIds.resize(mNumAgents);
		pegContacts.resize(mNumAgents);
		pegForces.resize(mNumAgents);
		targetPoses.resize(mNumAgents);
		ropeBodyIds.resize(mNumAgents);
		for (int i = 0; i < mNumAgents; i++)
		{
			ropeBodyIds[i].resize(ropeSegments);
		}
	}

	void AddChildEnvBodies(int ai, Transform gt)
	{
		// Create target peg holder
		Mesh* pegHolderMesh = ImportMesh("../../data/peg_holder.obj");
		pegHolderMesh->Transform(ScaleMatrix(0.03));

		NvFlexTriangleMeshId pegHolderMeshId = CreateTriangleMesh(pegHolderMesh, 0.005f);

		NvFlexRigidShape pegHolderShape;
		NvFlexMakeRigidTriangleMeshShape(&pegHolderShape, g_buffers->rigidBodies.size(), pegHolderMeshId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
		pegHolderShape.filter = 0x0;
		pegHolderShape.group = 0;
		pegHolderShape.material.friction = 1.0f;
		pegHolderShape.thickness = 0.005f;
		g_buffers->rigidShapes.push_back(pegHolderShape);

		float pegHolderDensity = 100.0f;
		NvFlexRigidBody pegHolderBody;
		NvFlexMakeRigidBody(g_flexLib, &pegHolderBody, Vec3(), Quat(), &pegHolderShape, &pegHolderDensity, 1);
		pegHolderBodyIds[ai] = g_buffers->rigidBodies.size();
		g_buffers->rigidBodies.push_back(pegHolderBody);

		// Create rope		
		NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["l_gripper_finger_joint"]];

		Vec3 startPos = Vec3(0.f, .75f, 1.f) + robotPoses[ai];
		NvFlexRigidPose prevJoint;
		int lastRopeBodyIndex = 0;
		const float bendingCompliance = 1.e+2f;
		const float torsionCompliance = 1.f;

		for (int i = 0; i < ropeSegments; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0, Quat(.0f, .0f, .707f, .707f)));
			shape.filter = 0x0;
			shape.group = 0;
			shape.material.rollingFriction = 0.001f;
			shape.material.friction = 0.25f;
			shape.user = UnionCast<void*>(linkMaterial);

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, startPos + Vec3(0.0f, -(i*linkLength*2.f + linkLength), 0.0f), Quat(), &shape, &ropeDensity, 1);

			ropeBodyIds[ai][i] = g_buffers->rigidBodies.size();
			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

			if (i == 0 && !connectRopeToRobot)
			{
				prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -linkLength, .0f), Quat());
				continue;
			}

			NvFlexRigidJoint joint;
			if (i == 0)
			{
				NvFlexMakeFixedJoint(&joint, handLeft.body0, bodyIndex,
					NvFlexMakeRigidPose(Vec3(0.1f, .0f, .0f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi*0.5)),
					NvFlexMakeRigidPose(0, 0));
			}
			else
			{
				NvFlexMakeFixedJoint(&joint, bodyIndex - 1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(.0f, linkLength, .0f), Quat()));
			}
			joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
			joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
			joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

			g_buffers->rigidJoints.push_back(joint);
			lastRopeBodyIndex = bodyIndex;
			prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -linkLength, .0f), Quat());
		}

		// Peg
		float scale = 0.03f;
		Mesh* pegMesh = ImportMesh("../../data/cylinder.obj");
		pegMesh->Transform(ScaleMatrix(scale));

		NvFlexTriangleMeshId pegId = CreateTriangleMesh(pegMesh, 0.005f);
		NvFlexRigidShape pegShape;
		NvFlexMakeRigidTriangleMeshShape(&pegShape, g_buffers->rigidBodies.size(), pegId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
		pegShape.material.friction = 1.0f;
		pegShape.thickness = 0.005f;
		pegShape.filter = 0x0;
		pegShape.group = 0;
		pegShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(.9f, .9f, .3f)));
		g_buffers->rigidShapes.push_back(pegShape);

		float pegDensity = 0.1f;
		NvFlexRigidBody pegBody;
		NvFlexMakeRigidBody(g_flexLib, &pegBody, startPos + Vec3(0.0f, -float(ropeSegments + 2)*linkLength*2.f, .0f), Quat(), &pegShape, &pegDensity, 1);
		pegMasses[ai] = pegBody.mass;
		pegBodyIds[ai] = g_buffers->rigidBodies.size();
		mapPegToAgent.emplace(pegBodyIds[ai], ai);
		g_buffers->rigidBodies.push_back(pegBody);

		// Connecting peg to rope
		NvFlexRigidJoint joint;
		int bodyIndex = g_buffers->rigidBodies.size();
		NvFlexMakeFixedJoint(&joint, lastRopeBodyIndex, bodyIndex - 1, NvFlexMakeRigidPose(Vec3(.0f, -4.f*linkLength, .0f), Quat()), NvFlexMakeRigidPose(0, 0));

		joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
		joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
		joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

		g_buffers->rigidJoints.push_back(joint);

		SampleTarget(ai);
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
			targetPoses[ai] = Vec3(Randf(-0.4f, 0.4f), tableHeight + 0.02f, Randf(0.4f, 0.8f));
		}
		else
		{
			targetPoses[ai] = Vec3(-.2f, tableHeight + 0.02f, .6f);
		}

		NvFlexRigidPose newPose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
		NvFlexSetRigidPose(&g_buffers->rigidBodies[pegHolderBodyIds[ai]], &newPose);
	}

	void SampleRopePeg(int ai)
	{
		float pegMass;
		if (sampleInitStates)
		{
			pegMass = pegMasses[ai] * Randf(0.8f, 1.1f);
		}
		else
		{
			pegMass = pegMasses[ai];
		}

		g_buffers->rigidBodies[pegBodyIds[ai]].mass = pegMass;
	}

	void SetInitArmPose(int ai)
	{
		rolls[ai] = 0.f;
		pitches[ai] = 0.f;
		yaws[ai] = -kPi/2;

		targetEffectorTranslations[ai] = initEffectorTranslation;
	}

	void ExtractChildState(int a, float* state, int& ct)
	{	
		// 7-9 target xyz
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = targetPoses[a][i];
		}

		// 10-12 peg xyz
		Transform pegPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[pegBodyIds[a]], (NvFlexRigidPose*)&pegPoseWorld);
		Vec3 pegTraLocal = pegPoseWorld.p - robotPoses[a];
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = pegTraLocal[i];
		}

		// 13-15 peg rpy
		float r, p, y;
		quat2rpy(pegPoseWorld.q, r, p, y);
		state[ct++] = r;
		state[ct++] = p;
		state[ct++] = y;

		// 16-21 peg velocities
		NvFlexRigidBody pegBody = g_buffers->rigidBodies[pegBodyIds[a]];
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = pegBody.linearVel[i];
		}
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = pegBody.angularVel[i];
		}
				
		// 22-23 contact flag and force
		state[ct++] = pegContacts[a] ? 1.f : 0.f;
		state[ct++] = pegForces[a];

		// +9 * ropeSegments rope segment translation and rotations
		Transform segPoseWorld;
		for (int i = 0; i < ropeSegments; i++)
		{
			int segBodyId = ropeBodyIds[a][i];			
			NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
			Vec3 segTraLocal = segPoseWorld.p - robotPoses[a];
			NvFlexRigidBody segBody = g_buffers->rigidBodies[segBodyId];

			for (int j = 0; j < 3; j++)
			{
				state[ct++] = segTraLocal[j];
				state[ct++] = segBody.linearVel[j];
				state[ct++] = segBody.angularVel[j];
			}
		}
	}

	void ResetAgent(int a)
	{
		// This needs to be called first to avoid overwriting rigid bodies of updated cubes
		for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
			g_buffers->rigidBodies[i] = initBodies[i];

		g_buffers->rigidShapes.map();
		SampleTarget(a);
		SampleRopePeg(a);
		g_buffers->rigidShapes.unmap();
		NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

		SetInitArmPose(a);

		RLFlexEnv::ResetAgent(a);
	}

	void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float rewTarget = 0.f, rewVertical = 0.f, rewAction = 0.f, rewSolved = 0.f;
		bool solved = false;
		float targetReachWeight = 4.f;
		float verticalAngleWeight = 1.f;
		
		// reward for getting close to peg holder
		Vec3 targetTra = Vec3(state[7], state[8], state[9]);
		Vec3 pegTra = Vec3(state[10], state[11], state[12]);
		Vec3 pegToHole = pegTra - targetTra;
		float distTarget = Length(pegToHole);
		rewTarget = targetReachWeight * exp(-4.f * distTarget);
		
		// reward for maintaining verticality of peg
		Quat pegRot = rpy2quat(state[13], state[14], state[15]);
		// TODO(jaliang) : not sure why vertical is about 57 degrees and not 90...but this works.
		float pegAngleToVertical = 57.2957f - RadToDeg(Dot(GetBasisVector1(pegRot), Vec3(0.f, 1.f, 0.f)));
		rewVertical = verticalAngleWeight * exp(-pegAngleToVertical);
		
		float* prevAction = GetPrevAction(a);
		for (int i = 0; i < mNumActions; i++)
		{
			rewAction += -0.25f * Pow(action[i] - prevAction[i], 2) / (float)mNumActions;
		}
		
		if (abs(pegToHole[0]) < 0.003f && abs(pegToHole[2]) < 0.003f // within planar radius of the hole
			&& 0.03f < pegToHole[1] && pegToHole[1] < 0.04f // check height
			&& pegAngleToVertical < 5.f) // ensure vertical peg. 
		{
			solved = true;
			rewSolved = 5.f;
		}
		
		dead = solved || pegTra[1] < tableHeight;		
		rew = rewTarget + rewVertical + rewAction + rewSolved;
	}

	void ClearContactInfo()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			pegContacts[a] = false;
			pegForces[a] = 0.f;
		}
	}

	void CheckPegContact(int body0, int body1, int& ai)
	{
		// check if body0 is a peg 
		ai = -1;
		if (mapPegToAgent.find(body0) != mapPegToAgent.end())
		{
			ai = mapPegToAgent.at(body0);
			// check if body1 is the peg holder corresponding to agent id. 
			if (body1 != pegHolderBodyIds[ai])
			{
				ai = -1;
			}
		}
	}

	void FinalizeContactInfo()
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

		float forceScale = 0.1f;

		NvFlexRigidContact* ct = &(rigidContacts[0]);

		int ai;
		for (int i = 0; i < numContacts; ++i)
		{
			ai = -1;

			CheckPegContact(ct[i].body0, ct[i].body1, ai);
			if (ai == -1)
			{
				CheckPegContact(ct[i].body1, ct[i].body0, ai);
			}

			if (ai != -1)
			{
				pegContacts[ai] = true;
				pegForces[ai] += forceScale * ct[i].lambda;
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}
};

class RLFetchRopeSimple: public RLFetchBase
{
public:

	// rope
	vector<vector<int>> ropeBodyIds;
	int ropeSegments;
	const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
	const float linkLength = 0.01f;
	const float linkWidth = 0.01f;
	const float ropeDensity = 1000.0f;
	const float bendingCompliance = 1.e+2f;
	const float torsionCompliance = 1.e-6f;
	const float ropeFriction = 0.7f;

	Vec3 startPos, startPos_;
	Vec3 redColor, greenColor;
	vector<vector<int>> linkRenderMaterials;

	vector<float> forceFingerLeft;
	vector<float> forceFingerRight;
	float graspForceThresh;

	unordered_map<int, int> mapFingerToAgent;
	unordered_map<int, bool> mapFingerToSide;

	vector<bool> ropeContacts;
	vector<float> ropeForces;
	float targetRopeTheta;
	float targetRopeStart;

	RLFetchRopeSimple()
	{
		controlType = eInverseDynamics;
		ropeSegments =  GetJsonVal(g_sceneJson, "RopeSegments", 16);
		targetRopeTheta = GetJsonVal(g_sceneJson, "TargetRopeTheta", 0.05f);
		targetRopeStart = GetJsonVal(g_sceneJson, "TargetRopeStart", 0.05f);
		startPos_ = Vec3(-0.1f, tableHeight + linkWidth * 0.5f + 0.02f, 0.5f);
		redColor = Vec3(1.0f, 0.02f, 0.06f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);
		initEffectorTranslation = Vec3(0.f, tableHeight + 0.2f, 0.5f);
	}

    int computeNumObservations()
    {
        // +2 finger contacts + 3(xyz,rpy) * rope segments
        if(doGripperControl)    return RLFetchBase::computeNumObservations() + 3 * ropeSegments + 2;
        else    return RLFetchBase::computeNumObservations() + 3 * ropeSegments;
    }

	void LoadChildEnv()
	{
		ropeBodyIds.resize(mNumAgents);
		linkRenderMaterials.resize(mNumAgents);
		
		for (int i = 0; i < mNumAgents; i++)
		{
			ropeBodyIds[i].resize(ropeSegments);
			linkRenderMaterials[i].resize(ropeSegments);
		}

		forceFingerLeft.resize(mNumAgents);
		forceFingerRight.resize(mNumAgents);
		forceLeft.resize(mNumAgents);
		forceRight.resize(mNumAgents);
		ropeContacts.resize(mNumAgents);
		ropeForces.resize(mNumAgents); 
	}

	void AddChildEnvBodies(int ai, Transform gt)
	{
		// initialize maps for robot fingers
		mapFingerToAgent.emplace(urdf->rigidNameMap["l_gripper_finger_link"], ai);
		mapFingerToAgent.emplace(urdf->rigidNameMap["r_gripper_finger_link"], ai);
		mapFingerToSide.emplace(urdf->rigidNameMap["l_gripper_finger_link"], true);
		mapFingerToSide.emplace(urdf->rigidNameMap["r_gripper_finger_link"], false);

		startPos = startPos_ + robotPoses[ai];

		NvFlexRigidPose prevJoint;
		for (int i=0; i < ropeSegments; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0,Quat(0.f, 0.707f, 0.f, 0.707f)));
			shape.filter = 0;
			shape.group = 0;
			shape.material.rollingFriction = 0.001f;
			shape.material.friction = ropeFriction;
			linkRenderMaterials[ai][i] = linkMaterial;
			shape.user = UnionCast<void*>(linkRenderMaterials[ai][i]);
			
			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, startPos + Vec3(0.0f, 0.0f, i*linkLength*2.0f + linkLength), Quat(), &shape, &ropeDensity, 1);
			ropeBodyIds[ai][i] = g_buffers->rigidBodies.size();

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

			if (i > 0)
			{
				NvFlexRigidJoint joint;				
				NvFlexMakeFixedJoint(&joint, bodyIndex-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, -linkLength), Quat()));

				joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

				g_buffers->rigidJoints.push_back(joint);
			}

			prevJoint = NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, linkLength), Quat());

		}

		SampleTarget(ai);
	}

	void SampleTarget(int ai)
	{
		return;
	}

	void SampleRope(int ai)
	{
		Quat quat = Quat();
		float theta = 0.f;;

		if(sampleInitStates)
		{
			startPos_ = Vec3(Randf(-targetRopeStart, targetRopeStart), tableHeight + linkWidth * 0.5f + 0.02f, 0.5f);
			theta = Randf(-targetRopeTheta, targetRopeTheta);
			quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * theta);
		}

		startPos = startPos_ + robotPoses[ai];

		for (int i=0; i < ropeSegments; ++i)
		{
			NvFlexRigidPose pose = NvFlexMakeRigidPose(startPos + Vec3((i*linkLength*2.0f + linkLength)*sin(theta), 0.0f, (i*linkLength*2.0f + linkLength)*cos(theta)), quat);
			NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
		}

	}

	void SetInitArmPose(int ai)
	{
		if(doGripperControl)    fingerWidths[ai] = fingerWidthMax;
		else    fingerWidths[ai] = fingerWidthMin;

		rolls[ai] = 0.f;
		pitches[ai] = 0.f;
		yaws[ai] = -kPi/2;

		targetEffectorTranslations[ai] = initEffectorTranslation;
	}

	void ExtractChildState(int a, float* state, int& ct)
	{	
		// +3 * ropeSegments rope segment translation
		Transform segPoseWorld;
		for (int i = 0; i < ropeSegments; i++)
		{
			int segBodyId = ropeBodyIds[a][i];			
			NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
			Vec3 segTraLocal = segPoseWorld.p - robotPoses[a];
			// NvFlexRigidBody segBody = g_buffers->rigidBodies[segBodyId];

			for (int j = 0; j < 3; j++)
			{
				if(relativeTarget)
				{	
					state[ct++] = segTraLocal[j] - state[j];
				}
				else
				{
					state[ct++] = segTraLocal[j];
				}
				
			}
		}

		// finger contacts
		if(doGripperControl)
		{
			state[ct++] = 0.1f * forceFingerLeft[a];
			state[ct++] = 0.1f * forceFingerRight[a];
		}
	}

	void ResetAgent(int a)
	{
		// This needs to be called first to avoid overwriting rigid bodies of updated cubes
		for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
			g_buffers->rigidBodies[i] = initBodies[i];

		g_buffers->rigidShapes.map();
		SampleTarget(a);
		SampleRope(a);
		g_buffers->rigidShapes.unmap();
		NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

		SetInitArmPose(a);

		RLFlexEnv::ResetAgent(a);
	}

   	void ColorTargetForReward(float distReward, int a)
   	{
		float x = max(exp(-12.0f * distReward), 0.f);
		g_renderMaterials[linkRenderMaterials[a][0]].frontColor = x * greenColor + (1.f - x) * redColor;
		g_renderMaterials[linkRenderMaterials[a][ropeSegments-1]].frontColor = x * greenColor + (1.f - x) * redColor;
    }

	void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		int firstRopeBodyId = ropeBodyIds[a][0];
		int lastRopeBodyId = ropeBodyIds[a][ropeSegments-1];
		
		Transform firstsegPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[firstRopeBodyId], (NvFlexRigidPose*)&firstsegPoseWorld);
		Vec3 firstsegTraLocal = firstsegPoseWorld.p - robotPoses[a];
		// NvFlexRigidBody startsegBody = g_buffers->rigidBodies[firstRopeBodyId];

		Transform lastsegPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[lastRopeBodyId], (NvFlexRigidPose*)&lastsegPoseWorld);
		Vec3 lastsegTraLocal = lastsegPoseWorld.p - robotPoses[a];
		// NvFlexRigidBody lastsegBody = g_buffers->rigidBodies[lastRopeBodyId];
		float distance = Length(lastsegTraLocal - firstsegTraLocal);

		if(!sparseRewards)
		{
			rew = -distance;
		}
		else
		{
            float distance_threshold = 0.1f;
            rew = (distance > distance_threshold) ? -1.f : 0.f;
		}
        if(renderTarget)
        {
        	ColorTargetForReward(distance, a);
        }
	}

	void ClearContactInfo()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			ropeContacts[a] = false;
			ropeForces[a] = 0.f;
		}
	}
	void CheckFingerRopeContact(int body0, int body1, int& ai)
	{	
		// check if body0 is a finger 
		ai = -1;
		if (mapFingerToAgent.find(body0) != mapFingerToAgent.end())
		{	
			ai = mapFingerToAgent.at(body0);
			// check if body1 is the cube corresponding to agent id. 
			for(int i = 0; i < ropeSegments; i++)			
			{	
				if (body1 == ropeBodyIds[ai][i])
				{	
					ai = -1;
					return;
				}
			}
		}
	}

	void FinalizeContactInfo()
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

		float forceScale = 0.1f;
		NvFlexRigidContact* ct = &(rigidContacts[0]);

		int ai;
		for (int i = 0; i < numContacts; ++i)
		{
			ai = -1;
			CheckFingerRopeContact(ct[i].body0, ct[i].body1, ai);
			if (ai == -1)
			{	
				CheckFingerRopeContact(ct[i].body1, ct[i].body0, ai);
			}

			if (ai != -1)
			{   
				ropeContacts[ai] = true;
				ropeForces[ai] += forceScale * ct[i].lambda;
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	void DoStats()
	{
		if (doStats)
		{
			int numSamples = 200;

			int start = Max(int(forceLeft.size()) - numSamples, 0);
			int end = Min(start + numSamples, int(forceLeft.size()));

			// convert from position changes to forces
			float units = -1.0f / Sqr(g_dt / g_numSubsteps);

			float height = 50.0f;
			float maxForce = 50.0f;  // What is maxForce?

			float dx = 1.0f;
			float sy = height / maxForce;

			float lineHeight = 10.0f;

			float rectMargin = 10.0f;
			float rectWidth = dx * numSamples + rectMargin * 4.0f;

			float x = float(g_screenWidth) - rectWidth - 20.0f;
			float y = 300.0f;

			DrawRect(x, y - height - rectMargin, rectWidth, 2.0f * height + rectMargin * 3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));

			x += rectMargin * 3.0f;

			DrawImguiString(int(x + dx * float(numSamples)), int(y + 55.0f), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

			DrawLine(x, y, x + float(numSamples) * dx, y, 1.0f, Vec3(1.0f));
			DrawLine(x, y - 50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

			float margin = 5.0f;

			DrawImguiString(int(x - margin), int(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
			DrawImguiString(int(x - margin), int(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
			DrawImguiString(int(x - margin), int(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

			for (int i = start; i < end - 1; ++i)
			{
				float fl0 = Clamp(forceLeft[i] * units, -maxForce, maxForce) * sy;
				float fr0 = Clamp(forceRight[i] * units, -maxForce, maxForce) * sy;

				float fl1 = Clamp(forceLeft[i + 1] * units, -maxForce, maxForce) * sy;
				float fr1 = Clamp(forceRight[i + 1] * units, -maxForce, maxForce) * sy;

				DrawLine(x, y + fl0, x + dx, y + fl1, 1.0f, Vec3(1.0f, 0.0f, 0.0f));
				DrawLine(x, y + fr0, x + dx, y + fr1, 1.0f, Vec3(0.0f, 1.0f, 0.0f));

				x += dx;
			}
		}
	}
};

class RLFetchRopePush: public RLFetchBase
{
public:

	// rope
	vector<vector<int>> ropeBodyIds;
	int ropeSegments;
	const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
	const float linkLength = 0.015f;
	const float linkWidth = 0.015f;
	const float ropeDensity = 1000.0f;
	const float bendingCompliance = 1.e+4f;
	const float torsionCompliance = 1.e-2f;
	const float ropeFriction = 0.1f;

	Vec3 startPos, startPos_;
	Vec3 redColor, greenColor;
	vector<vector<int>> linkRenderMaterials;

	vector<float> forceFingerLeft;
	vector<float> forceFingerRight;
	float graspForceThresh;

	unordered_map<int, int> mapFingerToAgent;
	unordered_map<int, bool> mapFingerToSide;

	vector<bool> ropeContacts;
	vector<float> ropeForces;
	
	vector<vector<int>> targetSphere;
	vector<vector<Vec3>> targetPoses;
	vector<vector<Quat>> targetQuat;

	string targetRopeShape;
	string defRopeShape = "I";
	float targetRopeTheta;

	RLFetchRopePush()
	{
		controlType = eInverseDynamics;
		ropeSegments =  GetJsonVal(g_sceneJson, "RopeSegments", 24);
		targetRopeShape = GetJsonVal(g_sceneJson, "TargetRopeShape", defRopeShape);
		targetRopeTheta = GetJsonVal(g_sceneJson, "TargetRopeTheta", 0.05f);
		startPos_ = Vec3(0.0f, tableHeight + linkWidth * 0.5f + 0.02f, 0.5f);
		redColor = Vec3(1.0f, 0.02f, 0.06f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);
		initEffectorTranslation = Vec3(0.0f, tableHeight + 0.15f, 0.6f);
	}

    int computeNumObservations()
    {
        // +2 finger contacts + (3+3)(xyz) * rope segments
        if(doGripperControl)    return RLFetchBase::computeNumObservations() + 6 * ropeSegments + 2;
        else    return RLFetchBase::computeNumObservations() + 6 * ropeSegments;
    }

	void LoadChildEnv()
	{
		ropeBodyIds.resize(mNumAgents);
		linkRenderMaterials.resize(mNumAgents);
		targetPoses.resize(mNumAgents);
		targetSphere.resize(mNumAgents);
		targetQuat.resize(mNumAgents);
		
		for (int i = 0; i < mNumAgents; i++)
		{
			ropeBodyIds[i].resize(ropeSegments);
			linkRenderMaterials[i].resize(ropeSegments);
			targetPoses[i].resize(ropeSegments);
			targetSphere[i].resize(ropeSegments);
			targetQuat[i].resize(ropeSegments);
		}

		forceFingerLeft.resize(mNumAgents);
		forceFingerRight.resize(mNumAgents);
		forceLeft.resize(mNumAgents);
		forceRight.resize(mNumAgents);
		ropeContacts.resize(mNumAgents);
		ropeForces.resize(mNumAgents); 
	}

	void AddChildEnvBodies(int ai, Transform gt)
	{
		// initialize maps for robot fingers
		mapFingerToAgent.emplace(urdf->rigidNameMap["l_gripper_finger_link"], ai);
		mapFingerToAgent.emplace(urdf->rigidNameMap["r_gripper_finger_link"], ai);
		mapFingerToSide.emplace(urdf->rigidNameMap["l_gripper_finger_link"], true);
		mapFingerToSide.emplace(urdf->rigidNameMap["r_gripper_finger_link"], false);

		startPos = startPos_ + robotPoses[ai];

		NvFlexRigidPose prevJoint;
		for (int i=0; i < ropeSegments; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0,Quat(0.f, 0.707f, 0.f, 0.707f)));
			shape.filter = 0;
			shape.group = 0;
			shape.material.rollingFriction = 0.001f;
			shape.material.friction = ropeFriction;
			linkRenderMaterials[ai][i] = linkMaterial;
			shape.user = UnionCast<void*>(linkRenderMaterials[ai][i]);
			
			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, startPos + Vec3(0.0f, 0.0f, i*linkLength*2.0f + linkLength), Quat(), &shape, &ropeDensity, 1);
			ropeBodyIds[ai][i] = g_buffers->rigidBodies.size();

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

			if (i > 0)
			{
				NvFlexRigidJoint joint;				
				NvFlexMakeFixedJoint(&joint, bodyIndex-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, -linkLength), Quat()));

				joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

				g_buffers->rigidJoints.push_back(joint);
			}

			prevJoint = NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, linkLength), Quat());

		}

		if(renderTarget)
		{
			for (int i=0; i < ropeSegments; ++i)
			{
				NvFlexRigidShape targetShape;
				NvFlexMakeRigidCapsuleShape(&targetShape, -1, linkWidth, linkLength, NvFlexMakeRigidPose(0, Quat(0.f, 0.707f, 0.f, 0.707f)));
				int renderMaterial = AddRenderMaterial(redColor);
				linkRenderMaterials[ai][i] = renderMaterial;
				targetShape.user = UnionCast<void*>(renderMaterial);
				targetShape.group = 1;
				targetSphere[ai][i] = g_buffers->rigidShapes.size();
				g_buffers->rigidShapes.push_back(targetShape);
			}
		}

		SampleTarget(ai);
	}

	void SampleTarget(int ai)
	{
		float shift_x;
		float shift_z;

		Transform segPoseWorld;
		if (sampleInitStates)
		{	
			shift_x = Randf(-0.15f, 0.15f);
			shift_z = Randf(-0.2f, -0.1f);
		}
		else
		{
			shift_x = 0.15f;
			shift_z = 0.1f;
		}

		for(int i=0; i<ropeSegments; i++)
		{	
			int segBodyId = ropeBodyIds[ai][i];			
			NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
			Vec3 segTraLocal = segPoseWorld.p - robotPoses[ai];
			Quat segQuatLocal = segPoseWorld.q;
			
			targetPoses[ai][i] = segTraLocal + Vec3(shift_x, 0.f, shift_z);
			targetQuat[ai][i] = segQuatLocal;
		}

		Vec3 targetstartPos = startPos + Vec3(shift_x, 0.f, shift_z);
		float theta = Randf(-targetRopeTheta, targetRopeTheta);

		if(targetRopeShape == "I")		CreateI(ai, targetstartPos, theta, Quat(0.f, 0.707f, 0.f, 0.707f), true);
		else if(targetRopeShape == "L") 	CreateL(ai, targetstartPos, theta, Quat(0.f, 0.707f, 0.f, 0.707f), true);
		else if(targetRopeShape == "V") 	CreateV(ai, targetstartPos, theta, Quat(0.f, 0.707f, 0.f, 0.707f), true);	
		else if(targetRopeShape == "U") 	CreateU(ai, targetstartPos, theta, Quat(0.f, 0.707f, 0.f, 0.707f), true);	
		else if(targetRopeShape == "O") 	CreateO(ai, targetstartPos, Quat(0.f, 0.707f, 0.f, 0.707f), true);	
		else
		{
			cout << string(50, '-') << endl;
			cout << "Rope shape should either be \"I\", \"L\", \"V\", \"O\" or \"U\". Default shape of \"I\" has been chosen to stop the program from crashing." << endl;
			cout << string(50, '-') << endl;
			CreateI(ai, startPos, theta, Quat(0.f, 0.707f, 0.f, 0.707f), true);
		}

		return;
	}

	void CreateI(int ai, Vec3 startPos, float theta, Quat quat, bool target=false)
	{
		NvFlexRigidPose pose;
		for (int i=0; i < ropeSegments; ++i)
		{
			pose = NvFlexMakeRigidPose(startPos + Vec3((i*linkLength*2.0f + linkLength)*sin(theta), 0.0f, (i*linkLength*2.0f + linkLength)*cos(theta)), quat);
			if(target)
			{
				g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;	
				targetPoses[ai][i] = pose.p;	
			}
			else
			{
				NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
			}
		}
	}

	void CreateO(int ai, Vec3 startPos, Quat quat, bool target=false)
	{
		NvFlexRigidPose pose;
		float radius = 0.1f; //(linkLength * ropeSegments) / (2 * kPi);
		float theta = (2*kPi/ropeSegments);

		// cout << "target: " << target << endl;
		for (int i=0; i < ropeSegments; ++i)
		{
			pose = NvFlexMakeRigidPose(startPos + Vec3(radius*cos(i*theta), 0.0f, radius*sin(i*theta)), quat);
			if(target)
			{
				g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;
				targetPoses[ai][i] = pose.p;	
			}
			else
			{
				NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
			}
			// printVec3(pose.p);
		}
	}

	void CreateL(int ai, Vec3 startPos, float theta, Quat quat, bool target=false)
	{
		NvFlexRigidPose pose;
		int pivotSegment = (int)ropeSegments/2;
		Vec3 pivotLoc;
		bool nextSegment = false;

		for (int i=0; i < ropeSegments; ++i)
		{
			if(!nextSegment)
			{
				pose = NvFlexMakeRigidPose(startPos + Vec3((i*linkLength*2.0f + linkLength)*sin(theta), 0.0f, (i*linkLength*2.0f + linkLength)*cos(theta)), quat);
				if(target)
				{
					g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;	
					targetPoses[ai][i] = pose.p;	
				}
				else
				{
					NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
				}
			}

			else
			{
				pose = NvFlexMakeRigidPose(pivotLoc + Vec3(((i-pivotSegment)*linkLength*2.0f + linkLength)*sin(kPi/2 - theta), 0.0f, ((i-pivotSegment)*linkLength*2.0f + linkLength)*cos(kPi/2 - theta)), quat);
				if(target)
				{
					g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;	
					targetPoses[ai][i] = pose.p;	
				}
				else
				{
					NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
				}
			}			

			if(i == pivotSegment)
			{
				pivotLoc = pose.p;
				nextSegment = true;
			}
		}
	}

	void CreateU(int ai, Vec3 startPos, float theta, Quat quat, bool target=false)
	{
		NvFlexRigidPose pose;
		int pivotSegment = (int)ropeSegments/3;
		int pivotSegment2 = 2 * (int)ropeSegments/3;
		Vec3 pivotLoc, pivotLoc2;
		bool firstSegment = true, secondSegment = false;

		for (int i=0; i < ropeSegments; ++i)
		{	
			if(firstSegment)
			{
				pose = NvFlexMakeRigidPose(startPos + Vec3((i*linkLength*2.0f + linkLength)*sin(theta), 0.0f, (i*linkLength*2.0f + linkLength)*cos(theta)), quat);
				if(target)
				{
					g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;
					targetPoses[ai][i] = pose.p;	
				}
				else
				{
					NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
				}
			}

			else if(secondSegment)
			{
				pose = NvFlexMakeRigidPose(pivotLoc + Vec3(((i-pivotSegment)*linkLength*2.0f + linkLength)*cos(theta), 0.0f, ((i-pivotSegment)*linkLength*2.0f + linkLength)*sin(theta)), quat);
				if(target)
				{
					g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;	
					targetPoses[ai][i] = pose.p;	
				}
				else
				{
					NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
				}
			}

			else
			{
				pose = NvFlexMakeRigidPose(pivotLoc2 + Vec3(((i-pivotSegment2)*linkLength*2.0f + linkLength)*sin(kPi - theta), 0.0f, ((i-pivotSegment2)*linkLength*2.0f + linkLength)*cos(kPi - theta)), quat);
				if(target)
				{
					g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;	
					targetPoses[ai][i] = pose.p;	
				}
				else
				{
					NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
				}
			}			

			if(i == pivotSegment)
			{
				pivotLoc = pose.p;
				firstSegment = false;
				secondSegment = true;
			}

			if(i == pivotSegment2)
			{
				pivotLoc2 = pose.p;
				secondSegment = false;
			}
		}
	}

	void CreateV(int ai, Vec3 startPos, float theta, Quat quat, bool target=false)
	{
		NvFlexRigidPose pose;
		int pivotSegment = (int)ropeSegments/2;
		Vec3 pivotLoc;
		bool nextSegment = false;

		for (int i=0; i < ropeSegments; ++i)
		{
			if(!nextSegment)
			{
				pose = NvFlexMakeRigidPose(startPos + Vec3((i*linkLength*2.0f + linkLength)*sin(theta), 0.0f, (i*linkLength*2.0f + linkLength)*cos(theta)), quat);
				if(target)
				{
					g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;	
					targetPoses[ai][i] = pose.p;	
				}
				else
				{
					NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
				}
			}

			else
			{
				pose = NvFlexMakeRigidPose(pivotLoc + Vec3(((i-pivotSegment)*linkLength*2.0f + linkLength)*sin(kPi - theta), 0.0f, ((i-pivotSegment)*linkLength*2.0f + linkLength)*cos(kPi - theta)), quat);
				if(target)
				{
					g_buffers->rigidShapes[targetSphere[ai][i]].pose = pose;	
					targetPoses[ai][i] = pose.p;	
				}
				else
				{
					NvFlexSetRigidPose(&g_buffers->rigidBodies[ropeBodyIds[ai][i]], &pose);
				}
			}			

			if(i == pivotSegment)
			{
				pivotLoc = pose.p;
				nextSegment = true;
			}
		}
	}

	void SampleRope(int ai)
	{
		Quat quat = Quat();
		float theta = 0.f;;

		if(sampleInitStates)
		{
			startPos_ = Vec3(Randf(-0.1f, 0.1f), tableHeight + linkWidth * 0.5f + 0.02f, 0.6f);
			theta = Randf(-0.05f, 0.05f);
			quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * theta);
		}

		startPos = startPos_ + robotPoses[ai];
		
		CreateI(ai, startPos, theta, quat, false);
	}

	void SetInitArmPose(int ai)
	{
		if(doGripperControl)    fingerWidths[ai] = 0.017f; //fingerWidthMax;
		else    fingerWidths[ai] = fingerWidthMin;

		rolls[ai] = 0.f;
		pitches[ai] = 0.f;
		yaws[ai] = -kPi/2;

		Transform segPoseWorld;
		int segBodyId = ropeBodyIds[ai][0];			
		NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
		Vec3 segTraLocal = segPoseWorld.p - robotPoses[ai];

		targetEffectorTranslations[ai] = segTraLocal + Vec3(0.f, 0.1f, 0.f);
	}

	void ExtractChildState(int a, float* state, int& ct)
	{	
		// +6 * ropeSegments rope segment translation
		Transform segPoseWorld;
		for (int i = 0; i < ropeSegments; i++)
		{
			int segBodyId = ropeBodyIds[a][i];			
			NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
			Vec3 segTraLocal = segPoseWorld.p - robotPoses[a];
			// NvFlexRigidBody segBody = g_buffers->rigidBodies[segBodyId];

			for (int j = 0; j < 3; j++)
			{
				if(relativeTarget)
				{	
					state[ct++] = segTraLocal[j] - state[j];
				}
				else
				{
					state[ct++] = segTraLocal[j];
				}
				
			}
		}

		// finger contacts
		if(doGripperControl)
		{
			state[ct++] = 0.1f * forceFingerLeft[a];
			state[ct++] = 0.1f * forceFingerRight[a];
		}

		for (int i = 0; i < ropeSegments; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				// RelativeTarget case has not been dealt with here
				state[ct++] = targetPoses[a][i][j];
				
			}
		}
	}

	void ResetAgent(int a)
	{
		// This needs to be called first to avoid overwriting rigid bodies of updated cubes
		for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
			g_buffers->rigidBodies[i] = initBodies[i];

		g_buffers->rigidShapes.map();
		SampleRope(a);
		SampleTarget(a);
		g_buffers->rigidShapes.unmap();
		NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

		SetInitArmPose(a);

		RLFlexEnv::ResetAgent(a);
	}

   	void ColorTargetForReward(float dist, int a)
   	{
		float x = Max(-log(4*dist), 0.f);
		g_renderMaterials[linkRenderMaterials[a][0]].frontColor = x * greenColor + (1.f - x) * redColor;
		g_renderMaterials[linkRenderMaterials[a][ropeSegments-1]].frontColor = x * greenColor + (1.f - x) * redColor;
    }

	void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float distance = 0.f;

		if(!sparseRewards)
		{
            for(int i=0; i<ropeSegments; ++i)
            {
				int RopeBodyId = ropeBodyIds[a][i];
            	Transform segPoseWorld;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[RopeBodyId], (NvFlexRigidPose*)&segPoseWorld);
				Vec3 segTraLocal = segPoseWorld.p - robotPoses[a];

				distance += Length(segTraLocal - targetPoses[a][i]);
            }
            rew = -distance;
		}
		else
		{
            float distance_threshold = 0.025f;
            // rew = (distance > distance_threshold) ? -1.f : 0.f;
            for(int i=0; i<ropeSegments; ++i)
            {
				int RopeBodyId = ropeBodyIds[a][i];
            	Transform segPoseWorld;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[RopeBodyId], (NvFlexRigidPose*)&segPoseWorld);
				Vec3 segTraLocal = segPoseWorld.p - robotPoses[a];
				
				distance += Length(segTraLocal - targetPoses[a][i]);
            }
            distance = distance/static_cast<float>(ropeSegments);
            rew = (distance > distance_threshold) ? -1.f : 0.f;

		}
        if(renderTarget)
        {
        	if(sparseRewards)	ColorTargetForReward(-rew, a);
			else 	ColorTargetForReward(distance, a);
        }
	}

	void ClearContactInfo()
	{
		for (int a = 0; a < mNumAgents; a++)
		{
			ropeContacts[a] = false;
			ropeForces[a] = 0.f;
		}
	}
	void CheckFingerRopeContact(int body0, int body1, int& ai)
	{	
		// check if body0 is a finger 
		ai = -1;
		if (mapFingerToAgent.find(body0) != mapFingerToAgent.end())
		{	
			ai = mapFingerToAgent.at(body0);
			// check if body1 is the cube corresponding to agent id. 
			for(int i = 0; i < ropeSegments; i++)			
			{	
				if (body1 == ropeBodyIds[ai][i])
				{	
					ai = -1;
					return;
				}
			}
		}
	}

	void FinalizeContactInfo()
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

		float forceScale = 0.1f;
		NvFlexRigidContact* ct = &(rigidContacts[0]);

		int ai;
		for (int i = 0; i < numContacts; ++i)
		{
			ai = -1;
			CheckFingerRopeContact(ct[i].body0, ct[i].body1, ai);
			if (ai == -1)
			{	
				CheckFingerRopeContact(ct[i].body1, ct[i].body0, ai);
			}

			if (ai != -1)
			{   
				ropeContacts[ai] = true;
				ropeForces[ai] += forceScale * ct[i].lambda;
			}
		}
		rigidContacts.unmap();
		rigidContactCount.unmap();
	}

	void DoStats()
	{
		if (doStats)
		{
			int numSamples = 200;

			int start = Max(int(forceLeft.size()) - numSamples, 0);
			int end = Min(start + numSamples, int(forceLeft.size()));

			// convert from position changes to forces
			float units = -1.0f / Sqr(g_dt / g_numSubsteps);

			float height = 50.0f;
			float maxForce = 50.0f;  // What is maxForce?

			float dx = 1.0f;
			float sy = height / maxForce;

			float lineHeight = 10.0f;

			float rectMargin = 10.0f;
			float rectWidth = dx * numSamples + rectMargin * 4.0f;

			float x = float(g_screenWidth) - rectWidth - 20.0f;
			float y = 300.0f;

			DrawRect(x, y - height - rectMargin, rectWidth, 2.0f * height + rectMargin * 3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));

			x += rectMargin * 3.0f;

			DrawImguiString(int(x + dx * float(numSamples)), int(y + 55.0f), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

			DrawLine(x, y, x + float(numSamples) * dx, y, 1.0f, Vec3(1.0f));
			DrawLine(x, y - 50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

			float margin = 5.0f;

			DrawImguiString(int(x - margin), int(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
			DrawImguiString(int(x - margin), int(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
			DrawImguiString(int(x - margin), int(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

			for (int i = start; i < end - 1; ++i)
			{
				float fl0 = Clamp(forceLeft[i] * units, -maxForce, maxForce) * sy;
				float fr0 = Clamp(forceRight[i] * units, -maxForce, maxForce) * sy;

				float fl1 = Clamp(forceLeft[i + 1] * units, -maxForce, maxForce) * sy;
				float fr1 = Clamp(forceRight[i + 1] * units, -maxForce, maxForce) * sy;

				DrawLine(x, y + fl0, x + dx, y + fl1, 1.0f, Vec3(1.0f, 0.0f, 0.0f));
				DrawLine(x, y + fr0, x + dx, y + fr1, 1.0f, Vec3(0.0f, 1.0f, 0.0f));

				x += dx;
			}
		}
	}
};

class RLFetchReachMultiGoal : public RLFetchReach, RLMultiGoalEnv
{
public:
    
    RLFetchReachMultiGoal()
	{
		mNumExtras = _GetNumExtras();
	}

	int computeNumObservations()
    {
        return RLFetchBase::computeNumObservations(); // no target
    }

	virtual int GetNumGoals()
	{
		return 3;
	}
	void PopulateExtra(int ai, float* extra)
	{
		_PopulateExtra(ai, extra);
	}
	void GetDesiredGoal(int ai, float* goal)
	{
		// Transform pose;
		// NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		// Vec3 effectorPose = pose.p - robotPoses[ai];

		for (int i = 0; i < 3; ++i)
		{
			// if(relativeTarget)
			// {
			// 	goal[i] = targetPoses[ai][i] - effectorPose[i];
			// }
			// else
			// {
				goal[i] = targetPoses[ai][i];
			// }
		}
	}
	void GetAchievedGoal(int ai, float* goal)
	{
		Transform pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		Vec3 effectorPose = pose.p - robotPoses[ai];
		for (int i = 0; i < 3; i++)
		{
			// if(relativeTarget)
			// {
			// 	goal[i] = 0.f;
			// }
			// else
			// {
				goal[i] = effectorPose[i];
			// }
		}
	}

	void ExtractChildState(int ai, float* state, int& ct)
	{   
		return;
	}

	Vec3 getTargetLocation(int ai, float* state)
    {
        if(relativeTarget)
		{
			return targetPoses[ai] - Vec3(state[0], state[1], state[2]);
		}
		else
		{
			return targetPoses[ai];
		}
    }

};

class RLFetchPushMultiGoal : public RLFetchPush, RLMultiGoalEnv
{
public:
    
    RLFetchPushMultiGoal()
	{
		mNumExtras = RLMultiGoalEnv::_GetNumExtras();
	}

	int computeNumObservations()
    {	// + 3 for object (no target)
        return RLFetchBase::computeNumObservations() + 3;
    }

	virtual int GetNumGoals()
	{
		return 3;
	}
	void PopulateExtra(int ai, float* extra)
	{
		RLMultiGoalEnv::_PopulateExtra(ai, extra);
	}
	void GetDesiredGoal(int ai, float* goal)
	{
		// Transform pose;
		// NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		// Vec3 effectorPose = pose.p - robotPoses[ai];

		for (int i = 0; i < 3; ++i)
		{
			// if(relativeTarget)
			// {
			// 	goal[i] = 0.f;
			// }
			// else
			// {
				goal[i] = targetPoses[ai][i];
			// }
		}
	}
	void GetAchievedGoal(int ai, float* goal)
	{
		// The achieved goal for this environment is the location of the cube which is different from the end-effector in Reach MultiGoal
		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];

		// Transform pose;
		// NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		// Vec3 effectorPose = pose.p - robotPoses[ai];

		for (int i = 0; i < 3; i++)
		{
			// if(relativeTarget)
			// {
			// 	goal[i] = cubeTraLocal[i] - targetPoses[ai][i]; // - effectorPose[i];
			// }
			// else
			// {
				goal[i] = cubeTraLocal[i];
			// }
		}
	}

	void ExtractChildState(int ai, float* state, int& ct)
	{   
		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];

		for (int i = 0; i < 3; ++i)
		{
			if(relativeTarget)
			{
				state[ct++] = cubeTraLocal[i] - state[i];
			}
			else
			{
				state[ct++] = cubeTraLocal[i];  
			}
		}
	}

	Vec3 getTargetLocation(int ai, float* state)
    {
        if(relativeTarget)
		{
			return targetPoses[ai] - getCubeLocation(ai, state);
		}
		else
		{
			return targetPoses[ai];
		}
    }

    Vec3 getCubeLocation(int ai, float* state)
    {	
    	// Returning only the absolute location because only the absolute position of cube is required by those functions calling this function
    	Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 Cube = cubePoseWorld.p - robotPoses[ai];

		return Cube;
    }

};

class RLFetchCubeMultiGoal : public RLFetchCube, RLMultiGoalEnv
{
public:
    
    RLFetchCubeMultiGoal()
	{
		mNumExtras = RLMultiGoalEnv::_GetNumExtras();
	}

	int computeNumObservations()
    {	
        // + 3 for cube, +2 finger contacts
        return RLFetchBase::computeNumObservations() + 5;
    }

	virtual int GetNumGoals()
	{
		return 3;
	}

	void PopulateExtra(int ai, float* extra)
	{
		RLMultiGoalEnv::_PopulateExtra(ai, extra);
	}

	void GetDesiredGoal(int ai, float* goal)
	{
		// Transform pose;
		// NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		// Vec3 effectorPose = pose.p - robotPoses[ai];

		for (int i = 0; i < 3; ++i)
		{
			// if(relativeTarget)
			// {
			// 	goal[i] = 0.f;
			// }
			// else
			// {
				goal[i] = targetPoses[ai][i];
			// }
		}
	}
	void GetAchievedGoal(int ai, float* goal)
	{
		// The achieved goal for this environment is the location of the cube which is different from the end-effector in Reach MultiGoal
		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];

		// Transform pose;
		// NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		// Vec3 effectorPose = pose.p - robotPoses[ai];

		for (int i = 0; i < 3; i++)
		{
			// if(relativeTarget)
			// {
			// 	goal[i] = cubeTraLocal[i] - targetPoses[ai][i]; // - effectorPose[i];
			// }
			// else
			// {
				goal[i] = cubeTraLocal[i];
			// }
		}
	}

	void ExtractChildState(int ai, float* state, int& ct)
	{   
		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];

		for (int i = 0; i < 3; ++i)
		{
			if(relativeTarget)
			{
				state[ct++] = cubeTraLocal[i] - state[i];
			}
			else
			{
				state[ct++] = cubeTraLocal[i];  
			}
		}
		state[ct++] = 0.1f * forceFingerLeft[ai];
		state[ct++] = 0.1f * forceFingerRight[ai];
	}

	Vec3 getTargetLocation(int ai, float* state)
    {
        if(relativeTarget)
		{
			return targetPoses[ai] - getCubeLocation(ai, state);
		}
		else
		{
			return targetPoses[ai];
		}
    }

    Vec3 getCubeLocation(int ai, float* state)
    {	
    	// Returning only the absolute location because only the absolute position of cube is required by those functions calling this function
    	Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 Cube = cubePoseWorld.p - robotPoses[ai];

		return Cube;
    }
};

class RLFetchRopeSimpleMultiGoal : public RLFetchRopeSimple, RLMultiGoalEnv
{
public:
    
    RLFetchRopeSimpleMultiGoal()
	{
		mNumExtras = RLMultiGoalEnv::_GetNumExtras();
	}

	virtual int GetNumGoals()
	{
		return 3;
	}

	void PopulateExtra(int ai, float* extra)
	{
		RLMultiGoalEnv::_PopulateExtra(ai, extra);
	}

	void GetDesiredGoal(int ai, float* goal)
	{
		// The desired goal for this environment is the location of the first end of the rope
		int firstRopeBodyId = ropeBodyIds[ai][0];
		
		Transform firstsegPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[firstRopeBodyId], (NvFlexRigidPose*)&firstsegPoseWorld);
		Vec3 firstsegTraLocal = firstsegPoseWorld.p - robotPoses[ai];

		for (int i = 0; i < 3; ++i)
		{
			goal[i] = firstsegTraLocal[i];
		}
	}
	void GetAchievedGoal(int ai, float* goal)
	{
		// The achieved goal for this environment is the location of the other end of the rope
		int lastRopeBodyId = ropeBodyIds[ai][ropeSegments-1];
		Transform lastsegPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[lastRopeBodyId], (NvFlexRigidPose*)&lastsegPoseWorld);
		Vec3 lastsegTraLocal = lastsegPoseWorld.p - robotPoses[ai];

		for (int i = 0; i < 3; i++)
		{
			goal[i] = lastsegTraLocal[i];
		}
	}
};

class RLFetchRopePushMultiGoal : public RLFetchRopePush, RLMultiGoalEnv
{
public:
    
    RLFetchRopePushMultiGoal()
	{
		mNumExtras = RLMultiGoalEnv::_GetNumExtras();
	}

    int computeNumObservations()
    {
        // +2 finger contacts + (3+3)(xyz) * rope segments
        if(doGripperControl)    return RLFetchBase::computeNumObservations() + 3 * ropeSegments + 2;
        else    return RLFetchBase::computeNumObservations() + 3 * ropeSegments;
    }

	virtual int GetNumGoals()
	{
		return 3*ropeSegments;
	}

	void PopulateExtra(int ai, float* extra)
	{
		RLMultiGoalEnv::_PopulateExtra(ai, extra);
	}

	void GetDesiredGoal(int ai, float* goal)
	{
		for (int i = 0; i < GetNumGoals(); i+=3)
		{
			goal[i] = targetPoses[ai][i/3][0];
			goal[i+1] = targetPoses[ai][i/3][1];
			goal[i+2] = targetPoses[ai][i/3][2];
		}
	}
	void GetAchievedGoal(int ai, float* goal)
	{
		Vec3 segTraLocal;
		for (int i = 0; i < GetNumGoals(); i++)
		{
			int RopeBodyId = ropeBodyIds[ai][(int)i/3];

			Transform segPoseWorld;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[RopeBodyId], (NvFlexRigidPose*)&segPoseWorld);
			segTraLocal = segPoseWorld.p - robotPoses[ai];
				
			goal[i] = segTraLocal[i%3];
		}
	}

	virtual float GetGoalDist(float* desiredGoal, float* achievedGoal) override
	{
		float distance = 0.f;
        for(int i=0; i<3*ropeSegments; i+=3)
        {
        	Vec3 desiredG = Vec3(desiredGoal[i], desiredGoal[i+1], desiredGoal[i+2]);
        	Vec3 achievedG = Vec3(achievedGoal[i], achievedGoal[i+1], achievedGoal[i+2]);
        	distance += Length(desiredG - achievedG);
        }

        distance = distance/static_cast<float>(ropeSegments);
        return distance;
	}



	void ExtractChildState(int ai, float* state, int& ct)
	{
		// +6 * ropeSegments rope segment translation
		Transform segPoseWorld;
		for (int i = 0; i < ropeSegments; i++)
		{
			int segBodyId = ropeBodyIds[ai][i];			
			NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
			Vec3 segTraLocal = segPoseWorld.p - robotPoses[ai];
			// NvFlexRigidBody segBody = g_buffers->rigidBodies[segBodyId];

			for (int j = 0; j < 3; j++)
			{
				if(relativeTarget)
				{	
					state[ct++] = segTraLocal[j] - state[j];	
				}
				else
				{
					state[ct++] = segTraLocal[j];
				}
				
			}
		}

		// finger contacts
		if(doGripperControl)
		{
			state[ct++] = 0.1f * forceFingerLeft[ai];
			state[ct++] = 0.1f * forceFingerRight[ai];
		}
    }
};


float* ReadAndClipSensor(int sensorId, int sensorDim, float minval = 0.f, float maxval = 1.f)
{
    float* sensorData = ReadSensor(sensorId);
    bool flag = true;
    for(int i = 0; i < sensorDim; i++)
    {
        if (std::isnan(sensorData[i]) && flag)
        {
            printf("sensor %d returned nan at %d\n", sensorId, i);
            //throw std::logic_error("sensor returned nan");            
            flag = false;

        }
        //use fmax / fmin because of behavior with nans
        sensorData[i] = fmax(fmin(sensorData[i], maxval), minval);
    }
    return sensorData;
}

class RLFetchReachSensor : public RLFetchReach
{
public:

	DepthRenderProfile sensorProfile;
	vector<int> agentSensorIds;
	int sensorSize;
   	int sensorDim;

    RLFetchReachSensor()
	{
        mNumActions = 0;
		g_drawSensors = true;
		sensorProfile = {
			0.0f, // minRange
			1.0f // maxRange
		};
		sensorSize = 64;
        sensorDim = sensorSize * sensorSize * 4;
        radius = 0.1;
	}

	void LoadChildEnv() override
	{
		RLFetchReach::LoadChildEnv();
		agentSensorIds.resize(mNumAgents);
	}

    int computeNumObservations() override
    {
        return RLFetchReach::computeNumObservations() + sensorDim;
    }

	virtual void AddChildEnvBodies(int ai, Transform gt) override
	{
		RLFetchReach::AddChildEnvBodies(ai, gt);

		// Adding sensor

        Transform sensorTransform;
        if (viewName == "default")
        {
            sensorTransform = Transform(gt.p + Vec3(0, 1.0, 0.2), rpy2quat(3*kPi/4, kPi, 0));
        }
        else if (viewName == "front")
        {
            sensorTransform = Transform(gt.p + Vec3(0, 0.9, 1.0), rpy2quat(3*kPi/4, 0, 0));
        }
        else
        {
            throw std::logic_error("invalid view name");
        }
        
		size_t sensorId = AddSensor(sensorSize, sensorSize, 0, sensorTransform,
				DegToRad(90), false, sensorProfile);
		agentSensorIds[ai] = static_cast<int>(sensorId);
	}

	void ExtractSensorState(int ai, float* state, int& ct) override
	{   
        int sensorId = agentSensorIds[ai];
        float* sensorData = ReadAndClipSensor(sensorId, sensorDim);
        for (int i = 0; i < sensorDim; i++)
        {
            state[ct++] = sensorData[i];
        }
	}

	virtual void SampleTarget(int ai) override
	{
		if (sampleInitStates)
		{
            if (initOnPlane)
            {
                targetPoses[ai] = Vec3(Randf(-0.5f, 0.5f), 0.5f, Randf(0.4f, 0.8f));
            }
            else
            {
                targetPoses[ai] = Vec3(Randf(-0.5f, 0.5f), Randf(0.4f, 0.8f), Randf(0.4f, 0.8f));
            }
		}
		else
		{
			targetPoses[ai] = Vec3(-0.3f, 0.5f, 0.5f);
		}
		NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
	}

   	void ColorTargetForReward(float distReward, int a) override
   	{
		float x = min(max(distReward, 0.f), 1.f);
		g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;
   	}

};

class RLFetchReachActive : public RLFetchReachSensor
{
public:

    float d;
    float obsSize;
    Vec3 limitsCamLow;
    Vec3 limitsCamHigh;
    vector<int> obstacleIndex;
    bool obstacleState;
    vector<Vec3> obstaclePoses;

    RLFetchReachActive()
    {
        d = 0.15;
        obsSize = 0.1;
        Vec3 delta = Vec3(0.25, 0.1, 0.1);
        limitsCamLow = Vec3(0, 0.9, 1.0) - delta;
        limitsCamHigh = Vec3(0, 0.9, 1.0) + delta;
    }

    virtual void ParseJson() override 
    {
        RLFetchReachSensor::ParseJson();
        obstacleState = GetJsonVal(g_sceneJson, "ObstacleState", false);
    }
    
    int computeNumActions() override
    {
        return RLFetchReachSensor::computeNumActions() + 3;
    }

    int computeNumObservations() override
    {
        return RLFetchReachSensor::computeNumObservations() + 3 + (obstacleState ? 3 : 0);
    }

    int computeCameraOffset()
    {
        return RLFetchReachSensor::computeNumActions();
    }
    
    void ApplyInverseDynamicsControl(int agentIndex) override
    {
        RLFetchReachSensor::ApplyInverseDynamicsControl(agentIndex);
        float* actions = GetAction(agentIndex);
        Vec3 camera_action = constructVec3(actions + computeCameraOffset());
        ApplyCameraAction(agentIndex, camera_action);
    }

    void ApplyCameraAction(int agentIndex, Vec3 camera_action)
    {
        int sensorId = agentSensorIds[agentIndex];
        Transform transform = GetSensorOrigin(sensorId);
        Transform newtransform = ComputeNewTransform(agentIndex, transform, camera_action);
        SetSensorOrigin(sensorId, newtransform);  
    }

    Vec3 Clamp3(Vec3 v, Vec3 low, Vec3 high)
    {
        return Vec3(Clamp(v.x, low.x, high.x),
                    Clamp(v.y, low.y, high.y),
                    Clamp(v.z, low.z, high.z));
    }
    
    Transform ComputeNewTransform(int agentIndex, Transform transform, Vec3 camera_action)
    {
        Vec3 base = robotPoses[agentIndex];
        transform.p += camera_action * 0.01;
        transform.p = Clamp3(transform.p, limitsCamLow+base, limitsCamHigh+base);
        return transform;
    }

	void ExtractChildState(int ai, float* state, int& ct) override
	{
        //state contains: gripper XYZ, target XYZ, cam XYZ, obstacle XYZ
        
		RLFetchReachSensor::ExtractChildState(ai, state, ct);

        int sensorId = agentSensorIds[ai];
        Transform transform = GetSensorOrigin(sensorId);
        state[ct++] = transform.p.x;
        state[ct++] = transform.p.y;
        state[ct++] = transform.p.z;

        if(obstacleState)
        {
            Vec3 p = obstaclePoses[ai];
            state[ct++] = p.x;
            state[ct++] = p.y;
            state[ct++] = p.z;            
        }
	}

    virtual void ResetAgent(int ai) override
    {
        RLFetchReachSensor::ResetAgent(ai);
        Vec3 base = robotPoses[ai];
        int sensorId = agentSensorIds[ai];
        Transform transform = Transform(base + Vec3(0, 0.9, 1.0), rpy2quat(3*kPi/4, 0, 0));
        SetSensorOrigin(sensorId, transform);
		g_buffers->rigidShapes.map();        
        SampleObstacle(ai);
		g_buffers->rigidShapes.unmap();        
    }

	virtual void AddChildEnvBodies(int ai, Transform gt) override
	{
        RLFetchReachSensor::AddChildEnvBodies(ai, gt);
        NvFlexRigidShape obstacle;
        NvFlexMakeRigidBoxShape(&obstacle, -1, obsSize, obsSize, 0.01, NvFlexMakeRigidPose(0,0));
        int renderMaterial = AddRenderMaterial(greenColor);
        obstacle.user = UnionCast<void*>(renderMaterial);
        obstacle.group = 1;
        obstacleIndex[ai] = g_buffers->rigidShapes.size();
        g_buffers->rigidShapes.push_back(obstacle);
		SampleObstacle(ai);
	}

    void SampleObstacle(int ai)
    {
        Vec3 p = (GetSensorOrigin(ai).p - targetPoses[ai] - robotPoses[ai])/2.0;
        p += Vec3(Randf(-d,d), Randf(-d,d), Randf(-d,d));
        p += robotPoses[ai] + targetPoses[ai];
        p.y += 0.05;
		NvFlexRigidPose pose = NvFlexMakeRigidPose(p, Quat());
        g_buffers->rigidShapes[obstacleIndex[ai]].pose = pose;
        obstaclePoses[ai] = p-robotPoses[ai];
    }

    void LoadChildEnv() override
    {
        RLFetchReachSensor::LoadChildEnv();
        obstacleIndex.resize(mNumAgents,-1);
		obstaclePoses.resize(mNumAgents, Vec3(0.f, 0.f, 0.f));
    }
};

class RLFetchReachActiveMoving : public RLFetchReachActive
{
public: 
    int frameCount;
    int memsize;
    vector<vector<float>> envmemory;
    
    RLFetchReachActiveMoving()
    {
        frameCount = 0;
        d = 0.05;
        obsSize = 0.1;
    }

    virtual int computeNumJoints()
    {
        //leave out memsize
        return RLFetchReachActive::computeNumActions();
    }

    virtual void ParseJson() override 
    {
        RLFetchReachActive::ParseJson();
        memsize = GetJsonVal(g_sceneJson, "MemSize", 0);
        printf("set memsize = %d\n", memsize);
    }

    virtual void PrepareScene() override
    {
        RLFetchReachActive::PrepareScene();
        envmemory.resize(mNumAgents);
        for(int i = 0; i < mNumAgents; i++)
        {
            envmemory[i].resize(memsize);
            for (int j = 0; j < memsize; j++)
            {
                envmemory[i][j] = 0.f;
            }
        }
    }

    void ResetAgent(int ai) override
    {
        RLFetchReachActive::ResetAgent(ai);
        for(int j = 0; j < memsize; j++)
        {
            envmemory[ai][j] = 0.f;
        }
    }
    
    int computeNumActions() override
    {
        return RLFetchReachActive::computeNumActions() + memsize + 1;
    }

    int computeNumObservations() override
    {
        return RLFetchReachActive::computeNumObservations() + memsize;
    }
    
    //every 30 steps, reposition the obstacle to keep things challenging
    virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) override
    {
        RLFetchReachActive::ComputeRewardAndDead(a, action, state, rew, dead);
        
        if (a == 0)
        {
            frameCount ++;
        }
        
        if (((frameCount+1) % 30) == 0)
        {
            SampleObstacle(a);
        }
    }

	void ExtractChildState(int ai, float* state, int& ct) override
	{
        //state contains: gripper XYZ, target XYZ, cam XYZ, obstacle XYZ, memory
        
		RLFetchReachActive::ExtractChildState(ai, state, ct);
        for(int i = 0; i < memsize; i++)
        {
            if (std::isnan(envmemory[ai][i]))
            {
                printf("mem is nan at %d %d\n", ai, i);
                throw std::logic_error("???");
            }
            state[ct++] = envmemory[ai][i];
        }
	}

    int computeMemoryOffset()
    {
        return RLFetchReachActive::computeNumActions();
    }
    
    void ApplyInverseDynamicsControl(int agentIndex) override
    {
        RLFetchReachActive::ApplyInverseDynamicsControl(agentIndex);
        float* actions = GetAction(agentIndex);
        float* memory_action = actions + computeMemoryOffset();
        ApplyMemoryAction(agentIndex, memory_action);
    }

    bool sample(float x)
	{
        x = (x + 1.f) / 2.0f; //now (0, 1)
        return Randf(0.f, 1.f) < x;
    }
    
    void ApplyMemoryAction(int ai, float* action)
    {
        bool do_write = sample(action[0]);
        if (do_write)
        {
            for(int i = 0; i < memsize; i++)
            {
                if(std::isnan(action[i+1]))
                {
                    printf("action being applied at %d %d is nan\n", ai, i);
                    throw std::logic_error("???");
                }
                envmemory[ai][i] = action[i+1];
            }
        }
    }
};

class RLFetchLR : public RLFetchReach
{
public:

    RLFetchLR()
    {
		limitsTra = {
			Vec2(-1.0f, 1.0f),
			Vec2(tableHeight, 0.8f),
			Vec2(0.f, 1.0f)
		};
    }

	virtual void SampleTarget(int ai) override
	{
        if (Randf(-1.0, 1.0) < 0)
        {
            targetPoses[ai] = Vec3(-0.3f, 0.9f, 0.5f);
        }
        else
        {
            targetPoses[ai] = Vec3(+0.3f, 0.9f, 0.5f);
        }
        
		NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
	}

	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) override
	{
        //warning: a different goal is used in the case of HER (see flex_vec_env.py)
        float target_is_right = getTargetLocation(a, state).x > 0.f ? 1.f : -1.f;
        float gripper_x = getEndEffectorLocation(a, state).x;
        rew = max(gripper_x * target_is_right, 0.f);
	}
};

class RLFetchLRHER : public RLFetchLR, RLMultiGoalEnv
{
public:
    RLFetchLRHER()
	{
		mNumExtras = _GetNumExtras();
	}
    
	int computeNumObservations() override
    {
        return RLFetchLR::computeNumObservations() - 3; // no target
    }

    bool IsSuccess(float *desiredGoal, float* achievedGoal) override
    {
        return (desiredGoal[0] * achievedGoal[0]) > 0.0;
    }
    
	virtual int GetNumGoals() override
	{
		return 3;
	}
    
	void PopulateExtra(int ai, float* extra) override
	{
		_PopulateExtra(ai, extra);
	}
    
	void GetDesiredGoal(int ai, float* goal) override
	{
		for (int i = 0; i < 3; ++i)
		{
            goal[i] = targetPoses[ai][i];
		}
	}
    
	void GetAchievedGoal(int ai, float* goal) override
	{
		Transform pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		Vec3 effectorPose = pose.p - robotPoses[ai];

        Vec3 goal_ = targetPoses[ai];
        if(effectorPose[0] * targetPoses[ai][0] < 0.0)
        {
            goal_.x = -goal_.x;
        }
        
		for (int i = 0; i < 3; i++)
		{
            goal[i] = goal_[i];
		}
	}

	void ExtractChildState(int ai, float* state, int& ct) override
	{
		return;
	}

	Vec3 getTargetLocation(int ai, float* state) override
    {
        return targetPoses[ai];
    }

};

class RLFetchLRSensor : public RLFetchReachSensor
{
public:

    RLFetchLRSensor()
    {
		limitsTra =
        {
			Vec2(-1.0f, 1.0f),
			Vec2(tableHeight, 0.8f),
			Vec2(0.f, 1.0f)
		};
    }

	virtual void SampleTarget(int ai) override
	{
        if (Randf(-1.f, 1.f) < 0.f)
		{
            targetPoses[ai] = Vec3(-0.3f, 0.9f, 0.5f);
        }
        else
        {
            targetPoses[ai] = Vec3(+0.3f, 0.9f, 0.5f);
        }
        
		NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
	}

	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) override
	{
        float target_is_right = getTargetLocation(a, state).x > 0.f ? 1.f : -1.f;
        float gripper_x = getEndEffectorLocation(a, state).x;
        rew = gripper_x * target_is_right;
	}
};

class RLFetchReachSensorHER : public RLFetchReachSensor, public RLMultiGoalEnv
{
public:

    RLFetchReachSensorHER()
    {
        mNumExtras = _GetNumExtras();
        mNumPyToC = 1; //indicator variable to use same target location as before
    }

    int computeNumObservations() override
    {
        return RLFetchReachSensor::computeNumObservations() - 3;
    }

    int GetNumGoals() override
    {
        return 3 + sensorDim;
    }

	void PopulateExtra(int ai, float* extra) override
	{
		_PopulateExtra(ai, extra);
	}

	void LoadChildEnv() override
	{
		RLFetchReachSensor::LoadChildEnv();
        targetImages.resize(mNumAgents);
        for(int i = 0; i < mNumAgents; i++)
        {
            targetImages[i].resize(sensorDim);
        }
	}

    void GetDesiredGoal(int ai, float* goal) override
    {
        if (!inPreperationMode(ai))
        {
            //printf("warning -- getting desired goal but not prepared\n");
            return; 
        }

		for (int i = 0; i < 3; i++)
		{
			goal[i] = targetPoses[ai][i];
		}

        for (int j = 0; j < sensorDim; j++)
        {
            goal[j+3] = targetImages[ai][j]; 
        }
    }
    
    void GetAchievedGoal(int ai, float* goal) override
    {
		Transform pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		Vec3 effectorPose = pose.p - robotPoses[ai];
		for (int i = 0; i < 3; i++)
		{
			goal[i] = effectorPose[i];
		}
        
        int sensorId = agentSensorIds[ai];
        float* sensorData = ReadAndClipSensor(sensorId, sensorDim);
        for (int j = 0; j < sensorDim; j++)
        {
            goal[j+3] = sensorData[j];
        }

        if(inPreperationMode(ai))
        {
            //also copy to targetImages
            for (int j = 0; j < sensorDim; j++)
            {
                targetImages[ai][j] = sensorData[j];
            }
        }
    }

	void ExtractChildState(int ai, float* state, int& ct) override
	{
        int sensorId = agentSensorIds[ai];
        float* sensorData = ReadAndClipSensor(sensorId, sensorDim);
        for (int i = 0; i < sensorDim; i++)
        {
            state[ct++] = sensorData[i];
        }
	}

	Vec3 getTargetLocation(int ai, float* state) override
    {
        return targetPoses[ai];
    }

    void PerformIDStep(int ai, float* actions, bool scale=true) override
    {
        if (!inPreperationMode(ai))
        {
            return RLFetchReachSensor::PerformIDStep(ai, actions, scale);
        }

        NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

        Vec3 targetPose = targetPoses[ai] + robotPoses[ai];
        
        effector0.pose0.p[0] = targetPose.x;
        effector0.pose0.p[1] = targetPose.y;
        effector0.pose0.p[2] = targetPose.z;        

		Quat q = rpy2quat(Randf(-kPi, kPi), Randf(-kPi, kPi), Randf(-kPi, kPi));
		effector0.pose0.q[0] = q.x;
		effector0.pose0.q[1] = q.y;
		effector0.pose0.q[2] = q.z;
		effector0.pose0.q[3] = q.w;

        g_buffers->rigidJoints[effectorJoints[ai]] = effector0;
    }

    bool inPreperationMode(int ai)
    {
        if (mPyToCBuf == nullptr)
        {
            return false;
        }
        return mPyToCBuf[ai * mNumPyToC + 0] > 1.f;
    }

	void SampleTarget(int ai) override
	{
        if (inPreperationMode(ai))
        {
            return;
        }
        RLFetchReachSensor::SampleTarget(ai);
	}
   
};


class RLFetchLRSensorHER : public RLFetchLRSensor, public RLMultiGoalEnv
{
public:

    Vec3 goalObsOffset;
    
    RLFetchLRSensorHER()
    {
        mNumExtras = _GetNumExtras();
        mNumPyToC = 1; //indicator variable to use same target location as before

    }

    virtual void PrepareHER(int x) 
    {
        if(x == 0)
        {
            for(int ai = 0; ai < mNumAgents; ai++)
            {
                int sensorId = agentSensorIds[ai];
                float* sensorData = ReadAndClipSensor(sensorId, sensorDim);
                for (int j = 0; j < sensorDim; j++)
                {
                    targetImages[ai][j] = sensorData[j];
                }
            }
        }
        else
        {
            throw std::logic_error("bad x value");
        }
    }

    bool IsSuccess(float *desiredGoal, float* achievedGoal) override
    {
        return (desiredGoal[0] * achievedGoal[0]) > 0.0;
    }
    
    int computeNumObservations() override
    {
        return RLFetchLRSensor::computeNumObservations() - 3;
    }

    int GetNumGoals() override
    {
        return 3 + sensorDim;
    }

	void PopulateExtra(int ai, float* extra) override
	{
		_PopulateExtra(ai, extra);
	}

	void LoadChildEnv() override
	{
		RLFetchLRSensor::LoadChildEnv();
        targetImages.resize(mNumAgents);
        for(int i = 0; i < mNumAgents; i++)
        {
            targetImages[i].resize(sensorDim);
        }
	}

    void GetDesiredGoal(int ai, float* goal) override
    {
		for (int i = 0; i < 3; i++)
		{
			goal[i] = targetPoses[ai][i];
		}
        for (int j = 0; j < sensorDim; j++)
		{
            goal[j+3] = targetImages[ai][j]; 
        }
    }
    
    void GetAchievedGoal(int ai, float* goal) override
    {
		Transform pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[ai]], (NvFlexRigidPose*)&pose);
		Vec3 effectorPose = pose.p - robotPoses[ai];

        Vec3 goal_ = targetPoses[ai];
        if(effectorPose[0] * targetPoses[ai][0] < 0.0)
        {
            goal_.x = -goal_.x;
        }
        
		for (int i = 0; i < 3; i++)
		{
            goal[i] = goal_[i];
		}
        
        int sensorId = agentSensorIds[ai];
        float* sensorData = ReadAndClipSensor(sensorId, sensorDim);
        for (int j = 0; j < sensorDim; j++)
        {
            goal[j+3] = sensorData[j];
        }
  
    }

	void ExtractChildState(int ai, float* state, int& ct) override
	{
        int sensorId = agentSensorIds[ai];
        float* sensorData = ReadAndClipSensor(sensorId, sensorDim);
        for (int i = 0; i < sensorDim; i++)
        {
            state[ct++] = sensorData[i];
        }
	}

	Vec3 getTargetLocation(int ai, float* state) override
    {
        return targetPoses[ai];
    }

    void PerformIDStep(int ai, float* actions, bool scale=true) override
    {
        if (!inPreperationMode(ai))
        {
            return RLFetchLRSensor::PerformIDStep(ai, actions, scale);
        }

        NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

        Vec3 offset = robotPoses[ai] + goalObsOffset;
        effector0.pose0.p[0] = offset.x;
        effector0.pose0.p[1] = offset.y;
        effector0.pose0.p[2] = offset.z;

		Quat q = rpy2quat(0.0, 0.0, -kPi/2.0);
		effector0.pose0.q[0] = q.x;
		effector0.pose0.q[1] = q.y;
		effector0.pose0.q[2] = q.z;
		effector0.pose0.q[3] = q.w;

        g_buffers->rigidJoints[effectorJoints[ai]] = effector0;
    }

    bool inPreperationMode(int ai)
    {
        if (mPyToCBuf == nullptr)
        {
            return false;
        }
        return mPyToCBuf[ai*mNumPyToC+0] > 1.0;
    }

	void SampleTarget(int ai) override
	{
        if (inPreperationMode(ai))
        {
            return;
        }
        RLFetchLRSensor::SampleTarget(ai);
        goalObsOffset = {(targetPoses[ai].x > 0 ? 1.f : -1.f) * Randf(0.05, 0.5),
                         Randf(0.45, 0.65),
                         Randf(0.2, 0.7)};
	}
    
};
