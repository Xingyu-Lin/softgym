#pragma once
#include <iostream>
#include <vector>
#include "rlbase.h"
#include "../urdf.h"
char sawyerUrdfPath[100];
char cupMeshPath[100];
char* make_path(char* full_path, std::string path);

// Collision settings: robot, table have group 0 and filter 0
class RLSawyerBase : public FlexGymBase2
{
public:
	URDFImporter* urdf;
	bool sampleInitStates;
    float tableHeight;
    vector<int> fingersLeft;
    vector<int> fingersRight;
    vector<int> effector;
    vector<int> base;
    
    vector<int> effectorJoints;
    vector<NvFlexRigidJoint> effectorJoints0;
    
    float speedTra, speedRot, speedGrip; // in meters or radians per timestep
    vector<Vec2> limitsTra;
    Vec2 limitsRot, limitsGrip;
    
    bool doStats;
    int hiddenMaterial;
    
    vector<Vec3> robotPoses;
    
    vector<float> fingerWidths;
    vector<float> rolls, pitches, yaws;
    vector<Vec3> targetEffectorTranslations;
    Vec3 initEffectorTranslation;
    
    vector<string> motors;
    vector<float> powers;
    
    vector<float> forceLeft;
    vector<float> forceRight;

	RLSawyerBase()
	{
		mNumAgents = 4;
		mNumActions = 8; // 3 tra + 4 rot + 1 gripper
		mNumObservations = 15; // see below lol
		mMaxEpisodeLength = 100; 

		controlType = eInverseDynamics;
        tableHeight = 0.55f;
		numPerRow = 20;
		spacing = 4.5f;

		speedTra = 5.0f * g_dt;
	    speedRot = 2 * kPi * g_dt;
		speedGrip = 0.1f * g_dt;
		limitsTra = {
			Vec2(-0.5f, 0.5f),
			Vec2(0.f, 1.5f),
			Vec2(0.f, 1.2f)
		};
        limitsRot = Vec2(-kPi, kPi);
		limitsGrip = Vec2(0.000f, 0.05f);
        initEffectorTranslation = Vec3(0.2f, 0.9f, 0.5f), 

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(8.6f, 0.9f, 3.5f);

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 4;
		g_params.numInnerIterations = 30;
		g_params.relaxationFactor = 0.75f;

		g_params.dynamicFriction = 1.25f;	// yes, this is a physically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;

		g_params.shapeCollisionMargin = 0.0015f;

		mDoLearning = g_doLearning;
		doStats = true;
		g_pause = false;
		g_drawPoints = false;
		g_drawCloth = true;  
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);
		if (g_sceneJson.find("SampleInitStates") != g_sceneJson.end())
		{
			sampleInitStates = g_sceneJson.value("SampleInitStates", sampleInitStates);
		}
		if (!sampleInitStates)
		{
			g_sceneLower = Vec3(-0.5f);
			g_sceneUpper = Vec3(0.4f, 0.8f, 0.4f);
		}
		
		LoadEnv();

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());

		initBodies.resize(g_buffers->rigidBodies.size());
		memcpy(&initBodies[0], &g_buffers->rigidBodies[0], sizeof(NvFlexRigidBody) * g_buffers->rigidBodies.size());

		if (mDoLearning)
		{
			init();
		}
	}

	// To be overwritten by child envs
	virtual void LoadChildEnv() {}

	void LoadEnv()
	{
		LoadChildEnv();

		// initialize data structures
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
		
        fingerWidths.resize(mNumAgents, 0.03f);
        rolls.resize(mNumAgents, 0.f);
        pitches.resize(mNumAgents, 0.f);
        yaws.resize(mNumAgents, -90.0f);

		effectorJoints0.clear();
		effector.resize(mNumAgents);
		fingersLeft.resize(mNumAgents);
		fingersRight.resize(mNumAgents);
        base.resize(mNumAgents);
		effectorJoints.resize(mNumAgents);
		robotPoses.resize(mNumAgents);
		effectorJoints0.resize(mNumAgents);
		targetEffectorTranslations.resize(mNumAgents);
        
        motors = {
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
            "head_pan",
        };
        
        int rbIt = 0;
        int jointsIt = 0;

		// hide collision shapes
		hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
		urdf = new URDFImporter(make_path(sawyerUrdfPath, "/data/sawyer"), "/sawyer_description/urdf/sawyer_with_gripper.urdf");
		
        powers.clear();
        
		// set up each env
		for (int ai = 0; ai < mNumAgents; ++ai)
		{
			Vec3 robotPos = Vec3((ai % numPerRow) * spacing, 0.0f, (ai / numPerRow) * spacing);
			Transform gt(robotPos + Vec3(0.f, 0.925f, 0.f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));
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
	}
    
    // To be overwritten by child envs
	virtual void AddChildEnvBodies(int ai, Transform gt) {}
    
    
    void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpowers) {}
    void AddAgentBodiesJointsCtlsPowers(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower,
												int rbIt, int jointsIt)
	{
		int startShape = g_buffers->rigidShapes.size();
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 10000.0f, 0.0f, 10.f, 0.01f, 10.f, 6.f, false);
		int endShape = g_buffers->rigidShapes.size();

		for (int i = startShape; i < endShape; i++)
		{
			g_buffers->rigidShapes[i].thickness = 0.001f;
		}
        
        // Add joint motors for later control
		int pi = 0;
		for (auto m : motors)
		{
			auto jointType = urdf->joints[urdf->urdfJointNameMap[m]]->type;
			if (jointType == URDFJoint::Type::CONTINUOUS || jointType == URDFJoint::Type::REVOLUTE)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[m], eNvFlexRigidJointAxisTwist)); // Or 0?
			}
			else if (jointType == URDFJoint::Type::PRISMATIC)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[m], eNvFlexRigidJointAxisX));
			}
			else
			{
				cout << "Error! Motor can't be a fixed joint" << endl;
			}
					
			float effort = urdf->joints[urdf->urdfJointNameMap[m]]->effort;
	//		cout << m << " power = " << effort << endl;
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
			}
			else if (j->type == URDFJoint::PRISMATIC)
			{
				joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
				joint.targets[eNvFlexRigidJointAxisX] = 0.02f;
				joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f;
				joint.damping[eNvFlexRigidJointAxisX] = 0.0f;
				joint.motorLimit[eNvFlexRigidJointAxisX] = 20.0f;
			}
		}
        
		effector[ai] = urdf->rigidNameMap["r_gripper_l_finger_tip"];

		// fix base in place, todo: add a kinematic body flag?
		g_buffers->rigidBodies[rbIt].invMass = 0.0f;
		(Matrix33&)g_buffers->rigidBodies[rbIt].invInertia = Matrix33();

		fingersLeft[ai] = urdf->jointNameMap["r_gripper_l_finger_joint"];
		fingersRight[ai] = urdf->jointNameMap["r_gripper_r_finger_joint"];
        base[ai] = urdf->rigidNameMap["base"];
        

		NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["right_endpoint"]];
        
        /* Makes sure starts in standard pose
        float startingAngles[] = {0.0f, -1.18f, 0.00f, 2.18f, 0.00f, 0.57f, 3.3161f};
        
        for(int i = 0; i < 7; i++){
            cout<<"Putting joint "<< ctrls[ai][i].first << " at angle " << startingAngles[i]<<endl;
            g_buffers->rigidJoints[ctrls[ai][i].first].targets[ctrls[ai][i].second] = startingAngles[i];
        }*/
        
		if (!mDoLearning || controlType == Control::eInverseDynamics)
		{
			// set up end effector targets
			NvFlexMakeFixedJoint(&effectorJoints0[ai], -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(0.2f, 0.7f, 0.5f) + robotPoses[ai],
				QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), -kPi) * QuatFromAxisAngle(Vec3(0.f, 1.f, 0.f), kPi * 0.5)), NvFlexMakeRigidPose(0, 0));
			for (int i = 0; i < 6; ++i)
			{
				effectorJoints0[ai].compliance[i] = 1e-4f;
				effectorJoints0[ai].damping[i] = 5e+3f;
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
		NvFlexMakeRigidBoxShape(&table, -1, 0.55f, tableHeight, 0.27f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.6f) + robotPoses[ai], Quat()));
		table.filter = 0x0;
		table.group = 0;
		table.material.friction = 0.7f;
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.17f, 0.24f, 0.4f)));

		g_buffers->rigidShapes.push_back(table);
	}
    
	// To be overwritten by child envs
	virtual void ExtractChildState(int ai, float* state, int ct) {}

	int ExtractState(int a, float* state, float* jointSpeeds)
	{
        // Prepare state (Justin - "Really not sure what's going on here")
		//--------------------
		int numJoints = motors.size();
		vector<float> joints(numJoints * 2, 0.f);
		vector<float> angles(numJoints, 0.f);
		vector<float> lows(numJoints, 0.f);
		vector<float> highs(numJoints, 0.f);

		GetAngles(a, angles, lows, highs);
		for (int i = 0; i < mNumActions; i++)
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
		
        NvFlexRigidPose pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[a]], &pose);

		Vec3 effectorPose1 = pose.p; //- robotPoses[a];

		// 0-2 end-effector translation
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = effectorPose1[i];
		}

		// 3-6 end-effector quat (we think its x,y,z,w)
		state[ct++] = pose.q[0];
        state[ct++] = pose.q[1];
        state[ct++] = pose.q[2];
        state[ct++] = pose.q[3];
        
        NvFlexRigidPose base_pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[base[a]], &base_pose);
        
        // 7-9 base translation
        state[ct++] = base_pose.p[0];
        state[ct++] = base_pose.p[1];
        state[ct++] = base_pose.p[2];
        
        // 10-13 base quat
        state[ct++] = base_pose.q[0];
        state[ct++] = base_pose.q[1];
        state[ct++] = base_pose.q[2];
        state[ct++] = base_pose.q[3];

		// 14 - end-effector finger width
		state[ct++] = fingerWidths[a];

		return ct;
	}

	void PopulateState(int ai, float* state)
	{
        float* jointSpeedsA = &jointSpeeds[ai][0];
		int ct = ExtractState(ai, state, jointSpeedsA);
		ExtractChildState(ai, state, ct);
	}
	
	// To be overwritten by child envs
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) {}

	float scaleActions(float minx, float maxx, float x)
	{
		x = 0.5f * (x + 1.f) * (maxx - minx) + minx;
		return x;
	}
    
    void ApplyTargetControl(int agentIndex)
	{
		vector<float> angles(mNumActions, 0.f);
		vector<float> lows(mNumActions, 0.f);
		vector<float> highs(mNumActions, 0.f);
		GetAngles(agentIndex, angles, lows, highs);

		float* actions = GetAction(agentIndex);
		for (int i = 0; i < mNumActions; i++)
		{
			float cc = Clamp(actions[i], -1.f, 1.f);

			float targetPos = 0.5f * (cc + 1.f) * (highs[i] - lows[i]) + lows[i];
			//	cout << "targetPos = " << targetPos << endl;
			//	cout << "low = " << lows[i] << " " << "high = " << highs[i] << endl;

			float smoothness = 0.01f;
			float pos = Lerp(angles[i], targetPos, smoothness);
				//	cout << "pos = " << pos << endl;
			//	g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] = pos;
                
                // Justin - "Won't this lose the last motor action?"
			if (i < (mNumActions - 1))
			{
				g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[ctrls[agentIndex][i].second] = pos;
			}
            // Justin - "Oh I see the finger is (going to be) used here..."
			else
			{
			//	g_buffers->rigidJoints[fingersLeft[agentIndex]].targets[eNvFlexRigidJointAxisX] = 0.f; //pos;
			//	g_buffers->rigidJoints[fingersRight[agentIndex]].targets[eNvFlexRigidJointAxisX] = 0.f; //pos;
			}
		}
	}
    
    void ApplyVelocityControl(int agentIndex, float df)
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
	}
    
	void ApplyInverseDynamicsControl(int agentIndex)
	{
		float* actions = GetAction(agentIndex);

		PerformIDStep(agentIndex, actions[0], actions[1], actions[2], actions[3], actions[4], actions[5], actions[6], actions[7], false);
        //cout<<"Applying inverse dynamics control" << endl;
	}

	void PerformIDStep(int ai, float targetx, float targety, float targetz, float x, float y, float z, float w, float newWidth, bool scale=true)
	{
		/*if (scale)
		{
			targetx = scaleActions(limitsTra[0][0], limitsTra[0][1], targetx);
			targety = scaleActions(limitsTra[1][0], limitsTra[1][1], targety);
			targetz = scaleActions(limitsTra[2][0], limitsTra[2][1], targetz);
			oroll = scaleActions(limitsRot[0], limitsRot[1], oroll);
			opitch = scaleActions(limitsRot[0], limitsRot[1], opitch);
			oyaw = scaleActions(limitsRot[0], limitsRot[1], oyaw);
			newWidth = scaleActions(limitsGrip[0], limitsGrip[1], newWidth);
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
		}*/

		NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

		/*float asmoothing = 0.05f;

        rolls[ai] = Lerp(rolls[ai], oroll,asmoothing);
		pitches[ai] = Lerp(pitches[ai], opitch, asmoothing);
		yaws[ai] = Lerp(yaws[ai], oyaw, asmoothing);		
        */
		const float smoothing = 0.1f;

		// low-pass filter controls otherwise it is too jerky
		targetEffectorTranslations[ai].x = Lerp(targetEffectorTranslations[ai].x, targetx, smoothing);
		targetEffectorTranslations[ai].y = Lerp(targetEffectorTranslations[ai].y, targety, smoothing);
		targetEffectorTranslations[ai].z = Lerp(targetEffectorTranslations[ai].z, targetz, smoothing);

		Vec3 targetEffectorTranslation = targetEffectorTranslations[ai];// + robotPoses[ai];
        cout<< "Robot " << ai << " moving to new position ("<< targetx<< "," << targety << "," <<targetz<<")\n";
		effector0.pose0.p[0] = targetx;//targetEffectorTranslation.x;
		effector0.pose0.p[1] = targety;//targetEffectorTranslation.y;
		effector0.pose0.p[2] = targetz;// targetEffectorTranslation.z;

		//Quat q = rpy2quat(rolls[ai], pitches[ai], yaws[ai]);
		effector0.pose0.q[0] = x;
		effector0.pose0.q[1] = y;
		effector0.pose0.q[2] = z;
		effector0.pose0.q[3] = w;

		g_buffers->rigidJoints[effectorJoints[ai]] = effector0;

		g_buffers->rigidJoints[fingersLeft[ai]].targets[eNvFlexRigidJointAxisX] = newWidth;
		g_buffers->rigidJoints[fingersRight[ai]].targets[eNvFlexRigidJointAxisX] = newWidth;
	}

	virtual void ResetAgent(int a) {}

	~RLSawyerBase()
	{
		if (urdf)
		{
			delete urdf;
		}
	}

    virtual void PreHandleCommunication() {}
	virtual void ClearContactInfo() {}
	virtual void FinalizeContactInfo() {}
	virtual void LockWrite() {} // Do whatever needed to lock write to simulation
	virtual void UnlockWrite() {} // Do whatever needed to unlock write to simulation

	void DoGui()
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
			imguiSlider("Gripper X", &targetx, limitsTra[0][0], limitsTra[0][1], 0.0001f);
			imguiSlider("Gripper Y", &targety, limitsTra[1][0], limitsTra[1][1], 0.0001f);
			imguiSlider("Gripper Z", &targetz, limitsTra[2][0], limitsTra[2][1], 0.0001f);
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

	void Update()
	{
		if (!mDoLearning)
		{
			// TODO(justinrose): add update if not learning
            
		}
	}

	void PostUpdate()
	{
		// joints are not read back by default
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer); // Do we need it?
	}
};

class RLSawyerVelocityControl: public FlexGymBase2
{
public:
    URDFImporter* urdf;
   
};


class RLSawyerReach : public RLSawyerBase
{
public:

	vector<int> targetSphere;
	vector<int> targetRenderMaterials;
	Vec3 redColor;
	Vec3 greenColor;
	vector<Vec3> targetPoses;
	float radius;

	RLSawyerReach()
	{
		mNumObservations = 10; // 7 state of endeffector + 3 target xyz
		
		doStats = false;

		radius = 0.04f;
		redColor = Vec3(1.0f, 0.04f, 0.07f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);
	}

	void LoadChildEnv()
	{
		targetPoses.resize(mNumAgents, Vec3(0.4f, 1.f, 0.8f));
		targetSphere.resize(mNumAgents, -1);
		targetRenderMaterials.resize(mNumAgents);
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
			targetPoses[ai] = Vec3(Randf(-0.5f, 0.5f), Randf(0.4f, 1.6f), Randf(0.25f, 1.1f));
		}
		else
		{
			targetPoses[ai] = Vec3(-0.3f, 0.5f, 0.5f);
		}
		NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
	}

	virtual void AddChildEnvBodies(int ai, Transform gt) 
	{
		NvFlexRigidShape targetShape;
		NvFlexMakeRigidSphereShape(&targetShape, -1, radius, NvFlexMakeRigidPose(0,0));

		int renderMaterial = AddRenderMaterial(redColor);
		targetRenderMaterials[ai] = renderMaterial;
		targetShape.user = UnionCast<void*>(renderMaterial);
		targetShape.group = 1;
		targetSphere[ai] = g_buffers->rigidShapes.size();
		g_buffers->rigidShapes.push_back(targetShape);

		SampleTarget(ai);
	}

	void ExtractChildState(int ai, float* state, int ct)
	{
		// 7-9 target xyz
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = targetPoses[ai][i];
		}
	}
	
	void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float reg = 0.f;
		float* prevAction = GetPrevAction(a);
		for (int i = 0; i < mNumActions; i++)
		{
			reg += Pow((action[i] - prevAction[i]), 2) / (float)mNumActions;
		}
		//	cout << "Penalty = " << reg << endl;

		float dist = Length(Vec3(state[0] - state[12], state[1] - state[13], state[2] - state[14]));
		float distReward = 2.f * exp(-5.f * dist) - 0.4f * reg;

		float x = max((distReward - 0.2f) / 1.79f, 0.f);

		g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;
		rew = distReward;
	}

	void ResetAgent(int ai)
	{
		for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second; i++)
		{
			g_buffers->rigidBodies[i] = initBodies[i];
		}

		g_buffers->rigidShapes.map();
		SampleTarget(ai);
		g_buffers->rigidShapes.unmap();
		NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

		RLFlexEnv::ResetAgent(ai);
	}
};

class RLSawyerCup : public RLSawyerBase
{
public:

    vector<int> targetCups;
    vector<int> targetRenderMaterials;
    vector<Vec3> targetPoses;
    float radius;

    RLSawyerCup()
	{
        //mNumObservations = 10; // 7 state of endeffector + 3 target xyz
        doStats = false;
        radius = 0.04f;
    }

    void LoadChildEnv()
	{
        targetPoses.resize(mNumAgents, Vec3(0.4f, 1.f, 0.8f));
        targetCups.resize(mNumAgents, -1);
        targetRenderMaterials.resize(mNumAgents);
    }

    void SampleTarget(int ai)
	{
        targetPoses[ai] = Vec3(-0.3f, 0.6f, 0.5f);
        NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
        NvFlexSetRigidPose(&g_buffers->rigidBodies[targetCups[ai]], &pose);
    }

    virtual void AddChildEnvBodies(int ai, Transform gt)
	{
        float density = 2000.0f;
        float scale = 1.f;
        Mesh* cupMesh = ImportMesh(make_path(cupMeshPath, "/data/cups/cup2_low.obj"));
        cupMesh->Transform(ScaleMatrix(scale));
        NvFlexTriangleMeshId shapeId = CreateTriangleMesh(cupMesh, 0.001f);
        NvFlexRigidShape shape;
        NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0,0), 1.f, 1.f, 1.f);
        shape.filter = 0x0;
		shape.material.friction = 1.0f;
		shape.material.torsionFriction = 0.1;
		shape.material.rollingFriction = 0.0f;
		shape.thickness = 0.001f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.f), Quat(), &shape, &density, 1);
        
        targetCups[ai] = g_buffers->rigidBodies.size();
        
		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);

        SampleTarget(ai);
    }

    void ExtractChildState(int ai, float* state, int ct)
	{
        // 7-9 target xyz
        for (int i = 0; i < 3; ++i)
		{
            state[ct++] = targetPoses[ai][i];
        }
    }

    void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
        float reg = 0.f;
        float* prevAction = GetPrevAction(a);
        for (int i = 0; i < mNumActions; i++)
		{
            reg += Pow((action[i] - prevAction[i]), 2) / (float) mNumActions;
        }
        //	cout << "Penalty = " << reg << endl;

        float dist = Length(Vec3(state[0] - state[12], state[1] - state[13], state[2] - state[14]));
        float distReward = 2.f * exp(-5.f * dist) - 0.4f * reg;

        float x = max((distReward - 0.2f) / 1.79f, 0.f);

        rew = distReward;
    }

    void ResetAgent(int ai)
	{
        for (int i = agentBodies[ai].first; i < (int) agentBodies[ai].second; i++)
		{
            g_buffers->rigidBodies[i] = initBodies[i];
        }

        g_buffers->rigidShapes.map();
        SampleTarget(ai);
        g_buffers->rigidShapes.unmap();
        NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

        RLFlexEnv::ResetAgent(ai);
    }
};
