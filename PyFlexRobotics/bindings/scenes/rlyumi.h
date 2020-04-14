#pragma once
#include <iostream>
#include <vector>
#include "rlbase.h"
#include "../urdf.h"

#if defined(WIN32) || defined(_WIN32)
#include "../external/eigen3/eigen/Dense"
#else
#include "external/eigen3/eigen/Dense"
#endif

#define RL_JSON_SAMPLE_INIT_JOINTS "SampleInitJoints"

#define RL_JSON_SIM_PARAMS_STOCHASTIC "SimParamsStochastic"
#define RL_JSON_SIM_PARAMS_PRESAMPLED "SimParamsPresampled"

#define RL_JSON_SIM_PARAMS_MEAN "SimParamsMean"
#define RL_JSON_SIM_PARAMS_MIN "SimParamsMin"
#define RL_JSON_SIM_PARAMS_MAX "SimParamsMax"
#define RL_JSON_SIM_PARAMS_COV "SimParamsCov"
#define RL_JSON_SIM_PARAMS_COV_INIT "SimParamsCovInit"
#define RL_JSON_SIM_PARAMS_SAMPLES "SimParamsSamples"
#define RL_JSON_INIT_POSITION "InitPosition"
#define RL_JSON_ROBOT_FREQUENCY "RobotFrequency"
#define RL_JSON_RESET_NUM_STEPS "ResetNumSteps"
#define RL_JSON_REWARD_WEIGHTS "RewardWeights"
#define RL_JSON_CONTROL_GRIPPER "ControlGripper"

struct normal_random_variable
{
    normal_random_variable() {}

    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](double x) { return dist(gen); });
    }
};

class RLYumiBase : public FlexGymBase2
{
public:
	URDFImporter* urdf;

	bool sampleInitStates = false;
	bool sampleInitJoints = false;
	bool controlGripper = false;

	int resetNumSteps = 100;
	float tableHeight;

	vector<int> fingersLeft;
	vector<int> fingersRight;
	vector<int> fingerLeftBody;
	vector<int> fingerRightBody;
	unordered_map<int, int> mapFingerLeftToAgent;
	unordered_map<int, int> mapFingerRightToAgent;

	vector<float> forceLeftAgent;
	vector<float> forceRightAgent;

	vector<int> effectorJoints;
	vector<NvFlexRigidJoint> effectorJoints0;

	float speedTra, speedRot, speedGrip; // in meters or radians per timestep
	vector<Vec2> limitsTra;
	Vec2 limitsRot, limitsGrip;

	bool doStats;
	int hiddenMaterial;

	vector<Vec3> robotPoses;
	vector<pair<int, int> > childBodies;

	vector<float> fingerWidths;
	vector<float> rolls, pitches, yaws;
	vector<Vec3> targetEffectorTranslations;
	Vec3 initEffectorTranslation;

	vector<string> motors_right;
	vector<string> motors_left;
	vector<float> powers;

	vector<float> forceLeft;
	vector<float> forceRight;

    vector<int> robotStartShape;
    vector<int> robotEndShape;

    vector<float> initJointAnglesRight;
	vector<Vec3> targetPoses;
    vector<int> targetSphere;

    float joint_limits_low[7] = {-2.940880f, -2.504547f,  -2.155482f, -5.061455f, -1.535890f, -3.996804f, -2.940880f};
    float joint_limits_high[7] = {2.940880f, 0.759218f, 1.396263f, 5.061455f, 2.408554f, 3.996804f, 2.940880f};

//    float initJointAnglesRight[8] = {-0.027727f,-0.493535f,0.702561f,
//       0.563854f,-1.278152f,-1.174275f,0.528319f, 0.01f};

    float initJointAnglesLeft[8] =
        {-0.37707386526964104f,-1.1284522471806202f, -1.5662578496902988f,
            0.40748309617726797f,0.7958395979066761f,-0.009977298498623277f,0.0f,0.01f};

    vector<vector<float>> initJointAnglesAgent;
	vector<int> originalFilters;

    vector<bool> firstReset;

    bool sim_params_stochastic = false;
    bool sim_params_presampled = false;
    const int num_robot_sim_params = 21;

    vector<vector<float>> sim_params_samples;
    vector<vector<float>> sim_params_orig;
    vector<float> sim_params_mean;
    vector<vector<float>> sim_params_cov;
    vector<float> sim_params_min;
    vector<float> sim_params_max;

    int num_sim_params = 0;
    normal_random_variable gauss_sample;

    int numRobotStates, numRobotExtras;
    float robotFrequency = 100.0f;
    float motorLimits[8] = {3000.0f, 3000.0f, 3000.0f, 3000.0f, 3000.0f, 3000.0f, 3000.0f, 3000.0f};

    vector<vector<float>> prevState;
    vector<vector<float>> currState;
    vector<vector<float>> prevExtra;
    vector<vector<float>> currExtra;

    vector<float> rewardWeights;

	RLYumiBase()
	{
		mNumAgents = 16;
		numPerRow = 4;
		mNumActions = 8; // 7 joints + 1 gripper
		//mNumObservations = 14; // Joint angles, end-effector pose and finger width

		mNumObservations = 7; // Joint angles
		mNumExtras = 6 + 3; // end-effector xyz and rpy

        numRobotExtras = mNumExtras;
		numRobotStates = mNumObservations;

		mMaxEpisodeLength = 100;
		controlType = ePosition;
		tableHeight = 0.39f;

		spacing = 3.5f;

		speedTra = 5.f * g_dt;
		speedRot = 2 * kPi * g_dt;
		speedGrip = 0.1f * g_dt;
		limitsTra = {
			Vec2(-0.5f, 0.5f),
			Vec2(0.f, 1.6f),
			Vec2(0.f, 1.2f)
		};
		limitsRot = Vec2(-kPi, kPi);
		limitsGrip = Vec2(0.002f, 0.05f);
		initEffectorTranslation = Vec3(-0.0092f, 0.7f, 0.3f);

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(8.6f, 0.9f, 3.5f);

		g_params.numPostCollisionIterations = 0;
		g_params.dynamicFriction = 0.8f;
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.shapeCollisionMargin = 0.005f;
		g_params.collisionDistance = 0.005f;

		g_params.solverType = eNvFlexSolverPCR;
		g_numSubsteps = 2;
		g_params.numIterations = 5; // 5; //4;
		g_params.numInnerIterations = 25; //20;
		g_params.relaxationFactor = 0.75f;

    	mDoLearning = g_doLearning;
		doStats = true;
		g_pause = true;
		g_drawPoints = false;
		g_drawCloth = true;

		initJointAnglesRight.resize(8, 0.0);
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);
		sim_params_stochastic = false;
        sim_params_presampled = false;
        sim_params_samples.resize(mNumAgents);
        sim_params_orig.resize(mNumAgents);
        firstReset.resize(mNumAgents, true);
   		targetSphere.resize(mNumAgents, -1);
        prevState.resize(mNumAgents);
        currState.resize(mNumAgents);
        prevExtra.resize(mNumAgents);
        currExtra.resize(mNumAgents);

        for (int i = 0; i < mNumAgents; i++)
		{
            currState[i].resize(mNumObservations, 0.0f);
            prevState[i].resize(mNumObservations, 0.0f);
            prevExtra[i].resize(mNumExtras, 0.0f);
            currExtra[i].resize(mNumExtras, 0.0f);
        }

        if (!g_sceneJson[RL_JSON_SIM_PARAMS_STOCHASTIC].is_null())
		{
            sim_params_stochastic = g_sceneJson[
                RL_JSON_SIM_PARAMS_STOCHASTIC].get<bool>();
            sim_params_mean = g_sceneJson[
                RL_JSON_SIM_PARAMS_MEAN].get<std::vector<float>>();

            num_sim_params = sim_params_mean.size();

            if (sim_params_stochastic)
			{
                sim_params_presampled = !g_sceneJson[RL_JSON_SIM_PARAMS_PRESAMPLED].is_null()
                    && g_sceneJson[RL_JSON_SIM_PARAMS_PRESAMPLED].get<bool>();

                if (sim_params_presampled)
				{
                    sim_params_samples = g_sceneJson[RL_JSON_SIM_PARAMS_SAMPLES].get<std::vector<std::vector<float>>>();
                }
				else
				{
                    if(g_sceneJson[RL_JSON_SIM_PARAMS_COV].is_null())
					{
                        // Initialize covariance diagonal.
                        vector<float> sim_params_cov_init = g_sceneJson[
                            RL_JSON_SIM_PARAMS_COV_INIT].get<std::vector<float>>();

                        sim_params_cov.resize(sim_params_cov_init.size());
                        for (int i = 0; i < sim_params_cov.size(); i++)
						{
                            sim_params_cov[i] = vector<float>(sim_params_cov.size(), 0.0);
                            sim_params_cov[i][i] = sim_params_cov_init[i];
                        }
                    }
					else
					{
                        sim_params_cov = g_sceneJson[
                            RL_JSON_SIM_PARAMS_COV].get<std::vector<std::vector<float>>>();
                    }

                    sim_params_min = g_sceneJson[RL_JSON_SIM_PARAMS_MIN].get<std::vector<float>>();
                    sim_params_max = g_sceneJson[RL_JSON_SIM_PARAMS_MAX].get<std::vector<float>>();
                }
            }
        }

    	if (g_sceneJson.find("SampleInitStates") != g_sceneJson.end())
		{
			sampleInitStates = g_sceneJson.value("SampleInitStates", sampleInitStates);
		}

        if (!g_sceneJson[RL_JSON_SAMPLE_INIT_JOINTS].is_null())
		{
            sampleInitJoints = g_sceneJson[RL_JSON_SAMPLE_INIT_JOINTS].get<bool>();
		}

		if (!g_sceneJson[RL_JSON_INIT_POSITION].is_null())
		{
            initJointAnglesRight = g_sceneJson[RL_JSON_INIT_POSITION].get<std::vector<float>>();
		}

        if (!g_sceneJson[RL_JSON_RESET_NUM_STEPS].is_null())
		{
            resetNumSteps = g_sceneJson[RL_JSON_RESET_NUM_STEPS].get<int>();
		}

        if (!g_sceneJson[RL_JSON_ROBOT_FREQUENCY].is_null())
		{
            robotFrequency =  g_sceneJson[RL_JSON_ROBOT_FREQUENCY].get<float>();
            g_dt = 1.0f / robotFrequency;
        }

        if (!g_sceneJson[RL_JSON_REWARD_WEIGHTS].is_null()) {
            rewardWeights =  g_sceneJson[RL_JSON_REWARD_WEIGHTS].get<vector<float>>();
        }

        if (!g_sceneJson[RL_JSON_CONTROL_GRIPPER].is_null())
		{
            controlGripper = g_sceneJson[RL_JSON_CONTROL_GRIPPER].get<bool>();
		}


		if (sim_params_stochastic && !sim_params_presampled)
		{
            Eigen::VectorXd mean(sim_params_mean.size());
            Eigen::MatrixXd cov(sim_params_cov.size(), sim_params_cov[0].size());
            for (int i = 0; i < sim_params_cov.size(); i++)
			{
                mean[i] = sim_params_mean[i];
                for (int j = 0; j < sim_params_cov[0].size(); j++)
				{
                    cov(i,j) = sim_params_cov[i][j];
                }
            }
            gauss_sample = normal_random_variable {mean, cov};
        }

		if (!sampleInitStates)
		{
			g_sceneLower = Vec3(-0.5f);
			g_sceneUpper = Vec3(0.4f, 0.8f, 0.4f);
		}

        LoadEnv();

		initJoints.resize(g_buffers->rigidJoints.size());
		memcpy(&initJoints[0], &g_buffers->rigidJoints[0], sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());

		if (mDoLearning)
		{
			init();
		}

	}

	// To be overwritten by child env
	virtual void LoadChildEnv() {}

	void LoadEnv()
	{
        initJointAnglesAgent.resize(mNumAgents, vector<float>());
        for (int i = 0; i < mNumAgents; i++)
		{
            initJointAnglesAgent[i].resize(mNumActions, 0.0);
            initJointAnglesAgent[i][0] = -17.54f / 180.0f * kPi;
            initJointAnglesAgent[i][1] = -12.17f / 180.0f * kPi;
            initJointAnglesAgent[i][2] = -1.44f / 180.0f * kPi;
            initJointAnglesAgent[i][3] = -20.35f / 180.0f * kPi;
            initJointAnglesAgent[i][4] = -48.56f / 180.0f * kPi;
            initJointAnglesAgent[i][5] = -1.0f / 180.0f * kPi;
            initJointAnglesAgent[i][6] = -22.89f / 180.0f * kPi;
        }


		LoadChildEnv();

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		fingerWidths.resize(mNumAgents, 0.03f);
		rolls.resize(mNumAgents, 0.f);
		pitches.resize(mNumAgents, 0.f);
		robotStartShape.resize(mNumAgents, 0);
		robotEndShape.resize(mNumAgents, 0);

		if (mDoLearning)
		{
			yaws.resize(mNumAgents, -kPi / 2.f);
		}
		else
		{
			yaws.resize(mNumAgents, -90.f);
		}

		effectorJoints0.clear();
		fingersLeft.resize(mNumAgents);
		fingersRight.resize(mNumAgents);
		fingerLeftBody.resize(mNumAgents);
		fingerRightBody.resize(mNumAgents);
		forceLeftAgent.resize(mNumAgents, 0.0f);
		forceRightAgent.resize(mNumAgents, 0.0f);
		effectorJoints.resize(mNumAgents);
		robotPoses.resize(mNumAgents);
		effectorJoints0.resize(mNumAgents);
		targetEffectorTranslations.resize(mNumAgents);

		motors_right = {
		    "yumi_joint_1_r",
		    "yumi_joint_2_r",
		    "yumi_joint_3_r",
		    "yumi_joint_4_r",
		    "yumi_joint_5_r",
		    "yumi_joint_6_r",
		    "yumi_joint_7_r",
		    "gripper_r_joint",
		    "gripper_r_joint_m",
		};

		motors_left = {
		    "yumi_joint_1_l",
		    "yumi_joint_2_l",
		    "yumi_joint_3_l",
		    "yumi_joint_4_l",
		    "yumi_joint_5_l",
		    "yumi_joint_6_l",
		    "yumi_joint_7_l",
		    "gripper_l_joint"
		};

		int rbIt = 0;
		int jointsIt = 0;

		// hide collision shapes
		hiddenMaterial = AddRenderMaterial(1.0f, 1.0f, 1.0f, true);

	    //new
	    bool useObjForCollision = true;
	    float dilation = 0.000f;
	    float thickness = 0.005f;
	    bool replaceCylinderWithCapsule=true;
	    int slicesPerCylinder = 20;
	    bool useSphereIfNoCollision = true;

	    urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi.urdf",
	    	useObjForCollision,
	    	dilation,
	    	thickness,
	    	replaceCylinderWithCapsule,
	    	slicesPerCylinder,
	    	useSphereIfNoCollision);

		Transform robot_transform(Vec3(0.0f, 0.025f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));
		powers.clear();
		for (int ai = 0; ai < mNumAgents; ++ai)
		{
			Vec3 robotPos = Vec3((ai % numPerRow) * spacing, 0.01f, (ai / numPerRow) * spacing);
			Transform gt(robotPos, QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f),
			             -kPi * 0.5f) * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));

			robotPoses[ai] = robotPos;

			int begin_agent_bodies = g_buffers->rigidBodies.size();
			AddAgentBodiesJointsCtlsPowers(ai, gt, ctrls[ai], motorPower[ai], rbIt, jointsIt);
            int begin_child_bodies = g_buffers->rigidBodies.size();
			AddChildEnvBodies(ai, gt);
			int end_agent_bodies = g_buffers->rigidBodies.size();
			agentBodies.push_back(make_pair(begin_agent_bodies, end_agent_bodies));
			childBodies.push_back(make_pair(begin_child_bodies, end_agent_bodies));

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

		originalFilters.resize(g_buffers->rigidShapes.size(), -1);
	}

	// To be overwritten by child envs
	virtual void AddChildEnvBodies(int ai, Transform gt) {}

	void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpowers) {}
	void AddAgentBodiesJointsCtlsPowers(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower,
												int rbIt, int jointsIt)
	{
	    SampleSimParams(ai);

        // physics variables
        float jointCompliance = 0.0;	// 1.e-6f;
        float jointDamping = 10.0f;		// 1.e+1f;
        float bodyArmature = 5.e-2f;	// 0.01;
        float bodyDamping = 10.f;		// 5.0f;
        float maxAngularVel = 50.0f;

		robotStartShape[ai] = g_buffers->rigidShapes.size();
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 5000.0f, jointCompliance, jointDamping, bodyArmature, bodyDamping, maxAngularVel, false);
		robotEndShape[ai] = g_buffers->rigidShapes.size();

		for (int i = robotStartShape[ai]; i < robotEndShape[ai]; i++)
		{
//            if (i == urdf->rigidShapeNameMap["gripper_r_finger_r"].first
//                 || i == urdf->rigidShapeNameMap["gripper_r_finger_l"].first)
//			{
//                g_buffers->rigidShapes[i].thickness += 0.005f;
//            }
//				else if (i == urdf->rigidShapeNameMap["gripper_r_base"].first)
//				{
//                g_buffers->rigidShapes[i].thickness += 0.005f;
//            }
            g_buffers->rigidShapes[i].filter = 0x1;
		}

		for (int i = 0; i < (int)urdf->joints.size(); i++)
		{
			URDFJoint* j = urdf->joints[i];
			NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[j->name]];
			if (j->type == URDFJoint::REVOLUTE)
			{
			    if (mDoLearning && controlType == Control::ePosition)
				{
			        joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
			    }
			    joint.targets[eNvFlexRigidJointAxisTwist] = 0.0f;
				joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-7f;
				joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+3f;
				if (mDoLearning) joint.motorLimit[eNvFlexRigidJointAxisTwist] = motorLimits[i];
			}
			else if (j->type == URDFJoint::PRISMATIC)
			{
    		    joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
				joint.targets[eNvFlexRigidJointAxisX] = 0.002f;
				joint.compliance[eNvFlexRigidJointAxisX] = 1.e-11f;
				joint.damping[eNvFlexRigidJointAxisX] = 0.0f;
//				if (mDoLearning) joint.motorLimit[eNvFlexRigidJointAxisX] = motorLimits[i];
			}
		}

        // Add controls to the right arm, set initial joint positions.
		for (int i = 0; i < motors_right.size(); i++)
		{
			auto jointType = urdf->joints[urdf->urdfJointNameMap[motors_right[i]]]->type;
     		NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[motors_right[i]]];
			if (jointType == URDFJoint::Type::REVOLUTE)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[motors_right[i]], eNvFlexRigidJointAxisTwist)); //3)); // Or 0?
//                joint.targets[eNvFlexRigidJointAxisTwist] = initJointAnglesRight[i];
			}
			else if (jointType == URDFJoint::Type::PRISMATIC)
			{
				ctrl.push_back(make_pair(urdf->jointNameMap[motors_right[i]], eNvFlexRigidJointAxisX));
//	            joint.targets[eNvFlexRigidJointAxisX] = 0.02f;
			}
			else
			{
				cout << "Error! Motor can't be a fixed joint" << endl;
			}

			float effort = urdf->joints[urdf->urdfJointNameMap[motors_right[i]]]->effort;
			mpower.push_back(effort);
		}

        // Set initial joint positions for the left arm, don't set the controls.
        for (int i = 0; i < motors_left.size(); i++)
		{
     		NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[motors_left[i]]];
        	auto jointType = urdf->joints[urdf->urdfJointNameMap[motors_left[i]]]->type;

     		if (jointType == URDFJoint::Type::CONTINUOUS || jointType == URDFJoint::Type::REVOLUTE)
			{
                joint.targets[eNvFlexRigidJointAxisTwist] = initJointAnglesLeft[i];
                if (mDoLearning) joint.motorLimit[eNvFlexRigidJointAxisTwist] = 1e3;//motorLimits[i];
            } else if (jointType == URDFJoint::Type::PRISMATIC)
			{
			    joint.targets[eNvFlexRigidJointAxisX] = initJointAnglesLeft[i];
                if (mDoLearning) joint.motorLimit[eNvFlexRigidJointAxisX] = 1e3;//motorLimits[i];
			}
		}

        fingerLeftBody[ai] = urdf->rigidNameMap["gripper_r_finger_l"];
        fingerRightBody[ai] = urdf->rigidNameMap["gripper_r_finger_r"];
		mapFingerLeftToAgent.emplace(urdf->rigidNameMap["gripper_r_finger_l"], ai);
		mapFingerRightToAgent.emplace(urdf->rigidNameMap["gripper_r_finger_r"], ai);

		// fix base in place, todo: add a kinematic body flag?
		g_buffers->rigidBodies[rbIt].invMass = 0.0f;
		(Matrix33&)g_buffers->rigidBodies[rbIt].invInertia = Matrix33();

		fingersLeft[ai] = urdf->jointNameMap["gripper_r_joint"];
		fingersRight[ai] = urdf->jointNameMap["gripper_r_joint_m"];

		NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["gripper_r_joint"]];

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
	}

	int ExtractState(int a, float* state)
	{
		int numJoints = motors_right.size();
		vector<float> angles(numJoints, 0.f);
		vector<float> lows(numJoints, 0.f);
		vector<float> highs(numJoints, 0.f);

	    GetAngles(a, angles, lows, highs);
        int ct = 0;

        // 0-6 joint angles (without gripper)
		for (int i = 0; i < numJoints-2; i++)
		{
			state[ct++] = angles[i];
		}

//		Transform pose;
//		NvFlexGetRigidPose(&g_buffers->rigidBodies[effector[a]], (NvFlexRigidPose*)&pose);
//
//		Vec3 effectorPose1 = pose.p - robotPoses[a];
//		// state 8-10: end-effector translation
//		for (int i = 0; i < 3; ++i)
//		{
//			state[ct++] = effectorPose1[i];
//		}
//
//		float roll, pitch, yaw;
//		quat2rpy(pose.q, roll, pitch, yaw);
//
//		// 11-13 end-effector rpy
//		state[ct++] = roll;
//		state[ct++] = pitch;
//		state[ct++] = yaw;

		return ct;
	}

	// To be overwritten by child env
	virtual void ExtractChildState(int ai, float* state, int ct) {}
	void PopulateState(int ai, float* state)
	{
		int ct = ExtractState(ai, state);
		ExtractChildState(ai, state, ct);

        for (int i = 0; i < mNumObservations; i++)
		{
	        prevState[ai][i] = currState[ai][i];
	        currState[ai][i] = state[i];
	    }
	}

    virtual void ExtractChildExtra(int ai, float* extras, int ct) {}
	void PopulateExtra(int ai, float* extras)
	{
	    int ct = 0;

	    Transform fingerLeftPose;
	    Transform fingerRightPose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[fingerLeftBody[ai]], (NvFlexRigidPose*)&fingerLeftPose);
		NvFlexGetRigidPose(&g_buffers->rigidBodies[fingerRightBody[ai]], (NvFlexRigidPose*)&fingerRightPose);
		Vec3 effectorPose = (fingerLeftPose.p + fingerRightPose.p) / 2.0 - robotPoses[ai];

		// extras 0-2: end-effector translation
		for (int i = 0; i < 3; ++i)
		{
			extras[ct++] = effectorPose[i];
		}

		float roll, pitch, yaw;
		quat2rpy(fingerLeftPose.q, roll, pitch, yaw);

		// extras 3-5 end-effector rpy
		extras[ct++] = roll;
		extras[ct++] = pitch;
		extras[ct++] = yaw;
		ExtractChildExtra(ai, extras, ct);

		for (int i = 0; i < mNumExtras; i++){
	        prevExtra[ai][i] = currState[ai][i];
	        currExtra[ai][i] = extras[i];
	    }

	    extras[ct++] = fingerLeftPose.p[1]- robotPoses[ai][1];
	    extras[ct++] = fingerRightPose.p[1]- robotPoses[ai][1];

		int numJoints = motors_right.size();
		vector<float> angles(numJoints, 0.f);
		vector<float> lows(numJoints, 0.f);
		vector<float> highs(numJoints, 0.f);
	    GetAngles(ai, angles, lows, highs);

	    if (angles[numJoints-2] > highs[numJoints-2] || angles[numJoints-2] < lows[numJoints-2]
	        || angles[numJoints-1] > highs[numJoints-1] || angles[numJoints-1] < lows[numJoints-1])
	    {
	        extras[ct++] = 1.0;
	    } else {
	        extras[ct++] = 0.0;
	    }
	    numRobotExtras = ct;
	    ExtractChildExtra(ai, extras, ct);
	}

	// To be overwritten by child env
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) {}
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float* extras, float& rew, bool& dead) {}

    void PreSimulation()
	{
        FlexGymBase2::PreSimulation();

        if (mDoLearning && mNumExtras > 0)
		{
            for (int a = 0; a < mNumAgents; ++a)
            {
                ComputeRewardAndDead(a, GetAction(a), &mObsBuf[a * mNumObservations], &mExtraBuf[a * mNumExtras],
                    mRewBuf[a], (bool&)mDieBuf[a]);
            }
		}
    }

	void ApplyTargetControl(int agentIndex)
	{
	    int numJoints = motors_right.size();
		vector<float> angles(numJoints, 0.f);
		vector<float> lows(numJoints, 0.f);
		vector<float> highs(numJoints, 0.f);
		GetAngles(agentIndex, angles, lows, highs);

		float* actions = GetAction(agentIndex);

		for (int i = 0; i < mNumActions; i++)
		{
		    float action = g_dt * actions[i];
     		//float cc =  Clamp(action, -1.f, 1.f);
    		//float targetPos = 0.5f * (cc + 1.f) * (highs[i] - lows[i]) + lows[i];

            if (i < (mNumActions - 1))
            {
                action = action * sim_params_samples[agentIndex][i+14];
//              float smoothness = 0.7f;
//	    	    float target = Lerp(currentTarget, targetPos, smoothness);
//              float target = Lerp(angles[i], targetPos, smoothness);

                float currentTarget = g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[eNvFlexRigidJointAxisTwist];
                float newTarget = Clamp(angles[i] + action, lows[i], highs[i]);
                float newTargetTarget = Clamp(currentTarget + action, lows[i], highs[i]);
//                if(i == 5)
//                    printf("ACTION %d --> angle %f action %f currTarget %f newTarget %f newTargetTarget %f\n",
//                        i, angles[i], action, currentTarget, newTarget, newTargetTarget);

                g_buffers->rigidJoints[ctrls[agentIndex][i].first].targets[eNvFlexRigidJointAxisTwist] = newTargetTarget;
            }
            else if (controlGripper)
            {
//              float smoothness = 0.2f;
//              float currentTarget = g_buffers->rigidJoints[fingersLeft[agentIndex]].targets[eNvFlexRigidJointAxisX];
//			    float target = Lerp(currentTarget, targetPos, smoothness);

                float currentTarget = g_buffers->rigidJoints[fingersLeft[agentIndex]].targets[eNvFlexRigidJointAxisX];
                float newTarget = Clamp(angles[i] + action, lows[i], highs[i]);
                float newTargetTarget = Clamp(currentTarget + action, lows[i], highs[i]);

				g_buffers->rigidJoints[fingersLeft[agentIndex]].targets[eNvFlexRigidJointAxisX] = newTargetTarget;
				g_buffers->rigidJoints[fingersRight[agentIndex]].targets[eNvFlexRigidJointAxisX] = newTargetTarget;
			}
		}
	}

	void ApplyInverseDynamicsControl(int agentIndex)
	{
		float* actions = GetAction(agentIndex);
		for (int i = 0; i < mNumActions; i++)
		{
			actions[i] = Clamp(actions[i], -1.f, 1.f);
		}

		PerformIDStep(agentIndex, actions[0], actions[1], actions[2], actions[3], actions[4], actions[5], actions[6]);
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
		}

		NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

		float asmoothing = 0.05f;

		rolls[ai] = Lerp(rolls[ai], oroll,asmoothing);
		pitches[ai] = Lerp(pitches[ai], opitch, asmoothing);
		yaws[ai] = Lerp(yaws[ai], oyaw, asmoothing);

		const float smoothing = 0.01f;

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

		float force = g_buffers->rigidJoints[fingersLeft[ai]].lambda[eNvFlexRigidJointAxisX];
		fingerWidths[ai] = Lerp(fingerWidths[ai], newWidth, smoothing);
		g_buffers->rigidJoints[fingersLeft[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
		g_buffers->rigidJoints[fingersRight[ai]].targets[eNvFlexRigidJointAxisX] = fingerWidths[ai];
	}

    void ResetAgent(int ai)
	{
	    SampleInitJointAngles(ai);
		SampleTarget(ai);

        for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second; i++)
        {
			g_buffers->rigidBodies[i] = initBodies[i];
		}

		for (int i = 0; i < mNumActions; i++)
		{
            if (i < (mNumActions - 1))
            {
                g_buffers->rigidJoints[ctrls[ai][i].first].targets[eNvFlexRigidJointAxisTwist] = initJointAnglesAgent[ai][i];
                g_buffers->rigidJoints[ctrls[ai][i].first].motorLimit[eNvFlexRigidJointAxisTwist] = 1.e3;
                g_buffers->rigidJoints[ctrls[ai][i].first].compliance[eNvFlexRigidJointAxisTwist] = 1.e-7f;
				g_buffers->rigidJoints[ctrls[ai][i].first].damping[eNvFlexRigidJointAxisTwist] = 1.e+3f;
            }
            else
            {
                g_buffers->rigidJoints[fingersLeft[ai]].targets[eNvFlexRigidJointAxisX] = initJointAnglesAgent[ai][i];
                g_buffers->rigidJoints[fingersRight[ai]].targets[eNvFlexRigidJointAxisX] = initJointAnglesAgent[ai][i];
//                g_buffers->rigidJoints[fingersLeft[ai]].motorLimit[eNvFlexRigidJointAxisX] = 1.e3;
//                g_buffers->rigidJoints[fingersRight[ai]].motorLimit[eNvFlexRigidJointAxisX] = 1.e3;
            }
		}

        // Move bodies to the desired joint angles. The motor limits were set high to move fast.
        if (ai == (mNumAgents - 1) && (sampleInitJoints || firstReset[ai]))
        {

       		g_buffers->rigidShapes.map();
            for (int i = 0; i < g_buffers->rigidShapes.size(); i++)
            {
                originalFilters[i] = g_buffers->rigidShapes[i].filter;
                g_buffers->rigidShapes[i].filter = 0x1;
            }

            g_buffers->rigidBodies.unmap();
            NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
            g_buffers->rigidJoints.unmap();
            NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());
            g_buffers->rigidShapes.unmap();
            NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

            // Give it some time to reach the initial position.
            for (int s = 0; s < resetNumSteps; s++)
            {
                NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
            }
            g_buffers->rigidShapes.map();
            NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
            g_buffers->rigidBodies.map();
            NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
            g_buffers->rigidJoints.map();

            // Set motor limits to the original values.
            for (int a = 0; a < mNumAgents; a++)
			{
                for (int i = 0; i < mNumActions; i++)
                {
                    if (i < (mNumActions - 1))
					{
                        g_buffers->rigidJoints[ctrls[a][i].first].motorLimit[eNvFlexRigidJointAxisTwist]
                            = motorLimits[i];
                        g_buffers->rigidJoints[ctrls[a][i].first].compliance[eNvFlexRigidJointAxisTwist]
                            = pow(10.0f, sim_params_samples[a][i]);
				        g_buffers->rigidJoints[ctrls[a][i].first].damping[eNvFlexRigidJointAxisTwist]
				            = pow(10.0f, sim_params_samples[a][i+7]);
                    }
                    else
					{
//                        g_buffers->rigidJoints[fingersLeft[a]].motorLimit[eNvFlexRigidJointAxisX] = motorLimits[i];
//                        g_buffers->rigidJoints[fingersRight[a]].motorLimit[eNvFlexRigidJointAxisX] = motorLimits[i];
                    }
                }

                if (!sampleInitJoints)
                {
                    for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
                    {
                        initBodies[i] = g_buffers->rigidBodies[i];
                    }
		        }
            }

            for (int i = 0; i < g_buffers->rigidShapes.size(); i++)
            {
                g_buffers->rigidShapes[i].filter = originalFilters[i];
            }
        }

        firstReset[ai] = false;
		RLFlexEnv::ResetAgent(ai);
	}

    virtual void SampleTarget(int ai){}

    void SampleInitJointAngles(int ai){

        float margin = 0.3f;
        for (int i = 0; i < mNumActions - 1; i++)
		{
            if (sampleInitJoints)
			{
                initJointAnglesAgent[ai][i] = Randf(
                    margin * joint_limits_low[i], margin * joint_limits_high[i]);
            }
			else
			{
                initJointAnglesAgent[ai][i] = initJointAnglesRight[i];
            }
        }
	}

    void SampleSimParams(int ai)
	{
        if (num_sim_params == 0)
		{
            return;
        }

        if (sim_params_stochastic)
		{
            if (!sim_params_presampled)
			{
                Eigen::VectorXd sample = gauss_sample();
                sim_params_samples[ai].resize(sample.size());
                for (int i = 0; i < sample.size(); i++)
                {
                    sim_params_samples[ai][i] = (float)sample[i];
                    sim_params_samples[ai][i] = max(sim_params_samples[ai][i], sim_params_min[i]);
                    sim_params_samples[ai][i] = min(sim_params_samples[ai][i], sim_params_max[i]);
                }
//                if (ai == 0)
//                    printf("Sim Params STOCHASTIC!\n");
            }
			else
			{
//                if (ai == 0)
//                    printf("Sim Params PRESAMPLED!\n");
            }
	    }
		else
		{
//	        if (ai == 0)
//	            printf ("Sim Params MEAN!\n");
	        sim_params_samples[ai].resize(sim_params_mean.size());
	        sim_params_samples[ai] = sim_params_mean;
	    }

//	    printf ("Sim Params Agent %d: [ ", ai);
//        for (int j = 0; j < sim_params_samples[ai].size(); j++)
//		{
//            printf ("%f ", sim_params_samples[ai][j]);
//        }
//        printf("]\n");
//        if (ai == mNumAgents - 1)
//            printf("-----------\n");
    }

	~RLYumiBase()
	{
		if (urdf)
		{
			delete urdf;
		}
	}

	float scaleActions(float minx, float maxx, float x)
	{
		x = 0.5f * (x + 1.f) * (maxx - minx) + minx;
		return x;
	}

	virtual void PreHandleCommunication() {} // Do whatever needed to be done before handling communication
	virtual void ClearContactInfo() {}

	virtual void FinalizeContactInfo() {
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
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

        for (int ai = 0; ai < mNumAgents; ai++)
        {
            forceLeftAgent[ai] = 0.0f;
            forceRightAgent[ai] = 0.0f;
        }

        for (int i = 0; i < numContacts; ++i)
        {
            if (mapFingerLeftToAgent.find(ct[i].body0) != mapFingerLeftToAgent.end())
            {
                forceLeftAgent[mapFingerLeftToAgent.at(ct[i].body0)] += forceScale * ct[i].lambda;
            }
            if (mapFingerRightToAgent.find(ct[i].body0) != mapFingerRightToAgent.end())
            {
                forceRightAgent[mapFingerRightToAgent.at(ct[i].body0)] += forceScale * ct[i].lambda;
            }

            if (mapFingerLeftToAgent.find(ct[i].body1) != mapFingerLeftToAgent.end())
            {
                forceLeftAgent[mapFingerLeftToAgent.at(ct[i].body1)] += forceScale * ct[i].lambda;
            }
            if (mapFingerRightToAgent.find(ct[i].body1) != mapFingerRightToAgent.end())
            {
                forceRightAgent[mapFingerRightToAgent.at(ct[i].body1)] += forceScale * ct[i].lambda;
            }
        }
        rigidContacts.unmap();
        rigidContactCount.unmap();
	}

	virtual void LockWrite() {} // Do whatever needed to lock write to simulation
	virtual void UnlockWrite() {} // Do whatever needed to unlock write to simulation

	virtual void DoGui()
	{
		if (!mDoLearning)
		{

            Transform fingerLeftPose;
            Transform fingerRightPose;
            NvFlexGetRigidPose(&g_buffers->rigidBodies[fingerLeftBody[0]], (NvFlexRigidPose*)&fingerLeftPose);
            NvFlexGetRigidPose(&g_buffers->rigidBodies[fingerRightBody[0]], (NvFlexRigidPose*)&fingerRightPose);
            Vec3 effectorPose = (fingerLeftPose.p + fingerRightPose.p) / 2.0 - robotPoses[0];

            NvFlexRigidJoint effector0_0 = g_buffers->rigidJoints[effectorJoints[0]];

			float targetx = effector0_0.pose0.p[0] - robotPoses[0].x;
			float targety = effector0_0.pose0.p[1] - robotPoses[0].y;
			float targetz = effector0_0.pose0.p[2] - robotPoses[0].z;

			float oroll = rolls[0];
			float opitch = pitches[0];
			float oyaw = yaws[0];

			imguiSlider("Gripper X", &targetx, -0.3f, 0.3f, 0.001f);
			imguiSlider("Gripper Y", &targety, 0.4f, 1.0f, 0.001f);
			imguiSlider("Gripper Z", &targetz, 0.0f, 0.8f, 0.001f);
			imguiSlider("Roll", &rolls[0], -180.0f, 180.0f, 0.01f);
			imguiSlider("Pitch", &pitches[0], -180.0f, 180.0f, 0.01f);
			imguiSlider("Yaw", &yaws[0], -180.0f, 180.0f, 0.01f);

			float newWidth = fingerWidths[0];
			imguiSlider("Finger Width", &newWidth, limitsGrip[0], limitsGrip[1], 0.001f);

			for (int ai = 0; ai < mNumAgents; ++ai)
			{
				NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoints[ai]];

				float f = 0.05f;

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

	virtual void DoStats()
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

	virtual void Update()
	{
		if (doStats)
		{
			for (int ai = 0; ai < 1; ++ai)
			{
				// record force on the 1st robot finger joints
//		        printf("FORCES FING L %d FING R %d --> %f %f\n", fingersLeft[ai], fingersRight[ai],
//                    g_buffers->rigidJoints[fingersLeft[ai]].lambda[eNvFlexRigidJointAxisX],
//                    g_buffers->rigidJoints[fingersRight[ai]].lambda[eNvFlexRigidJointAxisX]);
				forceLeft.push_back(g_buffers->rigidJoints[fingersLeft[ai]].lambda[eNvFlexRigidJointAxisX]);
				forceRight.push_back(g_buffers->rigidJoints[fingersRight[ai]].lambda[eNvFlexRigidJointAxisX]);
			}
		}
	}

	virtual void PostUpdate()
	{
		// joints are not read back by default
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
	}
};



class RLYumiReach : public RLYumiBase
{
public:

	vector<int> targetSphere;
	vector<int> targetRenderMaterials;
	Vec3 redColor;
	Vec3 greenColor;
	vector<Vec3> targetPoses;
	float radius;

	RLYumiReach()
	{
		mNumObservations = numRobotStates + 3; // state of the robot + 3 target xyz
		doStats = false;
		radius = 0.04f;
		redColor = Vec3(1.0f, 0.04f, 0.07f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);
	}

	void LoadChildEnv()
	{
		targetPoses.resize(mNumAgents, Vec3(0.4f, 1.f, 0.8f));
		targetRenderMaterials.resize(mNumAgents);
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
			targetPoses[ai] = Vec3(Randf(-0.5f, 0.2f), Randf(0.3f, 1.0f), Randf(0.3f, 0.7f));
		}
		else
		{
			targetPoses[ai] = Vec3(sim_params_samples[ai][num_robot_sim_params],
			                       sim_params_samples[ai][num_robot_sim_params+1],
			                       sim_params_samples[ai][num_robot_sim_params+2]);
		}

		NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], Quat());
		if (ai == 0) {
		    g_buffers->rigidShapes.map();
		}
		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
//		g_buffers->rigidShapes.unmap();
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
		// state 14-16: target xyz
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = targetPoses[ai][i];
		}
	}

	void ComputeRewardAndDead(int a, float* action, float* state, float* extras, float& rew, bool& dead)
	{
		float reg = 0.f;
		float* prevAction = GetPrevAction(a);
		for (int i = 0; i < mNumActions - 1; i++)
		{
//			reg += Pow((action[i] - prevAction[i]), 2) / (float)mNumActions;
			reg += Pow(action[i], 2);
		}
//		float dist = Length(Vec3(state[8] - state[numRobotStates],
//		                         state[9] - state[numRobotStates+1],
//		                         state[10] - state[numRobotStates+2]));

        float dist = Length(Vec3(extras[0] - state[numRobotStates],
                                 extras[1] - state[numRobotStates+1],
                                 extras[2] - state[numRobotStates+2]));

		float distReward = 2.f * exp(-5.f * dist);// - 0.4f * reg;

        rew = rewardWeights[0] * dist +
              rewardWeights[1] * reg;

		float x = max((distReward - 0.2f) / 1.79f, 0.f);
		g_renderMaterials[targetRenderMaterials[a]].frontColor = x * greenColor + (1.f - x) * redColor;
		dead = false;
	}
};


class RLYumiCabinet : public RLYumiBase
{
public:

	URDFImporter* cabinet_urdf;
	vector<Vec3> targetPoses;
	vector<int> topDrawerHandleBodyIds;
	vector<int> topDrawerHandleShapeIds;
	vector<int> topDrawerJoints;

    vector<Vec3> drawerPose;
    vector<int> cabinetStartShape;
    vector<int> cabinetEndShape;
    int cabinetMaterial;

	RLYumiCabinet()
	{
		mNumObservations = numRobotStates + 3; // state of the robot + 3 drawer handle xyz
		doStats = false;

		g_lightDir = Normalize(Vec3(-5.0f, 15.0f, -7.5f));
	}

	void LoadChildEnv()
	{
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

    	cabinetMaterial = AddRenderMaterial(Vec3(0.4f, 0.5f, 0.8f), 0, 0, true);
	}

	void SampleTarget(int ai)
	{
	    if (num_sim_params == 0) return;

        SampleSimParams(ai);
        int id = topDrawerHandleShapeIds[ai];
//        g_buffers->rigidShapes.map();
//        g_buffers->rigidShapes[id].geo.triMesh.scale[0] = sim_params_orig[ai][0] + sim_params_samples[ai][0];
//        g_buffers->rigidShapes[id].geo.triMesh.scale[1] = sim_params_orig[ai][1] + sim_params_samples[ai][1];
//        g_buffers->rigidShapes[id].geo.triMesh.scale[2] = sim_params_orig[ai][2] + sim_params_samples[ai][2];
//        g_buffers->rigidShapes[id].pose.p[0] = sim_params_orig[ai][3] + sim_params_samples[ai][3];
//        g_buffers->rigidShapes[id].pose.p[1] = sim_params_orig[ai][4] + sim_params_samples[ai][4];
//        g_buffers->rigidShapes[id].pose.p[0] = sim_params_orig[ai][5] + sim_params_samples[ai][5];
//        g_buffers->rigidShapes.unmap();
	}

	virtual void AddChildEnvBodies(int ai, Transform gt)
	{
//    	if(num_sim_params > 0)
//    	    sim_params_orig[ai].resize(num_sim_params);
//	    Transform cabinet_transform(Vec3(0.0f, 0.005f,0.23f) + robotPoses[ai],
//  		        QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(
//  		            Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));

  		 Transform cabinet_transform(Vec3(0.0f, -0.01f, 0.23f) + robotPoses[ai],
  		        QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(
  		            Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));

        cabinetStartShape[ai] = g_buffers->rigidShapes.size();
		int startBody = g_buffers->rigidBodies.size();
        cabinet_urdf->AddPhysicsEntities(cabinet_transform, cabinetMaterial, true, 1000.0f, 0.0f, 1e1f, 0.1f, 20.7f, 7.0f, false);
		cabinetEndShape[ai] = g_buffers->rigidShapes.size();
		int endBody = g_buffers->rigidBodies.size();

        for (int i = cabinetStartShape[ai]; i < cabinetEndShape[ai]; i++)
		{
			g_buffers->rigidShapes[i].material.friction = 0.8f;
//			g_buffers->rigidShapes[i].material.rollingFriction += 1.f;
            if (i == cabinet_urdf->rigidShapeNameMap["drawer_handle_top"].first)
			{
			    topDrawerHandleShapeIds[ai] = i;
                g_buffers->rigidShapes[i].thickness -= 0.007f;
//                if (num_sim_params > 0)
//				{
//                    sim_params_orig[ai][0] = g_buffers->rigidShapes[i].geo.triMesh.scale[0];
//                    sim_params_orig[ai][1] = g_buffers->rigidShapes[i].geo.triMesh.scale[1];
//                    sim_params_orig[ai][2] = g_buffers->rigidShapes[i].geo.triMesh.scale[2];
//                    sim_params_orig[ai][3] = g_buffers->rigidShapes[i].pose.p[0];
//                    sim_params_orig[ai][4] = g_buffers->rigidShapes[i].pose.p[1];
//                    sim_params_orig[ai][5] = g_buffers->rigidShapes[i].pose.p[0];
//                }
            }
            else if (i == cabinet_urdf->rigidShapeNameMap["drawer_top"].first)
			{
//                g_buffers->rigidShapes[i].thickness += 0.015f;
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
//
//                if (j->name == "drawer_top_joint") {
//                    topDrawerJoints[ai] = cabinet_urdf->jointNameMap[j->name];
//                }

                if (j->name == "drawer_bottom_joint") {
                    joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;	// 10^6 N/m
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
  	    drawerPose[ai] = pose.p - robotPoses[ai];

//		NvFlexRigidShape targetShape;
//		NvFlexMakeRigidSphereShape(&targetShape, -1, 0.04f, NvFlexMakeRigidPose(0,0));
//		int renderMaterial = AddRenderMaterial(Vec3(1.0f, 0.04f, 0.07f));
//		targetShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.0f,0.0f,0.0f)));
//		targetShape.group = 1;
//		targetSphere[ai] = g_buffers->rigidShapes.size();
//		g_buffers->rigidShapes.push_back(targetShape);
//        Vec3 targetOffset(0.0, 0.015, -0.32);
//		NvFlexRigidPose poseTarget = NvFlexMakeRigidPose(drawerPose[ai] + robotPoses[ai] + targetOffset, Quat());
//		g_buffers->rigidShapes[targetSphere[ai]].pose = poseTarget;
	}



void ExtractChildState(int ai, float* state, int ct)
	{
	    Transform pose;
	    NvFlexGetRigidPose(&g_buffers->rigidBodies[topDrawerHandleBodyIds[ai]],
			    (NvFlexRigidPose*)&pose);
  	    Vec3 drawer_pose = pose.p - robotPoses[ai];
        drawer_pose[1] += 0.015;
		// top drawer handle xyz
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = drawer_pose[i]; //drawerPose[ai][i];
		}

	}

	void ComputeRewardAndDead(int a, float* action, float* state, float* extras, float& rew, bool& dead)
	{
		float reg = 0.f;
		float* prevAction = GetPrevAction(a);
		for (int i = 0; i < mNumActions; i++)
		{
			reg += Pow((action[i] - prevAction[i]), 2.f) / (float)mNumActions;
//			reg += Pow(action[i], 2);
		}

		//-0.000000 0.681196 1.027004
		// Drawer handle position 0.000000 0.681200 1.028500
		// Drawer handle position at handle -0.001400 0.655344 0.697661

		// eff -0.002711  0.686056  0.664384
		// rpy 0.000000, PITCH 1.570796, YAW -0.003042
        // dra -0.000000 0.702200 1.026126

		float handle_z_rew = (state[numRobotStates+2] - 1.0f); //0.5*state[16];

        float forceRew = (abs(forceLeftAgent[a]) + abs(forceRightAgent[a]));
//		float dist = 0.5f * Length(Vec3(
//		            state[8] - state[numRobotStates],
//		            state[9] - state[numRobotStates+1],
//		            state[10] - (state[numRobotStates+2] - 0.361742f)));
//		float angleDist = 0.05f * (Length(Vec3(state[11] - 0.0f, state[12] - 1.570796f, state[13] - (-0.003042f)))
//		    + 2.0f*abs(state[11] - 0.0f + state[12] - 1.570796f + state[13] - (-0.003042f)));

		float dist = Length(Vec3(
			            extras[0] - state[numRobotStates],
		                extras[1] - state[numRobotStates+1],
		                extras[2] - (state[numRobotStates+2] - 0.361742f)));
		float angleDist = (Length(Vec3(extras[3] - 0.0f, extras[4] - 1.570796f, extras[5] - (-0.003042f)))
		    + 2.0f*abs(extras[3] - 0.0f + extras[4] - 1.570796f + extras[5] - (-0.003042f)));

        float aroundHandleRew = 0.0;
		if (extras[6] < state[numRobotStates+1] && extras[7] > state[numRobotStates+1]){
		    aroundHandleRew = 1.0;
		}

		float fingerOutOfLimitPenalty = 0.0;
		if (extras[8] > 0.0){
		    fingerOutOfLimitPenalty = 1.0;
		}

//        if (forceRew > 0.0) {
//            printf("FORC REW %f Dist %f Angle %f reg %f Z %f AROUND %f\n",
//                forceRew, dist, angleDist, reg, handle_z_rew, aroundHandleRew);
//        }
//
//        printf("REW dist %f angleDist %f reg %f handle_z_rew %f forceRew %f aroundHandleRew %f fingerLimit %f\n",
//                rewardWeights[0] * dist, rewardWeights[1] * angleDist, rewardWeights[2] * reg,
//                rewardWeights[3] * handle_z_rew, rewardWeights[4] * forceRew,
//                rewardWeights[5] * aroundHandleRew, rewardWeights[6] * fingerOutOfLimitPenalty);

        rew = rewardWeights[0] * dist +
              rewardWeights[1] * angleDist +
              rewardWeights[2] * reg +
              rewardWeights[3] * handle_z_rew +
              rewardWeights[4] * forceRew +
              rewardWeights[5] * aroundHandleRew +
              rewardWeights[6] * fingerOutOfLimitPenalty;

		dead = false;
	}
};


class RLYumiRopePeg : public RLYumiBase
{
public:
	vector<int> targetPegHolders;
	vector<float> pegMasses;
	vector<int> pegBodyIds;
	vector<int> pegHolderBodyIds;
	vector<int> pegHolderShapeIds;
	vector<bool> pegContacts;
	vector<float> pegForces;

	vector<vector<int>> ropeBodyIds;
	const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
	const bool connectRopeToRobot = true;
	float tableLength, tableWidth;
    Vec3 tableOrigin;

    // Table
    vector<float> tableHeight;
    vector<float> tableFriction;

    // Rope
    vector<int> ropeNumSegments;
    vector<float> ropeSegmentRollingFriction;
    vector<float> ropeSegmentFriction;
    vector<float> ropeSegmentInertiaShift;
    vector<float> ropeSegmentLength;
    vector<float> ropeSegmentWidth;
    vector<float> ropeDensity;
    vector<float> ropeBendingCompliance;
    vector<float> ropeTorsionCompliance;
	vector<float> ropeTorsionDamping;
	vector<float> ropeBendingDamping;

    // Peg
    vector<float> pegScaleX;
    vector<float> pegScaleY;
    vector<float> pegScaleZ;
    vector<float> pegFriction;
    vector<float> pegThickness;
    vector<float> pegDensity;
    vector<float> pegMassMult;

    // Peg holder
    vector<float> pegHolderDensity;
    vector<float> pegHolderScaleXZ;
    vector<float> pegHolderScaleY;
    vector<float> pegHolderFriction;
    vector<float> pegHolderThickness;
    vector<float> pegHolderPosX;
    vector<float> pegHolderPosY;
    vector<float> pegHolderPosZ;
    vector<float> pegHolderRoll;
    vector<float> pegHolderPitch;
    vector<float> pegHolderYaw;

	RLYumiRopePeg()
	{
	    mNumObservations = numRobotStates + 9;
		tableLength = 0.55f;
		tableWidth = 0.4f;
		tableOrigin = Vec3(0.0f, 0.0f, 0.7f);
		mNumExtras = 9 + 3;

	//	g_params.solverType = eNvFlexSolverPBD;
	//	g_numSubsteps = 4;
	//	g_params.numIterations = 50;// 5; //4;
	//	g_params.numInnerIterations = 20;//20;
	}

	void LoadChildEnv()
	{
		initEffectorTranslation = Vec3(0.f, 0.8, 0.5f);
		targetPegHolders.resize(mNumAgents);
		pegMasses.resize(mNumAgents);
		pegBodyIds.resize(mNumAgents);
		pegHolderBodyIds.resize(mNumAgents);
		pegHolderShapeIds.resize(mNumAgents);
		pegContacts.resize(mNumAgents);
		pegForces.resize(mNumAgents);
		targetPoses.resize(mNumAgents);
		targetSphere.resize(mNumAgents);
		ropeBodyIds.resize(mNumAgents);

        // Table
		tableHeight.resize(mNumAgents,                  0.0001f); // std 0.006
        tableFriction.resize(mNumAgents,                0.7f); // std 0.14

        // Rope
        ropeNumSegments.resize(mNumAgents,              10); // std 2
        ropeSegmentRollingFriction.resize(mNumAgents,   0.001f); // std 0.0002
        ropeSegmentFriction.resize(mNumAgents,          0.25f); // std 0.05
        ropeSegmentInertiaShift.resize(mNumAgents,      0.0f); // std 0.0001
        ropeSegmentLength.resize(mNumAgents,            0.014f); // std 0.0028
        ropeSegmentWidth.resize(mNumAgents,             0.004f); // std 0.0008
        ropeDensity.resize(mNumAgents,                  2500.f); // std 200.0
        ropeBendingCompliance.resize(mNumAgents,        1e+1f); // std 2.0
        ropeTorsionCompliance.resize(mNumAgents,        2.f); // std 0.4
        ropeTorsionDamping.resize(mNumAgents,           0.1f); // std 0.05
        ropeBendingDamping.resize(mNumAgents,           0.01f); // std 0.005

        // Peg
        pegScaleX.resize(mNumAgents,                    0.028f); // std 0.0056
        pegScaleY.resize(mNumAgents,                    0.03f); // std 0.006
        pegScaleZ.resize(mNumAgents,                    0.028f); // std 0.0056
        pegFriction.resize(mNumAgents,                  1.0f); // std 0.2
        pegThickness.resize(mNumAgents,                 0.005f); // std 0.0004
        pegDensity.resize(mNumAgents,                   400.0f); // std 80.0
        pegMassMult.resize(mNumAgents,                  1.0); // std 0.2

        // Peg holder
        pegHolderDensity.resize(mNumAgents,             1000.0f); // std 200.0
        pegHolderScaleXZ.resize(mNumAgents,              0.033f); // std 0.006
        pegHolderScaleY.resize(mNumAgents,              0.5f); // std 0.01
        pegHolderFriction.resize(mNumAgents,            1.0f); // std 0.2
        pegHolderThickness.resize(mNumAgents,           0.002f); // std 0.0004

        pegHolderPosX.resize(mNumAgents,                -0.15f); // std 0.04
        pegHolderPosY.resize(mNumAgents,               0.11f); // std 0.005
        pegHolderPosZ.resize(mNumAgents,                0.78f); // std 0.1
        pegHolderRoll.resize(mNumAgents,                -0.59f); // std 0.04
        pegHolderPitch.resize(mNumAgents,               0.0f); // std 0.005
        pegHolderYaw.resize(mNumAgents,                 0.0f); // std 0.1

//        pegHolderPosX.resize(mNumAgents,                -0.150710f); // std 0.04
//        pegHolderPosY.resize(mNumAgents,                0.03f + 0.02f); // std 0.005
//        pegHolderPosZ.resize(mNumAgents,                0.557756f); // std 0.1
//        pegHolderRoll.resize(mNumAgents,                0.0f); // std 0.04
//        pegHolderPitch.resize(mNumAgents,               0.0f); // std 0.005
//        pegHolderYaw.resize(mNumAgents,                 0.0f); // std 0.1
	}

    void SetSimParams(int ai)
	{
        if(num_sim_params == 0)
		{
            return;
        }
        SampleSimParams(ai);

        int ct = num_robot_sim_params;
        // Table
        tableHeight[ai] = sim_params_samples[ai][ct++];
        tableFriction[ai] = sim_params_samples[ai][ct++];

        // Rope
        ropeNumSegments[ai] = (int)sim_params_samples[ai][ct++];
        ropeSegmentRollingFriction[ai] = sim_params_samples[ai][ct++];
        ropeSegmentFriction[ai] = sim_params_samples[ai][ct++];
        ropeSegmentInertiaShift[ai] = sim_params_samples[ai][ct++];
        ropeSegmentLength[ai] = sim_params_samples[ai][ct++];
        ropeSegmentWidth[ai] = sim_params_samples[ai][ct++];
        ropeDensity[ai] = sim_params_samples[ai][ct++];
        ropeBendingCompliance[ai] = sim_params_samples[ai][ct++];
        ropeTorsionCompliance[ai] = sim_params_samples[ai][ct++];
        ropeTorsionDamping[ai] = sim_params_samples[ai][ct++];
        ropeBendingDamping[ai] = sim_params_samples[ai][ct++];

        // Peg
        pegScaleX[ai] = sim_params_samples[ai][ct++];
        pegScaleY[ai] = sim_params_samples[ai][ct++];
        pegScaleZ[ai] = sim_params_samples[ai][ct++];
        pegFriction[ai] = sim_params_samples[ai][ct++];
        pegThickness[ai] = sim_params_samples[ai][ct++];
        pegDensity[ai] = sim_params_samples[ai][ct++];
        pegMassMult[ai] = sim_params_samples[ai][ct++];

        // Peg holder
        pegHolderDensity[ai] = sim_params_samples[ai][ct++];
        pegHolderScaleXZ[ai] = sim_params_samples[ai][ct++];
        pegHolderScaleY[ai] = sim_params_samples[ai][ct++];
        pegHolderFriction[ai] = sim_params_samples[ai][ct++];
        pegHolderThickness[ai] = sim_params_samples[ai][ct++];
        pegHolderPosX[ai] = sim_params_samples[ai][ct++];
        pegHolderPosY[ai] = sim_params_samples[ai][ct++];
        pegHolderPosZ[ai] = sim_params_samples[ai][ct++];
        pegHolderRoll[ai] = sim_params_samples[ai][ct++];
        pegHolderPitch[ai] = sim_params_samples[ai][ct++];
        pegHolderYaw[ai] = sim_params_samples[ai][ct++];
    }

	void AddChildEnvBodies(int ai, Transform gt)
	{
	    SetSimParams(ai);
        ropeBodyIds[ai].resize(ropeNumSegments[ai]);
        tableOrigin = Vec3(0.0f, -robotPoses[ai][1], 0.7f);


//	    // Create table
//	    NvFlexRigidShape table;
//		NvFlexMakeRigidBoxShape(&table, -1, tableLength, 0.000005, tableWidth, NvFlexMakeRigidPose(tableOrigin + robotPoses[ai], Quat()));
////		table.filter = 0x0;
////		table.group = 0;
//		table.material.friction = tableFriction[ai];
//		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.17f, 0.24f, 0.4f)));
//
//		g_buffers->rigidShapes.push_back(table);

		// Create target peg holder
		Mesh* pegHolderMesh = ImportMesh("../../data/peg_holder_bottomright_top.obj");//"../../data/peg_holder.obj");
		pegHolderMesh->Transform(ScaleMatrix(Vec3(pegHolderScaleXZ[ai], pegHolderScaleY[ai], pegHolderScaleXZ[ai])));

		NvFlexTriangleMeshId pegHolderMeshId = CreateTriangleMesh(pegHolderMesh, 0.005f);

		NvFlexRigidShape pegHolderShape;
		NvFlexMakeRigidTriangleMeshShape(&pegHolderShape, g_buffers->rigidBodies.size(),
		    pegHolderMeshId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);

		pegHolderShape.filter = 0x4;
		pegHolderShape.group = 0;
		pegHolderShape.material.friction = pegHolderFriction[ai];
		pegHolderShape.thickness = pegHolderThickness[ai];
		pegHolderShapeIds[ai] = g_buffers->rigidShapes.size();
		g_buffers->rigidShapes.push_back(pegHolderShape);

		NvFlexRigidBody pegHolderBody;
		NvFlexMakeRigidBody(g_flexLib, &pegHolderBody, Vec3(), Quat(), &pegHolderShape, &pegHolderDensity[ai], 1);
		pegHolderBodyIds[ai] = g_buffers->rigidBodies.size();
		pegHolderBody.invMass = 0.0f;
		(Matrix33&)pegHolderBody.invInertia = Matrix33();
		g_buffers->rigidBodies.push_back(pegHolderBody);

		// Create rope
		NvFlexRigidJoint handRight = g_buffers->rigidJoints[urdf->jointNameMap["gripper_r_joint"]];

//		Vec3 startPos = Vec3(0.f, .75f, 1.f) + robotPoses[ai];
        Transform handRightPose;
	    NvFlexGetRigidPose(&g_buffers->rigidBodies[handRight.body0], (NvFlexRigidPose*)&handRightPose);
        Vec3 startPos = Vec3(0.01f, .0f, .1f) + handRightPose.p;

		NvFlexRigidPose prevJoint;
		int lastRopeBodyIndex = 0;

		for (int i = 0; i < ropeNumSegments[ai]; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, ropeSegmentWidth[ai], ropeSegmentLength[ai], NvFlexMakeRigidPose(0, Quat(.0f, .0f, .707f, .707f)));
			shape.filter = 0x1;
			shape.group = 0;

			shape.material.rollingFriction = ropeSegmentRollingFriction[ai];
			shape.material.friction = ropeSegmentFriction[ai];
			shape.user = UnionCast<void*>(linkMaterial);

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, startPos +
			    Vec3(0.0f, -(i*ropeSegmentLength[ai] * 2.f + ropeSegmentLength[ai]), 0.0f),
			    Quat(), &shape, &ropeDensity[ai], 1);
            (Matrix33&)body.inertia += Matrix33::Identity() * ropeSegmentInertiaShift[ai];
			body.maxAngularVelocity = 50.f;

            bool success;
            (Matrix33&)body.invInertia = Inverse((Matrix33&)body.inertia, success);

			ropeBodyIds[ai][i] = g_buffers->rigidBodies.size();
			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

			if (i == 0 && !connectRopeToRobot)
			{
				prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -ropeSegmentLength[ai], .0f), Quat());
				continue;
			}

			NvFlexRigidJoint joint;
			if (i == 0)
			{
				NvFlexMakeFixedJoint(&joint, handRight.body0, bodyIndex,
//					NvFlexMakeRigidPose(Vec3(0.01f, .0f, .1f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5)),
            		NvFlexMakeRigidPose(Vec3(0.03f, .0f, .1f), Quat()),//QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)),
					NvFlexMakeRigidPose(0, QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), -kPi * 0.5f)));
			}
			else
			{
				NvFlexMakeFixedJoint(&joint, bodyIndex - 1, bodyIndex, prevJoint,
									NvFlexMakeRigidPose(Vec3(0.f, ropeSegmentLength[ai], 0.f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)));
			}

			joint.compliance[eNvFlexRigidJointAxisTwist] = ropeTorsionCompliance[ai];
			joint.compliance[eNvFlexRigidJointAxisSwing1] = ropeBendingCompliance[ai];
			joint.compliance[eNvFlexRigidJointAxisSwing2] = ropeBendingCompliance[ai];

			joint.damping[eNvFlexRigidJointAxisTwist] = ropeTorsionDamping[ai];
			joint.damping[eNvFlexRigidJointAxisSwing1] = ropeBendingDamping[ai];
			joint.damping[eNvFlexRigidJointAxisSwing2] = ropeBendingDamping[ai];

			g_buffers->rigidJoints.push_back(joint);
			lastRopeBodyIndex = bodyIndex;
			prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -ropeSegmentLength[ai], .0f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f));
		}

		// Peg
		Mesh* pegMesh = ImportMesh("../../data/cylinder.obj");
		pegMesh->Transform(ScaleMatrix(Vec3(pegScaleX[ai], pegScaleY[ai], pegScaleZ[ai])));

		NvFlexTriangleMeshId pegId = CreateTriangleMesh(pegMesh, 0.005f);
		NvFlexRigidShape pegShape;
		NvFlexMakeRigidTriangleMeshShape(&pegShape, g_buffers->rigidBodies.size(), pegId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
		pegShape.material.friction = pegFriction[ai];
		pegShape.thickness = pegThickness[ai];
		pegShape.filter = 0x0;
		pegShape.group = 0;
		pegShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(.9f, .9f, .3f)));
		g_buffers->rigidShapes.push_back(pegShape);

		NvFlexRigidBody pegBody;
		NvFlexMakeRigidBody(g_flexLib, &pegBody, startPos + Vec3(0.0f, -float(ropeNumSegments[ai] + 2) * ropeSegmentLength[ai] * 2.f, 0.f),
							Quat(), &pegShape, &pegDensity[ai], 1);
		pegMasses[ai] = pegBody.mass;
		pegBodyIds[ai] = g_buffers->rigidBodies.size();
		pegBody.maxAngularVelocity = 50.f;
		g_buffers->rigidBodies.push_back(pegBody);

		// Connecting peg to rope
		NvFlexRigidJoint joint;
		int bodyIndex = g_buffers->rigidBodies.size();
		NvFlexMakeFixedJoint(&joint, lastRopeBodyIndex, bodyIndex - 1, 
						NvFlexMakeRigidPose(Vec3(0.f, -3.75f * ropeSegmentLength[ai], 0.f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)), NvFlexMakeRigidPose(0, QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)));

		joint.compliance[eNvFlexRigidJointAxisTwist] = 0.1f * ropeTorsionCompliance[ai];
		joint.compliance[eNvFlexRigidJointAxisSwing1] = 0.2f * ropeBendingCompliance[ai];
		joint.compliance[eNvFlexRigidJointAxisSwing2] = 0.2f * ropeBendingCompliance[ai];

		joint.damping[eNvFlexRigidJointAxisTwist] = ropeTorsionDamping[ai];
		joint.damping[eNvFlexRigidJointAxisSwing1] = ropeBendingDamping[ai];
		joint.damping[eNvFlexRigidJointAxisSwing2] = ropeBendingDamping[ai];

		g_buffers->rigidJoints.push_back(joint);

		SampleTarget(ai);

//		NvFlexRigidShape targetShape;
//		NvFlexMakeRigidSphereShape(&targetShape, -1, 0.04f, NvFlexMakeRigidPose(0,0));
//
//		int renderMaterial = AddRenderMaterial(Vec3(1.0f, 0.02f, 0.07f));
//		targetShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.0f,0.0f,0.0f)));
//		targetShape.group = 1;
//		targetSphere[ai] = g_buffers->rigidShapes.size();
//		g_buffers->rigidShapes.push_back(targetShape);
//
//		NvFlexRigidPose pose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai] + targetOffset, Quat());
//		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
			targetPoses[ai] = Vec3(Randf(-0.4f, 0.1f), tableHeight[ai] + 0.02f, Randf(0.4f, 0.8f));
		}
		else
		{
			targetPoses[ai] = Vec3(pegHolderPosX[ai], pegHolderPosY[ai], pegHolderPosZ[ai]) + robotPoses[ai];
		}

		Quat rot = rpy2quat(pegHolderRoll[ai], pegHolderPitch[ai], pegHolderYaw[ai]);


		NvFlexRigidPose newPose = NvFlexMakeRigidPose(targetPoses[ai], rot);
		NvFlexSetRigidPose(&g_buffers->rigidBodies[pegHolderBodyIds[ai]], &newPose);
//
//	    float pegMass;
//		if (sampleInitStates)
//		{
//			pegMass = pegMasses[ai] * Randf(0.8f, 1.1f);
//		}
//		else
//		{
//			pegMass = pegMasses[ai];
//		}
//
//		g_buffers->rigidBodies[pegBodyIds[ai]].mass = pegMass;
	}

	void ExtractChildState(int a, float* state, int ct)
	{
		// 0-2 target xyz
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = targetPoses[a][i] - robotPoses[a][i];
		}

		// 3-5 peg xyz
		Transform pegPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[pegBodyIds[a]], (NvFlexRigidPose*)&pegPoseWorld);
		Vec3 pegTraLocal = pegPoseWorld.p - robotPoses[a];
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = pegTraLocal[i];
		}

        // 6-8 previous peg xyz
		for (int i = 0; i < 3; ++i)
		{
			state[ct] = prevState[a][ct-3];
			ct++;
		}

//		// 6-8 peg rpy
//		float r, p, y;
//		quat2rpy(pegPoseWorld.q, r, p, y);
//		state[ct++] = r;
//		state[ct++] = p;
//		state[ct++] = y;

//		// 23-25 peg velocities
//		NvFlexRigidBody pegBody = g_buffers->rigidBodies[pegBodyIds[a]];
//		for (int i = 0; i < 3; ++i)
//		{
//			state[ct++] = pegBody.linearVel[i];
//		}
//		for (int i = 0; i < 3; ++i)
//		{
//			state[ct++] = pegBody.angularVel[i];
//		}
//
//		// 25-27 contact flag and force
//		state[ct++] = pegContacts[a] ? 1.f : 0.f;
//		state[ct++] = pegForces[a];
//
//		// +9 * ropeSegments rope segment translation and rotations
//		Transform segPoseWorld;
//		for (int i = 0; i < ropeSegments; i++)
//		{
//			int segBodyId = ropeBodyIds[a][i];
//			NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
//			Vec3 segTraLocal = segPoseWorld.p - robotPoses[a];
//			NvFlexRigidBody segBody = g_buffers->rigidBodies[segBodyId];
//
//			for (int j = 0; j < 3; j++)
//			{
//				state[ct++] = segTraLocal[j];
//				state[ct++] = segBody.linearVel[j];
//				state[ct++] = segBody.angularVel[j];
//			}
//		}
	}

    virtual void ExtractChildExtra(int a, float* extras, int ct) {
        Transform pegPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[pegBodyIds[a]], (NvFlexRigidPose*)&pegPoseWorld);

        // 0-3 peg rpy
		float r, p, y;
		quat2rpy(pegPoseWorld.q, r, p, y);
		extras[ct++] = r;
		extras[ct++] = p;
		extras[ct++] = y;
    }

	void ComputeRewardAndDead(int a, float* action, float* state, float* extras, float& rew, bool& dead)
	{
		float rewTarget = 0.f, rewVertical = 0.f, rewAction = 0.f, rewSolved = 0.f;
		bool solved = false;
		float targetReachWeight = 20.f;
		float verticalAngleWeight = 0.5f;

		// reward for getting close to peg holder
		Vec3 targetPos = Vec3(state[numRobotStates], state[numRobotStates+1], state[numRobotStates+2]);
		Vec3 pegPos = Vec3(state[numRobotStates+3], state[numRobotStates+4], state[numRobotStates+5]);
		Vec3 pegPrevPos = Vec3(prevState[a][numRobotStates+3], prevState[a][numRobotStates+4], prevState[a][numRobotStates+5]);

		Vec3 pegToTarget = pegPos - targetPos;
		float rewTargetL2 = Length(pegToTarget);
		float rewTargetL1 = abs(pegToTarget[0]) + abs(pegToTarget[1]) + abs(pegToTarget[2]);

		// reward for maintaining verticality of peg
		Quat pegRot = rpy2quat(extras[numRobotExtras], extras[numRobotExtras+1], extras[numRobotExtras+2]);
		float pegAngle0 = RadToDeg(Dot(GetBasisVector1(pegRot), Vec3(1.f, 0.f, 0.f)));
		float pegAngle1 = RadToDeg(Dot(GetBasisVector1(pegRot), Vec3(0.f, 1.f, 0.f)));
		float pegAngle2 = RadToDeg(Dot(GetBasisVector1(pegRot), Vec3(0.f, 0.f, 1.f)));

		float pegAngleToVertical = abs(pegAngle2 - (-45.0f));
//		rewVertical = verticalAngleWeight * exp(-0.1*pegAngleToVertical);
		rewVertical = pegAngleToVertical;

//		Quat pegPrevRot = rpy2quat(prevExtra[a][0], prevExtra[a][1], prevExtra[a][2]);
//		float pegPrevAngle = RadToDeg(Dot(GetBasisVector1(pegPrevRot), Vec3(0.f, 1.f, 0.f)));

        float pegPosChange = Length(pegPos - pegPrevPos);
        float pegRotChange = 0.0f;//abs(pegAngle - pegPrevAngle);
        float rewPegVel = pegPosChange;
        float rewPegRotChange = pegRotChange;

		float* prevAction = GetPrevAction(a);
		for (int i = 0; i < mNumActions - 1; i++)
		{
//			rewAction += -0.8f * Pow(action[i] - prevAction[i], 2) / (float)mNumActions;
    		rewAction += Pow(action[i], 2);
		}

        Vec3 targetOffset(0.0f, -0.04f, -0.04f * tan(pegHolderRoll[a]));
		Vec3 pegToSolved = pegPos - (targetPos + targetOffset);
		float rewDistSolved = Length(pegToSolved);

        if (rewDistSolved <  rewardWeights[8])
		{
			solved = true;
			rewSolved = 6.f;
		}

		float rewDevInit = 0.0;
		for (int i = 0; i < 7; i++) {
		    rewDevInit += abs(state[i] - initJointAnglesAgent[a][i]);
		}

		dead = false;//solved || pegPos[1] < tableHeight;
		rew = rewardWeights[0] * rewTargetL1 +
		      rewardWeights[1] * rewTargetL2 +
		      rewardWeights[2] * rewVertical +
		      rewardWeights[3] * rewSolved +
		      rewardWeights[4] * rewPegVel +
		      rewardWeights[5] * rewPegRotChange +
		      rewardWeights[6] * rewAction +
		      rewardWeights[7] * rewDistSolved;
	}

//	void ClearContactInfo()
//	{
//		for (int a = 0; a < mNumAgents; a++)
//		{
//			pegContacts[a] = false;
//			pegForces[a] = 0.f;
//		}
//	}
//
//	void CheckPegContact(int body0, int body1, int& ai)
//	{
//		// check if body0 is a peg
//		ai = -1;
//		if (mapPegToAgent.find(body0) != mapPegToAgent.end())
//		{
//			ai = mapPegToAgent.at(body0);
//			// check if body1 is the peg holder corresponding to agent id.
//			if (body1 != pegHolderBodyIds[ai])
//			{
//				ai = -1;
//			}
//		}
//	}
//
//	void FinalizeContactInfo()
//	{
//		rigidContacts.map();
//		rigidContactCount.map();
//		int numContacts = rigidContactCount[0];
//
//		// check if we overflowed the contact buffers
//		if (numContacts > g_solverDesc.maxRigidBodyContacts)
//		{
//			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
//			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
//		}
//
//		float forceScale = 0.1f;
//
//		NvFlexRigidContact* ct = &(rigidContacts[0]);
//
//		int ai;
//		for (int i = 0; i < numContacts; ++i)
//		{
//			ai = -1;
//
//			CheckPegContact(ct[i].body0, ct[i].body1, ai);
//			if (ai == -1)
//			{
//				CheckPegContact(ct[i].body1, ct[i].body0, ai);
//			}
//
//			if (ai != -1)
//			{
//				pegContacts[ai] = true;
//				pegForces[ai] += forceScale * ct[i].lambda;
//			}
//		}
//		rigidContacts.unmap();
//		rigidContactCount.unmap();
//	}
};

class RLYumiBallCup : public RLYumiBase
{
public:
	vector<int> targetCups;
	vector<float> ballMasses;
	vector<int> ballBodyIds;
	vector<int> cupBodyIds;
	vector<int> cupShapeIds;
	vector<bool> ballContacts;
	vector<float> ballForces;

	vector<vector<int>> ropeBodyIds;
	const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
	const bool connectRopeToRobot = true;
	float tableLength, tableWidth;
    Vec3 tableOrigin;

    // Table
    vector<float> tableHeight;
    vector<float> tableFriction;

    // Rope
    vector<int> ropeNumSegments;
    vector<float> ropeSegmentRollingFriction;
    vector<float> ropeSegmentFriction;
    vector<float> ropeSegmentInertiaShift;
    vector<float> ropeSegmentLength;
    vector<float> ropeSegmentWidth;
    vector<float> ropeDensity;
    vector<float> ropeBendingCompliance;
    vector<float> ropeTorsionCompliance;
	vector<float> ropeTorsionDamping;
	vector<float> ropeBendingDamping;

    // ball
    vector<float> ballScale;
    vector<float> ballFriction;
    vector<float> ballThickness;
    vector<float> ballDensity;
    vector<float> ballMassMult;

    // ball holder
    vector<float> cupDensity;
    vector<float> cupScaleX;
    vector<float> cupScaleY;
    vector<float> cupScaleZ;
    vector<float> cupFriction;
    vector<float> cupThickness;
    vector<float> cupPosX;
    vector<float> cupPosY;
    vector<float> cupPosZ;
    vector<float> cupRoll;
    vector<float> cupPitch;
    vector<float> cupYaw;

	RLYumiBallCup()
	{
	    mNumObservations = numRobotStates + 9;
		tableLength = 0.55f;
		tableWidth = 0.4f;
		tableOrigin = Vec3(0.0f, 0.0f, 0.7f);

	//	g_params.solverType = eNvFlexSolverPBD;
	//	g_numSubsteps = 4;
	//	g_params.numIterations = 50;// 5; //4;
	//	g_params.numInnerIterations = 20;//20;
	}

	void LoadChildEnv()
	{
		initEffectorTranslation = Vec3(0.f, 0.8, 0.5f);
		targetCups.resize(mNumAgents);
		ballMasses.resize(mNumAgents);
		ballBodyIds.resize(mNumAgents);
		cupBodyIds.resize(mNumAgents);
		cupShapeIds.resize(mNumAgents);
		ballContacts.resize(mNumAgents);
		ballForces.resize(mNumAgents);
		targetPoses.resize(mNumAgents);
		targetSphere.resize(mNumAgents);
		ropeBodyIds.resize(mNumAgents);

        // Table
//		tableHeight.resize(mNumAgents,                  0.005f); // std 0.006
//        tableFriction.resize(mNumAgents,                0.7f); // std 0.14

        // Rope
        ropeNumSegments.resize(mNumAgents,              7); // std 2
        ropeSegmentRollingFriction.resize(mNumAgents,   0.001f); // std 0.0002
        ropeSegmentFriction.resize(mNumAgents,          0.25f); // std 0.05
        ropeSegmentInertiaShift.resize(mNumAgents,      0.0f); // std 0.0001
        ropeSegmentLength.resize(mNumAgents,            0.014f); // std 0.0028
        ropeSegmentWidth.resize(mNumAgents,             0.004f); // std 0.0008
        ropeDensity.resize(mNumAgents,                  2500.f); // std 200.0
        ropeBendingCompliance.resize(mNumAgents,        1e+1f); // std 2.0
        ropeTorsionCompliance.resize(mNumAgents,        2.f); // std 0.4
        ropeTorsionDamping.resize(mNumAgents,           0.1f); // std 0.05
        ropeBendingDamping.resize(mNumAgents,           0.01f); // std 0.005

        // ball
        ballScale.resize(mNumAgents,                    0.02f); // std 0.0056
        ballFriction.resize(mNumAgents,                  1.0f); // std 0.2
        ballThickness.resize(mNumAgents,                 0.005f); // std 0.0004
        ballDensity.resize(mNumAgents,                   400.0f); // std 80.0
        ballMassMult.resize(mNumAgents,                  1.0); // std 0.2

        // Peg holder
        cupDensity.resize(mNumAgents,             1000.0f); // std 200.0
        cupScaleX.resize(mNumAgents,              1.f); // std 0.006
        cupScaleY.resize(mNumAgents,              0.6f); // std 0.01
        cupScaleZ.resize(mNumAgents,              1.f); // std 0.006
        cupFriction.resize(mNumAgents,            1.0f); // std 0.2
        cupThickness.resize(mNumAgents,           0.002f); // std 0.0004

//        cupPosX.resize(mNumAgents,                -0.12f); // std 0.04
//        cupPosY.resize(mNumAgents,               0.13f + 0.005f); // std 0.005
//        cupPosZ.resize(mNumAgents,                0.8f); // std 0.1
//        cupRoll.resize(mNumAgents,                0.0f); // std 0.04
//        cupPitch.resize(mNumAgents,               0.0f); // std 0.005
//        cupYaw.resize(mNumAgents,                 0.0f); // std 0.1

//        cupPosX.resize(mNumAgents,                -0.150710f); // std 0.04
//        cupPosY.resize(mNumAgents,                0.03f + 0.02f); // std 0.005
//        cupPosZ.resize(mNumAgents,                0.557756f); // std 0.1
//        cupRoll.resize(mNumAgents,                0.0f); // std 0.04
//        cupPitch.resize(mNumAgents,               0.0f); // std 0.005
//        cupYaw.resize(mNumAgents,                 0.0f); // std 0.1
	}

    void SetSimParams(int ai)
	{
        if(num_sim_params == 0)
		{
            return;
        }
        SampleSimParams(ai);

        int ct = num_robot_sim_params;

        // Rope
        ropeNumSegments[ai] = (int)sim_params_samples[ai][ct++];
        ropeSegmentRollingFriction[ai] = sim_params_samples[ai][ct++];
        ropeSegmentFriction[ai] = sim_params_samples[ai][ct++];
        ropeSegmentInertiaShift[ai] = sim_params_samples[ai][ct++];
        ropeSegmentLength[ai] = sim_params_samples[ai][ct++];
        ropeSegmentWidth[ai] = sim_params_samples[ai][ct++];
        ropeDensity[ai] = sim_params_samples[ai][ct++];
        ropeBendingCompliance[ai] = sim_params_samples[ai][ct++];
        ropeTorsionCompliance[ai] = sim_params_samples[ai][ct++];
        ropeTorsionDamping[ai] = sim_params_samples[ai][ct++];
        ropeBendingDamping[ai] = sim_params_samples[ai][ct++];

        // ball
        ballScale[ai] = sim_params_samples[ai][ct++];
        ballFriction[ai] = sim_params_samples[ai][ct++];
        ballThickness[ai] = sim_params_samples[ai][ct++];
        ballDensity[ai] = sim_params_samples[ai][ct++];
        ballMassMult[ai] = sim_params_samples[ai][ct++];

        // Peg holder
        cupDensity[ai] = sim_params_samples[ai][ct++];
        cupScaleX[ai] = sim_params_samples[ai][ct++];
        cupScaleY[ai] = sim_params_samples[ai][ct++];
        cupScaleZ[ai] = sim_params_samples[ai][ct++];
        cupFriction[ai] = sim_params_samples[ai][ct++];
        cupThickness[ai] = sim_params_samples[ai][ct++];
//        cupPosX[ai] = sim_params_samples[ai][ct++];
//        cupPosY[ai] = tableHeight[ai] + sim_params_samples[ai][ct++];
//        cupPosZ[ai] = sim_params_samples[ai][ct++];
//        cupRoll[ai] = sim_params_samples[ai][ct++];
//        cupPitch[ai] = sim_params_samples[ai][ct++];
//        cupYaw[ai] = sim_params_samples[ai][ct++];
    }

	void AddChildEnvBodies(int ai, Transform gt)
	{
	    SetSimParams(ai);
        ropeBodyIds[ai].resize(ropeNumSegments[ai]);

	    // Create table
//	    NvFlexRigidShape table;
//		NvFlexMakeRigidBoxShape(&table, -1, tableLength, tableHeight[ai], tableWidth, NvFlexMakeRigidPose(tableOrigin + robotPoses[ai], Quat()));
////		table.filter = 0x0;
////		table.group = 0;
//		table.material.friction = tableFriction[ai];
//		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.17f, 0.24f, 0.4f)));
//		g_buffers->rigidShapes.push_back(table);


		// Create target cup and attach it to the hand
        NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["gripper_r_joint"]];
        Transform handLeftPose;
	    NvFlexGetRigidPose(&g_buffers->rigidBodies[handLeft.body0], (NvFlexRigidPose*)&handLeftPose);
        Vec3 cupPos = Vec3(0.01f, .1f, .1f) + handLeftPose.p;

		Mesh* cupMesh = ImportMesh("../../data/cups/cup_nohold.obj");//"../../data/peg_holder.obj");
		cupMesh->Transform(ScaleMatrix(Vec3(cupScaleX[ai], cupScaleY[ai], cupScaleZ[ai])));

		NvFlexTriangleMeshId cupMeshId = CreateTriangleMesh(cupMesh, 0.005f);

		NvFlexRigidShape cupShape;
		NvFlexMakeRigidTriangleMeshShape(&cupShape, g_buffers->rigidBodies.size(),
		    cupMeshId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);

		cupShape.filter = 0x4;
		cupShape.group = 0;
		cupShape.material.friction = cupFriction[ai];
		cupShape.thickness = cupThickness[ai];
		cupShapeIds[ai] = g_buffers->rigidShapes.size();
		g_buffers->rigidShapes.push_back(cupShape);

		NvFlexRigidBody cupBody;
		NvFlexMakeRigidBody(g_flexLib, &cupBody, cupPos, Quat(), &cupShape, &cupDensity[ai], 1);
		cupBodyIds[ai] = g_buffers->rigidBodies.size();
//		cupBody.invMass = 0.0f;
//		(Matrix33&)cupBody.invInertia = Matrix33();
		g_buffers->rigidBodies.push_back(cupBody);

		NvFlexRigidJoint cupJoint;
		NvFlexMakeFixedJoint(&cupJoint, handLeft.body0, cupBodyIds[ai],
//					NvFlexMakeRigidPose(Vec3(0.01f, .0f, .1f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5)),
            		NvFlexMakeRigidPose(Vec3(-0.07f, .0f, .1f), Quat()),//QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)),
					NvFlexMakeRigidPose(0, QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), -kPi * 0.5f)));

		g_buffers->rigidJoints.push_back(cupJoint);


		// Create rope
		NvFlexRigidJoint handRight = g_buffers->rigidJoints[urdf->jointNameMap["gripper_r_joint_m"]];
//		Vec3 startPos = Vec3(0.f, .75f, 1.f) + robotPoses[ai];
        Transform handRightPose;
	    NvFlexGetRigidPose(&g_buffers->rigidBodies[handRight.body0], (NvFlexRigidPose*)&handRightPose);
        Vec3 startPos = Vec3(0.01f, .0f, .1f) + handRightPose.p;

		NvFlexRigidPose prevJoint;
		int lastRopeBodyIndex = 0;

		for (int i = 0; i < ropeNumSegments[ai]; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, ropeSegmentWidth[ai], ropeSegmentLength[ai], NvFlexMakeRigidPose(0, Quat(.0f, .0f, .707f, .707f)));
			shape.filter = 0x1;
			shape.group = 0;

			shape.material.rollingFriction = ropeSegmentRollingFriction[ai];
			shape.material.friction = ropeSegmentFriction[ai];
			shape.user = UnionCast<void*>(linkMaterial);

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, startPos +
			    Vec3(0.0f, -(i*ropeSegmentLength[ai] * 2.f + ropeSegmentLength[ai]), 0.0f),
			    Quat(), &shape, &ropeDensity[ai], 1);
            (Matrix33&)body.inertia += Matrix33::Identity() * ropeSegmentInertiaShift[ai];
			body.maxAngularVelocity = 50.f;

            bool success;
            (Matrix33&)body.invInertia = Inverse((Matrix33&)body.inertia, success);

			ropeBodyIds[ai][i] = g_buffers->rigidBodies.size();
			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

			if (i == 0 && !connectRopeToRobot)
			{
				prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -ropeSegmentLength[ai], .0f), Quat());
				continue;
			}

			NvFlexRigidJoint joint;
			if (i == 0)
			{
				NvFlexMakeFixedJoint(&joint, handRight.body0, bodyIndex,
//					NvFlexMakeRigidPose(Vec3(0.01f, .0f, .1f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5)),
            		NvFlexMakeRigidPose(Vec3(0.04f, .0f, .1f), Quat()),//QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)),
					NvFlexMakeRigidPose(0, QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), -kPi * 0.5f)));
			}
			else
			{
				NvFlexMakeFixedJoint(&joint, bodyIndex - 1, bodyIndex, prevJoint,
									NvFlexMakeRigidPose(Vec3(0.f, ropeSegmentLength[ai], 0.f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)));
			}

			joint.compliance[eNvFlexRigidJointAxisTwist] = ropeTorsionCompliance[ai];
			joint.compliance[eNvFlexRigidJointAxisSwing1] = ropeBendingCompliance[ai];
			joint.compliance[eNvFlexRigidJointAxisSwing2] = ropeBendingCompliance[ai];

			joint.damping[eNvFlexRigidJointAxisTwist] = ropeTorsionDamping[ai];
			joint.damping[eNvFlexRigidJointAxisSwing1] = ropeBendingDamping[ai];
			joint.damping[eNvFlexRigidJointAxisSwing2] = ropeBendingDamping[ai];

			g_buffers->rigidJoints.push_back(joint);
			lastRopeBodyIndex = bodyIndex;
			prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -ropeSegmentLength[ai], .0f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f));
		}


		// ball
		Mesh* ballMesh = ImportMesh("../../data/sphere.ply");
		ballMesh->Transform(ScaleMatrix(Vec3(ballScale[ai], ballScale[ai], ballScale[ai])));

		NvFlexTriangleMeshId ballId = CreateTriangleMesh(ballMesh, 0.005f);
		NvFlexRigidShape ballShape;
		NvFlexMakeRigidTriangleMeshShape(&ballShape, g_buffers->rigidBodies.size(), ballId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
		ballShape.material.friction = ballFriction[ai];
		ballShape.thickness = ballThickness[ai];
		ballShape.filter = 0x0;
		ballShape.group = 0;
		ballShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(.9f, .9f, .3f)));
		g_buffers->rigidShapes.push_back(ballShape);

		NvFlexRigidBody ballBody;
		NvFlexMakeRigidBody(g_flexLib, &ballBody, startPos + Vec3(0.0f, -float(ropeNumSegments[ai] + 2) * ropeSegmentLength[ai] * 2.f, 0.f),
							Quat(), &ballShape, &ballDensity[ai], 1);
		ballMasses[ai] = ballBody.mass;
		ballBodyIds[ai] = g_buffers->rigidBodies.size();
		ballBody.maxAngularVelocity = 50.f;
		g_buffers->rigidBodies.push_back(ballBody);

		// Connecting ball to rope
		NvFlexRigidJoint joint;
		int bodyIndex = g_buffers->rigidBodies.size();
		NvFlexMakeFixedJoint(&joint, lastRopeBodyIndex, bodyIndex - 1,
						NvFlexMakeRigidPose(Vec3(0.f, -3.75f * ropeSegmentLength[ai], 0.f), QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)), NvFlexMakeRigidPose(0, QuatFromAxisAngle(Vec3(0.f, 0.f, 1.f), kPi * 0.5f)));

		joint.compliance[eNvFlexRigidJointAxisTwist] = 0.1f * ropeTorsionCompliance[ai];
		joint.compliance[eNvFlexRigidJointAxisSwing1] = 0.2f * ropeBendingCompliance[ai];
		joint.compliance[eNvFlexRigidJointAxisSwing2] = 0.2f * ropeBendingCompliance[ai];

		joint.damping[eNvFlexRigidJointAxisTwist] = ropeTorsionDamping[ai];
		joint.damping[eNvFlexRigidJointAxisSwing1] = ropeBendingDamping[ai];
		joint.damping[eNvFlexRigidJointAxisSwing2] = ropeBendingDamping[ai];

		g_buffers->rigidJoints.push_back(joint);

		SampleTarget(ai);

//		NvFlexRigidShape targetShape;
//		NvFlexMakeRigidSphereShape(&targetShape, -1, 0.04f, NvFlexMakeRigidPose(0,0));
//
//		int renderMaterial = AddRenderMaterial(Vec3(1.0f, 0.04f, 0.07f));
//		targetShape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.0f,0.0f,0.0f)));
//		targetShape.group = 1;
//		targetSphere[ai] = g_buffers->rigidShapes.size();
//		g_buffers->rigidShapes.push_back(targetShape);
//
//        Transform cupPoseWorld;
//		NvFlexGetRigidPose(&g_buffers->rigidBodies[cupBodyIds[ai]], (NvFlexRigidPose*)&cupPoseWorld);
//		NvFlexRigidPose pose = NvFlexMakeRigidPose(cupPoseWorld.p, Quat());
//		g_buffers->rigidShapes[targetSphere[ai]].pose = pose;
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
			targetPoses[ai] = Vec3(Randf(-0.4f, 0.1f), tableHeight[ai] + 0.02f, Randf(0.4f, 0.8f));
		}
		else
		{
//			targetPoses[ai] = Vec3(cupPosX[ai], cupPosY[ai], cupPosZ[ai]);
		}

//		Quat rot = rpy2quat(cupRoll[ai], cupPitch[ai], cupYaw[ai]);

//
//		NvFlexRigidPose newPose = NvFlexMakeRigidPose(targetPoses[ai] + robotPoses[ai], rot);
//		NvFlexSetRigidPose(&g_buffers->rigidBodies[cupBodyIds[ai]], &newPose);
//
//	    float ballMass;
//		if (sampleInitStates)
//		{
//			ballMass = ballMasses[ai] * Randf(0.8f, 1.1f);
//		}
//		else
//		{
//			ballMass = ballMasses[ai];
//		}
//
//		g_buffers->rigidBodies[ballBodyIds[ai]].mass = ballMass;
	}

	void ExtractChildState(int a, float* state, int ct)
	{
		// 0-2 target xyz
		Transform cupPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cupBodyIds[a]], (NvFlexRigidPose*)&cupPoseWorld);
		Vec3 cupTraLocal = cupPoseWorld.p - robotPoses[a];
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = cupTraLocal[i];
		}

		// 3-5 ball xyz
		Transform ballPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[ballBodyIds[a]], (NvFlexRigidPose*)&ballPoseWorld);
		Vec3 ballTraLocal = ballPoseWorld.p - robotPoses[a];
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = ballTraLocal[i];
		}

        // 6-8 previous ball xyz
		for (int i = 0; i < 3; ++i)
		{
			state[ct] = prevState[a][ct-3];
			ct++;
		}

//		// 6-8 ball rpy
//		float r, p, y;
//		quat2rpy(ballPoseWorld.q, r, p, y);
//		state[ct++] = r;
//		state[ct++] = p;
//		state[ct++] = y;

//		// 23-25 ball velocities
//		NvFlexRigidBody ballBody = g_buffers->rigidBodies[ballBodyIds[a]];
//		for (int i = 0; i < 3; ++i)
//		{
//			state[ct++] = ballBody.linearVel[i];
//		}
//		for (int i = 0; i < 3; ++i)
//		{
//			state[ct++] = ballBody.angularVel[i];
//		}
//
//		// 25-27 contact flag and force
//		state[ct++] = ballContacts[a] ? 1.f : 0.f;
//		state[ct++] = ballForces[a];
//
//		// +9 * ropeSegments rope segment translation and rotations
//		Transform segPoseWorld;
//		for (int i = 0; i < ropeSegments; i++)
//		{
//			int segBodyId = ropeBodyIds[a][i];
//			NvFlexGetRigidPose(&g_buffers->rigidBodies[segBodyId], (NvFlexRigidPose*)&segPoseWorld);
//			Vec3 segTraLocal = segPoseWorld.p - robotPoses[a];
//			NvFlexRigidBody segBody = g_buffers->rigidBodies[segBodyId];
//
//			for (int j = 0; j < 3; j++)
//			{
//				state[ct++] = segTraLocal[j];
//				state[ct++] = segBody.linearVel[j];
//				state[ct++] = segBody.angularVel[j];
//			}
//		}
	}

    virtual void ExtractChildExtra(int a, float* extras, int ct) {
        Transform ballPoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[ballBodyIds[a]], (NvFlexRigidPose*)&ballPoseWorld);

        // 0-3 ball rpy
		float r, p, y;
		quat2rpy(ballPoseWorld.q, r, p, y);
		extras[ct++] = r;
		extras[ct++] = p;
		extras[ct++] = y;
    }

	void ComputeRewardAndDead(int a, float* action, float* state, float* extras, float& rew, bool& dead)
	{
		float rewTarget = 0.f, rewVertical = 0.f, rewAction = 0.f, rewSolved = 0.f;
		bool solved = false;
		float targetReachWeight = 20.f;
		float verticalAngleWeight = 0.5f;

		// reward for getting close to peg holder
		Vec3 targetPos = Vec3(state[numRobotStates], state[numRobotStates+1], state[numRobotStates+2]);
		Vec3 ballPos = Vec3(state[numRobotStates+3], state[numRobotStates+4], state[numRobotStates+5]);
		Vec3 ballPrevPos = Vec3(prevState[a][numRobotStates+3], prevState[a][numRobotStates+4], prevState[a][numRobotStates+5]);

		Vec3 ballToTarget = ballPos - targetPos;
		float distTarget = Length(ballToTarget);
		float distTargetL1 = abs(ballToTarget[0]) + abs(ballToTarget[1]) + abs(ballToTarget[2]);
		float rewBallY = ballPos[1];
		rewTarget = (distTarget + distTargetL1); //exp(-4.f * distTarget);

		// reward for maintaining verticality of ball
		Quat ballRot = rpy2quat(extras[0], extras[1], extras[2]);
		float ballAngle0 = RadToDeg(Dot(GetBasisVector1(ballRot), Vec3(1.f, 0.f, 0.f)));
		float ballAngle1 = RadToDeg(Dot(GetBasisVector1(ballRot), Vec3(0.f, 1.f, 0.f)));
		float ballAngle2 = RadToDeg(Dot(GetBasisVector1(ballRot), Vec3(0.f, 0.f, 1.f)));

		float ballAngleToVertical = ballAngle1 - 44.888748f + ballAngle2  - (-34.971664f); //abs(57.2957f - ballAngle1);
//		rewVertical = verticalAngleWeight * exp(-0.1*ballAngleToVertical);
		rewVertical = ballAngleToVertical;

//		Quat ballPrevRot = rpy2quat(prevExtra[a][0], prevExtra[a][1], prevExtra[a][2]);
//		float ballPrevAngle = RadToDeg(Dot(GetBasisVector1(ballPrevRot), Vec3(0.f, 1.f, 0.f)));

        float ballPosChange = Length(ballPos - ballPrevPos);
        float ballRotChange = 0.0f;//abs(ballAngle - ballPrevAngle);
        float rewBallVel = ballPosChange;
        float rewBallRotChange = ballRotChange;

		float* prevAction = GetPrevAction(a);
		for (int i = 0; i < mNumActions; i++)
		{
//			rewAction += -0.8f * Pow(action[i] - prevAction[i], 2) / (float)mNumActions;
    		rewAction += Pow(action[i], 2);
		}

        if (distTarget < 0.01f)
		{
			solved = true;
			rewSolved = 1.f;
		}

		float rewEndeffY = extras[1];
		float rewDevInit = 0.0;
		for (int i = 0; i < 7; i++) {
		    rewDevInit += abs(state[i] - initJointAnglesAgent[a][i]);
		}

		dead = false;//solved || ballPos[1] < tableHeight;
		rew = rewardWeights[0] * rewTarget +
		      rewardWeights[1] * rewEndeffY +
		      rewardWeights[2] * rewSolved +
		      rewardWeights[3] * rewDevInit +
		      rewardWeights[4] * rewBallY +
		      rewardWeights[5] * rewAction;

//		printf ("REW TARGET %f VERT %f ACT %f PEGVEL %f PEGROTCHANGE %f SOLV %f\n", rewTarget, rewVertical, rewAction, rewPegVel, rewPegRotChange, rewSolved);
	}

//	void ClearContactInfo()
//	{
//		for (int a = 0; a < mNumAgents; a++)
//		{
//			pegContacts[a] = false;
//			pegForces[a] = 0.f;
//		}
//	}
//
//	void CheckPegContact(int body0, int body1, int& ai)
//	{
//		// check if body0 is a peg
//		ai = -1;
//		if (mapPegToAgent.find(body0) != mapPegToAgent.end())
//		{
//			ai = mapPegToAgent.at(body0);
//			// check if body1 is the peg holder corresponding to agent id.
//			if (body1 != cupBodyIds[ai])
//			{
//				ai = -1;
//			}
//		}
//	}
//
//	void FinalizeContactInfo()
//	{
//		rigidContacts.map();
//		rigidContactCount.map();
//		int numContacts = rigidContactCount[0];
//
//		// check if we overflowed the contact buffers
//		if (numContacts > g_solverDesc.maxRigidBodyContacts)
//		{
//			printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
//			numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
//		}
//
//		float forceScale = 0.1f;
//
//		NvFlexRigidContact* ct = &(rigidContacts[0]);
//
//		int ai;
//		for (int i = 0; i < numContacts; ++i)
//		{
//			ai = -1;
//
//			CheckPegContact(ct[i].body0, ct[i].body1, ai);
//			if (ai == -1)
//			{
//				CheckPegContact(ct[i].body1, ct[i].body0, ai);
//			}
//
//			if (ai != -1)
//			{
//				pegContacts[ai] = true;
//				pegForces[ai] += forceScale * ct[i].lambda;
//			}
//		}
//		rigidContacts.unmap();
//		rigidContactCount.unmap();
//	}
};

class RLYumiCloth : public RLYumiBase
{
public:
	float tableLength, tableWidth;
	Vec3 tableOrigin;

	// Table
	float tableHeight = 0.27f;
	float tableFriction = 0.7f;

	// Cloth

	RLYumiCloth()
	{
		mNumAgents = 1;
		numPerRow = 2;
		mNumObservations = numRobotStates + 9;
		tableLength = 0.55f;
		tableWidth = 0.27f;
		tableOrigin = Vec3(0.0f, 0.0f, 0.7f);
	}

	void LoadChildEnv()
	{
		initEffectorTranslation = Vec3(0.f, 0.8, 0.5f);
	}

	void SetSimParams(int ai)
	{
		if (num_sim_params == 0)
		{
			return;
		}
		SampleSimParams(ai);

		int ct = 0;
		// Table
		tableHeight = sim_params_samples[ai][ct++];
		tableFriction = sim_params_samples[ai][ct++];

		// Cloth
	}

	void AddChildEnvBodies(int ai, Transform gt)
	{
		SetSimParams(ai);

		// Create table
		NvFlexRigidShape table;
		NvFlexMakeRigidBoxShape(&table, -1, tableLength, tableHeight, tableWidth, NvFlexMakeRigidPose(tableOrigin + robotPoses[ai], Quat()));
		table.filter = 0x0;
		table.group = 0;
		table.material.friction = tableFriction;
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.17f, 0.24f, 0.4f)));

		g_buffers->rigidShapes.push_back(table);

		// Cloth
		const float radius = 0.0075f;

		// PCR
		//   float stretchStiffness = 0.6f;
		//   float bendStiffness = 0.01f;
		//   float shearStiffness = 0.01f;

		// PBD
		g_params.solverType = eNvFlexSolverPBD;
		g_numSubsteps = 4;
		g_params.numIterations = 40;

		g_params.dynamicFriction = 1.25f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;

		g_params.radius = radius * 1.8f;
		g_params.collisionDistance = 0.006f;

		g_drawCloth = true;

		float stretchStiffness = 1.0f;
		float bendStiffness = 0.9f;
		float shearStiffness = 0.8f;

		int dimx = 60;
		int dimy = 40;

		float mass = 0.5f / (dimx * dimy);	// avg bath towel is 500-700g

		CreateSpringGrid(Vec3(-0.3f, 0.37f, 0.5f), dimx, dimy, 1, radius, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f / mass);

		g_params.radius = radius * 1.8f;
		g_params.collisionDistance = 0.005f;

		g_drawCloth = true;

		SampleTarget(ai);
	}

	void SampleTarget(int ai)
	{

	}

	void ExtractChildState(int a, float* state, int ct)
	{
		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = 0.f;
		}

		for (int i = 0; i < 3; ++i)
		{
			state[ct++] = 0.f;
		}

		state[ct++] = 0.f;
		state[ct++] = 0.f;
		state[ct++] = 0.f;
	}

	void ComputeRewardAndDead(int a, float* action, float* state, float* extras, float& rew, bool& dead)
	{
		float rewTarget = 0.f, rewVertical = 0.f, rewAction = 0.f, rewSolved = 0.f;
		bool solved = false;
		float targetReachWeight = 4.f;
		float verticalAngleWeight = 0.3f;

		float* prevAction = GetPrevAction(a);
		for (int i = 0; i < mNumActions; i++)
		{
			rewAction += -0.2f * Pow(action[i] - prevAction[i], 2) / (float)mNumActions;
			//    		rewAction += -0.5 * Pow(action[i], 2);
		}

		dead = false;
		rew = rewAction + rewTarget + rewVertical + rewSolved; // rewTarget + rewVertical + rewAction + rewSolved;
	}
};