#pragma once
#include <iostream>
#include <vector>
#include "rlbase.h"
#include "../urdf.h"


// Makes env with hand, table w/ no targets, no rewards. Provides interface for control and reading hand states.
class RLAllegroBase : public FlexGymBase2
{
public:
	URDFImporter* urdf;
	bool sampleInitStates;
	
	// general variables
	int numFingers, numJointsPerFinger;
	vector<vector<float>> fingerJointHomeAngles; // in degrees
	// limits for finger joints in rads. 
	vector<pair<float, float>> fingerJointAngleLimits, thumbJointAngleLimits;
	// limits for hand. rpy in degs.
	vector<pair<float, float>> handRpyLimits, handTranslationLimits;

	// per agent variables
	// num agents x num fingers x num joints per finger
	vector<vector<vector<float>>> fingerJointTargetAngles, fingerJointCurrentAngles; // in radians
	vector<vector<vector<int>>> allFingerJoints; // map to joint ids
	vector<Vec3> handTargetTranslations, handCurrentTranslations, handTargetRpys, handCurrentRpys; // in degrees
	vector<int> handBodyIds, handJointIds;
	vector<Vec3> robotPoses;

	// physics variables
	const float joinCompliance = 1.e-6f;
	const float jointDamping = 1.e+1f;
	const float bodyArmature = 1.e-5f;
	const float bodyDamping = 5.0f;

	int hiddenMaterial;

	RLAllegroBase()
	{
		mNumAgents = 10;
		mMaxEpisodeLength = 300;

		g_numSubsteps = 4;
		g_params.numIterations = 40;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(8.6f, 0.9f, 3.5f);

		numPerRow = 20;
		spacing = 3.5f;

		g_pause = true;
		mDoLearning = g_doLearning;

		g_params.dynamicFriction = 0.8f;
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.005f;

		g_pause = true;
		g_drawPoints = false;
		g_drawCloth = true;
		sampleInitStates = true;

		mNumActions = 22; // 4 fingers x 4 joints + 6DOF hand base
		mNumObservations = 22; // same as actions 

		controlType = ePosition;

		// allegro hand default values
		numFingers = 4;
		numJointsPerFinger = 4;

		// Set joint home angles (see http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Home_Position_Joint_Angles)
		fingerJointHomeAngles = {
			{ 0.f, -10.f, 45.f, 45.f, },
			{ 0.f, -10.f, 45.f, 45.f, },
			{ 5.f, -5.f, 50.f, 45.f, },
			{ 50.f, 25.f, 15.f, 45.f, },
		};
		
		// Joint limits in radians (see http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Limits)
		fingerJointAngleLimits = {
			{ -0.57f, 0.57f },
			{ -0.296f, 1.71f },
			{ -0.274f, 1.809f },
			{ -0.327f, 1.718f }
		};
		thumbJointAngleLimits = {
			{ 0.3636f, 1.4968f },
			{ -0.2050f, 1.2631f },
			{ -0.2897f, 1.744f },
			{ -0.2622f, 1.8199f }
		};
		// hand limits
		handRpyLimits = {
			{-180.f, 180.f},
			{-180.f, 180.f},
			{-180.f, 180.f}
		};
		handTranslationLimits = {
			{-.5f, .5f},
			{.4f, 1.f},
			{-.5f, .5f}
		};
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);
		if (g_sceneJson.find("SampleInitStates") != g_sceneJson.end())
		{
			sampleInitStates = g_sceneJson.value("SampleInitStates", sampleInitStates);
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
	virtual void AddEnvBodiesJoints(int ai, Transform gt) {};

	virtual void SampleInitHandFingers(int ai)
	{
		// TODO(jaliang): perform actual sampling?

		// hand
		handTargetRpys[ai] = Vec3(0.f, 0.f, -90.f);
		handCurrentRpys[ai] = Vec3(handTargetRpys[ai]);
		handTargetTranslations[ai] = Vec3(0.0f, 0.6f, 0.0f);
		handCurrentTranslations[ai] = Vec3(handTargetTranslations[ai]);

		Quat handQuat = rpy2quat(DegToRad(handTargetRpys[ai][0]), DegToRad(handTargetRpys[ai][1]), DegToRad(handTargetRpys[ai][2]));
		g_buffers->rigidJoints[handJointIds[ai]].pose0 = NvFlexMakeRigidPose(handTargetTranslations[ai] + robotPoses[ai], handQuat);

		// fingers
		// Set initial joint targets to home angles
		for (int i = 0; i < numFingers; i++)
		{
			for (int j = 0; j < numJointsPerFinger; j++)
			{
				fingerJointTargetAngles[ai][i][j] = DegToRad(fingerJointHomeAngles[i][j]);
				fingerJointCurrentAngles[ai][i][j] = fingerJointTargetAngles[ai][i][j];
				g_buffers->rigidJoints[allFingerJoints[ai][i][j]].targets[eNvFlexRigidJointAxisTwist] = fingerJointCurrentAngles[ai][i][j];
			}
		}
	}

	void AddAgentBodiesJointsCtlsPowers(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		handBodyIds[ai] = g_buffers->rigidBodies.size();

		int startShape = g_buffers->rigidShapes.size();
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, false, 1000.0f, joinCompliance, jointDamping, bodyArmature, bodyDamping,
			kPi * 8.0f, false);
		int endShape = g_buffers->rigidShapes.size();
		// set finger shape collision filters
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
				allFingerJoints[ai][i][j] = urdf->jointNameMap[jointName.str()];

				g_buffers->rigidJoints[allFingerJoints[ai][i][j]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
				ctrl.push_back(make_pair(allFingerJoints[ai][i][j], eNvFlexRigidJointAxisTwist));

				mpower.push_back(urdf->joints[urdf->urdfJointNameMap[jointName.str()]]->effort);
			}
		}
				
		// Configure hand joint
		NvFlexRigidJoint handJoint;
		NvFlexMakeFixedJoint(&handJoint, -1, handBodyIds[ai], NvFlexMakeRigidPose(0, 0), NvFlexMakeRigidPose(0, 0));
		handJointIds[ai] = g_buffers->rigidJoints.size();
		g_buffers->rigidJoints.push_back(handJoint);

		// Set initial hand location and finger joints
		SampleInitHandFingers(ai);

		// set up table
		NvFlexRigidShape table;
		NvFlexMakeRigidBoxShape(&table, -1, 0.6f, 0.4f, 0.3f, NvFlexMakeRigidPose(robotPoses[ai], Quat()));
		table.filter = 0x0;
		table.group = 0;
		table.material.friction = 0.7f;
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.17f, 0.24f, 0.4f)));

		g_buffers->rigidShapes.push_back(table);
	}

	// To be overwritten by child envs
	virtual void LoadChildEnv() {}

	void LoadEnv()
	{
		LoadChildEnv();

		// initialize data structures
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
		robotPoses.resize(mNumAgents);

		handTargetTranslations.resize(mNumAgents);
		handCurrentTranslations.resize(mNumAgents);
		handTargetRpys.resize(mNumAgents);
		handCurrentRpys.resize(mNumAgents);
		handBodyIds.resize(mNumAgents);
		handJointIds.resize(mNumAgents);

		allFingerJoints.resize(mNumAgents);
		fingerJointTargetAngles.resize(mNumAgents);
		fingerJointCurrentAngles.resize(mNumAgents);
		for (int i = 0; i < mNumAgents; i++)
		{
			allFingerJoints[i].resize(numFingers);
			fingerJointTargetAngles[i].resize(numFingers);
			fingerJointCurrentAngles[i].resize(numFingers);
			for (int j = 0; j < numFingers; j++)
			{
				allFingerJoints[i][j].resize(numJointsPerFinger);
				fingerJointTargetAngles[i][j].resize(numJointsPerFinger);
				fingerJointCurrentAngles[i][j].resize(numJointsPerFinger);
			}
		}

		// load urdf
		// hide collision shapes
		hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
		urdf = new URDFImporter("../../data", "allegro_hand_description/allegro_hand_description_right.urdf");
		
		// set up each env
		for (int ai = 0; ai < mNumAgents; ++ai)
		{
			Vec3 robotPos = Vec3((ai % numPerRow) * spacing, 0.0f, (ai / numPerRow) * spacing - 0.25f);
			Transform gt(robotPos + Vec3(0.f, 0.6f, 0.f), Quat());
			robotPoses[ai] = robotPos;

			int begin = g_buffers->rigidBodies.size();
			AddAgentBodiesJointsCtlsPowers(ai, gt, ctrls[ai], motorPower[ai]);
			AddEnvBodiesJoints(ai, gt);
			int end = g_buffers->rigidBodies.size();
			agentBodies.push_back(make_pair(begin, end));

			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
		}
	}

	// To be overwritten by child envs
	virtual void ExtractChildState(int ai, float* state, int ct) {}

	int ExtractState(int ai, float* state)
	{
		int ct = 0;
		// 0-15 finger joints
		for (int i = 0; i < numFingers; i++)
		{
			for (int j = 0; j < numJointsPerFinger; j++)
			{
				state[ct++] = fingerJointCurrentAngles[ai][i][j];
			}
		}

		// 16-18 hand translation
		for (int i = 0; i < 3; i++)
		{
			state[ct++] = handCurrentTranslations[ai][i];
		}
		
		// 19-21 hand rpy
		for (int i = 0; i < 3; i++)
		{
			state[ct++] = handCurrentRpys[ai][i];
		}

		return ct;
	}

	void PopulateState(int ai, float* state)
	{
		int ct = ExtractState(ai, state);
		ExtractChildState(ai, state, ct);
	}
	
	// To be overwritten by child envs
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) {}

	float scaleActions(float minx, float maxx, float x)
	{
		x = 0.5f * (x + 1.f) * (maxx - minx) + minx;
		return x;
	}

	void ApplyTargetControl(int ai)
	{
		float* actions = GetAction(ai);

		// updating target finger joint angles
		for (int i = 0; i < numFingers; i++)
		{
			for (int j = 0; j < numJointsPerFinger; j++)
			{
				float cc = Clamp(actions[i * numFingers + j], -1.f, 1.f);
				if (i == 3)
				{
					fingerJointTargetAngles[ai][i][j] = scaleActions(thumbJointAngleLimits[j].first, thumbJointAngleLimits[j].second, cc);
				}
				else
				{
					fingerJointTargetAngles[ai][i][j] = scaleActions(fingerJointAngleLimits[j].first, fingerJointAngleLimits[j].second, cc);
				}
			}
		}

		// updating target hand 
		for (int i = 0; i < 3; i++)
		{
			float cc = Clamp(actions[i + 16], -1.f, 1.f);
			handTargetTranslations[ai][i] = scaleActions(handTranslationLimits[i].first, handTranslationLimits[i].second, cc);
		}
		for (int i = 0; i < 3; i++)
		{
			float cc = Clamp(actions[i + 19], -1.f, 1.f);
			handTargetRpys[ai][i] = scaleActions(handRpyLimits[i].first, handRpyLimits[i].second, cc);
		}

		UpdateHandFingers(ai);
	}

	void UpdateHandFingers(int ai)
	{
		float smoothing = 0.05f;
		// move fingers
		for (int i = 0; i < numFingers; i++)
		{
			for (int j = 0; j < numJointsPerFinger; j++)
			{
				fingerJointCurrentAngles[ai][i][j] = Lerp(fingerJointCurrentAngles[ai][i][j], fingerJointTargetAngles[ai][i][j], smoothing);
				g_buffers->rigidJoints[allFingerJoints[ai][i][j]].targets[eNvFlexRigidJointAxisTwist] = fingerJointCurrentAngles[ai][i][j];
			}
		}

		// move hand
		NvFlexRigidJoint handJoint = g_buffers->rigidJoints[handJointIds[ai]];
		for (int i = 0; i < 3; i++)
		{
			handCurrentTranslations[ai][i] = Lerp(handCurrentTranslations[ai][i], handTargetTranslations[ai][i], smoothing);
			handJoint.pose0.p[i] = handCurrentTranslations[ai][i] + robotPoses[ai][i];
			handCurrentRpys[ai][i] = Lerp(handCurrentRpys[ai][i], handTargetRpys[ai][i], smoothing);
		}

		Quat currentQuat = rpy2quat(DegToRad(handCurrentRpys[ai][0]), DegToRad(handCurrentRpys[ai][1]), DegToRad(handCurrentRpys[ai][2]));
		handJoint.pose0.q[0] = currentQuat.x;
		handJoint.pose0.q[1] = currentQuat.y;
		handJoint.pose0.q[2] = currentQuat.z;
		handJoint.pose0.q[3] = currentQuat.w;
		g_buffers->rigidJoints[handJointIds[ai]] = handJoint;
	}

	virtual void ResetAgent(int a) {}

	~RLAllegroBase()
	{
		if (urdf)
		{
			delete urdf;
		}
	}

	virtual void ClearContactInfo() {}
	virtual void FinalizeContactInfo() {}
	virtual void LockWrite() {} // Do whatever needed to lock write to simulation
	virtual void UnlockWrite() {} // Do whatever needed to unlock write to simulation

	void DoGui()
	{
		if (!mDoLearning)
		{
			// TODO(jaliang): add slider controls
		}
	}

	void Update()
	{
		if (!mDoLearning)
		{
			// TODO(jaliang): add update if not learning
		}
	}

	void PostUpdate()
	{
		// joints are not read back by default
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer); // Do we need it?
	}

};

// Env for relocating an object to a target location
class RLAllegroObjectRelocation : public RLAllegroBase
{
public:

	// object variables (cubes)
	vector<pair<int, int>> cubeShapeBodyIds;
	vector<float> cubeMasses, cubeScales;

	// target sphere variables
	float targetSphereRadius;
	vector<int> targetRenderMaterials, targetSphereShapes;
	vector<Vec3> targetLocations;
	Vec3 redColor, greenColor;

	RLAllegroObjectRelocation()
	{
		mNumActions = 22; // 4 fingers x 4 joints + 6DOF hand base
		mNumObservations = 28; // same as actions + xyz of obj + xyz of target

		targetSphereRadius = 0.022f;
		redColor = Vec3(1.0f, 0.04f, 0.07f);
		greenColor = Vec3(0.06f, 0.92f, 0.13f);
	}

	void LoadChildEnv()
	{
		// init data structures
		targetLocations.resize(mNumAgents);
		cubeShapeBodyIds.resize(mNumAgents);
		targetRenderMaterials.resize(mNumAgents);
		targetSphereShapes.resize(mNumAgents);
		cubeMasses.resize(mNumAgents);
		cubeScales.resize(mNumAgents);
	}

	void SampleTarget(int ai)
	{
		if (sampleInitStates)
		{
			targetLocations[ai] = Vec3(Randf(-0.3f, 0.3f), Randf(0.5f, .8f), Randf(-.15f, .15f));
		}
		else
		{
			targetLocations[ai] = Vec3(-0.2f, 0.6f, 0.f);
		}

		NvFlexRigidPose newPose = NvFlexMakeRigidPose(targetLocations[ai] + robotPoses[ai], Quat());
		g_buffers->rigidShapes[targetSphereShapes[ai]].pose = newPose;
	}

	void SampleObject(int ai)
	{
		Vec3 pos; Quat quat; float cubeMass;
		if (sampleInitStates)
		{
			pos = Vec3(Randf(-0.2f, 0.2f), 0.4f + cubeScales[ai] * 0.5f + .02f, Randf(-.15f, .15f)) + robotPoses[ai];
			quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(-0.2f, 0.2f));
			cubeMass = cubeMasses[ai] * Randf(0.95f, 1.05f);
		}
		else
		{
			pos = Vec3(0.f, 0.4f + cubeScales[ai] * 0.5f + 0.02f, 0.f) + robotPoses[ai];
			quat = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), 0.f);
			cubeMass = cubeMasses[ai];
		}

		int	bodyId = cubeShapeBodyIds[ai].second;
		g_buffers->rigidBodies[bodyId].mass = cubeMass;
		NvFlexRigidPose pose = NvFlexMakeRigidPose(pos, quat);
		NvFlexSetRigidPose(&g_buffers->rigidBodies[bodyId], &pose);
	}

	// Add cubes and target spheres
	void AddEnvBodiesJoints(int ai, Transform gt) 
	{
		// Create target sphere
		NvFlexRigidShape targetShape;
		NvFlexMakeRigidSphereShape(&targetShape, -1, targetSphereRadius, NvFlexMakeRigidPose(0, 0));
		int renderMaterial = AddRenderMaterial(redColor);
		targetRenderMaterials[ai] = renderMaterial;
		targetShape.user = UnionCast<void*>(renderMaterial);
		targetShape.group = 1;
		targetSphereShapes[ai] = g_buffers->rigidShapes.size();
		g_buffers->rigidShapes.push_back(targetShape);

		SampleTarget(ai);

		// Create cube
		Mesh* cubeMesh = ImportMesh("../../data/box.ply");
		NvFlexTriangleMeshId cubeMeshId = CreateTriangleMesh(cubeMesh, 0.00125f);
		cubeShapeBodyIds[ai] = pair<int, int>(g_buffers->rigidShapes.size(), g_buffers->rigidBodies.size());

		float friction, density;
		if (sampleInitStates)
		{
			cubeScales[ai] = Randf(0.048f, 0.052f);
			density = Randf(480.f, 520.f);
			friction = Randf(0.95f, 1.15f);
		}
		else
		{
			cubeScales[ai] = 0.05f;
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

		SampleObject(ai);
	}

	void ExtractChildState(int ai, float* state, int ct)
	{
		// 22-24 xyz of cube
		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];
		for (int i = 0; i < 3; i++)
		{
			state[ct++] = cubeTraLocal[i];
		}
		
		// 25-27 xyz of target
		for (int i = 0; i < 3; i++)
		{
			state[ct++] = targetLocations[ai][i];
		}
	}

	void ComputeRewardAndDead(int ai, float* action, float* state, float& rew, bool& dead) 
	{
		Transform cubePoseWorld;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[cubeShapeBodyIds[ai].second], (NvFlexRigidPose*)&cubePoseWorld);
		Vec3 cubeTraLocal = cubePoseWorld.p - robotPoses[ai];
		float distObjToTarget = Length(cubeTraLocal - targetLocations[ai]);

		// sparse reward
		rew = 0.f;
		if (distObjToTarget < 0.1f)
		{
			rew += 10.f;
			if (distObjToTarget < 0.05f)
			{
				rew += 20.f;
			}
		}

		// color the target sphere
		float alpha = exp(-pow(distObjToTarget, 2)*50);
		g_renderMaterials[targetRenderMaterials[ai]].frontColor = alpha * greenColor + (1.f - alpha) * redColor;

		// ep is done if goal achieved or cube falls off table
		dead = false;
		if (distObjToTarget < 0.05f || cubeTraLocal[1] < 0.35f)
		{
			dead = true;
		}
	}

	void ResetAgent(int a) 
	{
		g_buffers->rigidShapes.map();
		g_buffers->rigidJoints.map();

		SampleTarget(a);
		SampleObject(a);
		SampleInitHandFingers(a);

		g_buffers->rigidShapes.unmap();
		g_buffers->rigidJoints.unmap();		
		
		RLFlexEnv::ResetAgent(a);
	}
};