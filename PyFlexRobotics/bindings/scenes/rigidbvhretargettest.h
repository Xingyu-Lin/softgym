#pragma once

#include "../mjcf.h"
#include "../Skeleton.h"
#include "rlbase.h"

class RigidBVHRetargetTest : public Scene
{
public:
	enum WORKMODE { DUMP_POSE, DUMP_FULL_STATE, DUMP_FULL_STATE_FORWARD_FACING, PREPARE_DATA, PREPARE_DATA_LOAD_STAT, DUMP_DATA_FRAME, DUMP_DATA_FRAME_WITH_VEL, NOTHING, DUMP_FULL_STATE_AND_JOINT_ANGLES };
	WORKMODE workmode;

	Skeleton* skeleton = NULL;
	float time;
	float to_meters;
	vector<pair<int, int> > jointToBVH;

	MJCFImporter* mjcf;

	vector<int> effectors;
	vector<int> robotDOFJoints;
	vector<float> robotDOFComplianceJoints;
	vector<int> robotTargetJoints;
	vector<int> robotBodies;
	vector<int> robotTinyBodies;
	vector<Vec4> bkPos;
	vector<Vec3> bkVel;

	vector<bool> robotJointParticipateInInverseDynamic;
	vector<NvFlexRigidBody> backupBodies;
	vector<NvFlexRigidBody> prevBodies;
	bool doInverseDynamic;
	vector<string> geo_joint;
	vector<int> bvh;
	vector<vector<Transform>> allTrans;
	string fileName;
	Transform worldTrans;
	Transform worldTransInv;
	vector<vector<Transform>> fullTrans;
	vector<vector<Vec3>> fullVels;
	vector<vector<Vec3>> fullAVels;

	// For dumping out target angles
	vector<vector<float>> jointAngles;
	vector<pair<int, NvFlexRigidJointAxis>> ctrls;

	string root;
	vector<string> frameFeatures;
	vector<vector<Vec3>> frameData;
	string frameFileName;
	string frameWithVelFileName;

	string fullFileName;
	string jointAnglesFileName;
	string fullForwardFileName;
	bool pull;
	int warmCount;
	bool render;
	virtual void KeyDown(int key)
	{
		if (key == 'b')
		{
			pull = !pull;
			cout << "Pull is " << pull << endl;
		}
	}

	virtual bool IsSkipSimulation()
	{
		return true;
	}

	virtual void PreSimulation()
	{
		if (g_frame > 30)
		{
			doInverseDynamic = false;
		}
		if (!g_pause || g_step)
		{
			if (!doInverseDynamic)
			{
				if (render)
				{
					NvFlexSetParams(g_solver, &g_params);
					NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
				}
				else
				{

					NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
					g_buffers->rigidBodies.map();
					NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
					g_buffers->rigidJoints.map();

					while (1)
					{
						Update();
						g_buffers->rigidBodies.unmap();
						NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
						g_buffers->rigidJoints.unmap();
						NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

						NvFlexSetParams(g_solver, &g_params);
						NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
						g_frame++;
						NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
						NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
						g_buffers->rigidBodies.map();
						g_buffers->rigidJoints.map();
						if (warmCount <= 0)
						{
							DumpData();
						}
					}
					g_buffers->rigidBodies.unmap();
					NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!
					g_buffers->rigidJoints.unmap();
					NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size()); // Need to set bodies here too!					}
				}
			}
			else
			{
				//const float effectorCompliance = 1e-12f;
				//const int effectorMaxIterations = 5;
				const float effectorCompliance = 1e-4f;
				const int effectorMaxIterations = 100000000;
				NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
				NvFlexCopyDesc cpd;
				if (g_buffers->positions.size() > 0)
				{

					cpd.srcOffset = 0;
					cpd.dstOffset = 0;

					NvFlexGetParticles(g_solver, g_buffers->positions.buffer, &cpd);
					NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, &cpd);
				}
				NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);

				g_buffers->rigidBodies.map();
				g_buffers->rigidJoints.map();
				if (g_buffers->positions.size() > 0)
				{
					g_buffers->positions.map();
					g_buffers->velocities.map();

					// Disable gravity
					// Disable all target angles
					// Make effector 0 compliance
					bkPos.resize(g_buffers->positions.size());
					bkVel.resize(g_buffers->velocities.size());
				}
				backupBodies.resize(g_buffers->rigidBodies.size());

				// Backup bodies
				memcpy(&backupBodies[0], &g_buffers->rigidBodies[0], sizeof(NvFlexRigidBody)*g_buffers->rigidBodies.size());

				// Go to prev IK
				if (prevBodies.size() > 0)
				{
					memcpy(&g_buffers->rigidBodies[0], &prevBodies[0], sizeof(NvFlexRigidBody)*g_buffers->rigidBodies.size());
				}
				else
				{
					prevBodies.resize(g_buffers->rigidBodies.size());
				}

				if (g_buffers->positions.size() > 0)
				{
					memcpy(&bkPos[0], &g_buffers->positions[0], sizeof(Vec4)*g_buffers->positions.size());
					memcpy(&bkVel[0], &g_buffers->velocities[0], sizeof(Vec3)*g_buffers->velocities.size());
				}

				// Make all effector joint to have very high compliance
				for (int i = 0; i < effectors.size(); i++)
				{
					for (int j = 0; j < 6; j++)
					{
						g_buffers->rigidJoints[effectors[i]].compliance[j] = effectorCompliance;
					}
					g_buffers->rigidJoints[effectors[i]].maxIterations = effectorMaxIterations;
				}

				// Make compliance of active target joints small
				for (int i = 0; i < robotTargetJoints.size(); i++)
				{
					if (robotJointParticipateInInverseDynamic[i])
					{
						g_buffers->rigidJoints[robotTargetJoints[i]].compliance[robotDOFJoints[i]] = 1e30f; // Basically ignore
					}
				}

				// Make tiny part heavier
				float mul = 1000.0f;
				float imul = 1.0f / mul;
				for (int i = 0; i < robotBodies.size(); i++)
				{
					(Vec3&)g_buffers->rigidBodies[robotBodies[i]].linearVel = Vec3(0.0f, 0.0f, 0.0f);
					(Vec3&)g_buffers->rigidBodies[robotBodies[i]].angularVel = Vec3(0.0f, 0.0f, 0.0f);
				}

				for (int i = 0; i < robotTinyBodies.size(); i++)
				{
					g_buffers->rigidBodies[robotTinyBodies[i]].mass *= mul;
					g_buffers->rigidBodies[robotTinyBodies[i]].invMass *= imul;
					//g_buffers->rigidBodies[robotTinyBodies[i]].linearDamping *= mul;
					//g_buffers->rigidBodies[robotTinyBodies[i]].angularDamping *= mul;
					for (int k = 0; k < 9; k++)
					{
						g_buffers->rigidBodies[robotTinyBodies[i]].inertia[k] *= mul;
						g_buffers->rigidBodies[robotTinyBodies[i]].invInertia[k] *= imul;
					}
				}

				if (g_buffers->positions.size() > 0)
				{
					g_buffers->positions.unmap();
					g_buffers->velocities.unmap();
				}
				g_buffers->rigidBodies.unmap();
				NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!

				g_buffers->rigidJoints.unmap();
				NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size()); // Need to set bodies here too!

				//int numActive = NvFlexGetActiveCount(g_solver);
				//NvFlexSetActiveCount(g_solver, 0);
				// Simulate fictitious step
				float bg[3];
				memcpy(bg, &g_params.gravity[0], sizeof(float) * 3);
				g_params.gravity[0] = 0.0f;
				g_params.gravity[1] = 0.0f;
				g_params.gravity[2] = 0.0f;
				NvFlexSetParams(g_solver, &g_params);
				NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

				NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
				g_buffers->rigidBodies.map();

				NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
				g_buffers->rigidJoints.map();

				if (g_buffers->positions.size() > 0)
				{
					g_buffers->positions.map();
					g_buffers->velocities.map();

					memcpy(&g_buffers->positions[0], &bkPos[0], sizeof(Vec4)*g_buffers->positions.size());
					memcpy(&g_buffers->velocities[0], &bkVel[0], sizeof(Vec3)*g_buffers->velocities.size());
					g_buffers->positions.unmap();
					g_buffers->velocities.unmap();
					cpd.elementCount = g_buffers->positions.size();
					NvFlexSetParticles(g_solver, g_buffers->positions.buffer, &cpd);
					NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, &cpd);
				}

				// Do something with the bodies and joints
				// Disable end effectors
				for (int i = 0; i < effectors.size(); i++)
				{
					for (int j = 0; j < 6; j++)
					{
						g_buffers->rigidJoints[effectors[i]].compliance[j] = 1e30f;
					}
				}

				// Restore target joint compliance
				for (int i = 0; i < robotTargetJoints.size(); i++)
				{
					// Compute target
					NvFlexRigidJoint& joint = g_buffers->rigidJoints[robotTargetJoints[i]];
					joint.compliance[robotDOFJoints[i]] = robotDOFComplianceJoints[i]; // Restore compliance

					if (robotJointParticipateInInverseDynamic[i])
					{
						NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
						NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

						Transform body0Pose;
						NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
						Transform body1Pose;
						NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

						Transform pose0 = body0Pose*Transform(joint.pose0.p, joint.pose0.q);
						Transform pose1 = body1Pose*Transform(joint.pose1.p, joint.pose1.q);
						Transform relPose = Inverse(pose0)*pose1;

						Quat qd = relPose.q;

						Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
						Quat qswing = qd*Inverse(qtwist);
						float twist = asin(qtwist.x)*2.0f;
						float swing1 = asin(qswing.y)*2.0f;
						float swing2 = asin(qswing.z)*2.0f;

						if (robotDOFJoints[i] == eNvFlexRigidJointAxisX)
						{
							joint.targets[eNvFlexRigidJointAxisX] = relPose.p.x;
						}
						else if (robotDOFJoints[i] == eNvFlexRigidJointAxisY)
						{
							joint.targets[eNvFlexRigidJointAxisY] = relPose.p.y;
						}
						else if (robotDOFJoints[i] == eNvFlexRigidJointAxisZ)
						{
							joint.targets[eNvFlexRigidJointAxisZ] = relPose.p.z;
						}
						else if (robotDOFJoints[i] == eNvFlexRigidJointAxisTwist)
						{
							joint.targets[eNvFlexRigidJointAxisTwist] = twist;
						}
						else if (robotDOFJoints[i] == eNvFlexRigidJointAxisSwing1)
						{
							joint.targets[eNvFlexRigidJointAxisSwing1] = swing1;
						}
						else if (robotDOFJoints[i] == eNvFlexRigidJointAxisSwing2)
						{
							joint.targets[eNvFlexRigidJointAxisSwing2] = swing2;
						}
						else
						{
							cout << "Can't handle this type of DOF yet!" << endl;
							exit(0);
						}
					}
				}

				// Restore bodies
				memcpy(&prevBodies[0], &g_buffers->rigidBodies[0], sizeof(NvFlexRigidBody)*g_buffers->rigidBodies.size());
				memcpy(&g_buffers->rigidBodies[0], &backupBodies[0], sizeof(NvFlexRigidBody)*g_buffers->rigidBodies.size());
				g_buffers->rigidBodies.unmap();
				NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!

				g_buffers->rigidJoints.unmap();
				NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size()); // Need to set bodies here too!

				//NvFlexSetActiveCount(g_solver, numActive);

				// Restore gravity
				memcpy(&g_params.gravity[0], bg, sizeof(float) * 3);

				// tick solver
				NvFlexSetParams(g_solver, &g_params);
				NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
			}

			g_frame++;
			g_step = false;
		}
	}

	RigidBVHRetargetTest()
	{
		render = false;
		pull = true;
		warmCount = 60;
		workmode = DUMP_FULL_STATE_AND_JOINT_ANGLES;//DUMP_FULL_STATE_AND_JOINT_ANGLES;
		prevBodies.clear();
		doInverseDynamic = false;
		//to_meters = 5.6444; // Actual value
		to_meters = 6.6f;
		//to_meters = 1.0f;

		//MJCFImporter("../../data/humanoid_symmetric.xml", Transform(Vec3(i, 3.0f, 5.0f), Quat()), ctrls, motors);
		//MJCFImporter("../../data/ant.xml", Transform(Vec3(i*2, 3.0f, 5.0f), Quat()), ctrls, motors);
		vector<float> motors;
		worldTrans = Transform(Vec3(0.0f, 0.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
		worldTransInv = Inverse(worldTrans);
		mjcf = new MJCFImporter("../../data/humanoid_mod.xml");
		mjcf->AddPhysicsEntities(worldTrans, ctrls, motors, false, true);

		if (skeleton)
		{
			delete skeleton;
		}
		skeleton = new Skeleton();
		//skeleton->loadFromBvh("../../data/bvh/NewCaptures10_000.bvh");
		//skeleton->loadFromBvh("../../data/bvh/01_06.bvh");
		/*
		skeleton->loadFromBvh("../../data/bvh/LocomotionFlat02_000.bvh");
		fileName = "../../data/bvh/LocomotionFlat02_000.pose";
		fullForwardFileName = "../../data/bvh/LocomotionFlat02_000.state";
		fullFileName = "../../data/bvh/LocomotionFlat02_000_full.state";
		jointAnglesFileName = "../../data/bvh/LocomotionFlat02_000_joint_angles.state";
		frameFileName = "../../data/bvh/motion_frame.dat";
		frameWithVelFileName = "../../data/bvh/motion_frame_with_vel.dat";
		*/
		/*
		skeleton->loadFromBvh("../../data/bvh/LocomotionFlat12_000.bvh");
		fileName = "../../data/bvh/LocomotionFlat12_000.pose";
		fullForwardFileName = "../../data/bvh/LocomotionFlat12_000.state";
		fullFileName = "../../data/bvh/LocomotionFlat12_000_full.state";
		jointAnglesFileName = "../../data/bvh/LocomotionFlat12_000_joint_angles.state";
		frameFileName = "../../data/bvh/motion_12_frame.dat";
		frameWithVelFileName = "../../data/bvh/motion_12_frame_with_vel.dat";
		*/
		/*
		skeleton->loadFromBvh("../../data/bvh/LocomotionFlat11_000.bvh");
		fileName = "../../data/bvh/LocomotionFlat11_000.pose";
		fullForwardFileName = "../../data/bvh/LocomotionFlat11_000.state";
		fullFileName = "../../data/bvh/LocomotionFlat11_000_full.state";
		jointAnglesFileName = "../../data/bvh/LocomotionFlat11_000_joint_angles.state";
		frameFileName = "../../data/bvh/motion_11_frame.dat";
		frameWithVelFileName = "../../data/bvh/motion_11_frame_with_vel.dat";
		*/
		//string prefix = "../../data/bvh/LocomotionFlat07_000";
		string prefix = string("../../data/bvh/") + g_argv[1];
		//string prefix = string("../../data/bvh/") + "NewCaptures11_000";
		skeleton->loadFromBvh(prefix + ".bvh");
		fileName = prefix + ".pose";
		fullForwardFileName = prefix + ".state";
		fullFileName = prefix + "_full.state";
		jointAnglesFileName = prefix + "_joint_angles.state";

		frameFileName = prefix + "_frame.dat";
		frameWithVelFileName = prefix + "_frame_with_vel.dat";

		frameData.clear();
		fullTrans.clear();
		fullVels.clear();
		fullAVels.clear();
		jointAngles.clear();
		//skeleton->loadFromBvh("../../data/bvh/137_29.bvh");
		//fileName = "../../data/bvh/137_29.pose";

		//skeleton->loadFromBvh("../../data/bvh/capture_007.bvh");
		//fileName = "../../data/bvh/capture_007.pose";

		//skeleton->loadFromBvh("../../data/bvh/01_06.bvh");
		//fileName = "../../data/bvh/01_06.pose";

		vector<Skeleton::Bone>& bones = skeleton->getBones();
		vector<Transform>& trans = skeleton->getBoneTransforms();

		vector<Vec3> gpos;
		vector<Vec3> bpos;

		geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
		bvh = { 0,12,13,25,26,27,18,19,20,7,8,9,2,3,4 };

		root = "torso1";
		frameFeatures = { "left_hand","right_hand","left_foot","right_foot","head" };

		//geo_joint = { "torso1","right_thigh", "right_foot","left_thigh","left_foot" };
		//bvh = { 13,7,9,2,4 };

		//vector<int> bvh = { 25,26,27,18,19,20,7,8,10,2,3,5 };
		skeleton->setAnimTime(0);
		for (int i = 0; i < geo_joint.size(); i++)
		{
			//for (int i = 0; i <5; i++) {
			if (mjcf->geoBodyPose.find(geo_joint[i]) == mjcf->geoBodyPose.end())
			{
				cout << geo_joint[i] << " not found" << endl;
			}
			auto p = mjcf->geoBodyPose[geo_joint[i]];
			Transform tt = trans[bvh[i]] * bones[bvh[i]].globalBindPose;
			tt.p *= to_meters;
			NvFlexRigidJoint fj;
			NvFlexMakeFixedJoint(&fj, -1, p.first, NvFlexMakeRigidPose(tt.p, Quat()), NvFlexMakeRigidPose(p.second.p, p.second.q));
			bpos.push_back(tt.p);

			Transform tf;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
			tf = tf*p.second;
			gpos.push_back(tf.p);

			for (int q = 0; q < 3; ++q)
			{
				fj.compliance[q] = 1e-8f;
				fj.damping[q] = 0.0f;
			}
			for (int q = 3; q < 6; q++)
			{
				fj.modes[q] = eNvFlexRigidJointModeFree;
			}
			jointToBVH.push_back(make_pair(g_buffers->rigidJoints.size(), bvh[i]));
			g_buffers->rigidJoints.push_back(fj);
		}

		vector<pair<int, int> > pis;
		pis.push_back(make_pair(13, 12));
		pis.push_back(make_pair(12, 0));
		pis.push_back(make_pair(2, 3));
		pis.push_back(make_pair(3, 4));
		pis.push_back(make_pair(18, 19));
		pis.push_back(make_pair(19, 20));
		pis.push_back(make_pair(2, 7));
		for (int i = 0; i < pis.size(); i++)
		{
			int ia = -1;
			int ib = -1;
			for (int a = 0; a < bvh.size(); a++)
			{
				if (bvh[a] == pis[i].first)
				{
					ia = a;
				}
				if (bvh[a] == pis[i].second)
				{
					ib = a;
				}
			}

			cout << "Distance between " << geo_joint[ia] << " and " << geo_joint[ib] << " is " << Length(gpos[ia] - gpos[ib]) << " but should be " << Length(bpos[ia] - bpos[ib]) << endl;
		}
		effectors.clear();

		robotDOFJoints.clear();
		robotDOFComplianceJoints.clear();

		robotTargetJoints.clear();
		robotBodies.clear();
		robotJointParticipateInInverseDynamic.clear();

		for (auto e : jointToBVH)
		{
			effectors.push_back(e.first);
		}

		// Setup active joint stiffness to that of gear
		for (auto m : mjcf->motors)
		{
			pair<int, NvFlexRigidJointAxis> aj = mjcf->activeJoints[mjcf->activeJointsNameMap[m->joint]];
			g_buffers->rigidJoints[aj.first].compliance[aj.second] = (1.0f / m->gear);
		}

		for (auto c : mjcf->activeJoints)
		{
			robotDOFJoints.push_back(c.second);
			g_buffers->rigidJoints[c.first].compliance[c.second] = 0.0f;
			robotDOFComplianceJoints.push_back(g_buffers->rigidJoints[c.first].compliance[c.second]);
			robotTargetJoints.push_back(c.first);
			robotJointParticipateInInverseDynamic.push_back(true);
			g_buffers->rigidJoints[c.first].compliance[c.second] = 1e30f; // Ignore for now
		}
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
		{
			robotBodies.push_back(i);
		}

		//skeleton->loadFromBvh("../../data/bvh/01_01.bvh");

		/*
		Mesh* terrain = ImportMesh("../../data/terrain.obj");
		terrain->Transform(RotationMatrix(-kPi*0.5f, Vec3(1.0f, 0.0f, 0.0f)));
		terrain->Normalize(10.0f);

		NvFlexTriangleMeshId terrainId = CreateTriangleMesh(terrain);

		NvFlexRigidShape terrainShape;
		NvFlexMakeRigidTriangleMeshShape(&terrainShape, -1, terrainId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
		terrainShape.filter = 0;

		g_buffers->rigidShapes.push_back(terrainShape);

		*/
		//g_params.numIterations = 100;
		g_params.numIterations = 20;// default
		g_params.dynamicFriction = 0.2f; // 0.2

		g_drawPoints = false;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);

		g_warmup = true;
		g_pause = false;
		time = 0.0f;

		if ((workmode == PREPARE_DATA) || (workmode == PREPARE_DATA_LOAD_STAT))
		{
			// Load poses
			//vector<string> poseNames = { "../../data/bvh/capture_007.pose", "../../data/bvh/LocomotionFlat02_000.pose","../../data/bvh/01_06.pose" };
			//vector<float> plabels = { 0, 1, 1 };

			//vector<string> poseNames = {"../../data/bvh/LocomotionFlat02_000.pose"};
			//vector<float> plabels = {  1 };

			const char* statFile = "../../data/bvh/motion_med_reduced.dat.inf";
			const char* motionFile = "../../data/bvh/motion_med_reduced.dat";
			const char* motionLabelFile = "../../data/bvh/motion_med_reduced.label";
			vector<string> poseNames = { "../../data/bvh/LocomotionFlat02_000.pose" };
			vector<float> plabels = { 0 };

			int numTrans = geo_joint.size();
			int skipFrame = 3;
			int hWindow = 3;
			/*
			int skipFrame = 6;
			int hWindow = 5;
			*/
			vector<vector<float>> data;
			vector<float> labels;
			vector<Vec3> features;

			for (int p = 0; p < poseNames.size(); p++)
			{
				FILE* f = fopen(poseNames[p].c_str(), "rb");

				int numFrames = 0;
				fread(&numFrames, 1, sizeof(int), f);

				for (int i = 0; i < numFrames; i++)
				{
					allTrans.push_back(vector<Transform>());
					allTrans.back().resize(geo_joint.size());
					fread(&allTrans.back()[0], sizeof(Transform), numTrans, f);
				}
				fclose(f);

				for (int i = skipFrame*hWindow; i < numFrames - skipFrame*hWindow - 1; i++)
				{
					// Everything relative to lwaist coordinate system of frame i
					Transform trans = allTrans[i][0];
					Transform itrans = Inverse(trans);
					features.clear();
					for (int f = -hWindow; f <= hWindow; f++)
					{
						for (int j = 0; j < numTrans; j++)
						{
							Vec3 pos = TransformPoint(itrans, allTrans[i + skipFrame*f][j].p);
							features.push_back(pos);
						}
					}
					data.push_back(vector<float>());
					for (int k = 0; k < features.size(); k++)
					{
						data.back().push_back(features[k].x);
						data.back().push_back(features[k].y);
						data.back().push_back(features[k].z);
					}
					labels.push_back(plabels[p]);
				}
			}

			float ax = 0.0, ay = 0.0, az = 0.0;
			float maxVal = 0.0f;
			int num = 0;
			for (int i = 0; i < data.size(); i++)
			{
				for (int j = 0; j < data[i].size(); j += 3)
				{
					ax += data[i][j];
					ay += data[i][j + 1];
					az += data[i][j + 2];
					num++;
				}
			}
			ax /= num;
			ay /= num;
			az /= num;

			float dx = 0.0, dy = 0.0, dz = 0.0;
			for (int i = 0; i < data.size(); i++)
			{
				for (int j = 0; j < data[i].size(); j += 3)
				{
					float tx = data[i][j];
					float ty = data[i][j + 1];
					float tz = data[i][j + 2];
					dx += (tx - ax)*(tx - ax);
					dy += (ty - ay)*(ty - ay);
					dz += (tz - az)*(tz - az);
				}
			}
			float isdx = 1.0f / sqrtf(dx / (num - 1));
			float isdy = 1.0f / sqrtf(dy / (num - 1));
			float isdz = 1.0f / sqrtf(dz / (num - 1));

			FILE* f;
			cout << "Each data point has " << data[0].size() << endl;
			if (workmode == PREPARE_DATA_LOAD_STAT)
			{
				f = fopen(statFile, "rt");
				fscanf(f, "%f %f %f %f %f %f\n", &ax, &ay, &az, &isdx, &isdy, &isdz);
				fclose(f);
				cout << "From file, Mean was " << ax << " " << ay << " " << az << " SD was " << (1.0 / isdx) << " " << (1.0 / isdy) << " " << (1.0 / isdz) << endl;
			}
			else
			{
				cout << "Mean was " << ax << " " << ay << " " << az << " SD was " << (1.0 / isdx) << " " << (1.0 / isdy) << " " << (1.0 / isdz) << endl;
				f = fopen(statFile, "wt");
				fprintf(f, "%f %f %f %f %f %f\n", ax, ay, az, isdx, isdy, isdz);
				fclose(f);

			}

			for (int i = 0; i < data.size(); i++)
			{
				for (int j = 0; j < data[i].size(); j += 3)
				{
					data[i][j] = (data[i][j] - ax)*isdx;
					data[i][j + 1] = (data[i][j + 1] - ay)*isdy;
					data[i][j + 2] = (data[i][j + 2] - az)*isdz;
				}
			}

			f = fopen(motionFile, "wt");
			for (int i = 0; i < data.size(); i++)
			{
				for (int j = 0; j < data[i].size(); j++)
				{
					if (j != 0)
					{
						fprintf(f, " ");
					}
					fprintf(f, "%lf", data[i][j]);
				}
				fprintf(f, "\n");
			}
			fclose(f);

			f = fopen(motionLabelFile, "wt");
			for (int i = 0; i < labels.size(); i++)
			{
				fprintf(f, "%f\n", labels[i]);
			}
			fclose(f);
		}
	}

	virtual void Update()
	{
		warmCount--;
		if (warmCount <= 0)
		{
			time += 1.f / 60.0f;
			skeleton->setAnimTime(time);
		}
		vector<Skeleton::Bone>& bones = skeleton->getBones();
		vector<Transform>& trans = skeleton->getBoneTransforms();
		Transform offset;
		for (int i = 0; i < jointToBVH.size(); i++)
		{
			Transform tt = trans[jointToBVH[i].second] * bones[jointToBVH[i].second].globalBindPose;
			tt.p *= to_meters;
			NvFlexRigidJoint& fj = g_buffers->rigidJoints[jointToBVH[i].first];
			if (workmode == DUMP_FULL_STATE_FORWARD_FACING)
			{
				// Rotate to forward facing
				if (i == 0)
				{
					Transform tmp = tt;
					tmp.p.x = 0.0f;
					tmp.p.z = 0.0f;
					Vec3 heading = Rotate(tmp.q, Vec3(1.0f, 0.0f, 0.0f));
					float angle = atan2(heading.z, heading.x);
					tmp.q = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), angle)*tmp.q;
					offset = tmp*Inverse(tt);
				}
				tt = offset*tt;
			}
			fj.pose0 = NvFlexMakeRigidPose(tt.p, Quat());

			for (int k = 0; k < 3; k++)
			{
				fj.compliance[k] = (pull) ? 1e-5f : 1e30f;
			}
			//if (i == 0) cout << "Set to " << tt.p.x << " " << tt.p.y << " " << tt.p.z << endl;
		}
	}

	virtual void DoStats()
	{
		if (warmCount > 0)
		{
			return;
		}
		vector<Skeleton::Bone>& bones = skeleton->getBones();
		vector<Transform>& trans = skeleton->getBoneTransforms();
		vector<Vec3> prevPos;
		vector<Vec3> allPos;
		for (int i = 0; i < bones.size(); i++)
		{
			Transform tt = trans[i] * bones[i].globalBindPose;
			tt.p *= to_meters;
			allPos.push_back(tt.p);
			bool duplicate = false;
			for (int j = 0; j < prevPos.size(); j++)
			{
				if (LengthSq(tt.p - prevPos[j]) < 1e-5)
				{
					duplicate = true;
					break;
				}
			}
			if (duplicate)
			{
				continue;
			}
			prevPos.push_back(tt.p);
			Vec3 sc = GetScreenCoord(tt.p);
			if (sc.z < 1.0f)
			{
				DrawImguiString(int(sc.x + 5.f), int(sc.y - 5.f), Vec3(1.f, 1.f, 0.f), 0, "%d", i);
			}
		}

		//		vector<Skeleton::Bone>& bones = skeleton->getBones();
		//	vector<Transform>& trans = skeleton->getBoneTransforms();
		DumpData();

	}
	void DumpData()
	{
		vector<Skeleton::Bone>& bones = skeleton->getBones();
		vector<Transform>& trans = skeleton->getBoneTransforms();

		for (int i = 0; i < geo_joint.size(); i++)
		{
			//for (int i = 0; i <5; i++) {
			if (mjcf->geoBodyPose.find(geo_joint[i]) == mjcf->geoBodyPose.end())
			{
				cout << geo_joint[i] << " not found" << endl;
			}
			auto p = mjcf->geoBodyPose[geo_joint[i]];
			Transform tt = trans[bvh[i]] * bones[bvh[i]].globalBindPose;
			tt.p *= to_meters;
			Transform tf;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
			tf = tf*p.second;
			Vec3 sc1 = GetScreenCoord(tt.p);
			Vec3 sc2 = GetScreenCoord(tf.p);
			if (sc1.z < 1.0f)
			{
				DrawImguiString(int(sc1.x + 5.0f), int(sc1.y - 5.0f), Vec3(1, 0, 1), 0, "%d", bvh[i]);
			}
			if (sc2.z < 1.0f)
			{
				DrawImguiString(int(sc2.x + 5.0f), int(sc2.y - 5.0f), Vec3(1, 0, 0), 0, "%d", bvh[i]);
			}

			/*
			NvFlexRigidJoint fj;
			NvFlexMakeFixedJoint(&fj, -1, p.first, NvFlexMakeRigidPose(tt.p, Quat()), NvFlexMakeRigidPose(p.second.p, p.second.q));
			for (int q = 0; q < 3; ++q)
			{
			fj.compliance[q] = 1e-5;
			fj.damping[q] = 0.0f;
			}
			for (int q = 3; q < 6; q++)
			{
			fj.modes[q] = eNvFlexRigidJointModeFree;
			}
			jointToBVH.push_back(make_pair(g_buffers->rigidJoints.size(), bvh[i]));
			g_buffers->rigidJoints.push_back(fj);
			*/
		}
		if ((!g_pause) && (time < skeleton->getAnimLength()))
		{
			if (workmode == DUMP_DATA_FRAME)
			{
				vector<Vec3> features;
				auto p = mjcf->geoBodyPose[root];
				Transform tf;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
				Transform invRoot = Inverse(tf);
				Vec3 rootPos = tf.p;
				Vec3 cvel = TransformVector(invRoot, (Vec3&)g_buffers->rigidBodies[p.first].linearVel);
				Vec3 cavel = TransformVector(invRoot, (Vec3&)g_buffers->rigidBodies[p.first].angularVel);
				Vec3 cup = tf.q * Vec3(0.0f, 0.0f, 1.0f); // Up is Z for mujoco
				features.push_back(cvel);
				features.push_back(cavel);
				features.push_back(cup);

				for (int i = 0; i < frameFeatures.size(); i++)
				{
					auto p = mjcf->geoBodyPose[frameFeatures[i]];

					NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
					tf = tf*p.second;
					features.push_back(TransformVector(invRoot, tf.p - rootPos));
				}
				frameData.push_back(features);
			}
			else if (workmode == DUMP_DATA_FRAME_WITH_VEL)
			{
				vector<Vec3> features;
				auto p = mjcf->geoBodyPose[root];
				Transform tf;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
				Transform invRoot = Inverse(tf);
				Vec3 rootPos = tf.p;
				Vec3 rootVel = (Vec3&)g_buffers->rigidBodies[p.first].linearVel;
				Vec3 cvel = TransformVector(invRoot, rootVel);
				Vec3 cavel = TransformVector(invRoot, (Vec3&)g_buffers->rigidBodies[p.first].angularVel);
				Vec3 cup = tf.q * Vec3(0.0f, 0.0f, 1.0f); // Up is Z for mujoco
				features.push_back(cvel);
				features.push_back(cavel);
				features.push_back(cup);

				for (int i = 0; i < frameFeatures.size(); i++)
				{
					auto p = mjcf->geoBodyPose[frameFeatures[i]];

					NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
					tf = tf*p.second;
					features.push_back(TransformVector(invRoot, tf.p - rootPos));
					features.push_back(TransformVector(invRoot, Cross((Vec3&)g_buffers->rigidBodies[p.first].angularVel, tf.p - (Vec3&)g_buffers->rigidBodies[p.first].com) + (Vec3&)g_buffers->rigidBodies[p.first].linearVel - rootVel));
				}
				frameData.push_back(features);
			}
			allTrans.push_back(vector<Transform>());
			//cout << "Store transform frame " << allTrans.size() - 1 << endl;
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[0], (NvFlexRigidPose*)&pose);

			//cout << pose.p.z << " " << g_buffers->rigidBodies[0].linearVel[2] << endl;

			for (int i = 0; i < geo_joint.size(); i++)
			{
				//for (int i = 0; i <5; i++) {
				if (mjcf->geoBodyPose.find(geo_joint[i]) == mjcf->geoBodyPose.end())
				{
					cout << geo_joint[i] << " not found" << endl;
				}
				auto p = mjcf->geoBodyPose[geo_joint[i]];
				Transform tf;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
				tf = tf*p.second;

				allTrans.back().push_back(tf);
				/*if (time < skeleton->getAnimLength()) {
				cout << tf.p.x << " " << tf.p.y << " " << tf.p.z << ", ";
				}
				*/
			}

			if (allTrans.size() > 10)
			{
				// Ignore first few frame to avoid transient
				fullTrans.push_back(vector<Transform>());
				fullVels.push_back(vector<Vec3>());
				fullAVels.push_back(vector<Vec3>());
				for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
				{
					Transform pose;
					NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&pose);
					pose = worldTransInv*pose;
					Vec3 lvel = Vec3(g_buffers->rigidBodies[i].linearVel), avel = Vec3(g_buffers->rigidBodies[i].angularVel);
					lvel = Rotate(worldTransInv.q, lvel);
					avel = Rotate(worldTransInv.q, avel);
					fullTrans.back().push_back(pose);
					fullVels.back().push_back(lvel);
					fullAVels.back().push_back(avel);
				}

				jointAngles.push_back(vector<float>());

				float prevTwist, prevSwing1, prevSwing2;
				Vec3 prevPos;
				int prevIdx = -1;
				for (int i = 0; i < (int)ctrls.size(); i++)
				{
					int qq = i;

					//hJointToD6[qq].first->calculateTransforms();
					float pos; //, vel;
					float low, high;
					NvFlexRigidJoint& joint = g_buffers->rigidJoints[ctrls[qq].first];
					if (ctrls[qq].first != prevIdx)
					{
						NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
						NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

						Transform body0Pose;
						NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
						Transform body1Pose;
						NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

						Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
						Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
						Transform relPose = Inverse(pose0)*pose1;

						prevPos = relPose.p;
						Quat qd = relPose.q;

						Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
						Quat qswing = qd*Inverse(qtwist);
						prevTwist = asin(qtwist.x)*2.0f;
						prevSwing1 = asin(qswing.y)*2.0f;
						prevSwing2 = asin(qswing.z)*2.0f;
						prevIdx = ctrls[qq].first;

						// If same, no need to recompute
					}

					int idx = ctrls[qq].second;
					low = joint.lowerLimits[idx];
					high = joint.upperLimits[idx];
					if (idx == 3)
					{
						pos = prevTwist;
					}
					else if (idx == 4)
					{
						pos = prevSwing1;
					}
					else if (idx == 5)
					{
						pos = prevSwing2;
					}
					else if (idx == 0)
					{
						pos = prevPos.x;
					}
					else if (idx == 1)
					{
						pos = prevPos.y;
					}
					else if (idx == 2)
					{
						pos = prevPos.z;
					}
					jointAngles.back().push_back(pos);
				}
			}
		}
		if (time >= skeleton->getAnimLength())
		{
			// Write to file and quit
			if (workmode == DUMP_DATA_FRAME_WITH_VEL)
			{
				FILE* f = fopen(frameWithVelFileName.c_str(), "wt");
				for (int i = 0; i < frameData.size(); i++)
				{
					for (int j = 0; j < frameData[i].size(); j++)
					{
						if (j > 0)
						{
							fprintf(f, " ");
						}
						fprintf(f, "%f %f %f", frameData[i][j].x, frameData[i][j].y, frameData[i][j].z);
					}
					fprintf(f, "\n");
				}
				fclose(f);
			}
			else if (workmode == DUMP_DATA_FRAME)
			{
				FILE* f = fopen(frameFileName.c_str(), "wt");
				for (int i = 0; i < frameData.size(); i++)
				{
					for (int j = 0; j < frameData[i].size(); j++)
					{
						if (j > 0)
						{
							fprintf(f, " ");
						}
						fprintf(f, "%f %f %f", frameData[i][j].x, frameData[i][j].y, frameData[i][j].z);
					}
					fprintf(f, "\n");
				}
				fclose(f);
			}
			else if (workmode == DUMP_POSE)
			{
				FILE* f = fopen(fileName.c_str(), "wb");
				int numFrames = allTrans.size();
				cout << "Writing out " << numFrames << " frames" << endl;
				fwrite(&numFrames, 1, sizeof(int), f);
				for (int i = 0; i < numFrames; i++)
				{
					fwrite(&allTrans[i][0], sizeof(Transform), allTrans[i].size(), f);
				}
				fclose(f);
			}
			else if ((workmode == DUMP_FULL_STATE_FORWARD_FACING) || (workmode == DUMP_FULL_STATE) || (workmode == DUMP_FULL_STATE_AND_JOINT_ANGLES))
			{
				FILE* f;
				if (workmode == DUMP_FULL_STATE_FORWARD_FACING)
				{
					f = fopen(fullForwardFileName.c_str(), "wb");
				}
				else if ((workmode == DUMP_FULL_STATE) || (workmode == DUMP_FULL_STATE_AND_JOINT_ANGLES))
				{
					f = fopen(fullFileName.c_str(), "wb");
				}
				int numFrames = fullTrans.size();
				cout << "Writing out " << numFrames << " frames of full data" << endl;
				fwrite(&numFrames, 1, sizeof(int), f);
				int numTrans = 0;
				if (numFrames > 0)
				{
					numTrans = fullTrans[0].size();
				}
				fwrite(&numTrans, 1, sizeof(int), f);

				for (int i = 0; i < numFrames; i++)
				{
					fwrite(&fullTrans[i][0], sizeof(Transform), fullTrans[i].size(), f);
					fwrite(&fullVels[i][0], sizeof(Vec3), fullVels[i].size(), f);
					fwrite(&fullAVels[i][0], sizeof(Vec3), fullAVels[i].size(), f);
				}
				fclose(f);

				if (workmode == DUMP_FULL_STATE_AND_JOINT_ANGLES)
				{
					f = fopen(jointAnglesFileName.c_str(), "wb");
					numFrames = jointAngles.size();
					fwrite(&numFrames, 1, sizeof(int), f);
					int numAngles = 0;
					if (numFrames > 0)
					{
						numAngles = jointAngles[0].size();
					}
					fwrite(&numAngles, 1, sizeof(int), f);
					for (int i = 0; i < numFrames; i++)
					{
						fwrite(&jointAngles[i][0], sizeof(float), numAngles, f);
					}
					fclose(f);
				}
			}
			exit(0);
		}

		//cout << endl;

		/*
		#define PrintDis(a,b) {cout<<"Distance "<<(a)<<" to "<<(b)<<" is "<<Length(allPos[a]-allPos[b])<<endl;}
		#define PrintDisY(a,b) {cout<<"Distance Y"<<(a)<<" to "<<(b)<<" is "<<fabs(allPos[a].y-allPos[b].y)<<endl;}
		PrintDis(2, 3);
		PrintDis(3, 4);
		PrintDis(18, 19);
		PrintDis(19, 20);
		PrintDis(18, 25);
		PrintDis(7, 2);
		PrintDis(17, 2);
		PrintDisY(17, 2);
		*/
	}
	virtual void Draw(int pass)
	{
		if (g_showHelp)
		{
			BeginLines();

			vector<Skeleton::Bone>& bones = skeleton->getBones();
			vector<Transform>& trans = skeleton->getBoneTransforms();
			for (int i = 0; i < bones.size(); i++)
			{
				Transform tt = trans[i] * bones[i].globalBindPose;
				tt.p *= to_meters;
				if (bones[i].parentNr >= 0)
				{
					Transform pp = trans[bones[i].parentNr] * bones[bones[i].parentNr].globalBindPose;
					pp.p *= to_meters;
					Vec3 parent = pp.p;
					Vec3 child = tt.p;//TransformPoint(tt, bones[i].localChildOffset);
					DrawLine(parent, child, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
					//cout << "Draw " << parent.x << " " << parent.y << " " << parent.z << " to " << child.x << " " << child.y << " " << child.z << " to " << endl;
				}
				DrawLine(tt.p + Vec3(-0.01f, 0.0f, 0.0f), tt.p + Vec3(0.01f, 0.0f, 0.0f), Vec4(1.0f));
				DrawLine(tt.p + Vec3(0.0f, -0.01f, 0.0f), tt.p + Vec3(0.0f, 0.01f, 0.0f), Vec4(1.0f));
				DrawLine(tt.p + Vec3(0.0f, 0.0f, -0.01f), tt.p + Vec3(0.0f, 0.0f, 0.01f), Vec4(1.0f));
			}

			EndLines();
		}
	}
};


class RigidBVHTestDumpFile : public Scene
{
public:
	enum MODE { TARGET_ANGLE, TARGET_POSE };
	MODE mode;
	MJCFImporter* mjcf;

	vector<vector<Transform>> fullTrans;
	vector<vector<Vec3>> fullVels;
	vector<vector<Vec3>> fullAVels;

	// For dumping out target angles
	vector<vector<float>> jointAngles;
	vector<pair<int, NvFlexRigidJointAxis>> ctrls;
	vector<float> motors;

	string fullFileName;
	string jointAnglesFileName;
	Transform worldTrans;
	Transform worldTransInv;
	int frame;

	vector<int> fjoints;
	Transform shift;
	vector<pair<int, Transform>> features;
	vector<int> feetOnFloor;
	int firstFrame;
	int lastFrame;
	RigidBVHTestDumpFile()
	{
		mode = TARGET_POSE;

		worldTrans = Transform(Vec3(0.0f, 0.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
		worldTransInv = Inverse(worldTrans);
		mjcf = new MJCFImporter("../../data/humanoid_mod.xml");
		mjcf->AddPhysicsEntities(worldTrans, ctrls, motors, true, true);

		string fprefix = g_argv[1];
		fullFileName = "../../data/bvh/" + fprefix + "_full.state";
		jointAnglesFileName = "../../data/bvh/" + fprefix + "_joint_angles.state";

		fullTrans.clear();
		fullVels.clear();
		fullAVels.clear();
		jointAngles.clear();

		FILE* f = fopen(fullFileName.c_str(), "rb");
		int numFrames;
		fread(&numFrames, 1, sizeof(int), f);
		fullTrans.resize(numFrames);
		fullVels.resize(numFrames);
		fullAVels.resize(numFrames);
		int numTrans;
		fread(&numTrans, 1, sizeof(int), f);

		for (int i = 0; i < numFrames; i++)
		{
			fullTrans[i].resize(numTrans);
			fullVels[i].resize(numTrans);
			fullAVels[i].resize(numTrans);

			fread(&fullTrans[i][0], sizeof(Transform), fullTrans[i].size(), f);
			fread(&fullVels[i][0], sizeof(Vec3), fullVels[i].size(), f);
			fread(&fullAVels[i][0], sizeof(Vec3), fullAVels[i].size(), f);
		}
		fclose(f);

		f = fopen(jointAnglesFileName.c_str(), "rb");
		fread(&numFrames, 1, sizeof(int), f);
		jointAngles.resize(numFrames);

		int numAngles;
		fread(&numAngles, 1, sizeof(int), f);
		for (int i = 0; i < numFrames; i++)
		{
			jointAngles[i].resize(numAngles);
			fread(&jointAngles[i][0], sizeof(float), numAngles, f);
		}
		fclose(f);

		if (mode == TARGET_POSE)
		{
			/*
			for (int i = 0; i < g_buffers->rigidBodies.size(); i++) {
			NvFlexRigidJoint fj;
			Transform mm = worldTrans * fullTrans[0][i];
			NvFlexMakeFixedJoint(&fj, -1, i, NvFlexMakeRigidPose(mm.p, mm.q), NvFlexMakeRigidPose(Vec3(), Quat()));
			for (int j = 0; j < 6; j++) {
			fj.compliance[j] = 1e-3f;
			}

			g_buffers->rigidJoints.push_back(fj);
			fjoints.push_back(i);
			}
			*/

		}
		vector<string> geo_joint = { "left_foot","right_foot" };
		features.clear();
		for (int i = 0; i < geo_joint.size(); i++)
		{
			auto p = mjcf->geoBodyPose[geo_joint[i]];
			features.push_back(p);
		}
		feetOnFloor.resize(fullTrans.size());
		for (int frameNum = 0; frameNum < fullTrans.size(); frameNum++)
		{
			int nf = 0;
			cout << frameNum << " -- ";
			for (int i = 0; i < features.size(); i++)
			{
				Vec3 pTarget = TransformPoint(fullTrans[frameNum][features[i].first - mjcf->firstBody], features[i].second.p);
				Vec3 vel = fullVels[frameNum][features[i].first - mjcf->firstBody];
				TransformPoint(worldTransInv, pTarget);
				//cout << "(" << i << " : " << pTarget.x << " " << pTarget.y << " " << pTarget.z << ") ";
				cout << pTarget.z << ":" << vel.z;
				if ((pTarget.z < 0.15) && (vel.z < 0.2f))
				{
					nf++;
				}
			}
			cout << endl;
			feetOnFloor[frameNum] = nf;

		}
		frame = 0;
		firstFrame = 0;
		lastFrame = fullTrans.size();
		shift = worldTrans*fullTrans[lastFrame - 1][0] * Inverse(worldTrans*fullTrans[firstFrame][0]);

		g_params.numIterations = 20;
		g_params.dynamicFriction = 0.2f;

		g_drawPoints = false;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_warmup = true;
		g_pause = false;
	}

	virtual void DoGui()
	{
		float ff = (float)firstFrame, lf = (float)lastFrame;
		imguiSlider("First Frame", &ff, 0.0f, 50.0f, 1.0f);
		imguiSlider("Last Frame", &lf, 80.0f, 150.0f, 1.0f);
		firstFrame = (int)ff;
		lastFrame = (int)lf;
	}

	virtual void Update()
	{
		frame++;
		frame = frame % (lastFrame - firstFrame);
		if (mode == TARGET_ANGLE)
		{
			for (int i = 0; i < (int)ctrls.size(); i++)
			{
				int qq = i;
				NvFlexRigidJoint& joint = g_buffers->rigidJoints[ctrls[qq].first + 1]; // Active joint
				joint.compliance[ctrls[qq].second] = 0.1f / motors[i];
				joint.targets[ctrls[qq].second] = jointAngles[frame + firstFrame][i];
				if (i == 20)
				{
					joint.targets[ctrls[qq].second] *= -1.0f;
				}
			}
		}
		else if (mode == TARGET_POSE)
		{

			for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
			{
				//(Transform&)g_buffers->rigidJoints[fjoints[i]].pose0 = worldTrans*fullTrans[frame + firstFrame][i];
				/*
				int pad = 20;
				Transform mm;
				if (frame >= (lastFrame - firstFrame) - pad) {
				Transform t0 = fullTrans[frame + firstFrame][i];
				Transform t1 = fullTrans[firstFrame - (lastFrame - (frame + firstFrame))][i];
				float f = (pad - (lastFrame - firstFrame)) / (float)(pad);

				}
				else {
				mm = worldTrans * fullTrans[frame + firstFrame][i];
				}*/
				Transform mm = worldTrans * fullTrans[frame + firstFrame][i];
				NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&mm);
			}
			if (frame == (lastFrame - firstFrame) - 1)
			{
				worldTrans = shift*worldTrans;
			}
		}
	}

	virtual void DoStats()
	{
		Vec3 sc(400.0f, 400.0f, 0.0f);
		DrawImguiString(int(sc.x + 5.f), int(sc.y - 5.f), Vec3(1.f, 1.f, 0.f), 0, "%d %d", frame, feetOnFloor[frame + firstFrame]);
	}

	virtual void Draw(int pass)
	{
	}
};


class RigidFullHumanoidTrackTargetAngles : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	int firstFrame;
	int lastFrame;
	vector<int> startFrame;
	vector<int> rightFoot;
	vector<int> leftFoot;

	vector<int> footFlag;

	vector<vector<Transform>> fullTrans;
	vector<vector<Vec3>> fullVels;
	vector<vector<Vec3>> fullAVels;
	vector<vector<float>> jointAngles;
	string fullFileName;

	vector < vector<pair<int, Transform>>> features;
	vector<string> geo_joint;
	float ax, ay, az;
	float isdx, isdy, isdz;
	bool useGeoMatching;
	int numFramesToProvideInfo;
	int baseNumObservations;
	bool showTargetMocap;
	vector<MJCFImporter*> tmjcfs;
	vector<pair<int, int>> tmocapBDs;
	bool useRelativePositionForInfo;
	bool killWhenFarEnough;
	int infoSkipFrame;
	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);

		LoadVec(f, startFrame);
		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);

		LoadVec(f, footFlag);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);

		SaveVec(f, startFrame);
		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);

		SaveVec(f, footFlag);

	}
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float& potential = potentials[a];
		float& potentialOld = potentialsOld[a];
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* joint_speeds = &jointSpeeds[a][0];
		int& jointsAtLimit = jointsAtLimits[a];

		//float& heading = headings[a];
		//float& upVec = upVecs[a];

		float electrCost = electricityCostScale * electricityCost;
		float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

		float alive = AliveBonus(state[0] + initialZ, p); //   # state[0] is body height above ground, body_rpy[1] is pitch
		dead = alive < 0.f;

		potentialOld = potential;
		potential = -walkTargetDist / (dt);
		if (potentialOld > 1e9)
		{
			potentialOld = potential;
		}

		float progress = potential - potentialOld;

		//-----------------------
		if (!useGeoMatching)
		{
			float targetVel = 0.8f;
			float marginVel = 0.1f;
			float progressRewardMag = 2.0f;
			if (fabs(progress - targetVel) < marginVel)
			{
				progress = progressRewardMag;
			}
			else
			{
				float error = fabs(progress - targetVel) - marginVel;
				//float errorRel = error / (targetVel - marginVel);
				progress = progressRewardMag*max(0.0f, 1.0f - error*error);
			}
		}
		else
		{
			int frameNum = (mFrames[a] + startFrame[a]) + firstFrame;
			float sumError = 0.0f;
			for (int i = 0; i < features[a].size(); i++)
			{
				Vec3 pTarget = TransformPoint(fullTrans[frameNum][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p);
				Transform cpose;

				NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
				Vec3 pCurrent = TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p));
				sumError += LengthSq(pCurrent - pTarget);
			}
			float err = sqrt(sumError / features[a].size());
			float maxE = 0.6f; // 1.0f, 0.1f
			progress = 2.0f*max(maxE - err, 0.0f) / maxE;

			//cout << a << " " << mFrames[a] << " " << err << " "<<progress<<endl;;
			if (killWhenFarEnough)
			{
				if (err >= maxE)
				{
					dead = true;
				}
			}
		}
		//------------------------
		float electricityCostCurrent = 0.0f;
		float sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			float vv = abs(action[i] * joint_speeds[i]);
			if (!isfinite(vv))
			{
				printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv, action[i], joint_speeds[i]);
			}

			if (!isfinite(action[i]))
			{
				printf("action at %d is infinite\n", i);
			}

			if (!isfinite(joint_speeds[i]))
			{
				printf("joint_speeds at %d is infinite\n", i);
			}

			sum += vv;
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl*joint_speed is infinite!\n");
		}

		//electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
		electricityCostCurrent += electrCost * sum / (float)ctrls[a].size();

		sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			sum += action[i] * action[i];
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
		}

		//electricity_costCurrent += stall_torque_cost * float(np.square(a).mean())
		electricityCostCurrent += stallTorqCost * sum / (float)ctrls[a].size();

		float jointsAtLimitCostCurrent = jointsAtLimitCost * jointsAtLimit;

		float feetCollisionCostCurrent = 0.0f;
		if (numCollideOther[a] > 0)
		{
			feetCollisionCostCurrent += footCollisionCost;
		}

		//cout << "heading = " << heading.x << " " << heading.y << " " << heading.z << endl;
		//float heading_rew = 0.2f*((heading.x > 0.5f) ? 1.0f: heading.x*2.0f); // MJCF3
		//float heading_rew = heading.x; // MJCF2
		//float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4

		//cout << mind << endl;

		// Heading was included, but actually probabably shouldn't, not sure about upvec to make it up right, but don't think so
		float rewards[5] =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent
		};

		//printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

		rew = 0.f;
		for (int i = 0; i < 5; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}
	}

	RigidFullHumanoidTrackTargetAngles()
	{
		showTargetMocap = true;
		useGeoMatching = true;
		useRelativePositionForInfo = true;
		killWhenFarEnough = true;
		infoSkipFrame = 1;
		if (!useGeoMatching)
		{
			firstFrame = 20;
			lastFrame = 110;
		}
		else
		{
			firstFrame = 30;
			lastFrame = 1100;
		}
		doFlagRun = false;
		loadPath = "../../data/humanoid_mod.xml";

		mNumAgents = 500;
		baseNumObservations = 52;
		mNumObservations = baseNumObservations;
		mNumActions = 21;
		mMaxEpisodeLength = 1000;
		//geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
		geo_joint = { "torso1","right_thigh", "right_foot","left_thigh","left_foot" };
		numFramesToProvideInfo = 3;


		g_numSubsteps = 4;
		g_params.numIterations = 20;
		//g_params.numIterations = 32; GAN4

		g_sceneLower = Vec3(-150.f, -250.f, -100.f);
		g_sceneUpper = Vec3(250.f, 150.f, 100.f);

		g_pause = false;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;

		numPerRow = 20;
		spacing = 10.f;

		numFeet = 2;

		//power = 0.41f; // Default
		powerScale = 0.25f; // Reduced power
		initialZ = 0.9f;

		electricityCostScale = 1.f;

		angleResetNoise = 0.f;
		angleVelResetNoise = 0.0f;
		velResetNoise = 0.0f;

		pushFrequency = 250;	// How much steps in average per 1 kick
		forceMag = 0.f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		if (numFramesToProvideInfo > 0)
		{
			mNumObservations += 3 * (numFramesToProvideInfo + 1)*geo_joint.size() + numFramesToProvideInfo; // Self, target current and futures
		}

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		LoadEnv();
		startFrame.resize(mNumAgents, 0);
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

			ppo_params.useGAN = false;
			ppo_params.resume = 4666; //3817;// 6727;
			ppo_params.timesteps_per_batch = 200;
			ppo_params.hid_size = 256;
			ppo_params.num_hid_layers = 2;
			ppo_params.optim_batchsize_per_agent = 64;
			ppo_params.optim_schedule = "adaptive";
			ppo_params.desired_kl = 0.01f;

			//string folder = "flexTrackTargetAngleModRetry2"; This is great!
			//string folder = "flexTrackTargetAngleGeoMatching_numFramesToProvideInfo_1";
			//string folder = "flexTrackTargetAngleTargetVel_BugFix_numFramesToProvideInfo_0";
			ppo_params.relativeLogDir = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far_0.6";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far_0.6_info_skip_20";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far";

			//string folder = "dummy";
			//string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";

			ppo_params.TryParseJson(g_sceneJson);


			fullFileName = "../../data/bvh/LocomotionFlat02_000_full.state";
			FILE* f = fopen(fullFileName.c_str(), "rb");
			int numFrames;
			fread(&numFrames, 1, sizeof(int), f);
			fullTrans.resize(numFrames);
			fullVels.resize(numFrames);
			fullAVels.resize(numFrames);
			cout << "Read " << numFrames << " frames of full data" << endl;

			int numTrans = fullTrans[0].size();
			fread(&numTrans, 1, sizeof(int), f);

			for (int i = 0; i < numFrames; i++)
			{
				fullTrans[i].resize(numTrans);
				fullVels[i].resize(numTrans);
				fullAVels[i].resize(numTrans);
				fread(&fullTrans[i][0], sizeof(Transform), fullTrans[i].size(), f);
				fread(&fullVels[i][0], sizeof(Vec3), fullVels[i].size(), f);
				fread(&fullAVels[i][0], sizeof(Vec3), fullAVels[i].size(), f);
			}
			fclose(f);

			string jointAnglesFileName = "../../data/bvh/LocomotionFlat02_000_joint_angles.state";

			jointAngles.clear();

			f = fopen(jointAnglesFileName.c_str(), "rb");
			fread(&numFrames, 1, sizeof(int), f);
			jointAngles.resize(numFrames);
			int numAngles;
			fread(&numAngles, 1, sizeof(int), f);
			for (int i = 0; i < numFrames; i++)
			{
				jointAngles[i].resize(numAngles);
				fread(&jointAngles[i][0], sizeof(float), numAngles, f);
			}

			fclose(f);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}

		for (int a = 0; a < mNumAgents; a++)
		{
			features.push_back(vector<pair<int, Transform>>());
			for (int i = 0; i < geo_joint.size(); i++)
			{
				auto p = mjcfs[a]->geoBodyPose[geo_joint[i]];
				features[a].push_back(p);
			}
		}

		const int greenMaterial = AddRenderMaterial(Vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, false);
		if (showTargetMocap)
		{
			tmjcfs.resize(mNumAgents);
			tmocapBDs.resize(mNumAgents);
			int tmpBody = g_buffers->rigidBodies.size();
			vector<pair<int, NvFlexRigidJointAxis>> ctrl;
			vector<float> mpower;
			for (int i = 0; i < mNumAgents; i++)
			{
				int sb = g_buffers->rigidShapes.size();
				Transform oo = agentOffset[i];
				oo.p.x += 2.0f;
				tmocapBDs[i].first = g_buffers->rigidBodies.size();
				tmjcfs[i] = new MJCFImporter(loadPath.c_str());
				tmjcfs[i]->AddPhysicsEntities(oo, ctrl, mpower, false);
				int eb = g_buffers->rigidShapes.size();
				for (int s = sb; s < eb; s++)
				{
					g_buffers->rigidShapes[s].user = UnionCast<void*>(greenMaterial);
					g_buffers->rigidShapes[s].filter = 1; // Ignore collsion, sort of
				}
				tmocapBDs[i].second = g_buffers->rigidBodies.size();
			}
			footFlag.resize(g_buffers->rigidBodies.size());
			for (int i = tmpBody; i < (int)g_buffers->rigidBodies.size(); i++)
			{
				footFlag[i] = -1;
			}
		}
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
				HandleCommunication();
				ClearContactInfo();
			}
			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!
			g_buffers->rigidJoints.unmap();
			NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size()); // Need to set bodies here too!
		}
	}

	virtual void Simulate()
	{
		// Random push to torso during training
		//int push_ai = Rand(0, pushFrequency - 1);

		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			int frameNum = 0;
			if (useGeoMatching)
			{
				frameNum = (mFrames[ai] + startFrame[ai]) + firstFrame;
			}
			else
			{
				frameNum = (mFrames[ai] + startFrame[ai]) % (lastFrame - firstFrame) + firstFrame;
			}

			if (showTargetMocap)
			{
				Transform tran = agentOffset[ai];
				tran.p.x += 2.0f;
				for (int i = tmocapBDs[ai].first; i < (int)tmocapBDs[ai].second; i++)
				{
					int bi = i - tmocapBDs[ai].first;
					Transform tt = tran * fullTrans[frameNum][bi];
					NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
					(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(tran.q, fullVels[frameNum][bi]);
					(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(tran.q, fullAVels[frameNum][bi]);
				}
			}

			for (int i = 0; i < (int)ctrls[ai].size(); i++)
			{
				int qq = i;
				NvFlexRigidJoint& joint = g_buffers->rigidJoints[ctrls[ai][qq].first + 1]; // Active joint
				joint.compliance[ctrls[ai][qq].second] = 0.1f / motorPower[ai][i];
				joint.targets[ctrls[ai][qq].second] = jointAngles[frameNum][i];

				//if (i == 20) joint.targets[ctrls[ai][qq].second] *= -1.0f;
			}

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
			for (int i = 0; i < ctrls[ai].size(); i++)
			{
				float cc = Clamp(actions[i], -1.f, 1.f);

				NvFlexRigidJoint& j = initJoints[ctrls[ai][i].first];
				NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
				NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
				Transform& pose0 = *((Transform*)&j.pose0);
				Transform gpose;
				NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
				Transform tran = gpose*pose0;

				Vec3 axis;
				if (ctrls[ai][i].second == 0)
				{
					axis = GetBasisVector0(tran.q);
				}
				if (ctrls[ai][i].second == 1)
				{
					axis = GetBasisVector1(tran.q);
				}
				if (ctrls[ai][i].second == 2)
				{
					axis = GetBasisVector2(tran.q);
				}

				Vec3 torque = axis * motorPower[ai][i] * cc * powerScale;
				a0.torque[0] += torque.x;
				a0.torque[1] += torque.y;
				a0.torque[2] += torque.z;
				a1.torque[0] -= torque.x;
				a1.torque[1] -= torque.y;
				a1.torque[2] -= torque.z;
			}

			if ((mFrames[ai] % 20 == 0) && torso[ai] != -1)
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
					pushForce.y *= 0.2f;
				}
				//cout << "Push agent " << ai << " with "<< pushForce.x<<" "<< pushForce.y<<" "<< pushForce.z<<endl;
				g_buffers->rigidBodies[torso[ai]].force[0] += pushForce.x;
				g_buffers->rigidBodies[torso[ai]].force[1] += pushForce.y;
				g_buffers->rigidBodies[torso[ai]].force[2] += pushForce.z;
			}
		}

		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
		g_buffers->rigidJoints.unmap();
		NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		g_frame++;
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
		g_buffers->rigidBodies.map();
		g_buffers->rigidJoints.map();
	}

	virtual void ResetAgent(int a)
	{
		//mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		startFrame[a] = rand() % (lastFrame - firstFrame);
		int aa = startFrame[a] + firstFrame;
		for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
		{
			int bi = i - agentBodies[a].first;
			Transform tt = agentOffset[a] * fullTrans[aa][bi];
			NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
			(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
			(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
		}


		RLWalkerEnv::ResetAgent(a);
	}

	virtual void DoStats()
	{
		if (showTargetMocap)
		{
			BeginLines(true);
			for (int i = 0; i < mNumAgents; i++)
			{
				DrawLine(g_buffers->rigidBodies[tmocapBDs[i].first].com, g_buffers->rigidBodies[agentBodies[i].first].com, Vec4(0.0f, 1.0f, 1.0f));
			}
			EndLines();

		}
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
					feetContact[ff / 2]++;
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
		// Original
		//return +2 if z > 0.78 else - 1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

		// Viktor: modified original one to enforce standing and walking high, not on knees
		// Also due to reduced electric cost bonus for living has been decreased
		if (z > 1.0)
		{
			return 1.5f;
		}
		else
		{
			return -1.f;
		}
	}

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		RLWalkerEnv<Transform, Vec3, Quat, Matrix33>::ExtractState(a, state, p, walkTargetDist, jointSpeeds, numJointsAtLimit, heading, upVec);
		int ct = baseNumObservations;
		if (numFramesToProvideInfo > 0)
		{
			Vec3* ttt = (Vec3*)&state[ct];
			for (int i = 0; i < features[a].size(); i++)
			{
				Transform cpose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
				Vec3 pCurrent = TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p));
				state[ct++] = pCurrent.x;
				state[ct++] = pCurrent.y;
				state[ct++] = pCurrent.z;
			}
			int frameNum = (mFrames[a] + startFrame[a]) + firstFrame;

			for (int q = 0; q < numFramesToProvideInfo; q++)
			{
				float sumError = 0.0f;
				for (int i = 0; i < features[a].size(); i++)
				{
					Vec3 pCurrent = ttt[i];
					Vec3 pTarget = TransformPoint(fullTrans[frameNum + q*infoSkipFrame][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p);
					sumError += LengthSq(pCurrent - pTarget);
					if (useRelativePositionForInfo)
					{
						state[ct++] = pTarget.x - pCurrent.x;
						state[ct++] = pTarget.y - pCurrent.x;
						state[ct++] = pTarget.z - pCurrent.x;
					}
					else
					{
						state[ct++] = pTarget.x;
						state[ct++] = pTarget.y;
						state[ct++] = pTarget.z;
					}
				}
				float err = sqrt(sumError / features[a].size());
				state[ct++] = err;
			}
		}
	}
};


class RigidFullHumanoidTrackTargetAnglesModified : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	int firstFrame;
	int lastFrame;
	vector<int> startFrame;
	vector<int> rightFoot;
	vector<int> leftFoot;

	vector<int> footFlag;

	vector<vector<Transform>> fullTrans;
	vector<vector<Vec3>> fullVels;
	vector<vector<Vec3>> fullAVels;
	vector<vector<float>> jointAngles;
	string fullFileName;

	vector < vector<pair<int, Transform>>> features;
	vector<string> geo_joint;
	float ax, ay, az;
	float isdx, isdy, isdz;
	int numFramesToProvideInfo;
	int baseNumObservations;
	bool showTargetMocap;
	vector<MJCFImporter*> tmjcfs;
	vector<pair<int, int>> tmocapBDs;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);

		LoadVec(f, startFrame);
		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);

		LoadVec(f, footFlag);
	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);

		SaveVec(f, startFrame);
		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);

		SaveVec(f, footFlag);

	}
	// Reward:
	//   Global pose error
	//	 Quat of torso error
	//   Position of torso error
	//   Velocity of torso error
	//   Angular velocity of torso error
	//	 Relative position with respect to torso of target and relative position with respect to torso of current
	//
	// State:
	// Quat of torso
	// Velocity of torso
	// Angular velocity of torso
	// Relative pos of geo_pos in torso's coordinate frame
	// Future frames:
	//				 Relative Pos of target torso in current torso's coordinate frame
	//				 Relative Quat of target torso in current torso's coordinate frame
	//				 Relative Velocity of target torso in current torso's coordinate frame
	//				 Relative Angular target velocity of torso in current torso's coordinate frame
	//               Relative target pos of geo_pos in current torso's coordinate frame
	// Look at 0, 1, 4, 16, 64 frames in future
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float& potential = potentials[a];
		float& potentialOld = potentialsOld[a];
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* joint_speeds = &jointSpeeds[a][0];
		int& jointsAtLimit = jointsAtLimits[a];

		//float& heading = headings[a];
		//float& upVec = upVecs[a];

		float electrCost = electricityCostScale * electricityCost;
		float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

		float alive = AliveBonus(state[0] + initialZ, p); //   # state[0] is body height above ground, body_rpy[1] is pitch
		dead = alive < 0.f;

		potentialOld = potential;
		potential = -walkTargetDist / (dt);
		if (potentialOld > 1e9)
		{
			potentialOld = potential;
		}

		float progress = potential - potentialOld;

		//-----------------------
		int frameNum = (mFrames[a] + startFrame[a]) + firstFrame;

		// Global error
		Transform targetTorso = fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		Transform cpose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
		Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;
		Vec3 targetVel = fullVels[frameNum][features[a][0].first - mjcfs[a]->firstBody];
		Vec3 currentVel = TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].linearVel);
		Vec3 targetAVel = fullAVels[frameNum][features[a][0].first - mjcfs[a]->firstBody];
		Vec3 currentAVel = TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].angularVel);

		float posError = Length(targetTorso.p - currentTorso.p);
		float velError = Length(targetVel - currentVel);
		float avelError = Length(targetAVel - currentAVel);
		Quat qE = targetTorso.q * Inverse(currentTorso.q);
		float sinHalfTheta = Length(qE.GetAxis());
		if (sinHalfTheta > 1.0f)
		{
			sinHalfTheta = 1.0f;
		}
		if (sinHalfTheta < -1.0f)
		{
			sinHalfTheta = -1.0f;
		}

		float quatError = asinf(sinHalfTheta)*2.0f;
		Transform itargetTorso = Inverse(targetTorso);
		Transform icurrentTorso = Inverse(currentTorso);

		// Local error
		float sumError = 0.0f;
		for (int i = 0; i < features[a].size(); i++)
		{
			Vec3 pTarget = TransformPoint(itargetTorso, TransformPoint(fullTrans[frameNum][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p));
			Transform cpose;

			NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
			Vec3 pCurrent = TransformPoint(icurrentTorso, TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p)));

			sumError += LengthSq(pCurrent - pTarget);
		}
		float localError = sqrt(sumError / features[a].size());
		if (a == 0)
		{
			//cout << "Agent " << a << " frame="<<mFrames[a]<<" posE=" << posError << " velE=" << velError << " avelE=" << avelError << " qE=" << quatError << " lE=" << localError << endl;
		}

		// For all but 07
		//if (posError > 0.6f) dead = true; // Position error > 0.6
		//if (quatError > kPi*0.5f) dead = true; // Angular error > 90 deg
		if (posError > 3.0f)
		{
			dead = true;    // Position error > 3.0, for 07 as it's fast
		}
		if (quatError > 2 * kPi)
		{
			dead = true;    // Angular error > 360 deg
		}

		float posWeight = 1.0f;
		float quatWeight = 1.0f;
		float velWeight = 0.2f;
		float avelWeight = 0.2f;
		float localWeight = 1.0f;

		progress = posWeight * max(0.6f - posError, 0.0f) / 0.6f + quatWeight * max(kPi*0.5f - quatError, 0.0f) / (kPi*0.5f)
				   + velWeight * max(2.0f - velError, 0.0f) / 2.0f + avelWeight * max(2.0f - avelError, 0.0f) / (2.0f)
				   + localWeight*max(0.1f - localError, 0.0f) / 0.1f;
		if (posError > 0.6f)
		{
			progress -= (posError - 0.6f)*posWeight*0.2f;
		}
		if (quatError > kPi*0.5f)
		{
			progress -= (quatError - kPi*0.5f)*quatWeight*0.2f;
		}

		//------------------------
		float electricityCostCurrent = 0.0f;
		float sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			float vv = abs(action[i] * joint_speeds[i]);
			if (!isfinite(vv))
			{
				printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv, action[i], joint_speeds[i]);
			}

			if (!isfinite(action[i]))
			{
				printf("action at %d is infinite\n", i);
			}

			if (!isfinite(joint_speeds[i]))
			{
				printf("joint_speeds at %d is infinite\n", i);
			}

			sum += vv;
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl*joint_speed is infinite!\n");
		}

		//electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
		electricityCostCurrent += electrCost * sum / (float)ctrls[a].size();

		sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			sum += action[i] * action[i];
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
		}

		//electricity_costCurrent += stall_torque_cost * float(np.square(a).mean())
		electricityCostCurrent += stallTorqCost * sum / (float)ctrls[a].size();

		float jointsAtLimitCostCurrent = jointsAtLimitCost * jointsAtLimit;

		float feetCollisionCostCurrent = 0.0f;
		if (numCollideOther[a] > 0)
		{
			feetCollisionCostCurrent += footCollisionCost;
		}

		//cout << "heading = " << heading.x << " " << heading.y << " " << heading.z << endl;
		//float heading_rew = 0.2f*((heading.x > 0.5f) ? 1.0f: heading.x*2.0f); // MJCF3
		//float heading_rew = heading.x; // MJCF2
		//		float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4

		//cout << mind << endl;
		// Heading was included, but actually probabably shouldn't, not sure about upvec to make it up right, but don't think so
		float rewards[5] =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent,
		};

		//printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

		rew = 0.f;
		for (int i = 0; i < 5; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}
	}

	RigidFullHumanoidTrackTargetAnglesModified()
	{
		showTargetMocap = false;
		firstFrame = 30;
		//lastFrame = 1100;
		//lastFrame = 3000; //02
		//lastFrame = 9000; //12
		//lastFrame = 7000; //11

		//lastFrame = 6400; //08
		lastFrame = 4900; //07

		doFlagRun = false;
		loadPath = "../../data/humanoid_mod.xml";

		mNumAgents = 500;
		baseNumObservations = 52;
		mNumObservations = baseNumObservations;
		mNumActions = 21;
		mMaxEpisodeLength = 1000;
		//geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
		geo_joint = { "torso1","right_thigh", "right_foot","left_thigh","left_foot" };
		numFramesToProvideInfo = 4;


		g_numSubsteps = 4;
		g_params.numIterations = 20;
		//g_params.numIterations = 32; GAN4

		g_sceneLower = Vec3(-150.f, -250.f, -100.f);
		g_sceneUpper = Vec3(250.f, 150.f, 100.f);

		g_pause = false;
		mDoLearning = g_doLearning;
		numRenderSteps = 60;


		numPerRow = 20;
		spacing = 30.f;

		numFeet = 2;

		//power = 0.41f; // Default
		powerScale = 0.25f; // Reduced power
		initialZ = 0.9f;

		electricityCostScale = 1.f;

		angleResetNoise = 0.f;
		angleVelResetNoise = 0.0f;
		velResetNoise = 0.0f;

		pushFrequency = 50;	// How much steps in average per 1 kick
		forceMag = 0.f;
	}

	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		if (numFramesToProvideInfo > 0)
		{
			mNumObservations += 10 + 3 * geo_joint.size() + numFramesToProvideInfo * (13 + 3 * geo_joint.size()); // Self, target current and futures
		}

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		LoadEnv();
		startFrame.resize(mNumAgents, 0);
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

			ppo_params.useGAN = false;
			ppo_params.resume = 3210;// 6727;
			ppo_params.timesteps_per_batch = 200;
			ppo_params.num_timesteps = 2000000000;
			ppo_params.hid_size = 256;
			ppo_params.num_hid_layers = 2;
			ppo_params.optim_batchsize_per_agent = 64;
			ppo_params.optim_schedule = "adaptive";
			ppo_params.desired_kl = 0.01f;

			//string folder = "flexTrackTargetAngleModRetry2"; This is great!
			//string folder = "flexTrackTargetAngleGeoMatching_numFramesToProvideInfo_1";
			//string folder = "flexTrackTargetAngleTargetVel_BugFix_numFramesToProvideInfo_0";
			//string folder = "flexTrackTargetAnglesModified_bigDB";
			//string folder = "flexTrackTargetAnglesModified_bigDB_12";
			ppo_params.relativeLogDir = "flexTrackTargetAnglesModified_bigDB_07";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far_0.6_info_skip_20";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far";

			//string folder = "dummy";
			//string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";

			ppo_params.TryParseJson(g_sceneJson);

			//fullFileName = "../../data/bvh/LocomotionFlat02_000_full.state";
			//fullFileName = "../../data/bvh/LocomotionFlat12_000_full.state";
			fullFileName = "../../data/bvh/LocomotionFlat07_000_full.state";
			FILE* f = fopen(fullFileName.c_str(), "rb");
			int numFrames;
			fread(&numFrames, 1, sizeof(int), f);
			fullTrans.resize(numFrames);
			fullVels.resize(numFrames);
			fullAVels.resize(numFrames);
			cout << "Read " << numFrames << " frames of full data" << endl;

			int numTrans = fullTrans[0].size();
			fread(&numTrans, 1, sizeof(int), f);

			for (int i = 0; i < numFrames; i++)
			{
				fullTrans[i].resize(numTrans);
				fullVels[i].resize(numTrans);
				fullAVels[i].resize(numTrans);
				fread(&fullTrans[i][0], sizeof(Transform), fullTrans[i].size(), f);
				fread(&fullVels[i][0], sizeof(Vec3), fullVels[i].size(), f);
				fread(&fullAVels[i][0], sizeof(Vec3), fullAVels[i].size(), f);
			}
			fclose(f);

			//string jointAnglesFileName = "../../data/bvh/LocomotionFlat02_000_joint_angles.state";
			//string jointAnglesFileName = "../../data/bvh/LocomotionFlat12_000_joint_angles.state";
			string jointAnglesFileName = "../../data/bvh/LocomotionFlat07_000_joint_angles.state";

			jointAngles.clear();

			f = fopen(jointAnglesFileName.c_str(), "rb");
			fread(&numFrames, 1, sizeof(int), f);
			jointAngles.resize(numFrames);
			int numAngles;
			fread(&numAngles, 1, sizeof(int), f);
			for (int i = 0; i < numFrames; i++)
			{
				jointAngles[i].resize(numAngles);
				fread(&jointAngles[i][0], sizeof(float), numAngles, f);
			}

			fclose(f);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}

		for (int a = 0; a < mNumAgents; a++)
		{
			features.push_back(vector<pair<int, Transform>>());
			for (int i = 0; i < geo_joint.size(); i++)
			{
				auto p = mjcfs[a]->geoBodyPose[geo_joint[i]];
				features[a].push_back(p);
			}
		}

		const int greenMaterial = AddRenderMaterial(Vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, false);
		if (showTargetMocap)
		{
			tmjcfs.resize(mNumAgents);
			tmocapBDs.resize(mNumAgents);
			int tmpBody = g_buffers->rigidBodies.size();

			vector<pair<int, NvFlexRigidJointAxis>> ctrl;
			vector<float> mpower;
			for (int i = 0; i < mNumAgents; i++)
			{
				int sb = g_buffers->rigidShapes.size();
				Transform oo = agentOffset[i];
				oo.p.x += 2.0f;
				tmocapBDs[i].first = g_buffers->rigidBodies.size();
				tmjcfs[i] = new MJCFImporter(loadPath.c_str());
				tmjcfs[i]->AddPhysicsEntities(oo, ctrl, mpower, false);
				int eb = g_buffers->rigidShapes.size();
				for (int s = sb; s < eb; s++)
				{
					g_buffers->rigidShapes[s].user = UnionCast<void*>(greenMaterial);
					g_buffers->rigidShapes[s].filter = 1; // Ignore collsion, sort of
				}
				tmocapBDs[i].second = g_buffers->rigidBodies.size();
			}

			footFlag.resize(g_buffers->rigidBodies.size());
			for (int i = tmpBody; i < (int)g_buffers->rigidBodies.size(); i++)
			{
				footFlag[i] = -1;
			}
		}
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
				HandleCommunication();
				ClearContactInfo();
			}
			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!
			g_buffers->rigidJoints.unmap();
			NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size()); // Need to set bodies here too!
		}
	}

	virtual void Simulate()
	{
		// Random push to torso during training
		//int push_ai = Rand(0, pushFrequency - 1);

		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			int frameNum = 0;
			frameNum = (mFrames[ai] + startFrame[ai]) + firstFrame;

			if (showTargetMocap)
			{
				Transform tran = agentOffset[ai];
				tran.p.x += 2.0f;
				for (int i = tmocapBDs[ai].first; i < (int)tmocapBDs[ai].second; i++)
				{
					int bi = i - tmocapBDs[ai].first;
					Transform tt = tran * fullTrans[frameNum][bi];
					NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
					(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(tran.q, fullVels[frameNum][bi]);
					(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(tran.q, fullAVels[frameNum][bi]);
				}
			}

			for (int i = 0; i < (int)ctrls[ai].size(); i++)
			{
				int qq = i;
				NvFlexRigidJoint& joint = g_buffers->rigidJoints[ctrls[ai][qq].first + 1]; // Active joint
				joint.compliance[ctrls[ai][qq].second] = 0.1f / motorPower[ai][i];
				joint.targets[ctrls[ai][qq].second] = jointAngles[frameNum][i];

				//if (i == 20) joint.targets[ctrls[ai][qq].second] *= -1.0f;
			}

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
			for (int i = 0; i < ctrls[ai].size(); i++)
			{
				float cc = Clamp(actions[i], -1.f, 1.f);

				NvFlexRigidJoint& j = initJoints[ctrls[ai][i].first];
				NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
				NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
				Transform& pose0 = *((Transform*)&j.pose0);
				Transform gpose;
				NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
				Transform tran = gpose*pose0;

				Vec3 axis;
				if (ctrls[ai][i].second == 0)
				{
					axis = GetBasisVector0(tran.q);
				}
				if (ctrls[ai][i].second == 1)
				{
					axis = GetBasisVector1(tran.q);
				}
				if (ctrls[ai][i].second == 2)
				{
					axis = GetBasisVector2(tran.q);
				}

				Vec3 torque = axis * motorPower[ai][i] * cc * powerScale;
				a0.torque[0] += torque.x;
				a0.torque[1] += torque.y;
				a0.torque[2] += torque.z;
				a1.torque[0] -= torque.x;
				a1.torque[1] -= torque.y;
				a1.torque[2] -= torque.z;
			}
			/*
			if (ai % pushFrequency == push_ai && torso[ai] != -1)
			{
			cout << "Push agent " << ai << endl;
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
			pushForce.y *= 0.2f;
			}
			g_buffers->rigidBodies[torso[ai]].force[0] += pushForce.x;
			g_buffers->rigidBodies[torso[ai]].force[1] += pushForce.y;
			g_buffers->rigidBodies[torso[ai]].force[2] += pushForce.z;
			}
			*/
		}

		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
		g_buffers->rigidJoints.unmap();
		NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		g_frame++;
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
		g_buffers->rigidBodies.map();
		g_buffers->rigidJoints.map();
	}

	virtual void ResetAgent(int a)
	{
		//mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		startFrame[a] = rand() % (lastFrame - firstFrame);
		int aa = startFrame[a] + firstFrame;
		for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
		{
			int bi = i - agentBodies[a].first;
			Transform tt = agentOffset[a] * fullTrans[aa][bi];
			NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
			(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
			(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
		}
		RLWalkerEnv::ResetAgent(a);
	}

	virtual void DoStats()
	{
		if (showTargetMocap)
		{
			BeginLines(true);
			for (int i = 0; i < mNumAgents; i++)
			{
				DrawLine(g_buffers->rigidBodies[tmocapBDs[i].first].com, g_buffers->rigidBodies[agentBodies[i].first].com, Vec4(0.0f, 1.0f, 1.0f));
			}
			EndLines();
		}
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
					feetContact[ff / 2]++;
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
		// Original
		//return +2 if z > 0.78 else - 1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

		// Viktor: modified original one to enforce standing and walking high, not on knees
		// Also due to reduced electric cost bonus for living has been decreased
		/*
		if (z > 1.0)
		{
		return 1.5f;
		}
		else
		{
		return -1.f;
		}*/
		return 1.5f;// Not die because of this
	}

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		RLWalkerEnv<Transform, Vec3, Quat, Matrix33>::ExtractState(a, state, p, walkTargetDist, jointSpeeds, numJointsAtLimit, heading, upVec);
		int ct = baseNumObservations;
		if (numFramesToProvideInfo > 0)
		{
			// State:
			// Quat of torso
			// Velocity of torso
			// Angular velocity of torso
			// Relative pos of geo_pos in torso's coordinate frame
			// Future frames:
			//				 Relative Pos of target torso in current torso's coordinate frame
			//				 Relative Quat of target torso in current torso's coordinate frame
			//				 Relative Velocity of target torso in current torso's coordinate frame
			//				 Relative Angular target velocity of torso in current torso's coordinate frame
			//               Relative target pos of geo_pos in current torso's coordinate frame
			// Look at 0, 1, 4, 16, 64 frames in future
			int frameNum = (mFrames[a] + startFrame[a]) + firstFrame;

			Transform cpose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
			Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;
			Transform icurrentTorso = Inverse(currentTorso);

			Vec3 currentVel = Rotate(icurrentTorso.q, TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].linearVel));
			Vec3 currentAVel = Rotate(icurrentTorso.q, TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].angularVel));

			state[ct++] = currentTorso.q.x;
			state[ct++] = currentTorso.q.y;
			state[ct++] = currentTorso.q.z;
			state[ct++] = currentTorso.q.w;

			state[ct++] = currentVel.x;
			state[ct++] = currentVel.y;
			state[ct++] = currentVel.z;

			state[ct++] = currentAVel.x;
			state[ct++] = currentAVel.y;
			state[ct++] = currentAVel.z;

			Vec3* ttt = (Vec3*)&state[ct];
			for (int i = 0; i < features[a].size(); i++)
			{
				Transform cpose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
				Vec3 pCurrent = TransformPoint(icurrentTorso, TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p)));
				state[ct++] = pCurrent.x;
				state[ct++] = pCurrent.y;
				state[ct++] = pCurrent.z;
			}

			for (int q = 0; q < numFramesToProvideInfo; q++)
			{
				if (q == 0)
				{
					frameNum = (mFrames[a] + startFrame[a]) + firstFrame;
				}
				else
				{
					frameNum = (mFrames[a] + startFrame[a]) + firstFrame + (1 << (2 * (q)));
				}

				Transform targetTorso = icurrentTorso*fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
				Vec3 targetVel = Rotate(icurrentTorso.q, fullVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]);
				Vec3 targetAVel = Rotate(icurrentTorso.q, fullAVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]);
				state[ct++] = targetTorso.p.x;
				state[ct++] = targetTorso.p.y;
				state[ct++] = targetTorso.p.z;

				state[ct++] = targetTorso.q.x;
				state[ct++] = targetTorso.q.y;
				state[ct++] = targetTorso.q.z;
				state[ct++] = targetTorso.q.w;

				state[ct++] = targetVel.x;
				state[ct++] = targetVel.y;
				state[ct++] = targetVel.z;

				state[ct++] = targetAVel.x;
				state[ct++] = targetAVel.y;
				state[ct++] = targetAVel.z;

				//float sumError = 0.0f;
				for (int i = 0; i < features[a].size(); i++)
				{
					Vec3 pCurrent = ttt[i];
					Vec3 pTarget = TransformPoint(icurrentTorso, TransformPoint(fullTrans[frameNum][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p));

					state[ct++] = pTarget.x - pCurrent.x;
					state[ct++] = pTarget.y - pCurrent.x;
					state[ct++] = pTarget.z - pCurrent.x;
				}
			}
		}
	}
};

class PushInfo
{
public:
	Vec3 pos;
	Vec3 force;
	int time;
};

class RigidFullHumanoidTrackTargetAnglesModifiedWithReducedControlAndDisturbance : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	int firstFrame;
	int lastFrame;
	vector<int> startFrame;
	vector<int> rightFoot;
	vector<int> leftFoot;

	vector<int> footFlag;

	vector<vector<Transform>> fullTrans;
	vector<vector<Vec3>> fullVels;
	vector<vector<Vec3>> fullAVels;
	vector<vector<float>> jointAngles;
	string fullFileName;

	vector < vector<pair<int, Transform>>> features;
	vector<string> geo_joint;
	float ax, ay, az;
	float isdx, isdy, isdz;
	int numFramesToProvideInfo;
	int baseNumObservations;
	bool showTargetMocap;
	vector<MJCFImporter*> tmjcfs;
	vector<pair<int, int>> tmocapBDs;
	vector<int> mFarCount;
	float maxFarItr; // More than this will die
	float farStartPos; // When to start consider as being far, PD will start to fall off
	float farStartQuat; // When to start consider as being far, PD will start to fall off
	float farEndPos; // PD will be 0 at this point and counter will start
	float farEndQuat; // PD will be 0 at this point and counter will start
	bool renderPush;
	bool limitForce; // Specify force limit
	bool pauseMocapWhenFar; // Implemented by decreasing startFrame
	bool useDifferentRewardWhenFell;
	bool halfRandomReset;
	vector<PushInfo> pushes;

	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);

		LoadVec(f, startFrame);
		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);

		LoadVec(f, footFlag);
		LoadVec(f, pushes);
		LoadVec(f, mFarCount);

	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);

		SaveVec(f, startFrame);
		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);

		SaveVec(f, footFlag);
		SaveVec(f, pushes);
		SaveVec(f, mFarCount);
	}

	// Reward:
	//   Global pose error
	//	 Quat of torso error
	//   Position of torso error
	//   Velocity of torso error
	//   Angular velocity of torso error
	//	 Relative position with respect to torso of target and relative position with respect to torso of current
	//
	// State:
	// Quat of torso
	// Velocity of torso
	// Angular velocity of torso
	// Relative pos of geo_pos in torso's coordinate frame
	// Future frames:
	//				 Relative Pos of target torso in current torso's coordinate frame
	//				 Relative Quat of target torso in current torso's coordinate frame
	//				 Relative Velocity of target torso in current torso's coordinate frame
	//				 Relative Angular target velocity of torso in current torso's coordinate frame
	//               Relative target pos of geo_pos in current torso's coordinate frame
	// Look at 0, 1, 4, 16, 64 frames in future
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float& potential = potentials[a];
		float& potentialOld = potentialsOld[a];
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* joint_speeds = &jointSpeeds[a][0];
		int& jointsAtLimit = jointsAtLimits[a];

		//float& heading = headings[a];
		//float& upVec = upVecs[a];

		float electrCost = electricityCostScale * electricityCost;
		float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

		float alive = AliveBonus(state[0] + initialZ, p); //   # state[0] is body height above ground, body_rpy[1] is pitch
		dead = alive < 0.f;

		potentialOld = potential;
		potential = -walkTargetDist / (dt);
		if (potentialOld > 1e9)
		{
			potentialOld = potential;
		}

		float progress = potential - potentialOld;
		float oprogress = progress;
		//-----------------------
		int frameNum = (mFrames[a] + startFrame[a]) + firstFrame;
		if (frameNum >= fullTrans.size())
		{
			frameNum = fullTrans.size() - 1;
		}

		// Global error
		Transform targetTorso = fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		Transform cpose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
		Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;
		Vec3 targetVel = fullVels[frameNum][features[a][0].first - mjcfs[a]->firstBody];
		Vec3 currentVel = TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].linearVel);
		Vec3 targetAVel = fullAVels[frameNum][features[a][0].first - mjcfs[a]->firstBody];
		Vec3 currentAVel = TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].angularVel);

		float posError = Length(targetTorso.p - currentTorso.p);
		float velError = Length(targetVel - currentVel);
		float avelError = Length(targetAVel - currentAVel);
		Quat qE = targetTorso.q * Inverse(currentTorso.q);
		float sinHalfTheta = Length(qE.GetAxis());
		if (sinHalfTheta > 1.0f)
		{
			sinHalfTheta = 1.0f;
		}
		if (sinHalfTheta < -1.0f)
		{
			sinHalfTheta = -1.0f;
		}

		float quatError = asinf(sinHalfTheta)*2.0f;
		Transform itargetTorso = Inverse(targetTorso);
		Transform icurrentTorso = Inverse(currentTorso);

		// Local error
		float sumError = 0.0f;
		for (int i = 0; i < features[a].size(); i++)
		{
			Vec3 pTarget = TransformPoint(itargetTorso, TransformPoint(fullTrans[frameNum][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p));
			Transform cpose;

			NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
			Vec3 pCurrent = TransformPoint(icurrentTorso, TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p)));

			sumError += LengthSq(pCurrent - pTarget);
		}

		float localError = sqrt(sumError / features[a].size());
		if (a == 0)
		{
			//cout << "Agent " << a << " frame="<<mFrames[a]<<" posE=" << posError << " velE=" << velError << " avelE=" << avelError << " qE=" << quatError << " lE=" << localError << endl;
		}

		// For all but 07
		//if (posError > 0.6f) dead = true; // Position error > 0.6
		//if (quatError > kPi*0.5f) dead = true; // Angular error > 90 deg
		//if (posError > .0f) dead = true; // Position error > 3.0, for 07 as it's fast
		//if (quatError > 2 * kPi) dead = true; // Angular error > 360 deg
		bool fell = false;
		if ((posError > farEndPos) || (quatError > farEndQuat))
		{
			fell = true;
			mFarCount[a]++;
			if (mFarCount[a] > maxFarItr)
			{
				dead = true;
			}
			if (pauseMocapWhenFar)
			{
				startFrame[a]--;
			}
		}
		else
		{
			mFarCount[a]--;
			if (mFarCount[a] < 0)
			{
				mFarCount[a] = 0;
			}
		}

		float posWeight = 1.0f;
		float quatWeight = 1.0f;
		float velWeight = 0.2f;
		float avelWeight = 0.2f;
		float localWeight = 1.0f;
		progress = posWeight * max(farStartPos - posError, 0.0f) / farStartPos + quatWeight * max(farStartQuat - quatError, 0.0f) / farStartQuat
				   + velWeight * max(2.0f - velError, 0.0f) / 2.0f + avelWeight * max(2.0f - avelError, 0.0f) / (2.0f)
				   + localWeight * max(0.1f - localError, 0.0f) / 0.1f;
		//------------------------
		//------------------------
		if (!useDifferentRewardWhenFell)
		{
			if (fell)
			{
				// If fall down, penalize height more severely
				progress -= fabs(currentTorso.p.z - targetTorso.p.z)*posWeight;
			}
			if (posError > farStartQuat)
			{
				progress -= (posError - farStartQuat)*posWeight*0.2f;
			}
			if (quatError > farStartQuat)
			{
				progress -= (quatError - farStartQuat)*quatWeight*0.2f;
			}
		}
		else
		{
			if (fell)
			{
				// If fell, use a different reward to first try to go to the target pose

				float zDif = 1.0f - max(targetTorso.p.z - currentTorso.p.z, 0.0f);
				float zWeight = 3.0f;
				float tmp = posWeight*(farEndPos - posError) / farEndPos + quatWeight*(farEndQuat - quatError) / farEndQuat + zWeight*zDif;
				progress = oprogress + tmp; // Use oprogress
				if (!isfinite(progress))
				{
					cout << "Fell = " << fell << " oprogress = " << oprogress << " tmp = " << tmp << endl;
				}
			}
		}
		if (!isfinite(progress))
		{
			cout << "Fell = " << fell << "pE = " << posError << " qE = " << quatError << endl;
		}
		//------------------------
		float electricityCostCurrent = 0.0f;
		float sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			float vv = abs(action[i] * joint_speeds[i]);
			if (!isfinite(vv))
			{
				printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv, action[i], joint_speeds[i]);
			}

			if (!isfinite(action[i]))
			{
				printf("action at %d is infinite\n", i);
			}

			if (!isfinite(joint_speeds[i]))
			{
				printf("joint_speeds at %d is infinite\n", i);
			}

			sum += vv;
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl*joint_speed is infinite!\n");
		}

		//electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
		electricityCostCurrent += electrCost * sum / (float)ctrls[a].size();

		sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			sum += action[i] * action[i];
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
		}

		//electricity_costCurrent += stall_torque_cost * float(np.square(a).mean())
		electricityCostCurrent += stallTorqCost * sum / (float)ctrls[a].size();

		float jointsAtLimitCostCurrent = jointsAtLimitCost * jointsAtLimit;

		float feetCollisionCostCurrent = 0.0f;
		if (numCollideOther[a] > 0)
		{
			feetCollisionCostCurrent += footCollisionCost;
		}

		//cout << "heading = " << heading.x << " " << heading.y << " " << heading.z << endl;
		//float heading_rew = 0.2f*((heading.x > 0.5f) ? 1.0f: heading.x*2.0f); // MJCF3
		//float heading_rew = heading.x; // MJCF2
		//		float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4

		//cout << mind << endl;
		// Heading was included, but actually probabably shouldn't, not sure about upvec to make it up right, but don't think so
		float rewards[5] =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent,

		};

		//printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

		rew = 0.f;
		for (int i = 0; i < 5; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}
	}

	RigidFullHumanoidTrackTargetAnglesModifiedWithReducedControlAndDisturbance()
	{
		limitForce = true;
		pauseMocapWhenFar = true;
		useDifferentRewardWhenFell = false;
		halfRandomReset = false;
		farStartPos = 0.6f; // When to start consider as being far, PD will start to fall off
		farStartQuat = kPi*0.5f; // When to start consider as being far, PD will start to fall off
		//farEndPos = 2.0f; // PD will be 0 at this point and counter will start, was 2.0
		farEndPos = 1.0f; // PD will be 0 at this point and counter will start, was 2.0
		farEndQuat = kPi*1.0f; // PD will be 0 at this point and counter will start

		showTargetMocap = true;
		renderPush = true;
		maxFarItr = 180.0f;
		//firstFrame = 30;
		firstFrame = 10;
		lastFrame = 38; // stand
		//lastFrame = 1100; // 02 easy
		//lastFrame = 3000; //02
		//lastFrame = 9000; //12
		//lastFrame = 7000; //11

		//lastFrame = 6400; //08
		//lastFrame = 4900; //07

		doFlagRun = false;
		loadPath = "../../data/humanoid_mod.xml";

		mNumAgents = 500;
		baseNumObservations = 52;
		mNumObservations = baseNumObservations;
		mNumActions = 21;
		//mMaxEpisodeLength = 1000;
		mMaxEpisodeLength = 300;

		//geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
		geo_joint = { "torso1","right_thigh", "right_foot","left_thigh","left_foot" };
		numFramesToProvideInfo = 4;


		g_numSubsteps = 4;
		g_params.numIterations = 20;
		//g_params.numIterations = 32; GAN4

		g_sceneLower = Vec3(-150.f, -250.f, -100.f);
		g_sceneUpper = Vec3(250.f, 150.f, 100.f);

		g_pause = true;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;

		numPerRow = 20;
		spacing = 30.f;

		numFeet = 2;

		//power = 0.41f; // Default
		//powerScale = 0.25f; // Reduced power
		//powerScale = 0.5f; // More power
		//powerScale = 0.41f; // More power
		powerScale = 0.41f; // More power
		//powerScale = 1.0f; // More power
		initialZ = 0.9f;

		//electricityCostScale = 1.f;
		electricityCostScale = 1.8f;

		angleResetNoise = 0.f;
		angleVelResetNoise = 0.0f;
		velResetNoise = 0.0f;

		//pushFrequency = 100;	// 200 How much steps in average per 1 kick // A bit too frequent
		pushFrequency = 200;	// 200 How much steps in average per 1 kick
		forceMag = 0.f; // 10000.0f
		//forceMag = 4000.f; // 10000.0f
		//forceMag = 10000.f; // 10000.0f Too much, can't learn anything useful fast...

		startShape.resize(mNumAgents, 0);
		endShape.resize(mNumAgents, 0);
		startBody.resize(mNumAgents, 0);
		endBody.resize(mNumAgents, 0);

	}
	void PrepareScene() override
	{
		ParseJsonParams(g_sceneJson);

		if (numFramesToProvideInfo > 0)
		{
			mNumObservations += 10 + 3 * geo_joint.size() + numFramesToProvideInfo * (13 + 3 * geo_joint.size()) + 1 + 1; // Self, target current and futures, far count
		}

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		LoadEnv();

		startFrame.resize(mNumAgents, 0);
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
		mFarCount.clear();
		mFarCount.resize(mNumAgents, 0);
		initRigidShapes.resize(g_buffers->rigidShapes.size());
		for (size_t i = 0; i < initRigidShapes.size(); i++)
		{
			initRigidShapes[i] = g_buffers->rigidShapes[i];
		}
		srand(211831);
		if (mDoLearning)
		{
			PPOLearningParams ppo_params;

			ppo_params.useGAN = false;
			ppo_params.resume = 10505;// 6727;
			ppo_params.timesteps_per_batch = 200000;
			ppo_params.num_timesteps = 2000000001;
			ppo_params.hid_size = 256;
			ppo_params.num_hid_layers = 2;
			ppo_params.optim_batchsize_per_agent = 64;
			ppo_params.optim_schedule = "adaptive";
			ppo_params.desired_kl = 0.01f; // 0.01f orig

			//string folder = "flexTrackTargetAngleModRetry2"; This is great!
			//string folder = "flexTrackTargetAngleGeoMatching_numFramesToProvideInfo_1";
			//string folder = "flexTrackTargetAngleTargetVel_BugFix_numFramesToProvideInfo_0";
			//string folder = "flexTrackTargetAnglesModified_bigDB";
			//string folder = "flexTrackTargetAnglesModified_bigDB_12";
			//string folder = "flexTrackTargetAnglesModified_bigDB_07";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far_0.6_info_skip_20";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_11_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far_forcemag_10000";
			//string folder = "track_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far_use_different_reward_when_fell";
			//string folder = "qqqq";
			//string folder = "track_02_far_end_1.0_pause_mocap_when_far_use_different_reward_when_fell_half_random_pose_stand";
			//string folder = "track_02_far_end_1.0_pause_mocap_when_far_use_different_reward_when_fell_half_random_pose";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_pow_1.0";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far_use_different_reward_when_fell";
			ppo_params.relativeLogDir = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02";
			//string folder = "dummy";
			//string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";

			ppo_params.TryParseJson(g_sceneJson);

			//fullFileName = "../../data/bvh/LocomotionFlat02_000_full.state";
			fullFileName = "../../data/bvh/140_09_full.state";

			//fullFileName = "../../data/bvh/LocomotionFlat12_000_full.state";
			//fullFileName = "../../data/bvh/LocomotionFlat11_000_full.state";
			//fullFileName = "../../data/bvh/LocomotionFlat07_000_full.state";
			FILE* f = fopen(fullFileName.c_str(), "rb");
			int numFrames;
			fread(&numFrames, 1, sizeof(int), f);
			fullTrans.resize(numFrames);
			fullVels.resize(numFrames);
			fullAVels.resize(numFrames);
			cout << "Read " << numFrames << " frames of full data" << endl;

			int numTrans = fullTrans[0].size();
			fread(&numTrans, 1, sizeof(int), f);

			for (int i = 0; i < numFrames; i++)
			{
				fullTrans[i].resize(numTrans);
				fullVels[i].resize(numTrans);
				fullAVels[i].resize(numTrans);
				fread(&fullTrans[i][0], sizeof(Transform), fullTrans[i].size(), f);
				fread(&fullVels[i][0], sizeof(Vec3), fullVels[i].size(), f);
				fread(&fullAVels[i][0], sizeof(Vec3), fullAVels[i].size(), f);
			}
			fclose(f);
			//string jointAnglesFileName = "../../data/bvh/LocomotionFlat11_000_joint_angles.state";
			//string jointAnglesFileName = "../../data/bvh/LocomotionFlat02_000_joint_angles.state";
			string jointAnglesFileName = "../../data/bvh/140_09_joint_angles.state";
			//string jointAnglesFileName = "../../data/bvh/LocomotionFlat12_000_joint_angles.state";
			//string jointAnglesFileName = "../../data/bvh/LocomotionFlat07_000_joint_angles.state";

			jointAngles.clear();

			f = fopen(jointAnglesFileName.c_str(), "rb");
			fread(&numFrames, 1, sizeof(int), f);
			jointAngles.resize(numFrames);
			int numAngles;
			fread(&numAngles, 1, sizeof(int), f);
			for (int i = 0; i < numFrames; i++)
			{
				jointAngles[i].resize(numAngles);
				fread(&jointAngles[i][0], sizeof(float), numAngles, f);
			}

			fclose(f);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}

		for (int a = 0; a < mNumAgents; a++)
		{
			features.push_back(vector<pair<int, Transform>>());
			for (int i = 0; i < geo_joint.size(); i++)
			{
				auto p = mjcfs[a]->geoBodyPose[geo_joint[i]];
				features[a].push_back(p);
			}
		}
		const int greenMaterial = AddRenderMaterial(Vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, false);
		if (showTargetMocap)
		{
			tmjcfs.resize(mNumAgents);
			tmocapBDs.resize(mNumAgents);
			int tmpBody = g_buffers->rigidBodies.size();

			vector<pair<int, NvFlexRigidJointAxis>> ctrl;
			vector<float> mpower;
			for (int i = 0; i < mNumAgents; i++)
			{
				int sb = g_buffers->rigidShapes.size();
				Transform oo = agentOffset[i];
				oo.p.x += 2.0f;
				tmocapBDs[i].first = g_buffers->rigidBodies.size();
				tmjcfs[i] = new MJCFImporter(loadPath.c_str());
				tmjcfs[i]->AddPhysicsEntities(oo, ctrl, mpower, false);
				int eb = g_buffers->rigidShapes.size();
				for (int s = sb; s < eb; s++)
				{
					g_buffers->rigidShapes[s].user = UnionCast<void*>(greenMaterial);
					g_buffers->rigidShapes[s].filter = 1; // Ignore collsion, sort of
				}
				tmocapBDs[i].second = g_buffers->rigidBodies.size();
			}

			footFlag.resize(g_buffers->rigidBodies.size());
			for (int i = tmpBody; i < (int)g_buffers->rigidBodies.size(); i++)
			{
				footFlag[i] = -1;
			}
		}
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
				HandleCommunication();
				ClearContactInfo();
			}

			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!
			g_buffers->rigidJoints.unmap();
			NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size()); // Need to set bodies here too!
		}
	}

	virtual void Simulate()
	{
		// Random push to torso during training
		int push_ai = Rand(0, pushFrequency - 1);

		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			int frameNum = 0;
			frameNum = (mFrames[ai] + startFrame[ai]) + firstFrame;
			if (frameNum >= fullTrans.size())
			{
				frameNum = fullTrans.size() - 1;
			}

			float pdScale = getPDScale(ai, frameNum);
			if (showTargetMocap)
			{
				Transform tran = agentOffset[ai];
				tran.p.x += 2.0f;
				for (int i = tmocapBDs[ai].first; i < (int)tmocapBDs[ai].second; i++)
				{
					int bi = i - tmocapBDs[ai].first;
					Transform tt = tran * fullTrans[frameNum][bi];
					NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
					(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(tran.q, fullVels[frameNum][bi]);
					(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(tran.q, fullAVels[frameNum][bi]);
				}
			}
			for (int i = 0; i < (int)ctrls[ai].size(); i++)
			{
				int qq = i;
				NvFlexRigidJoint& joint = g_buffers->rigidJoints[ctrls[ai][qq].first + 1]; // Active joint
				joint.compliance[ctrls[ai][qq].second] = 1.0f / (10.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f));
				//joint.compliance[ctrls[ai][qq].second] = 1.0f / (10.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f));
				//joint.compliance[ctrls[ai][qq].second] = 1.0f / (powerScale*motorPower[ai][i] * std::max(pdScale, 1e-12f));
				joint.targets[ctrls[ai][qq].second] = jointAngles[frameNum][i];
				if (limitForce)
				{
					//joint.motorLimit[ctrls[ai][qq].second] = 2.0f*motorPower[ai][i];
					joint.motorLimit[ctrls[ai][qq].second] = motorPower[ai][i];
				}
				//if (i == 20) joint.targets[ctrls[ai][qq].second] *= -1.0f;
			}
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
			for (int i = 0; i < ctrls[ai].size(); i++)
			{
				float cc = Clamp(actions[i], -1.f, 1.f);

				NvFlexRigidJoint& j = initJoints[ctrls[ai][i].first];
				NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
				NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
				Transform& pose0 = *((Transform*)&j.pose0);
				Transform gpose;
				NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
				Transform tran = gpose*pose0;

				Vec3 axis;
				if (ctrls[ai][i].second == 0)
				{
					axis = GetBasisVector0(tran.q);
				}
				if (ctrls[ai][i].second == 1)
				{
					axis = GetBasisVector1(tran.q);
				}
				if (ctrls[ai][i].second == 2)
				{
					axis = GetBasisVector2(tran.q);
				}

				Vec3 torque = axis * motorPower[ai][i] * cc * powerScale;
				a0.torque[0] += torque.x;
				a0.torque[1] += torque.y;
				a0.torque[2] += torque.z;
				a1.torque[0] -= torque.x;
				a1.torque[1] -= torque.y;
				a1.torque[2] -= torque.z;
			}

			if (ai % pushFrequency == push_ai && torso[ai] != -1)
			{
				//cout << "Push agent " << ai << endl;
				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[ai]], (NvFlexRigidPose*)&torsoPose);

				float z = torsoPose.p.y;
				Vec3 pushForce = Randf() * forceMag * RandomUnitVector();
				if (z > 1.f)
				{
					pushForce.z *= 0.2f;
				}
				else
				{
					pushForce.x *= 0.2f;
					pushForce.y *= 0.2f;
					pushForce.y *= 0.2f;
				}
				/*
				g_buffers->rigidBodies[torso[ai]].force[0] += pushForce.x;
				g_buffers->rigidBodies[torso[ai]].force[1] += pushForce.y;
				g_buffers->rigidBodies[torso[ai]].force[2] += pushForce.z;
				*/
				int bd = rand() % (agentBodies[ai].second - agentBodies[ai].first) + agentBodies[ai].first;
				g_buffers->rigidBodies[bd].force[0] += pushForce.x;
				g_buffers->rigidBodies[bd].force[1] += pushForce.y;
				g_buffers->rigidBodies[bd].force[2] += pushForce.z;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[bd], (NvFlexRigidPose*)&torsoPose);
				if (renderPush)
				{
					PushInfo pp;
					pp.pos = torsoPose.p;
					pp.force = pushForce;
					pp.time = 15;
					pushes.push_back(pp);
				}
			}
		}

		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
		g_buffers->rigidJoints.unmap();
		NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		g_frame++;
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
		g_buffers->rigidBodies.map();
		g_buffers->rigidJoints.map();
	}
	vector<int> startShape;
	vector<int> endShape;
	vector<int> startBody;
	vector<int> endBody;
	vector<NvFlexRigidShape> initRigidShapes;

	void GetShapesBounds(int start, int end, Vec3& totalLower, Vec3& totalUpper)
	{
		// calculates the union bounds of all the collision shapes in the scene
		Bounds totalBounds;

		for (int i = start; i < end; ++i)
		{
			NvFlexCollisionGeometry geo = initRigidShapes[i].geo;


			Vec3 localLower;
			Vec3 localUpper;

			GetGeometryBounds(geo, initRigidShapes[i].geoType, localLower, localUpper);
			Transform rpose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[initRigidShapes[i].body], (NvFlexRigidPose*)&rpose);
			Transform spose = rpose*(Transform&)initRigidShapes[i].pose;
			// transform local bounds to world space
			Vec3 worldLower, worldUpper;
			TransformBounds(localLower, localUpper, spose.p, spose.q, 1.0f, worldLower, worldUpper);

			totalBounds = Union(totalBounds, Bounds(worldLower, worldUpper));
		}

		totalLower = totalBounds.lower;
		totalUpper = totalBounds.upper;

	}
	virtual void ResetAgent(int a)
	{
		//mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		startFrame[a] = rand() % (lastFrame - firstFrame);
		int aa = startFrame[a] + firstFrame;
		if (aa >= fullTrans.size())
		{
			aa = fullTrans.size() - 1;
		}
		if ((a % 2 == 0) || (!halfRandomReset))
		{

			for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
			{
				int bi = i - agentBodies[a].first;
				Transform tt = agentOffset[a] * fullTrans[aa][bi];
				NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
				(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
				(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
			}
			Vec3 lower, upper;
			GetShapesBounds(startShape[a], endShape[a], lower, upper);
			for (int i = startBody[a]; i < endBody[a]; i++)
			{
				g_buffers->rigidBodies[i].com[1] -= lower.y;
			}
		}
		else
		{

			Transform trans = Transform(fullTrans[aa][0].p + Vec3(Randf() * 2.0f - 1.0f, Randf() * 2.0f - 1.0f, 0.0f), rpy2quat(Randf() * 2.0f * kPi, Randf() * 2.0f * kPi, Randf() * 2.0f * kPi));
			mjcfs[a]->reset(agentOffset[a] * trans, angleResetNoise, velResetNoise, angleVelResetNoise);
			Vec3 lower, upper;
			GetShapesBounds(startShape[a], endShape[a], lower, upper);
			for (int i = startBody[a]; i < endBody[a]; i++)
			{
				g_buffers->rigidBodies[i].com[1] -= lower.y;
			}
		}
		mFarCount[a] = 0;
		int frameNumFirst = aa;
		Transform targetPose = fullTrans[frameNumFirst][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		walkTargetX[a] = targetPose.p.x;
		walkTargetY[a] = targetPose.p.y;

		RLWalkerEnv::ResetAgent(a);
	}

	virtual void AddAgentBodiesAndJointsCtlsPowersPopulateTorsoPelvis(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		startShape[i] = g_buffers->rigidShapes.size();
		startBody[i] = g_buffers->rigidBodies.size();
		mjcfs.push_back(make_shared<MJCFImporter>(loadPath.c_str()));
		mjcfs.back()->AddPhysicsEntities(gt, ctrl, mpower);
		endShape[i] = g_buffers->rigidShapes.size();
		endBody[i] = g_buffers->rigidBodies.size();

		auto torsoInd = mjcfs[i]->bmap.find("torso");
		if (torsoInd != mjcfs[i]->bmap.end())
		{
			torso[i] = mjcfs[i]->bmap["torso"];
		}

		auto pelvisInd = mjcfs[i]->bmap.find("pelvis");
		if (pelvisInd != mjcfs[i]->bmap.end())
		{
			pelvis[i] = mjcfs[i]->bmap["pelvis"];
		}
	}

	virtual void DoStats()
	{
		if (showTargetMocap)
		{
			BeginLines(true);
			for (int i = 0; i < mNumAgents; i++)
			{
				DrawLine(g_buffers->rigidBodies[tmocapBDs[i].first].com, g_buffers->rigidBodies[agentBodies[i].first].com, Vec4(0.0f, 1.0f, 1.0f));
			}
			if (renderPush)
			{
				for (int i = 0; i < (int)pushes.size(); i++)
				{
					DrawLine(pushes[i].pos, pushes[i].pos + pushes[i].force*0.0005f, Vec4(1.0f, 0.0f, 1.0f));
					DrawLine(pushes[i].pos - Vec3(0.1f, 0.0f, 0.0f), pushes[i].pos + Vec3(0.1f, 0.0f, 0.0f), Vec4(1.0f, 1.0f, 1.0f));
					DrawLine(pushes[i].pos - Vec3(0.0f, 0.1f, 0.0f), pushes[i].pos + Vec3(0.0f, 0.1f, 0.0f), Vec4(1.0f, 1.0f, 1.0f));
					DrawLine(pushes[i].pos - Vec3(0.0f, 0.0f, 0.1f), pushes[i].pos + Vec3(0.0f, 0.0f, 0.1f), Vec4(1.0f, 1.0f, 1.0f));
					pushes[i].time--;
					if (pushes[i].time <= 0)
					{
						pushes[i] = pushes.back();
						pushes.pop_back();
						i--;
					}
				}
			}
			EndLines();
		}
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
					feetContact[ff / 2]++;
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
		// Original
		//return +2 if z > 0.78 else - 1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

		// Viktor: modified original one to enforce standing and walking high, not on knees
		// Also due to reduced electric cost bonus for living has been decreased
		/*
		if (z > 1.0)
		{
		return 1.5f;
		}
		else
		{
		return -1.f;
		}*/
		return 1.5f;// Not die because of this
	}

	float getPDScale(int a, int frameNum)
	{
		Transform targetTorso = fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		Transform cpose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
		Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;
		float posError = Length(targetTorso.p - currentTorso.p);
		Quat qE = targetTorso.q * Inverse(currentTorso.q);
		float sinHalfTheta = Length(qE.GetAxis());
		if (sinHalfTheta > 1.0f)
		{
			sinHalfTheta = 1.0f;
		}
		if (sinHalfTheta < -1.0f)
		{
			sinHalfTheta = -1.0f;
		}

		float quatError = asinf(sinHalfTheta)*2.0f;
		float pdPos = 1.0f - (posError - farStartPos) / (farEndPos - farStartPos);
		float pdQuat = 1.0f - (quatError - farStartQuat) / (farEndQuat - farStartQuat);
		float m = min(pdPos, pdQuat);
		if (m > 1.0f)
		{
			m = 1.0f;
		}
		if (m < 0.0f)
		{
			m = 0.0f;
		}
		return m;
	}

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		if (useDifferentRewardWhenFell)
		{
			int frameNumFirst = (mFrames[a] + startFrame[a]) + firstFrame;
			Transform targetPose = fullTrans[frameNumFirst][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
			walkTargetX[a] = targetPose.p.x;
			walkTargetY[a] = targetPose.p.y;
		}

		RLWalkerEnv<Transform, Vec3, Quat, Matrix33>::ExtractState(a, state, p, walkTargetDist, jointSpeeds, numJointsAtLimit, heading, upVec);
		int ct = baseNumObservations;
		if (numFramesToProvideInfo > 0)
		{
			// State:
			// Quat of torso
			// Velocity of torso
			// Angular velocity of torso
			// Relative pos of geo_pos in torso's coordinate frame
			// Future frames:
			//				 Relative Pos of target torso in current torso's coordinate frame
			//				 Relative Quat of target torso in current torso's coordinate frame
			//				 Relative Velocity of target torso in current torso's coordinate frame
			//				 Relative Angular target velocity of torso in current torso's coordinate frame
			//               Relative target pos of geo_pos in current torso's coordinate frame
			// Look at 0, 1, 4, 16, 64 frames in future
			int frameNum = (mFrames[a] + startFrame[a]) + firstFrame;
			if (frameNum >= fullTrans.size())
			{
				frameNum = fullTrans.size() - 1;
			}
			Transform cpose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
			Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;
			Transform icurrentTorso = Inverse(currentTorso);

			Vec3 currentVel = Rotate(icurrentTorso.q, TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].linearVel));
			Vec3 currentAVel = Rotate(icurrentTorso.q, TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].angularVel));


			state[ct++] = currentTorso.q.x;
			state[ct++] = currentTorso.q.y;
			state[ct++] = currentTorso.q.z;
			state[ct++] = currentTorso.q.w;

			state[ct++] = currentVel.x;
			state[ct++] = currentVel.y;
			state[ct++] = currentVel.z;

			state[ct++] = currentAVel.x;
			state[ct++] = currentAVel.y;
			state[ct++] = currentAVel.z;

			Vec3* ttt = (Vec3*)&state[ct];
			for (int i = 0; i < features[a].size(); i++)
			{
				Transform cpose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
				Vec3 pCurrent = TransformPoint(icurrentTorso, TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p)));
				state[ct++] = pCurrent.x;
				state[ct++] = pCurrent.y;
				state[ct++] = pCurrent.z;
			}

			for (int q = 0; q < numFramesToProvideInfo; q++)
			{
				if (q == 0)
				{
					frameNum = (mFrames[a] + startFrame[a]) + firstFrame;
				}
				else
				{
					frameNum = (mFrames[a] + startFrame[a]) + firstFrame + (1 << (2 * (q)));
				}
				if (frameNum >= fullTrans.size())
				{
					frameNum = fullTrans.size() - 1;
				}


				Transform targetTorso = icurrentTorso*fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
				Vec3 targetVel = Rotate(icurrentTorso.q, fullVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]);
				Vec3 targetAVel = Rotate(icurrentTorso.q, fullAVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]);
				state[ct++] = targetTorso.p.x;
				state[ct++] = targetTorso.p.y;
				state[ct++] = targetTorso.p.z;

				state[ct++] = targetTorso.q.x;
				state[ct++] = targetTorso.q.y;
				state[ct++] = targetTorso.q.z;
				state[ct++] = targetTorso.q.w;

				state[ct++] = targetVel.x;
				state[ct++] = targetVel.y;
				state[ct++] = targetVel.z;

				state[ct++] = targetAVel.x;
				state[ct++] = targetAVel.y;
				state[ct++] = targetAVel.z;

				//float sumError = 0.0f;
				for (int i = 0; i < features[a].size(); i++)
				{
					Vec3 pCurrent = ttt[i];
					Vec3 pTarget = TransformPoint(icurrentTorso, TransformPoint(fullTrans[frameNum][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p));

					state[ct++] = pTarget.x - pCurrent.x;
					state[ct++] = pTarget.y - pCurrent.x;
					state[ct++] = pTarget.z - pCurrent.x;

				}
			}
			state[ct++] = mFarCount[a] / maxFarItr; // When 1, die
			state[ct++] = getPDScale(a, frameNum);
		}
	}
};


class RigidFullHumanoidTrackTargetAnglesModifiedWithReducedControlAndDisturbanceMulti : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	int firstFrame;
	int lastFrame;
	vector<int> startFrame;
	vector<int> rightFoot;
	vector<int> leftFoot;

	vector<int> footFlag;
	vector<vector<int>> afeetInAir;
	vector<vector<vector<Transform>>> afullTrans;
	vector<vector<vector<Vec3>>> afullVels;
	vector<vector<vector<Vec3>>> afullAVels;
	vector<vector<vector<float>>> ajointAngles;
	string fullFileName;
	vector<int> agentAnim;

	vector < vector<pair<int, Transform>>> features;
	vector<string> geo_joint;
	float ax, ay, az;
	float isdx, isdy, isdz;
	int numFramesToProvideInfo;
	int baseNumObservations;
	bool showTargetMocap;
	vector<MJCFImporter*> tmjcfs;
	vector<pair<int, int>> tmocapBDs;
	vector<int> mFarCount;
	float maxFarItr; // More than this will die
	float farStartPos; // When to start consider as being far, PD will start to fall off
	float farStartQuat; // When to start consider as being far, PD will start to fall off
	float farEndPos; // PD will be 0 at this point and counter will start
	float farEndQuat; // PD will be 0 at this point and counter will start
	bool renderPush;
	bool limitForce; // Specify force limit
	bool pauseMocapWhenFar; // Implemented by decreasing startFrame
	bool useDifferentRewardWhenFell;
	bool halfRandomReset;
	bool morezok;
	bool useAllFrames;
	bool killWhenFall;
	bool switchAnimationWhenEnd;
	bool useDeltaPDController;
	bool useRelativeCoord;
	vector<PushInfo> pushes;
	vector<int> firstFrames;

	bool withContacts; // Has magnitude of contact force at knee, arm, sholder, head, etc..
	vector<string> contact_parts;
	vector<float> contact_parts_penalty_weight;
	vector<vector<Vec3>> contact_parts_force;
	vector<int> contact_parts_index;

	float earlyDeadPenalty;
	bool useCMUDB;
	bool alternateParts;
	bool correctedParts;
	bool killImmediately;
	bool providePreviousActions;
	bool forceLaterFrame;
	bool withPDFallOff;
	float flyDeadPenalty;
	vector<vector<float> > prevActions;
	vector<Transform> addedTransform;
	int numReload;
	float jointAngleNoise, velNoise, aavelNoise;
	vector<int> startShape;
	vector<int> endShape;
	vector<int> startBody;
	vector<int> endBody;
	vector<NvFlexRigidShape> initRigidShapes;
	bool allMatchPoseMode;
	bool useMatchPoseBrain;
	vector<bool> matchPoseMode;
	vector<float> lastRews;
	bool ragdollMode;
	bool changeAnim;
	bool doAppendTransform;
	bool halfSavedTransform;

	vector < vector<pair<int, int> >> transits;
	vector<vector<vector<Transform>>> tfullTrans;
	vector<vector<vector<Vec3>>> tfullVels;
	vector<vector<vector<Vec3>>> tfullAVels;
	vector<vector<vector<float>>> tjointAngles;
	bool useBlendAnim;
	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);
		LoadVec(f, startFrame);
		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);
		LoadVec(f, footFlag);
		LoadVec(f, agentAnim);
		LoadVec(f, mFarCount);

		LoadVec(f, pushes);
		LoadVec(f, firstFrames);

		LoadVec(f, contact_parts);
		LoadVec(f, contact_parts_penalty_weight);
		LoadVecVec(f, contact_parts_force);
		LoadVec(f, contact_parts_index);

		LoadVecVec(f, prevActions);
		LoadVec(f, addedTransform);


	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);
		SaveVec(f, startFrame);
		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);
		SaveVec(f, footFlag);
		SaveVec(f, agentAnim);
		SaveVec(f, mFarCount);

		SaveVec(f, pushes);
		SaveVec(f, firstFrames);

		SaveVec(f, contact_parts);
		SaveVec(f, contact_parts_penalty_weight);
		SaveVecVec(f, contact_parts_force);
		SaveVec(f, contact_parts_index);

		SaveVecVec(f, prevActions);
		SaveVec(f, addedTransform);
	}

	// Reward:
	//   Global pose error
	//	 Quat of torso error
	//   Position of torso error
	//   Velocity of torso error
	//   Angular velocity of torso error
	//	 Relative position with respect to torso of target and relative position with respect to torso of current
	//
	// State:
	// Quat of torso
	// Velocity of torso
	// Angular velocity of torso
	// Relative pos of geo_pos in torso's coordinate frame
	// Future frames:
	//				 Relative Pos of target torso in current torso's coordinate frame
	//				 Relative Quat of target torso in current torso's coordinate frame
	//				 Relative Velocity of target torso in current torso's coordinate frame
	//				 Relative Angular target velocity of torso in current torso's coordinate frame
	//               Relative target pos of geo_pos in current torso's coordinate frame
	// Look at 0, 1, 4, 16, 64 frames in future
	int forceDead;
	//vector<pair<float, float> > rewRec;
	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		int anum = agentAnim[a];
		vector<vector<Transform>>& fullTrans = (useBlendAnim) ? tfullTrans[a] : afullTrans[anum];
		vector<vector<Vec3>>& fullVels = (useBlendAnim) ? tfullVels[a] : afullVels[anum];
		vector<vector<Vec3>>& fullAVels = (useBlendAnim) ? tfullAVels[a] : afullAVels[anum];
		//vector<vector<float>>& jointAngles = ajointAngles[anum];
		float& potential = potentials[a];
		float& potentialOld = potentialsOld[a];
		float& p = ps[a];
		float& walkTargetDist = walkTargetDists[a];
		float* joint_speeds = &jointSpeeds[a][0];
		int& jointsAtLimit = jointsAtLimits[a];

		//float& heading = headings[a];
		//float& upVec = upVecs[a];

		float electrCost = electricityCostScale * electricityCost;
		float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

		float alive = AliveBonus(state[0] + initialZ, p); //   # state[0] is body height above ground, body_rpy[1] is pitch
		dead = alive < 0.f;

		potentialOld = potential;
		potential = -walkTargetDist / (dt);
		if (potentialOld > 1e9)
		{
			potentialOld = potential;
		}

		float progress = potential - potentialOld;
		float oprogress = progress;
		//-----------------------
		int frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a];
		if (frameNum >= fullTrans.size())
		{
			frameNum = fullTrans.size() - 1;
		}

		// Global error
		Transform targetTorso = addedTransform[a] * fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		Transform cpose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
		Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;
		Vec3 targetVel = Rotate(addedTransform[a].q, fullVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]);
		Vec3 currentVel = TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].linearVel);
		Vec3 targetAVel = Rotate(addedTransform[a].q, fullAVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]);
		Vec3 currentAVel = TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].angularVel);

		float posError = Length(targetTorso.p - currentTorso.p);
		float zError = 0.0f;
		if (morezok)
		{
			zError = max(0.0f, targetTorso.p.z - currentTorso.p.z);
		}
		else
		{
			zError = fabs(targetTorso.p.z - currentTorso.p.z);
		}
		if (matchPoseMode[a])
		{
			// More z is not OK
			zError = fabs(targetTorso.p.z - currentTorso.p.z);
		}
		float velError = Length(targetVel - currentVel);
		float avelError = Length(targetAVel - currentAVel);
		Quat qE = targetTorso.q * Inverse(currentTorso.q);
		float sinHalfTheta = Length(qE.GetAxis());
		if (sinHalfTheta > 1.0f)
		{
			sinHalfTheta = 1.0f;
		}
		if (sinHalfTheta < -1.0f)
		{
			sinHalfTheta = -1.0f;
		}

		float quatError = asinf(sinHalfTheta)*2.0f;
		Transform itargetTorso = Inverse(targetTorso);
		Transform icurrentTorso = Inverse(currentTorso);

		// Local error
		float sumError = 0.0f;
		for (int i = 0; i < features[a].size(); i++)
		{
			Vec3 pTarget = TransformPoint(itargetTorso, TransformPoint(addedTransform[a] * fullTrans[frameNum][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p));
			Transform cpose;

			NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
			Vec3 pCurrent = TransformPoint(icurrentTorso, TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p)));

			sumError += LengthSq(pCurrent - pTarget);
		}
		float localError = sqrt(sumError / features[a].size());
		if (a == 0)
		{
			//cout << "Agent " << a << " frame="<<mFrames[a]<<" posE=" << posError << " velE=" << velError << " avelE=" << avelError << " qE=" << quatError << " lE=" << localError << endl;
		}

		// For all but 07
		//if (posError > 0.6f) dead = true; // Position error > 0.6
		//if (quatError > kPi*0.5f) dead = true; // Angular error > 90 deg
		//if (posError > .0f) dead = true; // Position error > 3.0, for 07 as it's fast
		//if (quatError > 2 * kPi) dead = true; // Angular error > 360 deg
		if (!matchPoseMode[a])
		{
			if (killWhenFall)
			{
				if (killImmediately)
				{
					if (currentTorso.p.z < 0.8f*targetTorso.p.z)
					{
						dead = true;
					}
				}
				else
				{
					if (currentTorso.p.z < targetTorso.p.z - 0.2f)
					{
						mFarCount[a] += 10;
					}
				}
			}
		}
		bool fell = false;
		if (!matchPoseMode[a])
		{
			if ((posError > farEndPos) || (quatError > farEndQuat))
			{
				fell = true;
				mFarCount[a]++;
				if (pauseMocapWhenFar)
				{
					startFrame[a]--;
				}
			}
			else
			{
				mFarCount[a]--;
				if (mFarCount[a] < 0)
				{
					mFarCount[a] = 0;
				}
			}
		}
		else
		{
			startFrame[a]--; // Pause mocap
		}
		if ((matchPoseMode[a]) && (useMatchPoseBrain) && (!allMatchPoseMode))
		{
			if ((posError > farEndPos) || (quatError > farEndQuat))
			{
				// Still far, use matchPose mode still
			}
			else
			{
				// Not far anymore
				mFarCount[a] -= 30;
				if (mFarCount[a] < 0)
				{
					// Revert back to mocap tracking mode
					mFarCount[a] = 0;
					matchPoseMode[a] = false;
				}
			}
		}
		if (!matchPoseMode[a])
		{
			if (mFarCount[a] > maxFarItr)
			{
				if ((useMatchPoseBrain) && (!allMatchPoseMode))
				{
					matchPoseMode[a] = true;
				}
				else
				{
					dead = true;
				}
			}
		}
		if (matchPoseMode[a])
		{
			dead = false;
		}
		float posWeight = 1.0f;
		float quatWeight = 1.0f;
		//float velWeight = 0.2f;
		//float avelWeight = 0.2f;
		float velWeight = 1.0f; // More vel weight
		float avelWeight = 1.0f;

		//float localWeight = 1.0f;
		float localWeight = 3.0f; // More
		float zErrorMax = 1.0f;
		float zWeight = 2.0f;

		if (matchPoseMode[a])
		{
			zWeight += posWeight;
			quatWeight *= 30.0f;
			velWeight = 0.0f;
			avelWeight = 0.0f;
			localWeight *= 2.0f;
			//zWeight *= 10.0f; // more
			//zWeight *= 5.0f; // more but not too much
			//zWeight *= 10.0f; // more but not too much
			//posWeight = 0.0f; // Ignore global position, only care about z
			zWeight *= 20.0f;
			posWeight = 60.0f; // Care position too, a lot
			if (posError > farStartPos)
			{
				quatWeight = 0.0f; // Orientation does not matter, if far
				localWeight = 0.0f;
			}
		}
		//float localR = localWeight*max(0.1f - localError, 0.0f) / 0.1f;

		progress = posWeight*max(farStartPos - posError, 0.0f) / farStartPos + quatWeight*max(farStartQuat - quatError, 0.0f) / farStartQuat + velWeight*max(2.0f - velError, 0.0f) / 2.0f + avelWeight*max(2.0f - avelError, 0.0f) / (2.0f) + localWeight*max(0.1f - localError, 0.0f) / 0.1f + zWeight*pow(max(zErrorMax - zError, 0.0f) / zErrorMax, 2.0f);
		/*
		float otherR = progress;

		rewRec.push_back(make_pair(localR, otherR));
		if (rewRec.size() % 10000 == 1) {
			double sumL = 0.0, sumR = 0.0;
			for (int q = 0; q < rewRec.size(); q++) {
				sumL += rewRec[q].first;
				sumR += rewRec[q].second;
			}
			sumL /= rewRec.size();
			sumR /= rewRec.size();
			cout << "local = " << sumL << " all = " << sumR << " percent = " << (sumL / sumR)*100.0 << endl;
		}
		*/

		if (matchPoseMode[a])
		{
			float velPenalty = 0.2f;
			float avelPenalty = 0.2f;
			progress -= velPenalty * Length(currentVel);
			progress -= avelPenalty * Length(currentAVel);
		}
		if (!matchPoseMode[a])
		{
			if (!useDifferentRewardWhenFell)
			{
				if (fell)
				{
					// If fall down, penalize height more severely
					progress -= fabs(currentTorso.p.z - targetTorso.p.z)*posWeight;
				}
				if (posError > farStartQuat)
				{
					progress -= (posError - farStartQuat)*posWeight*0.2f;
				}
				if (quatError > farStartQuat)
				{
					progress -= (quatError - farStartQuat)*quatWeight*0.2f;
				}
			}
			else
			{
				if (fell)
				{
					// If fell, use a different reward to first try to go to the target pose

					float zDif = 1.0f - max(targetTorso.p.z - currentTorso.p.z, 0.0f);
					float zWeight = 3.0f;
					float tmp = posWeight*(farEndPos - posError) / farEndPos + quatWeight*(farEndQuat - quatError) / farEndQuat + zWeight*zDif;
					progress = oprogress + tmp; // Use oprogress
					if (!isfinite(progress))
					{
						cout << "Fell = " << fell << " oprogress = " << oprogress << " tmp = " << tmp << endl;
					}
				}
			}
		}
		if (withContacts)
		{
			float forceMul = 1.0f / 3000.0f;
			/*
			if (matchPoseMode[a]) {
			//forceMul *= 0.1f;
			forceMul *= 0.2f; // More penalty for contact
			for (int i = 0; i < contact_parts.size(); i++)
			{
			//				if (a == 0) {
			//				cout << i << " " << Length(contact_parts_force[a][i]) << endl;
			//}
			progress -= LengthSq(contact_parts_force[a][i])*contact_parts_penalty_weight[i] * forceMul;
			}

			}
			else {
			*/
			if (matchPoseMode[a])
			{
				forceMul *= 1.5f;
			}
			for (int i = 0; i < contact_parts.size(); i++)
			{
				//				if (a == 0) {
				//				cout << i << " " << Length(contact_parts_force[a][i]) << endl;
				//}
				progress -= Length(contact_parts_force[a][i])*contact_parts_penalty_weight[i] * forceMul;
			}
			//}

		}
		if (!isfinite(progress))
		{
			cout << "Fell = " << fell << "pE = " << posError << " qE = " << quatError << endl;
		}
		//------------------------
		float electricityCostCurrent = 0.0f;
		float sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			float vv = abs(action[i] * joint_speeds[i]);
			if (!isfinite(vv))
			{
				printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv, action[i], joint_speeds[i]);
				//exit(0);
			}

			if (!isfinite(action[i]))
			{
				printf("action at %d is infinite\n", i);
				//exit(0);
			}

			if (!isfinite(joint_speeds[i]))
			{
				printf("joint_speeds at %d is infinite\n", i);
				//exit(0);
			}

			sum += vv;
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl*joint_speed is infinite!\n");
			//exit(0);
		}

		//electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
		electricityCostCurrent += electrCost * sum / (float)ctrls[a].size();

		sum = 0.0f;
		for (int i = 0; i < ctrls[a].size(); i++)
		{
			sum += action[i] * action[i];
		}

		if (!isfinite(sum))
		{
			printf("Sum of ctl^2 is infinite!\n");
			//exit(0);
		}

		//electricity_costCurrent += stall_torque_cost * float(np.square(a).mean())
		electricityCostCurrent += stallTorqCost * sum / (float)ctrls[a].size();

		float jointsAtLimitCostCurrent = jointsAtLimitCost * jointsAtLimit;

		float feetCollisionCostCurrent = 0.0f;
		if (numCollideOther[a] > 0)
		{
			feetCollisionCostCurrent += footCollisionCost;
		}

		//cout << "heading = " << heading.x << " " << heading.y << " " << heading.z << endl;
		//float heading_rew = 0.2f*((heading.x > 0.5f) ? 1.0f: heading.x*2.0f); // MJCF3
		//float heading_rew = heading.x; // MJCF2
		//		float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4

		//cout << mind << endl;
		// Heading was included, but actually probabably shouldn't, not sure about upvec to make it up right, but don't think so
		float rewards[5] =
		{
			alive,
			progress,
			electricityCostCurrent,
			jointsAtLimitCostCurrent,
			feetCollisionCostCurrent,

		};


		//printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

		rew = 0.f;
		for (int i = 0; i < 5; i++)
		{
			if (!isfinite(rewards[i]))
			{
				printf("Reward %d is infinite\n", i);
			}
			rew += rewards[i];
		}

		if (!matchPoseMode[a])
		{
			if (currentTorso.p.z > targetTorso.p.z + 0.5f)
			{
				dead = true;
				rew += flyDeadPenalty;
			}
			else if (dead)
			{
				rew += earlyDeadPenalty;
			}
			// Run out of frame
			if ((mFrames[a] + startFrame[a]) + firstFrames[a] >= fullTrans.size())
			{
				dead = true;
				// No penalty
				//rew = mMaxEpisodeLength - mFrames[a];
			}
		}

		if (matchPoseMode[a])
		{
			rew *= 0.1f;
			//if (rew > 4.0f) dead = true;
		}
		dead = false;
		if (forceDead > 0)
		{
			dead = true;
			forceDead--;
		}

		lastRews[a] = rew;
		/*
		if (mFrames[a] % 60 == 59) {
		agentAnim[a] = rand() % afullTrans.size();
		int lf = max((int)afullTrans[agentAnim[a]].size() - 500, 38);
		int sf = 10;
		firstFrames[a] = sf;
		startFrame[a] = rand() % (lf - firstFrames[a]);
		}*/
	}

	vector<string> debugString;
	bool useVarPDAction;
	bool pureTorque;
	bool purePDController;
	bool hasTerrain;
	bool throwBox;
	bool clearBoxes;
	int bkNumBody;
	int bkNumShape;
	RigidFullHumanoidTrackTargetAnglesModifiedWithReducedControlAndDisturbanceMulti()
	{
		throwBox = false;
		hasTerrain = false;
		if (hasTerrain)
		{
			yOffset = 1.0f;
		}
		clearBoxes = false;
		rcount = 0;
		LoadTransforms();
		pureTorque = false;
		purePDController = false;
		useRelativeCoord = true;
		changeAnim = false;
		ragdollMode = false;
		doAppendTransform = false;
		halfRandomReset = false;
		halfSavedTransform = false;
		allMatchPoseMode = false;
		useMatchPoseBrain = true;
		useVarPDAction = true;
		jointAngleNoise = 0.2f;
		velNoise = 0.2f;
		aavelNoise = 0.2f;
		withPDFallOff = true;
		flyDeadPenalty = 0.0f; // Used to be same as early dead
		//earlyDeadPenalty = -200.0f; // Before it's 0
		earlyDeadPenalty = -400.0f; // Before it's 0
		useDeltaPDController = false; // Terrible IDEA :P
		useCMUDB = true;
		forceLaterFrame = false;
		switchAnimationWhenEnd = false;
		providePreviousActions = true;
		killWhenFall = true;
		killImmediately = false;
		alternateParts = false; // False seems better
		correctedParts = true;
		limitForce = false;
		withContacts = true;
		morezok = true;
		pauseMocapWhenFar = false;
		useDifferentRewardWhenFell = false;

		farStartPos = 0.6f; // When to start consider as being far, PD will start to fall off
		farStartQuat = kPi*0.5f; // When to start consider as being far, PD will start to fall off
		//farEndPos = 2.0f; // PD will be 0 at this point and counter will start, was 2.0
		farEndPos = 1.0f; // PD will be 0 at this point and counter will start, was 2.0
		farEndQuat = kPi*1.0f; // PD will be 0 at this point and counter will start

		showTargetMocap = true;
		renderPush = true;
		maxFarItr = 180.0f;
		//firstFrame = 30;
		useAllFrames = false;
		firstFrame = 10;
		lastFrame = 38; // stand
		//lastFrame = 1100; // 02 easy
		//lastFrame = 3000; //02
		//lastFrame = 9000; //12
		//lastFrame = 7000; //11

		//lastFrame = 6400; //08
		//lastFrame = 4900; //07

		doFlagRun = false;
		loadPath = "../../data/humanoid_mod.xml";

		mNumAgents = 2;
		baseNumObservations = 52;
		mNumObservations = baseNumObservations;
		mNumActions = 21;
		if (useVarPDAction)
		{
			mNumActions *= 2;
		}
		mMaxEpisodeLength = 2000; // longer episode
		if (allMatchPoseMode)
		{
			mMaxEpisodeLength = 180;
		}
		//mMaxEpisodeLength = 500;

		//geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
		geo_joint = { "torso1","right_thigh", "right_foot","left_thigh","left_foot" };
		//contact_parts = { "torso", "lwaist", "pelvis", "right_lower_arm", "right_upper_arm", "right_thigh", "right_shin", "right_foot", "left_lower_arm", "left_upper_arm", "left_thigh", "left_shin", "left_foot" };
		contact_parts = { "torso", "right_thigh", "right_foot", "left_thigh", "left_foot" };
		contact_parts_penalty_weight = { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };

		if (alternateParts)
		{
			geo_joint = { "torso1","right_lower_arm", "right_foot","left_lower_arm","left_foot" };
			contact_parts = { "torso", "right_lower_arm", "right_foot", "left_lower_arm", "left_foot" };
			contact_parts_penalty_weight = { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };
		}
		if (correctedParts)
		{
			geo_joint = { "torso1","right_upper_arm", "right_foot", "right_hand", "left_upper_arm","left_foot", "left_hand"};
			contact_parts = { "torso", "right_hand", "right_foot", "left_hand", "left_foot" };
			contact_parts_penalty_weight = { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };
		}
		contact_parts_force.resize(mNumAgents);
		for (int i = 0; i < mNumAgents; i++)
		{
			contact_parts_force[i].resize(contact_parts.size());
		}
		numFramesToProvideInfo = 4;
		if (numFramesToProvideInfo > 0)
		{
			mNumObservations += 10 + 3 * geo_joint.size() + numFramesToProvideInfo * (13 + 3 * geo_joint.size()) + 1 + 1; // Self, target current and futures, far count
		}
		if (withContacts)
		{
			mNumObservations += contact_parts.size() * 3;
		}
		if (providePreviousActions)
		{
			mNumObservations += mNumActions;
		}

		if (useDeltaPDController)
		{
			mNumObservations += mNumActions;    // Provide angles
		}
		g_numSubsteps = 4;
		//g_params.numIterations = 100;
		g_params.numIterations = 20;
		g_params.dynamicFriction = 1.0f; // 0.0
		g_params.staticFriction = 1.0f;
		//g_params.numIterations = 32; GAN4

		//		g_sceneLower = Vec3(-150.f, -250.f, -100.f);
		//g_sceneUpper = Vec3(250.f, 150.f, 100.f);
		g_sceneLower = Vec3(-7.0f);
		g_sceneUpper = Vec3(7.0f);

		g_pause = false;
		mDoLearning = true;
		numRenderSteps = 1;
		numReload = 600;

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		numPerRow = 20;
		spacing = 50.f;

		numFeet = 2;

		//power = 0.41f; // Default
		//powerScale = 0.25f; // Reduced power
		//powerScale = 0.5f; // More power
		//powerScale = 0.41f; // More power
		//powerScale = 0.41f; // Default
		powerScale = 0.82f; // More power
		//powerScale = 1.64f; // Even more power
		//powerScale = 1.0f; // More power
		initialZ = 0.9f;

		//electricityCostScale = 1.f;
		//electricityCostScale = 1.8f; //default
		electricityCostScale = 3.6f; //more
		//electricityCostScale = 7.2f; //even more

		angleResetNoise = 0.f;
		angleVelResetNoise = 0.0f;
		velResetNoise = 0.0f;

		pushFrequency = 100;	// 200 How much steps in average per 1 kick // A bit too frequent
		pushFrequency = 200;	// 200 How much steps in average per 1 kick
		forceMag = 0.0f;
		//forceMag = 2000.f; // 3/7/2018
		//forceMag = 0.f; // 10000.0f
		//forceMag = 4000.f; // 10000.0f
		//forceMag = 10000.f; // 10000.0f Too much, can't learn anything useful fast...

		startShape.resize(mNumAgents, 0);
		endShape.resize(mNumAgents, 0);
		startBody.resize(mNumAgents, 0);
		endBody.resize(mNumAgents, 0);

		LoadEnv();
		contact_parts_index.clear();
		contact_parts_index.resize(g_buffers->rigidBodies.size(), -1);
		for (int i = 0; i < mNumAgents; i++)
		{
			for (int j = 0; j < contact_parts.size(); j++)
			{
				contact_parts_index[mjcfs[i]->bmap[contact_parts[j]]] = i*contact_parts.size() + j;
			}
		}
		startFrame.resize(mNumAgents, 0);
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
		mFarCount.clear();
		mFarCount.resize(mNumAgents, 0);
		initRigidShapes.resize(g_buffers->rigidShapes.size());
		for (size_t i = 0; i < initRigidShapes.size(); i++)
		{
			initRigidShapes[i] = g_buffers->rigidShapes[i];
		}
		agentAnim.resize(mNumAgents, 0);
		firstFrames.resize(mNumAgents, 0);

		prevActions.resize(mNumAgents);
		for (int i = 0; i < mNumAgents; i++)
		{
			prevActions[i].resize(mNumActions, 0.0f);
		}
		addedTransform.resize(mNumAgents);
		lastRews.resize(mNumAgents);
		matchPoseMode.resize(mNumAgents, allMatchPoseMode);

		if (mDoLearning)
		{
			PPOLearningParams ppo_params;


			ppo_params.useGAN = false;
			//ppo_params.resume = 3457;// 6727;
			ppo_params.resume = 10325;// 6727;
			ppo_params.timesteps_per_batch = 20000;//400;
			ppo_params.num_timesteps = 2000000001;
			ppo_params.hid_size = 512;
			ppo_params.num_hid_layers = 2;
			ppo_params.optim_batchsize_per_agent = 64;

			ppo_params.optim_schedule = "adaptive";
			ppo_params.desired_kl = 0.01f; // 0.01f orig
			//ppo_params.optim_stepsize = 1e-4f;
			//ppo_params.optim_schedule = "constant";


			//ppo_params.desired_kl = 0.005f; // 0.01f orig

			//string folder = "flexTrackTargetAngleModRetry2"; This is great!
			//string folder = "flexTrackTargetAngleGeoMatching_numFramesToProvideInfo_1";
			//string folder = "flexTrackTargetAngleTargetVel_BugFix_numFramesToProvideInfo_0";
			//string folder = "flexTrackTargetAnglesModified_bigDB";
			//string folder = "flexTrackTargetAnglesModified_bigDB_12";
			//string folder = "flexTrackTargetAnglesModified_bigDB_07";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far_0.6_info_skip_20";
			//string folder = "flexTrackTargetAngleGeoMatching_BufFix_numFramesToProvideInfo_3_full_relativePose_kill_when_far";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_11_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far_forcemag_10000";
			//string folder = "track_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far_use_different_reward_when_fell";
			//string folder = "qqqq";
			//string folder = "track_02_far_end_1.0_pause_mocap_when_far_use_different_reward_when_fell_half_random_pose_stand";
			//string folder = "track_02_far_end_1.0_pause_mocap_when_far_use_different_reward_when_fell_half_random_pose";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_pow_1.0";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_multi_diff_z_pow_2_more_z_ok";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_multi_diff_z_pow_2_more_z_ok_hid_1024_all_anim_no_03_mirror_2000";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_multi_diff_z_pow_2_more_z_ok_hid_1024_all_anim_no_03_mirror";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_multi_diff_z_pow_2_more_z_ok_hid_1024_all_anim_no_03_mirror_just_flat_loco_and_stand_no_limit_force";

			//01/10/2018 string folder = "track_02_far_end_1.0_pause_mocap_stand_multi_diff_z_pow_2_more_z_ok_hid_1024_all_anim_no_03_mirror_just_flat_loco_and_stand_limit_force_wcontact_alternate_contact_kill_when_fall_0.8";
			//string folder = "track_multi_with_prev_action_delayed_kill_cmu_db";
			//string folder = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m200_no_force_limit_2000agents_more_power_400_per_batch";
			//string folder = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m200_no_force_limit_even_more_power_100itr";
			//string folder = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m200_no_force_limit_100itr_more_pd_gain_no_pd_fall_off";

			//string folder = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m200_no_force_limit_100itr_less_pd_gain";
			//string folder = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m200_no_force_limit_20itr_less_pd_gain";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_more_power";

			//CURRENT
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_more_pd_gain_more_electricity_cost_no_PD_fall";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_even_more_pd_gain_even_more_power_even_more_electricity_cost_from_0";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_more_pd_gain_more_power_more_electricity_cost_from_0_512_200";

			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_more_pd_gain_more_power_more_electricity_cost_from_0_512_400";
			//ppo_params.relativeLogDir = "track_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost_longer_episode_pd_damp_100_noise";
			//ppo_params.relativeLogDir = "track_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost_longer_episode_pd_damp_100_match_pose_fixed_more_z_weight";
			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_change_target_every_60";

			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_half_from_floor_push_2000_contact_sq_more_local_a_slight_bit_more_contact_penalty";
			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_half_from_floor_push_2000_contact_sq_more_local_a_slight_bit_more_contact_penalty_mimic";
			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_half_from_floor_push_2000_more_local_a_slight_bit_more_contact_penalty_mimic";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost_longer_episode";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_matchpose_less_pd_longer_ep_corrected";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_matchpose_less_pd_longer_ep_corrected_no_pd_morez";

			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected";


			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_pure_torque_corrected";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_pure_pd_corrected";
			ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_pd_action";

			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_pd_action_constant_step_size_1e-4";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_pd_action_matchpose";

			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_pure_torque";

			//ppo_params.relativeLogDir = "track_mimic_no_force_quat_reward_when_near_lower_power_less_PD_matchpose";

			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_less_power_less_kl";
			//ppo_params.relativeLogDir = "track_mimic_no_force_quat_reward_when_near_lower_power_no_PD_matchpose_more_pos_weight";


			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_matchpose";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_matchpose";
			//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_mimic";
			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_half_from_floor_push_2000_more_local_a_slight_bit_more_contact_penalty";

			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_force_limit_half_from_floor_push_2000_contact_sq_more_local_more_contact_penalty";
			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_force_limit_half_from_floor_push_2000_more_limit_more_penalty_for_contact_sq_more_local_more_contact_penalty";
			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_force_limit_half_from_floor_push_2000_more_limit_mimic";
			//ppo_params.relativeLogDir = "track_with_PD_more_qtoo_vel_penalty_pdd_180_re_mz_force_limit_half_from_floor_push_2000_mimic";


			//ppo_params.relativeLogDir = "track_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost_pd_damp_100_match_pose_fixed_z_weight_5_moreznotok_with_PD_more_qtoo_vel_penalty_pdd_180_re";
			//ppo_params.relativeLogDir = "track_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost_pd_damp_100_match_pose_fixed_z_weight_5_moreznotok_with_PD_more_qtoo_vel_penalty_pdd_180_re";


			//exit(0);
			//ppo_params.relativeLogDir = "track_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost_longer_episode_pd_damp_100";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost_longer_episode";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_pd_gain_power_electricity_cost_512_more_vel_cost";


			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_more_power";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_more_power_more_pd_gain";

			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_noises_more_power";
			//ppo_params.relativeLogDir = "track_multi_with_prev_action_delayed_CMU_split_256_earlydead_m400_fly_m0_no_force_limit_20itr_more_local_weight_noises_more_power_no_PD";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_multi_diff_z_pow_2_more_z_ok_hid_1024_all_anim_no_03_mirror_just_flat_loco_and_stand_limit_force_wcontact_kill_when_fall_0.8";
			//string folder = "track_02_far_end_1.0_pause_mocap_stand_multi_diff_z_pow_2_more_z_ok_hid_1024_all_anim_no_03_mirror_just_flat_loco_and_stand_limit_force";

			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0_force_limit_pow_0.41_elect_1.8_force_limit_to_motor_random_force_pause_mocap_when_far_use_different_reward_when_fell";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02";
			//string folder = "dummy";
			//string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";

			EnsureDirExists(ppo_params.workingDir + string("/") + ppo_params.relativeLogDir);

			char cm[5000];
			sprintf(cm, "copy ..\\..\\demo\\scenes\\rigidbvhretargettest.h c:\\new_baselines\\%s", ppo_params.relativeLogDir.c_str());
			cout << cm << endl;
			system(cm);
			//fullFileName = "../../data/bvh/LocomotionFlat02_000_full.state";

			ppo_params.TryParseJson(g_sceneJson);
			vector<string> fnames = { "140_09","140_03","140_04","140_02","140_01","139_18","139_17","139_16" };
			useAllFrames = true;
#if 0
			vector<string> fnames =
			{
				"139_17",
				"139_18",
				"140_01",
				"140_02",
				"140_03",
				"140_04",
				"140_08",
				"140_09",
				"LocomotionFlat01_000",
				"LocomotionFlat01_000_mirror",
				"LocomotionFlat02_000",
				"LocomotionFlat02_000_mirror",
				"LocomotionFlat02_001",
				"LocomotionFlat02_001_mirror",
				"LocomotionFlat03_000",
				"LocomotionFlat03_000_mirror",
				"LocomotionFlat04_000",
				"LocomotionFlat04_000_mirror",
				"LocomotionFlat05_000",
				"LocomotionFlat05_000_mirror",
				"LocomotionFlat06_001",
				"LocomotionFlat06_001_mirror",
				"LocomotionFlat07_000",
				"LocomotionFlat07_000_mirror",
				"LocomotionFlat08_000",
				"LocomotionFlat08_000_mirror",
				"LocomotionFlat08_001",
				"LocomotionFlat08_001_mirror",
				"LocomotionFlat09_000",
				"LocomotionFlat09_000_mirror",
				"LocomotionFlat10_000",
				"LocomotionFlat10_000_mirror",
				"LocomotionFlat11_000",
				"LocomotionFlat11_000_mirror",
				"LocomotionFlat12_000",
				"LocomotionFlat12_000_mirror"/*,
											 "NewCaptures01_000",
											 "NewCaptures01_000_mirror",
											 "NewCaptures02_000",
											 "NewCaptures02_000_mirror",
											 "NewCaptures03_000",
											 "NewCaptures03_000_mirror",
											 "NewCaptures03_001",
											 "NewCaptures03_001_mirror",
											 "NewCaptures03_002",
											 "NewCaptures03_002_mirror",
											 "NewCaptures04_000",
											 "NewCaptures04_000_mirror",
											 "NewCaptures05_000",
											 "NewCaptures05_000_mirror",
											 "NewCaptures07_000",
											 "NewCaptures07_000_mirror",
											 "NewCaptures08_000",
											 "NewCaptures08_000_mirror",
											 "NewCaptures09_000",
											 "NewCaptures09_000_mirror",
											 "NewCaptures10_000",
											 "NewCaptures10_000_mirror",
											 "NewCaptures11_000",
											 "NewCaptures11_000_mirror"*/
			};
#endif
			forceDead = 0;
			if (useCMUDB)
			{
				fnames.clear();

				ifstream inf("../../data/bvh/new/list.txt");
				string str;
				while (inf >> str)
				{
					fnames.push_back("new\\" + str);

				}
				inf.close();
			}
			/*
			fnames = { "old/139_16",
			"old/139_17",
			"old/139_18",
			"old/140_01",
			"old/140_02",
			"old/140_03",
			"old/140_04",
			"old/140_08",
			"old/140_09",
			"old/LocomotionFlat01_000",
			"old/LocomotionFlat01_000_mirror",
			"old/LocomotionFlat02_000",
			"old/LocomotionFlat02_000_mirror",
			"old/LocomotionFlat02_001",
			"old/LocomotionFlat02_001_mirror",
			"old/LocomotionFlat03_000",
			"old/LocomotionFlat03_000_mirror",
			"old/LocomotionFlat04_000",
			"old/LocomotionFlat04_000_mirror",
			"old/LocomotionFlat05_000",
			"old/LocomotionFlat05_000_mirror",
			"old/LocomotionFlat06_000",
			"old/LocomotionFlat06_000_mirror",
			"old/LocomotionFlat06_001",
			"old/LocomotionFlat06_001_mirror",
			"old/LocomotionFlat07_000",
			"old/LocomotionFlat07_000_mirror",
			"old/LocomotionFlat08_000",
			"old/LocomotionFlat08_000_mirror",
			"old/LocomotionFlat08_001",
			"old/LocomotionFlat08_001_mirror",
			"old/LocomotionFlat09_000",
			"old/LocomotionFlat09_000_mirror",
			"old/LocomotionFlat10_000",
			"old/LocomotionFlat10_000_mirror",
			"old/LocomotionFlat11_000",
			"old/LocomotionFlat11_000_mirror",
			"old/LocomotionFlat12_000",
			"old/LocomotionFlat12_000_mirror"};
			*/
#if 0
			afullTrans.resize(fnames.size());
			afullVels.resize(fnames.size());
			afullAVels.resize(fnames.size());
			ajointAngles.resize(fnames.size());
			ofstream airfiles;
			ofstream badfiles;
			airfiles.open("air.txt");
			badfiles.open("bad.txt");
			afeetInAir.resize(fnames.size());
			int numTotalFrames = 0;
			for (int q = 0; q < fnames.size(); q++)
			{
				vector<vector<Transform>>& fullTrans = afullTrans[q];
				vector<vector<Vec3>>& fullVels = afullVels[q];
				vector<vector<Vec3>>& fullAVels = afullAVels[q];
				vector<vector<float>>& jointAngles = ajointAngles[q];
				fullFileName = "../../data/bvh/" + fnames[q] + "_full.state";

				//fullFileName = "../../data/bvh/LocomotionFlat12_000_full.state";
				//fullFileName = "../../data/bvh/LocomotionFlat11_000_full.state";
				//fullFileName = "../../data/bvh/LocomotionFlat07_000_full.state";
				FILE* f = fopen(fullFileName.c_str(), "rb");
				bool bad = false;
				bool air = false;
				int numFrames = 0;
				if (!f)
				{
					bad = true;
				}
				else
				{

					fread(&numFrames, 1, sizeof(int), f);
					if (numFrames == 0)
					{
						bad = true;
						fclose(f);
					}
				}
				if (!bad)
				{
					fullTrans.resize(numFrames);
					fullVels.resize(numFrames);
					fullAVels.resize(numFrames);
					cout << "Read " << numFrames << " frames of full data from " << fnames[q] << endl;

					int numTrans = 0;

					fread(&numTrans, 1, sizeof(int), f);
					int airTime = 0;
					int rightF = mjcfs[0]->bmap["right_foot"] - mjcfs[0]->firstBody;
					int leftF = mjcfs[0]->bmap["left_foot"] - mjcfs[0]->firstBody;
					afeetInAir[q].resize(numFrames, 0);
					for (int i = 0; i < numFrames; i++)
					{
						fullTrans[i].resize(numTrans);
						fullVels[i].resize(numTrans);
						fullAVels[i].resize(numTrans);
						fread(&fullTrans[i][0], sizeof(Transform), fullTrans[i].size(), f);
						fread(&fullVels[i][0], sizeof(Vec3), fullVels[i].size(), f);
						fread(&fullAVels[i][0], sizeof(Vec3), fullAVels[i].size(), f);

						for (int k = 0; k < numTrans; k++)
						{
							if ((!isfinite(fullTrans[i][k].p.x)) ||
									(!isfinite(fullTrans[i][k].p.y)) ||
									(!isfinite(fullTrans[i][k].p.z)) ||
									(!isfinite(fullVels[i][k].x)) ||
									(!isfinite(fullVels[i][k].y)) ||
									(!isfinite(fullVels[i][k].z)) ||
									(!isfinite(fullAVels[i][k].x)) ||
									(!isfinite(fullAVels[i][k].y)) ||
									(!isfinite(fullAVels[i][k].z)) ||
									(!isfinite(fullTrans[i][k].q.x)) ||
									(!isfinite(fullTrans[i][k].q.y)) ||
									(!isfinite(fullTrans[i][k].q.z)) ||
									(!isfinite(fullTrans[i][k].q.w))
							   )
							{

								bad = true;
							}
						}
						int nf = 2;
						if ((fullTrans[i][rightF].p.z < 0.15) && (fullVels[i][rightF].z < 0.2f))
						{
							nf--;
						}
						if ((fullTrans[i][leftF].p.z < 0.15) && (fullVels[i][leftF].z < 0.2f))
						{
							nf--;
						}
						afeetInAir[q][i] = nf;
						if ((fullTrans[i][rightF].p.z > 0.4f) && (fullTrans[i][leftF].p.z > 0.4f))
						{
							airTime++;
						}
						else
						{
							airTime = 0;
						}
						if (airTime > 50)
						{
							air = true;
						}
					}
					fclose(f);

					// Now propagate frames with numfeet in air == 2backward for 20 frames, to disallow it from being a start frame
					//int inAirCount = 2;
					int ct = 0;
					for (int i = numFrames - 1; i >= 0; i--)
					{
						if (afeetInAir[q][i] == 2)
						{
							ct = 20;
						}
						else
						{
							ct--;
							if (ct < 0)
							{
								ct = 0;
							}
						}
						if (ct > 0)
						{
							afeetInAir[q][i] = 2;    // Mark as in air too
						}

					}
					//string jointAnglesFileName = "../../data/bvh/LocomotionFlat11_000_joint_angles.state";
					//string jointAnglesFileName = "../../data/bvh/LocomotionFlat02_000_joint_angles.state";
					string jointAnglesFileName = "../../data/bvh/" + fnames[q] + "_joint_angles.state";
					//string jointAnglesFileName = "../../data/bvh/LocomotionFlat12_000_joint_angles.state";
					//string jointAnglesFileName = "../../data/bvh/LocomotionFlat07_000_joint_angles.state";

					jointAngles.clear();

					f = fopen(jointAnglesFileName.c_str(), "rb");
					fread(&numFrames, 1, sizeof(int), f);
					jointAngles.resize(numFrames);
					int numAngles;
					fread(&numAngles, 1, sizeof(int), f);
					for (int i = 0; i < numFrames; i++)
					{
						jointAngles[i].resize(numAngles);
						fread(&jointAngles[i][0], sizeof(float), numAngles, f);
					}
					fclose(f);
				}
				if (bad || air)
				{
					if (bad)
					{
						badfiles << fnames[q] << endl;
						cout << fullFileName << " is bad :P" << endl;
					}
					if (air)
					{
						airfiles << fnames[q] << endl;
						cout << fullFileName << " is in air :P" << endl;
					}

					fnames[q] = fnames.back();
					q--;
					fnames.pop_back();
					afullTrans.pop_back();
					afullVels.pop_back();
					afullAVels.pop_back();
					ajointAngles.pop_back();
					afeetInAir.pop_back();
				}

			}

			airfiles.close();
			badfiles.close();
			numTotalFrames = 0;
			for (int q = 0; q < (int)afullTrans.size(); q++)
			{
				numTotalFrames += afullTrans[q].size();
			}
			FILE* f = fopen("alldbx.db", "wb");
			int num = afullTrans.size();
			fwrite(&num, sizeof(int), 1, f);
			int numBD = afullTrans[0][0].size();
			int numJ = ajointAngles[0][0].size();
			fwrite(&numBD, sizeof(int), 1, f);
			fwrite(&numJ, sizeof(int), 1, f);
			for (int i = 0; i < num; i++)
			{
				int numFrames = afullTrans[i].size();
				fwrite(&numFrames, sizeof(int), 1, f);
				for (int j = 0; j < numFrames; j++)
				{
					fwrite(&afullTrans[i][j][0], sizeof(Transform), numBD, f);
					fwrite(&afullVels[i][j][0], sizeof(Vec3), numBD, f);
					fwrite(&afullAVels[i][j][0], sizeof(Vec3), numBD, f);
					fwrite(&ajointAngles[i][j][0], sizeof(float), numJ, f);
				}
				fwrite(&afeetInAir[i][0], sizeof(int), numFrames, f);
			}
			fclose(f);
#else
			/*
			FILE* f = fopen("alldb.db", "wb");
			int num = afullTrans.size();
			fwrite(&num, sizeof(int), 1, f);
			int numBD = afullTrans[0][0].size();
			int numJ = ajointAngles[0][0].size();
			fwrite(&numBD, sizeof(int), 1, f);
			fwrite(&numJ, sizeof(int), 1, f);
			for (int i = 0; i < num; i++) {
				int numFrames = afullTrans[i].size();
				fwrite(&numFrames, sizeof(int), 1, f);
				for (int j = 0; j < numFrames; j++) {
					fwrite(&afullTrans[i][j][0], sizeof(Transform), numBD, f);
					fwrite(&afullVels[i][j][0], sizeof(Vec3), numBD, f);
					fwrite(&afullAVels[i][j][0], sizeof(Vec3), numBD, f);
					fwrite(&ajointAngles[i][j][0], sizeof(float), numJ, f);
				}
				fwrite(&afeetInAir[i][0], sizeof(int), numFrames, f);
			}
			fclose(f);
			exit(0);*/
			FILE* f = fopen("alldb.db", "rb");
			int num = afullTrans.size();
			fread(&num, sizeof(int), 1, f);
			afullTrans.resize(num);
			afullVels.resize(num);
			afullAVels.resize(num);
			ajointAngles.resize(num);
			afeetInAir.resize(num);

			int numBD;
			int numJ;
			fread(&numBD, sizeof(int), 1, f);
			fread(&numJ, sizeof(int), 1, f);
			for (int i = 0; i < num; i++)
			{
				//int numFrames = afullTrans[i].size();
				int numFrames = 0;
				fread(&numFrames, sizeof(int), 1, f);
				afullTrans[i].resize(numFrames);
				afullVels[i].resize(numFrames);
				afullAVels[i].resize(numFrames);
				ajointAngles[i].resize(numFrames);
				afeetInAir[i].resize(numFrames);
				for (int j = 0; j < numFrames; j++)
				{
					afullTrans[i][j].resize(numBD);
					afullVels[i][j].resize(numBD);
					afullAVels[i][j].resize(numBD);
					ajointAngles[i][j].resize(numJ);

					fread(&afullTrans[i][j][0], sizeof(Transform), numBD, f);
					fread(&afullVels[i][j][0], sizeof(Vec3), numBD, f);
					fread(&afullAVels[i][j][0], sizeof(Vec3), numBD, f);
					fread(&ajointAngles[i][j][0], sizeof(float), numJ, f);
				}
				fread(&afeetInAir[i][0], sizeof(int), numFrames, f);
			}
			fclose(f);

#endif
			//exit(0);
//			cout << "Total number of frames is " << numTotalFrames << endl;
//			exit(0);
			// Now, let's go through and split those > mMaxEpisodeLength frames into a bunch of files
			for (int q = 0; q < (int)afullTrans.size(); q++)
			{
				if (afullTrans[q].size() >= mMaxEpisodeLength * 2)
				{
					vector<int> tfeetInAir;
					vector<vector<Transform>> tfullTrans;
					vector<vector<Vec3>> tfullVels;
					vector<vector<Vec3>> tfullAVels;
					vector<vector<float>> tjointAngles;
					int st = afullTrans[q].size() - mMaxEpisodeLength;
					for (int i = 0; i < mMaxEpisodeLength; i++)
					{
						tfeetInAir.push_back(afeetInAir[q][st + i]);
						tfullTrans.push_back(afullTrans[q][st + i]);
						tfullVels.push_back(afullVels[q][st + i]);
						tfullAVels.push_back(afullAVels[q][st + i]);
						tjointAngles.push_back(ajointAngles[q][st + i]);
					}
					afeetInAir[q].resize(st);
					afullTrans[q].resize(st);
					afullVels[q].resize(st);
					afullAVels[q].resize(st);
					ajointAngles[q].resize(st);
					afeetInAir.push_back(tfeetInAir);
					afullTrans.push_back(tfullTrans);
					afullVels.push_back(tfullVels);
					afullAVels.push_back(tfullAVels);
					ajointAngles.push_back(tjointAngles);
					q--;

				}
			}
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		}

		// Look at joint Angles
#if 0
		float adiff = acos(kPi*5.0f / 180.0f);
		int numAngles = ajointAngles[0][0].size();
		vector < vector<pair<int, int> >> transits;
		transits.resize(ajointAngles.size());
		for (int a1 = 0; a1 < ajointAngles.size(); a1++)
		{
			for (int f1 = ajointAngles[a1].size() - 1; f1 < ajointAngles[a1].size(); f1++)
			{
				int mina2, minf2;
				double minSumD = 1e10f;
				for (int a2 = 0; a2 < ajointAngles.size(); a2++)
				{
					if (a1 == a2)
					{
						continue;
					}
					for (int f2 = (a1 == a2) ? (f1 + 1) : 0; f2 < ajointAngles[a2].size(); f2++)
					{
						Vec3 up1 = GetBasisVector2(afullTrans[a1][f1][0].q);
						Vec3 up2 = GetBasisVector2(afullTrans[a2][f2][0].q);
						if (Dot(up1, up2) > adiff)
						{
							continue;    // Up vec
						}
						double sumD = 0.0;
						for (int i = 0; i < numAngles; i++)
						{
							float da = ajointAngles[a1][f1][i] - ajointAngles[a2][f2][i];
							sumD += da*da;
						}
						sumD = sqrtf(sumD / numAngles);
						if (sumD < 0.1f)
						{
							transits[a1].push_back(make_pair(a2, f2));
						}
						else
						{
							if (sumD < minSumD)
							{
								minSumD = sumD;
								mina2 = a2;
								minf2 = f2;
							}
						}
					}
				}
				if (transits[a1].size() == 0)
				{
					transits[a1].push_back(make_pair(mina2, minf2));
				}
				cout << a1 << " -- ";
				for (int q = 0; q < transits[a1].size(); q++)
				{
					cout << transits[a1][q].first << ":" << transits[a1][q].second << " ";
				}
				cout << endl;
			}
		}

		FILE* tf = fopen("transitx.inf", "wb");
		int numa = transits.size();
		fwrite(&numa, sizeof(int), 1, tf);
		for (int i = 0; i < numa; i++)
		{
			int numt = transits[i].size();
			fwrite(&numt, sizeof(int), 1, tf);
			if (numt > 0)
			{
				fwrite(&transits[i][0], sizeof(pair<int, int>), numt, tf);
			}
		}
		fclose(tf);

#else
		FILE* tf = fopen("transit.inf", "rb");
		int numa = 0;
		fread(&numa, sizeof(int), 1, tf);
		transits.resize(numa);
		for (int i = 0; i < numa; i++)
		{
			int numt = 0;
			fread(&numt, sizeof(int), 1, tf);
			transits[i].resize(numt);
			if (numt > 0)
			{
				fread(&transits[i][0], sizeof(pair<int, int>), numt, tf);
			}
		}
		fclose(tf);
#endif

		debugString.resize(mNumAgents);
		tfullTrans.resize(mNumAgents);
		tfullVels.resize(mNumAgents);
		tfullAVels.resize(mNumAgents);
		tjointAngles.resize(mNumAgents);
		useBlendAnim = true && (!allMatchPoseMode);
		for (int a = 0; a < mNumAgents; a++)
		{
			features.push_back(vector<pair<int, Transform>>());
			for (int i = 0; i < geo_joint.size(); i++)
			{
				auto p = mjcfs[a]->geoBodyPose[geo_joint[i]];
				features[a].push_back(p);
			}
		}
		const int greenMaterial = AddRenderMaterial(Vec3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, false);
		if (showTargetMocap)
		{
			tmjcfs.resize(mNumAgents);
			tmocapBDs.resize(mNumAgents);
			int tmpBody = g_buffers->rigidBodies.size();
			vector<pair<int, NvFlexRigidJointAxis>> ctrl;
			vector<float> mpower;
			for (int i = 0; i < mNumAgents; i++)
			{
				int sb = g_buffers->rigidShapes.size();
				Transform oo = agentOffset[i];
				oo.p.x += 4.0f;
				tmocapBDs[i].first = g_buffers->rigidBodies.size();
				tmjcfs[i] = new MJCFImporter(loadPath.c_str());
				tmjcfs[i]->AddPhysicsEntities(oo, ctrl, mpower, false);
				int eb = g_buffers->rigidShapes.size();
				for (int s = sb; s < eb; s++)
				{
					g_buffers->rigidShapes[s].user = UnionCast<void*>(greenMaterial);
					g_buffers->rigidShapes[s].filter = 1; // Ignore collsion, sort of9
				}
				tmocapBDs[i].second = g_buffers->rigidBodies.size();
			}
			footFlag.resize(g_buffers->rigidBodies.size());
			for (int i = tmpBody; i < (int)g_buffers->rigidBodies.size(); i++)
			{
				footFlag[i] = -1;
			}
			contact_parts_index.resize(g_buffers->rigidBodies.size(), -1);
		}
		//	for (int i = 0; i < g_buffers->rigidShapes.size(); i++) {
//			g_buffers->rigidShapes[i].filter = 1;
		//}
		if (hasTerrain)
		{
			float t = 0.0f;
			Mesh* terrain = CreateTerrain(20.0f, 20.0f, 100, 100, Vec3(0.0f,0.0f,0.0f), Vec3(0.3f, 0.75f, 0.15f),
										  1 + (int)(6 * t), 0.05f + 0.2f * (float)t);
			terrain->Transform(TranslationMatrix(Point3(0.0f, 1.0f, 0)));

			NvFlexTriangleMeshId terrainId = CreateTriangleMesh(terrain);

			NvFlexRigidShape terrainShape;
			NvFlexMakeRigidTriangleMeshShape(&terrainShape, -1, terrainId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
			terrainShape.filter = 1;
			const int whiteMaterial = AddRenderMaterial(Vec3(0.3f, 0.3f, 0.3f), 0.0f, 0.0f, false, "checker2.png");
			terrainShape.user = UnionCast<void*>(whiteMaterial);

			g_buffers->rigidShapes.push_back(terrainShape);
		}
		bkNumBody = g_buffers->rigidBodies.size();
		bkNumShape = g_buffers->rigidShapes.size();
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
				numReload--;
				if (numReload == 0)
				{
					FILE* f = fopen("fps.txt", "rt");
					if (f)
					{
						fscanf(f, "%d", &numRenderSteps);
						fclose(f);
					}
					numReload = 600;
				}
				HandleCommunication();
				ClearContactInfo();
			}
			if (doAppendTransform)
			{
				doAppendTransform = false;
				AppendTransforms();
			}
			g_buffers->rigidBodies.unmap();
			NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!
			g_buffers->rigidJoints.unmap();
			NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size()); // Need to set bodies here too!
		}
	}

	virtual void Simulate()
	{

		if (changeAnim)
		{
			changeAnim = false;
			for (int a = 0; a < mNumAgents; a++)
			{
				/*
				agentAnim[a] = rand() % afullTrans.size();
				int lf = max((int)afullTrans[agentAnim[a]].size() - 500, 38);
				int sf = 10;
				firstFrames[a] = sf;
				startFrame[a] = rand() % (lf - firstFrames[a]);
				*/
				if (useBlendAnim)
				{
					// Generate blended anim
					//int anim = rand() % afullTrans.size();
					//int anim = agentAnim[a];
					//int f = startFrame[a] + firstFrames[a];
					tfullTrans[a].resize(mMaxEpisodeLength);
					tfullVels[a].resize(mMaxEpisodeLength);
					tfullAVels[a].resize(mMaxEpisodeLength);
					tjointAngles[a].resize(mMaxEpisodeLength);
					bool first = true;
					Transform trans;
					int anum = rand() % afullTrans.size();
					int f = rand() % afullTrans[anum].size();
					startFrame[a] = firstFrames[a] = 0;
					mFrames[a] = 0;
					Transform curPose;
					NvFlexGetRigidPose(&g_buffers->rigidBodies[agentBodies[a].first], (NvFlexRigidPose*)&curPose);
					curPose = agentOffsetInv[a] * curPose;
					trans = curPose * Inverse(afullTrans[anum][f][0]);
					trans.p.z = 0.0f; // No transform in z
					Vec3 e0 = GetBasisVector0(trans.q);
					Vec3 e1 = GetBasisVector1(trans.q);
					e0.z = 0.0f;
					e1.z = 0.0f;
					e0 = Normalize(e0);
					e1 = Normalize(e1);
					Vec3 e2 = Normalize(Cross(e0, e1));
					e1 = Normalize(Cross(e2, e0));
					Matrix33 mat = Matrix33(e0, e1, e2);
					trans.q = Quat(mat);


					for (int i = 0; i < mMaxEpisodeLength; i++)
					{
						int numLimbs = afullTrans[anum][f].size();
						/*;
						tfullTrans[a][i] = afullTrans[anum][f];
						tfullVels[a][i] = afullVels[anum][f];
						tfullAVels[a][i] = afullAVels[anum][f];
						tjointAngles[a][i] = ajointAngles[anum][f];
						*/
						tfullTrans[a][i].resize(numLimbs);
						tfullVels[a][i].resize(numLimbs);
						tfullAVels[a][i].resize(numLimbs);
						int numAngles = ajointAngles[anum][f].size();
						tjointAngles[a][i].resize(numAngles);

						for (int j = 0; j < numLimbs; j++)
						{
							//tfullTrans[a][i][j] = afullTrans[anum][f][j];
							//tfullVels[a][i][j] = afullVels[anum][f][j];
							//tfullAVels[a][i][j] = afullAVels[anum][f][j];


							tfullTrans[a][i][j] = trans*afullTrans[anum][f][j];
							tfullVels[a][i][j] = Rotate(trans.q, afullVels[anum][f][j]);
							tfullAVels[a][i][j] = Rotate(trans.q, afullAVels[anum][f][j]);

						}
						for (int j = 0; j < numAngles; j++)
						{
							//tjointAngles[a][i][j] = ajointAngles[anum][f][j];
							tjointAngles[a][i][j] = ajointAngles[anum][f][j];
						}
						f++;
						if (f == afullTrans[anum].size())
						{
							if (transits[anum].size() == 0)
							{
								if (first)
								{
									//cout << "Can't transit! anim " << anim << endl;
									first = false;
								}
								f--;
							}
							else
							{
								pair<int, int> tmp = transits[anum][rand() % transits[anum].size()];
								anum = tmp.first;
								f = tmp.second;
								// Now align body
								trans = tfullTrans[a][i][0] * Inverse(afullTrans[anum][f][0]);
								trans.p.z = 0.0f; // No transform in z
								Vec3 e0 = GetBasisVector0(trans.q);
								Vec3 e1 = GetBasisVector1(trans.q);
								e0.z = 0.0f;
								e1.z = 0.0f;
								e0 = Normalize(e0);
								e1 = Normalize(e1);
								Vec3 e2 = Normalize(Cross(e0, e1));
								e1 = Normalize(Cross(e2, e0));
								Matrix33 mat = Matrix33(e0, e1, e2);
								trans.q = Quat(mat);

							}
						}
					}
					int numLimbs = afullTrans[0][0].size();


					for (int i = 1; i < mMaxEpisodeLength; i++)
					{
						for (int j = 0; j < numLimbs; j++)
						{

							tfullVels[a][i][j] = (tfullTrans[a][i][j].p - tfullTrans[a][i - 1][j].p) / g_dt;
							tfullAVels[a][i][j] = DifferentiateQuat(tfullTrans[a][i][j].q, tfullTrans[a][i - 1][j].q, 1.0f / g_dt);
						}
					}
					startFrame[a] = firstFrames[a] = 0;

				}
			}

		}
		//cout << "g_camPos = Vec3(" << g_camPos.x << ", " << g_camPos.y << ", " << g_camPos.z << ");" << endl;
		//cout << "g_camAngle = Vec3(" << g_camAngle.x << ", " << g_camAngle.y << ", " << g_camAngle.z << ");" << endl;

		// Random push to torso during training
		int push_ai = Rand(0, pushFrequency - 1);

		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			int frameNum = 0;
			int anum = agentAnim[ai];

			vector<vector<Transform>>& fullTrans = (useBlendAnim) ? tfullTrans[ai] : afullTrans[anum];
			vector<vector<Vec3>>& fullVels = (useBlendAnim) ? tfullVels[ai] : afullVels[anum];
			vector<vector<Vec3>>& fullAVels = (useBlendAnim) ? tfullAVels[ai] : afullAVels[anum];
			vector<vector<float>>& jointAngles = (useBlendAnim) ? tjointAngles[ai] : ajointAngles[anum];
			frameNum = (mFrames[ai] + startFrame[ai]) + firstFrames[ai];
			if (frameNum >= fullTrans.size())
			{
				frameNum = fullTrans.size() - 1;
			}

			float pdScale = getPDScale(ai, frameNum);
			if (showTargetMocap)
			{
				Transform tran = agentOffset[ai];
				tran.p.x += 2.0f;
				for (int i = tmocapBDs[ai].first; i < (int)tmocapBDs[ai].second; i++)
				{
					int bi = i - tmocapBDs[ai].first;
					Transform tt = tran * addedTransform[ai] * fullTrans[frameNum][bi];
					NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
					(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(tran.q, Rotate(addedTransform[ai].q, fullVels[frameNum][bi]));
					(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(tran.q, Rotate(addedTransform[ai].q, fullAVels[frameNum][bi]));
				}
			}
			float* actions = GetAction(ai);
			for (int i = 0; i < (int)ctrls[ai].size(); i++)
			{
				int qq = i;
				NvFlexRigidJoint& joint = g_buffers->rigidJoints[ctrls[ai][qq].first + 1]; // Active joint
				//joint.compliance[ctrls[ai][qq].second] = 1.0f / (5.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f)); // less

				float sc = 1.0f;
				if (useVarPDAction)
				{
					float sc = 0.5f + 0.5f*actions[i + ctrls[ai].size()];
					if (sc > 1.0f)
					{
						sc = 1.0f;
					}
					if (sc < 0.0f)
					{
						sc = 0.0f;
					}
				}
				if (sc < 1e-20)
				{
					sc = 1e-20f; // Effectively 0
				}
				if (matchPoseMode[ai])
				{
					//joint.compliance[ctrls[ai][qq].second] = 1e12f;
					//joint.damping[ctrls[ai][qq].second] = 0.0f;
					joint.compliance[ctrls[ai][qq].second] = 1.0f / (5.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f)); // more
					joint.damping[ctrls[ai][qq].second] = 100.0f;

					joint.compliance[ctrls[ai][qq].second] /= sc;
					joint.damping[ctrls[ai][qq].second] *= sc;
					if (pdScale < 1e-6f)
					{
						joint.modes[ctrls[ai][qq].second] = eNvFlexRigidJointModeFree;
					}
					else
					{
						joint.modes[ctrls[ai][qq].second] = eNvFlexRigidJointModePosition;
					}
				}
				else
				{
					joint.modes[ctrls[ai][qq].second] = eNvFlexRigidJointModePosition;
					joint.compliance[ctrls[ai][qq].second] = 1.0f / (20.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f)); // more
					joint.damping[ctrls[ai][qq].second] = 100.0f;

					joint.compliance[ctrls[ai][qq].second] /= sc;
					joint.damping[ctrls[ai][qq].second] *= sc;
				}

				if (ragdollMode)
				{
					joint.compliance[ctrls[ai][qq].second] = 1e30f;
					joint.damping[ctrls[ai][qq].second] = 0.0f;
				}
				//joint.compliance[ctrls[ai][qq].second] = 1.0f / (40.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f)); // even more
				//joint.compliance[ctrls[ai][qq].second] = 1.0f / (10.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f)); // Default
				//joint.compliance[ctrls[ai][qq].second] = 1e10f / std::max(pdScale, 1e-12f); // none

				//joint.compliance[ctrls[ai][qq].second] = 1.0f / (10.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f));
				//joint.compliance[ctrls[ai][qq].second] = 1.0f / (powerScale*motorPower[ai][i] * std::max(pdScale, 1e-12f));
				joint.targets[ctrls[ai][qq].second] = jointAngles[frameNum][i];
				if (purePDController)
				{
					float cc = actions[i];
					if (cc < -1.0f)
					{
						cc = -1.0f;
					}
					if (cc > 1.0f)
					{
						cc = 1.0f;
					}
					joint.targets[ctrls[ai][qq].second] += cc*kPi;
				}
				if (useDeltaPDController)
				{
					float cc = actions[i];
					if (cc < -1.0f)
					{
						cc = -1.0f;
					}
					if (cc > 1.0f)
					{
						cc = 1.0f;
					}
					joint.targets[ctrls[ai][qq].second] += cc*kPi;
				}
				if (limitForce)
				{
					joint.motorLimit[ctrls[ai][qq].second] = 2.0f*motorPower[ai][i];
					//joint.motorLimit[ctrls[ai][qq].second] = motorPower[ai][i];
				}
				//if (i == 20) joint.targets[ctrls[ai][qq].second] *= -1.0f;
			}
			for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second; i++)
			{
				g_buffers->rigidBodies[i].force[0] = 0.0f;
				g_buffers->rigidBodies[i].force[1] = 0.0f;
				g_buffers->rigidBodies[i].force[2] = 0.0f;
				g_buffers->rigidBodies[i].torque[0] = 0.0f;
				g_buffers->rigidBodies[i].torque[1] = 0.0f;
				g_buffers->rigidBodies[i].torque[2] = 0.0f;
			}

			if (!useDeltaPDController && !purePDController)
			{
				for (int i = 0; i < ctrls[ai].size(); i++)
				{
					float cc = actions[i];
					prevActions[ai][i] = cc;

					if (useVarPDAction)
					{
						prevActions[ai][ctrls[ai].size()+i] = cc;
					}
					if (cc < -1.0f)
					{
						cc = -1.0f;
					}
					if (cc > 1.0f)
					{
						cc = 1.0f;
					}
					NvFlexRigidJoint& j = initJoints[ctrls[ai][i].first];
					NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
					NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
					Transform& pose0 = *((Transform*)&j.pose0);
					Transform gpose;
					NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
					Transform tran = gpose*pose0;

					Vec3 axis;
					if (ctrls[ai][i].second == 0)
					{
						axis = GetBasisVector0(tran.q);
					}
					if (ctrls[ai][i].second == 1)
					{
						axis = GetBasisVector1(tran.q);
					}
					if (ctrls[ai][i].second == 2)
					{
						axis = GetBasisVector2(tran.q);
					}

					if (!isfinite(cc))
					{
						cout << "Control of " << ai << " " << i << " is not finite!\n";
					}

					Vec3 torque = axis * motorPower[ai][i] * cc * powerScale;
					if (matchPoseMode[ai])
					{
						torque *= 0.5f; // Less power for match pose mode
					}
					if (ragdollMode)
					{
						torque = Vec3(0.0f, 0.0f, 0.0f);
					}
					a0.torque[0] += torque.x;
					a0.torque[1] += torque.y;
					a0.torque[2] += torque.z;
					a1.torque[0] -= torque.x;
					a1.torque[1] -= torque.y;
					a1.torque[2] -= torque.z;
				}

			}
			if (ai % pushFrequency == push_ai && torso[ai] != -1)
			{

				//cout << "Push agent " << ai << endl;
				Transform torsoPose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[ai]], (NvFlexRigidPose*)&torsoPose);

				float z = torsoPose.p.y;
				Vec3 pushForce = Randf() * forceMag * RandomUnitVector();
				if (z > 1.f)
				{
					pushForce.z *= 0.2f;
				}
				else
				{
					pushForce.x *= 0.2f;
					pushForce.y *= 0.2f;
					pushForce.y *= 0.2f;
				}
				/*
				g_buffers->rigidBodies[torso[ai]].force[0] += pushForce.x;
				g_buffers->rigidBodies[torso[ai]].force[1] += pushForce.y;
				g_buffers->rigidBodies[torso[ai]].force[2] += pushForce.z;
				*/
				int bd = rand() % (agentBodies[ai].second - agentBodies[ai].first) + agentBodies[ai].first;
				g_buffers->rigidBodies[bd].force[0] += pushForce.x;
				g_buffers->rigidBodies[bd].force[1] += pushForce.y;
				g_buffers->rigidBodies[bd].force[2] += pushForce.z;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[bd], (NvFlexRigidPose*)&torsoPose);
				if (renderPush)
				{
					PushInfo pp;
					pp.pos = torsoPose.p;
					pp.force = pushForce;
					pp.time = 15;
					pushes.push_back(pp);
				}
			}

		}

		g_buffers->rigidBodies.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
		g_buffers->rigidJoints.unmap();
		NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
		g_frame++;
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
		NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
		g_buffers->rigidBodies.map();
		g_buffers->rigidJoints.map();

		if (clearBoxes)
		{
			//bkNumBody = g_buffers->rigidBodies.size();
			//bkNumShape = g_buffers->rigidShapes.size();

			g_buffers->rigidShapes.map();
			g_buffers->rigidShapes.resize(bkNumShape);
			g_buffers->rigidShapes.unmap();
			NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
			g_buffers->rigidBodies.resize(bkNumBody);
			clearBoxes = false;

		}
		if (throwBox)
		{
			float bscale = 0.05f + Randf()*0.075f;

			Vec3 origin, dir;
			GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

			NvFlexRigidShape box;
			NvFlexMakeRigidBoxShape(&box, g_buffers->rigidBodies.size(), bscale, bscale, bscale, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));
			box.filter = 0;
			NvFlexRigidBody body;
			float box_density = 600.0f;
			NvFlexMakeRigidBody(g_flexLib, &body, origin, Quat(), &box, &box_density, 1);

			// set initial angular velocity
			body.angularVel[0] = 0.0f;
			body.angularVel[1] = 0.01f;
			body.angularVel[2] = 0.01f;
			body.angularDamping = 0.0f;
			(Vec3&)body.linearVel = dir*20.0f;

			g_buffers->rigidBodies.push_back(body);

			g_buffers->rigidShapes.map();
			g_buffers->rigidShapes.push_back(box);
			g_buffers->rigidShapes.unmap();
			NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
			throwBox = false;
			contact_parts_index.resize(g_buffers->rigidBodies.size(), -1);
			footFlag.resize(g_buffers->rigidBodies.size(), -1);

		}
	}

	void GetShapesBounds(int start, int end, Vec3& totalLower, Vec3& totalUpper)
	{
		// calculates the union bounds of all the collision shapes in the scene
		Bounds totalBounds;

		for (int i = start; i < end; ++i)
		{
			NvFlexCollisionGeometry geo = initRigidShapes[i].geo;


			Vec3 localLower;
			Vec3 localUpper;

			GetGeometryBounds(geo, initRigidShapes[i].geoType, localLower, localUpper);
			Transform rpose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[initRigidShapes[i].body], (NvFlexRigidPose*)&rpose);
			Transform spose = rpose*(Transform&)initRigidShapes[i].pose;
			// transform local bounds to world space
			Vec3 worldLower, worldUpper;
			TransformBounds(localLower, localUpper, spose.p, spose.q, 1.0f, worldLower, worldUpper);

			totalBounds = Union(totalBounds, Bounds(worldLower, worldUpper));
		}

		totalLower = totalBounds.lower;
		totalUpper = totalBounds.upper;

	}
	virtual void KeyDown(int key)
	{
		if (key == 'n')
		{
			forceDead = mNumAgents;
			if (rcount > 0)
			{
				rcount--;
			}
		}
		if (key == ',')
		{
			forceDead = mNumAgents;
			rcount++;
		}
		if (key == 'm')
		{
			forceDead = mNumAgents;
		}
		if (key == 'b')
		{
			changeAnim = true;
		}
		if (key == 'v')
		{
			//ragdollMode = !ragdollMode;
			clearBoxes = true;
		}
		//if (key == 'x') {
		//doAppendTransform = true;
		//}
		if (key == 'x')
		{
			throwBox = true;
		}
	}

	vector<Transform> savedTrans;
	vector<Vec3> savedVels;
	vector<Vec3> savedAVels;
	void LoadTransforms()
	{
		FILE* f = fopen("savedtrans.inf", "rb");
		while (1)
		{
			Transform tt;
			Vec3 vel;
			Vec3 avel;
			if (!fread(&tt, sizeof(Transform), 1, f))
			{
				break;    // EOF
			}
			fread(&vel, sizeof(Vec3), 1, f);
			fread(&avel, sizeof(Vec3), 1, f);
			savedTrans.push_back(tt);
			savedVels.push_back(vel);
			savedAVels.push_back(avel);
		}
		fclose(f);
	}

	void AppendTransforms()
	{
		FILE* f = fopen("savedtrans.inf", "ab");

		for (int a = 0; a < mNumAgents; a++)
		{

			for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
			{
				Transform tt;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
				tt = agentOffsetInv[a] * tt;
				Vec3 vel = (Vec3&)g_buffers->rigidBodies[i].linearVel;
				vel = Rotate(agentOffsetInv[a].q, vel);
				Vec3 avel = (Vec3&)g_buffers->rigidBodies[i].angularVel;
				avel = Rotate(agentOffsetInv[a].q, avel);
				fwrite(&tt, sizeof(Transform), 1, f);
				fwrite(&vel, sizeof(Vec3), 1, f);
				fwrite(&avel, sizeof(Vec3), 1, f);
				/*
				int bi = i - agentBodies[a].first;
				Transform tt = agentOffset[a] * fullTrans[aa][bi];
				NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
				Vec3 vel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
				(Vec3&)g_buffers->rigidBodies[i].linearVel = vel;

				Vec3 avel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
				(Vec3&)g_buffers->rigidBodies[i].angularVel = avel;*/
			}
		}
		fclose(f);
	}
	int rcount;
	virtual void ResetAgent(int a)
	{
		//mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		matchPoseMode[a] = allMatchPoseMode;
		addedTransform[a] = Transform(Vec3(), Quat());

		// Randomize frame until not near both feet in air frame
		while (1)
		{
			agentAnim[a] = rand() % afullTrans.size();
			agentAnim[a] = (rcount) % afullTrans.size();
			if (a == 0)
			{
				cout << "reset with anim = " << agentAnim[a] << endl;
			}
			if (!useAllFrames)
			{
				firstFrames[a] = firstFrame;
				startFrame[a] = rand() % (lastFrame - firstFrames[a]);
			}
			else
			{
				int lf = std::min(max(((int)afullTrans[agentAnim[a]].size()) - 500, (int)38), (int)afullTrans[agentAnim[a]].size());
				int sf = std::min(10, ((int)afullTrans[agentAnim[a]].size()) - 1);
				firstFrames[a] = sf;

				if (forceLaterFrame)
				{
					startFrame[a] = lf - 30;//rand() % (lf - firstFrames[a]);
				}
				else
				{
					startFrame[a] = rand() % (lf - firstFrames[a]);
					startFrame[a] = 0;
				}

			}
			//if (afeetInAir[agentAnim[a]][startFrame[a] + firstFrames[a]] < 2) break;
			break;
		}
		ostringstream oss;

		if (useBlendAnim)
		{
			// Generate blended anim
			//int anim = rand() % afullTrans.size();
			int anim = agentAnim[a];
			int f = startFrame[a] + firstFrames[a];
			oss << "Agent " << a << " use anim " << anim << " frame " << f;
			tfullTrans[a].resize(mMaxEpisodeLength);
			tfullVels[a].resize(mMaxEpisodeLength);
			tfullAVels[a].resize(mMaxEpisodeLength);
			tjointAngles[a].resize(mMaxEpisodeLength);
			int anum = agentAnim[a];
			bool first = true;
			Transform trans;
			for (int i = 0; i < mMaxEpisodeLength; i++)
			{
				int numLimbs = afullTrans[anum][f].size();
				/*;
				tfullTrans[a][i] = afullTrans[anum][f];
				tfullVels[a][i] = afullVels[anum][f];
				tfullAVels[a][i] = afullAVels[anum][f];
				tjointAngles[a][i] = ajointAngles[anum][f];
				*/
				tfullTrans[a][i].resize(numLimbs);
				tfullVels[a][i].resize(numLimbs);
				tfullAVels[a][i].resize(numLimbs);
				int numAngles = ajointAngles[anum][f].size();
				tjointAngles[a][i].resize(numAngles);

				for (int j = 0; j < numLimbs; j++)
				{
					//tfullTrans[a][i][j] = afullTrans[anum][f][j];
					//tfullVels[a][i][j] = afullVels[anum][f][j];
					//tfullAVels[a][i][j] = afullAVels[anum][f][j];


					tfullTrans[a][i][j] = trans*afullTrans[anum][f][j];
					tfullVels[a][i][j] = Rotate(trans.q, afullVels[anum][f][j]);
					tfullAVels[a][i][j] = Rotate(trans.q, afullAVels[anum][f][j]);

				}
				for (int j = 0; j < numAngles; j++)
				{
					//tjointAngles[a][i][j] = ajointAngles[anum][f][j];
					tjointAngles[a][i][j] = ajointAngles[anum][f][j];
				}
				f++;
				if (f == afullTrans[anum].size())
				{
					if (transits[anum].size() == 0)
					{
						if (first)
						{
							//cout << "Can't transit! anim " << anim << endl;
							first = false;
						}
						f--;
					}
					else
					{
						pair<int, int> tmp = transits[anum][rand() % transits[anum].size()];
						anum = tmp.first;
						f = tmp.second;
						// Now align body
						trans = tfullTrans[a][i][0] * Inverse(afullTrans[anum][f][0]);
						trans.p.z = 0.0f; // No transform in z
						Vec3 e0 = GetBasisVector0(trans.q);
						Vec3 e1 = GetBasisVector1(trans.q);
						e0.z = 0.0f;
						e1.z = 0.0f;
						e0 = Normalize(e0);
						e1 = Normalize(e1);
						Vec3 e2 = Normalize(Cross(e0, e1));
						e1 = Normalize(Cross(e2, e0));
						Matrix33 mat = Matrix33(e0, e1, e2);
						trans.q = Quat(mat);

						oss << " -- " << anum << ":" << f << ":" << i;
					}
				}
			}
			int numLimbs = afullTrans[0][0].size();


			for (int i = 1; i < mMaxEpisodeLength; i++)
			{
				for (int j = 0; j < numLimbs; j++)
				{

					tfullVels[a][i][j] = (tfullTrans[a][i][j].p - tfullTrans[a][i-1][j].p) / g_dt;
					tfullAVels[a][i][j] = DifferentiateQuat(tfullTrans[a][i][j].q, tfullTrans[a][i-1][j].q, 1.0f / g_dt);
				}
			}
			startFrame[a] = firstFrames[a] = 0;
			oss << endl;

		}
		debugString[a] = oss.str();
		//exit(0);
		int anum = agentAnim[a];

		if (matchPoseMode[a])
		{
			// Randomize anum (starting from another random frame)
			anum = rand() % afullTrans.size();
		}
		vector<vector<Transform>>& fullTrans = (useBlendAnim) ? tfullTrans[a] : afullTrans[anum];
		vector<vector<Vec3>>& fullVels = (useBlendAnim) ? tfullVels[a] : afullVels[anum];
		vector<vector<Vec3>>& fullAVels = (useBlendAnim) ? tfullAVels[a] : afullAVels[anum];
		//vector<vector<float>>& jointAngles = ajointAngles[anum];

		for (int i = 0; i < mNumActions; i++)
		{
			prevActions[a][i] = 0.0f;
		}
		int aa = startFrame[a] + firstFrames[a];

		if (matchPoseMode[a])
		{
			aa = rand() % fullTrans.size();
		}

		if (aa >= fullTrans.size())
		{
			aa = fullTrans.size() - 1;
		}
		if ((a % 2 == 0) || (!halfRandomReset))
		{
			if ((a % 2 == 0) && (halfSavedTransform))
			{
				int numPerA = (agentBodies[a].second - agentBodies[a].first);
				int num = savedTrans.size() / numPerA;
				int start = (rand() % num)*numPerA;
				while (savedTrans[start].p.z > 1.5f)
				{
					start = (rand() % num)*numPerA;
				}
				for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
				{
					int bi = (i - agentBodies[a].first) + start;
					Transform tt = agentOffset[a] * savedTrans[bi];
					NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
					Vec3 vel = Rotate(agentOffset[a].q, savedVels[bi]);
					(Vec3&)g_buffers->rigidBodies[i].linearVel = vel;

					Vec3 avel = Rotate(agentOffset[a].q, savedAVels[bi]);
					(Vec3&)g_buffers->rigidBodies[i].angularVel = avel;
				}
			}
			else
			{

				for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
				{
					int bi = i - agentBodies[a].first;
					Transform tt = agentOffset[a] * fullTrans[aa][bi];
					NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
					Vec3 vel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
					(Vec3&)g_buffers->rigidBodies[i].linearVel = vel;

					Vec3 avel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
					(Vec3&)g_buffers->rigidBodies[i].angularVel = avel;
				}
				//mjcfs[a]->applyJointAngleNoise(jointAngleNoise, velNoise, aavelNoise);

				Vec3 lower, upper;
				GetShapesBounds(startShape[a], endShape[a], lower, upper);
				for (int i = startBody[a]; i < endBody[a]; i++)
				{
					g_buffers->rigidBodies[i].com[1] -= (lower.y);
				}
			}
			//addedTransform[a].p.y = -lower.y;
		}
		else
		{

			Transform trans = Transform(fullTrans[aa][0].p + Vec3(Randf() * 2.0f - 1.0f, Randf() * 2.0f - 1.0f, 0.0f), rpy2quat(Randf() * 2.0f * kPi, Randf() * 2.0f * kPi, Randf() * 2.0f * kPi));
			mjcfs[a]->reset(agentOffset[a] * trans, angleResetNoise, velResetNoise, angleVelResetNoise);
			Vec3 lower, upper;
			GetShapesBounds(startShape[a], endShape[a], lower, upper);
			for (int i = startBody[a]; i < endBody[a]; i++)
			{
				g_buffers->rigidBodies[i].com[1] -= lower.y;
			}

			//addedTransform[a].p.y = -lower.y;
		}
		mFarCount[a] = 0;
		int frameNumFirst = aa;
		Transform targetPose = addedTransform[a] * fullTrans[frameNumFirst][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		walkTargetX[a] = targetPose.p.x;
		walkTargetY[a] = targetPose.p.y;


		RLWalkerEnv::ResetAgent(a);
	}

	virtual void AddAgentBodiesAndJointsCtlsPowersPopulateTorsoPelvis(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		startShape[i] = g_buffers->rigidShapes.size();
		startBody[i] = g_buffers->rigidBodies.size();
		mjcfs.push_back(make_shared<MJCFImporter>(loadPath.c_str()));
		mjcfs.back()->AddPhysicsEntities(gt, ctrl, mpower, true, true);
		endShape[i] = g_buffers->rigidShapes.size();
		endBody[i] = g_buffers->rigidBodies.size();

		for (int j = startBody[i]; j < endBody[i]; j++)
		{
			//	g_buffers->rigidBodies[j].angularDamping = 100.0f;
		}
		auto torsoInd = mjcfs[i]->bmap.find("torso");
		if (torsoInd != mjcfs[i]->bmap.end())
		{
			torso[i] = mjcfs[i]->bmap["torso"];
		}

		auto pelvisInd = mjcfs[i]->bmap.find("pelvis");
		if (pelvisInd != mjcfs[i]->bmap.end())
		{
			pelvis[i] = mjcfs[i]->bmap["pelvis"];
		}
	}

	virtual void DoStats()
	{
		if (showTargetMocap)
		{
			for (int i = 0; i < mNumAgents; i++)
			{
				Vec3 sc = GetScreenCoord((Vec3&)g_buffers->rigidBodies[agentBodies[i].first].com);
				if (matchPoseMode[i])
				{
					DrawImguiString(int(sc.x), int(sc.y + 35.0f), Vec3(1, 0, 1), 0, "Match Pose");
				}

				//DrawImguiString(int(sc.x), int(sc.y + 35.0f), Vec3(1, 0, 1), 0, "%d - %f", i, lastRews[i]);
				//DrawImguiString(int(sc.x), int(sc.y + 45.0f), Vec3(1, 0, 1), 0, "%s xx %d", debugString[i].c_str(), mFrames[i]);
			}
			/*
			BeginLines(true);

			for (int i = 0; i < mNumAgents; i++)
			{
				DrawLine(g_buffers->rigidBodies[tmocapBDs[i].first].com, g_buffers->rigidBodies[agentBodies[i].first].com, Vec4(0.0f, 1.0f, 1.0f));
			}
			if (renderPush)
			{
				for (int i = 0; i < (int)pushes.size(); i++)
				{
					DrawLine(pushes[i].pos, pushes[i].pos + pushes[i].force*0.0005f, Vec4(1.0f, 0.0f, 1.0f));
					DrawLine(pushes[i].pos - Vec3(0.1f, 0.0f, 0.0f), pushes[i].pos + Vec3(0.1f, 0.0f, 0.0f), Vec4(1.0f, 1.0f, 1.0f));
					DrawLine(pushes[i].pos - Vec3(0.0f, 0.1f, 0.0f), pushes[i].pos + Vec3(0.0f, 0.1f, 0.0f), Vec4(1.0f, 1.0f, 1.0f));
					DrawLine(pushes[i].pos - Vec3(0.0f, 0.0f, 0.1f), pushes[i].pos + Vec3(0.0f, 0.0f, 0.1f), Vec4(1.0f, 1.0f, 1.0f));
					pushes[i].time--;
					if (pushes[i].time <= 0)
					{
						pushes[i] = pushes.back();
						pushes.pop_back();
						i--;
					}
				}
			}

			EndLines();
			*/
		}
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
		if (withContacts)
		{
			for (int i = 0; i < mNumAgents; i++)
			{
				for (int j = 0; j < contact_parts.size(); j++)
				{
					contact_parts_force[i][j] = Vec3(0.0f, 0.0f, 0.0f);
				}
			}
		}
		NvFlexRigidContact* ct = &(rigidContacts[0]);
		for (int i = 0; i < numContacts; ++i)
		{
			if (withContacts)
			{
				if ((ct[i].body0 >= 0) && (contact_parts_index[ct[i].body0] >= 0))
				{
					int bd = contact_parts_index[ct[i].body0] / contact_parts.size();
					int p = contact_parts_index[ct[i].body0] % contact_parts.size();

					contact_parts_force[bd][p] -= ct[i].lambda*(Vec3&)ct[i].normal;
				}
				if ((ct[i].body1 >= 0) && (contact_parts_index[ct[i].body1] >= 0))
				{
					int bd = contact_parts_index[ct[i].body1] / contact_parts.size();
					int p = contact_parts_index[ct[i].body1] % contact_parts.size();

					contact_parts_force[bd][p] += ct[i].lambda*(Vec3&)ct[i].normal;
				}
			}
			if ((ct[i].body0 >= 0) && (footFlag[ct[i].body0] >= 0) && (ct[i].lambda > 0))
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
					feetContact[ff / 2]++;
				}
			}
			if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0))
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
		// Original
		//return +2 if z > 0.78 else - 1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

		// Viktor: modified original one to enforce standing and walking high, not on knees
		// Also due to reduced electric cost bonus for living has been decreased
		/*
		if (z > 1.0)
		{
		return 1.5f;
		}
		else
		{
		return -1.f;
		}*/
		return 1.5f;// Not die because of this
	}
	float getPDScale(int a, int frameNum)
	{
		if (pureTorque)
		{
			return 0.0f;
		}
		//return 1.0f;
		//if (matchPoseMode[a]) return 0.0f;
		if (!withPDFallOff)
		{
			return 1.0f;    // Always
		}
		int anum = agentAnim[a];
		vector<vector<Transform>>& fullTrans = (useBlendAnim) ? tfullTrans[a] : afullTrans[anum];
		//vector<vector<Vec3>>& fullVels = afullVels[anum];
		//vector<vector<Vec3>>& fullAVels = afullAVels[anum];
		//vector<vector<float>>& jointAngles = ajointAngles[anum];

		Transform targetTorso = addedTransform[a] * fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		Transform cpose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
		Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;
		float posError = Length(targetTorso.p - currentTorso.p);
		Quat qE = targetTorso.q * Inverse(currentTorso.q);
		float sinHalfTheta = Length(qE.GetAxis());
		if (sinHalfTheta > 1.0f)
		{
			sinHalfTheta = 1.0f;
		}
		if (sinHalfTheta < -1.0f)
		{
			sinHalfTheta = -1.0f;
		}

		float quatError = asinf(sinHalfTheta)*2.0f;
		float pdPos = 1.0f - (posError - farStartPos) / (farEndPos - farStartPos);
		float pdQuat = 1.0f - (quatError - farStartQuat) / (farEndQuat - farStartQuat);
		float m = min(pdPos, pdQuat);
		// Position matter now
		//if (matchPoseMode[a]) {
		//	m = pdQuat;
		//}
		if (m > 1.0f)
		{
			m = 1.0f;
		}
		if (m < 0.0f)
		{
			m = 0.0f;
		}
		return m;
	}
	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		int anum = agentAnim[a];
		int frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a];

#if 0
		if (switchAnimationWhenEnd)
		{
			if (frameNum >= afullTrans[anum].size())
			{
				//cout << "agent " << a << " out of frames" << endl;
				//Run out of frame, switch to a new animation
				Transform lastTrans = afullTrans[anum].back()[features[a][0].first - mjcfs[a]->firstBody];
				agentAnim[a] = rand() % afullTrans.size();
				anum = agentAnim[a];

				//vector<vector<Vec3>>& fullVels = afullVels[anum];
				//vector<vector<Vec3>>& fullAVels = afullAVels[anum];
				//vector<vector<float>>& jointAngles = ajointAngles[anum];
				if (!useAllFrames)
				{
					firstFrames[a] = firstFrame;
					startFrame[a] = rand() % (lastFrame - firstFrames[a]);
				}
				else
				{
					int lf = max((int)fullTrans.size(), 38);
					int sf = 10;
					firstFrames[a] = sf;
					startFrame[a] = rand() % (lf - firstFrames[a]);
				}
				startFrame[a] = startFrame[a] - mFrames[a];
				frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a];
				Vec3 xLast = Rotate(lastTrans.q, Vec3(1.0f, 0.0, 0.0f));
				Vec3 xCur = Rotate(fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody].q, Vec3(1.0f, 0.0, 0.0f));
				xLast.z = 0.0f;
				xCur.z = 0.0f;
				xLast = Normalize(xLast);
				xCur = Normalize(xCur);
				Vec3 axis = Normalize(Cross(xCur, xLast));
				float angle = Dot(xLast, xCur);
				if (Dot(axis, axis) < 1e-6f)
				{
					axis = Vec3(0.0f, 0.0f, 1.0f);
					angle = 0.0f;
				}
				Vec3 tt = lastTrans.p - fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody].p;
				tt.z = 0.0f;
				addedTransform[a] = Transform(tt, QuatFromAxisAngle(axis, angle));
			}
		}
#endif
		vector<vector<Transform>>& fullTrans = (useBlendAnim) ? tfullTrans[a] : afullTrans[anum];
		vector<vector<Vec3>>& fullVels = (useBlendAnim) ? tfullVels[a] : afullVels[anum];
		vector<vector<Vec3>>& fullAVels = (useBlendAnim) ? tfullAVels[a] : afullAVels[anum];
		vector<vector<float>>& jointAngles = (useBlendAnim) ? tjointAngles[a] : ajointAngles[anum];

		if (useDifferentRewardWhenFell)
		{
			int frameNumFirst = (mFrames[a] + startFrame[a]) + firstFrames[a];
			Transform targetPose = addedTransform[a] * fullTrans[frameNumFirst][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
			walkTargetX[a] = targetPose.p.x;
			walkTargetY[a] = targetPose.p.y;
		}

		RLWalkerEnv<Transform, Vec3, Quat, Matrix33>::ExtractState(a, state, p, walkTargetDist, jointSpeeds, numJointsAtLimit, heading, upVec);
		if (matchPoseMode[a])
		{
			//state[1] = state[2] = state[3] = 0.0f;
		}



		int ct = baseNumObservations;
		if (numFramesToProvideInfo > 0)
		{
			// State:
			// Quat of torso
			// Velocity of torso
			// Angular velocity of torso
			// Relative pos of geo_pos in torso's coordinate frame
			// Future frames:
			//				 Relative Pos of target torso in current torso's coordinate frame
			//				 Relative Quat of target torso in current torso's coordinate frame
			//				 Relative Velocity of target torso in current torso's coordinate frame
			//				 Relative Angular target velocity of torso in current torso's coordinate frame
			//               Relative target pos of geo_pos in current torso's coordinate frame
			// Look at 0, 1, 4, 16, 64 frames in future
			int frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a];
			if (frameNum >= fullTrans.size())
			{
				frameNum = fullTrans.size() - 1;
			}
			//cout << "Agent " << a << " use frame " << frameNum << endl;
			Transform cpose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);

			float yaw, pitch, roll;


			Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;


			getEulerZYX(currentTorso.q, yaw, pitch, roll);
			Matrix33 mat = Matrix33(
							   Vec3(cos(-yaw), sin(-yaw), 0.0f),
							   Vec3(-sin(-yaw), cos(-yaw), 0.0f),
							   Vec3(0.0f, 0.0f, 1.0f));

			Transform icurrentTorso = Inverse(currentTorso);
			if (useRelativeCoord)
			{
				icurrentTorso.q = Quat(mat);
			}

			Vec3 currentVel = Rotate(icurrentTorso.q, TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].linearVel));
			Vec3 currentAVel = Rotate(icurrentTorso.q, TransformVector(agentOffsetInv[a], (Vec3&)g_buffers->rigidBodies[features[a][0].first].angularVel));

			if (useRelativeCoord)
			{
				state[ct++] = roll;
				state[ct++] = pitch;
				state[ct++] = 0.0f;
				state[ct++] = 0.f;
			}
			else
			{
				state[ct++] = currentTorso.q.x;
				state[ct++] = currentTorso.q.y;
				state[ct++] = currentTorso.q.z;
				state[ct++] = currentTorso.q.w;
			}

			state[ct++] = currentVel.x;
			state[ct++] = currentVel.y;
			state[ct++] = currentVel.z;

			state[ct++] = currentAVel.x;
			state[ct++] = currentAVel.y;
			state[ct++] = currentAVel.z;

			Vec3* ttt = (Vec3*)&state[ct];
			for (int i = 0; i < features[a].size(); i++)
			{
				Transform cpose;
				NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&cpose);
				Vec3 pCurrent = TransformPoint(icurrentTorso, TransformPoint(agentOffsetInv[a], TransformPoint(cpose, features[a][i].second.p)));
				state[ct++] = pCurrent.x;
				state[ct++] = pCurrent.y;
				state[ct++] = pCurrent.z;
			}

			for (int q = 0; q < numFramesToProvideInfo; q++)
			{
				if (q == 0)
				{
					frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a];
				}
				else
				{
					frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a] + (1 << (2 * (q)));
				}
				if (frameNum >= fullTrans.size())
				{
					frameNum = fullTrans.size() - 1;
				}


				Transform targetTorso = icurrentTorso*addedTransform[a] * fullTrans[frameNum][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
				Vec3 targetVel = Rotate(icurrentTorso.q, Rotate(addedTransform[a].q, fullVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]));
				Vec3 targetAVel = Rotate(icurrentTorso.q, Rotate(addedTransform[a].q, fullAVels[frameNum][features[a][0].first - mjcfs[a]->firstBody]));

				if ((matchPoseMode[a]) && (q > 0))
				{
					// zero out everything
					targetTorso.p = Vec3(0.0f, 0.0f, 0.0f);
					targetTorso.q = Quat(0.0f, 0.0f, 0.0f, 0.0f);
					targetVel = Vec3(0.0f, 0.0f, 0.0f);
					targetAVel = Vec3(0.0f, 0.0f, 0.0f);
				}
				if (matchPoseMode[a])
				{
					// zero out position, so global position doesn't matter
					//targetTorso.p = Vec3(0.0f, 0.0f, 0.0f);
				}
				state[ct++] = targetTorso.p.x;
				state[ct++] = targetTorso.p.y;
				state[ct++] = targetTorso.p.z;

				state[ct++] = targetTorso.q.x;
				state[ct++] = targetTorso.q.y;
				state[ct++] = targetTorso.q.z;
				state[ct++] = targetTorso.q.w;

				state[ct++] = targetVel.x;
				state[ct++] = targetVel.y;
				state[ct++] = targetVel.z;

				state[ct++] = targetAVel.x;
				state[ct++] = targetAVel.y;
				state[ct++] = targetAVel.z;

				//float sumError = 0.0f;
				for (int i = 0; i < features[a].size(); i++)
				{
					Vec3 pCurrent = ttt[i];
					Vec3 pTarget = TransformPoint(icurrentTorso, TransformPoint(addedTransform[a] * fullTrans[frameNum][features[a][i].first - mjcfs[a]->firstBody], features[a][i].second.p));

					if ((matchPoseMode[a]) && (q > 0))
					{
						state[ct++] = 0.0f;
						state[ct++] = 0.0f;
						state[ct++] = 0.0f;
					}
					else
					{
						state[ct++] = pTarget.x - pCurrent.x;
						state[ct++] = pTarget.y - pCurrent.x;
						state[ct++] = pTarget.z - pCurrent.x;
					}

				}
			}
			if (matchPoseMode[a])
			{
				state[ct++] = 0.0f;
				//state[ct++] = 0.0f;
				state[ct++] = getPDScale(a, frameNum);
			}
			else
			{
				state[ct++] = mFarCount[a] / maxFarItr; // When 1, die
				state[ct++] = getPDScale(a, frameNum);
			}

			if (withContacts)
			{
				for (int i = 0; i < contact_parts.size(); i++)
				{
					if (useRelativeCoord)
					{
						Vec3 cf = Rotate(icurrentTorso.q, contact_parts_force[a][i]);
						state[ct++] = cf.x;
						state[ct++] = cf.y;
						state[ct++] = cf.z;
					}
					else
					{
						// TODO: This looks wrong to me :P
						state[ct++] = contact_parts_force[a][i].x;
						state[ct++] = contact_parts_force[a][i].y;
						state[ct++] = contact_parts_force[a][i].z;
					}
				}
			}
			if ((useMatchPoseBrain) && (!allMatchPoseMode) && (matchPoseMode[a]))
			{
				state[0] += 50.0f;
				//printf("Agent %d uses match pose\n", a);
			}
		}
		if (providePreviousActions)
		{
			for (int i = 0; i < mNumActions; i++)
			{
				state[ct++] = prevActions[a][i];
			}
		}
		if (useDeltaPDController)
		{
			frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a];
			if (frameNum >= fullTrans.size())
			{
				frameNum = fullTrans.size() - 1;
			}

			for (int i = 0; i < mNumActions; i++)
			{
				state[ct++] = jointAngles[frameNum][i];
			}

		}
	}
	virtual void CenterCamera(void)
	{
		g_camPos = Vec3(0.694362, 3.07111, 7.66372);
		g_camAngle = Vec3(-0.00523596, -0.254818, 0);
	}

};
#include "rlmocapmimic.h"
