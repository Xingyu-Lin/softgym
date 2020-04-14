#pragma once
#include "rigidbvhretargettest.h"
class RigidFullHumanoidDisturbanceRecoverTest : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	float farEndPos; // When to start consider as being far, PD will start to fall off
	float farEndQuat; // When to start consider as being far, PD will start to fall off
	int baseNumObservations;
	vector<int> rightFoot;
	vector<int> leftFoot;

	vector<int> footFlag;

	string fullFileName;

	bool renderPush;
	bool renderTarget;
	vector < vector<pair<int, Transform>>> features;
	vector<string> geo_joint;
	vector<int> mFarCount;
	vector<int> startShape;
	vector<int> endShape;
	vector<int> startBody;
	vector<int> endBody;
	vector<NvFlexRigidShape> initRigidShapes;
	float maxFarItr; // More than this will die
	bool withContacts; // Has magnitude of contact force at knee, arm, sholder, head, etc..
	vector<string> contact_parts;
	vector<float> contact_parts_penalty_weight;
	vector<vector<Vec3>> contact_parts_force;
	vector<int> contact_parts_index;

	vector<PushInfo> pushes;
	vector<Transform> targetPoses; // Need to sync with walk target!

	// Reward:
	//   Global pose error
	//	 Quat of torso error
	virtual void LoadRLState(FILE* f)
	{
		RLWalkerEnv::LoadRLState(f);

		LoadVec(f, rightFoot);
		LoadVec(f, leftFoot);

		LoadVec(f, footFlag);

		LoadVecVec(f, features);
		LoadVec(f, geo_joint);
		LoadVec(f, mFarCount);
		LoadVec(f, startShape);
		LoadVec(f, endShape);
		LoadVec(f, startBody);
		LoadVec(f, endBody);
		LoadVec(f,  initRigidShapes);
		LoadVec(f, contact_parts);
		LoadVec(f, contact_parts_penalty_weight);
		LoadVecVec(f, contact_parts_force);
		LoadVec(f, contact_parts_index);

		LoadVec(f, pushes);
		LoadVec(f, targetPoses); // Need to sync with walk target!

	}
	virtual void SaveRLState(FILE* f)
	{
		RLWalkerEnv::SaveRLState(f);

		SaveVec(f, rightFoot);
		SaveVec(f, leftFoot);

		SaveVec(f, footFlag);

		SaveVecVec(f, features);
		SaveVec(f, geo_joint);
		SaveVec(f, mFarCount);
		SaveVec(f, startShape);
		SaveVec(f, endShape);
		SaveVec(f, startBody);
		SaveVec(f, endBody);
		SaveVec(f, initRigidShapes);
		SaveVec(f, contact_parts);
		SaveVec(f, contact_parts_penalty_weight);
		SaveVecVec(f, contact_parts_force);
		SaveVec(f, contact_parts_index);

		SaveVec(f, pushes);
		SaveVec(f, targetPoses); // Need to sync with walk target!
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
		// Global error
		Transform targetTorso = targetPoses[a];
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
		float posWeight = 1.0f;
		float quatWeight = 1.0f;
		float op = progress;
		float zDif = 1.0f - max(targetTorso.p.z - currentTorso.p.z, 0.0f);
		float zWeight = 3.0f;
		float tmp = posWeight*(farEndPos - posError) / farEndPos + quatWeight*(farEndQuat - quatError) / farEndQuat + zWeight*zDif; // Use progress as well;
		progress += tmp;

		if ((posError > farEndPos) || (quatError > farEndQuat))
		{
			mFarCount[a]++;
			if (mFarCount[a] > maxFarItr)
			{
				dead = true;
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
		if (withContacts)
		{
			float forceMul = 1.0f / 3000.0f;
			for (int i = 0; i < contact_parts.size(); i++)
			{
//				if (a == 0) {
				//				cout << i << " " << Length(contact_parts_force[a][i]) << endl;
				//}
				progress -= Length(contact_parts_force[a][i])*contact_parts_penalty_weight[i] * forceMul;
			}

		}
		//------------------------
		float electricityCostCurrent = 0.0f;
		float sum = 0.0f;
		for (int i = 0; i < mNumActions; i++)
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
				printf("Agent %d Reward %d is infinite %f %f, pE = %f, qE = %f, sinHalfTheta=%f\n", a, i, op, tmp, posError, quatError, sinHalfTheta);
			}
			rew += rewards[i];
		}

	}

	RigidFullHumanoidDisturbanceRecoverTest()
	{
		farEndPos = 1.0f; // PD will be 0 at this point and counter will start, was 2.0
		farEndQuat = kPi*1.0f; // PD will be 0 at this point and counter will start
		maxFarItr = 180.0f;

		renderPush = true;
		renderTarget = true;

		doFlagRun = false;
		loadPath = "../../data/humanoid_mod.xml";
		withContacts = false;
		mNumAgents = 500;
		baseNumObservations = 52;
		mNumObservations = baseNumObservations;
		mNumActions = 21;
		mMaxEpisodeLength = 1000;

		//geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
		geo_joint = { "torso1","right_thigh", "right_foot","left_thigh","left_foot" };
		contact_parts = { "torso", "lwaist", "pelvis", "right_lower_arm", "right_upper_arm", "right_thigh", "right_shin", "right_foot", "left_lower_arm", "left_upper_arm", "left_thigh", "left_shin", "left_foot" };
		contact_parts_penalty_weight = { 1.0f, 0.7f, 0.7f, 0.1f, 0.2f, 0.2f, 0.1f, 0.0f,0.1f, 0.2f, 0.2f, 0.1f, 0.0f };
		contact_parts_force.resize(mNumAgents);
		for (int i = 0; i < mNumAgents; i++)
		{
			contact_parts_force[i].resize(contact_parts.size());
		}
		mNumObservations += 8;

		if (withContacts)
		{
			mNumObservations += contact_parts.size()*3;
		}
		g_numSubsteps = 4;
		g_params.numIterations = 20;
		//g_params.numIterations = 32; GAN4

		g_sceneLower = Vec3(-150.f, -250.f, -100.f);
		g_sceneUpper = Vec3(250.f, 150.f, 100.f);

		g_pause = false;
		mDoLearning = g_doLearning;
		numRenderSteps = 1;

		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);

		numPerRow = 20;
		spacing = 10.f;

		numFeet = 2;

		//powerScale = 0.41f; // Default
		powerScale = 0.25f; // Reduced power
		//powerScale = 0.5f; // More power
		initialZ = 0.9f;

		electricityCostScale = 1.8f; // Was 1.0f
		stallTorqueCostScale = 5.0f; // Was not specified

		angleResetNoise = 0.5f;
		angleVelResetNoise = 0.1f;
		velResetNoise = 0.1f;

		pushFrequency = 200;	// 200 How much steps in average per 1 kick
		forceMag = 4000.f; // 10000.0f
		//forceMag = 0.f; // 10000.0f
		targetPoses.resize(mNumAgents);
		mFarCount.clear();
		mFarCount.resize(mNumAgents, 0);
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
		initRigidShapes.resize(g_buffers->rigidShapes.size());
		for (size_t i = 0; i < initRigidShapes.size(); i++)
		{
			initRigidShapes[i] = g_buffers->rigidShapes[i];
		}


		if (mDoLearning)
		{
			PPOLearningParams ppo_params;


			ppo_params.useGAN = false;
			ppo_params.resume = 4026;//= 4664;//7107;// 6727;
			ppo_params.timesteps_per_batch = 20000;
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
			//string folder = "flexTestDisturbance_nopush";
			//string folder = "flexTestDisturbance_higher_withContacts";
			string folder = "flexTestDisturbance_higher_pow_0.25_ecost_1.8";
			//string folder = "flexTestDisturbance_withContacts";
			//string folder = "flexTestDisturbance";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02_far_end_1.0";
			//string folder = "flexTrackTargetAnglesModifiedWithReducedControlAndDisturbance_02";
			//string folder = "dummy";
			//string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";

			EnsureDirExists(ppo_params.workingDir + string("/") + folder);
			init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), folder.c_str());
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

	virtual void Simulate()
	{
		// Random push to torso during training
		int push_ai = Rand(0, pushFrequency - 1);

		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			/*
			int frameNum = 0;
			frameNum = (mFrames[ai] + startFrame[ai]) + firstFrame;
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
				joint.targets[ctrls[ai][qq].second] = jointAngles[frameNum][i];

				//if (i == 20) joint.targets[ctrls[ai][qq].second] *= -1.0f;
			}*/
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
			for (int i = 0; i < mNumActions; i++)
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
				NvFlexRigidJoint& j = initJoints[ctrls[ai][i].first];
				NvFlexRigidBody& a0 = g_buffers->rigidBodies[j.body0];
				NvFlexRigidBody& a1 = g_buffers->rigidBodies[j.body1];
				Transform& pose0 = *((Transform*)&j.pose0);
				Transform gpose;
				NvFlexGetRigidPose(&a0, (NvFlexRigidPose*)&gpose);
				Transform tran = gpose*pose0;

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
				g_buffers->rigidBodies[torso[ai]].force[0] += pushForce.x;
				g_buffers->rigidBodies[torso[ai]].force[1] += pushForce.y;
				g_buffers->rigidBodies[torso[ai]].force[2] += pushForce.z;

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
		targetPoses[a].p.x = Randf()*0.5f-0.25f;
		targetPoses[a].p.y = Randf()*0.5f-0.25f;
		targetPoses[a].p.z = 1.45f;//0.9f+Randf()*0.2f;
		targetPoses[a].q = rpy2quat(Randf()*0.1f - 0.05f, Randf()*0.1f - 0.05f, Randf() * 2.0f * kPi);//rpy2quat(Randf()*0.2f - 0.1f, Randf()*0.2f - 0.1f, Randf() * 2.0f * kPi);
		walkTargetX[a] = targetPoses[a].p.x;
		walkTargetY[a] = targetPoses[a].p.y;
		Transform trans = Transform(Vec3(0.0f,0.0f,0.0f), rpy2quat(Randf() * 2.0f * kPi, Randf() * 2.0f * kPi, Randf() * 2.0f * kPi));
		mjcfs[a]->reset(agentOffset[a]*trans, angleResetNoise, velResetNoise, angleVelResetNoise);
		Vec3 lower, upper;
		GetShapesBounds(startShape[a], endShape[a], lower, upper);
		for (int i = startBody[a]; i < endBody[a]; i++)
		{
			g_buffers->rigidBodies[i].com[1] -= lower.y;
		}

		/*
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
		mFarCount[a] = 0;
		*/
		mFarCount[a] = 0;
		RLWalkerEnv::ResetAgent(a);
	}

	virtual void DoStats()
	{
		BeginLines(true);
		if (renderTarget)
		{
			for (int i = 0; i < mNumAgents; i++)
			{
				Transform trans = agentOffset[i] * targetPoses[i];
				Vec3 x = GetBasisVector0(trans.q);
				Vec3 y = GetBasisVector1(trans.q);
				Vec3 z = GetBasisVector2(trans.q);
				DrawLine(trans.p, trans.p + x*0.5f, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
				DrawLine(trans.p, trans.p + y*0.5f, Vec4(0.0f, 1.0f, 0.0f, 1.0f));
				DrawLine(trans.p, trans.p + z*0.5f, Vec4(0.0f, 0.0f, 1.0f, 1.0f));

			}
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
					contact_parts_force[i][j] = Vec3(0.0f,0.0f,0.0f);
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

	virtual void ExtractState(int a, float* state,
							  float& p, float& walkTargetDist,
							  float* jointSpeeds, int& numJointsAtLimit,
							  float& heading, float& upVec)
	{
		RLWalkerEnv<Transform, Vec3, Quat, Matrix33>::ExtractState(a, state, p, walkTargetDist, jointSpeeds, numJointsAtLimit, heading, upVec);
		int ct = baseNumObservations;

		Transform targetTorso = targetPoses[a];
		Transform cpose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][0].first], (NvFlexRigidPose*)&cpose);
		Transform currentTorso = agentOffsetInv[a] * cpose*features[a][0].second;

		Quat itq = Inverse(currentTorso.q);
		Quat qE = targetTorso.q * itq;
		Vec3 posE = Rotate(itq, targetTorso.p - currentTorso.p);

		state[ct++] = posE.x;
		state[ct++] = posE.y;
		state[ct++] = posE.z;
		state[ct++] = qE.x;
		state[ct++] = qE.y;
		state[ct++] = qE.z;
		state[ct++] = qE.w;
		state[ct++] = mFarCount[a] / maxFarItr; // When 1, die

		if (withContacts)
		{
			for (int i = 0; i < contact_parts.size(); i++)
			{
				state[ct++] = contact_parts_force[a][i].x;
				state[ct++] = contact_parts_force[a][i].y;
				state[ct++] = contact_parts_force[a][i].z;
			}
		}
	}

};
