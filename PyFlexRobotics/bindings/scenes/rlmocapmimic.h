#include <sys/types.h>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "../helper/dirent.h"
#else
#include <dirent.h>
#endif
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define LOG_PATH "."
#else
#define LOG_PATH "/flex_gym_io"
#endif

class PairCompSecond 
{
	public:
		bool operator() (const pair<int, float>& a, const pair<int, float>& b) const {
			return a.second < b.second;			
		}
};

class RigidMocapMimic : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:
	vector<vector<int>> stats;
	bool statMode;
	bool testMode;
	float posWeight;
	float quatWeight;
	float velWeight;
	float avelWeight;

	float localWeight;
	float zErrorMax;
	float zWeight;
	
	float probRandomPose; // 0.0f default
	float probRandomFrame; // 0.0f default
	float probSavedPose; // 0.0f default


	int adjustFrameInterval;
	float probConsideredLearned;
	int minLearning;
	int frameFirstAdjust;

	float maxAngularVel;
	string outPath;
	int maxNumAnim;
	bool useActiveJoint;
	bool onWindows;
	int firstFrame;
	int lastFrame;

	vector<int> testAnims;
	vector<float> testAnimsProb;
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
	vector<float> agentAnimSuccessProb;

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
	void tryDumpSuccessFail()
	{
		if (g_frame - lastOutputDist >= 10000)
		{
			if (g_rank == 0) {
				char fn[200];
				sprintf(fn, "%s/failsuccess_%09d.txt", outPath.c_str(), g_frame);
				FILE* f = fopen(fn, "wt");
				fprintf(f, "Num fail = %d\n", (int)failAnims.size());
				fprintf(f, "Num success = %d\n", (int)successAnims.size());
				fprintf(f, "Fail = ");
				for (int i = 0; i < (int)failAnims.size(); i++)
				{
					fprintf(f, "%d ", (int)failAnims[i]);
				}
				fprintf(f, "\n");
				fprintf(f, "Succcess = ");
				for (int i = 0; i < (int)successAnims.size(); i++)
				{
					fprintf(f, "%d ", (int)successAnims[i]);
				}
				fprintf(f, "\n");
				fclose(f);
			}

			lastOutputDist = g_frame;
		}
	}

	virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
	{
		float posWeight = this->posWeight;
		float quatWeight = this->quatWeight;
		float velWeight = this->velWeight;
		float avelWeight = this->avelWeight;

		float localWeight = this->localWeight;
		float zErrorMax = this->zErrorMax;
		float zWeight = this->zWeight;

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
		
		if (testMode && !allMatchPoseMode) dead = false;
		if (forceDead > 0)
		{
			dead = true;
			forceDead--;
		}
		if (statMode) {
			if (dead || (mFrames[a] == mMaxEpisodeLength - 1)) {
				stats[startAnimNum[a]].push_back(mFrames[a]);
			}
		}
		if (autoAdjustDistributionNew) 
		{
			if (dead || (mFrames[a] == mMaxEpisodeLength - 3))
			{
				if (eligibleTest[a])
				{
					printf("Anim %d survive for %d len = %d\n", (int)startAnimNum[a], (int)mFrames[a], (int)afullTrans[startAnimNum[a]].size());
					// Record % length, normalize					
					//averageSurviveLength[startAnimNum[a]] = alphaSurvive*(mFrames[a] / (float)afullTrans[startAnimNum[a]].size()) + (1.0f - alphaSurvive)*averageSurviveLength[startAnimNum[a]];
					averageSurviveLength[startAnimNum[a]] = alphaSurvive*(mFrames[a] / (float)mMaxEpisodeLength) + (1.0f - alphaSurvive)*averageSurviveLength[startAnimNum[a]];
				}
			}
		}

		if (autoAdjustDistribution) 
		{
			// If get to 80%, say it's OK
			if ((!dead) && (mFrames[a] == (int)(mMaxEpisodeLength*0.8)))
			{
				// Successfully execute failed animation!
				if (eligibleTest[a])
				{
					// Eligible from moving from fail to success
					if (wasFail[a])
					{
						int newSuccessAnim = startAnimNum[a];
						bool found = false;
						for (int i = 0; i < failAnims.size(); i++)
						{
							if (failAnims[i] == newSuccessAnim)
							{
								failAnims[i] = failAnims.back();
								failAnims.pop_back();
								found = true;
								break;
							}
						}
						if (found)
						{
							printf("Anim %d executed successfully\n", newSuccessAnim);
							// Possible that it got removed already, by other agent no need to put many time
							failCount[newSuccessAnim] = 0;
							successAnims.push_back(newSuccessAnim);
							tryDumpSuccessFail();
						}
					}
					else
					{
						// Was success and keep being success

						failCount[startAnimNum[a]]--; // Subtract failure. 

					}
				}
			}
			else
			{
				// dead
				if (mFrames[a] < (int)(mMaxEpisodeLength*0.8))
				{
					// Early dead
					if (!wasFail[a])
					{
						if (eligibleTest[a])
						{
							// Was a success 
							// Need to remove from success :P						
							bool found = false;
							int newFailsAnim = startAnimNum[a];
							failCount[newFailsAnim]++;
							if (failCount[newFailsAnim] == 10) 
							//if (0)  // Never kick out once success
							{
								// If fail enough times, then allow for kicking out
								for (int i = 0; i < successAnims.size(); i++)
								{
									if (successAnims[i] == newFailsAnim)
									{
										successAnims[i] = successAnims.back();
										successAnims.pop_back();
										found = true;
										break;
									}
								}

								if (found)
								{
									// Possible that it got removed already, by other agent no need to put many time
									printf("Anim %d used to succeed but fail now\n", newFailsAnim);
									failAnims.push_back(newFailsAnim);
									tryDumpSuccessFail();
								}
							}
						}
					}
				}
			}
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
	int rcount;
	PPOLearningParams ppo_params;

	int lastOutputDist;
	bool autoAdjustDistribution;
	bool autoAdjustDistributionNew;
	vector<int> failAnims;
	vector<int> successAnims;
	vector<int> eligibleTest; // Eligible for moving from fail to success
	vector<int> startAnimNum;
	vector<int> wasFail; // Was this a fail animation?
	vector<int> failCount;

	vector<int> learningAnims;
	vector<int> learnedAnims;
	vector<int> hardAnims;
	vector<float> averageSurviveLength;
	float alphaSurvive;
	float successWeight;
	int frameProvideShiftMultiplier;
	float gainPDMimic;
	float gainPDMatchPose;
	float dampingMimic;
	float dampingMatchPose;
	bool autoResume;
	float terrainAmplitude;

	int getdir(string dir, vector<string> &files)
	{
		DIR *dp;
		struct dirent *dirp;
		if ((dp = opendir(dir.c_str())) == NULL) {
			cout << "Error(" << errno << ") opening " << dir << endl;
			return errno;
		}

		while ((dirp = readdir(dp)) != NULL) {
			files.push_back(string(dirp->d_name));
		}
		closedir(dp);
		return 0;
	}

	RigidMocapMimic()
	{
		statMode = false;
		probRandomPose = 0.0f;
		probRandomFrame = 0.0f;
		probSavedPose = 0.0;

		autoResume = true;
		testMode = false;
		posWeight = 1.0f;
		quatWeight = 1.0f;
		//float velWeight = 0.2f;
		//float avelWeight = 0.2f;
		velWeight = 1.0f; // More vel weight
		avelWeight = 1.0f;

		//float localWeight = 1.0f;
		localWeight = 3.0f; // More
		zErrorMax = 1.0f;
		zWeight = 2.0f;


		gainPDMimic = 20.0f;
		gainPDMatchPose = 5.0f;
		dampingMimic = 100.0f;
		dampingMatchPose = 100.0f;

		mNumAgents = 500;
		ppo_params.timesteps_per_batch = 200;
		adjustFrameInterval = 2000;
		probConsideredLearned = 0.7f;
		minLearning = 10;
		frameFirstAdjust = 102000;
		numFramesToProvideInfo = 4;
		frameProvideShiftMultiplier = 2;

		maxAngularVel = 64.0f;
		outPath = ".";
		LLL;

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			onWindows = true;
		#else 
			onWindows = false;
		#endif	
		lastOutputDist = -10000;
		autoAdjustDistribution = false;
		autoAdjustDistributionNew = true;
		successWeight = 0.2f;
		
		ppo_params.resume = 0;

		throwBox = false;
		hasTerrain = false;
		terrainAmplitude = 0.15f;
		if (hasTerrain)
		{
			yOffset = 1.2f;
		}

		clearBoxes = false;
		rcount = 1903;
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
		useMatchPoseBrain = false;
		useVarPDAction = false;
		jointAngleNoise = 0.2f;
		velNoise = 0.2f;
		aavelNoise = 0.2f;
		withPDFallOff = false;
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
		LLL;
		renderPush = true;
		maxFarItr = 30.0f;
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
		if (0)
		{
			FILE* f = fopen("testanims.txt", "rt");
			int an;
			float prob;
			while (fscanf(f, "%d:%f", &an, &prob) == 2) {
				testAnims.push_back(an);
				testAnimsProb.push_back(prob);
				//if (an == 2652) {
				if (an == 2662) {
					rcount = testAnims.size() - 1;
				}
			}
			fclose(f);
		}
	
		doFlagRun = false;
		loadPath = "../../data/humanoid_mod_mod_mass_new.xml";
		//loadPath = "../../data/humanoid_20_5.xml";

		LLL;

		g_numSubsteps = 4;
		//g_params.numIterations = 100;
		g_params.numIterations = 20;
		g_params.dynamicFriction = 1.0f; // 0.0
		g_params.staticFriction = 1.0f;


		g_numSubsteps = 2;
		g_params.numIterations = 6;
		g_params.numInnerIterations = 30;
		g_params.relaxationFactor = 0.75f;
		g_params.solverType = eNvFlexSolverPCR;
		g_params.contactRegularization = 1e-9f;

		//g_params.numIterations = 32; GAN4

		//		g_sceneLower = Vec3(-150.f, -250.f, -100.f);
		//g_sceneUpper = Vec3(250.f, 150.f, 100.f);
		g_sceneLower = Vec3(-7.0f);
		g_sceneUpper = Vec3(7.0f);

		g_pause = false;
		mDoLearning = true;
		numRenderSteps = 1;
		numReload = 600;
		LLL;

		numPerRow = 20;
		spacing = 50.f;

		numFeet = 2;

		//power = 0.41f; // Default
		//powerScale = 0.25f; // Less torque
		powerScale = 0.41f; // Normal torque
							//powerScale = 0.6f; // a bit more torque
							//powerScale = 0.41*1.5f; // a bit more torque
							//powerScale = 0.5f; // More power
							//powerScale = 0.41f; // More power
							//powerScale = 0.41f; // Default
							//powerScale = 0.82f; // More power
							//powerScale = 1.64f; // Even more power
							//powerScale = 1.0f; // More power
							//powerScale = 0.41f; // Default
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

		//ppo_params.optim_stepsize = 1e-4f;
		//ppo_params.optim_schedule = "constant";

		ppo_params.useGAN = false;
		//ppo_params.resume = 3457;// 6727;			
		LLL;
		ppo_params.num_timesteps = 2000000001;
		ppo_params.hid_size = 512;
		ppo_params.num_hid_layers = 2;
		ppo_params.optim_batchsize_per_agent = 64;

		ppo_params.optim_schedule = "adaptive";
		ppo_params.desired_kl = 0.01f; // 0.01f orig

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
		//ppo_params.relativeLogDir = "track_mimic_no_force_no_random_blend_relative_longer_ep_corrected_pd_action";
		//ppo_params.relativeLogDir = "track_mimic_cartwheel_init_rand_pure_torque";
		//ppo_params.relativeLogDir = "track_mimic_cartwheel_init_rand";
		//ppo_params.relativeLogDir = "track_mimic_cartwheel_init_rand_pure_torque";
		//ppo_params.relativeLogDir = "track_mimic_cartwheel_init_rand_pure_torque_pcr";
		//ppo_params.relativeLogDir = "track_mimic_cartwheel_init_rand_pcr";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_anim_good";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_less_power";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_less_torque";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_less_torque_0_only_maxfar30";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_normal_torque_0_only_maxfar30";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_less_torque_0_only_maxfar30_no_maxAng";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_normal_torque_0_only_maxfar30_no_maxAng";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_normal_torque_102_only_maxfar30_no_maxAng_blend";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_normal_torque_maxfar30_no_maxAng_blend";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_abitmore_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first10_db_blend_sca_PD_abitmore_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_db_blend_sca_PD_abitmore_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_db_blend_sca_PD_abitmore_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_novarac";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_db_blend_sca_PD_abitmore_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_novarac_80_percent_ok_fixed";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_onefail";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_tenfail";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first125_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_tenfail";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first25_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_neverfail";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_modhuman";

		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistributionNew_modhuman";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistributionNewFixed_modhuman_new";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistributionNewFixed_modhuman_new_novar";
		ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistributionNewFixed_modhuman_new_novar_fixbug";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first25_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_modhuman";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_abitm_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_modhuman";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first25_normal_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok_fixed_modhuman";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_first5_db_blend_sca_PD_abitmore_torque_maxfar30_no_maxAng_blend_autoAdjustDistribution_varac_80_percent_ok";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_less_torque_maxfar30_no_maxAng_blend";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_full_db_blend_sca_PD_normal_torque_maxfar30_no_maxAng";

		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_cartwheel_less_power";
		//ppo_params.relativeLogDir = "track_mimic_init_rand_pcr_cartwheel_less_power_more_pow_more_pd";

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
		LLL;
		useActiveJoint = false;
		maxNumAnim = 100000;
		char tmp[1000];
		char logdir[1000] = "\0";
		string configFileName = "xx";
		for (int i = 1; i < g_argc; ++i)
		{
			if (sscanf(g_argv[i], "-mimic=%s", &tmp))
			{
				configFileName = tmp;
				// Read config
				ifstream inf(tmp);
				string tmpLine;
				if (inf)
				{
					while (getline(inf, tmpLine))
					{
						cout << "Line: " << tmpLine<<endl;
						istringstream is(tmpLine);
						string field;
						is >> field;
						int t;						
						if (field == "powerScale") is >> powerScale; else							
							if (field == "useActiveJoint")
							{
								is >> t;
								useActiveJoint = t ? true : false;
							}
							else 
							if (field == "autoAdjustDistributionNew")
							{
								is >> t;
								autoAdjustDistributionNew = t ? true:false;
							} 
							else
							if (field == "maxNumAnim") is >> maxNumAnim; else
							if (field == "g_numSubsteps") is >> g_numSubsteps; else
							if (field == "numIterations") is >> g_params.numIterations; else
							if (field == "numInnerIterations") is >> g_params.numInnerIterations; else
							if (field == "relaxationFactor") is >> g_params.relaxationFactor; else
							if (field == "contactRegularization") is >> g_params.contactRegularization; else
							if (field == "maxFarItr") is >> maxFarItr; else

							if (field == "relativeLogDir") is >> ppo_params.relativeLogDir; else
							if (field == "workingDir") is >> ppo_params.workingDir; else
							if (field == "pythonPath") is >> ppo_params.pythonPath; else
							if (field == "hid_size") is >> ppo_params.hid_size; else
							if (field == "num_hid_layers") is >> ppo_params.num_hid_layers; else
							if (field == "optim_batchsize_per_agent") is >> ppo_params.optim_batchsize_per_agent; else
							if (field == "optim_schedule") is >> ppo_params.optim_schedule; else

							if (field == "desired_kl") is >> ppo_params.desired_kl; else
							if (field == "resume") is >> ppo_params.resume; else

							if (field == "adjustFrameInterval") is >> adjustFrameInterval; else
							if (field == "probConsideredLearned") is >> probConsideredLearned; else
							if (field == "minLearning") is >> minLearning; else
							if (field == "frameFirstAdjust") is >> frameFirstAdjust; else
							if (field == "mNumAgents")  is >> mNumAgents; else
							
							if (field == "timesteps_per_batch")  is >> ppo_params.timesteps_per_batch; else								

							if (field == "hid_size")  is >> ppo_params.hid_size; else
							if (field == "num_hid_layers")  is >> ppo_params.num_hid_layers; else
							if (field == "optim_batchsize_per_agent")  is >> ppo_params.optim_batchsize_per_agent; else
							if (field == "optim_schedule")  is >> ppo_params.optim_schedule; else
							if (field == "desired_kl")  is >> ppo_params.desired_kl; else

							if (field == "numFramesToProvideInfo")  is >> numFramesToProvideInfo; else
							if (field == "frameProvideShiftMultiplier")  is >> frameProvideShiftMultiplier; else
							if (field == "useVarPDAction")
							{
								is >> t;
								useVarPDAction = t ? true : false;
							} else
							if (field == "allMatchPoseMode") {
								is >> t;
								allMatchPoseMode = t ? true : false;
							} else
							if (field == "testMode") {
								is >> t;
								testMode = t ? true : false;
							}
							else
							if (field == "purePDController") {
								is >> t;
								purePDController = t ? true : false;
							}
							else
							if (field == "gainPDMimic")  is >> gainPDMimic; else
							if (field == "gainPDMatchPose")  is >> gainPDMatchPose; else
							if (field == "dampingMimic")  is >> dampingMimic; else
							if (field == "dampingMatchPose")  is >> dampingMatchPose; else

							if (field == "posWeight")  is >> posWeight; else
							if (field == "quatWeight")  is >> quatWeight; else
							if (field == "velWeight")  is >> velWeight; else
							if (field == "avelWeight")  is >> avelWeight; else
							if (field == "localWeight")  is >> localWeight; else
							if (field == "zErrorMax")  is >> zErrorMax; else
							if (field == "zWeight")  is >> zWeight; 
							else
								if (field == "autoResume") {
									is >> t;
									autoResume = t ? true : false;
								}
								else
							if (field == "halfRandomReset") {
								is >> t;
								halfRandomReset = t ? true : false;
							} else
							if (field == "halfSavedTransform") {
								is >> t;
								halfSavedTransform = t ? true : false;
							} else					
							if (field == "useMatchPoseBrain") {
								is >> t;
								useMatchPoseBrain = t ? true : false;
							}
							if (field == "pauseMocapWhenFar") {
								is >> t;
								pauseMocapWhenFar = t ? true : false;

							}
							else
							if (field == "probRandomPose")  is >> probRandomPose; else
							if (field == "probRandomFrame")  is >> probRandomFrame; else
							if (field == "probSavedPose")  is >> probSavedPose; else
							if (field == "pushFrequency") is >> pushFrequency; else
							if (field == "forceMag") is >> forceMag; else
							if (field == "yOffset") is >> yOffset; else
							if (field == "terrainAmplitude") is >> terrainAmplitude; else
							if (field == "hasTerrain") {
								is >> t;
								hasTerrain = t ? true : false;
							}							
					}
				}
			}
		}
		// ------------------ Load from YAML ----------------
		if (g_sceneJson.is_object()) {
			cout << "Has g_scene" << endl;
			powerScale = g_sceneJson.value("powerScale", powerScale);
			useActiveJoint = g_sceneJson.value("useActiveJoint", useActiveJoint);
			autoAdjustDistributionNew = g_sceneJson.value("autoAdjustDistributionNew", autoAdjustDistributionNew);
			maxNumAnim = g_sceneJson.value("maxNumAnim", maxNumAnim);
			g_numSubsteps = g_sceneJson.value("g_numSubsteps", g_numSubsteps);
			g_params.numIterations = g_sceneJson.value("numIterations", g_params.numIterations);
			g_params.numInnerIterations = g_sceneJson.value("numInnerIterations", g_params.numInnerIterations);
			g_params.relaxationFactor = g_sceneJson.value("relaxationFactor", g_params.relaxationFactor);
			g_params.contactRegularization = g_sceneJson.value("contactRegularization", g_params.contactRegularization);
			maxFarItr = g_sceneJson.value("maxFarItr", maxFarItr);

			ppo_params.relativeLogDir = g_sceneJson.value("relativeLogDir", ppo_params.relativeLogDir);
			ppo_params.workingDir = g_sceneJson.value("workingDir", ppo_params.workingDir);
			outPath = g_sceneJson.value("outPath", outPath);

			adjustFrameInterval = g_sceneJson.value("adjustFrameInterval", adjustFrameInterval);
			probConsideredLearned = g_sceneJson.value("probConsideredLearned", probConsideredLearned);
			minLearning = g_sceneJson.value("minLearning", minLearning);
			frameFirstAdjust = g_sceneJson.value("frameFirstAdjust", frameFirstAdjust);
			mNumAgents = g_sceneJson.value("mNumAgents", mNumAgents);

			numFramesToProvideInfo = g_sceneJson.value("numFramesToProvideInfo", numFramesToProvideInfo);
			frameProvideShiftMultiplier = g_sceneJson.value("frameProvideShiftMultiplier", frameProvideShiftMultiplier);

			useVarPDAction = g_sceneJson.value("useVarPDAction", useVarPDAction);
			allMatchPoseMode = g_sceneJson.value("allMatchPoseMode", allMatchPoseMode);
			testMode = g_sceneJson.value("testMode", testMode);
			frameProvideShiftMultiplier = g_sceneJson.value("frameProvideShiftMultiplier", frameProvideShiftMultiplier);

			purePDController = g_sceneJson.value("purePDController", purePDController);
			gainPDMimic = g_sceneJson.value("gainPDMimic", gainPDMimic);
			gainPDMatchPose = g_sceneJson.value("gainPDMatchPose", gainPDMatchPose);
			dampingMimic = g_sceneJson.value("dampingMimic", dampingMimic);
			dampingMatchPose = g_sceneJson.value("dampingMatchPose", dampingMatchPose);
			posWeight = g_sceneJson.value("posWeight", posWeight);
			quatWeight = g_sceneJson.value("quatWeight", quatWeight);

			velWeight = g_sceneJson.value("velWeight", velWeight);
			avelWeight = g_sceneJson.value("avelWeight", avelWeight);
			localWeight = g_sceneJson.value("localWeight", localWeight);
			zErrorMax = g_sceneJson.value("zErrorMax", zErrorMax);
			zWeight = g_sceneJson.value("zWeight", zWeight);

			autoResume = g_sceneJson.value("autoResume", autoResume);

			halfRandomReset = g_sceneJson.value("halfRandomReset", halfRandomReset);
			halfSavedTransform = g_sceneJson.value("halfSavedTransform", halfSavedTransform);
			useMatchPoseBrain = g_sceneJson.value("useMatchPoseBrain", useMatchPoseBrain);
			pauseMocapWhenFar = g_sceneJson.value("pauseMocapWhenFar", pauseMocapWhenFar);
			probRandomPose = g_sceneJson.value("probRandomPose", probRandomPose);
			probRandomFrame = g_sceneJson.value("probRandomFrame", probRandomFrame);
			probSavedPose = g_sceneJson.value("probSavedPose", probSavedPose);
			pushFrequency = g_sceneJson.value("pushFrequency", pushFrequency);
			forceMag = g_sceneJson.value("forceMag", forceMag);
			yOffset = g_sceneJson.value("yOffset", yOffset);
			terrainAmplitude = g_sceneJson.value("terrainAmplitude", terrainAmplitude);
			hasTerrain = g_sceneJson.value("hasTerrain", hasTerrain);
		}

		// ------------------------------------------

		size_t lastSlash = configFileName.find_last_of('/');
		if (lastSlash != string::npos)
		{
			configFileName = configFileName.substr(lastSlash + 1);
		}
		size_t lastBSlash = configFileName.find_last_of('\\');
		if (lastBSlash != string::npos)
		{
			configFileName = configFileName.substr(lastBSlash + 1);
		}
		size_t lastDot = configFileName.find_last_of('.');
		if (lastDot != string::npos)
		{
			configFileName = configFileName.substr(0, lastDot);
		}

		ppo_params.relativeLogDir = ppo_params.relativeLogDir + string("/out_") + string(configFileName);
		ppo_params.timesteps_per_batch = (testMode) ? 20000 : ppo_params.timesteps_per_batch;//400;
#ifdef NV_FLEX_GYM
#else
		// Need to provide explicit output path
		outPath = ppo_params.workingDir + "/" + ppo_params.relativeLogDir;		
#endif
		
		if (autoResume) 
		{
			// Try to find the latest resume point, as well as distribution
			vector<string> fnames;
			getdir(outPath, fnames);
			// Find latest log
			char pattern[500];
			sprintf(pattern, "%s-%%d", ppo_params.agent_name.c_str());
			int maxFrame = 0;

			// Look for agent save file
			for (int i = 0; i < fnames.size(); i++) 
			{
				int frame = 0;
				if (sscanf(fnames[i].c_str(), pattern, &frame)) 
				{
					if (frame > maxFrame) 
					{
						maxFrame = frame;
					}
				}
			}
			ppo_params.resume = maxFrame; // Resume here	
			printf("Auto resume at %d\n", maxFrame);
		}
//		showTargetMocap = (ppo_params.resume == 0) ? false : true;
//		mNumAgents = (ppo_params.resume == 0) ? mNumAgents : 2;
		
		showTargetMocap = testMode;
		mNumAgents = (testMode) ? 2 : mNumAgents;
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
		LLL;

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
			geo_joint = { "torso1","right_upper_arm", "right_foot", "right_hand", "left_upper_arm","left_foot", "left_hand" };
			contact_parts = { "torso", "right_hand", "right_foot", "left_hand", "left_foot" };
			contact_parts_penalty_weight = { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f };
		}

		contact_parts_force.resize(mNumAgents);
		for (int i = 0; i < mNumAgents; i++)
		{
			contact_parts_force[i].resize(contact_parts.size());
		}
		LLL;

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
		LLL;
		cout << "mNumObservations = " << mNumObservations << endl;
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
		LLL;
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
		agentAnimSuccessProb.resize(mNumAgents, 0.0f);

		firstFrames.resize(mNumAgents, 0);

		prevActions.resize(mNumAgents);
		for (int i = 0; i < mNumAgents; i++)
		{
			prevActions[i].resize(mNumActions, 0.0f);
		}
		addedTransform.resize(mNumAgents);
		lastRews.resize(mNumAgents);
		matchPoseMode.resize(mNumAgents, allMatchPoseMode);
		LLL;
		if (mDoLearning)
		{
			EnsureDirExists(ppo_params.workingDir + string("/") + ppo_params.relativeLogDir);
			/*
			char cm[5000];
			sprintf(cm, "copy ..\\..\\demo\\scenes\\rigidbvhretargettest.h c:\\new_baselines\\%s", ppo_params.relativeLogDir.c_str());
			cout << cm << endl;
			system(cm);
			*/
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
			LLL;
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
			LLL;
			FILE* f = fopen("../../data/alldb.db", "rb");
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

	/*
			if (maxNumAnim < afullTrans.size()) {
				afullTrans.resize(maxNumAnim);
				afeetInAir.resize(maxNumAnim);
				afullVels.resize(maxNumAnim);
				afullAVels.resize(maxNumAnim);
				ajointAngles.resize(maxNumAnim);			
			}
	*/
			if (maxNumAnim > afullTrans.size())
			{
				maxNumAnim = afullTrans.size();
			}
		}
		stats.resize(afullTrans.size());
		for (int i = 0; i < afullTrans.size(); i++) {
			stats[i].clear();
		}
		startAnimNum.resize(mNumAgents, -1);
		wasFail.resize(mNumAgents, -1);
		
		if (autoAdjustDistribution) 
		{
			//for (int q = 0; q < (int)afullTrans.size(); q++) 
			//for (int q = 0; q < 10; q++)
			//for (int q = 0; q < 125; q++)
			//for (int q = 0; q < 25; q++)			
			for (int q = 0; q < 5; q++)
			{
				failAnims.push_back(q);
			}
			failCount.resize(afullTrans.size(), 0);
			successAnims.clear();
			eligibleTest.resize(mNumAgents, 0);
		}
		if (autoAdjustDistributionNew)
		{
			// Initialize to everything
			for (int q = 0; q < (int)afullTrans.size(); q++)
			{
				hardAnims.push_back(q);				
			}
			learnedAnims.clear();
			learningAnims.clear();			
			eligibleTest.resize(mNumAgents, 0);
			averageSurviveLength.resize(afullTrans.size(), 0.0f);
			alphaSurvive = 0.1f;

			lastOutputDist = frameFirstAdjust - adjustFrameInterval;

			if (autoResume)
			{
				// Try to find the latest resume point, as well as distribution
				vector<string> fnames;
				getdir(outPath, fnames);
				// Find latest log
				int maxNum = 0;
				for (int i = 0; i < (int)fnames.size(); i++)
				{
					int num = 0;
					if (sscanf(fnames[i].c_str(), "hard_learning_learned.%d", &num)) {
						if (num > maxNum) {
							maxNum = num;
						}
					}
				}
				if (maxNum > 0) {
					printf("Auto resume autoAdjustDistributionNew at %d\n", maxNum);
					g_frame = maxNum;
					lastOutputDist = maxNum;
					char fn[500];
					sprintf(fn, "%s/hard_learning_learned.%09d.txt", outPath.c_str(), maxNum);
					FILE* f = fopen(fn, "rt");
					int numhard, numlearning, numlearned;
					fscanf(f, "Num hard = %d\n", &numhard);
					fscanf(f, "Num learning = %d\n", &numlearning);
					fscanf(f, "Num learned = %d\n", &numlearned);

					printf("Num hard = %d\n", numhard);
					printf("Num learning = %d\n", numlearning);
					printf("Num learned = %d\n", numlearned);
					hardAnims.resize(numhard);
					learningAnims.resize(numlearning);
					learnedAnims.resize(numlearned);
					averageSurviveLength.resize(afullTrans.size(), 0.0f);
					fscanf(f, "Hard = ");
					for (int i = 0; i < hardAnims.size(); i++)
					{
						float s = 0.0f;
						fscanf(f, "%d:%f ", &hardAnims[i], &s);
						averageSurviveLength[hardAnims[i]] = s;
					}
					fscanf(f, "\n");

					fscanf(f, "Learning = ");
					for (int i = 0; i < learningAnims.size(); i++)
					{
						float s = 0.0f;
						fscanf(f, "%d:%f ", &learningAnims[i], &s);
						averageSurviveLength[learningAnims[i]] = s;
					}
					fscanf(f, "\n");
					fscanf(f, "Learned = ");
					for (int i = 0; i < learnedAnims.size(); i++)
					{
						float s = 0.0f;
						fscanf(f, "%d:%f ", &learnedAnims[i], &s);
						averageSurviveLength[learnedAnims[i]] = s;
					}
					fscanf(f, "\n");
					fclose(f);
				}
			}
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
		FILE* tf = fopen("../../data/transit.inf", "rb");
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
				int sj = g_buffers->rigidJoints.size();
				Transform oo = agentOffset[i];
				oo.p.x += 4.0f;
				tmocapBDs[i].first = g_buffers->rigidBodies.size();
				tmjcfs[i] = new MJCFImporter(loadPath.c_str());
				tmjcfs[i]->AddPhysicsEntities(oo, ctrl, mpower, 10000.0f, false, false, true);
				int eb = g_buffers->rigidShapes.size();
				int ej = g_buffers->rigidJoints.size();
				for (int s = sb; s < eb; s++)
				{
					g_buffers->rigidShapes[s].user = UnionCast<void*>(greenMaterial);
					g_buffers->rigidShapes[s].filter = 1; // Ignore collsion, sort of9
				}
				for (int s = sj; s < ej; s++) {
					for (int j = 0; j < 6; j++) {
						if (g_buffers->rigidJoints[s].modes[j] == eNvFlexRigidJointModeLimitSpring) {
							g_buffers->rigidJoints[s].modes[j] = eNvFlexRigidJointModeLimit;
							g_buffers->rigidJoints[s].compliance[j] = 0.0f;
							g_buffers->rigidJoints[s].damping[j] = 0.0f;
						}
					}
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
			Mesh* terrain = CreateTerrain(20.0f, 20.0f, 100, 100, Vec3(0.0f, 0.0f, 0.0f), Vec3(0.3f, terrainAmplitude, 0.15f),
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
		printf("**** Init \n");
		init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
		printf("**** End Init \n");
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
					char tmp[300];
					sprintf(tmp, "%s/fps.txt", outPath.c_str());
					FILE* f = fopen(tmp, "rt");
					if (f)
					{
						fscanf(f, "%d", &numRenderSteps);
						fclose(f);
					}
					numReload = 600;
				}

#ifdef NV_FLEX_GYM
				Simulate();
				FinalizeContactInfo();
				for (int a = 0; a < mNumAgents; ++a)
				{
					PopulateState(a, &mObsBuf[a * mNumObservations]);
					if (mNumExtras > 0) PopulateExtra(a, &mExtraBuf[a * mNumExtras]);
					ComputeRewardAndDead(a, GetAction(a), &mObsBuf[a * mNumObservations], mRewBuf[a], (bool&)mDieBuf[a]);
				}
#else
				HandleCommunication();
#endif
			
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
		/*
		printf("outPath = %s\n", outPath.c_str());
		{
			char fn[500];
			sprintf(fn, "%s/test_path.%09d.txt", outPath.c_str(), g_frame);
			printf("%s\n", fn);
			FILE* f = fopen(fn, "wt");
			fprintf(f, "Test %d\n", g_frame);
			fclose(f);
		}
		*/
		if (autoAdjustDistributionNew)
		{
			if (g_frame - lastOutputDist > adjustFrameInterval) 
			{
				hardAnims.clear();
				learningAnims.clear();
				learnedAnims.clear();
				vector<pair<int, float>> aa;
				for (int i = 0; i < afullTrans.size(); i++)
				{
					aa.push_back(make_pair(i, averageSurviveLength[i]));
				}

				sort(aa.begin(), aa.end(), PairCompSecond());
				for (int i = afullTrans.size() - 1; i > 0; i--)
				{
					if (aa[i].second > probConsideredLearned)
					{
						// learned
						learnedAnims.push_back(aa[i].first);
					} else if (aa[i].second > 0.3f) {
						learningAnims.push_back(aa[i].first);
					} else {
						hardAnims.push_back(aa[i].first);
					}
				}

				// If learning anim < minLearning, take from hardAnims
				if (learningAnims.size() < minLearning)
				{
					int num = minLearning - learningAnims.size();
					if (num > hardAnims.size())
					{
						num = hardAnims.size();
					}
					for (int i = 0; i < num; i++)
					{
						learningAnims.push_back(hardAnims[i]);
					}
					if (num > 0)
					{
						hardAnims.erase(hardAnims.begin(), hardAnims.begin() + num);
					}
				}			
			
				if (g_rank == 0) {
					char fn[200];
					sprintf(fn, "%s/hard_learning_learned.%09d.txt", outPath.c_str(), g_frame);
					FILE* f = fopen(fn, "wt");
					fprintf(f, "Num hard = %d\n", (int)hardAnims.size());
					fprintf(f, "Num learning = %d\n", (int)learningAnims.size());
					fprintf(f, "Num learned = %d\n", (int)learnedAnims.size());
					fprintf(f, "Hard = ");
					for (int i = 0; i < (int)hardAnims.size(); i++)
					{
						fprintf(f, "%d:%0.2f ", (int)hardAnims[i], averageSurviveLength[hardAnims[i]]);
					}
					fprintf(f, "\n");

					fprintf(f, "Learning = ");
					for (int i = 0; i < (int)learningAnims.size(); i++)
					{
						fprintf(f, "%d:%0.2f ", (int)learningAnims[i], averageSurviveLength[learningAnims[i]]);
					}
					fprintf(f, "\n");

					fprintf(f, "Learned = ");
					for (int i = 0; i < (int)learnedAnims.size(); i++)
					{
						fprintf(f, "%d:%0.2f ", (int)learnedAnims[i], averageSurviveLength[learnedAnims[i]]);
					}
					fprintf(f, "\n");
					fclose(f);
				}
				/*
				for (int i = 0; i < averageSurviveLength.size(); i++) {
					if (averageSurviveLength[i] > 0.8f) learnedAnims.push_back(i); else
					if (averageSurviveLength[i] > 0.f) learnedAnims.push_back(i); else
				}*/
				lastOutputDist = g_frame;
			}
		}
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
					int anum = rand() % maxNumAnim;
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
					sc = 0.5f + 0.5f*actions[i + ctrls[ai].size()];
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

				//if (ai == 0) printf("%d:%0.2f ", i, sc);
				
				if (matchPoseMode[ai])
				{
					//joint.compliance[ctrls[ai][qq].second] = 1e12f;
					//joint.damping[ctrls[ai][qq].second] = 0.0f;
					joint.compliance[ctrls[ai][qq].second] = 1.0f / (gainPDMatchPose*motorPower[ai][i] * std::max(pdScale, 1e-12f)); // more
					joint.damping[ctrls[ai][qq].second] = dampingMatchPose;

					joint.compliance[ctrls[ai][qq].second] /= sc;
					//joint.damping[ctrls[ai][qq].second] *= sc;

				}
				else
				{
					joint.modes[ctrls[ai][qq].second] = eNvFlexRigidJointModePosition;
					joint.compliance[ctrls[ai][qq].second] = 1.0f / (gainPDMimic * motorPower[ai][i] * std::max(pdScale, 1e-12f));  // SCA
					//joint.compliance[ctrls[ai][qq].second] = 1.0f / (motorPower[ai][i] * std::max(pdScale, 1e-12f)); // less
					//joint.compliance[ctrls[ai][qq].second] = 1.0f / (5.0f*motorPower[ai][i] * std::max(pdScale, 1e-12f)); // more
					//joint.compliance[ctrls[ai][qq].second] = 1e30f;
					joint.damping[ctrls[ai][qq].second] = dampingMimic;
					//joint.damping[ctrls[ai][qq].second] = 0.0f;

					joint.compliance[ctrls[ai][qq].second] /= sc;
					//joint.damping[ctrls[ai][qq].second] *= sc;
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
					joint.targets[ctrls[ai][qq].second] = cc*kPi;
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
			//if (ai == 0) printf("\n");
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
						prevActions[ai][ctrls[ai].size() + i] = cc;
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

					if (!isfinite(cc))
					{
						cout << "Control of " << ai << " " << i << " is not finite!\n";
					}

					Vec3 torque = axis * motorPower[ai][i] * cc * powerScale;
					if (matchPoseMode[ai])
					{
						//torque *= 0.5f; // Less power for match pose mode
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
/*
		for (int i = 0; i < g_buffers->rigidBodies.size(); i++) {
			Vec3 tmp = (Vec3&)g_buffers->rigidBodies[i].angularVel;
			if (Length(tmp) > 60.0f) cout << "body " << i << " avel = " << Length(tmp) << endl;
		}
		*/
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
			float box_density = 4000.0f;
			NvFlexMakeRigidBody(g_flexLib, &body, origin, Quat(), &box, &box_density, 1);

			// set initial angular velocity
			body.angularVel[0] = 0.0f;
			body.angularVel[1] = 0.01f;
			body.angularVel[2] = 0.01f;
			body.angularDamping = 0.0f;
			(Vec3&)body.linearVel = dir*15.0f;

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

	Quat rpy2quat(float roll, float pitch, float yaw)
	{
		Quat q;
		// Abbreviations for the various angular functions
		float cy = cos(yaw * 0.5f);
		float sy = sin(yaw * 0.5f);
		float cr = cos(roll * 0.5f);
		float sr = sin(roll * 0.5f);
		float cp = cos(pitch * 0.5f);
		float sp = sin(pitch * 0.5f);

		q.w = cy * cr * cp + sy * sr * sp;
		q.x = cy * sr * cp - sy * cr * sp;
		q.y = cy * cr * sp + sy * sr * cp;
		q.z = sy * cr * cp - cy * sr * sp;
		return q;
		//return Quat(yaw, Vec3(0.0f, 0.0f, 1.0f))*Quat(pitch, Vec3(0.0f, 1.0f, 0.0f))*Quat(roll, Vec3(1.0f, 0.0f, 0.0f));
		//return Quat(roll, Vec3(1.0f, 0.0f, 0.0f))*Quat(pitch, Vec3(0.0f, 1.0f, 0.0f))* Quat(yaw, Vec3(0.0f, 0.0f, 1.0f));
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
		FILE* f = fopen("../../data/savedtrans.inf", "rb");
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
		FILE* f = fopen("../../data/savedtrans.inf", "ab");

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
	//int rcount;
	virtual void ResetAgent(int a)
	{
		//mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		matchPoseMode[a] = allMatchPoseMode;
		addedTransform[a] = Transform(Vec3(), Quat());

		// Randomize frame until not near both feet in air frame
		while (1)
		{
			agentAnim[a] = rand() % maxNumAnim;

			if (autoAdjustDistribution)
			{
				// Weight of success is successWeight time that of fail
				float successThreshold = successWeight*successAnims.size() / (successWeight*successAnims.size() + failAnims.size());
				if ((Randf() < successThreshold) && (successAnims.size() > 0))
				{
					agentAnim[a] = successAnims[rand() % successAnims.size()];
					wasFail[a] = false;
				}
				else
				{
					agentAnim[a] = failAnims[rand() % failAnims.size()];
					wasFail[a] = true;
				}
				startAnimNum[a] = agentAnim[a];
				eligibleTest[a] = false;
			}

			if (autoAdjustDistributionNew)
			{
				float pHard = 0.1f;
				float pLearning = 0.7f;
				float pLearned = 0.2f;
				if (learnedAnims.size() == 0) {
					pLearning += pLearned;
					pLearned = 0.0f;
				}
				if (learningAnims.size() == 0) {
					pHard += pLearning;
					pLearning = 0.0f;
				}
				if (hardAnims.size() == 0) {
					pLearning += pHard;
					pHard = 0.0f;
					if (learningAnims.size() == 0) {
						pLearned += pLearning;
						pLearning = 0.0f;
					}
				}
				// Sample
				float r = Randf();
				if (r >= 1.0f) r = 0.9999f;
				if (r < pHard) {
					agentAnim[a] = hardAnims[rand() % hardAnims.size()];
				}
				else
				if (r < pHard + pLearning) {
					agentAnim[a] = learningAnims[rand() % learningAnims.size()];
				}
				else {
					agentAnim[a] = learnedAnims[rand() % learnedAnims.size()];
				}
				startAnimNum[a] = agentAnim[a];
				eligibleTest[a] = false;
			}

			
			
			
			if (testMode && !allMatchPoseMode) {
				agentAnim[a] = (rcount) % maxNumAnim;
				if (testAnims.size() > 0) {
					int ind = rcount % testAnims.size();
					agentAnimSuccessProb[a] = testAnimsProb[ind];
					agentAnim[a] = testAnims[ind];
				}
				
				startAnimNum[a] = agentAnim[a];
				wasFail[a] = true;
			}
			
			if (statMode) {
				agentAnim[a] = (rcount) % afullTrans.size();
				startAnimNum[a] = agentAnim[a];
				rcount++;
				if (rcount % afullTrans.size() == 0) {
					char fname[5000];
					sprintf(fname, "report_%05d.txt", rcount);
					FILE* f = fopen(fname, "wt");
					for (int i = 0; i < stats.size(); i++) {
						for (int j = 0; j < stats[i].size(); j++) {
							if (j > 0) fprintf(f, " ");
							fprintf(f, "%d", stats[i][j]);
						}
						fprintf(f, "\n");
					}
					fclose(f);
				}
			}
			//agentAnim[a] = 102;
			if (a == 0)
			{
				cout << "reset with anim = " << agentAnim[a] << " len = "<< afullTrans[agentAnim[a]].size()<<endl;
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
					//startFrame[a] = rand() % (lf - firstFrames[a]);
					firstFrames[a] = 0;
					startFrame[a] = rand() % (afullTrans[agentAnim[a]].size());
					if (testMode && !allMatchPoseMode) startFrame[a] = 0;
				}
				if (statMode) {
					firstFrames[a] = 0;
					startFrame[a] = 0;
				}
				if (autoAdjustDistribution || autoAdjustDistributionNew) 
				{
					if (startFrame[a] <= afullTrans[agentAnim[a]].size() * 0.1f) 
					{
						//if (wasFail[a]) 
						//{
						eligibleTest[a] = true;
						//}
					}
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

					tfullVels[a][i][j] = (tfullTrans[a][i][j].p - tfullTrans[a][i - 1][j].p) / g_dt;
					tfullAVels[a][i][j] = DifferentiateQuat(tfullTrans[a][i][j].q, tfullTrans[a][i - 1][j].q, 1.0f / g_dt);
				}
			}
			startFrame[a] = firstFrames[a] = 0;
			oss << endl;

		}
		debugString[a] = oss.str();
		//exit(0);
		int anum = agentAnim[a];

		bool useRandomFrame = false;
		bool useSavedPose = false;
		bool useRandomPose = false;
		if (matchPoseMode[a]) {
			useRandomFrame = true;
		}
		if ((a % 2 == 0) && (halfSavedTransform))
		{
			useSavedPose = true;
		}
		if ((a % 2 == 0) && (halfRandomReset))
		{
			useRandomPose = true;
		}


		if (fabs(probRandomFrame + probRandomPose + probSavedPose) > 0.01f) 
		{
			// Non zero prob for special poses
			float f = Randf();
			if (f < probRandomFrame) 
			{
				useRandomFrame = true;
			} else
			if (f < probRandomFrame + probRandomPose) 
			{
				useRandomPose = true;
			} else
			if (f < probRandomFrame + probRandomPose + probSavedPose)
			{
				useSavedPose = true;
			}

		}

		if (useRandomFrame)
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
	
		if (useSavedPose) {
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
		} else
		if (useRandomPose) {
			Transform trans = Transform(fullTrans[aa][0].p + Vec3(Randf() * 2.0f - 1.0f, Randf() * 2.0f - 1.0f, 0.0f), rpy2quat(Randf() * 2.0f * kPi, Randf() * 2.0f * kPi, Randf() * 2.0f * kPi));
			mjcfs[a]->reset(agentOffset[a] * trans, angleResetNoise, velResetNoise, angleVelResetNoise);
			Vec3 lower, upper;
			GetShapesBounds(startShape[a], endShape[a], lower, upper);
			for (int i = startBody[a]; i < endBody[a]; i++)
			{
				g_buffers->rigidBodies[i].com[1] -= lower.y;
			}
		} else
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
		for (int i = startBody[a]; i < endBody[a]; i++)
		{
			g_buffers->rigidBodies[i].com[1] += yOffset;
		}
		mFarCount[a] = 0;
		int frameNumFirst = aa;
		Transform targetPose = addedTransform[a] * fullTrans[frameNumFirst][features[a][0].first - mjcfs[a]->firstBody] * features[a][0].second;
		walkTargetX[a] = targetPose.p.x;
		walkTargetY[a] = targetPose.p.y;


		RLWalkerEnv::ResetAgent(a);
	}

	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower)
	{
		startShape[i] = g_buffers->rigidShapes.size();
		startBody[i] = g_buffers->rigidBodies.size();
	


		mjcfs.push_back(make_shared<MJCFImporter>(loadPath.c_str()));

		int startJoint = g_buffers->rigidJoints.size();
		mjcfs.back()->AddPhysicsEntities(gt, ctrl, mpower, maxAngularVel, true, useActiveJoint);
		int endJoint = g_buffers->rigidJoints.size();
		if (useActiveJoint) 
		{
			/*
			int sj = startJoint;
			int ej = endJoint;
			for (int s = sj; s < ej; s++) {
				for (int j = 0; j < 6; j++) {
					if (g_buffers->rigidJoints[s].modes[j] == eNvFlexRigidJointModeLimitSpring) {
						//g_buffers->rigidJoints[s].modes[j] = eNvFlexRigidJointModeLimit;
						//g_buffers->rigidJoints[s].compliance[j] = 0.0f;
						//g_buffers->rigidJoints[s].damping[j] = 0.0f;
					}
				}
			}
			*/

			// Change to limit joint			
			for (int i = 0; i < ctrl.size(); i++) 
			{
				g_buffers->rigidJoints[ctrl[i].first - 1].modes[ctrl[i].second] = eNvFlexRigidJointModeLimit;
				g_buffers->rigidJoints[ctrl[i].first - 1].compliance[ctrl[i].second] = 0.0f;
				g_buffers->rigidJoints[ctrl[i].first - 1].damping[ctrl[i].second] = 0.0f;

			}
			
		}

		robotJoints.push_back({ startJoint, endJoint });
		endShape[i] = g_buffers->rigidShapes.size();
		endBody[i] = g_buffers->rigidBodies.size();


		torso.push_back(mjcfs[i]->bmap["torso"]);
		pelvis.push_back(mjcfs[i]->bmap["pelvis"]);
		head.push_back(mjcfs[i]->bmap["head"]);

		feet.push_back(mjcfs[i]->bmap["right_foot"]);
		feet.push_back(mjcfs[i]->bmap["left_foot"]);
	}

	virtual void DoStats()
	{
		if (showTargetMocap)
		{
		
#if 1
			
			for (int i = 0; i < mNumAgents; i++)
			{
				Vec3 cm(0.0, 0.0f, 0.0f);
				float totalMass = 0.0f;
				for (int j = agentBodies[i].first; j < agentBodies[i].second; j++) {
					cm += g_buffers->rigidBodies[j].mass*((Vec3&)g_buffers->rigidBodies[j].com);
					float mass = g_buffers->rigidBodies[j].mass;
					totalMass += mass;
					Vec3 sc = GetScreenCoord((Vec3&)g_buffers->rigidBodies[j].com);
					//DrawImguiString(int(sc.x), int(sc.y), Vec3(1, 0, 1), 0, "%0.2f", mass);
				}
				cm /= totalMass;
				cm.y = 0.0f;
				BeginLines(2.0f, true);
				DrawLine(cm, cm + Vec3(0.0f, 0.1f, 0.0f), Vec4(1.0f,0.0f,0.0f,1.0f));
				EndLines();
				Vec3 sc = GetScreenCoord(cm);
				//DrawImguiString(int(sc.x), int(sc.y), Vec3(1, 0, 0), 0, "cm");
				cm.y = 2.5f;
				sc = GetScreenCoord(cm);
				DrawImguiString(int(sc.x), int(sc.y), Vec3(1, 1, 0), 0, "Mimic animation %d, success prob %0.2f", startAnimNum[i], agentAnimSuccessProb[i]);
			}
#endif			
			/*
			for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
			{
				Vec3 sc = GetScreenCoord((Vec3&)g_buffers->rigidBodies[i].com);
				//DrawImguiString(int(sc.x), int(sc.y + 35.0f), Vec3(1, 0, 1), 0, "%d",i);
			}
			*/

			for (int i = 0; i < mNumAgents; i++)
			{
				Vec3 sc = GetScreenCoord((Vec3&)g_buffers->rigidBodies[agentBodies[i].first].com);
				if (matchPoseMode[i])
				{
					DrawImguiString(int(sc.x), int(sc.y + 35.0f), Vec3(1, 0, 1), 0, "Recovery");
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
					frameNum = (mFrames[a] + startFrame[a]) + firstFrames[a] + (1 << (frameProvideShiftMultiplier * (q)));
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
