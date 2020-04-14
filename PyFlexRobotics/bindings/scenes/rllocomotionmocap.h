#pragma once

#include "rlbase.h"

using namespace std;
using namespace tinyxml2;


#if 0
// Reproducing run with biped
class RigidFullHumanoidMocapInitMJCF : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;

    vector<int> footFlag;

    vector<vector<Transform>> fullTrans;
    vector<vector<Vec3>> fullVels;
    vector<vector<Vec3>> fullAVels;
    string fullFileName;

    virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
    {
        float& potential = potentials[a];
        float& potentialOld = potentialsOld[a];
        float& p = ps[a];
        float& walkTargetDist = walkTargetDists[a];
        float* joint_speeds = &joint_speedss[a][0];
        int& jointsAtLimit = joints_at_limits[a];
        float& heading = headings[a];
        float& upVec = upVecs[a];

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
        /*
        float targetVel = 0.8f;
        float marginVel = 0.1f;
        float progressRewardMag = 2.0f;
        if (fabs(progress - targetVel) < marginVel) {
        progress = progressRewardMag;
        }
        else {
        float error = fabs(progress - targetVel) - marginVel;
        float errorRel = error / (targetVel - marginVel);
        progress = progressRewardMag*max(0.0f, 1.0f - error*error);
        }
        */
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
        float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0 : 0.0f); // MJCF4
        float rewards[6] =
        {
            alive,
            progress,
            electricityCostCurrent,
            jointsAtLimitCostCurrent,
            feetCollisionCostCurrent,
            heading_rew
        };

        //printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

        rew = 0.f;
        for (int i = 0; i < 6; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }

    RigidFullHumanoidMocapInitMJCF()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 1000;

        g_numSubsteps = 4;
        g_params.numIterations = 32;

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 60;

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        numPerRow = 20;
        spacing = 100.f;

        numFeet = 2;

        power = 0.41f; // Default
        initialZ = 0.9f;

        electricityCostScale = 1.f;

        angleResetNoise = 0.075f;
        angleVelResetNoise = 0.05f;
        velResetNoise = 0.05f;

        pushFrequency = 250;	// How much steps in average per 1 kick
        forceMag = 1.f;

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "HumanoidFull_160";
            ppo_params.resume = 137;
            ppo_params.timesteps_per_batch = 124;
            ppo_params.hid_size = 160;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_stepsize = 1e-3;
            ppo_params.optim_batchsize_per_agent = 16;
            ppo_params.clip_param = 0.2;
            */
            //const char* working_dir = "C:/Deep_RL/baselines/baselines/ppo1";
            //const char* python_file = "C:/Users/vmakoviychuk/AppData/Local/Continuum/Anaconda3/python.exe";
            //string folder = "HumanoidFull";

            //-------------- Nuttapong ----------------
            ppo_params.resume = 0;
            ppo_params.timesteps_per_batch = 500;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;

            const char* working_dir = "c:/baselines";
            const char* python_file = "c:/python/python.exe";
            string folder = "flex_humanoid_mocap_init_target_speed";


            fullFileName = "../../data/bvh/LocomotionFlat02_000.state";
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

            init(ppo_params, python_file, working_dir, folder.c_str());
        }
    }

    ~RigidFullHumanoidMocapInitMJCF()
    {
        if (rigidContacts)
        {
            delete rigidContacts;
        }
        if (rigidContactCount)
        {
            delete rigidContactCount;
        }
    }


    virtual void ResetAgent(int a)
    {
        //mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);

        int aa = rand() % fullTrans.size();
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
                    feet_contact[ff] = 1;
                }
                else
                {
                    // foot contact with something other than ground
                    int ff = footFlag[ct[i].body0];
                    numCollideOther[ff / 2]++;
                }
            }
            if ((ct[i].body1 >= 0) && (footFlag[ct[i].body1] >= 0) && (ct[i].lambda > 0.f))
            {
                if (ct[i].body0 < 0)
                {
                    // foot contact with ground
                    int ff = footFlag[ct[i].body1];
                    feet_contact[ff] = 1;
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
        if (z > 0.795)
        {
            return 1.5f;
        }
        else
        {
            return -1.f;
        }
    }
};

#endif

class RigidFullHumanoidMocapInitMJCF : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;

    vector<int> footFlag;

    vector<vector<Transform>> fullTrans;
    vector<vector<Vec3>> fullVels;
    vector<vector<Vec3>> fullAVels;
    string fullFileName;

    virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
    {
        float& potential = potentials[a];
        float& potentialOld = potentialsOld[a];
        float& p = ps[a];
        float& walkTargetDist = walkTargetDists[a];
        float* joint_speeds = &jointSpeeds[a][0];
        int& jointsAtLimit = jointsAtLimits[a];
        float& heading = headings[a];
        float& upVec = upVecs[a];

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
        float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4
        float rewards[6] =
        {
            alive,
            progress,
            electricityCostCurrent,
            jointsAtLimitCostCurrent,
            feetCollisionCostCurrent,
            heading_rew
        };

        //printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

        rew = 0.f;
        for (int i = 0; i < 6; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }

    RigidFullHumanoidMocapInitMJCF()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 1000;

        g_numSubsteps = 4;
        g_params.numIterations = 32;

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 1;

        numPerRow = 20;
        spacing = 10.f;

        numFeet = 2;

        powerScale = 0.41f; // Default
        initialZ = 0.9f;

        electricityCostScale = 1.f;

        angleResetNoise = 0.075f;
        angleVelResetNoise = 0.05f;
        velResetNoise = 0.05f;

        pushFrequency = 250;	// How much steps in average per 1 kick
        forceMag = 0.f;

    }
    void PrepareScene() override
    {
        ParseJsonParams(g_sceneJson);

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "HumanoidFull_160";
            ppo_params.resume = 137;
            ppo_params.timesteps_per_batch = 124;
            ppo_params.hid_size = 160;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_stepsize = 1e-3;
            ppo_params.optim_batchsize_per_agent = 16;
            ppo_params.clip_param = 0.2;
            */

            //string folder = "HumanoidMocap";

            //-------------- Nuttapong ----------------
            ppo_params.resume = 0;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
			
			ppo_params.workingDir = "c:/baselines_ex";
			ppo_params.pythonPath = "c:/python/python.exe";
            ppo_params.relativeLogDir = "flexHumanoidMocapInitX_bugFix";

            ppo_params.TryParseJson(g_sceneJson);

            fullFileName = "../../data/bvh/LocomotionFlat02_000.state";
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

            init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }
    }

    virtual void ResetAgent(int a)
    {
        //mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
		int firstFrame = 20;
		int lastFrame = 110;
        int aa = rand() % (lastFrame - firstFrame) + firstFrame;
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
                    numCollideOther[ff / 2]++;
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
};


class RigidFullHumanoidMocapInitGANMJCF : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;

    vector<int> footFlag;

    vector<vector<Transform>> fullTrans;
    vector<vector<Vec3>> fullVels;
    vector<vector<Vec3>> fullAVels;
    string fullFileName;

    // Extra information for GAN
    int numFramesForExtra;
    vector<int> extraFrontIndex;
    vector<vector<Transform>> extras;
    vector < vector<pair<int, Transform>>> features;

    int skipFrame;
    int hWindow;
    vector<string> geo_joint;
    float ax, ay, az;
    float isdx, isdy, isdz;
    virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
    {
        float& potential = potentials[a];
        float& potentialOld = potentialsOld[a];
        float& p = ps[a];
        float& walkTargetDist = walkTargetDists[a];
        float* joint_speeds = &jointSpeeds[a][0];
        int& jointsAtLimit = jointsAtLimits[a];

        float& heading = headings[a];
        float& upVec = upVecs[a];

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
        /*
        float targetVel = 0.8f;
        float marginVel = 0.1f;
        float progressRewardMag = 2.0f;
        if (fabs(progress - targetVel) < marginVel) {
        	progress = progressRewardMag;
        }
        else {
        	float error = fabs(progress - targetVel) - marginVel;
        	float errorRel = error / (targetVel - marginVel);
        	progress = progressRewardMag*max(0.0f, 1.0f - error*error);
        }
        */
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
        float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4
        float rewards[6] =
        {
            alive,
            progress,
            electricityCostCurrent,
            jointsAtLimitCostCurrent,
            feetCollisionCostCurrent,
            heading_rew
        };

        //printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

        rew = 0.f;
        for (int i = 0; i < 6; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }

    RigidFullHumanoidMocapInitGANMJCF()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 1000;
        // GAN 4-11
        //skipFrame = 3;
        //hWindow = 3;
        skipFrame = 6; // GAN 12-
        hWindow = 5;

        g_numSubsteps = 4;
        g_params.numIterations = 20;
        //g_params.numIterations = 32; GAN4

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 1;

        numPerRow = 20;
        spacing = 10.f;

        numFeet = 2;

        //power = 0.41f; // Default
        powerScale = 0.25f; // Reduced power
        initialZ = 0.9f;

        electricityCostScale = 1.f;

        angleResetNoise = 0.075f;
        angleVelResetNoise = 0.05f;
        velResetNoise = 0.05f;

        pushFrequency = 250;	// How much steps in average per 1 kick
        forceMag = 0.f;

    }
    void PrepareScene() override
    {
        ParseJsonParams(g_sceneJson);

        geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
        numFramesForExtra = hWindow * 2 * skipFrame + 1;
        mNumExtras = geo_joint.size() * 3 * (2 * hWindow + 1); // 15 pos, xyz, 11 frames

        extraFrontIndex.resize(mNumAgents);
        extras.resize(mNumAgents);
        for (int i = 0; i < mNumAgents; i++)
        {
            extras[i].resize(numFramesForExtra*geo_joint.size());
            extras[i].clear();
        }

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "HumanoidFull_160";
            ppo_params.resume = 137;
            ppo_params.timesteps_per_batch = 124;
            ppo_params.hid_size = 160;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_stepsize = 1e-3;
            ppo_params.optim_batchsize_per_agent = 16;
            ppo_params.clip_param = 0.2;
            */
            //const char* working_dir = "C:/Deep_RL/baselines/baselines/ppo1";
            //const char* python_file = "C:/Users/vmakoviychuk/AppData/Local/Continuum/Anaconda3/python.exe";
            //string folder = "HumanoidFull";

            //-------------- Nuttapong ----------------

            /*
            GAN 4
            ppo_params.useGAN = true;
            ppo_params.resume = 3569;
            ppo_params.timesteps_per_batch = 500;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 128;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-3f;
            ppo_params.gan_reward_scale = 50.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 5.0f;
            ppo_params.gan_num_epochs = 5;
            */

            //GAN 5, rerun with GAN 8looks promising
            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 3170;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-5f; // GAN 5 and 8 1e-7, GAN 9 1e-6 (unlearn), GAN 10 1e-4, GAN 11 1e-5
            ppo_params.gan_reward_scale = 10.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            ppo_params.useGAN = true;
            ppo_params.resume = 7183;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-5f;  // GAN12 1e-7, GAN 13 1e-5, GAN 14 1e-7
            ppo_params.gan_reward_scale = 0.0f; // GAN before 14 1.0f
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;

            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 2057;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-6f; // 1e-6 for gan6   1e-4 for gan7
            ppo_params.gan_reward_scale = 1.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            //ppo_params.resume_non_disc = 2670;
            char NPath[5000];

            GetCurrentDir(5000, NPath);
            cout << NPath << endl;
            //GAN 4-11
            //ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion_med";
            ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion";

            ppo_params.relativeLogDir = "flex_humanoid_mocap_init_target_speed_gan_reduced_power_1em5";
            //string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";

            ppo_params.TryParseJson(g_sceneJson);

            fullFileName = "../../data/bvh/LocomotionFlat02_000.state";
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

            f = fopen((ppo_params.mocapPath + ".dat.inf").c_str(), "rt");
            fscanf(f, "%f %f %f %f %f %f\n", &ax, &ay, &az, &isdx, &isdy, &isdz);
            fclose(f);

            init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }

        for (int a = 0; a < mNumAgents; a++)
        {
            features.push_back(vector<pair<int, Transform>>());
            for (int i = 0; i < (int)geo_joint.size(); i++)
            {
                auto p = mjcfs[a]->geoBodyPose[geo_joint[i]];
                features[a].push_back(p);
            }
        }
    }

    virtual void PopulateExtra(int a, float* extra)
    {
        bool fillAll = false;
        if (extras[a].size() == 0)
        {
            fillAll = true;
            extras[a].resize(numFramesForExtra*geo_joint.size());
        }
        int back = (extraFrontIndex[a] + numFramesForExtra - 1) % numFramesForExtra;
        // Put latest frame in the back
        for (int i = 0; i < (int)geo_joint.size(); i++)
        {
            Transform t;
            NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&t);
            t = agentOffsetInv[a] * t * features[a][i].second;
            extras[a][geo_joint.size()*back + i] = t;
        }
        // If reset, duplicate to others
        if (fillAll)
        {
            for (int k = 0; k < numFramesForExtra-1; k++)
            {
                for (int i= 0; i< geo_joint.size(); i++)
                {
                    extras[a][geo_joint.size()*k + i] = extras[a][geo_joint.size()*back + i];
                }
            }
        }
        // Compute extra info, normalized by stats loaded from file
        // Everything relative to lwaist coordinate system of frame i
        int centerIndex = (extraFrontIndex[a] + skipFrame*hWindow) % numFramesForExtra;


        Transform trans = extras[a][centerIndex*geo_joint.size()];
        Transform itrans = Inverse(trans);

        int ind = 0;
        for (int f = -hWindow; f <= hWindow; f++)
        {
            int startExtraIndex = ((centerIndex + f*skipFrame + numFramesForExtra) % numFramesForExtra)*geo_joint.size();
            for (int j = 0; j < geo_joint.size(); j++)
            {
                Vec3 pos = TransformPoint(itrans, extras[a][startExtraIndex + j].p);
                extra[ind++] = (pos.x - ax)*isdx;
                extra[ind++] = (pos.y - ay)*isdy;
                extra[ind++] = (pos.z - az)*isdz;

            }
        }
        extraFrontIndex[a] = (extraFrontIndex[a] + 1) % numFramesForExtra;
        /*
        if (a == 0) {
        	cout << "Ind = " << ind << endl;
        	FILE* f = fopen("points.txt", "wt");
        	for (int i = 0; i < ind; i++) {
        		fprintf(f, "%f ", extra[i]);
        		if (i % 3 == 2) fprintf(f, "\n");
        	}
        	fclose(f);
        }*/
    }
    virtual void ResetAgent(int a)
    {
        //mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);

        int aa = rand() % fullTrans.size();
        for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
        {
            int bi = i - agentBodies[a].first;
            Transform tt = agentOffset[a] * fullTrans[aa][bi];
            NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
            (Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
            (Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
        }


        RLWalkerEnv::ResetAgent(a);
        extras[a].clear();
        extraFrontIndex[a] = 0;
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
                    numCollideOther[ff / 2]++;
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

};

class RigidFullHumanoidMocapInitTrackMJCF : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;

    vector<int> footFlag;

    vector<vector<Transform>> fullTrans;
    vector<vector<Vec3>> fullVels;
    vector<vector<Vec3>> fullAVels;
    string fullFileName;

    // Extra information for GAN
    int numFramesForExtra;
    vector<int> extraFrontIndex;
    vector<vector<Transform>> extras;
    vector < vector<pair<int, Transform>>> features;
    int skipFrame;
    int hWindow;
    vector<string> geo_joint;
    float ax, ay, az;
    float isdx, isdy, isdz;
    vector<float> mocapData;
    vector<float> tmpExtra;
    virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
    {
        float& potential = potentials[a];
        float& potentialOld = potentialsOld[a];
        float& p = ps[a];
        float& walkTargetDist = walkTargetDists[a];
        float* joint_speeds = &jointSpeeds[a][0];
        int& jointsAtLimit = jointsAtLimits[a];

        float& heading = headings[a];
        float& upVec = upVecs[a];

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
        /*
        float targetVel = 0.8f;
        float marginVel = 0.1f;
        float progressRewardMag = 2.0f;
        if (fabs(progress - targetVel) < marginVel) {
        progress = progressRewardMag;
        }
        else {
        float error = fabs(progress - targetVel) - marginVel;
        float errorRel = error / (targetVel - marginVel);
        progress = progressRewardMag*max(0.0f, 1.0f - error*error);
        }
        */
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
        float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4

        PopulateExtra(a, &tmpExtra[0]);
        float extra = 0.0f;
        int num = mocapData.size() / mNumExtras;
        float* m = &mocapData[0];
        float* f = &tmpExtra[0];
        float mind = 1e30f;
        for (int i = 0; i < num; i++)
        {
            f = &tmpExtra[0];
            float d = 0.0;
            for (int j = 0; j < mNumExtras; j++)
            {
                float dl = *f++ - *m++;
                d += dl*dl;
            }
            if (d < mind)
            {
                mind = d;
            }
        }
        mind = sqrt(mind / mNumExtras) ;
        //cout << mind << endl;
        extra = (0.6f - mind) * 3.f;
        float rewards[6] =
        {
            alive + extra,
            progress,
            electricityCostCurrent,
            jointsAtLimitCostCurrent,
            feetCollisionCostCurrent,
            heading_rew
        };


        //printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

        rew = 0.f;
        for (int i = 0; i < 6; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }

    RigidFullHumanoidMocapInitTrackMJCF()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 1000;
        // GAN 4-11
        skipFrame = 3;
        hWindow = 3;
        //skipFrame = 6; // GAN 12-
        //hWindow = 5;

        g_numSubsteps = 4;
        g_params.numIterations = 20;
        //g_params.numIterations = 32; GAN4

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 60;

        numPerRow = 20;
        spacing = 100.f;

        numFeet = 2;

        //power = 0.41f; // Default
        powerScale = 0.25f; // Reduced power
        initialZ = 0.9f;

        electricityCostScale = 1.f;

        angleResetNoise = 0.075f;
        angleVelResetNoise = 0.05f;
        velResetNoise = 0.05f;

        pushFrequency = 250;	// How much steps in average per 1 kick
        forceMag = 0.f;
    }

    void PrepareScene() override
    {
        ParseJsonParams(g_sceneJson);

        geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
        numFramesForExtra = hWindow * 2 * skipFrame + 1;
        mNumExtras = geo_joint.size() * 3 * (2 * hWindow + 1); // 15 pos, xyz, 11 frames
        tmpExtra.resize(mNumExtras);
        extraFrontIndex.resize(mNumAgents);
        extras.resize(mNumAgents);
        for (int i = 0; i < mNumAgents; i++)
        {
            extras[i].resize(numFramesForExtra*geo_joint.size());
            extras[i].clear();
        }

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "HumanoidFull_160";
            ppo_params.resume = 137;
            ppo_params.timesteps_per_batch = 124;
            ppo_params.hid_size = 160;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_stepsize = 1e-3;
            ppo_params.optim_batchsize_per_agent = 16;
            ppo_params.clip_param = 0.2;
            */
            //const char* working_dir = "C:/Deep_RL/baselines/baselines/ppo1";
            //const char* python_file = "C:/Users/vmakoviychuk/AppData/Local/Continuum/Anaconda3/python.exe";
            //string folder = "HumanoidFull";

            //-------------- Nuttapong ----------------

            /*
            GAN 4
            ppo_params.useGAN = true;
            ppo_params.resume = 3569;
            ppo_params.timesteps_per_batch = 500;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 128;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-3f;
            ppo_params.gan_reward_scale = 50.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 5.0f;
            ppo_params.gan_num_epochs = 5;
            */

            //GAN 5, rerun with GAN 8looks promising
            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 3170;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-5f; // GAN 5 and 8 1e-7, GAN 9 1e-6 (unlearn), GAN 10 1e-4, GAN 11 1e-5
            ppo_params.gan_reward_scale = 10.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            ppo_params.useGAN = false;
            ppo_params.resume = 0;
            ppo_params.timesteps_per_batch = 200;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;

            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 2057;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-6f; // 1e-6 for gan6   1e-4 for gan7
            ppo_params.gan_reward_scale = 1.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            //ppo_params.resume_non_disc = 2670;
            char NPath[5000];

            GetCurrentDir(5000, NPath);
            cout << NPath << endl;
            //GAN 4-11
            ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion_med";
            //ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion";

            ppo_params.relativeLogDir = "flex_humanoid_mocap_init_nearest_forward_reward";
            //string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";

            ppo_params.TryParseJson(g_sceneJson);

            fullFileName = "../../data/bvh/LocomotionFlat02_000.state";
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

            f = fopen((ppo_params.mocapPath + ".dat.inf").c_str(), "rt");
            fscanf(f, "%f %f %f %f %f %f\n", &ax, &ay, &az, &isdx, &isdy, &isdz);
            fclose(f);

            ifstream inf;
            inf.open((ppo_params.mocapPath + ".dat").c_str());
            float t;
            while (inf >> t)
            {
                mocapData.push_back(t);
            }
            inf.close();
            init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }

        for (int a = 0; a < mNumAgents; a++)
        {
            features.push_back(vector<pair<int, Transform>>());
            for (int i = 0; i < (int)geo_joint.size(); i++)
            {
                auto p = mjcfs[a]->geoBodyPose[geo_joint[i]];
                features[a].push_back(p);
            }
        }
    }

    virtual void PopulateExtra(int a, float* extra)
    {
        bool fillAll = false;
        if (extras[a].size() == 0)
        {
            fillAll = true;
            extras[a].resize(numFramesForExtra*geo_joint.size());
        }
        int back = (extraFrontIndex[a] + numFramesForExtra - 1) % numFramesForExtra;
        // Put latest frame in the back
        for (int i = 0; i < (int)geo_joint.size(); i++)
        {
            Transform t;
            NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&t);
            t = agentOffsetInv[a] * t * features[a][i].second;
            extras[a][geo_joint.size()*back + i] = t;
        }
        // If reset, duplicate to others
        if (fillAll)
        {
            for (int k = 0; k < numFramesForExtra - 1; k++)
            {
                for (int i = 0; i< geo_joint.size(); i++)
                {
                    extras[a][geo_joint.size()*k + i] = extras[a][geo_joint.size()*back + i];
                }
            }
        }
        // Compute extra info, normalized by stats loaded from file
        // Everything relative to lwaist coordinate system of frame i
        int centerIndex = (extraFrontIndex[a] + skipFrame*hWindow) % numFramesForExtra;


        Transform trans = extras[a][centerIndex*geo_joint.size()];
        Transform itrans = Inverse(trans);

        int ind = 0;
        for (int f = -hWindow; f <= hWindow; f++)
        {
            int startExtraIndex = ((centerIndex + f*skipFrame + numFramesForExtra) % numFramesForExtra)*geo_joint.size();
            for (int j = 0; j < geo_joint.size(); j++)
            {
                Vec3 pos = TransformPoint(itrans, extras[a][startExtraIndex + j].p);
                extra[ind++] = (pos.x - ax)*isdx;
                extra[ind++] = (pos.y - ay)*isdy;
                extra[ind++] = (pos.z - az)*isdz;

            }
        }
        extraFrontIndex[a] = (extraFrontIndex[a] + 1) % numFramesForExtra;
        /*
        if (a == 0) {
        cout << "Ind = " << ind << endl;
        FILE* f = fopen("points.txt", "wt");
        for (int i = 0; i < ind; i++) {
        fprintf(f, "%f ", extra[i]);
        if (i % 3 == 2) fprintf(f, "\n");
        }
        fclose(f);
        }*/
    }
    virtual void ResetAgent(int a)
    {
        //mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);

        int aa = rand() % fullTrans.size();
        for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
        {
            int bi = i - agentBodies[a].first;
            Transform tt = agentOffset[a] * fullTrans[aa][bi];
            NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
            (Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
            (Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
        }


        RLWalkerEnv::ResetAgent(a);
        extras[a].clear();
        extraFrontIndex[a] = 0;
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
};


class RigidFullHumanoidMocapInitNearestAndGANBlendMJCF : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;

    vector<int> footFlag;

    vector<vector<Transform>> fullTrans;
    vector<vector<Vec3>> fullVels;
    vector<vector<Vec3>> fullAVels;
    string fullFileName;

    // Extra information for GAN
    int numFramesForExtra;
    vector<int> extraFrontIndex;
    vector<vector<Transform>> extras;
    vector < vector<pair<int, Transform>>> features;
    int skipFrame;
    int hWindow;
    vector<string> geo_joint;
    float ax, ay, az;
    float isdx, isdy, isdz;
    virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
    {
        float& potential = potentials[a];
        float& potentialOld = potentialsOld[a];
        float& p = ps[a];
        float& walkTargetDist = walkTargetDists[a];
        float* joint_speeds = &jointSpeeds[a][0];
        int& jointsAtLimit = jointsAtLimits[a];
        float& heading = headings[a];
        float& upVec = upVecs[a];

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
            float errorRel = error / (targetVel - marginVel);
            progress = progressRewardMag*max(0.0f, 1.0f - error*error);
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
        float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0f : 0.0f); // MJCF4
        float rewards[6] =
        {
            alive,
            progress,
            electricityCostCurrent,
            jointsAtLimitCostCurrent,
            feetCollisionCostCurrent,
            heading_rew
        };

        //printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

        rew = 0.f;
        for (int i = 0; i < 6; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }
    RigidFullHumanoidMocapInitNearestAndGANBlendMJCF()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 1000;
        // GAN 4-11
        skipFrame = 3;
        hWindow = 3;
        //skipFrame = 6; // GAN 12-
        //hWindow = 5;

        g_numSubsteps = 4;
        g_params.numIterations = 20;
        //g_params.numIterations = 32; GAN4

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 1;

        numPerRow = 20;
        spacing = 10;

        numFeet = 2;

        //power = 0.41f; // Default
        powerScale = 0.25f; // Reduced power
        initialZ = 0.9f;

        electricityCostScale = 1.f;

        angleResetNoise = 0.075f;
        angleVelResetNoise = 0.05f;
        velResetNoise = 0.05f;

        pushFrequency = 250;	// How much steps in average per 1 kick
        forceMag = 0.f;

    }

    void PrepareScene() override
    {
        ParseJsonParams(g_sceneJson);

        //geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
        geo_joint = { "torso1","right_thigh", "right_foot","left_thigh","left_foot" };

        numFramesForExtra = hWindow * 2 * skipFrame + 1;
        mNumExtras = geo_joint.size() * 3 * (2 * hWindow + 1); // 15 pos, xyz, 11 frames

        extraFrontIndex.resize(mNumAgents);
        extras.resize(mNumAgents);
        for (int i = 0; i < mNumAgents; i++)
        {
            extras[i].resize(numFramesForExtra*geo_joint.size());
            extras[i].clear();
        }

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "HumanoidFull_160";
            ppo_params.resume = 137;
            ppo_params.timesteps_per_batch = 124;
            ppo_params.hid_size = 160;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_stepsize = 1e-3;
            ppo_params.optim_batchsize_per_agent = 16;
            ppo_params.clip_param = 0.2;
            */
            //const char* working_dir = "C:/Deep_RL/baselines/baselines/ppo1";
            //const char* python_file = "C:/Users/vmakoviychuk/AppData/Local/Continuum/Anaconda3/python.exe";
            //string folder = "HumanoidFull";

            //-------------- Nuttapong ----------------

            /*
            GAN 4
            ppo_params.useGAN = true;
            ppo_params.resume = 3569;
            ppo_params.timesteps_per_batch = 500;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 128;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-3f;
            ppo_params.gan_reward_scale = 50.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 5.0f;
            ppo_params.gan_num_epochs = 5;
            */

            //GAN 5, rerun with GAN 8looks promising
            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 3170;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-5f; // GAN 5 and 8 1e-7, GAN 9 1e-6 (unlearn), GAN 10 1e-4, GAN 11 1e-5
            ppo_params.gan_reward_scale = 10.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            ppo_params.useGAN = true;
            ppo_params.useDistance = false;
            ppo_params.useBlend = false;
            ppo_params.resume = 1851;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 4;
            ppo_params.gan_num_hid_layers = 0;
            ppo_params.gan_learning_rate = 1e-4f;  // GAN12 1e-7, GAN 13 1e-5, GAN 14 1e-7
            ppo_params.gan_reward_scale = 10.0f; // GAN before 14 1.0f
            ppo_params.gan_reward_to_retrain_discriminator = 2.0f;
            ppo_params.gan_num_epochs = 1;

            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 2057;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-6f; // 1e-6 for gan6   1e-4 for gan7
            ppo_params.gan_reward_scale = 1.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            //ppo_params.resume_non_disc = 2670;
            char NPath[5000];

            GetCurrentDir(5000, NPath);
            cout << NPath << endl;
            //GAN 4-11
            //ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion_med";
            ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion_med_reduced";

            //string folder = "flex_humanoid_mocap_init_target_speed_gan_reduced_power_1em5";
            //string folder = "flex_humanoid_mocap_init_python_gan_blend_learn_1e-6_rew_scale_1";
            //string folder = "flex_humanoid_mocap_init_fast_nogan_reduced_power_1em5";
            ppo_params.relativeLogDir = "flex_humanoid_mocap_init_python_gan_reduced_features_linear_net_rew_scale_10_retrain_2";

            ppo_params.TryParseJson(g_sceneJson);

            fullFileName = "../../data/bvh/LocomotionFlat02_000.state";
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

            f = fopen((ppo_params.mocapPath + ".dat.inf").c_str(), "rt");
            fscanf(f, "%f %f %f %f %f %f\n", &ax, &ay, &az, &isdx, &isdy, &isdz);
            fclose(f);

            init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }

        for (int a = 0; a < mNumAgents; a++)
        {
            features.push_back(vector<pair<int, Transform>>());
            for (int i = 0; i < (int)geo_joint.size(); i++)
            {
                auto p = mjcfs[a]->geoBodyPose[geo_joint[i]];
                features[a].push_back(p);
            }
        }
    }

    virtual void PopulateExtra(int a, float* extra)
    {
        bool fillAll = false;
        if (extras[a].size() == 0)
        {
            fillAll = true;
            extras[a].resize(numFramesForExtra*geo_joint.size());
        }
        int back = (extraFrontIndex[a] + numFramesForExtra - 1) % numFramesForExtra;
        // Put latest frame in the back
        for (int i = 0; i < (int)geo_joint.size(); i++)
        {
            Transform t;
            NvFlexGetRigidPose(&g_buffers->rigidBodies[features[a][i].first], (NvFlexRigidPose*)&t);
            t = agentOffsetInv[a] * t * features[a][i].second;
            extras[a][geo_joint.size()*back + i] = t;
        }
        // If reset, duplicate to others
        if (fillAll)
        {
            for (int k = 0; k < numFramesForExtra - 1; k++)
            {
                for (int i = 0; i< geo_joint.size(); i++)
                {
                    extras[a][geo_joint.size()*k + i] = extras[a][geo_joint.size()*back + i];
                }
            }
        }
        // Compute extra info, normalized by stats loaded from file
        // Everything relative to lwaist coordinate system of frame i
        int centerIndex = (extraFrontIndex[a] + skipFrame*hWindow) % numFramesForExtra;


        Transform trans = extras[a][centerIndex*geo_joint.size()];
        Transform itrans = Inverse(trans);

        int ind = 0;
        for (int f = -hWindow; f <= hWindow; f++)
        {
            int startExtraIndex = ((centerIndex + f*skipFrame + numFramesForExtra) % numFramesForExtra)*geo_joint.size();
            for (int j = 0; j < geo_joint.size(); j++)
            {
                Vec3 pos = TransformPoint(itrans, extras[a][startExtraIndex + j].p);
                extra[ind++] = (pos.x - ax)*isdx;
                extra[ind++] = (pos.y - ay)*isdy;
                extra[ind++] = (pos.z - az)*isdz;

            }
        }
        extraFrontIndex[a] = (extraFrontIndex[a] + 1) % numFramesForExtra;
        /*
        if (a == 0) {
        cout << "Ind = " << ind << endl;
        FILE* f = fopen("points.txt", "wt");
        for (int i = 0; i < ind; i++) {
        fprintf(f, "%f ", extra[i]);
        if (i % 3 == 2) fprintf(f, "\n");
        }
        fclose(f);
        }*/
    }
    virtual void ResetAgent(int a)
    {
        //mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);

        int aa = rand() % fullTrans.size();
        for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
        {
            int bi = i - agentBodies[a].first;
            Transform tt = agentOffset[a] * fullTrans[aa][bi];
            NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
            (Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
            (Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
        }

        RLWalkerEnv::ResetAgent(a);
        extras[a].clear();
        extraFrontIndex[a] = 0;
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
                    numCollideOther[ff / 2]++;
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
};

class RigidFullHumanoidDeepLocoRepro : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;
    vector<int> footFlag;

    vector<vector<Transform>> fullTrans;
    vector<vector<Vec3>> fullVels;
    vector<vector<Vec3>> fullAVels;
    string fullFileName;
    vector<pair<int, Transform>> features;
    vector<Transform> invFirstTorso;
    vector<int> resetNum;
    Transform invTargetFirstTorso;
    ofstream logRew;
    int countFlush;
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
        //float heading_rew = 0.5f*((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f*((upVec > 0.93f) ? 1.0 : 0.0f); // MJCF4

        int aa = mFrames[a];
        float difP = 0.0;
        float difQ = 0.0;
        float difV = 0.0;
        for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
        {
            int bi = i - agentBodies[a].first;
            Transform targetT = invTargetFirstTorso*fullTrans[aa][bi];
            Transform curT;
            NvFlexGetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&curT);
            curT = invFirstTorso[a] * curT;
            Vec3 difPos = (curT.p - targetT.p);
            difP += Dot(difPos, difPos);
            Quat difQuat = (curT.q - targetT.q);
            difQ += Dot(difQuat, difQuat);

            Vec3 targetVel = TransformVector(invTargetFirstTorso, fullVels[aa][bi]);
            Vec3 curVel = TransformVector(invFirstTorso[a],((Vec3&)g_buffers->rigidBodies[i].linearVel));
            Vec3 difVel = (curVel - targetVel);
            difV += Dot(difVel, difVel);

            //(Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
            //(Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
        }
        float pRMS = sqrt(difP / (agentBodies[a].second - agentBodies[a].first));
        float qRMS = sqrt(difQ / (agentBodies[a].second - agentBodies[a].first));
        float vRMS = sqrt(difV / (agentBodies[a].second - agentBodies[a].first));
        float pCoef = 1.0f;
        float qCoef = 1.0f;
        float vCoef = 1.0f/6.0f;
        float pWeight = 1.0f;
        float qWeight = 1.0f;
        float vWeight = 1.0f;

        float prew = pWeight*exp(-pRMS*pCoef);
        float qrew = qWeight*exp(-qRMS*qCoef);
        float vrew = vWeight*exp(-vRMS*vCoef);
        float rmatch = prew + qrew + vrew;
        //cout << "Agent " << a << " frame " << aa << " rms error = " << rmsError << " rmatch = "<<rmatch<<endl;
        if (a == 0)
        {
            logRew << a << " " << resetNum[a] << " " << mFrames[a] << " " << pRMS << " " << qRMS << " " << vRMS << " " << prew << " " << qrew << " " << vrew << " " << rmatch << endl;
            countFlush++;
            if (countFlush % 1000 == 0)
            {
                logRew.flush();
            }
        }
        float rewards[6] =
        {
            alive,
            progress,
            electricityCostCurrent,
            jointsAtLimitCostCurrent,
            feetCollisionCostCurrent,
            rmatch
        };

        //printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

        rew = 0.f;
        for (int i = 0; i < 6; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }

    RigidFullHumanoidDeepLocoRepro()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 200;

        g_numSubsteps = 4;
        g_params.numIterations = 32;

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 1;

        numPerRow = 20;
        spacing = 10.f;

        numFeet = 2;

        powerScale = 0.41f; // Default
        initialZ = 0.9f;

        electricityCostScale = 1.f;

        angleResetNoise = 0.075f;
        angleVelResetNoise = 0.05f;
        velResetNoise = 0.05f;

        pushFrequency = 250;	// How much steps in average per 1 kick
        forceMag = 0.f;
    }

    void PrepareScene() override
    {
        ParseJsonParams(g_sceneJson);

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "HumanoidFull_160";
            ppo_params.resume = 137;
            ppo_params.timesteps_per_batch = 124;
            ppo_params.hid_size = 160;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_stepsize = 1e-3;
            ppo_params.optim_batchsize_per_agent = 16;
            ppo_params.clip_param = 0.2;
            */
            //const char* working_dir = "C:/Deep_RL/baselines/baselines/ppo1";
            //const char* python_file = "C:/Users/vmakoviychuk/AppData/Local/Continuum/Anaconda3/python.exe";
            //string folder = "HumanoidFull";

            //-------------- Nuttapong ----------------
            ppo_params.resume = 4860;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;

            //string folder = "flex_humanoid_deep_loco_repro_with_progress";
            ppo_params.relativeLogDir = "flex_humanoid_deep_loco_repro_with_progress_reduced_features";

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
            //vector<string> geo_joint = { "lwaist","uwaist", "torso1", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin", "right_foot","left_thigh","left_shin","left_foot" };
            vector<string> geo_joint = { "left_hand","right_hand","left_foot","right_foot","head" };
            for (size_t i = 0; i < (int)geo_joint.size(); i++)
            {
                auto p = mjcfs[0]->geoBodyPose[geo_joint[i]];
                p.first -= agentBodies[0].first;
                features.push_back(p);
            }
            init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }
        invTargetFirstTorso = Inverse(fullTrans[0][0]);
        invFirstTorso.resize(mNumAgents);
        resetNum.resize(mNumAgents,0);
        logRew.open("log_deep_loco_rew.txt");
        countFlush = 0;
    }

    ~RigidFullHumanoidDeepLocoRepro()
    {
    }

    virtual void ResetAgent(int a)
    {
        //mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
        int aa = 0;
        for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
        {
            int bi = i - agentBodies[a].first;
            Transform tt = agentOffset[a] * fullTrans[aa][bi];
            NvFlexSetRigidPose(&g_buffers->rigidBodies[i], (NvFlexRigidPose*)&tt);
            (Vec3&)g_buffers->rigidBodies[i].linearVel = Rotate(agentOffset[a].q, fullVels[aa][bi]);
            (Vec3&)g_buffers->rigidBodies[i].angularVel = Rotate(agentOffset[a].q, fullAVels[aa][bi]);
        }

        invFirstTorso[a] = Inverse(agentOffset[a] * fullTrans[aa][0]);
        resetNum[a]++;
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
                    numCollideOther[ff / 2]++;
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
};



class RigidFullHumanoidSanityCheckMJCF : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;
    vector<int> footFlag;

    RigidFullHumanoidSanityCheckMJCF()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 1000;

        g_numSubsteps = 4;
        g_params.numIterations = 30;

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 60;

        numPerRow = 20;
        spacing = 100.f;

        numFeet = 2;

        powerScale = 0.205f;
        initialZ = 0.9f;
        terminationZ = 0.795f;

        electricityCostScale = 1.8f;
        stallTorqueCostScale = 5.f;

        maxX = 100.f;
        maxY = 100.f;
        maxFlagResetSteps = 175;

        angleResetNoise = 0.02f;
        angleVelResetNoise = 0.02f;
        velResetNoise = 0.02f;

        pushFrequency = 300;	// How much steps in average per 1 kick
        forceMag = 1.5f;

    }

    void PrepareScene() override
    {
        ParseJsonParams(g_sceneJson);

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "FH_flagrun_192_1e-3_linear_8_tz";
            ppo_params.resume = 0;
            ppo_params.num_timesteps = 256000000;
            ppo_params.timesteps_per_batch = 256;
            */
            ppo_params.optim_batchsize_per_agent = 32;
            ppo_params.optim_stepsize = 1e-3f;
            ppo_params.optim_epochs = 8;

            ppo_params.hid_size = 192;
            ppo_params.num_hid_layers = 2;

            ppo_params.clip_param = 0.2f;

            ppo_params.relativeLogDir = "humanoid_sanity_check";

            ppo_params.TryParseJson(g_sceneJson);

            //-------------- Nuttapong ----------------
        //    string folder = "flex_humanoid_22897509_with_armature_mod_alive";

            init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }
    }
    

    virtual void ResetAgent(int a)
    {
        mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);
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
                    numCollideOther[ff / 2]++;
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
        if (z > terminationZ) // 0.795 // Annealing!
        {
            return 1.5f;
        }
        else
        {
            return -1.f;
        }
    }
};

class RigidFullHumanoidMocapInitGANFrameFeatures : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
public:

    vector<int> rightFoot;
    vector<int> leftFoot;

    vector<int> footFlag;

    vector<vector<Transform>> fullTrans;
    vector<vector<Vec3>> fullVels;
    vector<vector<Vec3>> fullAVels;
    string fullFileName;

    string root;
    vector<string> frameFeatures;
    vector<vector<vector<float> > > featuresQueue;
    vector<int> frontIndex;
    int queueSize;
    int numEachFrame;
    bool withVel;
    virtual void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead)
    {
        float& potential = potentials[a];
        float& potentialOld = potentialsOld[a];
        float& p = ps[a];
        float& walkTargetDist = walkTargetDists[a];
        float* joint_speeds = &jointSpeeds[a][0];
        int& jointsAtLimit = jointsAtLimits[a];
        float& heading = headings[a];
        float& upVec = upVecs[a];

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
        float heading_rew = 0.5f * ((heading > 0.8f) ? 1.0f : heading / 0.8f) + 0.05f * ((upVec > 0.93f) ? 1.f : 0.0f); // MJCF4
        float rewards[6] =
        {
            alive,
            progress,
            electricityCostCurrent,
            jointsAtLimitCostCurrent,
            feetCollisionCostCurrent,
            heading_rew
        };

        //printf("%lf %lf %lf %lf %lf\n", rewards[0], rewards[1], rewards[2], rewards[3], rewards[4]);

        rew = 0.f;
        for (int i = 0; i < 6; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }

    RigidFullHumanoidMocapInitGANFrameFeatures()
    {
        doFlagRun = false;
        loadPath = "../../data/humanoid_mod.xml";

        withVel = true;
        mNumAgents = 500;
        mNumObservations = 52;
        mNumActions = 21;
        mMaxEpisodeLength = 1000;
        queueSize = 1;
        // GAN 4-11
        root = "torso1";
        frameFeatures = { "left_hand","right_hand","left_foot","right_foot","head" };

        vector<int> frontIndex;
        g_numSubsteps = 4;
        g_params.numIterations = 20;
        //g_params.numIterations = 32; GAN4

        g_sceneLower = Vec3(-150.f, -250.f, -100.f);
        g_sceneUpper = Vec3(250.f, 150.f, 100.f);

        g_pause = true;
        mDoLearning = g_doLearning;
        numRenderSteps = 60;

        numPerRow = 20;
        spacing = 100;

        numFeet = 2;


        powerScale = 0.25f; // Reduced power
        initialZ = 0.9f;

        electricityCostScale = 1.f;

        angleResetNoise = 0.075f;
        angleVelResetNoise = 0.05f;
        velResetNoise = 0.05f;

        pushFrequency = 250;	// How much steps in average per 1 kick
        forceMag = 0.f;

    }

    void PrepareScene() override
    {
        ParseJsonParams(g_sceneJson);

        if (withVel)
        {
            mNumExtras = 3 * (frameFeatures.size() * 2 + 3);
        }
        else
        {
            mNumExtras = 3 * (frameFeatures.size() + 3);
        }
        featuresQueue.resize(mNumAgents);
        frontIndex.resize(mNumAgents);
        for (int i = 0; i < mNumAgents; i++)
        {
            frontIndex[i] = 0;
            featuresQueue[i].resize(0);
        }
        numEachFrame = mNumExtras;
        mNumExtras *= queueSize;

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        LoadEnv();

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

            //------------- Viktor --------------------
            /*
            ppo_params.agent_name = "HumanoidFull_160";
            ppo_params.resume = 137;
            ppo_params.timesteps_per_batch = 124;
            ppo_params.hid_size = 160;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_stepsize = 1e-3;
            ppo_params.optim_batchsize_per_agent = 16;
            ppo_params.clip_param = 0.2;
            */
            //const char* working_dir = "C:/Deep_RL/baselines/baselines/ppo1";
            //const char* python_file = "C:/Users/vmakoviychuk/AppData/Local/Continuum/Anaconda3/python.exe";
            //string folder = "HumanoidFull";

            //-------------- Nuttapong ----------------

            /*
            GAN 4
            ppo_params.useGAN = true;
            ppo_params.resume = 3569;
            ppo_params.timesteps_per_batch = 500;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 128;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-3f;
            ppo_params.gan_reward_scale = 50.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 5.0f;
            ppo_params.gan_num_epochs = 5;
            */

            //GAN 5, rerun with GAN 8looks promising
            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 3170;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-5f; // GAN 5 and 8 1e-7, GAN 9 1e-6 (unlearn), GAN 10 1e-4, GAN 11 1e-5
            ppo_params.gan_reward_scale = 10.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            ppo_params.useGAN = true;
            ppo_params.useDistance = false;
            ppo_params.useBlend = false;
            ppo_params.resume = 0;
            ppo_params.timesteps_per_batch = 200;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-3f;  // 7714H8B used 1e-4
            ppo_params.gan_reward_scale = 0.f; //
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 8;
            ppo_params.gan_queue_size = queueSize;

            /*
            ppo_params.useGAN = true;
            ppo_params.resume = 2057;
            ppo_params.timesteps_per_batch = 20000;
            ppo_params.hid_size = 256;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_batchsize_per_agent = 64;
            ppo_params.gan_hid_size = 128;
            ppo_params.gan_num_hid_layers = 2;
            ppo_params.gan_learning_rate = 1e-6f; // 1e-6 for gan6   1e-4 for gan7
            ppo_params.gan_reward_scale = 1.0f;
            ppo_params.gan_reward_to_retrain_discriminator = 0.0f;
            ppo_params.gan_num_epochs = 1;
            */
            //ppo_params.resume_non_disc = 2670;
            char NPath[5000];

            GetCurrentDir(5000, NPath);
            cout << NPath << endl;
            //GAN 4-11
            //ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion_med";

            if (withVel)
            {
                ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion_frame_with_vel";
            }
            else
            {
                ppo_params.mocapPath = string(NPath) + "/../../data/bvh/motion_frame";
            }

            //string folder = "flex_humanoid_mocap_init_python_gan_frame_features_1e-3_use_4_epoch_forward_rewmul_80_update_always_fixed_new_with_vel_target_vel";
            //string folder = "flex_humanoid_mocap_init_python_gan_frame_features_1e-3_use_4_epoch_forward_rewmul_80_update_always_fixed_new_with_vel_queue_size_10";
            //string folder = "flex_humanoid_mocap_init_python_gan_frame_features_1e-3_forward_rewmul_20_update_always_fixed_new_with_vel_target_vel";
            ppo_params.relativeLogDir = "flex_mocap_init_nogan";

            ppo_params.TryParseJson(g_sceneJson);

            fullFileName = "../../data/bvh/LocomotionFlat02_000.state";

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

            init(ppo_params, ppo_params.pythonPath.c_str(), ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }

    }

    virtual void PopulateExtra(int a, float* extra)
    {
        int backIndex = (frontIndex[a] + queueSize - 1) % queueSize;
        bool duplicateToAll = false;
        if (featuresQueue[a].size() == 0)
        {
            duplicateToAll = true;
            // Handle special case
            featuresQueue[a].resize(queueSize);
            for (int j = 0; j < queueSize; j++)
            {
                featuresQueue[a][j].resize(numEachFrame);
            }
        }
        float* ptr = &featuresQueue[a][backIndex][0];
        auto p = mjcfs[a]->geoBodyPose[root];
        Transform tf;
        NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
        Transform invRoot = Inverse(tf);
        Vec3 rootPos = tf.p;
        Vec3 rootVel = (Vec3&)g_buffers->rigidBodies[p.first].linearVel;
        Vec3 cvel = TransformVector(invRoot, (Vec3&)g_buffers->rigidBodies[p.first].linearVel);
        Vec3 cavel = TransformVector(invRoot, (Vec3&)g_buffers->rigidBodies[p.first].angularVel);
        Vec3 cup = tf.q * Vec3(0.0f, 0.0f, 1.0f); // Up is Z for mujoco
        *(ptr++) = cvel.x;
        *(ptr++) = cvel.y;
        *(ptr++) = cvel.z;
        *(ptr++) = cavel.x;
        *(ptr++) = cavel.y;
        *(ptr++) = cavel.z;
        *(ptr++) = cup.x;
        *(ptr++) = cup.y;
        *(ptr++) = cup.z;

        for (int i = 0; i < frameFeatures.size(); i++)
        {
            auto p = mjcfs[a]->geoBodyPose[frameFeatures[i]];

            NvFlexGetRigidPose(&g_buffers->rigidBodies[p.first], (NvFlexRigidPose*)&tf);
            tf = tf*p.second;
            Vec3 tmp = TransformVector(invRoot, tf.p - rootPos);
            *(ptr++) = tmp.x;
            *(ptr++) = tmp.y;
            *(ptr++) = tmp.z;

            if (withVel)
            {
                Vec3 rvel = TransformVector(invRoot, Cross((Vec3&)g_buffers->rigidBodies[p.first].angularVel, tf.p - (Vec3&)g_buffers->rigidBodies[p.first].com) + (Vec3&)g_buffers->rigidBodies[p.first].linearVel - rootVel);
                *(ptr++) = rvel.x;
                *(ptr++) = rvel.y;
                *(ptr++) = rvel.z;
            }
        }

        if (duplicateToAll)
        {
            for (int j = 0; j < queueSize - 1; j++)
            {
                featuresQueue[a][j] = featuresQueue[a][backIndex];
            }
        }

        for (int i = 0; i < queueSize; i++)
        {
            int ind = (frontIndex[a] + i) % queueSize;
            memcpy(extra + i*numEachFrame, &featuresQueue[a][ind][0], sizeof(float)*numEachFrame);
        }

    }
    virtual void ResetAgent(int a)
    {
        //mjcfs[a]->reset(agentOffset[a], angleResetNoise, velResetNoise, angleVelResetNoise);

        int aa = rand() % fullTrans.size();
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
                    numCollideOther[ff / 2]++;
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
};