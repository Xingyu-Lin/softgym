#pragma once
#include "../../external/rl/RLFlexEnv.h"
#include "../urdf.h"
#include "rlbase.h"
#include <iostream>
#include <random>
#include <vector>

#include <assert.h>

class RLMinitaur : public RLWalkerEnv<Transform, Vec3, Quat, Matrix33>
{
  public:
    URDFImporter *urdf;

    int robotMaterial;
    vector<string> motors;
    vector<int> motorJoints;

    vector<float> powers;
    vector<float> prevZ;
    vector<string> feet;
    vector<int> feetInds[8];
    vector<int> footFlag;
    vector<string> legPairs;
    vector<string> knees;
    vector<int> kneesInds[8];
    vector<int> kneeFlag;
    const char *urdfFilename;
    vector<float> mp;

    vector<float> prevMotorAngles;

    bool randomizing;
    float mGravityRandom;
    float mTorqueRandom;
    float mFrictionLow;
    float mFrictionHigh;
    float vel_rand_std;
    float imu_rand_std;
    float vel_rand_bias;
    float imu_rand_bias;

    bool g_useTorqueControl;
    float maxTorque;
    float currTorque;
    float jointVelocityTorqueThresh;

    float mFriction;
    float mRollingFriction;
    float mTorsionFriction;
    float mMaxMotorAngularVelocity;

    float mMaxTorqueDifference;
    vector<float> appliedTorque;

    float mNumMotors;
    float targetSpeed;
    float mSpeedLow;
    float mSpeedHigh;

    int terrainInd;
    bool hardTerrain;

    RLMinitaur()
    {
        /* g_params.gravity[1] = 0.f; */

        urdfFilename = "minitaur_derpy_rotated_legs.urdf";
        mNumMotors = 8;
        mNumAgents = 100;
        numPerRow = 5;
        spacing = 4.f;
        yOffset = 0.4f;
        mMaxEpisodeLength = 1000;

        doFlagRun = false;
        maxFlagResetSteps = 500;
        mSpeedLow = 0.3f;
        mSpeedHigh = 0.3f;
        targetSpeed = mSpeedLow;

        maxTorque = 3.5f;
        currTorque = maxTorque;
        mFriction = 1.f;
        mRollingFriction = 0.05f;
        mTorsionFriction = 0.05f;

        randomizing = false;
        vel_rand_std = 0.00001f;
        vel_rand_bias = 0.00001f;
        imu_rand_std = 0.05f;
        imu_rand_bias = 0.05f;

        mTorqueRandom = 0.2f; // TODO: Expose this parameter to json or python
        mGravityRandom = 0.2f;
        mFrictionLow = 0.6f;
        mFrictionHigh = 1.2f;

        terrainInd = -1;
        hardTerrain = false;

        mMaxMotorAngularVelocity = 167.f;

        mMaxTorqueDifference = 5.f;
        appliedTorque.reserve(8);
        for (int i = 0; i < appliedTorque.capacity(); i++)
        {
            appliedTorque[i] = 0.f;
        }

        g_useTorqueControl = true;
        g_params.shapeCollisionMargin = 0.01f;
        g_params.numPostCollisionIterations = 16;
        g_numSubsteps = 4;
        g_params.numIterations = 50;
        g_params.dynamicFriction = 0.1f;
        g_params.staticFriction = 0.4f;
        g_params.wind[0] = 0.f;
        g_params.wind[1] = 0.f;
        g_params.wind[2] = 0.f;

        g_params.relaxationFactor = 1.0f;
        g_params.damping = 1.f;

        g_sceneLower = Vec3(-7.f, 0.f, 49.f);
        g_sceneUpper = Vec3(13.f, 0.6f, 50.f);

        g_pause = true;
        numRenderSteps = 1;

        initialZ = 0.19f;
        terminationZ = 0.16; // Overridden in minitaur.yaml

        g_numSubsteps = 2;
        g_params.numInnerIterations = 4;
        g_params.numIterations = 25;

        g_params.solverType = eNvFlexSolverPCR;

        electricityCostScale = 0.08f;
        stallTorqueCostScale = 0.f;
        footCollisionCost = -2.f;
        jointsAtLimitCost = -0.25f;
        jointVelocityTorqueThresh = 1.f;

        angleResetNoise = 0.0f;
        angleVelResetNoise = 0.1f;
        velResetNoise = 0.1f;

        preTransform =
            Transform(Vec3(), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi));
        pushFrequency = 300; // How much steps in average per 1 kick
        forceMag = 1.f;

        mp.clear();
        mp.resize(4, 1.f * kPi);

        prevMotorAngles.clear();
        prevMotorAngles.resize(mNumAgents * 8, 1e8);
    }

    virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt,
                                                vector<pair<int, NvFlexRigidJointAxis>> &ctrl,
                                                vector<float> &mpower)
    {
        // Adds the physical bodies
        urdf->AddPhysicsEntities(gt, robotMaterial, true, 1000.0f, 0.0f, 10.f,
                                 0.0005f, 10.f, 100.f, false, 1e10f, 0.0f, 0.001f);

        // Adds the Ankle Joints
        bool ankleJointNeedsAdding = true;
        if (ankleJointNeedsAdding)
        {
            for (size_t j = 0; j < legPairs.size(); j += 2)
            {
                int l0 = urdf->rigidNameMap[legPairs[j]];
                int l1 = urdf->rigidNameMap[legPairs[j + 1]];
                Transform t0, t1;
                NvFlexGetRigidPose(&g_buffers->rigidBodies[l0], (NvFlexRigidPose *)&t0);
                NvFlexGetRigidPose(&g_buffers->rigidBodies[l1], (NvFlexRigidPose *)&t1);

                NvFlexRigidJoint joint;

                Vec3 outsideTip = t0.p;
                outsideTip.y -= (0.100f - 0.024f);
                Transform outsideTipTF = Transform(outsideTip, Quat());

                Vec3 insideTip = t1.p;
                insideTip.y -= (0.100f);
                /* Transform insideTipTF = Transform(insideTip, Quat()); */

                Transform lt0, lt1;
                lt0 = Inverse(t0) * outsideTipTF;
                lt1 = Inverse(t1) * outsideTipTF;

                NvFlexMakeSphericalJoint(&joint, l0, l1,
                                         NvFlexMakeRigidPose(lt0.p, lt0.q),
                                         NvFlexMakeRigidPose(lt1.p, lt1.q));
                g_buffers->rigidJoints.push_back(joint);
            }
        }

        // Adds controls to the robot. I believe this allows us to move the motors
        for (size_t j = 0; j < motors.size(); j++)
        {
            ctrl.push_back(make_pair(urdf->jointNameMap[motors[j]], eNvFlexRigidJointAxisTwist));
            mpower.push_back(powers[j]);

            motorJoints.push_back(urdf->jointNameMap[motors[j]]);
        }

        // Keeps track of the feet
        for (size_t j = 0; j < feet.size(); j++)
        {
            feetInds[j][i] = urdf->rigidNameMap[feet[j]];
        }

        // Keeps track of the knees
        for (size_t j = 0; j < knees.size(); j++)
        {
            kneesInds[j][i] = urdf->rigidNameMap[knees[j]];
        }

        // Keeps track of the torso
        torso[i] = urdf->rigidNameMap["base_chassis_link"];
    }

    void PrepareScene() override
    {
        motorJoints.clear();

        ParseJsonParams(g_sceneJson);

        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);
        prevZ.resize(mNumAgents, initialZ);

        // hide collision shapes
        robotMaterial =
            AddRenderMaterial(Vec3(0.47f, 0.62f, 0.93f), 0.1f, 0.8f, false);

        urdf = new URDFImporter("../../data/minitaur", urdfFilename);

        feet = {"lower_leg_front_rightR_link", "lower_leg_front_rightL_link",
                "lower_leg_front_leftR_link", "lower_leg_front_leftL_link",
                "lower_leg_back_rightR_link", "lower_leg_back_rightL_link",
                "lower_leg_back_leftR_link", "lower_leg_back_leftL_link"};
        // Motors + joints used in observations (knees)
        /*    motors = { "motor_front_rightR_joint", "motor_front_rightL_joint",
"motor_front_leftL_joint", "motor_front_leftR_joint",
"motor_back_rightR_joint", "motor_back_rightL_joint",
"motor_back_leftL_joint", "motor_back_leftR_joint"
}; */

        motors = {"0", "1", "2", "3", "4", "5", "6", "7"};

        legPairs = {"lower_leg_back_rightR_link", "lower_leg_back_rightL_link",
                    "lower_leg_back_leftL_link", "lower_leg_back_leftR_link",
                    "lower_leg_front_rightR_link", "lower_leg_front_rightL_link",
                    "lower_leg_front_leftL_link", "lower_leg_front_leftR_link"};

        knees = {"upper_leg_front_rightR_link", "upper_leg_front_rightL_link",
                 "upper_leg_front_leftR_link", "upper_leg_front_leftL_link",
                 "upper_leg_back_rightR_link", "upper_leg_back_rightL_link",
                 "upper_leg_back_leftR_link", "upper_leg_back_leftL_link"};

        int numKnees = knees.size();
        numFeet = feet.size();

        mNumActions = 9; // 8 Motors + Target Speed
        // Where did these numbers come from?
        mNumObservations = 10 + motors.size() * 2 + numFeet;
        for (int i = 0; i < numFeet; i++)
        {
            feetInds[i].resize(mNumAgents);
        }

        for (int i = 0; i < numKnees; i++)
        {
            kneesInds[i].resize(mNumAgents);
        }

        for (auto m : motors)
        {
            float power = urdf->joints[urdf->jointNameMap[m]]->effort;

            cout << urdf->jointNameMap[m] << " power = " << power << endl;
            cout << m << " power = " << power << endl;

            power = 100.f;
            powers.push_back(power);
        }

        LoadEnv();
        for (int i = 0; i < (int)g_buffers->rigidShapes.size(); ++i)
        {
            g_buffers->rigidShapes[i].filter = 1;
            g_buffers->rigidShapes[i].material.friction = mFriction;
            g_buffers->rigidShapes[i].material.rollingFriction = mRollingFriction;
            g_buffers->rigidShapes[i].material.torsionFriction = mTorsionFriction;
        }

        footFlag.resize(g_buffers->rigidBodies.size());
        kneeFlag.resize(g_buffers->rigidBodies.size());
        for (int i = 0; i < (int)g_buffers->rigidBodies.size(); i++)
        {
            initBodies.push_back(g_buffers->rigidBodies[i]);
            footFlag[i] = -1;
            kneeFlag[i] = -1;
        }

        for (int i = 0; i < mNumAgents; i++)
        {
            for (int j = 0; j < numFeet; j++)
            {
                footFlag[feetInds[j][i]] = numFeet * i + j;
            }

            for (int j = 0; j < numKnees; j++)
            {
                kneeFlag[kneesInds[j][i]] = numKnees * i + j;
            }
        }

        if (hardTerrain)
        {
            Transform terrainTrans = Transform(Vec3(90.f, 0.f, 100.f), Quat());
            Vec3 offset = RandVec3() * 1.f;
            terrainInd = createTerrain(220.f, 250.f, 120, 130, offset, terrainTrans,
                                       Vec3(25.f, 0.5f, 25.f), 5, 0.41f);
        }

        if (mDoLearning)
        {
            PPOLearningParams ppo_params;

            ppo_params.agent_name = "Minitaur_128";
            ppo_params.resume = 0;
            ppo_params.timesteps_per_batch = 256;
            ppo_params.hid_size = 128;
            ppo_params.num_hid_layers = 2;
            ppo_params.optim_epochs = 10;
            ppo_params.optim_stepsize = 5e-4f;
            ppo_params.optim_schedule = "adaptive";
            ppo_params.desired_kl = 0.02f;
            ppo_params.optim_batchsize_per_agent = 32;
            ppo_params.clip_param = 0.2f;

            ppo_params.relativeLogDir = "Minitaur";

            ppo_params.TryParseJson(g_sceneJson);

            init(ppo_params, ppo_params.pythonPath.c_str(),
                 ppo_params.workingDir.c_str(), ppo_params.relativeLogDir.c_str());
        }
    }

    void DoGui() override
    {
        imguiSlider("Target Speed", &targetSpeed, -2.f, 2.f, 0.0001f);
    }

    ~RLMinitaur()
    {
        if (urdf)
        {
            delete urdf;
        }
    }

    virtual void LoadEnv()
    {
        ctrls.resize(mNumAgents);
        motorPower.resize(mNumAgents);

        torso.clear();
        torso.resize(mNumAgents, -1);

        for (int i = 0; i < mNumAgents; i++)
        {
            Vec3 pos =
                Vec3((i % numPerRow) * spacing, yOffset, (i / numPerRow) * spacing);

            // Rotates so the minitar is facing downward
            Quat rot = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi * 0.5);

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
        memcpy(&initJoints[0], &g_buffers->rigidJoints[0],
               sizeof(NvFlexRigidJoint) * g_buffers->rigidJoints.size());
    }

    virtual void SetLegPosition(std::vector<float> legAngles)
    {
        g_buffers->rigidJoints.unmap();
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
        g_buffers->rigidJoints.map();

        for (int ai = 0; ai < mNumAgents; ai++)
        {
            /* float *actions = GetAction(ai); */
            for (int i = 0; i < mNumMotors; i++)
            {
                float cc = legAngles[i];
                if (i < 4)
                {
                    cc = -cc;
                }
                g_buffers->rigidJoints[motorJoints[mNumActions * ai + i]]
                    .modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
                g_buffers->rigidJoints[motorJoints[mNumActions * ai + i]]
                    .targets[eNvFlexRigidJointAxisTwist] = cc;
            }
        }
        g_buffers->rigidJoints.unmap();
        NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer,
                             g_buffers->rigidJoints.size());

        g_buffers->rigidBodies.unmap();
        NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer,
                             g_buffers->rigidBodies.size());

        NvFlexSetParams(g_solver, &g_params);
        NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
        g_frame++;
        NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
        NvFlexGetRigidContacts(g_solver, rigidContacts.buffer,
                               rigidContactCount.buffer);
        g_buffers->rigidBodies.map();
    }

    virtual void ReleaseLegPosition()
    {
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
        g_buffers->rigidJoints.map();

        for (int ai = 0; ai < mNumAgents; ai++)
        {
            /* float *actions = GetAction(ai); */
            for (int i = 0; i < mNumMotors; i++)
            {
                g_buffers->rigidJoints[motorJoints[mNumActions * ai + i]]
                    .modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeFree;
            }
        }
        g_buffers->rigidJoints.unmap();
        NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer,
                             g_buffers->rigidJoints.size());
    }

    virtual void Simulate()
    { // Called Every Frame
        if (g_frame < 1)
        { // Only on startup
            // vector<float> initialLegAngles;
            // initialLegAngles = {0.5f * kPi, 0.5f * kPi, -0.5f * kPi, -0.5f * kPi,
            //                     0.5f * kPi, 0.5f * kPi, -0.5f * kPi, -0.5f * kPi};
            // SetLegPosition(initialLegAngles);
            // ReleaseLegPosition();
            g_frame++;
        }
        else
        {
            if (true)
            { // Set motor actions
                for (int ai = 0; ai < mNumAgents; ai++)
                { // Get each agent
                    for (int i = agentBodies[ai].first; i < (int)agentBodies[ai].second;
                         i++)
                    { // Zero the force on every body
                        g_buffers->rigidBodies[i].force[0] = 0.0f;
                        g_buffers->rigidBodies[i].force[1] = 0.0f;
                        g_buffers->rigidBodies[i].force[2] = 0.0f;
                        g_buffers->rigidBodies[i].torque[0] = 0.0f;
                        g_buffers->rigidBodies[i].torque[1] = 0.0f;
                        g_buffers->rigidBodies[i].torque[2] = 0.0f;
                    }

                    if (randomizing && g_frame % 1000 == 0)
                    {
                        /* cout << "RANDOM\n"; */
                        currTorque = Randf(maxTorque * (1.f - mTorqueRandom),
                                           maxTorque * (1.f + mTorqueRandom));
                        mFriction = Randf(mFrictionLow, mFrictionHigh);
                        /* mTorsionFriction = Randf(0.f, 0.05f); */
                        float gravMultiplier =
                            Randf(1.f - mGravityRandom, 1.f + mGravityRandom);
                        g_params.gravity[1] = -9.8f * gravMultiplier;
                        g_params.gravity[0] = 0.f + Randf(-mGravityRandom, mGravityRandom);
                        g_params.gravity[2] = 0.f + Randf(-mGravityRandom, mGravityRandom);
                        vel_rand_std = Randf(0.f, 0.00001f);
                        vel_rand_bias = Randf(-0.00001f, 0.00001f);
                        imu_rand_std = Randf(0.f, 0.05f);
                        imu_rand_bias = Randf(-0.05f, 0.05f);

                        if (g_frame % 10000 == 0)
                        {
                            targetSpeed = Randf(mSpeedLow, mSpeedHigh);
                            // cout << "Setting target speed to " << targetSpeed << endl;
                        }
                    }

                    float *actions;
                    if (g_useTorqueControl)
                    {
                        // Set the torque of every motor
                        actions = GetAction(ai);

                        setTorques(actions, ai);
                    }
                    else
                    {
                        // Using Position Control

                        // Desired extension and swing positions in leg space for each leg.
                        // Should be size 8. (e, s) for each leg
                        // TODO: Implement in C++. Currently implemented in python in
                        // gym/demo/sim2real/simtaur.py
                    }
                }
            }
        }

        // Run the simuation in flex
        g_buffers->rigidBodies.unmap();
        NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer,
                             g_buffers->rigidBodies.size());

        NvFlexSetParams(g_solver, &g_params);
        NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);
        g_frame++;
        NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
        NvFlexGetRigidContacts(g_solver, rigidContacts.buffer,
                               rigidContactCount.buffer);
        g_buffers->rigidBodies.map();
    }

    void setTorques(float *actions, int ai)
    {
        for (int i = 0; i < mNumMotors; i++)
        {
            float cc = actions[i];

            float maxTorqueBasedOnVelocity = currTorque;
            float jointVelocity = abs(jointSpeeds[ai][i]); // * 10.f;

            if (jointVelocity > mMaxMotorAngularVelocity)
            {
                cout << "######################\n";
            }
            if (jointVelocity > jointVelocityTorqueThresh)
            {
                maxTorqueBasedOnVelocity =
                    currTorque * (1 - (jointVelocity / mMaxMotorAngularVelocity));
            }

            cc = min(abs(cc), maxTorqueBasedOnVelocity) * sign(cc);

            if (abs(cc - appliedTorque[i]) > mMaxTorqueDifference)
            {
                /* cout << "diff is " << abs(cc - appliedTorque[i]) << endl; */
                /* cout << "cc was " << cc << "\tprevious torque " << appliedTorque[i]
* << "\tmotor number " << i << endl; */
                if (cc > appliedTorque[i])
                {
                    cc = appliedTorque[i] + mMaxTorqueDifference;
                }
                else
                {
                    cc = appliedTorque[i] - mMaxTorqueDifference;
                }
                /* cout << "cc is now " << cc << endl; */
            }
            appliedTorque[i] = cc;

            NvFlexRigidJoint &j = initJoints[ctrls[ai][i].first];
            NvFlexRigidBody &a0 = g_buffers->rigidBodies[j.body0];
            NvFlexRigidBody &a1 = g_buffers->rigidBodies[j.body1];
            Transform &pose0 = *((Transform *)&j.pose0);
            Transform gpose;
            NvFlexGetRigidPose(&a0, (NvFlexRigidPose *)&gpose);
            Transform tran = gpose * pose0;

            Vec3 axis;
            if (ctrls[ai][i].second == eNvFlexRigidJointAxisTwist)
            {
                axis = GetBasisVector0(tran.q);
            }
            if (ctrls[ai][i].second == eNvFlexRigidJointAxisSwing1)
            {
                axis = GetBasisVector1(tran.q);
            }
            if (ctrls[ai][i].second == eNvFlexRigidJointAxisSwing2)
            {
                axis = GetBasisVector2(tran.q);
            }

            Vec3 torque = axis * cc;

            torque.x = torque.x;
            torque.y = torque.y;
            torque.z = torque.z;
            a0.torque[2] += torque.z;
            a1.torque[2] -= torque.z;
        }
    }

    int sign(float value) { return ((0.f < value) - (value < 0.f)); }

    virtual void ExtractState(int a, float *state, float &p,
                              float &walkTargetDist, float *jointSpeeds,
                              int &numJointsAtLimit, float &heading,
                              float &upVec)
    {
        // Prepare state
        //--------------------
        numJointsAtLimit = 0;

        float joints[MaxActions * 2];
        float angles[MaxActions];
        float lows[MaxActions];
        float highs[MaxActions];

        int numBodies = GetNumBodies();

        GetAngles(a, angles, lows, highs);
        for (int i = 0; i < mNumMotors; i++)
        {
            int qq = i;

            float vel;
            float pos = angles[i];
            /* float low = lows[i]; */
            /* float high = highs[i]; */

            if (bPrevAngleValue[a][qq] > 1e8)
            {
                bPrevAngleValue[a][qq] = pos;
            }

            float pos_diff = pos - bPrevAngleValue[a][qq];
            if (fabs(pos_diff) < kPi)
            {
                vel = pos_diff / dt;
            }
            else
            {
                vel = (-(sign(pos) * 2.f * kPi) + pos_diff) / dt;
            }
            bPrevAngleValue[a][qq] = pos;

            pos += 3.f;
            // vel *= pow(10, 7);
            joints[2 * i] = pos;
            joints[2 * i + 1] = vel;

            jointSpeeds[i] = joints[2 * i + 1];
            if (fabs(joints[2 * i]) > 0.99f)
            {
                numJointsAtLimit++;
            }
        }

        Transform bodies[50];

        GetGlobalPose(a, bodies);
        Transform bodyPose = bodies[0];

        upVec = GetBasisVector2(bodyPose.q).z;
        float sumX = 0.0f;
        float sumY = 0.0f;
        int num = 0;
        for (int i = 0; i < numBodies; i++)
        {
            Vec3 pos = bodies[i].p;
            sumX += pos.x;
            sumY += pos.y;
            num++;
        }
        float z2 = 0;
        if (torso[a] != -1)
        {
            Transform torsoPose;
            NvFlexGetRigidPose(&g_buffers->rigidBodies[torso[a]],
                               (NvFlexRigidPose *)&torsoPose);
            z2 = torsoPose.p.y;
        }

        bodyXYZ[a] = Vec3(sumX / num, sumY / num, z2); // bodyPose.p.z);

        getEulerZYX(bodyPose.q, yaw, p, r);
        float x = bodyXYZ[a].x;
        float y = bodyXYZ[a].y;
        float z = bodyXYZ[a].z;

        if (initialZ > 1e9)
        {
            initialZ = z;
        }

        // Hard code target for now
        Vec3 toTarget = Vec3(walkTargetX[a] - bodyXYZ[a].x,
                             walkTargetY[a] - bodyXYZ[a].y, 0.0f);
        walkTargetDist = Length(toTarget);

        if (doFlagRun)
        {
            if (flagRunSteps[a] > maxFlagResetSteps || walkTargetDist < 1.f)
            {
                resetTarget(a, false);
                toTarget = Vec3(walkTargetX[a] - bodyXYZ[a].x,
                                walkTargetY[a] - bodyXYZ[a].y, 0.0f);
                walkTargetDist = Length(toTarget);

                potentialsOld[a] = -walkTargetDist / dt;
                potentials[a] = -walkTargetDist / dt;
            }
            flagRunSteps[a]++;
        }

        /* float walkTargetTheta = */
        /*     atan2(walkTargetY[a] - bodyXYZ[a].y, walkTargetX[a] - bodyXYZ[a].x);
*/

        toTarget = Normalize(toTarget);
        heading = Dot(GetBasisVector0(bodyPose.q), toTarget);
        /* float angleToTarget = walkTargetTheta - yaw; */

        Matrix33 mat =
            Matrix33(Vec3(cos(-yaw), sin(-yaw), 0.0f),
                     Vec3(-sin(-yaw), cos(-yaw), 0.0f), Vec3(0.0f, 0.0f, 1.0f));

        Vec3 vel = GetLinearVel(a, 0);
        Vec3 bvel = mat * vel;
        float vx = bvel.x;
        float vy = bvel.y;
        float vz = bvel.z;

        /* cout << vel_rand_std << " " << vel_rand_bias << " " << imu_rand_std << "
* " << imu_rand_bias << endl; */
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<float> vx_gen{0.3f * vx, vel_rand_std};
        float random_vx = vx_gen(gen) + vel_rand_bias;
        std::normal_distribution<float> vy_gen{0.3f * vy, vel_rand_std};
        float random_vy = vy_gen(gen) + vel_rand_bias;
        std::normal_distribution<float> vz_gen{0.3f * vz, vel_rand_std};
        float random_vz = vz_gen(gen) + vel_rand_bias;
        std::normal_distribution<float> r_gen{r, imu_rand_std};
        float random_r = r_gen(gen) + imu_rand_bias;
        std::normal_distribution<float> p_gen{p, imu_rand_std};
        float random_p = p_gen(gen) + imu_rand_bias;
        std::normal_distribution<float> yaw_gen{yaw, imu_rand_std};
        float random_yaw = yaw_gen(gen) + imu_rand_bias;

        /* cout << r << " " << random_r << " " << p << " " << random_p << " " << yaw
* << " " << random_yaw << endl; */
        float more[9] = {targetSpeed, 0, z, random_vx, random_vy,
                         random_vz, random_r, random_p, random_yaw};
        /* cout << "target speed in state " << more[0] << endl; */

        int ct = 0;
        for (int i = 0; i < 9; i++)
        {
            state[ct++] = more[i];
        }

        for (int i = 0; i < mNumMotors * 2; i++)
        {
            state[ct++] = joints[i];
        }

        float forceScale = 0.2f;
        for (int i = 0; i < numFeet; i++)
        {
            state[ct++] = forceScale * feetContact[numFeet * a + i];
        }
    }

    void ResetAgent(int a)
    {

        /* for (auto pz : prevZ) */
        /* { */
        /*   pz = initialZ; */
        /* } */
        for (int i = agentBodies[a].first; i < (int)agentBodies[a].second; i++)
        {
            g_buffers->rigidBodies[i] = initBodies[i];
        }
        for (int i = 0; i < 8; i++)
        {
            prevMotorAngles[a * 8 + 1] = 1e8;
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

    virtual bool IsSkipSimulation() { return true; }

    virtual void ComputeRewardAndDead(int a, float *action, float *state,
                                      float &rew, bool &dead)
    {
        float &potential = potentials[a];
        float &potentialOld = potentialsOld[a];
        /* float &p = ps[a]; */
        float &walkTargetDist = walkTargetDists[a];
        float *jointSpeedsA = &jointSpeeds[a][0];
        /* int &numJointsAtLimitA = jointsAtLimits[a]; */
        /* float &heading = headings[a]; */
        /* float &upVec = upVecs[a]; */

        float electrCost = electricityCostScale;
        float stallTorqCost = stallTorqueCostScale * stallTorqueCost;

        float alive = AliveBonus(state[2], state[6],
                                 state[7]); //   # state[2] is body height above
        //   ground, state[6] is roll, state[7] is pitch
        dead = alive < 0.f;

        potentialOld = potential;
        potential = -walkTargetDist / dt;
        if (potentialOld > 1e9)
        {
            potentialOld = potential;
        }

        float progress = potential - potentialOld;
        progress = min(progress, 1.f / 0.3f);
        if (progress > 100.f)
        {
            printf("progress is infinite %f %f %f \n", progress, potential,
                   potentialOld);
        }

        float electricityCostCurrent = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < mNumMotors; i++)
        {
            float vv =
                abs(action[i] * jointSpeedsA[i]); // Take motor power into account
            if (!isfinite(vv))
            {
                printf("vv at %d is infinite, vv = %lf, ctl = %lf, js =%lf\n", i, vv,
                       action[i], jointSpeedsA[i]);
            }

            if (!isfinite(action[i]))
            {
                printf("action at %d is infinite\n", i);
            }

            if (!isfinite(jointSpeedsA[i]))
            {
                printf("jointSpeeds at %d is infinite\n", i);
            }

            sum += vv;
        }

        if (!isfinite(sum))
        {
            printf("Sum of ctl*joint_speed is infinite!\n");
        }

        electricityCostCurrent += electrCost * sum / (float)mNumMotors;

        sum = 0.0f;
        for (int i = 0; i < mNumMotors; i++)
        {
            sum += action[i] * action[i];
        }

        if (!isfinite(sum))
        {
            printf("Sum of ctl^2 is infinite!\n");
        }

        electricityCostCurrent += stallTorqCost * sum / (float)mNumMotors;

        /* float jointsAtLimitCostCurrent = */
        /*     jointsAtLimitCost * (float)numJointsAtLimitA; */

        float feetCollisionCostCurrent = 0.0f;
        if (numCollideOther[a] > 0)
        {
            feetCollisionCostCurrent += footCollisionCost;
        }

        /* float headingRew = 0.5f * ((heading > 0.8f) ? 1.f : heading / 0.8f) + */
        /*                    upVecWeight * ((upVec > 0.93f) ? 1.f : 0.f); */
        /* float dzRew = -0.5f * abs(state[2] - prevZ[a]); */
        prevZ[a] = state[2];

        float correctSpeedScalar = 2.f;
        // targetSpeed = action[8];
        float incorrectSpeedCost = abs(targetSpeed - progress);
        float correctSpeedReward = 100;
        if (incorrectSpeedCost != 0)
        {
            correctSpeedReward = (1 / incorrectSpeedCost) * correctSpeedScalar;
        }
        /* cout << "incorrectSpeedCost " << incorrectSpeedCost << endl; */
        /* cout << progress << endl; */
        /* cout << "alive " << alive << "\tprogress " << progress << "\telectricity
* " << electricityCostCurrent << endl; */
        /* cout << "targetSpeed " << targetSpeed << "\tspeed " << progress <<
* "\treward " << correctSpeedReward << "\tcost " << incorrectSpeedCost <<
* endl; */

        const int numRewards = 3;
        /* progress = min(progress, 1.8f); */
        float rewards[numRewards] = {
            alive,
            /* progress, */
            electricityCostCurrent, correctSpeedReward,
            /* jointsAtLimitCostCurrent, */
            /* feetCollisionCostCurrent, */
            /* headingRew, */
            /* dzRew */
        };

        rew = 0.f;
        for (int i = 0; i < numRewards; i++)
        {
            if (!isfinite(rewards[i]))
            {
                printf("Reward %d is infinite\n", i);
            }
            rew += rewards[i];
        }
    }

    virtual void FinalizeContactInfo()
    {
        rigidContacts.map();
        rigidContactCount.map();
        int numContacts = rigidContactCount[0];

        // check if we overflowed the contact buffers
        if (numContacts > g_solverDesc.maxRigidBodyContacts)
        {
            printf("Overflowing rigid body contact buffers (%d > %d). Contacts will "
                   "be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n",
                   numContacts, g_solverDesc.maxRigidBodyContacts);
            numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
        }

        /* for (auto fc : feetContact) */
        /* { */
        /*   fc = 0.f; */
        /* } */

        NvFlexRigidContact *ct = &(rigidContacts[0]);
        for (int i = 0; i < numContacts; ++i)
        {
            if ((ct[i].body0 >= 0) &&
                (footFlag[ct[i].body0] >= 0 || kneeFlag[ct[i].body0] >= 0) &&
                (ct[i].lambda > 0.f))
            {
                //	cout << "lambda = " << ct[i].lambda << endl;
                if (ct[i].body1 < 0 && footFlag[ct[i].body0] >= 0 &&
                    kneeFlag[ct[i].body0] < 0)
                {
                    // foot contact with ground
                    int ff = footFlag[ct[i].body0];
                    feetContact[ff] += ct[i].lambda;
                }
                else
                {
                    // foot contact with something other than ground or knee contact with
                    // the ground
                    int ff = footFlag[ct[i].body0];
                    numCollideOther[ff / numFeet]++;
                }
            }

            if ((ct[i].body1 >= 0) &&
                (footFlag[ct[i].body1] >= 0 || kneeFlag[ct[i].body1] >= 0) &&
                (ct[i].lambda > 0.f))
            {
                if (ct[i].body0 < 0 && footFlag[ct[i].body1] >= 0 &&
                    kneeFlag[ct[i].body1] < 0)
                {
                    // foot contact with ground
                    int ff = footFlag[ct[i].body1];
                    feetContact[ff] += ct[i].lambda;
                }
                else
                {
                    // foot contact with something other than ground or knee contact with
                    // the ground
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
                walkTargetX[a] = Randf(-maxX, maxX) - bodyXYZ[a].x;
                walkTargetY[a] = Randf(-maxY, maxY) - bodyXYZ[a].y;
            }

            flagRunSteps[a] = 0;
        }
        else
        {
            walkTargetX[a] = 1000.f;
            walkTargetY[a] = 0.f;
        }
    }

    // Not used, implemented as it's an abstract function
    float AliveBonus(float z, float pitch)
    {
        if (z > terminationZ)
        {
            return 1.f; // Lower value due to the additional reward contribution from
                        // following the target velocity
        }
        else
        {
            return -1.f;
        }
    }

    virtual float AliveBonus(float height, float roll, float pitch)
    {
        bool debugging = true;
        if (!debugging)
        {
            float tooHigh = 1;
            bool heightOk = height > terminationZ && height < tooHigh;
            /* cout << "height " << height << " heightOk " << heightOk << endl; */
            bool rollPitchOk =
                abs(abs(roll) - kPi) < 0.5 * kPi && abs(pitch) < 0.5 * kPi;
            if (heightOk && rollPitchOk)
            {
                float aliveRew = 0.5f;
                return aliveRew;
            }
            else
            {
                return -100.f;
            }
        }
        else
        {
            return 1.f;
        }
    }
};
