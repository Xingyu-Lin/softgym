#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"
#include "rlbase.h"


class RLCarter : public CarterBase
{
public:
    URDFImporter* urdf;

    bool sampleInitStates;
    map<string, int> jointMap;
    map<string, int> activeJointMap;
    bool flagRun = false;

    vector<Vec3> robotPoses;
	vector<string> motors;
    int hiddenMaterial;
	Mesh mesh;

	float angularVelocity;
	float left_angularVelocity;
	float right_angularVelocity;
	float front_angularVelocity;
	float pivot_angularVelocity;
	virtual void AddAgentBodiesJointsCtlsPowers(int i, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower){}
    void AddAgentBodiesJointsCtlsPowers(int ai, Transform gt, vector<pair<int, NvFlexRigidJointAxis>>& ctrl, vector<float>& mpower,
												int rbIt, int jointsIt)
	{
		int startShape = g_buffers->rigidShapes.size(); //Try removing this
		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, false,  10000.0f, 0.0f, 10.f, 0.01f, 10.f, 6.f, false);
		int endShape = g_buffers->rigidShapes.size();
		for (int i = startShape; i < endShape; i++)
		{
			g_buffers->rigidShapes[i].thickness = 0.001f;
		}
	}

	RLCarter()
    {
		mNumAgents = 2;
		numPerRow = 10;
		spacing = 6.f;
		mDoLearning = false;
	//	controlType = eVelocity;
		mNumActions = 2; //angular and linear velocity
		mNumObservations = 2;
		mMaxEpisodeLength = 1000;

		g_params.shapeCollisionMargin = 0.01f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(8.6f, 0.9f, 3.5f);

		g_pause = false;
        g_params.solverType = eNvFlexSolverPCR;
        g_numSubsteps = 2;
        g_params.numIterations = 4;
		g_params.numInnerIterations = 15;
		g_params.relaxationFactor = 0.75f;
        g_params.shapeCollisionMargin = 0.0015f;
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
		// for (int i=0; i < g_buffers->rigidShapes.size()-1; ++i)
		// {
		// 	g_buffers->rigidShapes[i].filter = 0;
		// }
		if (mDoLearning)
		{
			init();
		}

		angularVelocity = 0.f;
		left_angularVelocity = 0.f;
		right_angularVelocity = 0.f;
		front_angularVelocity = 0.f;
		pivot_angularVelocity = 0.f;

		for (int i = 0; i < g_buffers->rigidJoints.size(); i++)
		{
			g_buffers->rigidJoints[i].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeVelocity;
			g_buffers->rigidJoints[i].targets[eNvFlexRigidJointAxisTwist] = -DegToRad(180.0f);
		}
		cout << "Total joints :" << g_buffers->rigidJoints.size() << endl;
    }

    void LoadEnv()
	{
		// initialize data structures
		ctrls.resize(mNumAgents);
		motorPower.resize(mNumAgents);
        robotPoses.resize(mNumAgents);

        motors = {
            "left_wheel",
            "right_wheel",
        };
        
        int rbIt = 0;
        int jointsIt = 0;
        hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
        urdf = new URDFImporter("../../data", "carter/urdf/carter.urdf");		
        
		// set up each env
		for (int ai = 0; ai < mNumAgents; ++ai)
		{
			Vec3 robotPos = Vec3((ai % numPerRow) * spacing, 0.0f, (ai / numPerRow) * spacing);
			Transform gt(robotPos + Vec3(0.f, 0.25f, 0.f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));
			robotPoses[ai] = robotPos;

			int begin = g_buffers->rigidBodies.size();
			AddAgentBodiesJointsCtlsPowers(ai, gt, ctrls[ai], motorPower[ai], rbIt, jointsIt);
			int end = g_buffers->rigidBodies.size();
			agentBodies.push_back(make_pair(begin, end));
            
            rbIt = g_buffers->rigidBodies.size();
            jointsIt = g_buffers->rigidJoints.size();

			agentOffsetInv.push_back(Inverse(gt));
			agentOffset.push_back(gt);
		}
	}

	~RLCarter()
	{
		if (urdf)
		{
			delete urdf;
		}
	}
	virtual void ApplyCustomVelocityControl(int agent, float df=1.0f)
	{
		float* actions = GetAction(agent);
		cout<<"Applying action"<<actions[0]<<"&"<<actions[1]<<" to "<<agent<<endl;
		g_buffers->rigidJoints[agent * 4].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeVelocity;
		g_buffers->rigidJoints[agent * 4 + 1].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeVelocity;
		g_buffers->rigidJoints[agent * 4].targets[eNvFlexRigidJointAxisTwist] = actions[0];
		g_buffers->rigidJoints[agent * 4 + 1].targets[eNvFlexRigidJointAxisTwist] = actions[1];
		g_buffers->rigidJoints[agent * 4 + 2].targets[eNvFlexRigidJointAxisTwist] = 0.f;
		g_buffers->rigidJoints[agent * 4 + 3].targets[eNvFlexRigidJointAxisTwist] = 0.f;
		//g_buffers->rigidJoints[agent * 4 + 2].targets[eNvFlexRigidJointAxisTwist] = 0;
		//g_buffers->rigidJoints[agent * 4 + 3].targets[eNvFlexRigidJointAxisTwist] = 0;
	}
	virtual void ApplyCarterActions()
	{
		cout<<"Entered APPLY CARTER"<<endl;
		// Do whatever needed with the action to transition to the next state
		for (int ai = 0; ai < mNumAgents; ai++)
		{
			ApplyCustomVelocityControl(ai);
		}
	}
	virtual void Simulate()
	{
		cout<<"Entered SIMULATE"<<endl;
		ApplyCarterActions();
		//ApplyForces();
		SimulateMapUnmap();
	}
    void ResetAgent(int a)
    {
    }

    virtual void LockWrite()
    {
    }

    virtual void UnlockWrite()
    {
    }

    virtual bool IsSkipSimulation()
    {
        return true;
    }

    virtual void FinalizeContactInfo()
    {
    }

	virtual void resetTarget(int a, bool firstTime = true)
	{
	}

    virtual void DoGui()
    {
		if(!mDoLearning)
		{
			imguiSlider("Left Wheel", &left_angularVelocity, -2.f * kPi, 2.f * kPi, 0.01f);
			imguiSlider("Right Wheel", &right_angularVelocity, -2.f * kPi, 2.f * kPi, 0.01f);
			imguiSlider("Front Wheel", &front_angularVelocity, -2.f * kPi, 2.f * kPi, 0.01f);
			imguiSlider("Pivot", &pivot_angularVelocity, -2.f * kPi, 2.f * kPi, 0.01f);
			for (int i = 0; i < mNumAgents; i++)
			{
				g_buffers->rigidJoints[i * 4].targets[eNvFlexRigidJointAxisTwist] = left_angularVelocity;
				g_buffers->rigidJoints[i * 4 + 1].targets[eNvFlexRigidJointAxisTwist] = right_angularVelocity;
				g_buffers->rigidJoints[i * 4 + 2].targets[eNvFlexRigidJointAxisTwist] = front_angularVelocity;
				g_buffers->rigidJoints[i * 4 + 3].targets[eNvFlexRigidJointAxisTwist] = pivot_angularVelocity;
			}
		}
    }

    virtual void DoStats()
    {
    }

    virtual void Update()
    {
        //Remain empty
    }

	virtual void PostUpdate()
	{
		// joints are not read back by default
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
	}

	virtual void Draw(int pass)
	{
	}

	virtual float AliveBonus(float z, float pitch)
	{
		return -1.f;
	}

    void ComputeRewardAndDead(int a, float* action, float* state, float& rew, bool& dead) 
    {

    }

    void ClearContactInfo(){};
};


