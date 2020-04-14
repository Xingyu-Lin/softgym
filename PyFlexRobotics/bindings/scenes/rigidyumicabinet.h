#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"

class RigidYumiCabinet : public Scene
{
public:

    URDFImporter* robot_urdf;
    URDFImporter* cabinet_urdf;

    vector<Transform> rigidTrans;
    map<string, int> jointMap;
    map<string, int> activeJointMap;
	int effectorJoint0;
	int effectorJoint1;
	
	int fingerLeft;
	int fingerRight;
	float fingerWidth = 0.03f;
	float fingerWidthMin = 0.0f;
	float roll, pitch, yaw;

    RigidYumiCabinet()
    {
		roll = 0.0f;
		pitch = 0.0f;
		yaw = -90.0f;
        rigidTrans.clear();
	
		robot_urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi.urdf");
		cabinet_urdf = new URDFImporter("../../data/", "sektion_cabinet_model/urdf/sektion_cabinet.urdf");
        
		Transform robot_transform(Vec3(0.0f, 0.025f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
		Transform cabinet_transform(Vec3(0.0f, 0.025f, 0.3f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
//		Transform cabinet_transform(Vec3(0.0f, 0.35f, 1.12f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

        robot_urdf->AddPhysicsEntities(robot_transform, hiddenMaterial, true, 1000.0f, 0.0f, 1e1f, 0.01f, 20.7f, 7.0f, false);
        cabinet_urdf->AddPhysicsEntities(cabinet_transform, hiddenMaterial, true, 1000.0f, 0.0f, 1e1f, 0.01f, 20.7f, 7.0f, false);
        
		for (int i = 0; i < (int)robot_urdf->joints.size(); i++)
        {
            URDFJoint* j = robot_urdf->joints[i];
            NvFlexRigidJoint& joint = g_buffers->rigidJoints[robot_urdf->jointNameMap[j->name]];
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

            }
        }


        for (int i = 0; i < (int)cabinet_urdf->joints.size(); i++)
        {
            URDFJoint* j = cabinet_urdf->joints[i];
            NvFlexRigidJoint& joint = g_buffers->rigidJoints[cabinet_urdf->jointNameMap[j->name]];
            if (j->type == URDFJoint::REVOLUTE)
            {
                joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-8f;	// 10^6 N/m
                joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+3f;	// 5*10^5 N/m/s
            }
            else if (j->type == URDFJoint::PRISMATIC)
            {
//                joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
//                joint.targets[eNvFlexRigidJointAxisX] = 0.02f;
                joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f;
                joint.damping[eNvFlexRigidJointAxisX] = 1.e+3f;

            }
        }


        // fix base in place, todo: add a kinematic body flag?
        g_buffers->rigidBodies[0].invMass = 0.0f;
        (Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();

        // fix the cabinet
        g_buffers->rigidBodies[22].invMass = 0.0f;
        (Matrix33&)g_buffers->rigidBodies[22].invInertia = Matrix33();

		NvFlexRigidJoint handLeft = g_buffers->rigidJoints[robot_urdf->jointNameMap["gripper_l_joint"]];
		NvFlexRigidJoint handRight = g_buffers->rigidJoints[robot_urdf->jointNameMap["gripper_r_joint"]];

        // set up end effector0 targets
		{
			NvFlexRigidJoint joint;
			NvFlexMakeFixedJoint(&joint, -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(0.2f, 0.7f, 0.5f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -kPi*0.5f)), NvFlexMakeRigidPose(0,0));
			for (int i = 0; i < 6; ++i)
			{
				joint.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
				joint.damping[i] = 1.e+3f;
				//effectorJoint0.maxIterations = 30;
			}

			effectorJoint0 = g_buffers->rigidJoints.size();
			g_buffers->rigidJoints.push_back(joint);
		}

		// set up end effector1 targets
		{
			NvFlexRigidJoint joint;
			NvFlexMakeFixedJoint(&joint, -1, handRight.body0, NvFlexMakeRigidPose(Vec3(-0.2f, 0.7f, 0.5f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -kPi*0.5f)), NvFlexMakeRigidPose(0,0));
			for (int i = 0; i < 6; ++i)
			{
				joint.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
				joint.damping[i] = 1.e+3f;
				//effectorJoint0.maxIterations = 30;
			}

			effectorJoint1 = g_buffers->rigidJoints.size();
			g_buffers->rigidJoints.push_back(joint);
		}

		fingerLeft = robot_urdf->jointNameMap["gripper_l_joint"];
		fingerRight = robot_urdf->jointNameMap["gripper_l_joint_m"];

        // Set the collision filters
		for (int i = 0; i < (int)g_buffers->rigidShapes.size(); ++i)
		{
		    g_buffers->rigidShapes[i].filter = 0;
		}

//      NvFlexMakeRigidBoxShape(NvFlexRigidShape* shape, int body, float hx, float hy, float hz, NvFlexRigidPose pose)

//        NvFlexRigidShape table;
//        NvFlexMakeRigidBoxShape(&table, -1, 1.0f, 0.01f, 1.0f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 1.2f), Quat()));
//        table.filter = 0;
//        table.material.friction = 0.7f;
//    	  table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));
//        g_buffers->rigidShapes.push_back(table);
//
        g_numSubsteps = 2;
        g_params.numIterations = 30;

        g_params.dynamicFriction = 1.25f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;
		g_params.numPostCollisionIterations = 15;

        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.01f;

        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);

        g_pause = true;

        g_drawPoints = false;
	
    }

    virtual void DoGui()
    {

        NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoint0];

        float targetx = effector0.pose0.p[0];
        float targety = effector0.pose0.p[1];
        float targetz = effector0.pose0.p[2];

		float oroll = roll;
		float opitch = pitch;
		float oyaw = yaw;
		imguiSlider("Gripper X", &targetx, -0.4f, 0.4f, 0.0001f);
        imguiSlider("Gripper Y", &targety, 0.0f, 1.0f, 0.0005f);
        imguiSlider("Gripper Z", &targetz, 0.0f, 1.0f, 0.0005f);
		imguiSlider("Roll", &roll, -180.0f, 180.0f, 0.01f);
		imguiSlider("Pitch", &pitch, -180.0f, 180.0f, 0.01f);
		imguiSlider("Yaw", &yaw, -180.0f, 180.0f, 0.01f);
		float f = 0.1f;

		roll = Lerp(oroll, roll, f);
		pitch = Lerp(opitch, pitch, f);
		yaw = Lerp(oyaw, yaw, f);

        const float smoothing = 0.05f;

        // low-pass filter controls otherwise it is too jerky
        float newx = Lerp(effector0.pose0.p[0], targetx, smoothing);
        float newy = Lerp(effector0.pose0.p[1], targety, smoothing);
        float newz = Lerp(effector0.pose0.p[2], targetz, smoothing);

        effector0.pose0.p[0] = newx;
        effector0.pose0.p[1] = newy;
        effector0.pose0.p[2] = newz;

		Quat q = rpy2quat(roll*kPi / 180.0f, pitch*kPi / 180.0f, yaw*kPi / 180.0f);
		effector0.pose0.q[0] = q.x;
		effector0.pose0.q[1] = q.y;
		effector0.pose0.q[2] = q.z;
		effector0.pose0.q[3] = q.w;

		// mirror effector transform to other side of Yumi
		NvFlexRigidJoint effector1 = g_buffers->rigidJoints[effectorJoint1];
		
		effector1.pose0.p[0] = -newx;
		effector1.pose0.p[1] =  newy;
		effector1.pose0.p[2] =  newz;

		(Quat&)effector1.pose0.q = rpy2quat(roll*kPi / 180.0f, -pitch*kPi / 180.0f, -yaw*kPi / 180.0f);

        g_buffers->rigidJoints[effectorJoint0] = effector0;
		g_buffers->rigidJoints[effectorJoint1] = effector1;

		
        float newWidth = fingerWidth;
        imguiSlider("Finger Width", &newWidth, fingerWidthMin, 0.05f, 0.001f);

        fingerWidth = Lerp(fingerWidth, newWidth, smoothing);

        g_buffers->rigidJoints[fingerLeft].targets[eNvFlexRigidJointAxisX] = fingerWidth;
        g_buffers->rigidJoints[fingerRight].targets[eNvFlexRigidJointAxisX] = fingerWidth;
		
    }

    virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }

};


