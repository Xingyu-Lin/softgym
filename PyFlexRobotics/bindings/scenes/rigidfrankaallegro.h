#pragma once

#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include "../urdf.h"
#include "../deformable.h"

class RigidFrankaAllegro : public Scene
{
public:

	vector<vector<int>> fingerJoints;
	int numFingers, numJointsPerFinger;
	vector<vector<float>> fingerJointTargetAngles; // in degrees
	vector<vector<float>> fingerJointHomeAngles; // in degrees

	float handTranslationDeltas;
	float handRotationDeltas;
	float handGraspValue; // 0 hand in home position -> 1 hand in closed position. this is a very rough control, used to speed up interacting w/ environment.
	float jointRotationDeltas; // in degrees

	int handBodyId, handJointId;
	vector<pair<float, float>> fingerJointAngleLimits, thumbJointAngleLimits;

	bool useGraspControl; // If true, provide 1 slider for grasping. if false, provide sliders for all joints individually.

	static const int frankaNumJoints = 7;
	float frankaTargets[frankaNumJoints] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 20.0f, 0.0f};

    RigidFrankaAllegro()
    {

    	int frankaEndBody;
    	Transform frankaEndPose;

		// load Franka
		{
			Transform gt(Vec3(0.0f, 0.05f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) * QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));

			int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

			const float jointCompliance = 1.e-5f;
			const float jointDamping = 0.0f;
			const float bodyArmature = 1.e-4f;
			const float bodyDamping = 0.0f;
			
		    //new
		    bool useObjForCollision = true;
		    float dilation = 0.000f;
		    float thickness = 0.005f;
		    bool replaceCylinderWithCapsule=true;
		    int slicesPerCylinder = 20;
		    bool useSphereIfNoCollision = true;

		    URDFImporter* urdf = new URDFImporter("../../data/", "franka_description/robots/franka.urdf",
		    	useObjForCollision,
		    	dilation,
		    	thickness,
		    	replaceCylinderWithCapsule,
		    	slicesPerCylinder,
		    	useSphereIfNoCollision);

		     urdf->AddPhysicsEntities(gt, hiddenMaterial, true, true, 5000.0f, jointCompliance, jointDamping, bodyArmature, bodyDamping, 45.0f, false);

		     // fix base
		     g_buffers->rigidBodies[0].invMass =0.0f;
		     (Matrix33&)(g_buffers->rigidBodies[0].invInertia) = Matrix33();

		     frankaEndBody = g_buffers->rigidBodies.size()-1;
		     NvFlexGetRigidPose(&g_buffers->rigidBodies.back(), (NvFlexRigidPose*)&frankaEndPose);

		     for (int i=0; i < g_buffers->rigidJoints.size(); ++i)
		     {
		     	NvFlexRigidJoint& joint = g_buffers->rigidJoints[i];

		     	joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
			    joint.targets[eNvFlexRigidJointAxisTwist] = 0.0f;
			    joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-7f;
			    joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+2f;
				//joint.motorLimit[eNvFlexRigidJointAxisTwist] = 50.0f;
		     }

		     delete urdf;
		}

		// load Allegro
		{
			numFingers = 4; 
			numJointsPerFinger = 4;
			jointRotationDeltas = 1.f;
			handTranslationDeltas = 0.005f;
			handRotationDeltas = 0.05f; // in degrees

			// Set joint home angles (see http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Home_Position_Joint_Angles)
			fingerJointHomeAngles = {
				{ 0.f, -10.f, 45.f, 45.f, },
				{ 0.f, -10.f, 45.f, 45.f, },
				{ 5.f, -5.f, 50.f, 45.f, },
				{ 70.f, 15.f, 15.f, 45.f, },
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
			
			useGraspControl = true;
			handGraspValue = 0.f;

			// load URDF
			int allegroHandBody = g_buffers->rigidBodies.size();
			URDFImporter* urdf = new URDFImporter("../../data", "allegro_hand_description/allegro_hand_description_right.urdf");
			
			// hide collision shapes
			const int hiddenMaterial = AddRenderMaterial(0.5f, 0.5f, 0.5f, true);

			const float jointCompliance = 1.e-5f;
			const float jointDamping = 0.0f;
			const float bodyArmature = 1.e-4f;
			const float bodyDamping = 0.0f;

			Transform handLocalPose = Transform(Vec3(0.0f, 0.0f, 0.1f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi));
			Transform allegroPose = frankaEndPose*handLocalPose;		
			
	        urdf->AddPhysicsEntities(allegroPose, hiddenMaterial, true, false, 1000.0f, jointCompliance, jointDamping, bodyArmature, bodyDamping,
									FLT_MAX, false);

			
			// adjust thickness
			for (int i=0; i < g_buffers->rigidShapes.size(); ++i)
				g_buffers->rigidShapes[i].thickness = 0.0025f;
		
			// Configure finger joints
			ostringstream jointName;
			fingerJoints.resize(numFingers);
			for (int i = 0; i < numFingers; i++)
			{
				fingerJoints[i].resize(numJointsPerFinger);
				for (int j = 0; j < numJointsPerFinger; j++)
				{
					jointName.str("");
					jointName << "joint_" << i * numFingers + j << ".0";
					fingerJoints[i][j] = urdf->jointNameMap[jointName.str()];
					
					g_buffers->rigidJoints[fingerJoints[i][j]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
					g_buffers->rigidJoints[fingerJoints[i][j]].compliance[eNvFlexRigidJointAxisTwist] = 1.e-3f;		// unknown
					g_buffers->rigidJoints[fingerJoints[i][j]].damping[eNvFlexRigidJointAxisTwist] = 0.01f;			// unknown 
					g_buffers->rigidJoints[fingerJoints[i][j]].motorLimit[eNvFlexRigidJointAxisTwist] = 2.0f; 		// actual limit from specs is 0.9Nm or 10Nm in overdrive
				}
			}

			// Set initial target to home angles
			fingerJointTargetAngles.resize(numFingers);
			for (int i = 0; i < numFingers; i++)
			{
				fingerJointTargetAngles[i] = fingerJointHomeAngles[i];
			}

			// Configure hand joint
			NvFlexRigidJoint handJoint;
			NvFlexMakeFixedJoint(&handJoint, frankaEndBody, allegroHandBody, (NvFlexRigidPose&)handLocalPose, NvFlexMakeRigidPose(0, 0));

			handJointId = g_buffers->rigidJoints.size();
			g_buffers->rigidJoints.push_back(handJoint);

			delete urdf;
		}

        g_numSubsteps = 2;
        g_params.numIterations = 6;

        g_params.solverType = eNvFlexSolverPCR;
        g_params.geometricStiffness = 0.0f;
        g_params.frictionMode = eNvFlexFrictionModeFull;
	
        g_params.relaxationFactor = 0.75f;
        g_params.shapeCollisionMargin = 0.01f;

        g_sceneLower = Vec3(-0.5f);
        g_sceneUpper = Vec3(0.5f);

        g_pause = true;

        g_drawPoints = false;
        g_drawCloth = false;

		// Box object
		float scale = 0.09f; 
		float density = 250.0f;

		Mesh* boxMesh = ImportMesh("../../data/box.ply"); 
		boxMesh->Transform(ScaleMatrix(scale));

		for (int i = 0; i < 1; ++i)
		{
			NvFlexTriangleMeshId shapeId = CreateTriangleMesh(boxMesh, 0.00125f);

			NvFlexRigidShape shape;
			NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
			shape.filter = 0x0;
			shape.material.friction = 1.0f;
			shape.material.torsionFriction = 0.1;
			shape.material.rollingFriction = 0.0f;
			shape.thickness = 0.001f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.2f * (float)i, 0.35f, 0.6f), Quat(), &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
		}

		// table
		NvFlexRigidShape box;
		NvFlexMakeRigidBoxShape(&box, -1, 0.5f, 0.1f, 0.5f, NvFlexMakeRigidPose(Vec3(0.0f, 0.1f, 1.0f),0));
		box.filter = 0;
		box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.35f, 0.45f, 0.65f)));

		g_buffers->rigidShapes.push_back(box);
    }

	virtual void CenterCamera()
	{
		g_camPos = Vec3(-0.010622f, 0.555237f, 0.584409f);
		g_camAngle = Vec3(0.017453f, -0.263545f, 0.000000f);
	}

	virtual void DoGui()
	{

		//--------------------------
		// Allegro controls

		if (imguiCheck("Grasp Control", &useGraspControl))
			useGraspControl = !useGraspControl;

		if (useGraspControl)
		{
			// Hand opening
			imguiSlider("Grasp", &handGraspValue, 0.f, 1.f, 0.01f);
		}
		else
		{
			// Finger joints
			char strBuf[50];
			for (int i = 0; i < numFingers; i++)
			{
				sprintf(strBuf, "Finger %d", i);
				imguiLabel(strBuf);
				for (int j = 0; j < numJointsPerFinger; j++)
				{
					float jointAngle = fingerJointTargetAngles[i][j];
					sprintf(strBuf, "Link %d", j);

					float lo, hi;
					if (j == 3)
					{
						lo = thumbJointAngleLimits[j].first;
						hi = thumbJointAngleLimits[j].second;
					}
					else
					{
						lo = fingerJointAngleLimits[j].first;
						hi = fingerJointAngleLimits[j].second;
					}

					imguiSlider(strBuf, &jointAngle, RadToDeg(lo), RadToDeg(hi), jointRotationDeltas);
					fingerJointTargetAngles[i][j] = jointAngle;
				}
			}
		}

		//-----------------------------
		// Franka controls

        imguiSlider("Joint 0", &frankaTargets[0],RadToDeg(-2.8973f), RadToDeg(2.8973f), 0.01f);
        imguiSlider("Joint 1", &frankaTargets[1],RadToDeg(-1.7628f), RadToDeg(1.7628f), 0.01f);
        imguiSlider("Joint 2", &frankaTargets[2],RadToDeg(-2.8973f), RadToDeg(2.8973f), 0.01f);
        imguiSlider("Joint 3", &frankaTargets[3],RadToDeg(-3.0718f), RadToDeg(-0.0698f), 0.01f);
        imguiSlider("Joint 4", &frankaTargets[4],RadToDeg(-2.8973f), RadToDeg(2.8973f), 0.01f);
        imguiSlider("Joint 5", &frankaTargets[5],RadToDeg(-0.0175f), RadToDeg(3.7525f), 0.01f);
        imguiSlider("Joint 6", &frankaTargets[6],RadToDeg(-2.8973f), RadToDeg(2.8973f), 0.01f);

        for (int i = 0; i < frankaNumJoints; i++)
        {
			const float smoothing = 0.05f;

			float current = g_buffers->rigidJoints[i].targets[eNvFlexRigidJointAxisTwist];
			float target = DegToRad(frankaTargets[i]);

			g_buffers->rigidJoints[i].targets[eNvFlexRigidJointAxisTwist] = Lerp(current, target, smoothing);
		}
	}

	virtual void Update()
	{
		const float smoothing = 0.05f;

		// Grasp
		if (useGraspControl)
		{
			for (int i = 0; i < 4; i++)
			{
				for (int j = 1; j < 4; j++)
				{
					if (i == 3 && j == 1) continue; // skip the 2nd link for thumb
					fingerJointTargetAngles[i][j] = (RadToDeg(fingerJointAngleLimits[j].second) 
												- fingerJointHomeAngles[i][j]) * handGraspValue 
												+ fingerJointHomeAngles[i][j];
				}
			}
		}
		 
		// clear control torques
		for (int i=0; i < g_buffers->rigidBodies.size(); ++i)
			(Vec3&)g_buffers->rigidBodies[i].torque = Vec3(0.0f);

		// Finger Joints
		for (int i = 0; i < numFingers; i++)
		{
			float jointScale = 1.f;
			for (int j = 0; j < numJointsPerFinger; j++)
			{
				float currentAngleRad = g_buffers->rigidJoints[fingerJoints[i][j]].targets[eNvFlexRigidJointAxisTwist];
				float nextAngleRad = Lerp(currentAngleRad, DegToRad(fingerJointTargetAngles[i][j]), smoothing);

				g_buffers->rigidJoints[fingerJoints[i][j]].targets[eNvFlexRigidJointAxisTwist] 
							= jointScale * nextAngleRad + (1.f - jointScale) * currentAngleRad;						

				jointScale -= 0.16f * (float)j;
			}
		}


	}

	virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }
};


