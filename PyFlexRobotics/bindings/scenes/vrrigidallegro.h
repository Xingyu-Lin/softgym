#pragma once

#include "../vr/vr.h"

#if FLEX_VR

#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include "../urdf.h"

class VRRigidAllegro : public Scene
{
public:

	enum Mode
	{
		eRigidCube,
		eRigidBunny,
		eFemFrog
	};

	Mode mode;

	Mesh mesh;

	vector<vector<int>> fingerJoints;
	int numFingers, numJointsPerFinger;
	vector<vector<float>> fingerJointTargetAngles; // in degrees
	vector<vector<float>> fingerJointHomeAngles; // in degrees

	float handTranslationDeltas;
	float handRotationDeltas;
	float handGraspValue; // 0 hand in home position -> 1 hand in closed position. this is a very rough control, used to speed up interacting w/ environment.
	float jointRotationDeltas; // in degrees

	Vec3 baseLinkTargetTranslation, baseLinkTargetRpy, baseLinkCurrentRpy; // in degrees

	int handBodyId, handJointId;
	vector<pair<float, float>> fingerJointAngleLimits, thumbJointAngleLimits;

	bool useGraspControl; // If true, provide 1 slider for grasping. if false, provide sliders for all joints individually.

	VRRigidAllegro(Mode mode) : mode(mode)
    {
		// default vals
		numFingers = 4; 
		numJointsPerFinger = 4;
		jointRotationDeltas = 1.f;
		handTranslationDeltas = 0.005f;
		handRotationDeltas = .05f; // in degrees

		// Set joint home angles (see http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Home_Position_Joint_Angles)
		fingerJointHomeAngles = {
			{ 0.f, -10.f, 45.f, 45.f, },
			{ 0.f, -10.f, 45.f, 45.f, },
			{ 5.f, -5.f, 50.f, 45.f, },
			{ 50.f, 25.f, 15.f, 45.f, },
		};
		handGraspValue = 0.f;

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

		// load URDF
		handBodyId = g_buffers->rigidBodies.size();
		URDFImporter* urdf = new URDFImporter("../../data", "allegro_hand_description/allegro_hand_description_right.urdf");
        
		baseLinkTargetTranslation = Vec3(0.0f, 0.5f, 0.0f);
		baseLinkTargetRpy = Vec3(0.f, 0.f, -90.f);
		baseLinkCurrentRpy = Vec3(baseLinkTargetRpy);
		Quat baseLinkHomeQuat = rpy2quat(baseLinkTargetRpy[0], baseLinkTargetRpy[1], baseLinkTargetRpy[2]);
		Transform gt(baseLinkTargetTranslation, baseLinkHomeQuat);

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.5f, 0.5f, 0.5f, true);

		const float joinCompliance = 1.e-6f;
		const float jointDamping = 1.e+1f;
		const float bodyArmature = 1.e-4f;
		const float bodyDamping = 5.0f;
		
        urdf->AddPhysicsEntities(gt, hiddenMaterial, true, false, 1000.0f, joinCompliance, jointDamping, bodyArmature, bodyDamping, 
								kPi * 8.0f, false);
		
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
				g_buffers->rigidJoints[fingerJoints[i][j]].compliance[eNvFlexRigidJointAxisTwist] = 5e-3f;
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
		NvFlexMakeFixedJoint(&handJoint, -1,handBodyId, NvFlexMakeRigidPose(baseLinkTargetTranslation, baseLinkHomeQuat), NvFlexMakeRigidPose(0, 0));
		for (int i = 0; i < 6; ++i)
		{
			handJoint.compliance[i] = 1.e-4f;
			handJoint.damping[i] = 1.e+3f;
		}
		handJointId = g_buffers->rigidJoints.size();
		g_buffers->rigidJoints.push_back(handJoint);

        g_numSubsteps = 4;
        g_params.numIterations = 40;
	//	g_params.numPostCollisionIterations = 15;

        g_params.dynamicFriction = 0.8f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
    //    g_params.sleepThreshold = 0.02f;
		
        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.001f;

        g_sceneLower = Vec3(-0.5f);
        g_sceneUpper = Vec3(0.5f);

        g_pause = true;

        g_drawPoints = false;
        g_drawCloth = false;

		delete urdf;

		if (mode == eRigidCube)
		{
			// Box object
			float scale = 0.09f; // 1.f;
			float density = 250.0f;

			Mesh* boxMesh = ImportMesh("../../data/box.ply"); //ImportMesh("../../data/bunny.ply");
			boxMesh->Transform(ScaleMatrix(scale));

			for (int i = 0; i < 1; ++i)
			{
				NvFlexTriangleMeshId shapeId = CreateTriangleMesh(boxMesh, 0.00125f);

				NvFlexRigidShape shape;
				NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
				shape.filter = 0x0;
				shape.material.friction = 1.0f;
				shape.material.torsionFriction = 0.1f;
				shape.material.rollingFriction = 0.0f;
				shape.thickness = 0.001f;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.2f * (float)i, 0.05f, 0.f), Quat(), &shape, &density, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);
			}
		}
		else if (mode == eRigidBunny)
		{
			// Bunny
			float scale = 1.f;
			float density = 250.0f;

			Mesh* boxMesh = ImportMesh("../../data/bunny.ply");
			boxMesh->Transform(ScaleMatrix(scale));

			for (int i = 0; i < 1; ++i)
			{
				NvFlexTriangleMeshId shapeId = CreateTriangleMesh(boxMesh, 0.00125f);

				NvFlexRigidShape shape;
				NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
				shape.filter = 0x0;
				shape.material.friction = 1.0f;
				shape.material.torsionFriction = 0.1f;
				shape.material.rollingFriction = 0.0f;
				shape.thickness = 0.001f;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.2f * (float)i, 0.05f, 0.f), Quat(), &shape, &density, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);
			}
		}
		else if (mode == eFemFrog)
		{
			const float invMass = 50.f;

			g_params.numIterations = 45;

			CreateTetMesh("../../data/froggy.tet", Vec3(-0.05f, 0.05f, 0.f), 0.26f, invMass, 0,
				NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));

			g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

			g_tetraMaterials.resize(0);
			g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+8f, 0.4f, 0.0f));
		}
    }

	virtual void DoGui()
	{
		// Hand base location
		float x = baseLinkTargetTranslation[0];
		float y = baseLinkTargetTranslation[1];
		float z = baseLinkTargetTranslation[2];
		float roll = baseLinkTargetRpy[0];
		float pitch = baseLinkTargetRpy[1];
		float yaw = baseLinkTargetRpy[2];
		
		imguiLabel("Hand");
		imguiSlider("X", &x, -0.5f, 0.5f, handTranslationDeltas);
		imguiSlider("Y", &y, 0.0f, 0.5f, handTranslationDeltas);
		imguiSlider("Z", &z, -0.5f, 0.5f, handTranslationDeltas);
		imguiSlider("roll", &roll, -240.f, 240.f, handRotationDeltas);
		imguiSlider("pitch", &pitch, -180.f, 180.f, handRotationDeltas);
		imguiSlider("yaw", &yaw, -180.f, 180.f, handRotationDeltas);

		baseLinkTargetTranslation[0] = x;
		baseLinkTargetTranslation[1] = y;
		baseLinkTargetTranslation[2] = z;
		baseLinkTargetRpy[0] = roll;
		baseLinkTargetRpy[1] = pitch;
		baseLinkTargetRpy[2] = yaw;

		if (useGraspControl)
		{
			// Hand opening
			imguiSlider("grasp", &handGraspValue, 0.f, 1.f, 0.01f);
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
					imguiLabel(strBuf);

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
	}

	size_t selectedController = VrSystem::InvalidControllerIndex;

	virtual void Update()
	{
		Quat newVrOrientation;
		float newVrGrasp;
		bool haveVrData = false;

		if (selectedController != VrSystem::InvalidControllerIndex)
		{
			VrControllerState state;
			if (g_vrSystem->GetControllerState(selectedController, state))
			{
				if (state.buttonPressed)
				{
					selectedController = VrSystem::InvalidControllerIndex;
				}
				else
				{
					baseLinkTargetTranslation = state.pos;

					Matrix33 rotationMat;
					for (int i = 0; i < 3; i++)
					{
						for (int j = 0; j < 3; j++)
						{
							rotationMat.cols[i][j] = state.orientation.columns[i][j];
						}
					}
					newVrOrientation = Quat(rotationMat);

					newVrOrientation = newVrOrientation * QuatFromAxisAngle(Vec3(0, 1, 0), kPi) * QuatFromAxisAngle(Vec3(0, 0, 1), -kPi * .5f);
					newVrGrasp = state.triggerValue;

					haveVrData = true;
					/*
					NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoint];

					NvFlexRigidPose newPose = NvFlexMakeRigidPose(state.pos, Quat());

					g_buffers->rigidShapes[debugSphere].pose = newPose;
					//	NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

					float currentWidth = fingerWidth;

					float newFingerWidth = fingerWidthMax + (fingerWidthMin - fingerWidthMax) * state.triggerValue;
					fingerWidth = Lerp(currentWidth, newFingerWidth, 0.05);

					const float smoothing = 0.1f;

					Vec3 desiredPos = state.pos;

					// low-pass filter controls otherwise it is too jerky
					float newx = Lerp(effector0.pose0.p[0], desiredPos[0], smoothing);
					float newy = Lerp(effector0.pose0.p[1], desiredPos[1], smoothing);
					float newz = Lerp(effector0.pose0.p[2], desiredPos[2], smoothing);

					effector0.pose0.p[0] = newx;
					effector0.pose0.p[1] = newy;
					effector0.pose0.p[2] = newz;

					g_buffers->rigidJoints[effectorJoint] = effector0;
					*/
				}
			}
		}
		else
		{
			for (size_t i = 0, contSize(g_vrSystem->GetNumControllers()); i < contSize; ++i)
			{
				VrControllerState state;
				if (g_vrSystem->GetControllerState(i, state) && state.buttonPressed)
				{
					selectedController = i;
				}
			}
		}

		//const float smoothing = 0.05f;
		const float smoothing = 1.0f;

		// Hand
		NvFlexRigidJoint handJoint = g_buffers->rigidJoints[handJointId];
		for (int i = 0; i < 3; i++)
		{
			handJoint.pose0.p[i] = Lerp(handJoint.pose0.p[i], baseLinkTargetTranslation[i], smoothing);
			baseLinkCurrentRpy[i] = Lerp(baseLinkCurrentRpy[i], baseLinkTargetRpy[i], smoothing);
		}

		/*Quat currentQuat = rpy2quat(DegToRad(baseLinkCurrentRpy[0]), DegToRad(baseLinkCurrentRpy[1]), DegToRad(baseLinkCurrentRpy[2]));
		handJoint.pose0.q[0] = currentQuat.x;
		handJoint.pose0.q[1] = currentQuat.y;
		handJoint.pose0.q[2] = currentQuat.z;
		handJoint.pose0.q[3] = currentQuat.w;*/
		if (haveVrData)
		{
			handJoint.pose0.q[0] = newVrOrientation.x;
			handJoint.pose0.q[1] = newVrOrientation.y;
			handJoint.pose0.q[2] = newVrOrientation.z;
			handJoint.pose0.q[3] = newVrOrientation.w;
		}
		g_buffers->rigidJoints[handJointId] = handJoint;

		if (haveVrData)
		{
			// Grasp
			if (useGraspControl)
			{
				for (int i = 0; i < 4; i++)
				{
					for (int j = 1; j < 4; j++)
					{
						if (i == 3 && j == 1) continue; // skip the 2nd link for thumb
						fingerJointTargetAngles[i][j] = (RadToDeg(fingerJointAngleLimits[j].second)
							- fingerJointHomeAngles[i][j]) * newVrGrasp
							+ fingerJointHomeAngles[i][j];
					}
				}
			}
		}

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

	virtual void Draw(int pass)
	{
		if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i = 0; i < g_buffers->triangles.size(); ++i)
				mesh.m_indices[i] = g_buffers->triangles[i];

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i = 0; i < g_buffers->tetraIndices.size(); i += 4)
			{
				float vonMises = fabsf(g_buffers->tetraStress[i / 4]);

				//printf("%f\n", vonMises);

				averageStress[g_buffers->tetraIndices[i + 0]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i + 1]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i + 2]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i + 3]] += Vec2(vonMises, 1.0f);

				rangeMin = Min(rangeMin, vonMises);
				rangeMax = Max(rangeMax, vonMises);
			}

			//printf("%f %f\n", rangeMin,rangeMax);

			rangeMin = 0.0f;//Min(rangeMin, vonMises);
			rangeMax = 0.5f;//Max(rangeMax, vonMises);

			for (int i = 0; i < g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x / averageStress[i].y);
				/*
				if (g_buffers->tetraMaterials[i] == 0)
				mesh.m_colours[i] = Colour::kGreen;
				else
				mesh.m_colours[i] = Colour::kRed;
				*/
			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);

		if (pass == 0)
		{

			SetFillMode(true);

			DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.001f);

			SetFillMode(false);

		}
	}
};

#endif // #if FLEX_VR