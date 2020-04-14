#pragma once

#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include "../urdf.h"
#include "../deformable.h"

class RigidAllegro : public Scene
{
public:

	enum Mode
	{
		eDexnet1,
		eDexnet2,
		eDexnet3,
		eDexnet4,
		eDexnet5,
		eDexnet6,
		eDexnet7,
		eDexnet8,
		eDexnet9,
		eDexnet10,
		eDexnet11,
		eDexnet12,
		eRigidCube,
		eRigidBunny,
		eSoftTomato,
		eSoftFish,
		eSoftBread,
		eSoftSandwich,
		eSoftCube,
		eSoftSphere,
		eCloth,
		eRope
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

	DeformableMesh* deformable = NULL;

	bool drawMesh = true;

    RigidAllegro(Mode mode) : mode(mode)
    {
		// default vals
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
		handBodyId = g_buffers->rigidBodies.size();
		URDFImporter* urdf = new URDFImporter("../../data", "allegro_hand_description/allegro_hand_description_right.urdf");

       
		baseLinkTargetTranslation = Vec3(0.0f, 0.5f, 0.0f);
		baseLinkTargetRpy = Vec3(-90.f, 0.f, -90.f);
		baseLinkCurrentRpy = Vec3(baseLinkTargetRpy);
		Quat baseLinkHomeQuat = rpy2quat(baseLinkTargetRpy[0], baseLinkTargetRpy[1], baseLinkTargetRpy[2]);
		Transform gt(baseLinkTargetTranslation, baseLinkHomeQuat);

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.5f, 0.5f, 0.5f, true);

		const float jointCompliance = 1.e-5f;
		const float jointDamping = 0.0f;
		const float bodyArmature = 1.e-4f;
		const float bodyDamping = 0.0f;
		
        urdf->AddPhysicsEntities(gt, hiddenMaterial, true, false, 1000.0f, jointCompliance, jointDamping, bodyArmature, bodyDamping, 
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
		NvFlexMakeFixedJoint(&handJoint, -1,handBodyId, NvFlexMakeRigidPose(baseLinkTargetTranslation, baseLinkHomeQuat), NvFlexMakeRigidPose(0, 0));
		for (int i = 0; i < 6; ++i)
		{
			handJoint.compliance[i] = 1.e-4f;
			handJoint.damping[i] = 1.e+2f;
		}
		handJointId = g_buffers->rigidJoints.size();
		g_buffers->rigidJoints.push_back(handJoint);

        g_numSubsteps = 2;
        g_params.numIterations = 8;

        g_params.solverType = eNvFlexSolverPCR;
        g_params.geometricStiffness = 0.0f;

        g_params.dynamicFriction = 0.8f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 0.0f;
		
        g_params.relaxationFactor = 0.75f;
        g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.01f;

        g_sceneLower = Vec3(-0.5f);
        g_sceneUpper = Vec3(0.5f);

        g_pause = true;

        g_drawPoints = false;
        g_drawCloth = false;

		delete urdf;

		if (mode == eDexnet1 ||
			mode == eDexnet2 ||
			mode == eDexnet3 ||
			mode == eDexnet4 ||
			mode == eDexnet5 ||
			mode == eDexnet6 ||
			mode == eDexnet7 ||
			mode == eDexnet8 ||
			mode == eDexnet9 ||
			mode == eDexnet10 ||
			mode == eDexnet11 ||
			mode == eDexnet12)
		{
			g_params.frictionMode = eNvFlexFrictionModeFull;

			baseLinkTargetTranslation.x = 0.0f;
			baseLinkTargetTranslation.y = 0.5f;
			baseLinkTargetTranslation.z = 0.0f;
			baseLinkTargetRpy[2] = -110.0f;

			float scale = 1.0f;
			float density = 250.0f;
		
			const char* meshes[] 	= 
			{
				"../../data/mini_dexnet/bar_clamp.obj",
				"../../data/mini_dexnet/climbing_hold.obj",
				"../../data/mini_dexnet/endstop_holder.obj",
				"../../data/mini_dexnet/gearbox.obj",
				"../../data/mini_dexnet/mount1.obj",
				"../../data/mini_dexnet/mount2.obj",
				"../../data/mini_dexnet/nozzle.obj",
				"../../data/mini_dexnet/part1.obj",
				"../../data/mini_dexnet/part3.obj",
				"../../data/mini_dexnet/pawn.obj",
				"../../data/mini_dexnet/pipe_connector.obj",
				"../../data/mini_dexnet/turbine_housing.obj",
				"../../data/mini_dexnet/vase.obj"
			};

			const int numMeshes = sizeof(meshes)/sizeof(const char*);
			const int meshIndex = mode - eDexnet1;

			Mesh* mesh = ImportMesh(meshes[meshIndex]);
			mesh->Transform(ScaleMatrix(scale));

			NvFlexTriangleMeshId shapeId = CreateTriangleMesh(mesh, 0.0f);

			NvFlexRigidShape shape;
			NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
			shape.filter = 0x0;
			shape.material.friction = 0.8f;
			shape.material.torsionFriction = 0.1;
			shape.material.rollingFriction = 0.0f;
			shape.thickness = 0.001f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.3f, 0.f), Quat(), &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
		}
		else if (mode == eRope)
		{
			// Rope object (capsules)
			int segments = 32;

			const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
			const float linkLength = 0.0125f;
			const float linkWidth = 0.01f;
			const float density = 1000.0f;
			const float bendingCompliance = 1.e-2f;
			const float torsionCompliance = 1.e-6f;

			Vec3 startPos = Vec3(-0.3f, 0.5f, 0.45f);
			NvFlexRigidPose prevJoint;
			for (int i=0; i < segments; ++i)
			{
				int bodyIndex = g_buffers->rigidBodies.size();

				NvFlexRigidShape shape;
				NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0,0));
				shape.filter = 0;
				shape.material.rollingFriction = 0.001f;
				shape.material.friction = 0.25f;
				shape.user = UnionCast<void*>(linkMaterial);
				

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, startPos + Vec3(i*linkLength*2.0f + linkLength, 0.0f, 0.0f), Quat(), &shape, &density, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);

				if (i > 0)
				{
					NvFlexRigidJoint joint;				
					NvFlexMakeFixedJoint(&joint, bodyIndex-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), Quat()));

					joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
					joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
					joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

					g_buffers->rigidJoints.push_back(joint);
				}

				prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), Quat());

			}
		}
		else if (mode == eCloth)
        {
            // Cloth object
            const float radius = 0.00625f;

            float stretchStiffness = 0.6f;
            float bendStiffness = 0.01f;
            float shearStiffness = 0.01f;

            int dimx = 80;
            int dimy = 40;

            float mass = 0.5f/(dimx*dimy);	// avg bath towel is 500-700g

            CreateSpringGrid(Vec3(-0.3f, 0.5f, 0.15f), dimx, dimy, 1, radius, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f/mass);

            g_params.radius = radius*1.8f;
			g_params.collisionDistance = 0.005f;

			NvFlexRigidShape box;
			NvFlexMakeRigidBoxShape(&box, -1, 0.5f, 0.1f, 0.5f, NvFlexMakeRigidPose(Vec3(0.0f, 0.1f, 0.0f),0));
			box.filter = 0;
			box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f)));

			g_buffers->rigidShapes.push_back(box);

			/*
			// sphere for draping
			NvFlexRigidShape shape;
			NvFlexMakeRigidSphereShape(&shape, -1, 0.06f, NvFlexMakeRigidPose(Vec3(0.0f, 0.3f, 0.25f),Quat()));

			g_buffers->rigidShapes.push_back(shape);
			*/

			g_drawCloth = true;			
        }
        else if (mode == eRigidCube)
		{
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
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.2f * (float)i, 0.35f, 0.f), Quat(), &shape, &density, 1);

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
				shape.material.torsionFriction = 0.1;
				shape.material.rollingFriction = 0.0f;
				shape.thickness = 0.001f;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.2f * (float)i, 0.25f, 0.f), Quat(), &shape, &density, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);
			}
		}
		else
		{
			g_params.collisionDistance = 0.002f;

			g_params.numIterations = 4;
			g_params.numInnerIterations = 50;
			g_params.damping = 0.0f;
			g_params.radius = 0.01f;

			const float left = -0.25f;

			if (mode == eSoftTomato)
			{
				const float radius = 0.02f;
				const float density = 1000.0f;		// mostly water
				const float stiffness  = 1.e+5f;
				const float poissons = 0.45f;
				const float damping = 0.0f;

				deformable = CreateDeformableMesh("../../data/tomato/tomato.obj", "../../data/tomato/tomato.tet", Vec3(left, radius + 0.3f, 0.0f), Quat(), radius, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));
				
				g_tetraMaterials.resize(0);
				g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));			
			
			}

			if (mode == eSoftFish)
			{
				/*
				const float radius = 0.04f;
				const float density = 800.0f;
				const float stiffness  = 1.e+5f;
				const float poissons = 0.4f;
				const float damping = 0.001f;

				deformable = CreateDeformableMesh("../../data/trout/trout.obj", "../../data/trout/trout.tet", Vec3(left, radius + 0.3f, 0.0f), Quat(), radius, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));

				g_tetraMaterials.resize(0);
				g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));
				*/
			}

			if (mode == eSoftSphere)
			{
				const float radius = 0.04f;
				const float density = 500.0f;
				const float stiffness  = 1.e+4f;
				const float poissons = 0.45f;
				const float damping = 0.01f;

				deformable = CreateDeformableMesh("../../data/icosphere.obj", "../../data/icosphere.tet", Vec3(0.0f, radius + 0.3f, 0.0f), Quat(), radius, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));

				g_tetraMaterials.resize(0);
				g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));

				for (int i=0; i < g_buffers->positions.size(); ++i)
					g_buffers->positions[i].w = 7000.0f;

				baseLinkTargetTranslation.x = 0.0f;
				baseLinkTargetTranslation.y = 0.5f;
				baseLinkTargetTranslation.z = 0.0f;

				drawMesh = false;
			}

			if (mode == eSoftBread)
			{
				const float radius = 0.03f;
				const float density = 250.0f;
				const float stiffness = 1.e+5f;
				const float poissons = 0.35f;
				const float damping = 0.0f;

				deformable = CreateDeformableMesh("../../data/bread/Bread_1_Low_Poly.obj", "../../data/bread/bread.tet", Vec3(left, radius + 0.3f, 0.0f), Quat(), radius, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));

				g_tetraMaterials.resize(0);
				g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));

			}

			if (mode == eSoftSandwich)
			{
				const float radius = 1.0f;
				const float density = 250.0f;
				const float stiffness = 1.e+5f;
				const float poissons = 0.4f;
				const float damping = 0.0f;

				deformable = CreateDeformableMesh("../../data/sandwich/sandwich.obj", "../../data/sandwich/sandwich.tet", Vec3(left, 0.3f, 0.0f), Quat(), radius, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));

				g_tetraMaterials.resize(0);
				g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));

			}			

			if (mode == eSoftCube)
			{

				const int beamx = 5;
				const int beamy = 5;
				const int beamz = 20;
				const float radius = 0.01f;
				const float density = 250.0f;
				const float stiffness  = 1.e+4f;
				const float poissons = 0.4f;
				const float damping = 0.0f;
				
				CreateTetGrid(Vec3(0.0f, 0.3f, -beamz*radius/2), beamx, beamy, beamz, radius, radius, radius, density, ConstantMaterial<0>, false);	

				g_tetraMaterials.resize(0);
				g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));

			}

			float mass =  0.0f;
			for (int i=0; i < g_buffers->positions.size(); ++i)
				mass += 1.0f/g_buffers->positions[i].w;

			printf("mass: %f\n", mass);
		}

		// table
		NvFlexRigidShape box;
		NvFlexMakeRigidBoxShape(&box, -1, 0.5f, 0.1f, 0.5f, NvFlexMakeRigidPose(Vec3(0.0f, 0.1f, 0.0f),0));
		box.filter = 0;
		box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.35f, 0.45f, 0.65f)));

		g_buffers->rigidShapes.push_back(box);
    }

	virtual void CenterCamera()
	{
		g_camPos = Vec3(-0.010622f, 0.555237f, 0.584409f);
		g_camAngle = Vec3(0.017453f, -0.263545f, 0.000000f);

		if (mode == eSoftSphere)
		{
			g_camPos = Vec3(0.0f, 0.497352f, -0.498977f);
			g_camAngle = Vec3(3.146828f, -0.315905f, 0.000000f);
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
		imguiSlider("Y", &y, 0.0f, 1.0f, handTranslationDeltas);
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

		if (imguiCheck("Draw Mesh", drawMesh))
			drawMesh = !drawMesh;
	}

	virtual void Update()
	{
		const float smoothing = 0.05f;

		// Hand
		NvFlexRigidJoint handJoint = g_buffers->rigidJoints[handJointId];
		for (int i = 0; i < 3; i++)
		{
			handJoint.pose0.p[i] = Lerp(handJoint.pose0.p[i], baseLinkTargetTranslation[i], smoothing);
			baseLinkCurrentRpy[i] = Lerp(baseLinkCurrentRpy[i], baseLinkTargetRpy[i], smoothing);
		}

		Quat currentQuat = rpy2quat(DegToRad(baseLinkCurrentRpy[0]), DegToRad(baseLinkCurrentRpy[1]), DegToRad(baseLinkCurrentRpy[2]));
		handJoint.pose0.q[0] = currentQuat.x;
		handJoint.pose0.q[1] = currentQuat.y;
		handJoint.pose0.q[2] = currentQuat.z;
		handJoint.pose0.q[3] = currentQuat.w;
		g_buffers->rigidJoints[handJointId] = handJoint;

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

		if (deformable)
			UpdateDeformableMesh(deformable);

	}

	virtual void Sync()
	{
	    if (g_buffers->tetraIndices.size())
	    {   	
	        NvFlexSetFEMGeometry(g_solver, g_buffers->tetraIndices.buffer, g_buffers->tetraRestPoses.buffer, g_buffers->tetraMaterials.buffer, g_buffers->tetraMaterials.size());
	    }
	}

	virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }

	virtual void Draw(int pass)
	{
		if (mode < eDexnet12+1)
		{
			if (pass == 0)
			{
				SetFillMode(true);

				DrawRigidShapes(true,1);

				SetFillMode(false);
			}
		}

		if (drawMesh && deformable)
		{
			// visual mesh
			DrawDeformableMesh(deformable);
		}
		else
		{
			// tetrahedral mesh
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

				rangeMin = 0.0f;
				rangeMax = 0.5f;

				for (int i = 0; i < g_buffers->positions.size(); ++i)
				{
					mesh.m_normals[i] = Vec3(g_buffers->normals[i]);
					mesh.m_positions[i] = Point3(g_buffers->positions[i]) + Vec3(g_buffers->normals[i])*g_params.collisionDistance*1.5f;

					mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x / averageStress[i].y);					
				}
			}

			DrawMesh(&mesh, g_renderMaterials[0]);

			if (pass == 0 && g_buffers->triangles.size())
			{

				SetFillMode(true);

				DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], g_params.collisionDistance*1.5f);

				SetFillMode(false);

			}
		}
	}
};


