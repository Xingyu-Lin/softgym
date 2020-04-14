#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"
#include "../vr/vr.h"

#if FLEX_VR

class VRFetch : public Scene
{
public:

	enum Mode
	{
		eCloth,
		eRigid,
		eSoft,
		eRopeCapsules,
		eRopeParticles,
		eRopePeg,
		eSandBucket,
		eFlexibleBeam,
	};

	Mode mode;

    URDFImporter* urdf;

    vector<Transform> rigidTrans;
    map<string, int> jointMap;
    map<string, int> activeJointMap;

	Vec3 robotPos;
	Vec3 controllerPos;
	int debugSphere;

	int effectorJoint;

	int fingerLeft;
	int fingerRight;

	float fingerWidth = 0.03f;
	float fingerWidthMin = 0.005f;; //0.002f;
	float fingerWidthMax = 0.05f;
	float roll, pitch, yaw;

	bool hasFluids = false;

	float numpadJointTraSpeed = 0.1f / 60.f; // 10cm/s under 60 fps
	float numpadJointRotSpeed = 10 / 60.f; // 10 deg/s under 60 fps
	float numpadJointRotDir = 1.f; // direction of rotation
	float numpadFingerSpeed = 0.02f / 60.f; // 2cm/s under 60 fps

	VRFetch(Mode mode) : mode(mode)
    {
		roll = 0.0f;
		pitch = 0.0f;
		yaw = -90.0f;
        rigidTrans.clear();

        urdf = new URDFImporter("../../data/fetch_ros-indigo-devel", "fetch_description/robots/fetch.urdf", false);

		robotPos = Vec3(0.0f, 0.05f, -0.25f);
        
		Transform gt(robotPos, QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

        urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 1000.0f, 0.0f, 1e1f, 0.01f, 20.7f, 7.0f, false);
        
		for (int i = 0; i < (int)urdf->joints.size(); i++)
        {
            URDFJoint* j = urdf->joints[i];
            NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[j->name]];
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
                joint.damping[eNvFlexRigidJointAxisX] = 10.0f;

            }
        }
        // fix base in place, todo: add a kinematic body flag?
        g_buffers->rigidBodies[0].invMass = 0.0f;
        (Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();

        fingerLeft = urdf->jointNameMap["l_gripper_finger_joint"];
        fingerRight = urdf->jointNameMap["r_gripper_finger_joint"];

        NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["l_gripper_finger_joint"]];

        // set up end effector targets
        NvFlexRigidJoint effectorJoint0;
    //    NvFlexMakeFixedJoint(&effectorJoint0, -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(0.2f, 0.7f, 0.5f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -kPi*0.5f)), NvFlexMakeRigidPose(0, 0));
		NvFlexMakeFixedJoint(&effectorJoint0, -1, urdf->rigidNameMap["gripper_link"], NvFlexMakeRigidPose(Vec3(0.2f, 0.7f, 0.5f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -kPi*0.5f)), NvFlexMakeRigidPose(0, 0));

        effectorJoint = g_buffers->rigidJoints.size();
        g_buffers->rigidJoints.push_back(effectorJoint0);

		// Make fingers heavier for ID control
		float mul = 25.0f;
		float imul = 1.0f / mul;
		g_buffers->rigidBodies[fingerLeft].mass *= mul;
		g_buffers->rigidBodies[fingerLeft].invMass *= imul;
		for (int k = 0; k < 9; k++)
		{
			g_buffers->rigidBodies[fingerLeft].inertia[k] *= mul;
			g_buffers->rigidBodies[fingerLeft].invInertia[k] *= imul;
		}

		g_buffers->rigidBodies[fingerRight].mass *= mul;
		g_buffers->rigidBodies[fingerRight].invMass *= imul;
		for (int k = 0; k < 9; k++)
		{
			g_buffers->rigidBodies[fingerRight].inertia[k] *= mul;
			g_buffers->rigidBodies[fingerRight].invInertia[k] *= imul;
		}

		g_buffers->rigidJoints[fingerRight].targets[eNvFlexRigidJointAxisX] = 0.02f;
		g_buffers->rigidJoints[fingerLeft].targets[eNvFlexRigidJointAxisX] = 0.02f;

		for (int i = 0; i < 6; ++i)
		{
			effectorJoint0.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
			effectorJoint0.damping[i] = 2.e+3f;
			//effectorJoint0.maxIterations = 30;
		}

		// table
        NvFlexRigidShape table;
        NvFlexMakeRigidBoxShape(&table, -1, 0.55f, 0.4f, 0.25f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.6f), Quat()));
        table.filter = 1;
        table.material.friction = 0.7f;
		table.thickness = 0.025f;
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));

        g_buffers->rigidShapes.push_back(table);
		
		forceLeft.resize(0);
		forceRight.resize(0);


		g_params.solverType = eNvFlexSolverPCR;;
		g_numSubsteps = 1;
		g_params.numIterations = 6;
		g_params.numInnerIterations = 20;

	//	g_numSubsteps = 4;
	//	g_params.numIterations = 40;
	//			g_params.numPostCollisionIterations = 0;

		g_params.dynamicFriction = 1.25f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.001f;

		g_sceneLower = Vec3(-0.5f);
		g_sceneUpper = Vec3(0.1f, 0.55f, 0.0f);
		g_drawPoints = false;

		const float radius = 0.05f;
		NvFlexRigidShape starget;
		controllerPos = Vec3(Randf(-0.5f, 0.5f), Randf(0.4f, 1.6f), Randf(0.25f, 1.1f));
		NvFlexRigidPose newPose = NvFlexMakeRigidPose(controllerPos + robotPos, Quat());

		NvFlexMakeRigidSphereShape(&starget, -1, radius, newPose);

		starget.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.1f, 0.24f, 0.64f)));

		debugSphere = g_buffers->rigidShapes.size();
		g_buffers->rigidShapes.push_back(starget);

        if (mode == eRigid)
        {
            // Box object
            float scale = 0.04f;
			float density = 500.0f;

            Mesh* boxMesh = ImportMesh("../../data/box.ply");
            boxMesh->Transform(ScaleMatrix(scale));

            for (int i = 0; i < 2; ++i)
            {
                NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.00125f);

                NvFlexRigidShape box;
                NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
                box.filter = 0;
                box.material.friction = 1.0f;
                box.material.torsionFriction = 0.1f;
				box.material.rollingFriction = 0.0f;
				box.thickness = 0.001f;
				box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.97f - 0.85f * float(i), 0.04f + 0.8f * float(i), 0.04f)));
				
                NvFlexRigidBody body;
                NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-0.15f + 0.2f * (float)i, 0.4f + scale*0.5f + 0.01f + 0.05f * (float)i, 0.52f), Quat() /*Normalize(RandVec3()), Randf(0.f, kPi))*/, &box, &density, 1);

                g_buffers->rigidBodies.push_back(body);
                g_buffers->rigidShapes.push_back(box);
            }

			NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.00125f);
		
			NvFlexRigidShape box;
			NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
			box.filter = 0;
			box.material.friction = 1.0f;
			box.material.torsionFriction = 0.1f;
			box.material.rollingFriction = 0.0f;
			box.thickness = 0.0012f;
			box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.04f, 0.07f, 0.98f)));
		
			NvFlexRigidBody body;
			Quat rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * Randf(0.1f, 0.2f));
		
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(Randf(0.1f, 0.25f), 0.4f + scale*0.5f + 0.04f, Randf(0.5f, 0.6f)), rot, &box, &density, 1);
		//	NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-0.1f + Randf(0.3f, 0.4f), 0.4f + scale*0.5f + 0.04f, 0.6f), Quat(), &box, &density, 1);
		
			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(box);
			
			g_params.numPostCollisionIterations = 15;
        }

        if (mode == eSoft)
        {
            // FEM object
            int beamx = 6;
            int beamy = 6;
            int beamz = 24;

            float radius = 0.01f;
            float mass = 0.5f/(beamx*beamy*beamz);	// 500gm total sweight

            //Todo: fix params
        	CreateTetMesh("../../data/dragonNormalized.tet", Vec3(-0.25f, 0.45f, 0.55f), 0.4f, 1.0f / mass, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));
        	//CreateTetraGrid(Vec3(-beamx/2*radius, 0.475f, 0.5f), beamx, beamy, beamz, radius, radius, radius, 1.0f/mass, ConstantMaterial<0>, false);

            g_params.radius = radius;
			g_params.collisionDistance = 0.0f;

            g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

            g_tetraMaterials.resize(0);
            g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+9f, 0.4f, 0.0005f));

			g_drawCloth = false;
        }

        if (mode == eCloth)
        {
			g_params.numIterations = 45;

            // Cloth object
            const float radius = 0.00625f;

            float stretchStiffness = 1.0f;
            float bendStiffness = 0.9f;
            float shearStiffness = 0.8f;

            int dimx = 72;
            int dimy = 40;

            float mass = 0.5f/(dimx*dimy);	// avg bath towel is 500-700g

            CreateSpringGrid(Vec3(-0.3f, 0.5f, 0.4f), dimx, dimy, 1, radius, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f/mass);

            g_params.radius = radius*1.8f;
			g_params.collisionDistance = 0.004f;

			g_drawCloth = true;
        }

		if (mode == eRopeParticles)
		{
			// Rope object (particles)

			const float radius = 0.025f;
			g_params.radius = radius;	// some overlap between particles for more robust self collision
			g_params.dynamicFriction = 1.0f;
			g_params.collisionDistance = radius*0.5f;

			// do not allow fingers to close more than this to prevent pushing through grippers
			fingerWidthMin = g_params.collisionDistance;

			const int segments = 64;

			const float stretchStiffness = 0.9f;
			const float bendStiffness = 0.8f;

			const float mass = 0.5f;///segments;	// assume 1kg rope

			Rope r;
			CreateRope(r, Vec3(-0.3f, 0.5f, 0.45f), Vec3(1.0f, 0.0f, 0.0f), stretchStiffness, bendStiffness, segments, segments*radius*0.5f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), 0.0f, 1.0f/mass);

			g_ropes.push_back(r);
		}

		if (mode == eRopeCapsules)
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
		
		if (mode == eRopePeg)
		{
			// Raising start end-effector position to make room for rope
			g_buffers->rigidJoints[effectorJoint].pose0.p[1] = 1.f;

			// Rope (capsules)
			int segments = 12;
			const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
			const float linkLength = 0.01f;
			const float linkWidth = 0.005f;
			const float density = 1.f;
			const bool connectRopeToRobot = true;

			Vec3 startPos = Vec3(0.2f, 1.f, 0.5f);
			NvFlexRigidPose prevJoint;
			int lastRopeBodyIndex;
			const float bendingCompliance = 1.e+2f;
			const float torsionCompliance = 1.f;
			for (int i = 0; i < segments; ++i) {
				int bodyIndex = g_buffers->rigidBodies.size();

				NvFlexRigidShape shape;
				NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0, Quat(.0f, .0f, .707f, .707f)));
				shape.filter = 0;
				shape.material.rollingFriction = 0.001f;
				shape.material.friction = 0.25f;
				shape.user = UnionCast<void*>(linkMaterial);

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, startPos + Vec3(0.0f, -(i*linkLength*2.f + linkLength), 0.0f), Quat(), &shape, &density, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);

				if (i == 0 && !connectRopeToRobot) 
				{
					prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -linkLength, .0f), Quat());
					continue;
				}

				NvFlexRigidJoint joint;
				if (i == 0) 
				{					
					NvFlexMakeFixedJoint(&joint, handLeft.body0, bodyIndex, 
						NvFlexMakeRigidPose(Vec3(0.1f, .0f, .0f), QuatFromAxisAngle(Vec3(.0f, .0f, 1.f), kPi*0.5)),
						NvFlexMakeRigidPose(0,0));
				} else 
				{
					NvFlexMakeFixedJoint(&joint, bodyIndex - 1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(.0f, linkLength, .0f), Quat()));
				}
				joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

				g_buffers->rigidJoints.push_back(joint);
				lastRopeBodyIndex = bodyIndex;
				prevJoint = NvFlexMakeRigidPose(Vec3(.0f, -linkLength, .0f), Quat());
			}

			// Peg
			float scale = 0.03f;
			Mesh* pegMesh = ImportMesh("../../data/cylinder.obj");			
			pegMesh->Transform(ScaleMatrix(scale));

			NvFlexTriangleMeshId pegId = CreateTriangleMesh(pegMesh, 0.005f);
			NvFlexRigidShape peg;
			NvFlexMakeRigidTriangleMeshShape(&peg, g_buffers->rigidBodies.size(), pegId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
			peg.filter = 0x0;
			peg.material.friction = 1.0f;
			peg.thickness = 0.005f;
			peg.user = UnionCast<void*>(AddRenderMaterial(Vec3(.9f, .9f, .3f)));
			g_buffers->rigidShapes.push_back(peg);

			float pegDensity = 0.1f;
			NvFlexRigidBody pegBody;
			NvFlexMakeRigidBody(g_flexLib, &pegBody, startPos + Vec3(0.0f, -float(segments+2)*linkLength*2.f, .0f), Quat(), &peg, &pegDensity, 1);
			g_buffers->rigidBodies.push_back(pegBody);

			// Connecting peg to rope
			NvFlexRigidJoint joint;
			int bodyIndex = g_buffers->rigidBodies.size();
			NvFlexMakeFixedJoint(&joint, lastRopeBodyIndex, bodyIndex - 1, NvFlexMakeRigidPose(Vec3(.0f, -4.f*linkLength, .0f), Quat()), NvFlexMakeRigidPose(0, 0));

			joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
			joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
			joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

			g_buffers->rigidJoints.push_back(joint);

			// Peg Holder
			Mesh* pegHolderMesh = ImportMesh("../../data/peg_holder.obj");
			pegHolderMesh->Transform(ScaleMatrix(scale));

			NvFlexTriangleMeshId pegHolderId = CreateTriangleMesh(pegHolderMesh, 0.005f);

			NvFlexRigidShape pegHolder;
			NvFlexMakeRigidTriangleMeshShape(&pegHolder, g_buffers->rigidBodies.size(), pegHolderId, NvFlexMakeRigidPose(0,0), 1.f, 1.f, 1.f);
			pegHolder.filter = 0x0;
			pegHolder.material.friction = 1.0f;
			pegHolder.thickness = 0.005f;
			g_buffers->rigidShapes.push_back(pegHolder);

			float pegHolderDensity = 100.0f;
			NvFlexRigidBody pegHolderBody;
			NvFlexMakeRigidBody(g_flexLib, &pegHolderBody, Vec3(-.2f, .42f, .6f), Quat(), &pegHolder, &pegHolderDensity, 1);
			g_buffers->rigidBodies.push_back(pegHolderBody);
			g_buffers->rigidBodies.back().invMass = 0.0f;
			(Matrix33&)g_buffers->rigidBodies.back().invInertia = Matrix33();
		}

		if (mode == eSandBucket)
		{			
			// Bucket
			Vec3 center = Vec3(-0.1f, 0.43f, 0.5f);
			float widthH = 0.12f;
			float lengthH = 0.2f;
			float heightH = 0.02f;
			float edge = 0.002f;

			Vec3 grayColor = Vec3(0.6f, 0.6f, 0.65f);

			NvFlexRigidShape sideNear;
			NvFlexMakeRigidBoxShape(&sideNear, -1, lengthH, heightH, edge, NvFlexMakeRigidPose(center + Vec3(0.f, 0.f, -widthH), Quat()));
			sideNear.filter = 0;
			sideNear.material.friction = 0.7f;
			sideNear.user = UnionCast<void*>(AddRenderMaterial(grayColor));
			g_buffers->rigidShapes.push_back(sideNear);

			NvFlexRigidShape sideFar;
			NvFlexMakeRigidBoxShape(&sideFar, -1, lengthH, heightH, edge, NvFlexMakeRigidPose(center + Vec3(0.f, 0.f, widthH), Quat()));
			sideFar.filter = 0;
			sideFar.material.friction = 0.7f;
			sideFar.user = UnionCast<void*>(AddRenderMaterial(grayColor));
			g_buffers->rigidShapes.push_back(sideFar);

			NvFlexRigidShape sideLeft;
			NvFlexMakeRigidBoxShape(&sideLeft, -1, edge, heightH, widthH - edge, NvFlexMakeRigidPose(center + Vec3(lengthH, 0.f, 0.f), Quat()));
			sideLeft.filter = 0;
			sideLeft.material.friction = 0.7f;
			sideLeft.user = UnionCast<void*>(AddRenderMaterial(grayColor));
			g_buffers->rigidShapes.push_back(sideLeft);

			NvFlexRigidShape sideRight;
			NvFlexMakeRigidBoxShape(&sideRight, -1, edge, heightH, widthH - edge, NvFlexMakeRigidPose(center + Vec3(-lengthH, 0.f, 0.f), Quat()));
			sideRight.filter = 0;
			sideRight.material.friction = 0.7f;
			sideRight.user = UnionCast<void*>(AddRenderMaterial(grayColor));
			g_buffers->rigidShapes.push_back(sideRight);

			// Sand
			float radius = 0.005f;

			float fluidWidth = widthH * 2;
			float fluidLength = lengthH * 2;
			float fluidHeight = heightH * 0.75f;

			int particleWidth = 40;
			int particleLength = 60;
			int particleHeight = 10;

			float particleMass = 0.001f;

			CreateParticleGrid(center - Vec3(lengthH-edge*2, 0, widthH-edge*2), particleLength, particleHeight, particleWidth, radius, Vec3(0.0f), 1.0f/particleMass, false, .0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide));
			g_colors[0] = Colour(0.805f, 0.702f, 0.401f);

			g_drawPoints = true;
			g_params.radius = radius;
			g_params.staticFriction = 1.0f;
			g_params.dynamicFriction = 0.5f;
			g_params.particleCollisionMargin = 0.001f;
			g_params.shapeCollisionMargin = 0.001f;
			g_params.sleepThreshold = g_params.radius*0.05f;

			hasFluids = true;

			// Top-down sensor
			DepthRenderProfile p = {
				0.2f, // minRange
				0.7f // maxRange
			};
			AddSensor(256, 256, 0, Transform(center + Vec3(0, 0.3f, 0), rpy2quat(kPi/2, kPi, 0)), DegToRad(60), true, p);
			g_drawSensors = true;			
		}

		if (mode == eFlexibleBeam)
		{
			const float linkLength = 0.035f;
			const float linkWidth = 0.02f;
			const float linkHeight = 0.0015f;
			const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));
			const float density = 1000.0f;
			const int numLinks = 6;
			const float bendingCompliance = 5e-2f;
			const float torsionCompliance = 1.e-6f;
			
			Vec3 beamStartPos = Vec3(0.1f, 0.45f, 0.3f);
			Quat localFrame = Quat();
			NvFlexRigidPose prevJoint;
			for (int i = 0; i < numLinks; ++i)
			{
				int bodyIndex = g_buffers->rigidBodies.size();

				NvFlexRigidShape shape;
				NvFlexMakeRigidBoxShape(&shape, bodyIndex, linkWidth, linkHeight, linkLength, NvFlexMakeRigidPose(0, 0));
				shape.user = UnionCast<void*>(linkMaterial);
				shape.filter = 0;
				shape.material.friction = 1.f;
				shape.material.rollingFriction = 1.f;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, beamStartPos + Vec3(0.f, 0.f, i*linkLength*2.f + linkLength), Quat(), &shape, &density, 1);

				g_buffers->rigidShapes.push_back(shape);
				g_buffers->rigidBodies.push_back(body);

				NvFlexRigidJoint joint;
				if (i == 0)
				{
					NvFlexMakeFixedJoint(&joint, handLeft.body0, bodyIndex,
						NvFlexMakeRigidPose(Vec3(0.12f, .0f, .0f), QuatFromAxisAngle(Vec3(.0f, 1.f, 0.f), kPi*0.5)),
						NvFlexMakeRigidPose(0, 0));
				}
				else
				{
					NvFlexMakeFixedJoint(&joint, bodyIndex - 1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(0.f, 0.f, -linkLength), localFrame));
				}
				joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;
				g_buffers->rigidJoints.push_back(joint);

				prevJoint = NvFlexMakeRigidPose(Vec3(0.f, 0.f, linkLength), localFrame);
			}
				
			// Moving gripper to init position
			roll = 90.f;
			pitch = -90.f;
			yaw = 90.f;
			NvFlexRigidJoint effectorJoint0 = g_buffers->rigidJoints[effectorJoint];
			Vec3 endEffectorStartPos = Vec3(0.2f, 0.7f, 0.3f);
			effectorJoint0.pose0.p[0] = endEffectorStartPos.x;
			effectorJoint0.pose0.p[1] = endEffectorStartPos.y;
			effectorJoint0.pose0.p[2] = endEffectorStartPos.z;
			g_buffers->rigidJoints[effectorJoint] = effectorJoint0;
			fingerWidth = .017f;		

			// Beam container
			Vec3 grayColor = Vec3(0.6f, 0.6f, 0.65f);
			Vec3 containerMiddle = endEffectorStartPos + Vec3(0.f, .25f, .85f);
			Vec3 topBottomDeltaPos = Vec3(0.f, 0.025f, 0.f);

			NvFlexRigidShape containerBottom;
			NvFlexMakeRigidBoxShape(&containerBottom, -1, linkWidth * 3.f, linkWidth * 0.2f, linkLength * numLinks,
				NvFlexMakeRigidPose(containerMiddle - topBottomDeltaPos, Quat()));
			containerBottom.filter = 0;
			containerBottom.material.friction = 0.7f;
			containerBottom.user = UnionCast<void*>(AddRenderMaterial(grayColor));
			g_buffers->rigidShapes.push_back(containerBottom);

			NvFlexRigidShape containerTop;
			NvFlexMakeRigidBoxShape(&containerTop, -1, linkWidth * 3.f, linkWidth * 0.2f, linkLength * numLinks,
				NvFlexMakeRigidPose(containerMiddle + topBottomDeltaPos, Quat()));
			containerTop.filter = 0;
			containerTop.material.friction = 0.7f;
			containerTop.user = UnionCast<void*>(AddRenderMaterial(grayColor));
			g_buffers->rigidShapes.push_back(containerTop);

			g_camSpeed = 0.03f;
			g_params.shapeCollisionMargin = 0.01f;
			
		}

		/*
		// add camera, todo: read correct links etc from URDF, right now these are thrown away
		const int headLink = urdf->rigidNameMap["head_tilt_link"];
		AddPrimesenseSensor(headLink, Transform(Vec3(-0.1f, 0.2f, 0.0f), rpy2quat(-1.57079632679f, 0.0f, -1.57079632679f)), 1.f, hasFluids);
		*/

        g_pause = true;              
    }

    virtual void DoGui()
    {
        NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoint];

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

        g_buffers->rigidJoints[effectorJoint] = effector0;

        float newWidth = fingerWidth;
        imguiSlider("Finger Width", &newWidth, fingerWidthMin, fingerWidthMax, 0.001f);

        fingerWidth = Lerp(fingerWidth, newWidth, smoothing);
		
        g_buffers->rigidJoints[fingerLeft].targets[eNvFlexRigidJointAxisX] = fingerWidth;
        g_buffers->rigidJoints[fingerRight].targets[eNvFlexRigidJointAxisX] = fingerWidth;
    }

    virtual void DoStats()
    {
        int numSamples = 200;

        int start = Max(int(forceLeft.size())-numSamples, 0);
        int end = Min(start + numSamples, int(forceLeft.size()));

        // convert from position changes to forces
        float units = -1.0f/Sqr(g_dt/g_numSubsteps);

        float height = 50.0f;
        float maxForce = 10.0f;

        float dx = 1.0f;
        float sy = height/maxForce;

        float lineHeight = 10.0f;

        float rectMargin = 10.0f;
        float rectWidth = dx*numSamples + rectMargin*4.0f;

        float x = g_screenWidth - rectWidth - 20.0f;
        float y = 300.0f;

        DrawRect(x, y - height - rectMargin, rectWidth, 2.0f*height + rectMargin*3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));

        x += rectMargin*3.0f;

        DrawImguiString(static_cast<int>(x + dx*numSamples), static_cast<int>(y + 55.0f), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

        DrawLine(x, y, x + static_cast<float>(numSamples)*dx, y, 1.0f, Vec3(1.0f));
        DrawLine(x, y -50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

        float margin = 5.0f;

        const int guiStringX = static_cast<int>(x - margin);
        DrawImguiString(guiStringX, static_cast<int>(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
        DrawImguiString(guiStringX, static_cast<int>(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
        DrawImguiString(guiStringX, static_cast<int>(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

        for (int i=start; i < end-1; ++i)
        {
        	float fl0 = Clamp(forceLeft[i]*units, -maxForce, maxForce)*sy;
        	float fr0 = Clamp(forceRight[i]*units, -maxForce, maxForce)*sy;

        	float fl1 = Clamp(forceLeft[i+1]*units, -maxForce, maxForce)*sy;
        	float fr1 = Clamp(forceRight[i+1]*units, -maxForce, maxForce)*sy;

        	DrawLine(x, y + fl0, x + dx, y + fl1, 1.0f, Vec3(1.0f, 0.0f, 0.0f));
        	DrawLine(x, y + fr0, x + dx, y + fr1, 1.0f, Vec3(0.0f, 1.0f, 0.0f));

        	x += dx;
        }
    }

    std::vector<float> forceLeft;
    std::vector<float> forceRight;

	size_t selectedController = VrSystem::InvalidControllerIndex;

    virtual void Update()
    {
        // record force on finger joints
        forceLeft.push_back(g_buffers->rigidJoints[fingerLeft].lambda[eNvFlexRigidJointAxisX]);
        forceRight.push_back(g_buffers->rigidJoints[fingerRight].lambda[eNvFlexRigidJointAxisX]);

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
					NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoint];

					NvFlexRigidPose newPose = NvFlexMakeRigidPose(state.pos, Quat());

					g_buffers->rigidShapes[debugSphere].pose = newPose;
					//	NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

					float currentWidth = fingerWidth;

					float newFingerWidth = fingerWidthMax + (fingerWidthMin - fingerWidthMax) * state.triggerValue;
					fingerWidth = Lerp(currentWidth, newFingerWidth, 0.05f);

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
		/*
    	// move end-effector via numpad
		NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoint];

		Vec3 contrPos = Vec3(0.f);
			
		bool res = g_vrSystem->GetControllerPosition(0, contrPos);

		NvFlexRigidPose newPose = NvFlexMakeRigidPose(contrPos, Quat());

		g_buffers->rigidShapes[debugSphere].pose = newPose;
	//	NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());

		/*
    	if (g_numpadPressedState[SDLK_KP_PLUS])
		{
			fingerWidth = min(fingerWidth + numpadFingerSpeed, fingerWidthMax);
		}
		if (g_numpadPressedState[SDLK_KP_MINUS])
		{
			fingerWidth = max(fingerWidth - numpadFingerSpeed, fingerWidthMin);
		}

		const float smoothing = 0.05f;

		Vec3 desiredPos = contrPos; // -robotPos;

		// low-pass filter controls otherwise it is too jerky
		float newx = Lerp(effector0.pose0.p[0], desiredPos[0], smoothing);
		float newy = Lerp(effector0.pose0.p[1], desiredPos[1], smoothing);
		float newz = Lerp(effector0.pose0.p[2], desiredPos[2], smoothing);

		effector0.pose0.p[0] = newx;
		effector0.pose0.p[1] = newy;
		effector0.pose0.p[2] = newz;

		g_buffers->rigidJoints[effectorJoint] = effector0;
		*/
    /*	// x
		if (g_numpadPressedState[SDLK_KP_4])
		{
			joint.pose0.p[0] -= numpadJointTraSpeed;
		}
		if (g_numpadPressedState[SDLK_KP_6])
		{
			joint.pose0.p[0] += numpadJointTraSpeed;
		}		
		// y
		if (g_numpadPressedState[SDLK_KP_9])
		{
			joint.pose0.p[1] += numpadJointTraSpeed;
		}
		if (g_numpadPressedState[SDLK_KP_7])
		{
			joint.pose0.p[1] -= numpadJointTraSpeed;
		}
		// z
		if (g_numpadPressedState[SDLK_KP_5])
		{
			joint.pose0.p[2] += numpadJointTraSpeed;
		}
		if (g_numpadPressedState[SDLK_KP_8])
		{
			joint.pose0.p[2] -= numpadJointTraSpeed;
		}
		// rpy
		Quat currentRot = Quat(joint.pose0.q);
		if (g_numpadPressedState[SDLK_KP_0])
		{
			numpadJointRotDir *= -1;
		}
		if (g_numpadPressedState[SDLK_KP_1])
		{
			float deltaRollAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaRoll = QuatFromAxisAngle(Vec3(1, 0, 0), deltaRollAngle);
			currentRot = deltaRoll * currentRot;
			roll += deltaRollAngle;
		}
		if (g_numpadPressedState[SDLK_KP_2])
		{
			float deltaYawAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaYaw = QuatFromAxisAngle(Vec3(0, 1, 0), deltaYawAngle);
			currentRot = deltaYaw * currentRot;
			yaw += deltaYawAngle;
		}
		if (g_numpadPressedState[SDLK_KP_3])
		{
			float deltaPitchAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaPitch = QuatFromAxisAngle(Vec3(0, 0, 1), deltaPitchAngle);
			currentRot = deltaPitch * currentRot;
			pitch += deltaPitchAngle;
		}
		if (g_numpadPressedState[SDLK_KP_PLUS])
		{
			fingerWidth = min(fingerWidth + numpadFingerSpeed, fingerWidthMax);
		}
		if (g_numpadPressedState[SDLK_KP_MINUS])
		{
			fingerWidth  = max(fingerWidth - numpadFingerSpeed, fingerWidthMin);
		}

		joint.pose0.q[0] = currentRot.x;
		joint.pose0.q[1] = currentRot.y;
		joint.pose0.q[2] = currentRot.z;
		joint.pose0.q[3] = currentRot.w; */

	//	g_buffers->rigidJoints[effectorJoint] = joint;
    }

    virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }

	Mesh mesh;

	virtual void Draw(int pass)
	{		
		// FEM stain visualization
		if (mode == eSoft)
		{
			if (pass == 1)
			{
				mesh.m_positions.resize(g_buffers->positions.size());
				mesh.m_normals.resize(g_buffers->normals.size());
				mesh.m_colours.resize(g_buffers->positions.size());
				mesh.m_indices.resize(g_buffers->triangles.size());

				for (int i=0; i < g_buffers->triangles.size(); ++i)
					mesh.m_indices[i] = g_buffers->triangles[i];

				float rangeMin = FLT_MAX;
				float rangeMax = -FLT_MAX;

				std::vector<Vec2> averageStress(mesh.m_positions.size());

				// calculate average Von-Mises stress on each vertex for visualization
				for (int i=0; i < g_buffers->tetraIndices.size(); i+=4)
				{
					float vonMises = fabsf(g_buffers->tetraStress[i/4]);

					//printf("%f\n", vonMises);

					averageStress[g_buffers->tetraIndices[i+0]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i+1]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i+2]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i+3]] += Vec2(vonMises, 1.0f);

					rangeMin = Min(rangeMin, vonMises);
					rangeMax = Max(rangeMax, vonMises);
				}

				//printf("%f %f\n", rangeMin,rangeMax);

				rangeMin = 0.0f;//Min(rangeMin, vonMises);
				rangeMax = 0.5f;//Max(rangeMax, vonMises);

				for (int i=0; i < g_buffers->positions.size(); ++i)
				{
					mesh.m_positions[i] = Point3(g_buffers->positions[i]);
					mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

					mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x/averageStress[i].y);
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
	}

};

#endif // #if FLEX_VR