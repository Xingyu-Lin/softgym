#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"

class RigidYumi : public Scene
{
public:

	enum Mode
	{
		eCloth,
		eRigid,
		eSoft,
		eRopeCapsules,
		eRopeParticles,
	};

	Mode mode;

    URDFImporter* urdf;

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

    RigidYumi(Mode mode) : mode(mode)
    {
		roll = 0.0f;
		pitch = 0.0f;
		yaw = -90.0f;
        rigidTrans.clear();
	
        //urdf = new URDFImporter("../../data/fetch_ros-indigo-devel", "fetch_description/robots/fetch.urdf");
		urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi.urdf");
        
		Transform gt(Vec3(0.0f, 0.025f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));

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
                joint.damping[eNvFlexRigidJointAxisX] = 0.0f;

            }
        }
		/*
        // fix base in place, todo: add a kinematic body flag?
        g_buffers->rigidBodies[0].invMass = 0.0f;
        (Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();
		*/

		NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["gripper_l_joint"]];
		NvFlexRigidJoint handRight = g_buffers->rigidJoints[urdf->jointNameMap["gripper_r_joint"]];

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

		fingerLeft = urdf->jointNameMap["gripper_l_joint"];
		fingerRight = urdf->jointNameMap["gripper_l_joint_m"];


        NvFlexRigidShape table;
        NvFlexMakeRigidBoxShape(&table, -1, 0.55f, 0.2f, 0.25f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.6f), Quat()));
        table.filter = 0;
        table.material.friction = 0.7f;	
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));

        g_buffers->rigidShapes.push_back(table);
		
        if (mode == eRigid)
        {
            // Box object
            NvFlexRigidShape capsule;
            NvFlexMakeRigidCapsuleShape(&capsule, 0, 0.125f, 0.25f, NvFlexMakeRigidPose(0,0));

            float scale = 1.0f;

            Mesh* boxMesh = ImportMesh("../../data/bar_clamp/bar_clamp.obj");
            boxMesh->Transform(ScaleMatrix(scale));

            for (int i = 0; i < 1; ++i)
            {
                NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.001f);

                NvFlexRigidShape box;
                //NvFlexMakeRigidBoxShape(&box, g_buffers->rigidBodies.size(), 0.125f*scale, 0.125f*scale, 0.125f*scale, NvFlexMakeRigidPose(0,0));
                NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
                box.filter = 0x0;
                box.material.friction = 1.0f;
				box.material.torsionFriction = 0.1f;
                box.thickness = 0.005f;
				box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.98f, 0.25f, 0.125f)));

				const float density = 100.0f;

                NvFlexRigidBody body;
                NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-0.1f + 0.2f * (float)i, 0.4f + scale*0.5f + 0.01f + 0.05f * (float)i, 0.6f), Quat(), &box, &density, 1);

                g_buffers->rigidBodies.push_back(body);
                g_buffers->rigidShapes.push_back(box);
            }
        }

        if (mode == eSoft)
        {
            // FEM object
            int beamx = 4;
            int beamy = 4;
            int beamz = 4;

            float radius = 0.02f;
            float mass = 0.5f/(beamx*beamy*beamz);	// 500gm total sweight

            //CreateTetraGrid(Vec3(-beamx/2*radius, 0.475f, 0.6f), beamx, beamy, beamz, radius, radius, radius, 1.0f/mass, ConstantMaterial<0>, false);
			CreateTetMesh("../../data/dragonNormalized.tet", Vec3(-0.25f, 0.45f, 0.55f), 0.4f, 1.0f/mass, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));

            g_params.radius = radius;
			g_params.collisionDistance = radius;

            g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

            g_tetraMaterials.resize(0);
            g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+9f, 0.4f, 0.01f));

			g_drawCloth = false;
			g_drawMesh = false;

			g_params.collisionDistance = 0.0025f;
        }

        if (mode == eCloth)
        {
            // Cloth object
            const float radius = 0.00625f;

            float stretchStiffness = 1.0f;
            float bendStiffness = 0.9f;
            float shearStiffness = 0.8f;

            int dimx = 80;
            int dimy = 40;

            float mass = 0.5f/(dimx*dimy);	// avg bath towel is 500-700g

            CreateSpringGrid(Vec3(-0.3f, 0.5f, 0.45f), dimx, dimy, 1, radius, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f/mass);

            g_params.radius = radius*1.8f;
			g_params.collisionDistance = 0.005f;

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
					
					const float bendingCompliance = 1.e-2f;
					const float torsionCompliance = 1.e-6f;

					joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
					joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
					joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

					g_buffers->rigidJoints.push_back(joint);
				}

				prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), Quat());

			}
		}


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


