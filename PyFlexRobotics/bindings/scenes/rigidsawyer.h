#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"

class RigidSawyer : public Scene
{
public:

	enum Mode
	{
		eCloth,
		eRigid,
		eSoft,
		eRopeCapsules,
	};

	Mode mode;

    URDFImporter* urdf;

    vector<Transform> rigidTrans;
    map<string, int> jointMap;
    map<string, int> activeJointMap;
	int effectorJoint;

	int fingerLeft;
	int fingerRight;
	float fingerWidth = 0.03f;
	float fingerWidthMin = 0.0f;
	float fingerWidthMax = 0.05f;
	float roll, pitch, yaw;
	bool hasFluids = false;
	float numpadJointTraSpeed = 0.1f / 60.f; // 10cm/s under 60 fps
	float numpadJointRotSpeed = 10 / 60.f; // 10 deg/s under 60 fps
	float numpadJointRotDir = 1.f; // direction of rotation
	float numpadFingerSpeed = 0.02f / 60.f; // 2cm/s under 60 fps

	RigidSawyer(Mode mode = eRigid) : mode(mode)
    {
		g_numSubsteps = 4;
		g_params.numIterations = 30;
		g_params.numPostCollisionIterations = 10;

		g_params.shapeCollisionMargin = 0.04f;

		roll = 0.0f;
		pitch = 0.0f;
		yaw = -90.0f;
        rigidTrans.clear();
	
        urdf = new URDFImporter("../../data/sawyer", "/sawyer_description/urdf/sawyer_with_gripper.urdf", false,  0.005f, 0.005f, true, 20, false); // sawyer_with_gripper.urdf
        
		Transform gt(Vec3(0.0f, 0.3f, -0.25f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) 
					* QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

        urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 1000.0f, 0.0f, 1e1f, 0.01f, 20.7f, 7.0f, false, 1e-7f, 10.0f, 0);
        
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
        // fix base in place, todo: add a kinematic body flag?
        g_buffers->rigidBodies[0].invMass = 0.0f;
        (Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();

        fingerLeft = urdf->jointNameMap["r_gripper_l_finger_joint"];
        fingerRight = urdf->jointNameMap["r_gripper_r_finger_joint"];
		fingerWidthMin = 0.002f;

        NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["right_endpoint"]];

        // set up end effector targets
        NvFlexRigidJoint effectorJoint0;
        NvFlexMakeFixedJoint(&effectorJoint0, -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(0.2f, 0.7f, 0.5f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -kPi*0.5f)), NvFlexMakeRigidPose(0,0));
        for (int i = 0; i < 6; ++i)
        {
            effectorJoint0.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
            effectorJoint0.damping[i] = 1.e+3f;
            //effectorJoint0.maxIterations = 30;
        }

        effectorJoint = g_buffers->rigidJoints.size();
        g_buffers->rigidJoints.push_back(effectorJoint0);

    //    NvFlexRigidShape table;
    //    NvFlexMakeRigidBoxShape(&table, -1, 0.55f, 0.4f, 0.25f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.6f), Quat()));
    //    table.filter = 0;
    //    table.material.friction = 0.7f;	
	//	table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));
	//
    //    g_buffers->rigidShapes.push_back(table);
		
		forceLeft.resize(0);
		forceRight.resize(0);

		g_numSubsteps = 4;
		g_params.numIterations = 30;

		g_params.dynamicFriction = 1.25f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);
		g_drawPoints = false;

        if (mode == eRigid)
        {
            // Box object
            NvFlexRigidShape capsule;
            NvFlexMakeRigidCapsuleShape(&capsule, 0, 0.125f, 0.25f, NvFlexMakeRigidPose(0,0));

            float scale = 0.05f;

            Mesh* boxMesh = ImportMesh("../../data/box.ply");
            boxMesh->Transform(ScaleMatrix(scale));

            for (int i = 0; i < 1; ++i)
            {
                NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.005f);

                NvFlexRigidShape box;
                //NvFlexMakeRigidBoxShape(&box, g_buffers->rigidBodies.size(), 0.125f*scale, 0.125f*scale, 0.125f*scale, NvFlexMakeRigidPose(0,0));
                NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
                box.filter = 0x0;
                box.material.friction = 1.0f;
                box.thickness = 0.005f;

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
            int beamx = 6;
            int beamy = 6;
            int beamz = 24;

            float radius = 0.01f;
            float density = 500.0f;

            CreateTetGrid(Vec3(-beamx/2*radius, 0.475f, 0.5f), beamx, beamy, beamz, radius, radius, radius, density, ConstantMaterial<0>, false);

            g_params.radius = radius;
			g_params.collisionDistance = 0.0f;

            g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

            g_tetraMaterials.resize(0);
            g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+9f, 0.4f, 0.0005f));

			g_drawCloth = false;
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

			for (int i = 0; i < segments; ++i)
			{
				int bodyIndex = g_buffers->rigidBodies.size();

				NvFlexRigidShape shape;
				NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0,0));
				shape.filter = 0;
				shape.material.rollingFriction = 0.001f;
				shape.material.friction = 0.25f;
				shape.user = UnionCast<void*>(linkMaterial);
				
				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, startPos + Vec3(i * linkLength * 2.0f + linkLength, 0.0f, 0.0f), Quat(), &shape, &density, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);

				const float bendingCompliance = 1.e-2f;
				const float torsionCompliance = 1.e-6f;

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

		// add camera, todo: read correct links etc from URDF, right now these are thrown away
	//	const int headLink = urdf->rigidNameMap["head_tilt_link"];
	//	AddPrimesenseSensor(headLink, Transform(Vec3(-0.1f, 0.2f, 0.0f), rpy2quat(-1.57079632679f, 0.0f, -1.57079632679f)), 0.5f, hasFluids);

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
		imguiSlider("Gripper X", &targetx, -0.5f, 0.5f, 0.0001f);
        imguiSlider("Gripper Y", &targety, 0.0f, 1.5f, 0.0001f);
        imguiSlider("Gripper Z", &targetz, -0.5f, 1.2f, 0.0001f);
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
        float rectWidth = dx * float(numSamples) + rectMargin * 4.0f;

        float x = float(g_screenWidth) - rectWidth - 20.0f;
        float y = 300.0f;

        DrawRect(x, y - height - rectMargin, rectWidth, 2.0f * height + rectMargin * 3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));

        x += rectMargin * 3.0f;

        DrawImguiString(int(x + dx * float(numSamples)), int(y + 55.0f), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

        DrawLine(x, y, x + float(numSamples) * dx, y, 1.0f, Vec3(1.0f));
        DrawLine(x, y -50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

        float margin = 5.0f;

        DrawImguiString(int(x - margin), int(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
        DrawImguiString(int(x - margin), int(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
        DrawImguiString(int(x - margin), int(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

        for (int i = start; i < end - 1; ++i)
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

    virtual void Update()
    {
        // record force on finger joints
        forceLeft.push_back(g_buffers->rigidJoints[fingerLeft].lambda[eNvFlexRigidJointAxisX]);
        forceRight.push_back(g_buffers->rigidJoints[fingerRight].lambda[eNvFlexRigidJointAxisX]);

		// move end-effector via numpad
		NvFlexRigidJoint joint = g_buffers->rigidJoints[effectorJoint];
		// x
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
		joint.pose0.q[3] = currentRot.w;
		g_buffers->rigidJoints[effectorJoint] = joint;
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
				for (int i = 0; i < g_buffers->tetraIndices.size(); i += 4)
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

				rangeMin = 0.0f; //Min(rangeMin, vonMises);
				rangeMax = 0.5f; //Max(rangeMax, vonMises);

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


