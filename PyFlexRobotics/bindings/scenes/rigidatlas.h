#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"

class RigidAtlas : public Scene
{
public:

    URDFImporter* urdf;

    vector<Transform> rigidTrans;
    map<string, int> jointMap;
    map<string, int> activeJointMap;
	int effectorJoint;

	int fingerLeft;
	int fingerRight;
	float fingerWidth = 0.03f;
	float fingerWidthMin = 0.0f;
	float roll, pitch, yaw;

	Mesh mesh;

    RigidAtlas()
    {
		// soft mat
		const float density = 1000.0f;

		int beamx = 20;
		int beamy = 2;
		int beamz = 10;

		CreateTetGrid(Vec3(-1.0f, 0.02f, 0.5f), beamx, beamy, beamz, 0.1f, 0.1f, 0.1f, density, ConstantMaterial<0>, true);
		g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

		g_tetraMaterials.resize(0);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+9f, 0.4f, 0.0f));

		roll = 0.0f;
		pitch = 0.0f;
		yaw = -90.0f;
        rigidTrans.clear();
	
		// box
        NvFlexRigidShape box;
        NvFlexMakeRigidBoxShape(&box, -1, 0.5f, 0.5f, 0.25f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));
        box.filter = 0;
        box.material.friction = 0.7f;	
		box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));

        g_buffers->rigidShapes.push_back(box);
		
		urdf = new URDFImporter("../../data", "atlas_description/urdf/atlas_v5_simple_shapes.urdf");
        
		Transform gt(Vec3(0.0f, 2.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

        urdf->AddPhysicsEntities(gt, hiddenMaterial, true, true, 1000.0f, 0.0f, 1e1f, 0.01f, 20.7f, 7.0f, false);
        
        forceLeft.resize(0);
        forceRight.resize(0);

        g_numSubsteps = 4;
        g_params.numIterations = 30;

        g_params.dynamicFriction = 1.25f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;
		
        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.005f;

        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);

        g_pause = true;

        g_drawPoints = false;
        g_drawCloth = false;

    }

    virtual void DoGui()
    {
        /*
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
        imguiSlider("Finger Width", &newWidth, fingerWidthMin, 0.05f, 0.001f);

        fingerWidth = Lerp(fingerWidth, newWidth, smoothing);

        g_buffers->rigidJoints[fingerLeft].targets[eNvFlexRigidJointAxisX] = fingerWidth;
        g_buffers->rigidJoints[fingerRight].targets[eNvFlexRigidJointAxisX] = fingerWidth;
        */
    }

    virtual void DoStats()
    {
        /*
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

        DrawImguiString(x + dx*numSamples, y + 55.0f, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

        DrawLine(x, y, x + numSamples*dx, y, 1.0f, Vec3(1.0f));
        DrawLine(x, y -50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));

        float margin = 5.0f;

        DrawImguiString(x - margin, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
        DrawImguiString(x - margin, y + height - lineHeight, Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
        DrawImguiString(x - margin, y - height, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

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
        */
    }

    std::vector<float> forceLeft;
    std::vector<float> forceRight;

    virtual void Update()
    {
        // record force on finger joints
        //forceLeft.push_back(g_buffers->rigidJoints[fingerLeft].lambda[eNvFlexRigidJointAxisX]);
        //forceRight.push_back(g_buffers->rigidJoints[fingerRight].lambda[eNvFlexRigidJointAxisX]);
    }

    virtual void PostUpdate()
    {
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }

	virtual void Draw(int pass)
	{		
		if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i = 0; i < (int)g_buffers->triangles.size(); ++i)
			{
				mesh.m_indices[i] = g_buffers->triangles[i];
			}

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i=0; i < (int)g_buffers->tetraIndices.size(); i += 4)
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

			for (int i = 0; i < (int)g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x/averageStress[i].y);
				/*
				if (g_buffers->tetraMaterials[i] == 0)
					mesh.m_colours[i] = Colour::kGreen;
				else
					mesh.m_colours[i] = Colour::kRed;
				*/

			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);
	}
};


