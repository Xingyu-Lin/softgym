#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"


class RigidURDF : public Scene
{
public:
    URDFImporter* p_urdf;
    
	RigidURDF()
    {
        LLL;
        //p_urdf = new URDFImporter("../../data/r2d2/", "r2d2.urdf");
        LLL;
        //p_urdf = new URDFImporter("../../data", "yumi_description/urdf/yumi2.urdf");
        //p_urdf = new URDFImporter("../../data/PR2/", "../../data/PR2/pr2.urdf");
        //p_urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi.urdf");

        p_urdf = new URDFImporter("../../data/fetch_ros-indigo-devel", "fetch_description/robots/fetch.urdf");
        Transform gt(Vec3(0.0f, 0.1f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
        p_urdf->AddPhysicsEntities(gt, 0, false, 1000.0f, 0.0f, 1e1f, 0.0f, 0.0f, 7.0f, true, 1e-2f, 1e1f);

        // fix base in place, todo: add a kinematic body flag?
        g_buffers->rigidBodies[0].invMass = 0.0f;
        (Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();

        NvFlexRigidJoint handLeft = g_buffers->rigidJoints[p_urdf->activeJointNameMap["l_gripper_finger_link"]];
        NvFlexRigidJoint handRight = g_buffers->rigidJoints[p_urdf->activeJointNameMap["r_gripper_finger_link"]];


        LLL;
        // set up end effector targets
        NvFlexRigidJoint effectorJoint0, effectorJoint1;
        NvFlexMakeFixedJoint(&effectorJoint0, -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(0.2f, 0.5f, 0.5f), Quat()), NvFlexMakeRigidPose(0,0));
        for (int i=0; i < 6; ++i)
        {
            effectorJoint0.compliance[i] = 1.e-8f;
            effectorJoint0.damping[i] = 1.e+3f;
            effectorJoint0.maxIterations = 40;
        }

        NvFlexMakeFixedJoint(&effectorJoint1, -1, handRight.body0, NvFlexMakeRigidPose(Vec3(-0.2f, 0.5f, 0.5f), Quat()), NvFlexMakeRigidPose(0,0));
        for (int i=0; i < 6; ++i)
        {
            effectorJoint1.compliance[i] = 1.e-8f;
            effectorJoint1.damping[i] = 1.e+3f;
            effectorJoint1.maxIterations = 40;
        }

        g_buffers->rigidJoints.push_back(effectorJoint0);
        g_buffers->rigidJoints.push_back(effectorJoint1);
        LLL;

        NvFlexRigidShape table;
        NvFlexMakeRigidBoxShape(&table, -1, 0.5f, 0.25f, 0.25f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.5f), Quat()));
        table.filter = 0;
        LLL;
        g_buffers->rigidShapes.push_back(table);

        if (0)
        {

            // manipulation object
            NvFlexRigidShape capsule;
            NvFlexMakeRigidCapsuleShape(&capsule, 0, 0.125f, 0.25f, NvFlexMakeRigidPose(0,0));

            float scale = 0.04f;

            Mesh* boxMesh = ImportMesh("../../data/box.ply");
            boxMesh->Transform(ScaleMatrix(scale));

            NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.005f);

            NvFlexRigidShape box;
            //NvFlexMakeRigidBoxShape(&box, g_buffers->rigidBodies.size(), 0.125f*scale, 0.125f*scale, 0.125f*scale, NvFlexMakeRigidPose(0,0));
            NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0,0), 1.0f, 1.0f, 1.0f);
            box.filter = 0x0;
            box.material.friction = 1.0f;
            box.thickness = 0.005f;

			const float density = 10.0f;

            NvFlexRigidBody body;
            NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.25f + scale*0.5f + 0.01f, 0.6f), Quat(), &box, &density, 1);

            g_buffers->rigidBodies.push_back(body);
            g_buffers->rigidShapes.push_back(box);
        }
        else
        {
            // FEM object
            int beamx = 4;
            int beamy = 4;
            int beamz = 4;

            float radius = 0.02f;
            float density = 500.0f;

            CreateTetGrid(Vec3(-beamx/2*radius, 0.275f, 0.5f), beamx, beamy, beamz, radius, radius, radius, density, ConstantMaterial<0>, false);

            g_params.radius = radius;

            g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

            g_tetraMaterials.resize(0);
            g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+9f, 0.4f, 0.005f));
        }

        g_numSubsteps = 4;
        g_params.numIterations = 50;
        g_params.dynamicFriction = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;
        g_params.collisionDistance = 0.01f;
        LLL;
        g_sceneLower = Vec3(-2.0f);
        g_sceneUpper = Vec3(2.0f);

        g_pause = true;
        LLL;
        g_drawPoints = false;
        g_drawCloth = false;
    }

    virtual void DoGui()
    {
        int i = g_buffers->rigidJoints.size()-2;

        NvFlexRigidJoint effector0 = g_buffers->rigidJoints[i];
        NvFlexRigidJoint effector1 = g_buffers->rigidJoints[i+1];

        float targetx = effector0.pose0.p[0];
        float targety = effector0.pose0.p[1];
        float targetz = effector0.pose0.p[2];

        imguiSlider("Gripper X", &targetx, -0.25f, 0.25f, 0.0001f);
        imguiSlider("Gripper Y", &targety, 0.0f, 1.0f, 0.001f);
        imguiSlider("Gripper Z", &targetz, 0.0f, 1.0f, 0.001f);

        // low-pass filter controls otherwise it is too jerky
        float newx = Lerp(effector0.pose0.p[0], targetx, 0.5f);
        float newy = Lerp(effector0.pose0.p[1], targety, 0.5f);
        float newz = Lerp(effector0.pose0.p[2], targetz, 0.5f);


        effector0.pose0.p[0] = newx;
        effector0.pose0.p[1] = newy;
        effector0.pose0.p[2] = newz;

        // mirror over the x-plane
        effector1.pose0.p[0] = -newx;
        effector1.pose0.p[1] = newy;
        effector1.pose0.p[2] = newz;

        g_buffers->rigidJoints[i+0] = effector0;
        g_buffers->rigidJoints[i+1] = effector1;

    }


    virtual void Draw(int pass)
    {
        if (pass == 1)
        {
            mesh.m_positions.resize(g_buffers->positions.size());
            mesh.m_normals.resize(g_buffers->normals.size());
            mesh.m_colours.resize(g_buffers->positions.size());
            mesh.m_indices.resize(g_buffers->triangles.size());

            for (int i=0; i < g_buffers->triangles.size(); ++i)
            {
                mesh.m_indices[i] = g_buffers->triangles[i];
            }

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
                /*
                if (g_buffers->tetraMaterials[i] == 0)
                	mesh.m_colours[i] = Colour::kGreen;
                else
                	mesh.m_colours[i] = Colour::kRed;
                */

            }
        }

        DrawMesh(&mesh, g_renderMaterials[0]);

        // wireframe overlay
        if (pass == 0)
        {

            SetFillMode(true);

			if (g_buffers->triangles.size())
				DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.00f);
            
			DrawRigidShapes(true);

            SetFillMode(false);

        }
    }


    Mesh mesh;

};


class RigidURDF2 : public Scene
{
public:


	URDFImporter* p_urdf;
	URDFImporter* p_urdf2;
	vector<string> motors;
	float mv[8];
	RigidURDF2()
	{
		LLL;
		//p_urdf = new URDFImporter("../../data/r2d2/", "r2d2.urdf");
		LLL;
		//p_urdf = new URDFImporter("../../data", "yumi_description/urdf/yumi3.urdf");
		//p_urdf = new URDFImporter("../../data/PR2/", "../../data/PR2/pr2.urdf");
		//p_urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi.urdf");
		p_urdf = new URDFImporter("../../data", "minitaur_original.urdf", false, 0.005f, 0.005f, true, 20);
		//p_urdf = new URDFImporter("../../data/urdfs/Fish_800_tex", "Fish_800_tex.urdf", false, 0.005f, 0.005f, true, 20);
		Transform gt(Vec3(0.0f, 0.7f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi*0.5f));
		p_urdf->AddPhysicsEntities(gt, 0, true, 1000.0f, 0.0f, 1e1f, 0.0f, 0.0f, 7.0f, true, 1e-2f, 1e1f);
		p_urdf->LumpFixedJointsAndSaveURDF("mini.urdf");

		int bd = g_buffers->rigidBodies.size();
		p_urdf2 = new URDFImporter("../../bin/win64/", "mini.urdf", false, 0.005f, 0.005f, true, 20);

		Transform gt2(Vec3(1.0f, 0.7f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi*0.5f));
		//p_urdf = new URDFImporter("../../data/fetch_ros-indigo-devel", "fetch_description/robots/fetch.urdf");
		//Transform gt(Vec3(0.0f, 0.1f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
		p_urdf2->AddPhysicsEntities(gt2, 0, true, 1000.0f, 0.0f, 1e1f, 0.0f, 0.0f, 7.0f, true, 1e-2f, 1e1f);


		NvFlexRigidJoint fj, fj2;

		NvFlexMakeFixedJoint(&fj, -1, 0, (NvFlexRigidPose&)gt, NvFlexMakeRigidPose(0, 0));
		NvFlexMakeFixedJoint(&fj2, -1, bd, (NvFlexRigidPose&)gt2, NvFlexMakeRigidPose(0, 0));
		g_buffers->rigidJoints.push_back(fj);
		g_buffers->rigidJoints.push_back(fj2);

		motors = { "0", "1", "2", "3", "4", "5", "6", "7","8","9","10","11","12","13","14","15" };

		// fix base in place, todo: add a kinematic body flag?
		//g_buffers->rigidBodies[0].invMass = 0.0f;
		//(Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();

		g_numSubsteps = 4;
		g_params.numIterations = 50;
		g_params.dynamicFriction = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;
		g_params.collisionDistance = 0.01f;
		LLL;
		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
		LLL;
		g_drawPoints = false;
		g_drawCloth = false;
		for (int i = 0; i < motors.size(); i++) {
			mv[i] = 0.0f;
		}
		for (int i = 0; i < g_buffers->rigidShapes.size(); i++) {
			g_buffers->rigidShapes[i].filter = 1;
		}
	}
	virtual void DoGui()
	{
		for (int i = 0; i < motors.size(); i++) {

			imguiSlider(motors[i].c_str(), &mv[i], -kPi + 0.01f, kPi - 0.01f, 0.0001f);
			g_buffers->rigidJoints[p_urdf->activeJointNameMap[motors[i]]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
			g_buffers->rigidJoints[p_urdf->activeJointNameMap[motors[i]]].targets[eNvFlexRigidJointAxisTwist] = mv[i];

			g_buffers->rigidJoints[p_urdf2->activeJointNameMap[motors[i]]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
			g_buffers->rigidJoints[p_urdf2->activeJointNameMap[motors[i]]].targets[eNvFlexRigidJointAxisTwist] = mv[i];
		}
	}
};

