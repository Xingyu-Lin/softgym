#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"
class RigidYumiGripper : public Scene
{
public:

    URDFImporter* urdf;

    float roll, pitch, yaw;
    float gripperWidth;

    int effectorIndex;

    const float dilation = 0.001f;
    const float thickness = 0.001f;


    void LoadSimpleOBJ(const char* path, const Transform& transform, bool flip =false)
    {
        Mesh* m = ImportMesh(path);

        if (flip)
        {
            m->Flip();
        }

        NvFlexTriangleMeshId mesh = CreateTriangleMesh(m, dilation);

        NvFlexRigidShape shape;
        NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), mesh, NvFlexMakeRigidPose(0,0), 1.0f, 1.0f, 1.0f);
        shape.filter = 0;
        shape.material.friction = 1.0f;
        shape.thickness = thickness;

		const float density = 1000.0f;

        NvFlexRigidBody body;
        NvFlexMakeRigidBody(g_flexLib, &body, transform.p, transform.q, &shape, &density, 1);

        g_buffers->rigidShapes.push_back(shape);
        g_buffers->rigidBodies.push_back(body);


    }

    RigidYumiGripper()
    {
        gripperWidth = 0.025f;

        roll = 90.0f;
        pitch = 0.0f;
        yaw = 0.0f;


        urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi_gripper.urdf", true, dilation, thickness);
        //urdf = new URDFImporter("../../data/r2d2/", "r2d2.urdf");
        //urdf = new URDFImporter("../../data/PR2/", "../../data/PR2/pr2.urdf");
        //urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi.urdf");

        Transform gt(Vec3(0.0f, 0.6f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
        urdf->AddPhysicsEntities(gt, 0, false, 1000.0f, 0.0f, 1e1f, 0.0f, 0.0f, 7.0f, true, 1e-7f, 1e1f);

        NvFlexRigidBody& gripperBase = g_buffers->rigidBodies[urdf->rigidNameMap["gripper_r_base"]];
        float factor = 0.05f;
        gripperBase.mass *= factor;
        gripperBase.invMass /= factor;
        (Matrix33&)gripperBase.inertia *= factor;
        (Matrix33&)gripperBase.invInertia *= (1.0f / factor);

        NvFlexRigidJoint handRight = g_buffers->rigidJoints[urdf->jointNameMap["gripper_r_joint"]];

        // set up end effector targets
        NvFlexRigidJoint effectorJoint0;
        NvFlexMakeFixedJoint(&effectorJoint0, -1, handRight.body0, NvFlexMakeRigidPose(Vec3(0.2f, 0.5f, 0.5f), Quat()), NvFlexMakeRigidPose(0, 0));

        for (int i = 0; i < 6; ++i)
        {
            effectorJoint0.compliance[i] = 1e-5f;
            effectorJoint0.damping[i] = 0.0f;
        }

        effectorJoint0.maxIterations = 30;


        effectorIndex = g_buffers->rigidJoints.size();
        g_buffers->rigidJoints.push_back(effectorJoint0);
        for (int i = 0; i < g_buffers->rigidBodies.size(); ++i)
        {
            g_buffers->rigidBodies[i].invMass = 10.0f;
            (Matrix33&)g_buffers->rigidBodies[i].invInertia = Matrix33::Identity()*10;
        }


        /*
        tableIndex = g_buffers->rigidBodies.size();
        NvFlexRigidShape table;
        NvFlexMakeRigidBoxShape(&table, tableIndex, 0.5f, 0.125f, 0.15f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));
        table.filter = 0;


        NvFlexRigidBody body;
        NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.2f, 0.5f), Quat(), &table, 1000000.0, 1);
        g_buffers->rigidBodies.push_back(body);
        g_buffers->rigidShapes.push_back(table);


        // manipulation object

        NvFlexRigidShape capsule;
        NvFlexMakeRigidCapsuleShape(&capsule, 0, 0.125f, 0.25f, NvFlexMakeRigidPose(0, 0));

        float scale = 0.01875*1.5;

        Mesh* boxMesh = ImportMesh("../../data/box.ply");
        boxMesh->Transform(ScaleMatrix(scale));

        NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.005f);

        NvFlexRigidShape box;
        //NvFlexMakeRigidBoxShape(&box, g_buffers->rigidBodies.size(), 0.125f*scale, 0.125f*scale, 0.125f*scale, NvFlexMakeRigidPose(0,0));
        NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
        box.filter = 0x0;
        box.material.friction = 1.0f;
        box.thickness = 0.005f;

        NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 1.25f + scale*0.5f + 0.01f, 0.6f), Quat(), &box, 10.0, 1);


        g_buffers->rigidBodies.push_back(body);
        g_buffers->rigidShapes.push_back(box);


        int beamx = 4;
        int beamy = 4;
        int beamz = 4;

        float radius = 0.0075f;

        CreateTetraGrid(Vec3(-beamx / 2 * radius, 1.0f, 0.5f), beamx, beamy, beamz, radius, radius, radius, 10000.0f, ConstantMaterial<0>, false);

        g_params.radius = radius;

        g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

        g_tetraMaterials.resize(0);
        g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+9f, 0.4f, 0.005f));
        */

        float x = -0.1f;
        float dx = 0.2f;

        LoadSimpleOBJ("../../data/dex-net/mini_dexnet/bar_clamp.obj", Transform(Vec3(x, 0.25f, 0.5f)));
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/mini_dexnet/endstop_holder.obj", Transform(Vec3(x, 0.25f, 0.5f)), true);
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/mini_dexnet/gearbox.obj", Transform(Vec3(x, 0.25f, 0.5f)));
        x += dx;

        LoadSimpleOBJ("../../data/dex-net/kit/CatSitting_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)));
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/kit/Clown_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)));
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/kit/DanishHam_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)));
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/kit/Fish_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)), true);
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/kit/Wineglass_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)), true);
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/kit/Dog_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)), true);
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/kit/GreenCup_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)));
        x += dx;
        LoadSimpleOBJ("../../data/dex-net/kit/Tortoise_800_tex.obj", Transform(Vec3(x, 0.25f, 0.5f)));
        x += dx;

        // set friction on all shapes
        for (int i = 0; i < g_buffers->rigidShapes.size(); i++)
        {
            NvFlexRigidShape& shape = g_buffers->rigidShapes[i];

            shape.material.friction = 0.8f;
            shape.material.restitution = 0.0f;
            shape.material.compliance = 0.0f;
        }

        g_numSubsteps = 4;
        g_params.numIterations = 50;
        g_params.numPostCollisionIterations = 10;

        g_params.dynamicFriction = 1.0f;
        g_params.staticFriction = 1.0f;
        g_params.shapeCollisionMargin = 0.02f;

        g_sceneLower = Vec3(0.0f);
        g_sceneUpper = Vec3(1.0f);

        g_pause = false;

    }

    ~RigidYumiGripper()
    {
    }

    virtual void DoGui()
    {
        NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorIndex];

        Vec3 ePos = effector0.pose0.p;

        Vec3 oePos = ePos;
        float oroll = roll;
        float opitch = pitch;
        float oyaw = yaw;

        imguiSlider("Gripper X", &ePos.x, -0.5f, 0.5f, 0.001f);
        imguiSlider("Gripper Y", &ePos.y, 0.0f, 1.0f, 0.001f);
        imguiSlider("Gripper Z", &ePos.z, 0.0f, 1.0f, 0.001f);
        imguiSlider("Roll", &roll, -180.0f, 180.0f, 0.01f);
        imguiSlider("Pitch", &pitch, -180.0f, 180.0f, 0.01f);
        imguiSlider("Yaw", &yaw, -180.0f, 180.0f, 0.01f);
        imguiSlider("Gripper W", &gripperWidth, 0.0f, 0.025f, 0.00001f);

        const float smoothing = 0.1f;

        roll = Lerp(oroll, roll, smoothing);
        pitch = Lerp(opitch, pitch, smoothing);
        yaw = Lerp(oyaw, yaw, smoothing);

        effector0.pose0.p[0] = Lerp(oePos.x, ePos.x, smoothing);
        effector0.pose0.p[1] = Lerp(oePos.y, ePos.y, smoothing);
        effector0.pose0.p[2] = Lerp(oePos.z, ePos.z, smoothing);

        Quat q = rpy2quat(roll*kPi / 180.0f, pitch*kPi / 180.0f, yaw*kPi / 180.0f);
        effector0.pose0.q[0] = q.x;
        effector0.pose0.q[1] = q.y;
        effector0.pose0.q[2] = q.z;
        effector0.pose0.q[3] = q.w;

        g_buffers->rigidJoints[urdf->activeJointNameMap["gripper_r_joint"]].targets[eNvFlexRigidJointAxisX] = gripperWidth;
        g_buffers->rigidJoints[urdf->activeJointNameMap["gripper_r_joint_m"]].targets[eNvFlexRigidJointAxisX] = gripperWidth;

        g_buffers->rigidJoints[effectorIndex] = effector0;

    }
    virtual void Update()
    {
    }
    virtual void Draw(int pass)
    {

        if (pass == 0)
        {
            SetFillMode(true);

            DrawRigidShapes(true);

            SetFillMode(false);
        }
    }
};


