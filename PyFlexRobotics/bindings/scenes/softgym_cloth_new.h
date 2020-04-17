#pragma once
#include <iostream>
#include <vector>

#include "../urdf.h"
#include "../deformable.h"

char urdfPath[100];
char boxMeshPath[100];
char* make_path(char* full_path, std::string path);

class softgymCloth : public Scene
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
	int effectorJoint;

	int fingerLeft;
	int fingerRight;
	float fingerWidth = 0.03f;
	float fingerWidthMin = 0.002f;
	float fingerWidthMax = 0.05f;
	float roll, pitch, yaw;
	bool hasFluids = false;
	float numpadJointTraSpeed = 0.1f / 60.f; // 10cm/s under 60 fps
	float numpadJointRotSpeed = 10 / 60.f; // 10 deg/s under 60 fps
	float numpadJointRotDir = 1.f; // direction of rotation
	float numpadFingerSpeed = 0.02f / 60.f; // 2cm/s under 60 fps
    
    DeformableMesh* deformable = NULL;

    bool drawMesh = true;

    int scalesJoint = -1;
    int scalesRoot = -1;

    int headPanJoint = -1;
    int headTiltJoint = -1;

    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;

	softgymCloth(Mode mode) : mode(mode)
    {
//		roll = 0.0f;
//		pitch = 0.0f;
//		yaw = -90.0f;
//        rigidTrans.clear();
//        urdf = new URDFImporter(make_path(urdfPath, "/data/fetch_ros-indigo-devel"), "fetch_description/robots/fetch.urdf", false);

//		Transform gt(Vec3(0.0f, 0.0f, -0.25f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)*QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));

		// hide collision shapes
//		const int hiddenMaterial = AddRenderMaterial(0.5f, 0.5f, 0.5f, true);

//        urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 1000.0f, 0.0f, 1e1f, 0.01f, 20.7f, 7.0f, false);

//        for (int i=0; i < g_buffers->rigidShapes.size(); ++i)
//            g_buffers->rigidShapes[i].thickness = 0.001f;

//		for (int i = 0; i < (int)urdf->joints.size(); i++)
//        {
//            URDFJoint* j = urdf->joints[i];
//            NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[j->name]];
//
//            if (j->type == URDFJoint::REVOLUTE)
//            {
//                joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-8f;	// 10^6 N/m
//                joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+4f;	// 5*10^5 N/m/s
//            }
//            else if (j->type == URDFJoint::PRISMATIC)
//            {
//                joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
//                joint.targets[eNvFlexRigidJointAxisX] = joint.lowerLimits[eNvFlexRigidJointAxisX];
//                joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f;
//                joint.damping[eNvFlexRigidJointAxisX] = 0.0f;//1.e+4;
//            }
//        }

        // fix base in place, todo: add a kinematic body flag?
//        g_buffers->rigidBodies[0].invMass = 0.0f;
//        (Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();
//
//        fingerLeft = urdf->jointNameMap["l_gripper_finger_joint"];
//        fingerRight = urdf->jointNameMap["r_gripper_finger_joint"];

//        NvFlexRigidJoint* fingers[2] = { &g_buffers->rigidJoints[fingerLeft], &g_buffers->rigidJoints[fingerRight] };
//        for (int i=0; i < 2; ++i)
//        {
//            fingers[i]->modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
//            fingers[i]->targets[eNvFlexRigidJointAxisX] = 0.02f;
//            fingers[i]->compliance[eNvFlexRigidJointAxisX] = 1.e-6f;
//            fingers[i]->damping[eNvFlexRigidJointAxisX] = 0.0f;
//			fingers[i]->motorLimit[eNvFlexRigidJointAxisX] = 40.0f;
//        }

//        headPanJoint = urdf->jointNameMap["head_pan_joint"];
//        headTiltJoint = urdf->jointNameMap["head_tilt_joint"];
//
//        NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["l_gripper_finger_joint"]];

        // set up end effector targets
//        NvFlexRigidJoint effectorJoint0;
//        NvFlexMakeFixedJoint(&effectorJoint0, -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(-0.2f, 0.7f, 0.5f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -kPi*0.5f)), NvFlexMakeRigidPose(0,0));
//        for (int i = 0; i < 6; ++i)
//        {
//            effectorJoint0.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
//            effectorJoint0.damping[i] = 1.e+3f;
//            //effectorJoint0.maxIterations = 30;
//        }

//        effectorJoint = g_buffers->rigidJoints.size();
//        g_buffers->rigidJoints.push_back(effectorJoint0);

		// table
        NvFlexRigidShape table;
        // Half x, y, z
        NvFlexMakeRigidBoxShape(&table, -1, 0.27f, 0.15f, 0.3f, NvFlexMakeRigidPose(Vec3(-0.04f, 0.0f, 0.0f), Quat()));
        table.filter = 0;
        table.material.friction = 0.95f;
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.35f, 0.45f, 0.65f)));

        float density = 1000.0f;
        NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(1.0f, 1.0f, 0.0f), Quat(), &table, &density, 1);

        g_buffers->rigidShapes.push_back(table);
        g_buffers->rigidBodies.push_back(body);

        // Box object
        float scaleBox = 0.05f;
        float densityBox = 2000000000.0f;

        Mesh* boxMesh = ImportMesh(make_path(boxMeshPath, "/data/box.ply"));
        boxMesh->Transform(ScaleMatrix(scaleBox));

        NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.00125f);

        NvFlexRigidShape box;
        NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
        box.filter = 0x0;
        box.material.friction = 1.0f;
        box.material.torsionFriction = 0.1;
        box.material.rollingFriction = 0.0f;
        box.thickness = 0.00125f;

        NvFlexRigidBody boxBody;
        NvFlexMakeRigidBody(g_flexLib, &boxBody, Vec3(0.21f, 0.3f, -0.1375f), Quat(), &box, &density, 1);

        g_buffers->rigidBodies.push_back(boxBody);
        g_buffers->rigidShapes.push_back(box);

//        g_buffers->rigidJoints[fingerLeft].damping[eNvFlexRigidJointAxisX] = 0.0f;
//        g_buffers->rigidJoints[fingerLeft].motorLimit[eNvFlexRigidJointAxisX] = 40.0f;
//
//        g_buffers->rigidJoints[fingerRight].damping[eNvFlexRigidJointAxisX] = 0.0f;
//        g_buffers->rigidJoints[fingerRight].motorLimit[eNvFlexRigidJointAxisX] = 40.0f;

        g_params.numPostCollisionIterations = 15;


//        NvFlexRigidShape heavyBlock;
//        // Half x, y, z
//        NvFlexMakeRigidBoxShape(&heavyBlock, -1, 0.05f, 0.05f, 0.05f, NvFlexMakeRigidPose(Vec3(0.2f, 0.5f, -0.2f), Quat()));
//        table.filter = 0;
//        table.material.friction = 0.95f;
//		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.85f, 0.15f, 0.15f)));
//
//        float densityHeavyBlock = 100000.0f;
//        NvFlexRigidBody bodyHeavyBlock;
//		NvFlexMakeRigidBody(g_flexLib, &bodyHeavyBlock, Vec3(1.0f, 1.0f, 0.0f), Quat(), &heavyBlock, &densityHeavyBlock, 1);
//
//        g_buffers->rigidShapes.push_back(heavyBlock);
//        g_buffers->rigidBodies.push_back(bodyHeavyBlock);


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
        g_params.gravity[1] = -9.81f;

		/*
		// add camera, todo: read correct links etc from URDF, right now these are thrown away
		const int headLink = urdf->rigidNameMap["head_tilt_link"];
		AddPrimesenseSensor(headLink, Transform(Vec3(-0.1f, 0.2f, 0.0f), rpy2quat(-1.57079632679f, 0.0f, -1.57079632679f)), 1.f, hasFluids);
		*/

//        g_pause = true;
//
//        if (mode == eFlexibleBeam)
//        {
//            FILE* file = fopen("beam.bin", "rb");
//            if (file)
//            {
//                fread(&g_buffers->rigidBodies[0], sizeof(NvFlexRigidBody), g_buffers->rigidBodies.size(), file);
//                fclose(file);
//            }
//        }
        }
    void Initialize(py::array_t<float> scene_params, int thread_idx=0)
    {
        // Cloth
//        const float radius = 0.00625f;

//            float stretchStiffness = 0.8f;
//            float bendStiffness = 0.25f;
//            float shearStiffness = 0.25f;


//            float mass = 0.5f/(dimx*dimy);	// avg bath towel is 500-700g

//            CreateSpringGrid(Vec3(-0.3f, 0.5f, 0.45f), dimx, dimy, 1, radius, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f/mass);

//            g_params.radius = radius*1.8f;
//			g_params.collisionDistance = 0.005f;
//
//			g_drawCloth = true;



        auto ptr = (float *) scene_params.request().ptr;
	    float initX = ptr[0];
	    float initY = ptr[1];
	    float initZ = ptr[2];

		int dimx = (int)ptr[3]; //64;
		int dimz = (int)ptr[4]; //32;
		float radius = 0.00625f;
//
        int render_type = ptr[8]; // 0: only points, 1: only mesh, 2: points + mesh

        cam_x = ptr[9];
        cam_y = ptr[10];
        cam_z = ptr[11];
        cam_angle_x = ptr[12];
        cam_angle_y = ptr[13];
        cam_angle_z = ptr[14];
        cam_width = int(ptr[15]);
        cam_height = int(ptr[16]);

		float stretchStiffness = ptr[5]; //0.9f;
		float bendStiffness = ptr[6]; //1.0f;
		float shearStiffness = ptr[7]; //0.9f;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
		float mass = float(ptr[17])/(dimx*dimz);	// avg bath towel is 500-700g
	    CreateSpringGrid(Vec3(initX, -initY, initZ), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f/mass);
//
//
		g_numSubsteps = 4;
		g_params.numIterations = 30;

		g_params.dynamicFriction = 0.75f;
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);
		g_drawPoints = false;
//
        g_params.radius = radius*1.8f;
        g_params.collisionDistance = 0.005f;
//
        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >>1;
        g_drawSprings = false;
    }
    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }
};