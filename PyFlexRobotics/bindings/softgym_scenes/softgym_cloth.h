#pragma once
#include <iostream>
#include <vector>

#include "../urdf.h"
#include "../deformable.h"
#include "softgym_sawyer.h"


char boxMeshPath[100];
char* make_path(char* full_path, std::string path);

class SoftgymCloth : public Scene
{
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;
    char urdfPath[100];
    SoftgymSawyer* ptrRobot = NULL;

	SoftgymCloth(){}

    SoftgymSawyer* getPtrRobot() {return ptrRobot;}

    void Initialize(py::array_t<float> scene_params = py::array_t<float>(),
                    py::array_t<float> robot_params = py::array_t<float>(), int thread_idx = 0)
    {

        auto ptr = (float *) scene_params.request().ptr;
	    float initX = ptr[0];
	    float initY = ptr[1];
	    float initZ = ptr[2];

		int dimx = (int)ptr[3]; //64;
		int dimz = (int)ptr[4]; //32;
		float radius = 0.00625f;

        int render_type = ptr[8]; // 0: only points, 1: only mesh, 2: points + mesh

        cam_x = ptr[9];
        cam_y = ptr[10];
        cam_z = ptr[11];
        cam_angle_x = ptr[12];
        cam_angle_y = ptr[13];
        cam_angle_z = ptr[14];
        cam_width = int(ptr[15]);
        cam_height = int(ptr[16]);

        // Load robot
        auto ptrRobotParams = (float *) robot_params.request().ptr;
        if (ptrRobotParams!=NULL &&  robot_params.size()>0) // Use robot
        {
            cout<<robot_params.size();
            ptrRobot = new SoftgymSawyer();
            ptrRobot->Initialize(robot_params); // XY: For some reason this has to be before creation of other rigid body
        }

        // Cloth
        float stretchStiffness = ptr[5]; //0.9f;
		float bendStiffness = ptr[6]; //1.0f;
		float shearStiffness = ptr[7]; //0.9f;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
		float mass = float(ptr[17])/(dimx*dimz);	// avg bath towel is 500-700g
        int flip_mesh = int(ptr[18]); // Flip half
	    CreateSpringGrid(Vec3(initX, -initY, initZ), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f/mass);
	    // Flip the last half of the mesh for the folding task
	    if (flip_mesh)
	    {
	        int size = g_buffers->triangles.size();
//	        for (int j=int((dimz-1)*3/8); j<int((dimz-1)*5/8); ++j)
//	            for (int i=int((dimx-1)*1/8); i<int((dimx-1)*3/8); ++i)
//	            {
//	                int idx = j *(dimx-1) + i;
//
//	                if ((i!=int((dimx-1)*3/8-1)) && (j!=int((dimz-1)*3/8)))
//	                    swap(g_buffers->triangles[idx* 3 * 2], g_buffers->triangles[idx*3*2+1]);
//	                if ((i != int((dimx-1)*1/8)) && (j!=int((dimz-1)*5/8)-1))
//	                    swap(g_buffers->triangles[idx* 3 * 2 +3], g_buffers->triangles[idx*3*2+4]);
//                }
	        for (int j=0; j<int((dimz-1)); ++j)
	            for (int i=int((dimx-1)*1/8); i<int((dimx-1)*1/8)+5; ++i)
	            {
	                int idx = j *(dimx-1) + i;

	                if ((i!=int((dimx-1)*1/8+4)))
	                    swap(g_buffers->triangles[idx* 3 * 2], g_buffers->triangles[idx*3*2+1]);
	                if ((i != int((dimx-1)*1/8)))
	                    swap(g_buffers->triangles[idx* 3 * 2 +3], g_buffers->triangles[idx*3*2+4]);
                }
        }
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

        g_params.radius = radius*1.8f;
        g_params.collisionDistance = 0.005f;

        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >>1;
        g_drawSprings = false;


        if (ptrRobotParams!=NULL &&  robot_params.size()>0) // Use robot
        {
            // Table
            NvFlexRigidShape table;
            // Half x, y, z
            NvFlexMakeRigidBoxShape(&table, -1, 0.64f, 0.55f, 0.4f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));
            table.filter = 0;
            table.material.friction = 0.95f;
            table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.35f, 0.45f, 0.65f)));

            float density = 1000.0f;
            NvFlexRigidBody body;
            NvFlexMakeRigidBody(g_flexLib, &body, Vec3(1.0f, 1.0f, 0.0f), Quat(), &table, &density, 1);

            g_buffers->rigidShapes.push_back(table);
            g_buffers->rigidBodies.push_back(body);
        }

        bool hasFluids = false;
        DepthRenderProfile p = {
			0.f, // minRange
			5.f // maxRange
		};
        if (g_render) // ptr[19] is whether to use a depth sensor
        {
            printf("adding a sensor in softgym_cloth!\n");
            AddSensor(cam_width, cam_height,  0,  Transform(Vec3(cam_x, cam_y, cam_z), rpy2quat(cam_angle_x, cam_angle_y, cam_angle_z)),  DegToRad(60.f), hasFluids, p);
        }

        // Box object
//        float scaleBox = 0.05f;
//        float densityBox = 2000000000.0f;

//        Mesh* boxMesh = ImportMesh(make_path(boxMeshPath, "/data/box.ply"));
//        boxMesh->Transform(ScaleMatrix(scaleBox));
//
//        NvFlexTriangleMeshId boxId = CreateTriangleMesh(boxMesh, 0.00125f);
//
//        NvFlexRigidShape box;
//        NvFlexMakeRigidTriangleMeshShape(&box, g_buffers->rigidBodies.size(), boxId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
//        box.filter = 0x0;
//        box.material.friction = 1.0f;
//        box.material.torsionFriction = 0.1;
//        box.material.rollingFriction = 0.0f;
//        box.thickness = 0.00125f;
//
//        NvFlexRigidBody boxBody;
//        NvFlexMakeRigidBody(g_flexLib, &boxBody, Vec3(0.21f, 0.7f, -0.1375f), Quat(), &box, &density, 1);
//
//        g_buffers->rigidBodies.push_back(boxBody);
//        g_buffers->rigidShapes.push_back(box);

//        g_params.numPostCollisionIterations = 15;

    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }
};