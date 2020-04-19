#include "softgym_sawyer.h"
class softgym_FlagCloth : public Scene {
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;

	softgym_FlagCloth(){}
    SoftgymSawyer* getPtrRobot() {return ptrRobot;}

    float get_param_float(py::array_t<float> scene_params, int idx)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float out = ptr[idx];
        return out;
    }
    //params ordering: xpos, ypos, zpos, xsize, zsize, stretch, bend, shear
    // render_type, cam_X, cam_y, cam_z, angle_x, angle_y, angle_z, width, height
	void Initialize(py::array_t<float> scene_params = py::array_t<float>(),
                    py::array_t<float> robot_params = py::array_t<float>(), int thread_idx = 0)
	{
//	    cout << "initing" << endl;
        auto ptr = (float *) scene_params.request().ptr;
	    float initX = ptr[0];
	    float initY = ptr[1];
	    float initZ = ptr[2];

		int dimx = (int)ptr[3]; //64;
		int dimz = (int)ptr[4]; //32;
		float radius = 0.05f;



        int render_type = ptr[8]; // 0: only points, 1: only mesh, 2: points + mesh

        cam_x = ptr[9];
        cam_y = ptr[10];
        cam_z = ptr[11];
        cam_angle_x = ptr[12];
        cam_angle_y = ptr[13];
        cam_angle_z = ptr[14];
        cam_width = int(ptr[15]);
        cam_height = int(ptr[16]);

        //float radius = 0.05f;

        // Load robot
        auto ptrRobotParams = (float *) robot_params.request().ptr;
        if (ptrRobotParams!=NULL &&  robot_params.size()>0) // Use robot
        {
            cout<<robot_params.size();
            ptrRobot = new SoftgymSawyer();
            ptrRobot->Initialize(robot_params); // XY: For some reason this has to be before creation of other rigid body

            // For SoftGym 1.0 release, only have table when there is a robot
            // table
            NvFlexRigidShape table;
            // Half x, y, z
            NvFlexMakeRigidBoxShape(&table, -1, 0.55f, 0.55f, 0.34f, NvFlexMakeRigidPose(Vec3(-0.04f, 0.0f, 0.0f), Quat()));
            table.filter = 0;
            table.material.friction = 0.95f;
            table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.35f, 0.45f, 0.65f)));

            float density = 1000.0f;
            NvFlexRigidBody body;
            NvFlexMakeRigidBody(g_flexLib, &body, Vec3(1.0f, 1.0f, 0.0f), Quat(), &table, &density, 1);

            g_buffers->rigidShapes.push_back(table);
            g_buffers->rigidBodies.push_back(body);
        }
		float stretchStiffness = ptr[5]; //0.9f;
		float bendStiffness = ptr[6]; //1.0f;
		float shearStiffness = ptr[7]; //0.9f;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);

	    CreateSpringGrid(Vec3(initX, -initY, initZ), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f);
		//CreateSpringGrid(Vec3(0.0f, -1.0f, -3.0f), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f);

        const int c1 = 0;
        const int c2 = dimx * (dimz - 1);

		//g_buffers->positions[c1].w = 0.0f;
		//g_buffers->positions[c2].w = 0.0f;

        // add tethers
        for (int i = 0; i < int(g_buffers->positions.size()); ++i) {
            // hack to rotate cloth
            // swap(g_buffers->positions[i].y, g_buffers->positions[i].z);
            g_buffers->positions[i].y *= -1.0f;

            g_buffers->velocities[i] = RandomUnitVector() * 0.1f;

            float minSqrDist = FLT_MAX;

            //if (i!=c1 && i!=c2)
            if (1) {
                float stiffness = -0.8f;
                float give = 0.1f;

                float sqrDist = LengthSq(Vec3(g_buffers->positions[c1]) - Vec3(g_buffers->positions[c2]));

                if (sqrDist < minSqrDist) {
//                     CreateSpring(c1, i, stiffness, give);
//                     CreateSpring(c2, i, stiffness, give);

                    minSqrDist = sqrDist;
                }
            }
        }

        g_params.radius = radius * 1.0f;
        g_params.dynamicFriction = 0.25f;
        g_params.dissipation = 0.0f;
        g_params.numIterations = 4;
        g_params.drag = 0.06f;
        g_params.relaxationFactor = 1.0f;

        g_numSubsteps = 2;

//        cout<<"render_type: "<<  render_type<<endl;
        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >>1;
        g_drawSprings = false;
        g_windFrequency *= 2.0f;
        g_windStrength = 0.0f;
//        cout << "finish init" << endl;
    }


    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }

    void Update() {
        const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
        const float kNoise = fabsf(Perlin1D(g_windTime * 0.05f, 2, 0.25f));
        Vec3 wind = g_windStrength * kWindDir * Vec3(kNoise, kNoise * 0.1f, -kNoise * 0.1f);

        g_params.wind[0] = wind.x;
        g_params.wind[1] = wind.y;
        g_params.wind[2] = wind.z;
    }
};

