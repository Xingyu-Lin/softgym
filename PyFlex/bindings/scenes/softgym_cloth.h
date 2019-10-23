
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

    softgym_FlagCloth(const char *name) : Scene(name) {}

    void Initialize(py::array_t<float> scene_params, int thread_idx = 0) {
        auto ptr = (float *) scene_params.request().ptr;

        int dimx = (int) ptr[0]; //64;
        int dimz = (int) ptr[1]; //32;
        int render_type = (int) ptr[2]; // 0: only points, 1: only mesh, 2: points + mesh

        cam_x = ptr[3];
        cam_y = ptr[4];
        cam_z = ptr[5];
        cam_angle_x = ptr[6];
        cam_angle_y = ptr[7];
        cam_angle_z = ptr[8];
        cam_width = int(ptr[9]);
        cam_height = int(ptr[10]);

        float radius = 0.05f;

        float stretchStiffness = 0.9f;
        float bendStiffness = 1.0f;
        float shearStiffness = 0.9f;
        int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);

        CreateSpringGrid(Vec3(0.0f, -0.025f, -3.0f), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness,
                         shearStiffness, 0.0f, 1.0f);

        const int c1 = 0;
        const int c2 = dimx * (dimz - 1);

        g_buffers->positions[c1].w = 0.0f;
        g_buffers->positions[c2].w = 0.0f;

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
                    // CreateSpring(c1, i, stiffness, give);
                    // CreateSpring(c2, i, stiffness, give);

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

        // draw options
        g_drawPoints = render_type == 0 || render_type == 2;
        g_drawMesh = render_type == 1 || render_type == 2;
        g_drawSprings = false;
        g_windFrequency *= 2.0f;
        g_windStrength = 10.0f;

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

