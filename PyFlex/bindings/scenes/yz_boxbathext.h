class yz_BoxBathExt: public Scene
{
public:

	yz_BoxBathExt(const char* name, bool dam) : Scene(name), mDam(dam) {}

	char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}

	float rand_float(float LO, float HI) {
        return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(HI-LO)));
    }

    void swap(float* a, float* b) {
	    float tmp = *a;
	    *a = *b;
	    *b = tmp;
	}

	void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	{
		float radius = 0.1f;

		// deforming bunny
		float s = radius*0.5f;
		float m = 0.25f;
		int group = 1;

		char bunny_path[100];
		char box_path[100];
		char sphere_path[100];
		make_path(bunny_path, "/data/bunny.ply");
		make_path(box_path, "/data/box.ply");
		make_path(sphere_path, "/data/sphere.ply");

        srand(time(NULL) + thread_idx);

        auto ptr = (float *) scene_params.request().ptr;
        // float x = rand_float(0.45, 1.0);
        // float z = rand_float(0.05, 0.15);
        // float scale = rand_float(0.5, 0.1);
        float x = ptr[0];
        float y = ptr[1];
        float z = ptr[2];
        float px = ptr[3];
        float pz = ptr[4];

        // void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)

        //CreateParticleShape(GetFilePathByPlatform(box_path).c_str(), Vec3(px, -0.0155f, pz), Vec3(x, y, z), 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), m, true, 1.0f, NvFlexMakePhase(group++, 0), false, 0.0f);
		//CreateParticleShape(GetFilePathByPlatform(box_path).c_str(), Vec3(x_1, -0.0155f, z_1), 0.2f, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), m, true, 1.0f, NvFlexMakePhase(group++, 0), false, 0.0f);
		//CreateParticleShape(GetFilePathByPlatform(bunny_path).c_str(), Vec3(3.0f, 0.0f, 0.0f), 0.5f, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f);
		//CreateParticleShape(GetFilePathByPlatform(sphere_path).c_str(), Vec3(1.0f, 0.0f, 1.0f), 0.45f, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f);
		//CreateParticleShape(GetFilePathByPlatform(bunny_path).c_str(), Vec3(2.0f, 0.0f, 1.0f), 0.5f, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f);
		//CreateParticleShape(GetFilePathByPlatform(box_path).c_str(), Vec3(2.0f, 0.0f, 0.0f), 0.45f, 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), m, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f);

		g_numSolidParticles = g_buffers->positions.size();		

		float restDistance = radius*0.55f;

		if (mDam)
		{
		    // void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)

			CreateParticleGrid(Vec3(0.0f, 0.0f, 0.0f), 8, 15, 8, restDistance, Vec3(0.0f), 1.0f, false, 0.0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid), 0.005f);
			g_lightDistance *= 0.5f;
		}

		CreateParticleShape(GetFilePathByPlatform(box_path).c_str(), Vec3(px, -0.0155f, pz), Vec3(x, y, z), 0.0f, s, Vec3(0.0f, 0.0f, 0.0f), m, true, 1.0f, NvFlexMakePhase(group++, 0), false, 0.0f);

		g_sceneLower = Vec3(-0.015f, 0.0f, 0.0f);
		g_sceneUpper = Vec3(1.2f, 0.0f, 0.0f);

		g_numSubsteps = 2;

		g_params.radius = radius;
		g_params.dynamicFriction = 0.01f; 
		g_params.viscosity = 2.0f;
		g_params.numIterations = 4;
		g_params.vorticityConfinement = 40.0f;
		g_params.fluidRestDistance = restDistance;
		g_params.solidPressure = 0.f;
		g_params.relaxationFactor = 0.0f;
		g_params.cohesion = 0.02f;
		g_params.collisionDistance = 0.01f;		

		g_maxDiffuseParticles = 0;
		g_diffuseScale = 0.5f;

		g_fluidColor = Vec4(0.113f, 0.425f, 0.55f, 1.f);

		Emitter e1;
		e1.mDir = Vec3(1.0f, 0.0f, 0.0f);
		e1.mRight = Vec3(0.0f, 0.0f, -1.0f);
		e1.mPos = Vec3(radius, 1.f, 0.65f);
		e1.mSpeed = (restDistance/g_dt)*2.0f; // 2 particle layers per-frame
		e1.mEnabled = true;

		g_emitters.push_back(e1);

		// g_numExtraParticles = 48*1024;

		g_lightDistance = 1.8f;

		g_params.numPlanes = 5;

		g_waveFloorTilt = 0.0f;
		g_waveFrequency = 1.5f;
		g_waveAmplitude = 2.0f;
		
		g_warmup = false;

		// draw options		
		g_drawPoints = true;
		g_drawMesh = false;
		g_drawEllipsoids = false;
		g_drawDiffuse = true;
	}

	bool mDam;
};
