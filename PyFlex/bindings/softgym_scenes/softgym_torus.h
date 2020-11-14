class SoftgymTorus : public Scene
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
	char torus_path[100];

	
	SoftgymTorus(const char* name) : Scene(name) {}

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
	    // scene_params

	    auto ptr = (float *) scene_params.request().ptr;

		float radius = ptr[0];
		float rest_dis_coef = ptr[1];
		
		int num = ptr[2]; // # of torus
		float size = ptr[3];
		float lowerx = ptr[4];
		float torus_height = ptr[5];
		float lowerz = ptr[6];
		float static_friction = float(ptr[7]);
		float dynamic_friction = float(ptr[8]);

	    cam_x = ptr[9];
		cam_y = ptr[10];
		cam_z = ptr[11];
		cam_angle_x = ptr[12];
		cam_angle_y = ptr[13];
		cam_angle_z = ptr[14];
		cam_width = int(ptr[15]);
		cam_height = int(ptr[16]);
		int render = int(ptr[17]);


		// printf("num: %d  size: %f\n", num, size);
		// printf("radius: %f \n", radius);

		/*
		The main particle radius is set via NvFlexParams::radius, which is the “interaction radius”.
		Particles closer than this distance will be able to affect each other.
		*/
		int group = 0;
		for (int i=0; i < num; ++i)
			// make_path(box_high_path, "/data/box_high.ply")
			// CreateParticleShape(GetFilePathByPlatform("../../data/torus.obj").c_str(), Vec3(4.5f, 2.0f + radius*2.0f*i, 1.0f), size, 0.0f, radius*0.5f, Vec3(0.0f, 0.0f, 0.0f), 0.125f, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f);
			// void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, 
			// 		float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
			CreateParticleShape(make_path(torus_path, "/data/torus.obj"), Vec3(lowerx + (i % 3) * torus_height / 3., 
				torus_height*(i+1), lowerz + (i % 3) * torus_height / 3.), size, 0.0f, radius*0.5f, Vec3(0.0f, 0.0f, 0.0f), 0.125f, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f);

		g_numSolidParticles = g_buffers->positions.size();	

		// float restDistance = radius*0.55f;
		float restDistance = radius * rest_dis_coef;

		// to make gif
		// g_capture = true;

		// void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius,
		// Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)
		// jitter controls the randomness in particle positions.
		// radius controls the particle radius / rest distance of the particle.
		// if radius / rest_radius is large, then the fluid is more smoothing, as particles interact with more neighbors.
	
		g_lightDistance *= 0.5f;

		g_sceneLower = Vec3(-2.0f, 0.0f, -1.0f);
		g_sceneUpper = Vec3(2.0f, 1.0f, 1.0f);

		g_numSubsteps = 2;

		g_params.radius = radius;
		g_params.numIterations = 4;
		g_params.staticFriction = static_friction;
		g_params.dynamicFriction = dynamic_friction;
		// g_params.fluidRestDistance = restDistance;
		// g_params.collisionDistance = restDistance;
		// g_params.shapeCollisionMargin = radius / 10.;

		g_maxDiffuseParticles = 0;
		g_diffuseScale = 0.5f;

		Emitter e1;
		e1.mDir = Vec3(1.0f, 0.0f, 0.0f);
		e1.mRight = Vec3(0.0f, 0.0f, -1.0f);
		e1.mPos = Vec3(radius, 1.f, 0.65f);
		e1.mSpeed = (restDistance/g_dt)*2.0f; // 2 particle layers per-frame
		e1.mEnabled = true;

		g_emitters.push_back(e1);

		// g_numExtraParticles = 48*1024;

		g_lightDistance = 1.8f;

		// g_params.numPlanes = 5;

		g_waveFloorTilt = 0.0f;
		g_waveFrequency = 1.5f;
		g_waveAmplitude = 2.0f;

		g_warmup = false;

		// std::cout << "render: " << render << endl;
		// draw options
		if (render == 0) {	// particle mode render
			g_drawPoints = true;
			g_drawMesh = false;
			g_drawEllipsoids = false;
			g_drawDiffuse = true;
		}

		else { //human mode render
			g_drawDensity = true;
			g_drawDiffuse = true;
			g_drawEllipsoids = true;
			g_drawPoints = false;
		}

	}

	virtual void CenterCamera(void)
	{
		g_camPos = Vec3(cam_x, cam_y, cam_z);
		g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
		g_screenHeight = cam_height;
		g_screenWidth = cam_width;
	}

	bool mDam;
};
