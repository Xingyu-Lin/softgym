
class SoftgymFluid : public Scene
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

	SoftgymFluid(const char* name) : Scene(name) {}

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
		float cohesion = ptr[2];
		float viscosity = ptr[3];
		float surfaceTension = ptr[4];
		float adhesion = ptr[5];
		float vorticityConfinement = ptr[6];
		float solidpressure = ptr[7];

	    float x = ptr[8];
	    float y = ptr[9];
	    float z = ptr[10];
	    float dim_x = ptr[11];
	    float dim_y = ptr[12];
	    float dim_z = ptr[13];
		cam_x = ptr[14];
		cam_y = ptr[15];
		cam_z = ptr[16];
		cam_angle_x = ptr[17];
		cam_angle_y = ptr[18];
		cam_angle_z = ptr[19];
		cam_width = int(ptr[20]);
		cam_height = int(ptr[21]);
		int render = int(ptr[22]);


		/*
		The main particle radius is set via NvFlexParams::radius, which is the “interaction radius”.
		Particles closer than this distance will be able to affect each other.
		*/
		// float radius = 0.1f;

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
		CreateParticleGrid(Vec3(x, y, z), dim_x, dim_y, dim_z, restDistance,
			Vec3(0.0f), 1.0f, false, 0.0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid), 0.005f);

		g_lightDistance *= 0.5f;

		g_sceneLower = Vec3(-2.0f, 0.0f, -1.0f);
		g_sceneUpper = Vec3(2.0f, 1.0f, 1.0f);

		g_numSubsteps = 2;

		g_params.radius = radius;
		g_params.dynamicFriction = 0.01f;
		g_params.viscosity =  viscosity; //2.0f;
		g_params.numIterations = 4;
		g_params.vorticityConfinement = vorticityConfinement;// 40.0f;
		g_params.fluidRestDistance = restDistance;
		g_params.solidPressure = solidpressure; //0.f;
		g_params.relaxationFactor = 0.0f;
		g_params.collisionDistance = 0.0033f;
		g_params.cohesion = cohesion; //0.01f*viscosity;
		

		Vec4 nomral_water_color = Vec4(0.113f, 0.425f, 0.55f, 1.f);

		g_fluidColor = nomral_water_color;

		g_maxDiffuseParticles = 0;
		g_diffuseScale = 0.5f;

		Emitter e1;
		e1.mDir = Vec3(1.0f, 0.0f, 0.0f);
		e1.mRight = Vec3(0.0f, 0.0f, -1.0f);
		e1.mPos = Vec3(radius, 1.f, 0.65f);
		e1.mSpeed = (restDistance/g_dt)*2.0f; // 2 particle layers per-frame
		e1.mEnabled = true;

		g_emitters.push_back(e1);

		g_lightDistance = 1.8f;


		g_waveFloorTilt = 0.0f;
		g_waveFrequency = 1.5f;
		g_waveAmplitude = 2.0f;

		g_warmup = false;

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
