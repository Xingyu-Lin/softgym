
class yz_RiceFall: public Scene
{
public:

	yz_RiceFall(const char* name) : Scene(name) {}

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
	    float viscosity = 10.0f;
	    float dissipation = 2.0f;

		float radius = 0.1f;
		float restDistance = radius * 0.5f;

        srand(time(NULL) + thread_idx);
        float x_0 = rand_float(0.15, 0.25);
        float x_1 = rand_float(0.15, 0.25);
        float y_0 = rand_float(0.1, 0.25);
        float y_1 = rand_float(0.55, 0.7);
        float z_0 = rand_float(0.05, 0.15);
        float z_1 = rand_float(0.05, 0.15);

        if (rand_float(-1, 1) > 0) {
            swap(&y_0, &y_1);
        }

		g_solverDesc.featureMode = eNvFlexFeatureModeSimpleFluids;
		g_params.radius = radius;

		g_params.numIterations = 3;
		g_params.vorticityConfinement = 0.0f;
		g_params.fluidRestDistance = restDistance;
		g_params.smoothing = 0.35f;
		g_params.relaxationFactor = 1.f;
		g_params.restitution = 0.0f;
		g_params.collisionDistance = 0.00125f;
		g_params.shapeCollisionMargin = g_params.collisionDistance*0.25f;
		g_params.dissipation = dissipation;

		g_params.gravity[1] *= 4.0f;

		g_fluidColor = Vec4(1.0f, 1.0f, 1.0f, 0.0f);
		g_meshColor = Vec3(0.7f, 0.8f, 0.9f)*0.7f;

		g_params.dynamicFriction = 1.0f;
		g_params.staticFriction = 0.0f;
		g_params.viscosity = 20.0f + 20.0f*viscosity;
		g_params.adhesion = 0.1f*viscosity;
		g_params.cohesion = 0.05f*viscosity;
		g_params.surfaceTension = 0.0f;


        // void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)
		CreateParticleGrid(Vec3(x_0, y_0, z_0), 4, 4, 4, restDistance, Vec3(0.0f), 1.0f, false, 0.0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide), 0.0f);
		CreateParticleGrid(Vec3(x_1, y_1, z_1), 5, 5, 5, restDistance, Vec3(0.0f), 1.0f, false, 0.0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide), 0.0f);
		g_lightDistance *= 0.5f;

		g_sceneLower = Vec3(-0.0f, 0.0f, 0.0f);
		g_sceneUpper = Vec3(0.6f, 0.0f, 0.4f);

		g_numSubsteps = 2;

		// draw options
		g_drawPoints = true;
		g_drawEllipsoids = false;
		g_drawDiffuse = true;
	}
};
