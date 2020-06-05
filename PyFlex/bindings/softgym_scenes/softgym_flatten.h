class softgym_FlattenCloth: public Scene
{
public:

	softgym_FlattenCloth(const char* name) : Scene(name) {}

	void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	{

	    //for (int i=0; i < 5; i++)
			//AddRandomConvex(10, Vec3(i*2.0f, 0.0f, Randf(0.0f, 2.0f)), minSize, maxSize, Vec3(0.0f, 1.0f, 0.0f), Randf(0.0f, k2Pi));
			//AddBox(Vec3(2.0f, 0.5f, Randf(0.0f, 2.0f)),Vec3(i*2.0f, 0.0f, Randf(0.0f, 2.0f)));
        auto ptr = (float *) scene_params.request().ptr;
        float pickp = ptr[0];
        int pickpoint = (int) pickp;

		int dimx = (int) ptr[1]; //64;
		int dimz = (int) ptr[2]; //32;
		float radius = 0.05f;

		float stretchStiffness = 0.9f;
		float bendStiffness = 1.0f;
		float shearStiffness = 0.9f;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
		cout << "Phase: " << phase << endl;

		CreateSpringGrid(Vec3(0.0f, 2.0f, 2.0f), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f);


        g_buffers->positions[pickpoint].w = 0;
        for (int i=0; i < int(g_buffers->positions.size()); ++i)
		{
			// hack to rotate cloth

			g_buffers->velocities[i] = RandomUnitVector()*0.1f;

			float minSqrDist = FLT_MAX;

			if (i != pickpoint)
			{
				float stiffness = -0.8f;
				float give = 0.1f;

				//float sqrDist = LengthSq(Vec3(g_buffers->positions[c1])-Vec3(g_buffers->positions[c2]));


                //CreateSpring(pickpoint, i, stiffness, give);
                //CreateSpring(c2, i, stiffness, give);


			}
		}
		g_solverDesc.featureMode = eNvFlexFeatureModeSimpleSolids;

		g_params.radius = radius*1.5f;
		g_params.dynamicFriction = 0.5f;
		g_params.dissipation = 0.0f;
		g_params.numIterations = 4;
		g_params.drag = 0.06f;
		g_params.relaxationFactor = 1.0f;
		g_params.collisionDistance = 0.00125f;
		g_params.adhesion = 0.0f;
		g_params.shapeCollisionMargin = g_params.collisionDistance*0.25f;

        float viscosity = 10.0f;
	    float dissipation = 2.0f;

		//g_params.dynamicFriction = 1.0f;
		g_params.staticFriction = 1.0f;
		//_params.particleFriction = 2.0;
		g_params.viscosity = 20.0f + 20.0f*viscosity;
		//g_params.adhesion = 0.1f*viscosity;
		g_params.cohesion = 0.05f*viscosity;
		g_params.surfaceTension = 0.0f;

		g_numSubsteps = 2;

		// draw options
		g_drawPoints = true;
		g_drawSprings = false;
		g_drawMesh = false;
		g_windFrequency *= 2.0f;
		g_windStrength = 0;//10.0f;

    }

    void Update()
	{
		const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
		const float kNoise = fabsf(Perlin1D(g_windTime*0.05f, 2, 0.25f));
		Vec3 wind = g_windStrength*kWindDir*Vec3(kNoise, kNoise*0.1f, -kNoise*0.1f);
        cout<< "here" << endl;
		g_params.wind[0] = 0;//wind.x;
		g_params.wind[1] = 0;//wind.y;
		g_params.wind[2] = 0;//wind.z;
	}
};