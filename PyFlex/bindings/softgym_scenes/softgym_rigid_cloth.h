class SoftgymRigidCloth : public Scene
{
public:

	SoftgymRigidCloth(const char* name) : Scene(name) {}

    float radius = 0.025f;
    int group=0;

    void Connect(int idx1, int idx2)
    {
        // Create rope
		Vec3 attachPosition = Vec3(g_buffers->positions[idx1]);
		Rope r;
        int start = g_buffers->positions.size();

        r.mIndices.push_back(idx1);

        Vec3 d0 = Vec3(g_buffers->positions[idx2])-attachPosition;
        CreateRope(r, attachPosition , Normalize(d0), 1.2f, 3, Length(d0), NvFlexMakePhase(group, 0), 0.0f, 10.f, 0.f);

        r.mIndices.push_back(idx2);

        g_ropes.push_back(r);
        int end = g_buffers->positions.size()-1;


        CreateSpring(idx1, start, 1.f, -0.9);
        CreateSpring(end, idx2, 1.f, -0.9);
    }

	void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	{

		// Parse scene_params
		auto ptr = (float *) scene_params.request().ptr;
		int dimx = ptr[0], dimy = ptr[1], dimz = ptr[2]; // Scale
		float invMass = float(ptr[3]), rigidStiffness = float(ptr[4]);
		int drawpoints = int(ptr[5]), drawmesh=int(ptr[6]), ropescale = float(ptr[7]);

        float sx = dimx * radius, sy = dimy* radius, sz = dimz*radius;
		// Create plates
	    char box_path[100];
		strcpy(box_path, getenv("PYFLEXROOT"));
		strcat(box_path, "/data/box.ply");
        Mesh* mesh = ImportMesh(GetFilePathByPlatform(box_path).c_str());
        group=0;
		for (int i=0; i< 2; ++i)
		{
		    CreateParticleShape(mesh,
		    Vec3( (3*radius + sx) * i, radius, 0.0f), // lower
		    Vec3(sx, sy, sz), .0f, // scale and rotation
		    radius,  Vec3(0.0f, 0.0f, 0.0f), // spacing and velocity
		    invMass, true, rigidStiffness, NvFlexMakePhase(group++, 0), true, 0.0f); //invMass, rigid, rigidStiffness, phase, skin, jitter
		}

        int linkInterval = 4;
        for (int i=0; i< dimz; i+= linkInterval)
            Connect(dimz*(dimx-1) + i, dimz*dimx +i);
        if ((dimz-1)%linkInterval)
            Connect(dimz*dimx-1, dimz*(dimx+1) -1);

        // Without this, pressing M, i.e. do not rendering mesh will crash
//        CreateSpringGrid(Vec3(-1.f, -1.f, -1.f), 20, 20, 1, radius, NvFlexMakePhase(group++, eNvFlexPhaseSelfCollide), 1, 0.8, 0.8, Vec3(0.0f), 1.1f);



        g_params.radius = 0.1f; // Following the parachute example. Not sure why the params need to be set to be larger than the acutal radius
		g_params.fluidRestDistance = radius;
		g_params.numIterations = 4;
		g_params.viscosity = 0.0f;
		g_params.dynamicFriction = 0.5f;
		g_params.staticFriction = 2.f;
		g_params.particleCollisionMargin = 0.0f;
		g_params.collisionDistance = g_params.fluidRestDistance*0.5f;

		g_maxDiffuseParticles = 64*1024;
		g_diffuseScale = 0.25f;
		g_diffuseShadow = false;
		g_diffuseColor = 2.5f;
		g_diffuseMotionScale = 1.5f;
		g_params.diffuseThreshold *= 0.01f;
		g_params.diffuseBallistic = 35;

		g_numSubsteps = 5;

		// draw options
		g_drawEllipsoids = false;
		g_drawPoints = false;
		g_drawDiffuse = false;
		g_drawSprings = 0;

		g_ropeScale = 0.2f;
		g_warmup = false;
//		g_pause=true;
	}
};



