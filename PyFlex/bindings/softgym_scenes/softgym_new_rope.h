
class Softgym_NewRope : public Scene
{
public:

	Softgym_NewRope(const char* name) : Scene(name) {}

    float cam_x;
	float cam_y;
	float cam_z;
	float cam_angle_x;
	float cam_angle_y;
	float cam_angle_z;
	int cam_width;
	int cam_height;

    // init_x, init_y, init_z, stiffness, segments, length, invmass
	void Initialize(py::array_t<float> scene_params, int thread_idx=0)
	{

        auto ptr = (float *) scene_params.request().ptr;
        float init_x = ptr[0];
        float init_y = ptr[1];
        float init_z = ptr[2];
        float stretchstiffness = ptr[3];
        float bendingstiffness = ptr[4];
        float radius = ptr[5]; // used to determine the num of segments
        float segment = ptr[6];
        float mass = ptr[7];
        float scale = ptr[8];
		// printf("radius is: %f", radius);
		// printf("segment num is: %f", segment);

        cam_x = ptr[9];
		cam_y = ptr[10];
		cam_z = ptr[11];
		cam_angle_x = ptr[12];
		cam_angle_y = ptr[13];
		cam_angle_z = ptr[14];
		cam_width = int(ptr[15]);
		cam_height = int(ptr[16]);
		// int render = int(ptr[15]);


		int group = 0;
		float bendStiffness = 0.8f;
		float shearStiffness = 0.8f;
		// Yufei: I am not sure why, but the render gives an error if I do not add this cloth here.
		CreateSpringGrid(Vec3(-5, 0, 0.0f), 2, 2, 1, radius, NvFlexMakePhase(group++, eNvFlexPhaseSelfCollide), stretchstiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.1f);

        Rope r;

        Vec3 d0 = Vec3(1, 0, 0);
        Vec3 start = Vec3(init_x, init_y, init_z);
        // void CreateRope(Rope& rope, Vec3 start, Vec3 dir, float stiffness, int segments, float length, int phase, 
        //    float spiralAngle=0.0f, float invmass=1.0f, float give=0.075f)
        // CreateRope(r, attachPosition, Normalize(d0), 1.2f, int(Length(d0)/radius*1.1f), Length(d0), 
        //     NvFlexMakePhase(group++, 0), 0.0f, 0.5f, 0.0f);

		// printf("segment num is: %f", segment);
        CreateRope(r, start, d0, stretchstiffness, int(segment), segment * radius * 0.5, 
            NvFlexMakePhase(group++, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), 0.0f, 1 / mass, 0.075f, bendingstiffness);

        g_ropes.push_back(r);
		

	   	g_params.radius = radius;
		g_params.numIterations = 4;
		g_params.dynamicFriction = 1.0f;
		// g_params.staticFriction = 0.8f;
		g_params.collisionDistance = radius*0.5f;
		
		g_maxDiffuseParticles = 64*1024;
		g_diffuseScale = 0.25f;		
		g_diffuseShadow = false;
		g_diffuseColor = 2.5f;
		g_diffuseMotionScale = 1.5f;
		g_params.diffuseThreshold *= 0.01f;
		g_params.diffuseBallistic = 35;

		g_windStrength = 0.0f;
		g_windFrequency = 0.0f;

		g_numSubsteps = 2;

		// draw options		
		g_drawEllipsoids = false;
		g_drawPoints = false;
		g_drawDiffuse = false;
		g_drawSprings = 0;

		g_ropeScale = scale;
		g_warmup = false;
	}

    virtual void CenterCamera(void)
	{
		g_camPos = Vec3(cam_x, cam_y, cam_z);
		g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
		g_screenHeight = cam_height;
		g_screenWidth = cam_width;
	}
};



class ParachutingBunnies2 : public Scene
{
public:

	ParachutingBunnies2(const char* name) : Scene(name) {}

	void Initialize(py::array_t<float> scene_params, int thread_idx=0)
	{
		float stretchStiffness = 1.0f;
		float bendStiffness = 0.8f;
		float shearStiffness = 0.8f;

		int dimx = 32;
		int dimy = 32;
		float radius = 0.055f;

		float height = 10.0f;
		float spacing = 1.5f;
		int numBunnies = 2;
		int group = 0;

		for (int i=0; i < numBunnies; ++i)
		{
			CreateSpringGrid(Vec3(i*dimx*radius, height + i*spacing, 0.0f), dimx, dimy, 1, radius, NvFlexMakePhase(group++, eNvFlexPhaseSelfCollide), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.1f);

			const int startIndex = i*dimx*dimy;

			int corner0 = startIndex + 0;
			int corner1 = startIndex + dimx-1;
			int corner2 = startIndex + dimx*(dimy-1);
			int corner3 = startIndex + dimx*dimy-1;

			CreateSpring(corner0, corner1, 1.f,-0.1f);
			CreateSpring(corner1, corner3, 1.f,-0.1f);
			CreateSpring(corner3, corner2, 1.f,-0.1f);
			CreateSpring(corner0, corner2, 1.f,-0.1f);
		}

		for (int i=0; i < 1; ++i)
		{		
			Vec3 velocity = RandomUnitVector()*1.0f;
			float size = radius*8.5f;

			CreateParticleShape(GetFilePathByPlatform("../../data/bunny.ply").c_str(), Vec3(i*dimx*radius + radius*0.5f*dimx - 0.5f*size, height + i*spacing-0.5f, radius*0.5f*dimy - 0.5f), size, 0.0f, radius, velocity, 0.15f, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.0f);			

			const int startIndex = i*dimx*dimy;
			const int attachIndex = g_buffers->positions.size()-1;
			g_buffers->positions[attachIndex].w = 2.0f;

			int corner0 = startIndex + 0;
			int corner1 = startIndex + dimx-1;
			int corner2 = startIndex + dimx*(dimy-1);
			int corner3 = startIndex + dimx*dimy-1;

			Vec3 attachPosition = (Vec3(g_buffers->positions[corner0]) + Vec3(g_buffers->positions[corner1]) + Vec3(g_buffers->positions[corner2]) + Vec3(g_buffers->positions[corner3]))*0.25f;
			attachPosition.y = height + i*spacing-0.5f;

			if (1)
			{
				int c[4] = {corner0, corner1, corner2, corner3};

				// for (int i=0; i < 4; ++i)
				// {
					Rope r;

					// int start = g_buffers->positions.size();
	
					// r.mIndices.push_back(attachIndex);

					Vec3 d0 = Vec3(g_buffers->positions[c[i]])-attachPosition;
					printf("length of d0: ", Length(d0));
					d0 = Vec3(Length(d0), 0, 0);
					CreateRope(r, Vec3(0, 0, 0), Normalize(d0), 1.2f, int(Length(d0)/radius*1.1f), Length(d0), NvFlexMakePhase(group++, 0), 0.0f, 0.5f, 0.0f);

					// r.mIndices.push_back(c[i]);
					g_ropes.push_back(r);

					// int end = g_buffers->positions.size()-1;
					

					// CreateSpring(attachIndex, start, 1.2f, -0.5f);
					// CreateSpring(c[i], end, 1.0f);
				// }
			}
		}


		g_params.radius = 0.1f;
		g_params.fluidRestDistance = radius;
		g_params.numIterations = 4;
		g_params.viscosity = 1.0f;
		g_params.dynamicFriction = 0.05f;
		g_params.staticFriction = 0.0f;
		g_params.particleCollisionMargin = 0.0f;
		g_params.collisionDistance = g_params.fluidRestDistance*0.5f;
		g_params.vorticityConfinement = 120.0f;
		g_params.cohesion = 0.0025f;
		g_params.drag = 0.06f;
		g_params.lift = 0.f;
		g_params.solidPressure = 0.0f;
		g_params.smoothing = 1.0f;
		g_params.relaxationFactor = 1.0f;
		
		g_maxDiffuseParticles = 64*1024;
		g_diffuseScale = 0.25f;		
		g_diffuseShadow = false;
		g_diffuseColor = 2.5f;
		g_diffuseMotionScale = 1.5f;
		g_params.diffuseThreshold *= 0.01f;
		g_params.diffuseBallistic = 35;

		g_windStrength = 0.0f;
		g_windFrequency = 0.0f;

		g_numSubsteps = 2;

		// draw options		
		g_drawEllipsoids = true;
		g_drawPoints = false;
		g_drawDiffuse = true;
		g_drawSprings = 0;

		g_ropeScale = 0.2f;
		g_warmup = false;
	}
};
