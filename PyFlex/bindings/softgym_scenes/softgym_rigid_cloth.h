class SoftgymRigidCloth : public Scene
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

	SoftgymRigidCloth(const char* name) : Scene(name) {}

    float radius = 0.02f;
    int group=0;

    void CreateMeshTriangle(int x, int y, int z)
    {
        g_buffers->triangles.push_back(x);
        g_buffers->triangles.push_back(y);
        g_buffers->triangles.push_back(z);
        g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
    }

	void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	{

		// Parse scene_params
		auto ptr = (float *) scene_params.request().ptr;
		int dimx = ptr[0], dimy = ptr[1], dimz = ptr[2]; // Dimension
		int numPiece = ptr[3];
		float invMass = float(ptr[4]), rigidStiffness = float(ptr[5]);

		cam_x = ptr[6];
        cam_y = ptr[7];
        cam_z = ptr[8];
        cam_angle_x = ptr[9];
        cam_angle_y = ptr[10];
        cam_angle_z = ptr[11];
        cam_width = int(ptr[12]);
        cam_height = int(ptr[13]);


        float sx = dimx * radius, sy = dimy* radius, sz = dimz*radius;

		// Create plates
	    char box_path[100];
		strcpy(box_path, getenv("PYFLEXROOT"));
		strcat(box_path, "/data/box.ply");
        Mesh* mesh = ImportMesh(GetFilePathByPlatform(box_path).c_str());
        group=2;
        int cloth_dimx = 2;

		for (int i=0; i< numPiece; ++i)
		{
		    CreateParticleShape(mesh,
		    Vec3( ((cloth_dimx-0.5f)*radius + sx) * i, radius, 0.0f), // lower
		    Vec3(sx, sy, sz), .0f, // scale and rotation
		    radius,  Vec3(0.0f, 0.0f, 0.0f), // spacing and velocity
		    invMass, true, rigidStiffness, NvFlexMakePhase(group, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), true, 0.0f); //invMass, rigid, rigidStiffness, phase, skin, jitter
		}

        float stretchStiffness = 0.9f;
		float bendStiffness = 1.0f;
		float shearStiffness = 0.9f;
		int phase = NvFlexMakePhase(group, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
		float mass = float(100.)/(cloth_dimx *dimz);	// avg bath towel is 500-700g



	    // Create cloth connection
	    for (int i=0; i< numPiece-1; ++i)
		{
		    int start_idx_left = i * dimx * dimz + dimx * dimz - dimz;
		    int start_idx_right = (i+1) * dimx * dimz;
		    int cloth_size = cloth_dimx + dimz;
            int cloth_start_idx = g_buffers->positions.size();

	        CreateSpringGrid(Vec3(sx, radius, -radius/2.), cloth_dimx, dimz, 1, radius, phase,
	                        stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f/mass);
	        for (int j=0; j<dimz; ++j)
	        {
	            CreateSpring(start_idx_left +j, cloth_start_idx + j * cloth_dimx , stretchStiffness, -0.95);
	            CreateSpring(start_idx_right +j, cloth_start_idx + j * cloth_dimx + cloth_dimx-1, stretchStiffness, -0.95);
                // Create cross springs
                if (j<dimz-1)
                {
                    CreateSpring(start_idx_left +j+1, cloth_start_idx + j * cloth_dimx , stretchStiffness, -0.95);
                    CreateSpring(start_idx_left +j, cloth_start_idx + (j+1) * cloth_dimx , stretchStiffness, -0.95);

                    CreateSpring(start_idx_right +j +1, cloth_start_idx + j * cloth_dimx + cloth_dimx-1, stretchStiffness, -0.95);
                    CreateSpring(start_idx_right +j, cloth_start_idx + (j+1) * cloth_dimx + cloth_dimx-1, stretchStiffness, -0.95);
                }
                // Create additional mesh
                if (j<dimz-1)
                {
                    int x = start_idx_left +j;
                    int y = cloth_start_idx + j * cloth_dimx;
	                CreateMeshTriangle(x, y+cloth_dimx, x+1);
	                CreateMeshTriangle(x, y, y+cloth_dimx);
	                x = start_idx_right + j;
                    y = cloth_start_idx + j * cloth_dimx + cloth_dimx-1;
                    CreateMeshTriangle(y, x+1, y + cloth_dimx);
                    CreateMeshTriangle(y, x, x+1);
	            }
            }
		}


        g_params.radius = radius ;
//		g_params.fluidRestDistance = radius;
		g_params.numIterations = 8;
		g_params.dynamicFriction = 0.5f;
		g_params.staticFriction = 0.5f;
		g_params.dissipation = 0.01f;
		g_params.particleCollisionMargin = g_params.radius*0.05f;

		g_numSubsteps = 5;

        // CLoth env
        g_params.dissipation = 0.0f;


		// draw options
		g_drawEllipsoids = false;
		g_drawPoints = false;
		g_drawDiffuse = false;
		g_drawCloth = true;
		g_drawSprings = 0;
		g_warmup = false;
//		g_pause=true;
	}

	virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        if (cam_height & cam_width){
            g_screenHeight = cam_height;
            g_screenWidth = cam_width;
        }
    }
};



