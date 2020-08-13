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
        group=0;
		for (int i=0; i< numPiece; ++i)
		{
		    CreateParticleShape(mesh,
		    Vec3( (3*radius + sx) * i, radius, 0.0f), // lower
		    Vec3(sx, sy, sz), .0f, // scale and rotation
		    radius,  Vec3(0.0f, 0.0f, 0.0f), // spacing and velocity
		    invMass, true, rigidStiffness, NvFlexMakePhase(group++, 0), true, 0.0f); //invMass, rigid, rigidStiffness, phase, skin, jitter
		}
//		cout<<"here"<<g_buffers->triangles.size()<<end;
//		cout<<g_mesh->m_colours.size()<<end;
//		g_mesh->m_colours[i] = 1.25f*colors[((unsigned int)(phase))%7];

        if (numPiece ==1)
        {
            const Colour colors[7] =
            {
                Colour(0.0f, 0.5f, 1.0f),
                Colour(0.797f, 0.354f, 0.000f),
                Colour(0.000f, 0.349f, 0.173f),
                Colour(0.875f, 0.782f, 0.051f),
                Colour(0.01f, 0.170f, 0.453f),
                Colour(0.673f, 0.111f, 0.000f),
                Colour(0.612f, 0.194f, 0.394f)
            };

            for (int i=0; i<int(g_mesh->GetNumVertices()*0.6); ++i)
                g_mesh->m_colours[i] = 1.5f*colors[5];
            for (int i=int(g_mesh->GetNumVertices()*0.6); i<g_mesh->GetNumVertices(); ++i)
                g_mesh->m_colours[i] = 1.5f*colors[6];
        }

        g_params.radius = radius ;
		g_params.fluidRestDistance = radius;
		g_params.numIterations = 15;
		g_params.dynamicFriction = 0.5f;
		g_params.staticFriction = 2.f;
		g_params.dissipation = 0.01f;
		g_params.particleCollisionMargin = g_params.radius*0.05f;
		g_params.particleFriction = 100000.;


		g_numSubsteps = 5;

		// draw options
		g_drawEllipsoids = false;
		g_drawPoints = false;
		g_drawDiffuse = false;
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



