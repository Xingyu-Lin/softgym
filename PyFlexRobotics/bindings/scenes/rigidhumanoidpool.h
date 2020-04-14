
class RigidHumanoidPool : public Scene
{
public:
	string humanoidLoadPath;

	float particleRadius, poolWidth, poolHeight, poolLength, poolEdge;
	Vec3 poolOrigin;
	
	vector<NvFlexRigidShape> poolWalls;	
	Vec3* points;

	float particleDensity;

	RigidHumanoidPool()
	{
		humanoidLoadPath = "../../data/humanoid_20_5.xml";
		particleRadius = 0.07f;
		poolOrigin = Vec3(-0.5f, 0.f, 5.f);
		poolWidth = 1.f;
		poolLength = 4.f;
		poolEdge = 0.05f;
		poolHeight = 1.7f;

		g_sceneUpper = Vec3(0.f, 4.f, 0.f);

		particleDensity = 1.25e3f; // Water density will e less due to not dense particle placement. Let's consider salt water.

		AddPoolWalls();
		AddPool();
		AddHumanoid();
		SetSimParams();
	}

	void AddHumanoid()
	{
		MJCFImporter mjcf = MJCFImporter(humanoidLoadPath.c_str());
		Transform gt = Transform(poolOrigin + Vec3(0.f, 4.f, -0.6f), Quat());
		vector<pair<int, NvFlexRigidJointAxis>> ctrl;
		vector<float> mpower;
		mjcf.AddPhysicsEntities(gt, ctrl, mpower);
	}

	void AddPoolWalls()
	{
		poolWalls.resize(0);
		float wallHeight = poolHeight / 2.5f;

		NvFlexRigidShape leftWall;
		NvFlexMakeRigidBoxShape(&leftWall, -1, poolEdge, poolHeight, poolLength + poolEdge * 2.f,
			NvFlexMakeRigidPose(poolOrigin + Vec3(-poolWidth - poolEdge, wallHeight, 0.f), Quat()));
		poolWalls.push_back(leftWall);

		NvFlexRigidShape rightWall;
		NvFlexMakeRigidBoxShape(&rightWall, -1, poolEdge, poolHeight, poolLength + poolEdge * 2.f,
			NvFlexMakeRigidPose(poolOrigin + Vec3(poolWidth + poolEdge, wallHeight, 0.f), Quat()));
		poolWalls.push_back(rightWall);

		NvFlexRigidShape frontWall;
		NvFlexMakeRigidBoxShape(&frontWall, -1, poolWidth + 2.f * poolEdge, poolHeight, poolEdge,
			NvFlexMakeRigidPose(poolOrigin + Vec3(0.f, wallHeight, - poolLength - poolEdge), Quat()));
		poolWalls.push_back(frontWall);

		NvFlexRigidShape backWall;
		NvFlexMakeRigidBoxShape(&backWall, -1, poolWidth + 2.f * poolEdge, poolHeight, poolEdge,
			NvFlexMakeRigidPose(poolOrigin + Vec3(0.f, wallHeight, poolLength + poolEdge), Quat()));
		poolWalls.push_back(backWall);

		for (int i = 0; (unsigned int)i < poolWalls.size(); i++)
		{
			poolWalls[i].filter = 0;
			poolWalls[i].material.friction = 0.7f;
			poolWalls[i].user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));
			g_buffers->rigidShapes.push_back(poolWalls[i]);
		}
	}

	void AddPool()
	{
		float fluidWidth = poolWidth;
		float fluidHeight = poolHeight;
		float fluidLength = poolLength;
		int particleWidth = int(2.f * fluidWidth / particleRadius);
		int particleHeight = int(fluidHeight / particleRadius);
		int particleLength = int(2.f * fluidLength / particleRadius);

		float mass = 4.f / 3.f * kPi * pow(particleRadius, 3.f) * particleDensity;
		CreateParticleGrid(poolOrigin - Vec3(fluidWidth, 0.f, fluidLength), particleWidth, particleHeight, particleLength,
			particleRadius, Vec3(0.f), 1.f / mass, false, 0.0f, // 1.0582 achieves about the same water : human density ratio w/ the humanoid
			NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid ));
	}

	void SetSimParams()
	{
		g_params.radius = 0.1f;
		g_params.fluidRestDistance = particleRadius;
		g_params.numIterations = 10;
		g_params.viscosity = .5f;
		g_params.dynamicFriction = 0.1f;
		g_params.staticFriction = 0.0f;
		g_params.particleCollisionMargin = 0.0f;
		g_params.collisionDistance = g_params.fluidRestDistance * 0.5f;
		g_params.vorticityConfinement = 120.0f;
		g_params.cohesion = 0.0025f;
		g_params.drag = 0.06f;
		g_params.lift = 0.f;
		g_params.solidPressure = 0.0f;
		g_params.smoothing = 1.0f;
		g_params.relaxationFactor = 1.0f;

		g_maxDiffuseParticles = 64 * 1024;
		g_diffuseScale = 0.25f;
		g_diffuseShadow = false;
		g_diffuseColor = 2.5f;
		g_diffuseMotionScale = 1.5f;
		g_params.diffuseThreshold *= 0.01f;
		g_params.diffuseBallistic = 35;
		g_numSubsteps = 2;

		// draw options		
		g_drawEllipsoids = true;
		g_drawPoints = false;
		g_drawDiffuse = true;
		g_ropeScale = 0.2f;
		g_warmup = false;
		g_pause = true;
	}
};
