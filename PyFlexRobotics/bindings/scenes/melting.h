
class Melting : public Scene
{
public:

	Melting()
	{
		g_params.radius = 0.1f;

		g_params.numIterations = 2;
		g_params.dynamicFriction = 0.25f;
		g_params.dissipation = 0.0f;
		g_params.viscosity = 0.0f;
		g_params.cohesion = 0.0f;
		g_params.fluidRestDistance = g_params.radius*0.6f;
		g_params.smoothing = 0.5f;

		const float spacing = g_params.radius*0.5f;

		Mesh* mesh = ImportMesh(GetFilePathByPlatform("../../data/bunny.ply").c_str());

		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);
		float size = 1.2f;

		for (int i = 0; i < 1; ++i)
			for (int j = 0; j < 3; ++j)
				CreateParticleShape(mesh, Vec3(-2.0f + j*size, 3.0f + j*size, i*size), size, 0.0f, spacing, Vec3(0.0f, 0.0f, 0.0f), 1.0f, true, 1.f, phase, false, 0.0f);

		delete mesh;

		// plinth
		AddBox(2.0f, Vec3(0.0f, 1.0f, 0.0f));

		g_numSubsteps = 2;

		// draw options		
		g_drawPoints = true;
		g_drawMesh = false;

		mFrame = 0;
	}

	virtual void Update()
	{
		const int start = 130;

		if (mFrame >= start)
		{
			float stiffness = max(0.0f, 1.0f - (mFrame - start) / 100.0f);

			for (int i = 0; i < g_buffers->shapeMatchingCoefficients.size(); ++i)
				g_buffers->shapeMatchingCoefficients[i] = stiffness;

			g_params.cohesion = Lerp(0.05f, 0.0f, stiffness);
		}

		++mFrame;
	}

	virtual void Sync()
	{
		NvFlexSetRigids(g_solver, g_buffers->shapeMatchingOffsets.buffer, g_buffers->shapeMatchingIndices.buffer, g_buffers->shapeMatchingLocalPositions.buffer, g_buffers->shapeMatchingLocalNormals.buffer, g_buffers->shapeMatchingCoefficients.buffer, g_buffers->shapeMatchingPlasticThresholds.buffer, g_buffers->shapeMatchingPlasticCreeps.buffer, g_buffers->shapeMatchingRotations.buffer, g_buffers->shapeMatchingTranslations.buffer, g_buffers->shapeMatchingOffsets.size() - 1, g_buffers->shapeMatchingIndices.size());
	}

	int mFrame;
};