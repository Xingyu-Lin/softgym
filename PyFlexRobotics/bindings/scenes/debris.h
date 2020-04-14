
class RigidDebris : public Scene
{
public:

	struct Instance
	{
		Vec3 mTranslation;
		Quat mRotation;
		float mLifetime;

		int mGroup;
		int mParticleOffset;

		int mMeshIndex;
	};

	struct MeshBatch
	{
		RenderMesh* mMesh;
		NvFlexExtAsset* mAsset;

		std::vector<Matrix44> mInstanceTransforms;
	};

	struct MeshAsset
	{
		const char* file;
		float scale;
	};

	RigidDebris()
	{
		float radius = 0.1f;

		const int numMeshes = 8;
		MeshAsset meshes[numMeshes] =
		{
			{ "../../data/rocka.ply", 0.2f },
			{ "../../data/box.ply", 0.1f },
			{ "../../data/torus.obj", 0.3f },
			{ "../../data/rockd.ply", 0.2f },
			{ "../../data/banana.obj", 0.3f },
			{ "../../data/rocka.ply", 0.2f },
			{ "../../data/box.ply", 0.1f },
			{ "../../data/rockd.ply", 0.2f },
			//"../../data/rockf.ply"
		};

		for (int i = 0; i < numMeshes; ++i)
		{
			Mesh* mesh = ImportMesh(GetFilePathByPlatform(meshes[i].file).c_str());
			mesh->Normalize(meshes[i].scale);

			const float spacing = radius*0.5f;

			MeshBatch b;
			b.mAsset = NvFlexExtCreateRigidFromMesh((float*)&mesh->m_positions[0], int(mesh->m_positions.size()), (int*)&mesh->m_indices[0], mesh->m_indices.size(), spacing, -spacing*0.5f);
			b.mMesh = CreateRenderMesh(mesh);

			mBatches.push_back(b);
		}

		Mesh* level = ImportMeshFromBin(GetFilePathByPlatform("../../data/testzone.bin").c_str());
		level->Transform(TranslationMatrix(Point3(-10.0f, 0.0f, 10.0f)));

		NvFlexTriangleMeshId mesh = CreateTriangleMesh(level);
		AddTriangleMesh(mesh, Vec3(), Quat(), 1.0f);

		delete level;

		g_params.radius = radius;
		g_params.dynamicFriction = 0.6f;
		g_params.staticFriction = 0.35f;
		g_params.particleFriction = 0.25f;
		g_params.dissipation = 0.0f;
		g_params.numIterations = 4;
		g_params.viscosity = 0.0f;
		g_params.drag = 0.0f;
		g_params.lift = 0.0f;
		g_params.numPlanes = 0;
		g_params.collisionDistance = radius*0.5f;
		g_params.particleCollisionMargin = radius*0.25f;

		g_numExtraParticles = 32000;

		g_drawPoints = false;

		g_numSubsteps = 2;

		g_lightDistance *= 3.0f;

		mAttractForce = 0.0f;

		mGroupCounter = 0;
		mInstances.resize(0);
	}

	void Update()
	{
		// copy transforms out
		for (int i = 0; i < int(mInstances.size()); ++i)
		{
			mInstances[i].mTranslation = g_buffers->shapeMatchingTranslations[i];
			mInstances[i].mRotation = g_buffers->shapeMatchingRotations[i];
		}

		if (g_emit)
		{
			// emit new debris
			int numToEmit = 1;//Rand()%8;

			int particleOffset = NvFlexGetActiveCount(g_solver);

			for (int i = 0; i < numToEmit; ++i)
			{
				// choose a random mesh to emit
				const int meshIndex = Rand() % mBatches.size();

				NvFlexExtAsset* asset = mBatches[meshIndex].mAsset;

				// check we can fit in the container
				if (int(g_buffers->positions.size()) - particleOffset < asset->numParticles)
					break;

				Instance inst;
				inst.mLifetime = 1000.0f;// Randf(5.0f, 60.0f);
				inst.mParticleOffset = particleOffset;
				inst.mRotation = QuatFromAxisAngle(UniformSampleSphere(), Randf()*k2Pi);

				float spread = 0.2f;
				inst.mTranslation = g_emitters[0].mPos + Vec3(Randf(-spread, spread), Randf(-spread, spread), 0.0f);
				inst.mMeshIndex = meshIndex;

				Vec3 linearVelocity = g_emitters[0].mDir*15.0f;//*Randf(5.0f, 10.0f);//Vec3(Randf(0.0f, 10.0f), 0.0f, 0.0f);
				Vec3 angularVelocity = Vec3(UniformSampleSphere()*Randf()*k2Pi);

				inst.mGroup = mGroupCounter++;

				const int phase = NvFlexMakePhase(inst.mGroup, 0);

				// generate initial particle positions
				for (int j = 0; j < asset->numParticles; ++j)
				{
					Vec3 localPos = Vec3(&asset->particles[j * 4]) - Vec3(&asset->shapeCenters[0]);

					g_buffers->positions[inst.mParticleOffset + j] = Vec4(inst.mTranslation + inst.mRotation*localPos, 1.0f);
					g_buffers->velocities[inst.mParticleOffset + j] = linearVelocity + Cross(angularVelocity, localPos);
					g_buffers->phases[inst.mParticleOffset + j] = phase;
				}

				particleOffset += asset->numParticles;

				mInstances.push_back(inst);
			}
		}

		// destroy old debris pieces
		for (int i = 0; i < int(mInstances.size());)
		{
			Instance& inst = mInstances[i];

			inst.mLifetime -= g_dt;

			if (inst.mLifetime <= 0.0f)
			{
				inst = mInstances.back();
				mInstances.pop_back();
			}
			else
			{
				++i;
			}
		}

		// compact instances 
		static std::vector<Vec4> particles(g_buffers->positions.size());
		static std::vector<Vec3> velocities(g_buffers->velocities.size());
		static std::vector<int> phases(g_buffers->phases.size());

		g_buffers->shapeMatchingTranslations.resize(0);
		g_buffers->shapeMatchingRotations.resize(0);
		g_buffers->shapeMatchingCoefficients.resize(0);
		g_buffers->shapeMatchingIndices.resize(0);
		g_buffers->shapeMatchingLocalPositions.resize(0);
		g_buffers->shapeMatchingOffsets.resize(0);

		// start index
		g_buffers->shapeMatchingOffsets.push_back(0);

		// clear mesh batches
		for (int i = 0; i < int(mBatches.size()); ++i)
			mBatches[i].mInstanceTransforms.resize(0);

		numActive = 0;

		for (int i = 0; i < int(mInstances.size()); ++i)
		{
			Instance& inst = mInstances[i];

			NvFlexExtAsset* asset = mBatches[inst.mMeshIndex].mAsset;

			for (int j = 0; j < asset->numParticles; ++j)
			{
				particles[numActive + j] = g_buffers->positions[inst.mParticleOffset + j];
				velocities[numActive + j] = g_buffers->velocities[inst.mParticleOffset + j];
				phases[numActive + j] = g_buffers->phases[inst.mParticleOffset + j];
			}

			g_buffers->shapeMatchingCoefficients.push_back(1.0f);
			g_buffers->shapeMatchingTranslations.push_back(inst.mTranslation);
			g_buffers->shapeMatchingRotations.push_back(inst.mRotation);

			for (int j = 0; j < asset->numShapeIndices; ++j)
			{
				g_buffers->shapeMatchingLocalPositions.push_back(Vec3(&asset->particles[j * 4]) - Vec3(&asset->shapeCenters[0]));
				g_buffers->shapeMatchingIndices.push_back(asset->shapeIndices[j] + numActive);
			}

			g_buffers->shapeMatchingOffsets.push_back(g_buffers->shapeMatchingIndices.size());

			mInstances[i].mParticleOffset = numActive;

			// Draw transform
			Matrix44 xform = TranslationMatrix(Point3(inst.mTranslation - inst.mRotation*Vec3(asset->shapeCenters)))*RotationMatrix(inst.mRotation);
			mBatches[inst.mMeshIndex].mInstanceTransforms.push_back(xform);

			numActive += asset->numParticles;
		}

		// update particle buffers
		g_buffers->positions.assign(&particles[0], particles.size());
		g_buffers->velocities.assign(&velocities[0], velocities.size());
		g_buffers->phases.assign(&phases[0], phases.size());

		// rebuild active indices
		g_buffers->activeIndices.resize(numActive);
		for (int i = 0; i < numActive; ++i)
			g_buffers->activeIndices[i] = i;

		if (mAttractForce != 0.0f)
		{
			const Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));

			Vec3 attractPos = g_camPos + forward*5.0f;
			float invRadius = 1.0f / 5.0f;

			for (int i = 0; i < int(g_buffers->velocities.size()); ++i)
			{
				Vec3 dir = Vec3(g_buffers->positions[i]) - attractPos;
				float d = Length(dir);

				g_buffers->velocities[i] += Normalize(dir)*Randf(0.0, 1.0f)*mAttractForce*Max(0.0f, 1.0f - d*invRadius);
			}
		}
	}

	virtual void PostInitialize()
	{
		g_sceneLower = Vec3(-5.0f, 0.0f, 0.0f);
		g_sceneUpper = g_sceneLower + Vec3(10.0f, 10.0f, 5.0f);
	}

	virtual void Sync()
	{
		NvFlexSetRigids(g_solver, g_buffers->shapeMatchingOffsets.buffer, g_buffers->shapeMatchingIndices.buffer, g_buffers->shapeMatchingLocalPositions.buffer, g_buffers->shapeMatchingLocalNormals.buffer, g_buffers->shapeMatchingCoefficients.buffer, g_buffers->shapeMatchingPlasticThresholds.buffer, g_buffers->shapeMatchingPlasticCreeps.buffer,   g_buffers->shapeMatchingRotations.buffer, g_buffers->shapeMatchingTranslations.buffer, g_buffers->shapeMatchingOffsets.size() - 1, g_buffers->shapeMatchingIndices.size());
	}

	virtual void KeyDown(int key)
	{
		if (key == 'B')
		{
			float bombStrength = 10.0f;

			Vec3 bombPos = g_emitters[0].mPos + g_emitters[0].mDir*5.0f;
			bombPos.y -= 5.0f;

			for (int i = 0; i < int(g_buffers->velocities.size()); ++i)
			{
				Vec3 dir = Vec3(g_buffers->positions[i]) - bombPos;

				g_buffers->velocities[i] += Normalize(dir)*bombStrength*Randf(0.0, 1.0f);
			}
		}

		if (key == 'V')
		{
			if (mAttractForce == 0.0f)
				mAttractForce = -1.5f;
			else
				mAttractForce = 0.0f;
		}
	}

	void Draw(int pass)
	{
		if (!g_drawMesh)
			return;

		RenderMaterial materials[8];
		memset(materials, 0, sizeof(materials));

		for (int i=0; i < 8; ++i)
			materials[i].frontColor = Vec3(g_colors[i]);	

		for (int b = 0; b < int(mBatches.size()); ++b)
		{
			if (mBatches[b].mInstanceTransforms.size())
			{
				DrawRenderMeshInstances(mBatches[b].mMesh, &mBatches[b].mInstanceTransforms[0], mBatches[b].mInstanceTransforms.size(), materials[b%8]);
			}
		}
	}

	float mAttractForce;

	std::vector<MeshBatch> mBatches;
	int numActive;

	int mGroupCounter;
	std::vector<Instance> mInstances;
};
