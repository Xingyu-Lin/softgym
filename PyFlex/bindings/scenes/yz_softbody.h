

class yz_SoftBody : public Scene
{

public:
	yz_SoftBody(const char* name) :
		Scene(name),
		mRadius(0.1f),
		mRelaxationFactor(1.0f),
		mPlinth(false),
		plasticDeformation(false)
	{
		const Vec3 colorPicker[7] =
		{
			Vec3(0.0f, 0.5f, 1.0f),
			Vec3(0.797f, 0.354f, 0.000f),
			Vec3(0.000f, 0.349f, 0.173f),
			Vec3(0.875f, 0.782f, 0.051f),
			Vec3(0.01f, 0.170f, 0.453f),
			Vec3(0.673f, 0.111f, 0.000f),
			Vec3(0.612f, 0.194f, 0.394f)
		};
		memcpy(mColorPicker, colorPicker, sizeof(Vec3) * 7);
	}

	float mRadius;
	float mRelaxationFactor;
	bool mPlinth;
	float mTime;

	Vec3 mColorPicker[7];

	struct Instance
	{
		Instance(const char* mesh) :

			mFile(mesh),
			mColor(0.5f, 0.5f, 1.0f),

			mScale(2.0f),
			mTranslation(0.0f, 1.0f, 0.0f),

			mClusterSpacing(1.0f),
			mClusterRadius(0.0f),
			mClusterStiffness(0.5f),

			mLinkRadius(0.0f),
			mLinkStiffness(1.0f),

			mGlobalStiffness(0.0f),

			mSurfaceSampling(0.0f),
			mVolumeSampling(4.0f),

			mSkinningFalloff(2.0f),
			mSkinningMaxDistance(100.0f),

			mClusterPlasticThreshold(0.0f),
			mClusterPlasticCreep(0.0f)
		{}

		const char* mFile;
		Vec3 mColor;

		Vec3 mScale;
		Vec3 mTranslation;

		float mClusterSpacing;
		float mClusterRadius;
		float mClusterStiffness;

		float mLinkRadius;
		float mLinkStiffness;

		float mGlobalStiffness;

		float mSurfaceSampling;
		float mVolumeSampling;

		float mSkinningFalloff;
		float mSkinningMaxDistance;

		float mClusterPlasticThreshold;
		float mClusterPlasticCreep;
	};

	/*
	void Update()
	{
		ClearShapes();

		// let cloth settle on object
		float startTime = 0.0f;

		float time = Max(0.0f, mTime-startTime);
		float lastTime = Max(0.0f, time-g_dt);

		const float translationSpeed = 0.45f;

		Vec3 pos_0 = Vec3(translationSpeed*(1.0f-cosf(time)) - 1.0f, 0.6f, gripper.z);
		Vec3 prevPos_0 = Vec3(translationSpeed*(1.0f-cosf(lastTime)) - 1.0f, 0.6f, gripper.z);

		Vec3 pos_1 = Vec3(translationSpeed*(cosf(time)) + 1.0f, 0.6f, gripper.z);
		Vec3 prevPos_1 = Vec3(translationSpeed*(cosf(lastTime)) + 1.0f, 0.6f, gripper.z);

		Quat rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi / 2.0f);
		Quat prevRot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi / 2.0f);

		AddBox(Vec3(0.8f, 0.1f, 0.2f), pos_0, rot);
		AddBox(Vec3(0.8f, 0.1f, 0.2f), pos_1, rot);
        // AddCapsule(0.2f, 1.0f, pos_0, rot);
        // AddCapsule(0.2f, 1.0f, pos_1, rot);

		g_buffers->shapePrevPositions[0] = Vec4(prevPos_0, 0.0f);
		g_buffers->shapePrevRotations[0] = prevRot;

		g_buffers->shapePrevPositions[1] = Vec4(prevPos_1, 0.0f);
		g_buffers->shapePrevRotations[1] = prevRot;

		UpdateShapes();

		mTime += g_dt;
	}
	*/

	std::vector<Instance> mInstances;

private:

	struct RenderingInstance
	{
		Mesh* mMesh;
		std::vector<int> mSkinningIndices;
		std::vector<float> mSkinningWeights;
		vector<Vec3> mRigidRestPoses;
		Vec3 mColor;
		int mOffset;
	};

	std::vector<RenderingInstance> mRenderingInstances;

	bool plasticDeformation;


public:
	virtual void AddInstance(Instance instance)
	{
		this->mInstances.push_back(instance);
	}

	virtual void AddStack(Instance instance, int xStack, int yStack, int zStack, bool rotateColors = false)
	{
		Vec3 translation = instance.mTranslation;
		for (int x = 0; x < xStack; ++x)
		{
			for (int y = 0; y < yStack; ++y)
			{
				for (int z = 0; z < zStack; ++z)
				{
					instance.mTranslation = translation + Vec3(x*(instance.mScale.x + 1), y*(instance.mScale.y + 1), z*(instance.mScale.z + 1))*mRadius;
					if (rotateColors) {
						instance.mColor = mColorPicker[(x*yStack*zStack + y*zStack + z) % 7];
					}
					this->mInstances.push_back(instance);
				}
			}
		}
	}

	char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}

	float rand_float(float LO, float HI) {
        return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(HI-LO)));
    }

	virtual void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	{
	    // scene_params:
	    // x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep

	    srand(time(NULL) + thread_idx);
	    this->mInstances.clear();

	    // Add instance
	    char bunny_path[100];
		char box_path[100];
		char sphere_path[100];
		make_path(bunny_path, "/data/bunny.ply");
		make_path(box_path, "/data/box.ply");
		make_path(sphere_path, "/data/sphere.ply");

        Instance stackBox(box_path);

        /*
        float x = rand_float(8.0f, 10.0f);
        float y = rand_float(8.0f, 10.0f);
        float z = rand_float(8.0f, 10.0f);
        float clusterStiffness = rand_float(0.4f, 0.8f);
        float clusterPlasticThreshold = rand_float(0.00001f, 0.00003f);
        float clusterPlasticCreep = rand_float(0.1f, 0.3f);
         */

        auto ptr = (float *) scene_params.request().ptr;
        float x = ptr[0];
        float y = ptr[1];
        float z = ptr[2];
        float clusterStiffness = ptr[3];
        float clusterPlasticThreshold = ptr[4];
        float clusterPlasticCreep = ptr[5];

        stackBox.mScale = Vec3(x, y, z);
        stackBox.mClusterSpacing = 10.0f;
        stackBox.mClusterRadius = 0.0f;
        stackBox.mClusterStiffness = clusterStiffness;
        stackBox.mGlobalStiffness = 0.0f;
        stackBox.mClusterPlasticThreshold = clusterPlasticThreshold;
        stackBox.mClusterPlasticCreep = clusterPlasticCreep;
        stackBox.mTranslation.x = -x/20.0f;
        stackBox.mTranslation.y = 0.0f;
        stackBox.mTranslation.z = -z/20.0f;
        AddInstance(stackBox);


		float radius = mRadius;

		g_sceneLower = Vec3(-0.5f, 0.0f, 0.0f);
		g_sceneUpper = Vec3(1.0f, 2.0f, 0.6f);

		// no fluids or sdf based collision
		// g_solverDesc.featureMode = eNvFlexFeatureModeSimpleSolids;

		g_params.radius = radius;
		g_params.dynamicFriction = 2.0f;
		g_params.particleFriction = 0.25f;
		g_params.numIterations = 4;
		g_params.collisionDistance = radius*0.75f;

		g_params.relaxationFactor = mRelaxationFactor;

		g_windStrength = 0.0f;

		g_numSubsteps = 2;

		// draw options
		g_drawPoints = true;
		g_drawMesh = false;
		g_drawDiffuse = true;

		g_wireframe = false;
		g_drawSprings = false;
		g_drawBases = false;

		g_buffers->rigidOffsets.push_back(0);

		mRenderingInstances.resize(0);

		// build soft bodies 
		for (int i = 0; i < int(mInstances.size()); i++)
			CreateSoftBody(mInstances[i], mRenderingInstances.size());

		if (mPlinth) 
			AddPlinth();

		// fix any particles below the ground plane in place
		for (int i = 0; i < int(g_buffers->positions.size()); ++i)
			if (g_buffers->positions[i].y < 0.0f)
				g_buffers->positions[i].w = 0.0f;

		// expand radius for better self collision
		g_params.radius *= 1.5f;

		g_lightDistance *= 1.5f;

		mTime = 0.0f;
    }

    void CreateSoftBody(Instance instance, int group = 0)
    {
        RenderingInstance renderingInstance;

        Mesh* mesh = ImportMesh(GetFilePathByPlatform(instance.mFile).c_str());
        mesh->Normalize();
        mesh->Transform(TranslationMatrix(Point3(instance.mTranslation))*ScaleMatrix(instance.mScale*mRadius));

        renderingInstance.mMesh = mesh;
        renderingInstance.mColor = instance.mColor;
        renderingInstance.mOffset = g_buffers->rigidTranslations.size();

        double createStart = GetSeconds();

        // create soft body definition
        NvFlexExtAsset* asset = NvFlexExtCreateSoftFromMesh(
                (float*)&renderingInstance.mMesh->m_positions[0],
                renderingInstance.mMesh->m_positions.size(),
                (int*)&renderingInstance.mMesh->m_indices[0],
                renderingInstance.mMesh->m_indices.size(),
                mRadius,
                instance.mVolumeSampling,
                instance.mSurfaceSampling,
                instance.mClusterSpacing*mRadius,
                instance.mClusterRadius*mRadius,
                instance.mClusterStiffness,
                instance.mLinkRadius*mRadius,
                instance.mLinkStiffness,
                instance.mGlobalStiffness,
                instance.mClusterPlasticThreshold,
                instance.mClusterPlasticCreep);

        double createEnd = GetSeconds();

        // create skinning
        const int maxWeights = 4;

        renderingInstance.mSkinningIndices.resize(renderingInstance.mMesh->m_positions.size()*maxWeights);
        renderingInstance.mSkinningWeights.resize(renderingInstance.mMesh->m_positions.size()*maxWeights);

        for (int i = 0; i < asset->numShapes; ++i)
            renderingInstance.mRigidRestPoses.push_back(Vec3(&asset->shapeCenters[i * 3]));

        double skinStart = GetSeconds();

        NvFlexExtCreateSoftMeshSkinning(
                (float*)&renderingInstance.mMesh->m_positions[0],
                renderingInstance.mMesh->m_positions.size(),
                asset->shapeCenters,
                asset->numShapes,
                instance.mSkinningFalloff,
                instance.mSkinningMaxDistance,
                &renderingInstance.mSkinningWeights[0],
                &renderingInstance.mSkinningIndices[0]);

        double skinEnd = GetSeconds();

        printf("Created soft in %f ms Skinned in %f\n", (createEnd - createStart)*1000.0f, (skinEnd - skinStart)*1000.0f);

        const int particleOffset = g_buffers->positions.size();
        const int indexOffset = g_buffers->rigidOffsets.back();

        // add particle data to solver
        for (int i = 0; i < asset->numParticles; ++i)
        {
            g_buffers->positions.push_back(&asset->particles[i * 4]);
            g_buffers->velocities.push_back(0.0f);

            const int phase = NvFlexMakePhase(group, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
            g_buffers->phases.push_back(phase);
        }

        // add shape data to solver
        for (int i = 0; i < asset->numShapeIndices; ++i)
            g_buffers->rigidIndices.push_back(asset->shapeIndices[i] + particleOffset);

        for (int i = 0; i < asset->numShapes; ++i)
        {
            g_buffers->rigidOffsets.push_back(asset->shapeOffsets[i] + indexOffset);
            g_buffers->rigidTranslations.push_back(Vec3(&asset->shapeCenters[i * 3]));
            g_buffers->rigidRotations.push_back(Quat());
            g_buffers->rigidCoefficients.push_back(asset->shapeCoefficients[i]);
        }


        // add plastic deformation data to solver, if at least one asset has non-zero plastic deformation coefficients, leave the according pointers at NULL otherwise
        if (plasticDeformation)
        {
            if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
            {
                for (int i = 0; i < asset->numShapes; ++i)
                {
                    g_buffers->rigidPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
                    g_buffers->rigidPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
                }
            }
            else
            {
                for (int i = 0; i < asset->numShapes; ++i)
                {
                    g_buffers->rigidPlasticThresholds.push_back(0.0f);
                    g_buffers->rigidPlasticCreeps.push_back(0.0f);
                }
            }
        }
        else
        {
            if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
            {
                int oldBufferSize = g_buffers->rigidCoefficients.size() - asset->numShapes;

                g_buffers->rigidPlasticThresholds.resize(oldBufferSize);
                g_buffers->rigidPlasticCreeps.resize(oldBufferSize);

                for (int i = 0; i < oldBufferSize; i++)
                {
                    g_buffers->rigidPlasticThresholds[i] = 0.0f;
                    g_buffers->rigidPlasticCreeps[i] = 0.0f;
                }

                for (int i = 0; i < asset->numShapes; ++i)
                {
                    g_buffers->rigidPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
                    g_buffers->rigidPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
                }

                plasticDeformation = true;
            }
        }

        // add link data to the solver
        for (int i = 0; i < asset->numSprings; ++i)
        {
            g_buffers->springIndices.push_back(asset->springIndices[i * 2 + 0]);
            g_buffers->springIndices.push_back(asset->springIndices[i * 2 + 1]);

            g_buffers->springStiffness.push_back(asset->springCoefficients[i]);
            g_buffers->springLengths.push_back(asset->springRestLengths[i]);
        }

        NvFlexExtDestroyAsset(asset);

        mRenderingInstances.push_back(renderingInstance);
    }

    virtual void Draw(int pass)
    {
        if (!g_drawMesh)
            return;

        for (int s = 0; s < int(mRenderingInstances.size()); ++s)
        {
            const RenderingInstance& instance = mRenderingInstances[s];

            Mesh m;
            m.m_positions.resize(instance.mMesh->m_positions.size());
            m.m_normals.resize(instance.mMesh->m_normals.size());
            m.m_indices = instance.mMesh->m_indices;

            for (int i = 0; i < int(instance.mMesh->m_positions.size()); ++i)
            {
                Vec3 softPos;
                Vec3 softNormal;

                for (int w = 0; w < 4; ++w)
                {
                    const int cluster = instance.mSkinningIndices[i * 4 + w];
                    const float weight = instance.mSkinningWeights[i * 4 + w];

                    if (cluster > -1)
                    {
                        // offset in the global constraint array
                        int rigidIndex = cluster + instance.mOffset;

                        Vec3 localPos = Vec3(instance.mMesh->m_positions[i]) - instance.mRigidRestPoses[cluster];

                        Vec3 skinnedPos = g_buffers->rigidTranslations[rigidIndex] + Rotate(g_buffers->rigidRotations[rigidIndex], localPos);
                        Vec3 skinnedNormal = Rotate(g_buffers->rigidRotations[rigidIndex], instance.mMesh->m_normals[i]);

                        softPos += skinnedPos*weight;
                        softNormal += skinnedNormal*weight;
                    }
                }

                m.m_positions[i] = Point3(softPos);
                m.m_normals[i] = softNormal;
            }

            DrawMesh(&m, instance.mColor);
        }
    }
};
