

class SoftBody : public Scene
{

public:
	
	SoftBody() :
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

	virtual void Initialize()
	{
		float radius = mRadius;

		// no fluids or sdf based collision
		g_solverDesc.featureMode = eNvFlexFeatureModeSimpleSolids;

		g_params.radius = radius;
		g_params.dynamicFriction = 0.35f;
		g_params.particleFriction = 0.25f;
		g_params.numIterations = 4;
		g_params.collisionDistance = radius*0.75f;

		g_params.relaxationFactor = mRelaxationFactor;

		g_windStrength = 0.0f;

		g_numSubsteps = 2;

		// draw options
		g_drawPoints = false;
		g_wireframe = false;
		g_drawSprings = false;
		g_drawBases = false;

		g_buffers->shapeMatchingOffsets.push_back(0);

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
	}

	void CreateSoftBody(Instance instance, int group = 0)
	{
		RenderingInstance renderingInstance;

		Mesh* mesh = ImportMesh(GetFilePathByPlatform(instance.mFile).c_str());
		mesh->Normalize();
		mesh->Transform(TranslationMatrix(Point3(instance.mTranslation))*ScaleMatrix(instance.mScale*mRadius));

		renderingInstance.mMesh = mesh;
		renderingInstance.mColor = instance.mColor;
		renderingInstance.mOffset = g_buffers->shapeMatchingTranslations.size();

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
		const int indexOffset = g_buffers->shapeMatchingOffsets.back();

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
			g_buffers->shapeMatchingIndices.push_back(asset->shapeIndices[i] + particleOffset);

		for (int i = 0; i < asset->numShapes; ++i)
		{
			g_buffers->shapeMatchingOffsets.push_back(asset->shapeOffsets[i] + indexOffset);
			g_buffers->shapeMatchingTranslations.push_back(Vec3(&asset->shapeCenters[i * 3]));
			g_buffers->shapeMatchingRotations.push_back(Quat());
			g_buffers->shapeMatchingCoefficients.push_back(asset->shapeCoefficients[i]);
		}


		// add plastic deformation data to solver, if at least one asset has non-zero plastic deformation coefficients, leave the according pointers at NULL otherwise
		if (plasticDeformation)
		{
			if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
			{
				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->shapeMatchingPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
					g_buffers->shapeMatchingPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
				}
			}
			else
			{
				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->shapeMatchingPlasticThresholds.push_back(0.0f);
					g_buffers->shapeMatchingPlasticCreeps.push_back(0.0f);
				}
			}
		}
		else 
		{
			if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
			{
				int oldBufferSize = g_buffers->shapeMatchingCoefficients.size() - asset->numShapes;

				g_buffers->shapeMatchingPlasticThresholds.resize(oldBufferSize);
				g_buffers->shapeMatchingPlasticCreeps.resize(oldBufferSize);

				for (int i = 0; i < oldBufferSize; i++)
				{
					g_buffers->shapeMatchingPlasticThresholds[i] = 0.0f;
					g_buffers->shapeMatchingPlasticCreeps[i] = 0.0f;
				}

				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->shapeMatchingPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
					g_buffers->shapeMatchingPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
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

						Vec3 skinnedPos = g_buffers->shapeMatchingTranslations[rigidIndex] + Rotate(g_buffers->shapeMatchingRotations[rigidIndex], localPos);
						Vec3 skinnedNormal = Rotate(g_buffers->shapeMatchingRotations[rigidIndex], instance.mMesh->m_normals[i]);

						softPos += skinnedPos*weight;
						softNormal += skinnedNormal*weight;
					}
				}

				m.m_positions[i] = Point3(softPos);
				m.m_normals[i] = softNormal;
			}

			RenderMaterial mat;
			mat.frontColor = Vec3(instance.mColor);
			mat.backColor = Vec3(instance.mColor);
			mat.specular = 0.0f;
			mat.roughness = 1.0f;

			DrawMesh(&m, mat);
		}
	}
};



class SoftOctopus : public SoftBody
{
public:

	SoftOctopus()
	{
		Instance octopus("../../data/softs/octopus.obj");
		octopus.mScale = Vec3(32.0f);
		octopus.mClusterSpacing = 2.75f;
		octopus.mClusterRadius = 3.0f;
		octopus.mClusterStiffness = 0.15f;
		octopus.mSurfaceSampling = 1.0f;

		AddStack(octopus, 1, 3, 1);

		Initialize();
	}
};

class SoftRope : public SoftBody
{
public:

	SoftRope()
	{
	    Instance rope("../../data/rope.obj");
		rope.mScale = Vec3(50.0f);
		rope.mClusterSpacing = 1.5f;
		rope.mClusterRadius = 0.0f;
		rope.mClusterStiffness = 0.55f;

		AddInstance(rope);

		Initialize();
	}
};


class SoftCloth : public SoftBody
{
public:

	SoftCloth()
	{
	    Instance cloth("../../data/box_ultra_high.ply");
		cloth.mScale = Vec3(20.0f, 0.2f, 20.0f);
		cloth.mClusterSpacing = 1.0f;
		cloth.mClusterRadius = 2.0f;
		cloth.mClusterStiffness = 0.2f;
		cloth.mLinkRadius = 2.0f;
		cloth.mLinkStiffness = 1.0f;
		cloth.mSkinningFalloff = 1.0f;
		cloth.mSkinningMaxDistance = 100.f;
		
		mRadius = 0.05f;
		AddInstance(cloth);

		Initialize();
	}
};

class SoftTeapot : public SoftBody
{
public:

	SoftTeapot()
	{
		Instance teapot("../../data/teapot.ply");
		teapot.mScale = Vec3(25.0f);
		teapot.mClusterSpacing = 3.0f;
		teapot.mClusterRadius = 0.0f;
		teapot.mClusterStiffness = 0.1f;	
	
		AddInstance(teapot);

		Initialize();
	}
};

class SoftArmadillo : public SoftBody
{
public:

	SoftArmadillo()
	{
		Instance armadillo("../../data/armadillo.ply");
		armadillo.mScale = Vec3(25.0f);
		armadillo.mClusterSpacing = 3.0f;
		armadillo.mClusterRadius = 0.0f;

		AddInstance(armadillo);

		Initialize();
	}
};


class SoftBunny : public SoftBody
{
public:

	SoftBunny()
	{
		Instance softbunny("../../data/bunny.ply");
		softbunny.mScale = Vec3(20.0f);
		softbunny.mClusterSpacing = 3.5f;
		softbunny.mClusterRadius = 0.0f;
		softbunny.mClusterStiffness = 0.2f;
    
		AddInstance(softbunny);
		
		Initialize();
	}
};

class PlasticBunnies : public SoftBody
{
public:

	PlasticBunnies()
	{
		Instance plasticbunny("../../data/bunny.ply");
		plasticbunny.mScale = Vec3(10.0f);
		plasticbunny.mClusterSpacing = 1.0f;
		plasticbunny.mClusterRadius = 0.0f;
		plasticbunny.mClusterStiffness = 0.0f;
		plasticbunny.mGlobalStiffness = 1.0f;
		plasticbunny.mClusterPlasticThreshold = 0.0015f;
		plasticbunny.mClusterPlasticCreep = 0.15f;
		plasticbunny.mTranslation[1] = 5.0f;

		mPlinth = true;
		AddStack(plasticbunny, 1, 10, 1, true);

		Initialize();
	}
};

class PlasticStack : public SoftBody
{
public:

	PlasticStack()
	{
		Instance stackBox("../../data/box_high.ply");
		stackBox.mScale = Vec3(10.0f);
		stackBox.mClusterSpacing = 1.5f;
		stackBox.mClusterRadius = 0.0f;
		stackBox.mClusterStiffness = 0.0f;
		stackBox.mGlobalStiffness = 1.0f;
		stackBox.mClusterPlasticThreshold = 0.0015f;
		stackBox.mClusterPlasticCreep = 0.25f;
		stackBox.mTranslation[1] = 1.0f;
		
		Instance stackSphere("../../data/sphere.ply");
		stackSphere.mScale = Vec3(10.0f);
		stackSphere.mClusterSpacing = 1.5f;
		stackSphere.mClusterRadius = 0.0f;
		stackSphere.mClusterStiffness = 0.0f;
		stackSphere.mGlobalStiffness = 1.0f;
		stackSphere.mClusterPlasticThreshold = 0.0015f;
		stackSphere.mClusterPlasticCreep = 0.25f;
		stackSphere.mTranslation[1] = 2.0f;
		
		AddInstance(stackBox);
		AddInstance(stackSphere);

		for (int i = 0; i < 3; i++)
		{
			stackBox.mTranslation[1] += 2.0f;
			stackSphere.mTranslation[1] += 2.0f;
			AddInstance(stackBox);
			AddInstance(stackSphere);
		}	

		Initialize();
	}
};