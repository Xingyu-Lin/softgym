
class FEM : public Scene
{
public:

	FEM(const char* file) : file(file) 
	{
		Initialize();
	}
	
	FEM(int x, int y, int z, bool bottomFixed, bool topFixed=false) : file(NULL), beamx(x), beamy(y), beamz(z), bottomFixed(bottomFixed), topFixed(topFixed)
	{
		Initialize();
	}

	const char* file;
	Mesh mesh;

	Vec3 handPos;
	float handWidth;
	float handRot;

	int beamx;
	int beamy;
	int beamz;
	bool bottomFixed;
	bool topFixed;

	virtual void Initialize()
	{
		const float density = 1000.0f;

		if (!file)
		{
			const float radius = 0.1f;

			Vec3 lower = Vec3(-radius*beamx/2, 0.1f + 0.02f, -radius*beamz/2);

			CreateTetGrid(Transform(lower, Quat()), beamx, beamy, beamz, radius, radius, radius, density, ConstantMaterial<0>, bottomFixed, topFixed);
		}
		else
		{
			CreateTetMesh(file, Vec3(0.0f, 1.0f, 0.0f), 2.0f, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));
		}

		g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

		g_tetraMaterials.resize(0);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+9f, 0.45f, 0.0f));

		g_params.dynamicFriction = 1.0f;
		g_params.staticFriction = 1.0f;

		g_params.radius = 0.05f;
		g_params.collisionDistance = 0.07f;
		g_params.shapeCollisionMargin = 0.08f;

		g_params.relaxationMode = NvFlexRelaxationMode::eNvFlexRelaxationGlobal;
		g_params.relaxationFactor = 0.25f;
		
		//g_params.relaxationFactor = 0.0f;

		g_params.numIterations = 100;
		g_numSubsteps = 4;

		// draw options		
		g_drawPoints = false;
		g_drawMesh = false;
		g_drawCloth = false;
		g_pause = true;

		g_params.numPlanes = 0;

		Vec3 lower, upper;
		GetParticleBounds(lower, upper);

		handPos = 0.5f*(lower + upper) + Vec3(0.0f, 0.5f, 0.0f);
		handPos.y = upper.y + 0.5f;

		handWidth = upper.x - lower.x + 0.5f;
		handRot = 0.0f;

		float radius = 0.125f;
		float halfHeight = 0.5f;

#if 0

		{

			NvFlexRigidShape shape1;
			NvFlexMakeRigidCapsuleShape(&shape1, 0, radius, halfHeight, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body1;
			NvFlexMakeRigidBody(g_flexLib, &body1, handPos - Vec3(handWidth, 0.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f), &shape1, &density, 1);

			g_buffers->rigidShapes.push_back(shape1);
			g_buffers->rigidBodies.push_back(body1);
		}

		{
			NvFlexRigidShape shape1;
			NvFlexMakeRigidCapsuleShape(&shape1, 1, radius, halfHeight, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body1;
			NvFlexMakeRigidBody(g_flexLib, &body1, handPos + Vec3(handWidth, 0.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f), &shape1, &density, 1);

			g_buffers->rigidShapes.push_back(shape1);
			g_buffers->rigidBodies.push_back(body1);
		}

		{
			NvFlexRigidPose t;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[0], &t);

			NvFlexRigidJoint joint;
			NvFlexMakeFixedJoint(&joint, -1, 0, t, NvFlexMakeRigidPose(0,0));
			g_buffers->rigidJoints.push_back(joint);
		}

		{

			NvFlexRigidPose t;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[1], &t);

			NvFlexRigidJoint joint;
			NvFlexMakeFixedJoint(&joint, -1, 1, t, NvFlexMakeRigidPose(0,0));
			g_buffers->rigidJoints.push_back(joint);
		}
#endif

	}

	virtual void DoGui()
	{

#if 0
		imguiSlider("Hand Pos (x)", &handPos.x, -1.0f, 1.0f, 0.0005f);
		imguiSlider("Hand Pos (y)", &handPos.y, 0.0f, 3.0f, 0.0005f);
		imguiSlider("Hand Pos (z)", &handPos.z, -1.0f, 1.0f, 0.0005f);
		imguiSlider("Hand Rot", &handRot, -kPi * 2.0f, kPi * 2.0f, 0.0005f);

		imguiSlider("Hand Width", &handWidth, -3.0f, 3.0f, 0.001f);

		g_buffers->rigidJoints[0].pose0.p[1] = handPos.y;
		g_buffers->rigidJoints[1].pose0.p[1] = handPos.y;

		g_buffers->rigidJoints[0].targets[eNvFlexRigidJointAxisY] = handWidth;
		g_buffers->rigidJoints[1].targets[eNvFlexRigidJointAxisY] = -handWidth;
#endif

	}

	virtual void Draw(int pass)
	{		
		float sum = 0.0f;

		if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i=0; i < g_buffers->triangles.size(); ++i)
				mesh.m_indices[i] = g_buffers->triangles[i];

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i=0; i < g_buffers->tetraIndices.size(); i+=4)
			{
				float vonMises = fabsf(g_buffers->tetraStress[i/4]);

				sum += vonMises;

				//printf("%f\n", vonMises);

				averageStress[g_buffers->tetraIndices[i+0]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+1]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+2]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+3]] += Vec2(vonMises, 1.0f);

				rangeMin = Min(rangeMin, vonMises);
				rangeMax = Max(rangeMax, vonMises);
			}

			//printf("%f %f\n", rangeMin,rangeMax);

			rangeMin = 0.0f;//Min(rangeMin, vonMises);
			rangeMax = 0.5f;//Max(rangeMax, vonMises);

			for (int i = 0; i < g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x/averageStress[i].y);
			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);

		if (pass == 0)
		{

			SetFillMode(true);
			
			DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.001f);
			
			SetFillMode(false);

		}
	}
};

class FEMTwist : public Scene
{
public:

	FEMTwist(int x, int y, int z, bool bottomFixed, bool topFixed=false) : file(NULL), beamx(x), beamy(y), beamz(z), bottomFixed(bottomFixed), topFixed(topFixed)
	{
		Initialize();
	}

	const char* file;
	Mesh mesh;

	Vec3 handPos;
	float handWidth;
	float handRot;

	int beamx;
	int beamy;
	int beamz;
	bool bottomFixed;
	bool topFixed;

	virtual void Initialize()
	{
		const float density = 1000.0f;

		if (!file)
		{
			if (1)
			{
				const float radius = 0.1f;

				Vec3 lower = Vec3(-radius*beamx/2, 0.02f, -radius*beamz/2);

				CreateTetGrid(Transform(lower, Quat()), beamx, beamy, beamz, radius, radius, radius, density, ConstantMaterial<0>, bottomFixed, topFixed);
			}
			else
			{
				Mesh* tet = CreateTetrahedron();

				for (int i=0; i < int(tet->GetNumVertices()); ++i)
				{
					g_buffers->positions.push_back(Vec4(Vec3(tet->m_positions[i]), 1.0f));
					g_buffers->velocities.push_back(0.0f);
					g_buffers->phases.push_back(NvFlexMakePhase(0, 0));

					g_buffers->tetraIndices.push_back(i);
				}

				g_buffers->positions[0].w = 0.0f;
				g_buffers->positions[1].w = 0.0f;
				g_buffers->positions[3].w = 0.0f;

				g_drawPoints = true;

				// calculate rest poses
				for (int i=0; i < g_buffers->tetraIndices.size(); i+=4)
				{
					Vec4 x0 = g_buffers->positions[i*4+0];
					Vec4 x1 = g_buffers->positions[i*4+1];
					Vec4 x2 = g_buffers->positions[i*4+2];
					Vec4 x3 = g_buffers->positions[i*4+3];

					x1 -= Vec4(Vec3(x0), 0.0f);
					x2 -= Vec4(Vec3(x0), 0.0f);
					x3 -= Vec4(Vec3(x0), 0.0f);

					bool success;
					Matrix33 Q = Matrix33(Vec3(x1), Vec3(x2), Vec3(x3));
					Matrix33 rest = Inverse(Q, success);

					g_buffers->tetraRestPoses.push_back(rest);
				}

				g_buffers->tetraMaterials.push_back(0);
			}
		}
		else
		{
			CreateTetMesh(file, Vec3(0.0f, 1.0f, 0.0f), 2.0f, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));
		}

		g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

		g_tetraMaterials.resize(0);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+6f, 0.5f, 0.0f));

		g_params.dynamicFriction = 1.0f;
		g_params.staticFriction = 1.0f;

		g_params.radius = 0.05f;
		g_params.collisionDistance = 0.07f;
		g_params.shapeCollisionMargin = 0.08f;

		g_params.relaxationMode = NvFlexRelaxationMode::eNvFlexRelaxationGlobal;
		g_params.relaxationFactor = 0.25f;
		
		//g_params.relaxationFactor = 0.0f;

		g_params.numIterations = 100;
		g_numSubsteps = 4;

		// draw options		
		g_drawPoints = false;
		g_drawMesh = false;
		g_drawCloth = false;
		g_pause = true;

		g_params.numPlanes = 0;

		Vec3 lower, upper;
		GetParticleBounds(lower, upper);

		handPos = 0.5f*(lower + upper) + Vec3(0.0f, 0.5f, 0.0f); //Vec3(0.0f, 1.0f, 0.0f);
		handPos.y = upper.y + 0.5f;

		handWidth = upper.x - lower.x + 0.5f;
		handRot = 0.0f;

		Update();
	}

	virtual void Update()
	{

		const int numFrames = 300;
		float targetAngle = kPi;
		float targetPos = 0.5f;

		float deltaAngle = targetAngle/numFrames;
		float deltaPos = targetPos/numFrames;

		if (topFixed && g_frame < numFrames)
		{
			Quat r = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), deltaAngle);

			for (int i=0; i < g_buffers->positions.size(); ++i)
			{
				Vec4 p = g_buffers->positions[i];

				if (p.w == 0.0f && p.y > 0.5f)
				{
					Vec3 x = Rotate(r, Vec3(p));

					x.y += deltaPos;

					g_buffers->positions[i].x = x.x;
					g_buffers->positions[i].y = x.y;
					g_buffers->positions[i].z = x.z;

				}

			}
		}
	}

	virtual void Draw(int pass)
	{		
		float sum = 0.0f;

		if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i=0; i < g_buffers->triangles.size(); ++i)
				mesh.m_indices[i] = g_buffers->triangles[i];

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i=0; i < g_buffers->tetraIndices.size(); i+=4)
			{
				float vonMises = fabsf(g_buffers->tetraStress[i/4]);

				sum += vonMises;

				//printf("%f\n", vonMises);

				averageStress[g_buffers->tetraIndices[i+0]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+1]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+2]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+3]] += Vec2(vonMises, 1.0f);

				rangeMin = Min(rangeMin, vonMises);
				rangeMax = Max(rangeMax, vonMises);
			}

			rangeMin = 0.0f;//Min(rangeMin, vonMises);
			rangeMax = 0.5f;//Max(rangeMax, vonMises);

			for (int i = 0; i < g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x/averageStress[i].y);
			}
		}


		DrawMesh(&mesh, g_renderMaterials[0]);

		if (pass == 0)
		{

			SetFillMode(true);
			
			DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.001f);
			
			SetFillMode(false);

		}
	}

};


class FEMPoisson : public Scene
{
public:

	Mesh mesh;

	FEMPoisson()
	{
		const float density = 1000.0f;

		CreateTetGrid(Transform(Vec3(0.0f, 0.5f, 0.0f), Quat()), 20, 1, 10, 0.1f, 0.05f, 0.1f, density, ConstantMaterial<0>, false, false, true, true);

		g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

		g_tetraMaterials.resize(0);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+6f, 0.45f));
	
		g_params.dynamicFriction = 0.9f;
		g_params.staticFriction = 1.0f;

		g_params.radius = 0.0f;
		g_params.collisionDistance = 0.02f;
		g_params.shapeCollisionMargin = 0.02f;

		g_params.relaxationMode = NvFlexRelaxationMode::eNvFlexRelaxationGlobal;
		g_params.relaxationFactor = 0.25f;
		
		//g_params.relaxationFactor = 0.0f;

		g_params.numIterations = 50;
		g_numSubsteps = 2;

		// draw options		
		g_drawPoints = false;
		g_drawMesh = false;
		g_drawCloth = false;
		g_pause = true;

		g_lightDistance *= 2.0f;


		Vec3 lower, upper;
		GetParticleBounds(lower, upper);

		Update();

        FILE* file = fopen("poisson.bin", "rb");
        if (file)
        {
            fread(&g_buffers->positions[0], sizeof(Vec4), g_buffers->positions.size(), file);
            fclose(file);
        }            
	}

	virtual void KeyDown(int key)
	{

        g_buffers->positions.map();

        FILE* file = fopen("poisson.bin", "wb");
        if (file)
        {
            fwrite(&g_buffers->positions[0], sizeof(Vec4), g_buffers->positions.size(), file);
            fclose(file);
        }

        g_buffers->positions.unmap();

	}

	virtual void CenterCamera()
	{
		g_camPos = Vec3(3.004286f, 2.886685f, 5.639095f);
		g_camAngle = Vec3(0.001745f, -0.387462f, 0.000000f);
	}

	virtual void DoGui()
	{
	}

	virtual void Update()
	{
		
		for (int i=0; i < g_buffers->positions.size(); ++i)
		{
			float invDt = 1.0f / g_dt;
			if (g_buffers->positions[i].w == 0.0f && g_buffers->positions[i].x > 1.9f)
			{
				Vec4 prev = g_buffers->positions[i];
				g_buffers->positions[i].x = Lerp(2.0f, 6.0f, sinf(-kPi*0.5f + g_frame*g_dt)*0.5f + 0.5f);
				g_buffers->velocities[i] = Vec3(g_buffers->positions[i] - prev) * invDt;
			}
		}

		/*
		std::vector<Vec4> prevPositions;
		std::vector<Quat> prevRotations;

		for (int i=0; i < g_buffers->shapePositions.size(); ++i)
		{
			prevPositions.push_back(g_buffers->shapePositions[i]);
			prevRotations.push_back(g_buffers->shapeRotations[i]);
		}

		ClearShapes();

		float radius = 0.125f;
		float halfHeight = 0.5f;

		Quat rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), handRot);

		AddCapsule(radius, halfHeight, handPos - rot*Vec3(-0.5*handWidth, 0.0f, 0.0f), rot*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f));
		AddCapsule(radius, halfHeight, handPos - rot*Vec3(+0.5*handWidth, 0.0f, 0.0f), rot*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f));
		AddCapsule(radius, handWidth*0.5f, handPos + Vec3(0.0f, halfHeight, 0.0f), rot*Quat());

		// set previous for friction
		for (int i=0; i < prevPositions.size(); ++i)
		{
			g_buffers->shapePrevPositions[i] = prevPositions[i];
			g_buffers->shapePrevRotations[i] = prevRotations[i];
		}

		UpdateShapes();
		*/
	}

	virtual void Draw(int pass)
	{		
		if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i=0; i < g_buffers->triangles.size(); ++i)
				mesh.m_indices[i] = g_buffers->triangles[i];

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i=0; i < g_buffers->tetraIndices.size(); i+=4)
			{
				float vonMises = fabsf(g_buffers->tetraStress[i/4]);

				//printf("%f\n", vonMises);

				averageStress[g_buffers->tetraIndices[i+0]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+1]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+2]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+3]] += Vec2(vonMises, 1.0f);

				rangeMin = Min(rangeMin, vonMises);
				rangeMax = Max(rangeMax, vonMises);
			}

			//printf("%f %f\n", rangeMin,rangeMax);

			rangeMin = 0.0f;//Min(rangeMin, vonMises);
			rangeMax = 0.5f;//Max(rangeMax, vonMises);

			for (int i=0; i < g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x/averageStress[i].y);
			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);

		if (pass == 0)
		{

			SetFillMode(true);
			
			DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.001f);
			
			SetFillMode(false);

		}
	}
};




class FEMNet : public Scene
{
public:

	Mesh mesh;

	FEMNet()
	{
		const float density = 1000.0f;

		CreateTetGrid(Transform(Vec3(0.0f, 0.5f, 0.0f), Quat()), 20, 1, 10, 0.1f, 0.05f, 0.1f, density, ConstantMaterial<0>, false, false, true, true);

		g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

		g_tetraMaterials.resize(0);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+8f, 0.45f));

		// sphere drop
		NvFlexRigidShape shape;
		NvFlexMakeRigidSphereShape(&shape, 0, 0.25f, NvFlexMakeRigidPose(0,0));

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(1.0f, 1.5f, 0.5f), Quat(), &shape, &density, 1);

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);

	
		g_params.dynamicFriction = 0.9f;
		g_params.staticFriction = 1.0f;

		g_params.radius = 0.0f;
		g_params.collisionDistance = 0.02f;
		g_params.shapeCollisionMargin = 0.02f;

		g_params.relaxationMode = NvFlexRelaxationMode::eNvFlexRelaxationGlobal;
		g_params.relaxationFactor = 0.25f;
		
		//g_params.relaxationFactor = 0.0f;

		g_params.numIterations = 50;
		g_numSubsteps = 2;

		// draw options		
		g_drawPoints = false;
		g_drawMesh = false;
		g_drawCloth = false;
		g_pause = true;

		g_lightDistance *= 2.0f;


		Vec3 lower, upper;
		GetParticleBounds(lower, upper);

		Update();
	}

	virtual void Draw(int pass)
	{		
		if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i=0; i < g_buffers->triangles.size(); ++i)
				mesh.m_indices[i] = g_buffers->triangles[i];

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i=0; i < g_buffers->tetraIndices.size(); i+=4)
			{
				float vonMises = fabsf(g_buffers->tetraStress[i/4]);

				//printf("%f\n", vonMises);

				averageStress[g_buffers->tetraIndices[i+0]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+1]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+2]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+3]] += Vec2(vonMises, 1.0f);

				rangeMin = Min(rangeMin, vonMises);
				rangeMax = Max(rangeMax, vonMises);
			}

			//printf("%f %f\n", rangeMin,rangeMax);

			rangeMin = 0.0f;//Min(rangeMin, vonMises);
			rangeMax = 0.5f;//Max(rangeMax, vonMises);

			for (int i=0; i < g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x/averageStress[i].y);
			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);

		if (pass == 0)
		{

			SetFillMode(true);
			
			DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.001f);
			
			SetFillMode(false);

		}
	}
};


class FEMStickSlip : public Scene
{
public:

	Mesh mesh;

	FEMStickSlip()
	{
		const float density = 1000.0f;

		CreateTetGrid(Transform(Vec3(0.0f, 1.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0))), 20, 4, 4, 0.1f, 0.1f, 0.1f, density, ConstantMaterial<0>, false, false, true, false);

		g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

		g_tetraMaterials.resize(0);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+8f, 0.45f));

		g_params.dynamicFriction = 0.75f;
		g_params.staticFriction = 1.0f;

		g_params.radius = 0.0f;
		g_params.collisionDistance = 0.02f;
		g_params.shapeCollisionMargin = 0.02f;

		g_params.relaxationMode = NvFlexRelaxationMode::eNvFlexRelaxationGlobal;
		g_params.relaxationFactor = 0.25f;
		
		//g_params.relaxationFactor = 0.0f;

		g_params.numIterations = 50;
		g_numSubsteps = 2;

		// draw options		
		g_drawPoints = false;
		g_drawMesh = false;
		g_drawCloth = false;
		g_pause = true;

		g_lightDistance *= 2.0f;


		Vec3 lower, upper;
		GetParticleBounds(lower, upper);
	}

	float speed = 0.5f;

	virtual void Update()
	{
		//set velocity on inv mass particles
		for (int i=0; i < g_buffers->positions.size(); ++i)
		{
			if (g_buffers->positions[i].w == 0.0f)
			{
				g_buffers->velocities[i] = Vec3(speed, 0.0f, 0.0f);
			}
		}
	}


	virtual void DoGui()
	{
		imguiSlider("Speed", &speed, 0.0f, 10.0f, 0.0001f);

	}

	virtual void Draw(int pass)
	{		
		if (pass == 1)
		{
			mesh.m_positions.resize(g_buffers->positions.size());
			mesh.m_normals.resize(g_buffers->normals.size());
			mesh.m_colours.resize(g_buffers->positions.size());
			mesh.m_indices.resize(g_buffers->triangles.size());

			for (int i=0; i < g_buffers->triangles.size(); ++i)
				mesh.m_indices[i] = g_buffers->triangles[i];

			float rangeMin = FLT_MAX;
			float rangeMax = -FLT_MAX;

			std::vector<Vec2> averageStress(mesh.m_positions.size());

			// calculate average Von-Mises stress on each vertex for visualization
			for (int i=0; i < g_buffers->tetraIndices.size(); i+=4)
			{
				float vonMises = fabsf(g_buffers->tetraStress[i/4]);

				//printf("%f\n", vonMises);

				averageStress[g_buffers->tetraIndices[i+0]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+1]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+2]] += Vec2(vonMises, 1.0f);
				averageStress[g_buffers->tetraIndices[i+3]] += Vec2(vonMises, 1.0f);

				rangeMin = Min(rangeMin, vonMises);
				rangeMax = Max(rangeMax, vonMises);
			}

			//printf("%f %f\n", rangeMin,rangeMax);

			rangeMin = 0.0f;//Min(rangeMin, vonMises);
			rangeMax = 0.5f;//Max(rangeMax, vonMises);

			for (int i=0; i < g_buffers->positions.size(); ++i)
			{
				mesh.m_positions[i] = Point3(g_buffers->positions[i]);
				mesh.m_normals[i] = Vec3(g_buffers->normals[i]);

				mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x/averageStress[i].y);
			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);

		if (pass == 0)
		{

			SetFillMode(true);
			
			DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.001f);
			
			SetFillMode(false);

		}
	}
};
