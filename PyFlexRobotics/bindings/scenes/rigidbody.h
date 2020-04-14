#include "../mjcf.h"
#include "../urdf.h"


class RigidFEM : public Scene
{
public:

	const char* file;
	Mesh mesh;

	RigidFEM(const char* f=NULL) : file(f)
	{
		if (1)
		{
			float blockDensity = 1000.0f;

			if (1 || !file)
			{
				int beamx = 20;
				int beamy = 8;
				int beamz = 8;

				CreateTetGrid(Vec3(0.0f, 0.02f, 0.0f), beamx, beamy, beamz, 0.1f, 0.1f, 0.1f, blockDensity, ConstantMaterial<0>, true);	
			}
			else
			{
				CreateTetMesh(file, Vec3(0.0f, 0.5f, 0.0f), 0.5f, blockDensity, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));
			}

			g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

			g_tetraMaterials.resize(0);
			g_tetraMaterials.push_back(IsotropicMaterialCompliance(1.e+5f, 0.4f, 0.0f));		
		}

		Vec3 lower, upper;
		GetParticleBounds(lower, upper);

		Vec3 center = 0.5f * (lower + upper);		

		// drop a rigid ball on the FEM block
		float capsuleDensity = 1000.0f;

		NvFlexRigidShape capsule;		
		NvFlexMakeRigidCapsuleShape(&capsule, 0, 0.25f, 0.5f, NvFlexMakeRigidPose(0,0));

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, center + Vec3(0.0f, 1.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), kPi*0.5f), &capsule, &capsuleDensity, 1);

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(capsule);

		g_params.dynamicFriction = 0.9f;
		g_params.staticFriction = 1.0f;

		g_params.radius = 0.05f;
		g_params.collisionDistance = 0.01f;
		g_params.shapeCollisionMargin = 0.01f;

		g_params.collisionDistance = 1.e-4f;
		g_params.dynamicFriction = 0.2f;

		g_numSubsteps = 2;

		g_params.solverType = eNvFlexSolverPCR;
		g_params.numIterations = 4;

		g_params.relaxationFactor = 0.75f;

		// draw options		
		g_drawPoints = false;
		g_drawMesh = false;
		g_drawCloth = false;
		g_pause = true;

		g_sceneLower = -1.0f;
		g_sceneUpper = 1.0f;
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
				/*
				if (g_buffers->tetraMaterials[i] == 0)
					mesh.m_colours[i] = Colour::kGreen;
				else
					mesh.m_colours[i] = Colour::kRed;
				*/
			}
		}

		DrawMesh(&mesh, g_renderMaterials[0]);

		if (pass == 0)
		{
			SetFillMode(true);

			if (g_buffers->triangles.size())
				DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3]);
			
			SetFillMode(false);
		}
	}
};


class RigidCloth : public Scene
{
public:

	RigidCloth()
	{
		int clothStart = 0;
		int dimx = 20;
		int dimy = 20;

		CreateSpringGrid(Vec3(-1.0f, 0.5f, -1.0f), dimx, dimy, 1, 0.5f, 0, 1.0f, 1.0f, 1.0f, Vec3(0.0f), 1.0f);

		int corner0 = clothStart + 0;
		int corner1 = clothStart + dimx-1;
		int corner2 = clothStart + dimx*(dimy-1);
		int corner3 = clothStart + dimx*dimy-1;

		g_buffers->positions[corner0].w = 0.0f;
		g_buffers->positions[corner1].w = 0.0f;
		g_buffers->positions[corner2].w = 0.0f;
		g_buffers->positions[corner3].w = 0.0f;

		if (0)
		{
			float scale = 0.5f;
			float density = 1000.0f;

			NvFlexRigidShape sphere;		
			NvFlexMakeRigidCapsuleShape(&sphere, 0, 0.125f, 0.25f, NvFlexMakeRigidPose(0,0));

		
			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 1.0f, 0.0f), Quat(), &sphere, &density,  1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(sphere);
		}

		if (1)
		{
			vector<pair<int, NvFlexRigidJointAxis>> ctrls;
			vector<float> motors;
		
			for (int i=0; i < 10; ++i)
			{
				MJCFImporter mj("../../data/humanoid_symmetric.xml");
				mj.AddPhysicsEntities(Transform(Vec3((float)i, 1.0f + (float)i, 0.0f), Quat()), ctrls, motors);
			}
		}
		

		g_params.numIterations = 20;
		g_params.dynamicFriction = 0.2f;

		g_params.numPlanes = 0;

		g_drawPoints = false;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
	}
};


class RigidTerrain : public Scene
{
public:

	RigidTerrain()
	{
		float dz = 0.f;
		for (int t = 0; t < 3; ++t)
		{
			if (1)
			{
				vector<pair<int, NvFlexRigidJointAxis>> ctrls;
				vector<float> motors;

				for (int i = 0; i < 10; ++i)
				{
					MJCFImporter mj("../../data/humanoid.xml");
					mj.AddPhysicsEntities(Transform(Vec3(0.0f + (float)i, 3.0f, dz), Quat()), ctrls, motors);
				}
			}

			Mesh* terrain = CreateTerrain(10.0f, 5.0f, 40, 20, RandVec3() * 10.0f, Vec3(0.3f, 1.0f, 0.15f),
				1 + 6 * t, 0.05f + 0.2f * (float)t);
			terrain->Transform(TranslationMatrix(Point3(0.0f, 1.0f, dz)));

			NvFlexTriangleMeshId terrainId = CreateTriangleMesh(terrain);

			NvFlexRigidShape terrainShape;
			NvFlexMakeRigidTriangleMeshShape(&terrainShape, -1, terrainId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
			terrainShape.filter = 0;

			g_buffers->rigidShapes.push_back(terrainShape);

			dz -= 7.f;
		}

		g_params.numIterations = 25;
		g_params.dynamicFriction = 0.2f;

		g_drawPoints = false;

		g_sceneLower = Vec3(0.0f);
		g_sceneUpper = Vec3(5.f, 3.5f, 1.25f);

		g_pause = true;

		/*
		// create rays
		int numRays = 100000;

		rays = new NvFlexVector<NvFlexRay>(g_flexLib, numRays);
		hits = new NvFlexVector<NvFlexRayHit>(g_flexLib, numRays);

		rays->map();

		for (int i = 0; i < numRays; ++i)
		{
			Vec3 origin = Vec3(Lerp(g_sceneLower.x, g_sceneUpper.x, Randf()),
							Lerp(g_sceneLower.y, g_sceneUpper.y, Randf()),
							Lerp(g_sceneLower.z, g_sceneUpper.z, Randf()));

			Vec3 dir = RandomUnitVector();

			NvFlexRay ray;
			(Vec3&)ray.start = origin;
			(Vec3&)ray.dir = dir;
			ray.filter = 0;
			ray.group = -1;
			ray.maxT = 1.0f;

			(*rays)[i] = ray;
		}

		rays->unmap();
	}

	NvFlexVector<NvFlexRay>* rays;
	NvFlexVector<NvFlexRayHit>* hits;
	
	virtual void Update()
	{
		NvFlexRayCast(g_solver, rays->buffer, hits->buffer, rays->size());
	}

	virtual void Draw(int pass) 
	{
		if (pass == 0)
		{
			hits->map();
			rays->map();

			BeginLines();

			for (int i=0; i < hits->size(); ++i)
			{
				NvFlexRay ray = (*rays)[i];
				NvFlexRayHit hit = (*hits)[i];

				if (hit.shape != -1)
				{
					DrawLine(Vec3(ray.start), Vec3(ray.start) + Vec3(ray.dir)*hit.t, Vec4(1.0f));
				}
			}
			
			EndLines();

			hits->unmap();
			rays->unmap();
			
		}	
		*/
	}
};


class RigidGyroscopic : public Scene
{
public:

	RigidGyroscopic()
	{
		float scale = 0.5f;

		NvFlexRigidShape axis;
		NvFlexMakeRigidBoxShape(&axis, 0, 0.25f*scale, 0.1f*scale, 0.1f*scale, NvFlexMakeRigidPose(Vec3(0.25f*scale + 0.05f*scale, 0.0f, 0.0f), Quat()));

		NvFlexRigidShape tip;
		NvFlexMakeRigidBoxShape(&tip, 0, 0.05f*scale, 0.2f*scale, 1.0f*scale, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));

		NvFlexRigidShape shapes[2] = { axis, tip};
		const float densities[2] = { 100.0f, 100.0f };

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 2.0f*scale, 0.0f), Quat(), shapes, densities, 2);

		// set initial angular velocity
		body.angularVel[0] = 25.0f;
		body.angularVel[1] = 0.01f;
		body.angularVel[2] = 0.01f;
		body.angularDamping = 0.1f;

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(axis);		
		g_buffers->rigidShapes.push_back(tip);

		g_params.gravity[1] = 0.0f;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
	}
};


class RigidFixedJoint : public Scene
{
public:

	RigidFixedJoint()
	{
		const float linkLength = 0.125f;
		const float linkWidth = 0.05f;

		const float density = 1000.0f;
		const float height = 2.0f;
	
		Quat localFrame = Quat();//QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), DegToRad(90.0f));

		NvFlexRigidPose prevJoint = NvFlexMakeRigidPose(Vec3(0.0f, height, 0.0f), localFrame);

		for (int i=0; i < 6; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, bodyIndex, linkLength, linkWidth, linkWidth, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(i*linkLength*2.0f + linkLength, height, 0.0f), Quat(), &shape, &density, 1);

			NvFlexRigidJoint joint;				
			NvFlexMakeFixedJoint(&joint, i-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), localFrame));

			prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), localFrame);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);

		}

		g_numSubsteps = 4;
		g_params.numIterations = 20;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
	}

};


class RigidAngularMotor : public Scene
{
public:

	RigidAngularMotor()
	{
		const float linkLength = 0.125f;
		const float linkWidth = 0.05f;

		const float density = 1000.0f;
		const float height = 2.0f;
	
		Quat localFrame = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), DegToRad(-90.0f));

		NvFlexRigidPose prevJoint = NvFlexMakeRigidPose(Vec3(0.0f, height, 0.0f), localFrame);

		for (int i=0; i < 4; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, bodyIndex, linkLength, linkWidth, linkWidth, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(i*linkLength*2.0f + linkLength, height, 0.0f), Quat(), &shape, &density, 1);

			if (i == 0)
				printf("Link Mass: %f\n", body.mass);

			NvFlexRigidJoint joint;				
			NvFlexMakeFixedJoint(&joint, i-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), localFrame));

			if (i==0)
			{
				joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeVelocity;
				joint.targets[eNvFlexRigidJointAxisTwist] = DegToRad(180.0f);
			}
			else
			{
				joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeFree;
			}

			prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), localFrame);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);

		}

		// attach end weight
		if (0)
		{
			NvFlexRigidShape weight;
			NvFlexMakeRigidBoxShape(&weight, g_buffers->rigidBodies.size(), 0.25f, 0.25f, 0.25f, NvFlexMakeRigidPose(0,0));

			Transform prevPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies.back(), (NvFlexRigidPose*)&prevPose);
			
			Transform endLink = prevPose*(Transform&)prevJoint;
			endLink.p.x += 0.25f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, endLink.p, Quat(), &weight, &density, 1);

			printf("Load Mass: %f\n", body.mass);


			NvFlexRigidJoint joint;				
			NvFlexMakeSphericalJoint(&joint, weight.body-1, weight.body, prevJoint, NvFlexMakeRigidPose(Vec3(-0.25f, 0.0f, 0.0f), Quat()));

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(weight);
			g_buffers->rigidJoints.push_back(joint);
		}
				
		g_numSubsteps = 4;
		g_params.numIterations = 20;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
	}
};


class RigidHingeJoint : public Scene
{
public:

	RigidHingeJoint()
	{
		const float linkLength = 0.125f;
		const float linkWidth = 0.05f;

		const float density = 1000.0f;
		const float height = 2.0f;
	
		Quat localFrame = Quat();//QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), DegToRad(90.0f));

		NvFlexRigidPose prevJoint = NvFlexMakeRigidPose(Vec3(0.0f, height, 0.0f), localFrame);

		for (int i=0; i < 8; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, bodyIndex, linkLength, linkWidth, linkWidth, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(i*linkLength*2.0f + linkLength, height, 0.0f), Quat(), &shape, &density, 1);

			NvFlexRigidJoint joint;
			NvFlexMakeHingeJoint(&joint, i-1, i, prevJoint, NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), Quat()), eNvFlexRigidJointAxisSwing2, -DegToRad(30.0f), DegToRad(30.0f));

			prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), localFrame);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);
		}

		g_numSubsteps = 4;
		g_params.numIterations = 20;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
	}
};


class RigidSphericalJoint : public Scene
{
public:

	RigidSphericalJoint()
	{
		const float linkLength = 0.125f*0.5f;
		const float linkWidth = 0.025f;

		const float density = 1000.0f;
		const float height = 2.5f;
	
		Quat localFrame = Quat();//QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), DegToRad(90.0f));

		// world frame
		NvFlexRigidBody anchor;
		NvFlexMakeRigidBody(g_flexLib, &anchor, Vec3(0.0f, height, 0.0f), Quat(), NULL, NULL, 0);
		anchor.invMass = 0.0f;

		g_buffers->rigidBodies.push_back(anchor);	


		NvFlexRigidPose prevJoint = NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), localFrame);

		for (int i=0; i < 16; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(i*linkLength*2.0f + linkLength, height, 0.0f), Quat(), &shape, &density, 1);
			
			/*
			(Matrix33&)body.inertia += Matrix33::Identity()*0.1f;

			bool success;
			(Matrix33&)body.invInertia = Inverse((Matrix33&)body.inertia, success);
			*/


			NvFlexRigidJoint joint;				
			NvFlexMakeSphericalJoint(&joint, bodyIndex-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), Quat()));

			prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), localFrame);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);

		}

		// attach end weight
		if (1)
		{
			NvFlexRigidShape weight;
			NvFlexMakeRigidBoxShape(&weight, g_buffers->rigidBodies.size(), 0.25f, 0.25f, 0.25f, NvFlexMakeRigidPose(0,0));

			Transform prevPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies.back(), (NvFlexRigidPose*)&prevPose);
			
			Transform endLink = prevPose*(Transform&)prevJoint;
			endLink.p.x += 0.025f + 0.05f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, endLink.p, endLink.q, &weight, &density, 1);

			// some initial motion 
			//(Vec3&)body.linearVel = Vec3(0.0f, 0.0f, -1.0f);

			NvFlexRigidJoint joint;				
			NvFlexMakeSphericalJoint(&joint, weight.body-1, weight.body, prevJoint, NvFlexMakeRigidPose(Vec3(-0.075f, 0.0f, 0.0f), Quat()));

			/*
			// limit the platforms movement
			joint.modes[eNvFlexRigidJointAxisSwing1] = eNvFlexRigidJointModeLimit;
			joint.modes[eNvFlexRigidJointAxisSwing2] = eNvFlexRigidJointModeLimit;
			joint.lowerLimits[eNvFlexRigidJointAxisSwing1] = -DegToRad(30.0f);
			joint.upperLimits[eNvFlexRigidJointAxisSwing1] = DegToRad(30.0f);
			joint.lowerLimits[eNvFlexRigidJointAxisSwing2] = -DegToRad(30.0f);
			joint.upperLimits[eNvFlexRigidJointAxisSwing2] = DegToRad(30.0f);
			*/

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(weight);
			g_buffers->rigidJoints.push_back(joint);
		}

		g_numSubsteps = 1;
		g_params.numIterations = 5;
		g_params.numInnerIterations = 40;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
	}
};


class RigidCosserat : public Scene
{
public:

	float twist;

	RigidCosserat()
	{
		int segments = 64;

		const int linkMaterial = AddRenderMaterial(Vec3(0.805f, 0.702f, 0.401f));

		const float linkLength = 0.0125f;
		const float linkWidth = 0.01f;

		const float density = 1000.0f;

		Vec3 startPos = Vec3(-0.3f, 1.0f, 0.45f);

		NvFlexRigidPose prevJoint;

		for (int i=0; i < segments; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(linkMaterial);
					
			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, startPos + Vec3(i*linkLength*2.0f + linkLength, 0.0f, 0.0f), Quat(), &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

			if (i > 0)
			{
				NvFlexRigidJoint joint;				
				NvFlexMakeFixedJoint(&joint, bodyIndex-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), Quat()));
					
				const float bendingCompliance = 1.e-1f;
				const float torsionCompliance = 1.e-3f;

				joint.compliance[eNvFlexRigidJointAxisTwist] = torsionCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing1] = bendingCompliance;
				joint.compliance[eNvFlexRigidJointAxisSwing2] = bendingCompliance;

				g_buffers->rigidJoints.push_back(joint);
			}

			prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), Quat());
		}

		/*
		// fix ends
		NvFlexRigidJoint leftJoint;
		NvFlexMakeFixedJoint(&leftJoint, -1, 0, NvFlexMakeRigidPose(startPos, Quat()), NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), Quat()));

		NvFlexRigidJoint rightJoint;
		NvFlexMakeFixedJoint(&rightJoint, -1, g_buffers->rigidBodies.size()-1,  NvFlexMakeRigidPose(startPos + Vec3(segments*linkLength*2.0f, 0.0f, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), Quat()));

		g_buffers->rigidJoints.push_back(leftJoint);
		g_buffers->rigidJoints.push_back(rightJoint);
		*/
		g_buffers->rigidBodies[0].invMass = 0.0f;
		(Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();

		g_buffers->rigidBodies.back().invMass = 0.0f;
		(Matrix33&)g_buffers->rigidBodies.back().invInertia = Matrix33();

		g_numSubsteps = 4;
		g_params.numIterations = 40;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;

		twist = 0.0f;
	}

	void DoGui()
	{	
//		imguiSlider("Twist", &g_buffers->rigidJoints[0].targets[eNvFlexRigidJointAxisTwist], 0.0f, kPi*10.0f, 0.001f);
		
		imguiSlider("Separation", &g_buffers->rigidBodies[0].com[0], -0.5f, 1.0f, 0.001f);

		Quat q = QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), twist);
		imguiSlider("Twist", &twist, 0.0f, kPi*20.0f, 0.001f);

		(Quat&)g_buffers->rigidBodies[0].theta = q;
	}
};


class RigidPrismaticJoint : public Scene
{
public:

	RigidPrismaticJoint()
	{
		const float density = 1000.0f;
	
		{
			// axis
			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, 0, 1.0f, 0.05f, 0.05f, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 1.0f, 0.0f), Quat(), &shape, &density, 1);
		
			//body.torque[2] = -0.1f;

			NvFlexRigidJoint joint;				
			NvFlexMakeHingeJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(0.0f, 1.0f, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(), Quat()), eNvFlexRigidJointAxisSwing2, -DegToRad(60.0f), DegToRad(60.0f));
		
			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);
		}

		// attach slider
		{
			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, 1, 0.1f, 0.1f, 0.1f, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.2f, 1.0f, 0.0f), Quat(), &shape, &density, 1);
		
			NvFlexRigidJoint joint;				
			NvFlexMakePrismaticJoint(&joint, 0, 1, NvFlexMakeRigidPose(Vec3(), Quat()), NvFlexMakeRigidPose(Vec3(), Quat()), eNvFlexRigidJointAxisX, -1.0f, 1.0f);
		
			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);
		}

		g_numSubsteps = 4;
		g_params.numIterations = 20;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_pause = true;
	}

	virtual void Update()
	{
		(Vec3&)g_buffers->rigidBodies[0].angularVel = Vec3(0.0f, 0.0f, sinf(g_frame*g_dt));
	}
};


class RigidMobile : public Scene
{
public:

	RigidMobile()
	{
		const float density = 1000.0f;

		BuildRecursive(-1, Vec3(0.0f, 3.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), 0, 1.0f);

		g_numSubsteps = 4;
		g_params.numIterations = 20;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_drawJoints = true;

		g_pause = true;
	}

	void BuildRecursive(int parent, Vec3 parentPos, Vec3 parentOffset, int depth, float spacing)
	{
		if (depth > 5)
			return;

		int bodyIndex = g_buffers->rigidBodies.size();

		NvFlexRigidShape shape;
		NvFlexMakeRigidBoxShape(&shape, bodyIndex, spacing*0.5f, spacing*0.5f, spacing*0.5f, NvFlexMakeRigidPose(0,0));

		float density = 1000.0f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, parentPos + parentOffset, Quat(), &shape, &density, 1);
	
		// joint to parent
		NvFlexRigidJoint joint;
		NvFlexMakeSphericalJoint(&joint, parent, bodyIndex, NvFlexMakeRigidPose(depth==0?parentPos:Vec3(0.0f, 0.5f*parentOffset.y, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(0.0f, -0.5f*parentOffset.y, 0.0f), Quat()));

		joint.targets[0] = parentOffset.x;
		joint.targets[2] = parentOffset.z;		

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);
		g_buffers->rigidJoints.push_back(joint);

		// left
		BuildRecursive(bodyIndex, parentPos + parentOffset, Vec3(spacing, -spacing, 0.0f), depth + 1, spacing*0.5f);
		// right
		BuildRecursive(bodyIndex, parentPos + parentOffset, Vec3(-spacing, -spacing, 0.0f), depth + 1, spacing*0.5f);
		// up
		BuildRecursive(bodyIndex, parentPos + parentOffset, Vec3(0.0f, -spacing, -spacing), depth + 1, spacing*0.5f);
		// down
		BuildRecursive(bodyIndex, parentPos + parentOffset, Vec3(0.0f, -spacing, spacing), depth + 1, spacing*0.5f);

	}

	void Update()
	{
		(Vec3&)g_buffers->rigidBodies[0].angularVel = Vec3(0.0f, sinf(g_frame*g_dt), 0.0f);
	}
};


class RigidSpring : public Scene
{
public:

	RigidSpring()
	{
		NvFlexRigidShape box;
		NvFlexMakeRigidBoxShape(&box, 0, 0.25f, 0.25f, 0.25f, NvFlexMakeRigidPose(0,0));

		float density = 1000.0f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 1.0f, 0.0f), Quat(), &box, &density, 1);

		NvFlexRigidJoint joint;
		NvFlexMakeFixedJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(0.0f, 1.0f, 0.0f), Quat()), NvFlexMakeRigidPose(0,0));
		
		// set a target offset of 1m
		joint.targets[eNvFlexRigidJointAxisX] = 1.0f;
		joint.compliance[eNvFlexRigidJointAxisX] = 1.e-2f; // 100N/m
		joint.damping[eNvFlexRigidJointAxisX] = 10.0f;
		joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;

		g_params.gravity[1] = 0.0f;

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(box);
		g_buffers->rigidJoints.push_back(joint);

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_drawJoints = true;

		g_pause = true;
	}

	virtual void Update()
	{
		//printf("%f %f\n", g_buffers->rigidJoints[0].lambda[0], g_buffers->rigidBodies[0].com[0]);
	}

	virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }
};

class RigidSpringHard : public Scene
{
public:

	RigidSpringHard()
	{
		NvFlexRigidShape box;
		NvFlexMakeRigidBoxShape(&box, 0, 0.125f, 0.125f, 0.125f, NvFlexMakeRigidPose(0,0));

		float density = 1000.0f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(1.0f, 1.0f, 0.0f), Quat(), &box, &density, 1);

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(box);


		{
			NvFlexRigidJoint joint;
			NvFlexMakeFixedJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(0.0f, 1.0f, 0.0f), Quat()), NvFlexMakeRigidPose(0,0));
			
			// set a target offset of 1m
			joint.targets[eNvFlexRigidJointAxisX] = 0.0f;
			joint.compliance[eNvFlexRigidJointAxisX] = 0.0f; 
			joint.damping[eNvFlexRigidJointAxisX] = 0.0f;
			joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;

			g_buffers->rigidJoints.push_back(joint);

		}

		{
			NvFlexRigidJoint joint;
			NvFlexMakeFixedJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(2.0f, 1.0f, 0.0f), Quat()), NvFlexMakeRigidPose(0,0));
			
			// set a target offset of 1m
			joint.targets[eNvFlexRigidJointAxisX] = 0.0f;
			joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f; 
			joint.damping[eNvFlexRigidJointAxisX] = 0.0f;
			joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;

			g_buffers->rigidJoints.push_back(joint);

		}

		g_params.systemRegularization = 1.e-7f;
		g_params.gravity[1] = 0.0f;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_drawJoints = true;

		g_pause = true;
	}

	virtual void Update()
	{
		//printf("%f %f\n", g_buffers->rigidJoints[0].lambda[0], g_buffers->rigidBodies[0].com[0]);
	}

	virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }
};





class RigidOverlap : public Scene
{
public:
	
	RigidOverlap()
	{
		NvFlexTriangleMeshId boxMesh = CreateTriangleMesh(ImportMesh("../../data/box.ply"));

		NvFlexRigidShape staticShape;
		//NvFlexMakeRigidBoxShape(&staticShape, -1, 0.5f, 0.5f, 0.5f, NvFlexMakeRigidPose(0,0));
		NvFlexMakeRigidTriangleMeshShape(&staticShape, -1, boxMesh, NvFlexMakeRigidPose(0,0), 1.0f, 1.0f, 1.0f);
		staticShape.filter = 0;

		g_buffers->rigidShapes.push_back(staticShape);

		NvFlexRigidShape boxShape;	
		//NvFlexMakeRigidBoxShape(&boxShape, 0, 0.5f, 0.1f, 0.1f, NvFlexMakeRigidPose(0,0));
		NvFlexMakeRigidTriangleMeshShape(&boxShape, 0, boxMesh, NvFlexMakeRigidPose(0,0), 1.0f, 0.1f, 0.1f);
		boxShape.filter = 0;

		float density = 1000.0f;
		
		NvFlexRigidBody boxBody;
		NvFlexMakeRigidBody(g_flexLib, &boxBody, Vec3(0.5f, 0.5f, 0.0f) + Normalize(Vec3(1.0f, 1.0f, 0.0f))*0.11f, QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(-45.0f)), &boxShape, &density, 1);
		
		g_buffers->rigidBodies.push_back(boxBody);
		g_buffers->rigidShapes.push_back(boxShape);

		//g_params.collisionDistance = FLT_MIN;

		printf("mass: %f\n", boxBody.mass);

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_params.numIterations = 20;

		g_pause = true;
	}

};


class RigidCollision : public Scene
{
public:	
	
	RigidCollision()
	{
		NvFlexRigidShape shape;	
		//NvFlexMakeRigidCapsuleShape(&shape, 0, 0.1f, 0.25f, NvFlexMakeRigidPose(0,0));
		//NvFlexMakeRigidSphereShape(&shape, 0, 0.1f, NvFlexMakeRigidPose(0,0));
		//NvFlexMakeRigidBoxShape(&shape, 0, 0.25f, 0.1f, 0.2f, NvFlexMakeRigidPose(0,0));

		Mesh* m = ImportMesh("../../data/cylinder.obj");
		m->Normalize();

		NvFlexTriangleMeshId mesh = CreateTriangleMesh(m);
		NvFlexMakeRigidTriangleMeshShape(&shape, 0, mesh, NvFlexMakeRigidPose(Vec3(0.0f, 0.6f, 0.0f), Quat()), 0.5f, 0.5f, 0.5f);

		float density = 1000.0f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

		//(Vec3&)body.linearVel = Vec3(0.2f, 0.0f, 0.2f);
		//(Vec3&)body.angularVel = Vec3(0.0f, 10.0f, 2.0f);

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);

		//g_params.collisionDistance = FLT_MIN;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_params.numIterations = 20;

		g_pause = true;
	}
};

class RigidJointLimits : public Scene
{
public:

	RigidJointLimits()
	{
		NvFlexRigidShape shape;
		NvFlexMakeRigidBoxShape(&shape, 0, 0.5f, 0.2f, 0.3f, NvFlexMakeRigidPose(0,0));
		//NvFlexMakeRigidCapsuleShape(&shape, 0, 0.1f, 0.5f, NvFlexMakeRigidPose(0,0));

		float density = 100.0f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.6f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f), &shape, &density, 1);

		NvFlexRigidJoint joint;
		//NvFlexMakeSphericalJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(0.0f, 0.1f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f)), NvFlexMakeRigidPose(Vec3(-0.5f, 0.0f, 0.0f), Quat()));
		NvFlexMakeFixedJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(0.0f, 0.1f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f)), NvFlexMakeRigidPose(Vec3(-0.5f, 0.0f, 0.0f), Quat()));

		joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeFree;
		joint.damping[eNvFlexRigidJointAxisTwist] = 10.0f;
		joint.compliance[eNvFlexRigidJointAxisTwist] = 0.001f;

		//body.angularVel[1] = 0.5f;
		body.angularDamping = 0.0f;

		//joint.modes[eNvFlexRigidJointAxisSwing1] = eNvFlexRigidJointModeLimit;
		//joint.modes[eNvFlexRigidJointAxisSwing2] = eNvFlexRigidJointModeLimit;

/*
		joint.lowerLimits[eNvFlexRigidJointAxisSwing1] = -DegToRad(30.0f);
		joint.upperLimits[eNvFlexRigidJointAxisSwing1] =  DegToRad(30.0f);

		joint.lowerLimits[eNvFlexRigidJointAxisSwing2] = -DegToRad(30.0f);
		joint.upperLimits[eNvFlexRigidJointAxisSwing2] =  DegToRad(30.0f);
*/
		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);
		g_buffers->rigidJoints.push_back(joint);

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_numSubsteps = 2;
		g_params.numIterations = 20;
		g_params.shapeCollisionMargin = 0.01f;

		g_pause = true;
	}

	void Update()
	{
		//g_buffers->rigidJoints.back().targets[eNvFlexRigidJointAxisTwist] += 0.01f;

		g_buffers->rigidBodies.back().torque[1] = 4.0f;

	}
};


class RigidStack : public Scene
{
public:
	
	RigidStack(const char* filename) 
	{
		const int height = 6;

		Mesh* mesh = ImportMesh(filename);
		mesh->Normalize(0.5f);

		NvFlexTriangleMeshId meshId = CreateTriangleMesh(mesh);
		
		//NvFlexRigidShape staticShape;
		//NvFlexMakeRigidTriangleMeshShape(&staticShape, -1, meshId, NvFlexMakeRigidPose(0,0), 1.0f, 1.0f, 1.0f);
		//g_buffers->rigidShapes.push_back(staticShape);
		int dx = 2;
		int dy = 2;

		for (int x=0; x < dx; ++x)
		{
			for (int y=0; y < dy; ++y)
			{
				Vec3 offset(x*1.0f, 1.0f, y*1.0f);

				for (int i=0; i < height; ++i)
				{
					//float offset = 1.0f;

					NvFlexRigidShape shape;	
					//NvFlexMakeRigidCapsuleShape(&shape, i, 0.1f, 0.25f, NvFlexMakeRigidPose(0,0));
					//NvFlexMakeRigidSphereShape(&shape, i, 0.1f, NvFlexMakeRigidPose(0,0));
					//NvFlexMakeRigidBoxShape(&shape, i, 0.25f, 0.1f, 0.2f, NvFlexMakeRigidPose(0,0));
					NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), meshId, NvFlexMakeRigidPose(0,0), 1.0f, 1.0f, 1.0f);

					// collide all shapes
					shape.filter = 0;
					shape.material.friction = 0.7f;

					float density = 1000.0f;

					NvFlexRigidBody body;
					NvFlexMakeRigidBody(g_flexLib, &body, offset + Vec3(0.0f, i*0.6f, 0.0f), Quat(), &shape, &density, 1);

					g_buffers->rigidBodies.push_back(body);
					g_buffers->rigidShapes.push_back(shape);
				}
			}
		}
		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_numSubsteps = 2;
		g_params.numIterations = 30;
		g_params.shapeCollisionMargin = 0.05f;

		g_pause = true;
	}
};


class RigidCapsuleStack : public Scene
{
public:
	
	RigidCapsuleStack()
	{
		const int height = 8;
		const float mu = 0.6f;

		Quat rotation;

		NvFlexRigidShape staticShape;

		NvFlexMakeRigidBoxShape(&staticShape, -1, 1.1f, 0.5f, 1.0f, NvFlexMakeRigidPose(Vec3(-1.0f, 0.5f, -1.0f), Quat()));
		staticShape.filter = 0;

		g_buffers->rigidShapes.push_back(staticShape);

		RenderMaterial mat;
		mat.frontColor = Vec3(0.1f);
		mat.backColor = Vec3(0.7f);
		mat.gridScale = 12.0f;
		

		for (int i=0; i < height; ++i)
		{
			NvFlexRigidShape shape0, shape1;	
			NvFlexMakeRigidCapsuleShape(&shape0, i*2+0, 0.1f, 0.6f, NvFlexMakeRigidPose(0,0));
			NvFlexMakeRigidCapsuleShape(&shape1, i*2+1, 0.1f, 0.6f, NvFlexMakeRigidPose(0,0));

			// collide all shapes
			shape0.filter = 0;
			shape0.material.friction = mu;

			shape1.filter = 0;
			shape1.material.friction = mu;

			Vec3 offset = rotation*Vec3(0.0f, 0.0f, 0.4f);

			float density = 1000.0f;

			NvFlexRigidBody body0, body1;
			NvFlexMakeRigidBody(g_flexLib, &body0, offset + Vec3(-0.9f, 1.15f + i*0.2f, -0.9f), rotation, &shape0, &density, 1);
			NvFlexMakeRigidBody(g_flexLib, &body1, -offset + Vec3(-0.9f, 1.15f + i*0.2f, -0.9f), rotation, &shape1, &density, 1);

			// to avoid missing collisions
			body0.maxAngularVelocity = 4.0f;
			body1.maxAngularVelocity = 4.0f;

			g_buffers->rigidBodies.push_back(body0);
			g_buffers->rigidBodies.push_back(body1);

			g_buffers->rigidShapes.push_back(shape0);
			g_buffers->rigidShapes.push_back(shape1);

			rotation = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), kPi/2.0f)*rotation;
		}

		// weight
		if (1)
		{
			const float size = 1.0f;

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, g_buffers->rigidBodies.size(), size, size, size, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[1%8]), 0.1f));
			shape.material.friction = mu;
			shape.thickness = 0.05f;

			const float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-0.9f, 2.75f + size, -0.9f), Quat(), &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

		}

		printf("capsule mass: %f\n", g_buffers->rigidBodies[0].mass);
		printf("box mass: %f\n", g_buffers->rigidBodies.back().mass);

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_numSubsteps = 2;
		g_params.numIterations = 20;
		g_params.shapeCollisionMargin = 0.05f;

		g_pause = true;
	}
};


class RigidFriction : public Scene
{
public:
	
	const float g = 9.81f;
	const float v0 = 2.5f;
	const float x0 = 0.0f;

	RigidFriction()
	{

		for (int i=0; i < 3; ++ i)
		{
			// analytic params
			const float mu = 0.1f*(i+1);

			// analytic solution
			const float tstop = v0 / (mu*g);
			const float xstop = x0 + v0*tstop - 0.5f*mu*g*tstop*tstop;

			const float radius = 0.5f;

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, i, 0.5f, 0.5f, 0.5f, NvFlexMakeRigidPose(0,0));
			
			// friction
			shape.material.friction = mu;

			float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.51f, i*1.25f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

			(Vec3&)body.linearVel = Vec3(v0, 0.0f, 0.0f);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
		}

		g_params.collisionDistance = FLT_MIN;
		g_params.shapeCollisionMargin = 0.01f;
		g_params.relaxationFactor = 1.f;

		g_numSubsteps = 1;

		g_params.numIterations = 8;
		g_params.solverType = eNvFlexSolverLDLT; 
		g_params.relaxationFactor = 0.5f; 

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_params.gravity[1] = -g;

		g_pause = true;
	}

	virtual void Draw(int pass)
	{
		for (int i=0; i < 3; ++ i)
		{
			// analytic params
			const float mu = 0.1f*(i+1);

			// analytic solution
			const float tstop = v0 / (mu*g);
			const float xstop = x0 + v0*tstop - 0.5f*mu*g*tstop*tstop;

			BeginLines();
			DrawLine(Vec3(xstop, 0.0f, i*1.25f), Vec3(xstop, 1.0f, i*1.25f), Vec4(0.0f, 1.0f, 0.0f, 0.0f));
			EndLines();
		}
	}
};


class RigidFrictionAniso : public Scene
{
public:
	
	const float g = 9.81f;
	const float v0 = 2.5f;	
	const float x0 = 0.0f;
		
	const float mu2 = 0.1f;
	const float mu1 = mu2/sqrtf(0.25f);

	const float numSpheres = 12;

	RigidFrictionAniso()
	{
	
		for (int i=0; i < numSpheres; ++i)
		{
			const float radius = 0.5f;

			NvFlexRigidShape shape;
			//NvFlexMakeRigidBoxShape(&shape, i, 0.5f, 0.5f, 0.5f, NvFlexMakeRigidPose(0,0));
			NvFlexMakeRigidSphereShape(&shape, i, 0.5f, NvFlexMakeRigidPose(0,0));

			shape.filter = 1;
			shape.material.friction = mu2;

			float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.5001f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

			float theta = i*(k2Pi/numSpheres);
			Vec3 dir = Normalize(Vec3(cosf(theta), 0.0f, sinf(theta)));

			(Vec3&)body.linearVel = dir*v0;

			(Matrix33&)body.invInertia = Matrix33();
			(Matrix33&)body.inertia = Matrix33();

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

		}

		g_params.collisionDistance = FLT_MIN;
		g_params.shapeCollisionMargin = 0.01f;
		g_params.relaxationFactor = 1.f;

		g_numSubsteps = 1;

		g_params.numIterations = 8;
		g_params.solverType = eNvFlexSolverLDLT; 
		g_params.relaxationFactor = 0.5f; 

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_params.gravity[1] = -g;

		g_pause = true;
	}
	
};



class RigidTorsionFriction : public Scene
{
public:
	
	RigidTorsionFriction()
	{
		const float radius = 0.5f;

		// render material
		RenderMaterial mat;
		mat.frontColor = Vec3(0.1f);
		mat.backColor = Vec3(0.7f);
		mat.gridScale = 12.0f;

		const int renderMaterial = AddRenderMaterial(mat);

		for (int i=0; i < 3; ++i)
		{
			NvFlexRigidShape shape;
			NvFlexMakeRigidSphereShape(&shape, i, radius, NvFlexMakeRigidPose(0,0));

			// friction
			shape.material.torsionFriction = 0.01f + 0.01f*i;
			shape.material.rollingFriction = 0.0f;
			shape.material.friction = 0.0f;
			shape.user = UnionCast<void*>(renderMaterial);

			const float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(radius*2.2f*i, radius, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

			(Vec3&)body.linearVel = Vec3(0.0f, 0.0f, 0.0f);
			(Vec3&)body.angularVel = Vec3(0.0f, -5.0f, 0.0f);

			body.angularDamping = 0.0f;
			body.linearDamping = 0.0f;

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
		}

		g_params.numIterations = 10;
		g_params.frictionMode = eNvFlexFrictionModeFull;	// enable torsional friction in Newton solver

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(4.0f);

		g_pause = true;
	}
};


class RigidRollingFriction : public Scene
{
public:
	
	RigidRollingFriction()
	{
		const float radius = 0.5f;

		// render material
		RenderMaterial mat;
		mat.frontColor = Vec3(0.1f);
		mat.backColor = Vec3(0.7f);
		mat.gridScale = 12.0f;

		const int renderMaterial = AddRenderMaterial(mat);

		for (int i=0; i < 3; ++i)
		{
			NvFlexRigidShape shape;
			NvFlexMakeRigidSphereShape(&shape, i, radius, NvFlexMakeRigidPose(0,0));

			// friction
			shape.material.torsionFriction = 0.0f;
			shape.material.rollingFriction = (0.01f + 0.1f*i);
			shape.material.friction = 0.8f;
			shape.user = UnionCast<void*>(renderMaterial);

			float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, radius, radius*2.2f*i), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);
			
			(Vec3&)body.linearVel = Vec3(5.0f, 0.0f, 0.0f);
			(Vec3&)body.angularVel = Vec3(0.0f, 0.0f, 0.0f);

			body.linearDamping = 0.0f;
			body.angularDamping = 0.0f;
			
			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
		}

		g_params.numIterations = 10;

		g_sceneLower = Vec3(0.0f);
		g_sceneUpper = Vec3(10.0f, 2.0f, 8.0f);

		g_pause = true;
	}
};


class RigidComplementarity1 : public Scene
{
public:
	
	RigidComplementarity1()
	{
		const float radius = 0.5f;
		
		NvFlexRigidShape shape;
		NvFlexMakeRigidSphereShape(&shape, 0, radius, NvFlexMakeRigidPose(0,0));

		const float density = 1000.0f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, radius-0.005f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

		body.angularDamping = 0.0f;
		body.linearDamping = 0.0f;

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);

		// add a joint pulling the sphere upwards
		NvFlexRigidJoint joint;
		NvFlexMakeFixedJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(0.0f, 2.0f, 0.0f), Quat()), NvFlexMakeRigidPose(0,0));
		
		joint.compliance[eNvFlexRigidJointAxisY] = 1.e-2f;

		g_buffers->rigidJoints.push_back(joint);

		g_numSubsteps = 1;
		g_params.numIterations = 10;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(4.0f);

		g_pause = true;

		g_drawJoints = true;
	}
};


class RigidComplementarity : public Scene
{
public:
	
	RigidComplementarity()
	{
		// box
		NvFlexRigidShape staticShape;

		NvFlexMakeRigidBoxShape(&staticShape, -1, 1.1f, 0.5f, 1.0f, NvFlexMakeRigidPose(Vec3(-1.0f, 0.5f, -1.0f), Quat()));
		staticShape.filter = 0;

		g_buffers->rigidShapes.push_back(staticShape);

		const float radius = 0.1f;

		NvFlexRigidShape shape;
		NvFlexMakeRigidCapsuleShape(&shape, 0, radius, radius*4.0f, NvFlexMakeRigidPose(0,0));

		const float density = 1000.0f;

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.25f, 1.0f + radius - 0.005f, -0.5f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

		body.angularDamping = 0.0f;
		body.linearDamping = 0.0f;

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);

		g_numSubsteps = 1;
		g_params.numIterations = 10;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(4.0f);

		g_pause = true;

		g_drawJoints = true;
	}
};

class RigidTippeTop : public Scene
{
public:

    RigidTippeTop()
    {
        float scale = 1.0f;

		const float ballRadius = 0.025f*scale;	
		const float stickRadius = 0.01f*scale;
		const float stickLength = 0.005f*scale;

		const float mu = 0.6f;

		NvFlexRigidShape ballShape;
		NvFlexMakeRigidSphereShape(&ballShape, 0, ballRadius, NvFlexMakeRigidPose(0,0));
		ballShape.material.friction = mu;

		NvFlexRigidShape capsuleShape;
		NvFlexMakeRigidCapsuleShape(&capsuleShape, 0, stickRadius, stickLength, NvFlexMakeRigidPose(Vec3(ballRadius + stickLength, 0.0f, 0.0f), Quat()));		
		capsuleShape.material.friction = mu;

		NvFlexRigidShape shapes[2] = { ballShape, capsuleShape };
		const float densities[2] = { 1000.0f, 1000.0f };

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, ballRadius, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.25f), shapes, densities, 2);

		(Vec3&)body.angularVel = Vec3(0.0f, 150.0f, 0.0f);
		(Vec3&)body.linearVel = Vec3(0.1f, 0.0f, 0.2f);
		body.maxAngularVelocity = 1000.0f;
		body.angularDamping = 0.00f;
		body.linearDamping = 0.0f;

		(Vec3&)body.com -= Vec3(0.0f, -0.01020f, 0.0f);
		(Vec3&)body.origin += Vec3(0.0f, -0.01020f, 0.0f);

		Matrix33 I = Matrix33::Identity()*3.75e-6f;
		bool success;

		(Matrix33&)body.inertia = I;
		(Matrix33&)body.invInertia = Inverse(I, success);
		
		body.mass = 0.0015f;
		body.invMass = 1.0f/body.mass;

		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(ballShape);
		g_buffers->rigidShapes.push_back(capsuleShape);

		/*
		PxShape* base = gPhysics->createShape(PxSphereGeometry(ballRadius), *material, true);
		PxShape* capsule = gPhysics->createShape(PxCapsuleGeometry(stickRadius, stickLength), *material, true);
		capsule->setLocalPose(PxTransform(PxVec3(0.0f, ballRadius + stickLength, 0.0f), PxQuat(PxPi*0.5f, PxVec3(0.0f, 0.0f, 1.0f))));
	
		PxRigidDynamic* body = gPhysics->createRigidDynamic(PxTransform(PxVec3(0.0f, ballRadius, 0.0f), PxQuat(PxPi*0.25f, PxVec3(0.0f, 0.0f, 1.0f))));//PxQuat(-0.126f, -0.316f, 0.065f, -0.938f)));
		body->attachShape(*base);
		body->attachShape(*capsule);

		body->setAngularVelocity(PxVec3(0.0f, 150.0f, 0.0f));
		body->setLinearVelocity(PxVec3(0.1f, 0.0f, 0.2f));

		body->setLinearDamping(0.0f);
		body->setAngularDamping(0.001f);
		body->setMaxAngularVelocity(1000.0f);

		body->setMass(0.0015f);
		body->setMassSpaceInertiaTensor(PxVec3(3.75e-6f));
		body->setCMassLocalPose(PxTransform(0.0f, -0.0048f, 0.0f));	
	
		gScene->addActor(*body);
		gScene->setGravity(PxVec3(0.0f, -9.8f, 0.0f));

		*/

        g_sceneLower = Vec3(-0.5f);
        g_sceneUpper = Vec3(0.5f);

        g_numSubsteps = 4;
        g_params.numIterations = 20;

        g_pause = true;
    }
};

class RigidHand : public Scene
{
public:

	RigidHand()
	{
		vector<pair<int, NvFlexRigidJointAxis>> ctrls;
		vector<float> motors;

		MJCFImporter mj("../../data/MPL/include_MPL.xml");
		mj.AddPhysicsEntities(Transform(Vec3(0.0f, 1.0f, 0.0f), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi * 0.5f)), 
					ctrls, motors, false, true);

	//	MJCFImporter mj("../../data/MPL/include_MPL.xml");
	//  mj.AddPhysicsEntities(Transform(Vec3(0.0f, 1.0f, 0.0f), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi * 0.5f)), 
	//				ctrls, motors, false);

	//	MJCFImporter mj("../../data/robotics/hand/hand.xml");
    //  mj.AddPhysicsEntities(Transform(Vec3(0.0f, 1.0f, 0.0f), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), kPi * 0.5f)),
	//				ctrls, motors, false);

		NvFlexRigidPose pose;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[1], &pose);

		NvFlexRigidJoint anchor;
		NvFlexMakeFixedJoint(&anchor, -1, 1, pose, NvFlexMakeRigidPose(0,0));
	
		g_buffers->rigidJoints.push_back(anchor);

		g_numSubsteps = 4;
		g_params.numIterations = 30;
		g_params.dynamicFriction = 0.2f;

		g_drawPoints = false;

		g_sceneLower = Vec3(-0.5f);
		g_sceneUpper = Vec3(0.5f, 1.f, 0.5f);

		g_pause = true;
	}

	virtual void DoGui()
	{
		/*
		for (int i = 0; i < motors.size(); i++)
		{
			char tmp[30];

			imguiSlider(motors[i].c_str(), &targetP[i], -kPi + 0.01f, kPi - 0.01f, 0.0001f);
			g_buffers->rigidJoints[p_urdf->activeJointNameMap[motors[i]]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
			g_buffers->rigidJoints[p_urdf->activeJointNameMap[motors[i]]].targets[eNvFlexRigidJointAxisTwist] = targetP[i];
		}
		*/
	}
};

class RigidHeavyStack : public Scene
{
public:

	RigidHeavyStack()
	{

		const float baseSize = 0.1f;
		const float growthFactor = 2.0f;

		const int n = 5;

		float size = baseSize;
		float height = 0.01f;
		
		for (int i=0; i < 5; ++i)		
		{
			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, i, size, size, size, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[i%8]), 0.1f));
			shape.material.friction = 0.5f;

			const float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, height + size, 0.0f), Quat(), &shape, &density, 1);

			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidBodies.push_back(body);
	
			height += 2.0f*(size + 0.25f);

			size *= growthFactor;

			printf("body %d mass: %f\n", i, body.mass);
		}


		g_pause = true;

		g_sceneLower = Vec3(-4.0f);
		g_sceneUpper = Vec3(4.0f);

		g_params.numIterations = 40;
		g_params.solverType = eNvFlexSolverPCR;

		g_params.relaxationFactor = 0.75f;
	}

	virtual void CenterCamera()
	{
		g_camPos = Vec3(0.0f, 2.372088f, 14.998753f);
		g_camAngle = Vec3(-0.008727f, 0.068068f, 0.000000f);
	}

};

class RigidHeavyButton : public Scene
{
public:

	RigidHeavyButton()
	{
		// button
		{
			const float size = 0.1f;

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, 0, size, size*0.1f, size, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[0%8]), 0.1f));

			const float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.25f, 0.0f), Quat(), &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
		}

		// weight
		if (1)
		{
			const float size = 0.5f;

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, 1, size, size, size, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[1%8]), 0.1f));

			const float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 0.5f + size, 0.0f), Quat(), &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

		}

		// spring
		NvFlexRigidJoint joint;
		NvFlexMakeFixedJoint(&joint, -1, 0, NvFlexMakeRigidPose(0,0), NvFlexMakeRigidPose(0,0));

		joint.compliance[eNvFlexRigidJointAxisY] = 1.e-5f;
		joint.targets[eNvFlexRigidJointAxisY] = 0.25f;

		g_buffers->rigidJoints.push_back(joint);

		g_pause = true;
		
		g_drawJoints = true;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);
		
	}
};

class RigidGranularCompression : public Scene
{
public:

	RigidGranularCompression()
	{
		const int n = 512;
		const float radius = 0.05f;

		std::vector<Vec3> points(n);

		int created = PoissonSampleBox3D(Vec3(radius*2.0f), Vec3(1.0f-radius*2.0f), radius*2.0f, &points[0], n, 2000);
		points.resize(created);

		for (int i=0; i < int(points.size()); ++i)
		{
			NvFlexRigidShape shape;
			NvFlexMakeRigidSphereShape(&shape, i, radius, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.material.friction = 0.8f;

			RenderMaterial mat;
			mat.frontColor = Vec3(g_colors[i%8]);
			mat.backColor = Vec3(0.7f);
			mat.gridScale = 10.0f;

			shape.user = UnionCast<void*>(AddRenderMaterial(mat));

			const float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, points[i], Quat(), &shape, &density, 1);
			body.angularDamping = 0.1f;			

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);	
		}

		// weight
		if (1)
		{
			const float size = 0.5;

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, g_buffers->rigidBodies.size(), size, size, size, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[1%8]), 0.1f));

			const float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(size, 1.0f + size, size), Quat(), &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

		}

		g_params.numPlanes = 5;

		g_pause = true;

		g_drawJoints = true;

		g_sceneLower = Vec3(0.0f);
		g_sceneUpper = Vec3(1.0f);
	}
};


class RigidArch : public Scene
{
public:

	RigidArch()
	{
	
		const float insideWidth = 1.0f;
		const float insideHeight = 2.0f;

		const float thickness = 0.5f;

		const int sections = 19;

		const float density = 1000.0f;

		Vec3 prevInside;
		Vec3 prevOutside;

		for (int i=0; i <= sections; ++i)
		{
			float t = SmoothStep(0.0f, 1.0f, float(i)/sections);

			Vec3 inside = EvaluateParabola(insideWidth, insideHeight, t);
			Vec3 outside = EvaluateParabola(insideWidth + thickness, insideHeight + thickness*0.5f, t);

			if (i > 0)
			{
				const Vec3 offset = Vec3(0.0f, 0.0f, -thickness);

				Vec3 vertices[] = 
				{
					prevInside,
					prevOutside,
					prevInside + offset,
					inside,
					outside,
					prevOutside + offset,
					inside + offset,
					outside + offset
				};

				const float density = 1000.0f;

				ConvexMeshBuilderNew builder;
				builder.BuildFromPoints(vertices, 8);

				Mesh m;
				m.m_positions.assign(builder.vertices.begin(), builder.vertices.end());
				
				for (int i=0; i < builder.triangles.size(); ++i)
				{
					m.m_indices.push_back(builder.triangles[i].vertices[0]);
					m.m_indices.push_back(builder.triangles[i].vertices[1]);
					m.m_indices.push_back(builder.triangles[i].vertices[2]);
				}

				m.CalculateFaceNormals();

				// convert to triangle mesh
				NvFlexTriangleMeshId triangleMesh = CreateTriangleMesh(&m, 0.01f);
			
				// create shape and body
				NvFlexRigidShape shape;
				NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), triangleMesh, NvFlexMakeRigidPose(0,0), 1.0f, 1.0f, 1.0f);
				shape.filter = 0;
				shape.material.friction = 0.6f;
				shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[i%7]), 0.5f));

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f), Quat(), &shape, &density, 1);
		
				g_buffers->rigidShapes.push_back(shape);
				g_buffers->rigidBodies.push_back(body);

			}

			prevInside = inside;
			prevOutside = outside;
		}

		g_pause = true;

		g_drawJoints = true;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_params.solverType = eNvFlexSolverLDLT;
		g_params.numIterations = 40;

		g_params.contactRegularization = 1.e-5f;
		g_params.relaxationFactor = 0.5f;
	}

	Vec3 EvaluateParabola(float width, float height, float t)
	{
		float a = -height/(width*width);
		float b = height;

		const float x = Lerp(-width, width, t);
		const float y = x*x*a + b;

		return Vec3(x, y, 0.0f);
	}

	bool spawn = false;

	void KeyDown(int key)
	{
		// create a rigid body
		if (key == 'f')
			spawn = true;
	}

	void Update()
	{
		if (spawn)
		{
			spawn = false;

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, g_buffers->rigidBodies.size(), 0.6f, 0.1f, 0.3, NvFlexMakeRigidPose(0, 0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[g_buffers->rigidBodies.size()%7]), 0.5f));

			float density = 1000.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 4.0f, -0.1f), Quat(), &shape, &density, 1);

			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidBodies.push_back(body);
		}
	}

	void Draw(int pass)
	{
		return;

		BeginLines();
				
		float insideWidth = 1.0f;
		float insideHeight = 2.0f;

		float thickness = 0.25f;

		int sections = 10;

		Vec3 prevInside;
		Vec3 prevOutside;

		for (int i=0; i <= sections; ++i)
		{
			float t = float(i)/sections;

			Vec3 inside = EvaluateParabola(insideWidth, insideHeight, t);
			Vec3 outside = EvaluateParabola(insideWidth + thickness, insideHeight + thickness, t);

			if (i > 0)
			{
				DrawLine(prevInside, inside, Vec4(1.0f));
				DrawLine(prevOutside, outside, Vec4(1.0f));
			}
			/*
			ConvexMeshBuilderNew convexBuilder;
			convexBuilder.AddVertex();
			convexBuilder.AddVertex();
			convexBuilder.AddVertex();
			convexBuilder.AddVertex();

			// convert to triangle mesh
			NvFlexTriangleMeshId triMesh = CreateTriangleMesh();

			// add rigid shape and bodies
			*/

			prevInside = inside;
			prevOutside = outside;
		}

		EndLines();
	}
};


class RigidTower : public Scene
{
public:

	RigidTower()
	{
		const int height = 10;

		const float scale = 1.0f;

		const float strutWidth = 0.15f*scale;
		const float strutHeight = 0.5f*scale;
		const float strutDepth = 0.15f*scale;

		const float plankWidth = 0.15f*scale;
		const float plankLength = 2.0f*scale;
		const float plankDepth = 0.4f*scale;

		float y = 0.0f;

		float thickness = 0.025f*scale;

		float mu = 0.4f;
		float density = 1000.0f;

		for (int i=0; i < height; ++i)
		{

			// struts
			{
				NvFlexRigidShape shape;
				NvFlexMakeRigidBoxShape(&shape, g_buffers->rigidBodies.size(), strutWidth*0.5f, strutHeight*0.5f, strutDepth*0.5f, NvFlexMakeRigidPose(0,0));
				shape.filter = 0;
				shape.material.friction = mu;
				shape.thickness = thickness;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-plankLength*0.4f, y + thickness + strutHeight*0.5f, 0.0f), Quat(), &shape, &density, 1);
		
				g_buffers->rigidShapes.push_back(shape);
				g_buffers->rigidBodies.push_back(body);
			}

			{
				NvFlexRigidShape shape;
				NvFlexMakeRigidBoxShape(&shape, g_buffers->rigidBodies.size(), strutWidth*0.5f, strutHeight*0.5f, strutDepth*0.5f, NvFlexMakeRigidPose(0,0));
				shape.filter = 0;
				shape.material.friction = mu;
				shape.thickness = thickness;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(plankLength*0.4f, y + thickness + strutHeight*0.5f, 0.0f), Quat(), &shape, &density, 1);
		
				g_buffers->rigidShapes.push_back(shape);
				g_buffers->rigidBodies.push_back(body);
			}


			// plank
			{
				NvFlexRigidShape shape;
				NvFlexMakeRigidBoxShape(&shape, g_buffers->rigidBodies.size(), plankLength*0.5f, plankWidth*0.5f, plankDepth*2.0f, NvFlexMakeRigidPose(0,0));
				shape.filter = 0;
				shape.material.friction = mu;
				shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(g_colors[3]), 0.5f));
				shape.thickness = thickness;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, y + strutHeight + plankWidth*0.5f + thickness*3.0f, 0.0f), Quat(), &shape, &density, 1);
		
				g_buffers->rigidShapes.push_back(shape);
				g_buffers->rigidBodies.push_back(body);
			}


			y += strutHeight + thickness*4.0f + plankWidth;
		}

		
		// disable rotations
		//for (int i=0; i < g_buffers->rigidBodies.size(); ++i)
			//(Matrix33&)g_buffers->rigidBodies[i].invInertia = Matrix33();
		

		g_pause = true;

		g_drawJoints = true;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f, 10.0f, 2.0f);

		
		g_numSubsteps = 1;

		g_params.solverType = eNvFlexSolverLDLT;		
		//g_params.solverType = eNvFlexSolverPCG1;
		//g_params.numInnerIterations = 100;
		//g_params.shapeCollisionMargin = thickness;

		g_params.numIterations = 100;
		g_params.numLineIterations = 0;

		g_params.relaxationFactor = 1.0f;		

		g_dt = 1.0f/60.0f;

	}

};

class RigidPendulum : public Scene
{
public:

	RigidPendulum()
	{
		const float density = 1000.0f;

		NvFlexRigidShape shape;
		NvFlexMakeRigidBoxShape(&shape, 0, 0.05f, 0.5f, 0.05f, NvFlexMakeRigidPose(0,0));

		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0f, 1.0f, 0.0f), Quat(), &shape, &density, 1);

		NvFlexRigidJoint joint;
		NvFlexMakeFixedJoint(&joint, -1, 0, NvFlexMakeRigidPose(Vec3(0.0f, 0.5f, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(0.0f, -0.5f, 0.0f), Quat()));

		joint.modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModeFree;


		g_buffers->rigidBodies.push_back(body);
		g_buffers->rigidShapes.push_back(shape);
		g_buffers->rigidJoints.push_back(joint);

		g_params.solverType = eNvFlexSolverPCR;

		g_sceneLower = -1.0f;
		g_sceneUpper = 1.0f;
	}

	virtual void Update()
	{		
        NvFlexRigidBody& body = g_buffers->rigidBodies[0];
        NvFlexRigidJoint& joint = g_buffers->rigidJoints[0];

        // clear control torques
        (Vec3&)body.torque = 0.0f;

        float linPos[3];
        float linRate[3];
        float angPos[3];
        float angRate[3];

        // compute joint error and derivatives
        NvFlexGetRigidJointState(&joint, NULL, &g_buffers->rigidBodies[joint.body1], linPos, linRate, angPos, angRate);

		const Vec3 axis = Vec3(1.0f, 0.0f, 0.0f);

		float err = angPos[0] - target;
		float derr = angRate[0];
		float dderr = -(gain*(err + g_dt*derr) + damping*derr)/(Dot(axis, Matrix33(body.inertia)*axis) + damping*g_dt);
	
		float action;

		if (stable)
		{
			// use semi-implicit Stable-PID from Liu et al. Georgia Tech 
			action = gain*(err + derr*g_dt) + damping*(derr + dderr*g_dt);
		}
		else			
		{
			// explicit PID
			action = gain*err + damping*derr;
		}

		Vec3 torque = axis*action;

		body.torque[0] -= torque.x;
		body.torque[1] -= torque.y;
		body.torque[2] -= torque.z;
	}

	float target = 0.0f;

	void DoGui()
	{
		imguiSlider("Target", &target, -1.57f, 1.57f, 0.001f);
		imguiSlider("PD Gain", &gain, 0.0f, 1000.0f, 0.0001f);
		imguiSlider("PD Damping", &damping, 0.0f, 100.f, 0.0001f);
		
		// use the stable PD from Liu et al.
		if (imguiCheck("PD Stable", stable))
			stable = !stable;
	}

	bool stable = true;

	float gain = 500.0f;
	float damping = 5.0f;
};

//------------------------------------------------------------------------------
class RigidTablePile : public Scene
{
public:

	RigidTablePile()
    {
    	int height = 6;
		int dx = 3;
		int dy = 3;

		float yspacing = 1.0f;
		float xyspacing = 1.0f;

		RandInit();

		Vec3 offset(-0.5f*xyspacing*(dx+0.5f), 1.0f, 0.0f);
		//Vec3 offset(-1.0f, 1.0f, 0.0f);


		for (int x=0; x < dx; ++x)
		{
			for (int y=0; y < dy; ++y)
			{
				for (int i=0; i < height; ++i)
				{
					Vec3 pos = offset + Vec3((x+Randf(0.4f))*xyspacing,
											i*yspacing,
											(y+Randf(0.1f))*xyspacing);
					AddTable(pos);
				}
			}
		}

        //g_sceneLower = Vec3(-0.6f, -0.4f, -0.4f);
        //g_sceneUpper = Vec3(0.2f, 0.4f, 0.4f);
        g_sceneLower = Vec3(-5.0f, -0.4f, -2.0f);
        g_sceneUpper = Vec3(5.0f, 1.0f, 2.0f);

        g_params.geometricStiffness = 0.0f;
        g_params.relaxationFactor = 0.75f;
        g_params.systemTolerance = 1.e-3f;

        g_params.solverType = eNvFlexSolverPCR;
        g_params.numIterations = 6;
        g_params.numInnerIterations = 20;

        g_pause = true;

        g_params.numPlanes = 1;


        g_numSubsteps = 1;  g_params.solverType = eNvFlexSolverPCR;  g_params.numInnerIterations = 40; g_params.scalingMode = 1; g_params.numIterations = 100; g_params.systemTolerance = 1.e-3f; 

        FILE* file = fopen("tables.bin", "rb");
        if (file)
        {
        	fread(&g_buffers->rigidBodies[0], sizeof(NvFlexRigidBody), g_buffers->rigidBodies.size(), file);
        	fclose(file);
        }
    }

    NvFlexTriangleMeshId mesh;

    void CenterCamera()
    {
    	g_camPos = Vec3(-0.378588f, 1.351021f, 6.120290f);
    	g_camAngle = Vec3(0.0f, -0.013963f, 0.000000f);
    }

    void KeyDown(int key)
    {
		if (key == 'f')
		{
			g_buffers->rigidBodies.map();

			FILE* file = fopen("tables.bin", "wb");
			if (file)
			{
				fwrite(&g_buffers->rigidBodies[0], sizeof(NvFlexRigidBody), g_buffers->rigidBodies.size(), file);
				fclose(file);
			}

			g_buffers->rigidBodies.unmap();
		}
    }

    void AddTable(const Vec3& pos)
	{
		float friction = 0.7f;

		RenderMaterial material;
		material.frontColor = Vec3(0.805f, 0.702f, 0.401f);//Vec3(0.35f, 0.45f, 0.65f);//Vec3(0.7f);
		material.backColor = Vec3(0.0f);
		material.roughness = 0.5f;
		material.specular = 0.5f;

		void* user = UnionCast<void*>(AddRenderMaterial(material));


		if (1)
		{
			// Table top
			NvFlexRigidShape shape[5];
			NvFlexMakeRigidBoxShape(&shape[0], g_buffers->rigidBodies.size(),
					0.55f/2, 0.05f/2, 0.55f/2, NvFlexMakeRigidPose(0,0));

			// 4 legged table		
			const float offset = 0.25f;

			NvFlexMakeRigidBoxShape(&shape[1], g_buffers->rigidBodies.size(),
					0.05f/2, 0.40f/2, 0.05f/2,
					NvFlexMakeRigidPose(Vec3(-offset,-0.225,-offset),0));

			NvFlexMakeRigidBoxShape(&shape[2], g_buffers->rigidBodies.size(),
					0.05f/2, 0.40f/2, 0.05f/2,
					NvFlexMakeRigidPose(Vec3(offset,-0.225,-offset),0));

			NvFlexMakeRigidBoxShape(&shape[3], g_buffers->rigidBodies.size(),
					0.05f/2, 0.40f/2, 0.05f/2,
					NvFlexMakeRigidPose(Vec3(offset,-0.225,offset),0));

			NvFlexMakeRigidBoxShape(&shape[4], g_buffers->rigidBodies.size(),
					0.05f/2, 0.40f/2, 0.05f/2,
					NvFlexMakeRigidPose(Vec3(-offset,-0.225,offset),0));

			float densities[5] = {1000.0f,1000.0f,1000.0f,1000.0f,1000.0f};
			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body,pos, Quat(), shape,
					densities, 5);

			// assign render material
			for (int i=0; i < 5; ++i)
			{
				shape[i].filter = 0; // collide all shapes
				shape[i].material.friction = friction;		
				shape[i].user = user;

				g_buffers->rigidShapes.push_back(shape[i]);
			}
			g_buffers->rigidBodies.push_back(body);
		}
	}
};



//------------------------------------------------------------------------------
class RigidBroccoli : public Scene
{
public:

	RigidBroccoli()
    {
    	int height = 5;
		int dx = 4;
		int dz = 4;

		float xspacing = 0.5f;
		float yspacing = 0.5f;
		float zspacing = 0.5f;

		Vec3 offset(-0.5f*xspacing*dx, 0.5f, 0.0f);

		Mesh* visual = ImportMesh("../../data/broccoli/broccoli.obj");
		Mesh* collision = ImportMesh("../../data/broccoli/broccoli_collision.obj");

		float scale = 0.05f;

		visual->Transform(ScaleMatrix(scale));
		collision->Transform(ScaleMatrix(scale));

		for (int x=0; x < dx; ++x)
		{
			for (int z=0; z < dz; ++z)
			{
				for (int i=0; i < height; ++i)
				{
					Vec3 pos = offset + Vec3(x*xspacing, i*yspacing, z*zspacing);

					float density = 1000.0f;
					float thickness = 0.015f;
					float friction = 0.5f;

					CreateRigidBodyFromMesh("../../data/broccoli/", visual, collision, pos, Quat(), thickness, density, friction);
				}
			}
		}

        g_sceneLower = Vec3(-5.0f, -0.4f, -2.0f);
        g_sceneUpper = Vec3(5.0f, 1.0f, 2.0f);

		g_params.numIterations = 40;
        g_params.shapeCollisionMargin = 0.01f;
        g_params.geometricStiffness = 0.0f;

        g_numSubsteps = 2;
        g_pause = true;
    }
};



// fwds
void InitScene(int scene, bool centerCamera);
void UpdateFrame(void);

struct Experiment
{
	Experiment(std::string file, std::string name, int length) : file(file), name(name), numFrames(length) {}

	void Run()
	{
		NvFlexStartExperiment(file.c_str(), name.c_str());

		for (int i=0; i < runFuncs.size(); ++i)
		{
			const int experimentScene = g_sceneFactories.size()-1;

			// override scene factor with the experiment creation func, rely on being the last factory
			g_sceneFactories[experimentScene].mFactory = runFuncs[i];

			// run for some time
			InitScene(experimentScene, false);

			g_pause = false;

			// run for n iterations
			for (int f=0; f < numFrames; ++f)
			{
				// enable debug instrumentation on last frame only
				if (f == numFrames-1)
				{
					startFuncs[i]();

					NvFlexStartExperimentRun(runLabels[i].c_str());
				}

				UpdateFrame();

				if (f == numFrames-1)
					NvFlexStopExperimentRun();
			}
		}

		NvFlexStopExperiment();
	}

	void AddRun(const char* label, std::function<Scene*()> sceneCreate, std::function<void()> sceneStart)
	{
		runFuncs.push_back(sceneCreate);
		startFuncs.push_back(sceneStart);

		runLabels.push_back(label);
	}

	std::vector<std::function<Scene*()>> runFuncs;
	std::vector<std::function<void()>> startFuncs;

	std::vector<std::string> runLabels;

	int numFrames = 30;

	std::string name;
	std::string file;

};

std::vector<Experiment> g_experiments;

template <typename T>
void AddRScalingTest(const char* filename, const char* title, int frames)
{
	const int outerIterations = 100;
	const int innerIterations = 40;
    
	Experiment e(filename, title, frames);
	e.AddRun("r=1", [=]() { T* scene = new T(); return scene; }, [=]() { g_numSubsteps = 1; g_params.solverType = eNvFlexSolverPCR; g_params.numInnerIterations = innerIterations; g_params.scalingMode = 2;  g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-4f;  });
	e.AddRun("r=h^2", [=]() { T* scene = new T(); return scene; } , [=]() { g_numSubsteps = 1; g_params.solverType = eNvFlexSolverPCR; g_params.numInnerIterations = innerIterations; g_params.scalingMode = 0;  g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-4f;   });
	e.AddRun("r=h^2diag(A)", [=]() { T* scene = new T(); return scene; } , [=]() { g_numSubsteps = 1; g_params.solverType = eNvFlexSolverPCR; g_params.numInnerIterations = innerIterations; g_params.scalingMode = 1; g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-4f;  });

	g_experiments.push_back(e);
}

template <typename T>
void AddSolverTest(const char* filename, const char* title, int frames)
{
	const int outerIterations = 100;
	const int innerIterations = 40;

	Experiment e(filename, title, frames);
	e.AddRun("Jacobi", [=]() {  T* scene = new T(); return scene; } , [=]() { g_numSubsteps = 1;  g_params.solverType = eNvFlexSolverJacobi; g_params.numInnerIterations = innerIterations; g_params.scalingMode = 1;  g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-6f;  } );
	e.AddRun("Gauss-Seidel", [=]() { T* scene = new T(); return scene; } , [=]() { g_numSubsteps = 1;   g_params.solverType = eNvFlexSolverGaussSeidel; g_params.numInnerIterations = innerIterations; g_params.scalingMode = 1;  g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-6f; } );
	//e.AddRun("LDLT", [=]() { T* scene = new T(); return scene; }, [=]() { g_numSubsteps = 1; g_params.relaxationFactor = 0.75f; g_params.solverType = eNvFlexSolverLDLT; g_params.numInnerIterations = innerIterations; g_params.scalingMode = 1; g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-3f; });
	e.AddRun("PCG", [=]() { T* scene = new T(); return scene; }, [=]() { g_numSubsteps = 1; g_params.solverType = eNvFlexSolverPCG2; g_params.numInnerIterations = innerIterations; g_params.scalingMode = 1;  g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-6f;   } );
	e.AddRun("PCR", [=]() { T* scene = new T();return scene; }, [=]() {  g_numSubsteps = 1;   g_params.solverType = eNvFlexSolverPCR;  g_params.numInnerIterations = innerIterations; g_params.scalingMode = 1; g_params.numIterations = outerIterations; g_params.systemTolerance = 1.e-4f;  } );

	g_experiments.push_back(e);
}

/*
struct RigidFetchBeam : public RigidFetch
{
	RigidFetchBeam() : RigidFetch(RigidFetch::eFlexibleBeam) {}
};
*/

void RegisterExperimentScenes()
{
	// create scene functor for creating experiments
	RegisterScene("Experiment", []() { return nullptr; });

	// r-scaling tests
	//AddRScalingTest<RigidFriction>("rscaling_box.m", "box", 10);
	//AddRScalingTest<RigidCollision>("rscaling_cylinder.m", "cylinder", 30);
	AddRScalingTest<RigidHeavyStack>("rscaling_heavy.m", "heavy", 60);
	AddRScalingTest<RigidTower>("rscaling_tower.m", "tower", 30);
	AddRScalingTest<RigidTablePile>("rscaling_table.m", "pile", 1);
	AddRScalingTest<RigidArch>("rscaling_arch.m", "arch", 30);
	//AddRScalingTest<RigidComplementarity>("rscaling_simple.m", "simple", 1);
	AddRScalingTest<RigidCapsuleStack>("rscaling_capsule.m", "capsule", 30);

	// solver tests
	AddSolverTest<RigidHeavyStack>("solver_heavy.m", "heavy", 60);
	//AddSolverTest<RigidTower>("solver_tower.m", "tower", 30);
	//AddSolverTest<RigidTablePile>("solver_table.m", "pile", 30);
	//AddSolverTest<RigidCapsuleStack>("solver_capsule.m", "capsule", 30);
	AddSolverTest<RigidArch>("solver_arch.m", "arch", 30);
	//AddSolverTest<RigidFetchBeam>("solver_beam.m", "beam", 1);
	AddSolverTest<FEMPoisson>("solver_fem.m", "fem", 1);
}

void RunExperiments(const char* filter)
{
	for (int i=0; i < g_experiments.size(); ++i)
	{
		if (strstr(g_experiments[i].file.c_str(), filter))
		{
			g_experiments[i].Run();
		}
	}
}


class RigidBodyPile : public Scene
{
public:

	RigidBodyPile ()
	{
		Colour defaultColor = Colour(71.0f / 255.0f, 165.0f / 255.0f, 1.0f);
        Colour remoteShapeColor = Colour(100.0f / 255.0f, 100.0f / 255.0f, 100.0f / 255.0f);
        Colour sharedShapeColor = Colour(200.0f / 255.0f, 200.0f / 255.0f, 200.0f / 255.0f);
        Colour bulldozerColor = Colour(255.0f / 255.0f, 255.0f / 255.0f, 0.0f);

        
        int dim = 10;
        float xzpileSize = 0.4f;
        float boxSpacing = xzpileSize/(float)dim;
        float boxRadius = (boxSpacing/2.0f)*0.8f;

        float ypileSize = 0.2f;
        int ydim = int (ypileSize / boxSpacing);
        //int ydim = 2;
        //int ydim = 3;

		const float density = 1000.0f;

        Vec3 pileOrigin(-1.f, 0.f, -xzpileSize / 2.f);

        printf("boxRadius %f\n",boxRadius);

        int boxCount = 0;
        for(int z = 0; z!=dim; z++)
        {
            for(int y=0; y!=ydim; y++)
            {
                for (int x = 0; x != dim; x++)
                {
                    NvFlexRigidShape shape;
                    NvFlexMakeRigidBoxShape(&shape, boxCount, boxRadius, boxRadius, boxRadius, NvFlexMakeRigidPose(0, 0));
                    shape.filter = 0;
                    shape.material.friction = 1.0f;

                    shape.thickness = boxRadius*0.2f;

                    NvFlexRigidBody body;
                    NvFlexMakeRigidBody(g_flexLib, &body, pileOrigin + Vec3(0.5f * boxSpacing + float(x) * boxSpacing,
						0.5f * boxSpacing + float(y) * boxSpacing, 0.5f * boxSpacing + float(z) * boxSpacing), Quat(), &shape, &density, 1);

                    g_buffers->rigidBodies.push_back(body);
                    g_buffers->rigidShapes.push_back(shape);
                    boxCount++;
                }
            }
        }
        printf("boxCount %d\n",boxCount);

        NvFlexRigidShape shape;
        NvFlexMakeRigidBoxShape(&shape, boxCount, 0.01f, 0.1f, 0.4f, NvFlexMakeRigidPose(0, 0));
        shape.filter = 0;
        shape.material.friction = 0.0f;
        NvFlexRigidBody body;
        NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-2.0f, 0.1f, 0.0f), Quat(), &shape, &density, 1);
        //NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-0.3f, 0.1f, -1.0f), Quat(), &shape, 1, 1.0f);
        //NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-1.0f, 0.1f, 0.0f), Quat(), &shape, 1, 1.0f);
        body.linearVel[0] = 0.2f;

        // Make kinematic
        body.mass = 0.0f;
		body.invMass = 0.0f;
		(Matrix33&)body.inertia = Matrix33();
        (Matrix33&)body.invInertia = Matrix33();

        g_buffers->rigidBodies.push_back(body);
        g_buffers->rigidShapes.push_back(shape);

        g_sceneLower = Vec3(-0.6f, -0.4f, -0.4f);
        g_sceneUpper = Vec3(0.2f, 0.4f, 0.4f);

        g_numSubsteps = 4;
        g_params.numIterations = 30;
        g_params.shapeCollisionMargin = boxRadius * 0.5f;	
	}

};
