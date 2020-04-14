#pragma once

#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include "../deformable.h"

const int kNumFingers = 3;

class PneuNetFinger : public Scene
{
public:

	enum Mode
	{
		eCube,
		eBanana,
		eSphere
	};

	Mode mode;

	Mesh mesh;

	DeformableMesh* fingers[kNumFingers];

	bool drawMesh = false;

	float stiffness  = 5.e+6f;
	float poissons = 0.47f;
	float damping = 0.001f;
	
	// finger scale
	float fingerScale = 0.1f;
	
	Vec3 fingerTranslation;
	float fingerRotation = 0.0f;

	RenderMesh* baseMesh = NULL;

    float tableHeight = 0.1f;

    std::vector<float> activation;

    PneuNetFinger(Mode mode) : mode(mode)
    {
	    g_numSubsteps = 2;
        g_params.numIterations = 5;
        g_params.numInnerIterations = 80;

        g_params.solverType = eNvFlexSolverPCR;
        g_params.geometricStiffness = 0.0f;

        g_params.dynamicFriction = 0.8f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 0.0f;
		
        g_params.relaxationFactor = 0.75f;
        g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.001f;

        g_sceneLower = Vec3(-0.5f);
        g_sceneUpper = Vec3(0.5f);

        g_pause = true;

        g_drawPoints = false;
        g_drawCloth = false;

        g_numSubsteps = 2;

        for (int i=0; i < kNumFingers; ++i)
        {
			const float density = 1500.0f;

			const int startVertex = g_buffers->positions.size();

			fingers[i] = CreateDeformableMesh("../../data/pneunet/finger.obj", "../../data/pneunet/finger.tet", Vec3(0.0f, tableHeight + 0.15f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), i*(k2Pi/kNumFingers))*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(15.0f)), fingerScale, density, 0, NvFlexMakePhase(0, NvFlexPhase::eNvFlexPhaseSelfCollide | NvFlexPhase::eNvFlexPhaseSelfCollideFilter));

			g_tetraMaterials.resize(0);

			g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));
			g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness*0.1f, poissons, damping));

			for (int v=0; v < fingers[i]->tetMesh->vertices.size(); ++v)
			{	
				if (fingers[i]->tetMesh->vertices[v].x > -0.1f)
					g_buffers->positions[startVertex+v].w = 0.0f;
				else
					g_buffers->positions[startVertex+v].w = 2000.0f;
			}
		}

		// activation array to store per-element activation value
		activation.resize(g_buffers->tetraRestPoses.size());

		if (mode == eSphere)
		{
			float radius = 0.05f;

			NvFlexRigidShape shape;
			NvFlexMakeRigidSphereShape(&shape, 0, radius, NvFlexMakeRigidPose(0,0));
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.9372549f, 0.611274f, 0.31f)));

			float density = 250.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0001f, 0.1f + radius + 0.01f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

			printf("mass: %f\n", body.mass);


			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidBodies.push_back(body);
		}
		else if (mode == eCube)
		{
			float radius = 0.03f;

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, 0, radius, radius, radius, NvFlexMakeRigidPose(0,0));
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.9372549f, 0.611274f, 0.31f)));
			shape.thickness = 0.01f;

			float density = 250.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0001f, 0.1f + radius + 0.01f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

			printf("mass: %f\n", body.mass);


			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidBodies.push_back(body);
		}		

		else if (mode == eBanana)
		{
			float radius = 0.03f;

			NvFlexTriangleMeshId mesh = CreateTriangleMesh(ImportMesh("../../data/banana.obj"));
			float meshScale = 0.7f;

			NvFlexRigidShape shape;
			//NvFlexMakeRigidBoxShape(&shape, 0, radius, radius, radius, NvFlexMakeRigidPose(0,0));
			NvFlexMakeRigidTriangleMeshShape(&shape, 0, mesh, NvFlexMakeRigidPose(0,0), meshScale, meshScale, meshScale);			
			shape.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.9372549f, 0.811274f, 0.31f)));
			shape.thickness = 0.01f;

			float density = 250.0f;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(0.0001f, 0.1f + radius + 0.01f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), DegToRad(0.0f)), &shape, &density, 1);

			printf("mass: %f\n", body.mass);


			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidBodies.push_back(body);
		}		

		// table
		NvFlexRigidShape box;
		NvFlexMakeRigidBoxShape(&box, -1, 0.5f, tableHeight, 0.5f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f),0));
		box.filter = 0;
		box.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.35f, 0.45f, 0.65f)));

		g_buffers->rigidShapes.push_back(box);

		// base mesh (visual only)
		//float baseHeight = 0.02f;
		//float baseRadius = 0.025f;
		//Mesh* base = CreateCylinder(40, baseRadius, baseHeight, true);
		//base->Transform(TranslationMatrix(Point3(0.0f, baseHeight, 0.0f)));
		Mesh* base = ImportMeshFromObj("../../data/pneunet/base.obj");		
		
		baseMesh = CreateRenderMesh(base);

		delete base;
    }
	

	virtual void DoGui()
	{
		if (imguiCheck("Draw Mesh", drawMesh))
			drawMesh = !drawMesh;

		float logStiffness = log10f(stiffness);
		imguiSlider("Stiffness (log)", &logStiffness, 0.f, 10.0f, 0.0001f);
		imguiSlider("Poisson", &poissons, 0.1f, 0.5f, 0.0001f);
		imguiSlider("Damping", &damping, 0.0f, 1.0f, 0.0001f);

		stiffness = powf(10.0f, logStiffness);
		g_tetraMaterials[0] = IsotropicMaterialCompliance(stiffness, poissons, damping);

		float newInflation = inflation;
		imguiSlider("Finger Inflation", &newInflation, 0.4f, 2.0f, 0.0001f);
		inflation = Lerp(inflation, newInflation, 0.05f);
		
		Vec3 newTranslation = fingerTranslation;
		imguiSlider("Finger X", &newTranslation.x, -0.1f, 0.1f, 0.0001f);
		imguiSlider("Finger Y", &newTranslation.y, -0.1f, 0.1f, 0.0001f);
		imguiSlider("Finger Z", &newTranslation.z, -0.1f, 0.1f, 0.0001f);

		float newRotation = fingerRotation;
		imguiSlider("Finger Rotation", &newRotation, 0.0f, k2Pi, 0.00001f);

		fingerTranslation = Lerp(fingerTranslation, newTranslation, 0.05f);
		fingerRotation = Lerp(fingerRotation, newRotation, 0.05f);
	}

	float inflation = 1.0f;

	virtual void CenterCamera()
	{
		g_camPos = Vec3(-0.176218f, 0.283473f, 0.446055f);
		g_camAngle = Vec3(-0.375247f, -0.092502f, 0.000000f);
	}

	virtual void Update()
	{
		int startTet = 0;

		for (int i=0; i < kNumFingers; ++i)
		{		
			// inflation along the x-axis
			Matrix33 pressure = Matrix33::Identity()*(1.0f/fingerScale);
			pressure(0,0) *= inflation;

			DeformableMesh* deformable = fingers[i];

			for (int j=0; j < deformable->tetMesh->tetraRestPoses.size(); ++j)
			{
				// calculate centroid
				Vec3 a = Vec3(deformable->tetMesh->vertices[deformable->tetMesh->tetraIndices[j*4+0]]);
				Vec3 b = Vec3(deformable->tetMesh->vertices[deformable->tetMesh->tetraIndices[j*4+1]]);
				Vec3 c = Vec3(deformable->tetMesh->vertices[deformable->tetMesh->tetraIndices[j*4+2]]);
				Vec3 d = Vec3(deformable->tetMesh->vertices[deformable->tetMesh->tetraIndices[j*4+3]]);

				Vec3 centroid = (a + b + c + d)*0.25f;

				// inflate the chambers only
				if (centroid.y > 0.05f)
				{
					g_buffers->tetraRestPoses[startTet + j] = deformable->tetMesh->tetraRestPoses[j]*pressure;

					// store activation value (just for visualization)
					activation[startTet + j] = inflation-1.0f;
				}
			}

			UpdateDeformableMesh(deformable);		

			startTet += deformable->tetMesh->tetraRestPoses.size();
		}

		// translate fixed particles
		Quat r = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), fingerRotation);

		for (int i=0; i < g_buffers->positions.size(); ++i)
		{
			if (g_buffers->positions[i].w == 0.0f)
			{
				Vec3 p = r*Vec3(g_buffers->restPositions[i]) + fingerTranslation;

				g_buffers->positions[i] = Vec4(p, 0.0f);
			}
		}
	}

	virtual void Sync()
	{
	    if (g_buffers->tetraIndices.size())
	    {   
	    	NvFlexSetFEMMaterials(g_solver, &g_tetraMaterials[0], g_tetraMaterials.size());
	        NvFlexSetFEMGeometry(g_solver, g_buffers->tetraIndices.buffer, g_buffers->tetraRestPoses.buffer, g_buffers->tetraMaterials.buffer, g_buffers->tetraMaterials.size());
	    }
	}

	virtual void Draw(int pass)
	{
		RenderMaterial baseMaterial;
		baseMaterial.metallic = 0.0f;
		baseMaterial.frontColor = Vec3(0.99f, 0.98f, 0.97f)*0.75f;
		baseMaterial.backColor = Vec3(0.97f);
		baseMaterial.roughness = 0.2f;
		baseMaterial.specular = 0.8f;

		DrawRenderMesh(baseMesh, TranslationMatrix(Point3(fingerTranslation) + Vec3(0.0f, tableHeight + 0.165f, 0.0f))*RotationMatrix(fingerRotation, Vec3(0.0f, 1.0f, 0.0f)), baseMaterial);

		if (drawMesh)
		{
			for (int i=0; i < kNumFingers; ++i)
			{
				// visual mesh
				DrawDeformableMesh(fingers[i]);
			}
		}
		else
		{
			// tetrahedral mesh
			if (pass == 1)
			{
				mesh.m_positions.resize(g_buffers->positions.size());
				mesh.m_normals.resize(g_buffers->normals.size());
				mesh.m_colours.resize(g_buffers->positions.size());
				mesh.m_indices.resize(g_buffers->triangles.size());

				for (int i = 0; i < g_buffers->triangles.size(); ++i)
					mesh.m_indices[i] = g_buffers->triangles[i];

				float rangeMin = FLT_MAX;
				float rangeMax = -FLT_MAX;

				std::vector<Vec2> averageStress(mesh.m_positions.size());

				// calculate average Von-Mises stress on each vertex for visualization
				for (int i = 0; i < g_buffers->tetraIndices.size(); i += 4)
				{
					float vonMises = fabsf(g_buffers->tetraStress[i / 4]) + activation[i/4];

					averageStress[g_buffers->tetraIndices[i + 0]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i + 1]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i + 2]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i + 3]] += Vec2(vonMises, 1.0f);

					rangeMin = Min(rangeMin, vonMises);
					rangeMax = Max(rangeMax, vonMises);
				}

				rangeMin = 0.0f;
				rangeMax = 1.0f;

				for (int i = 0; i < g_buffers->positions.size(); ++i)
				{
					mesh.m_normals[i] = Vec3(g_buffers->normals[i]);
					mesh.m_positions[i] = Point3(g_buffers->positions[i]) + Vec3(g_buffers->normals[i])*g_params.collisionDistance;

					mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, fabsf(averageStress[i].x / averageStress[i].y));					
				}
			}

			DrawMesh(&mesh, g_renderMaterials[0]);

			if (pass == 0 && g_buffers->triangles.size())
			{

				SetFillMode(true);

				DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], g_params.collisionDistance);

				SetFillMode(false);

			}
		}
	}
};


