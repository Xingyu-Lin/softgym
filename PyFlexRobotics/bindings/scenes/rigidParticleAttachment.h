#include "../mjcf.h"
#include "../urdf.h"
#include "../../core/maths.h"


#pragma once
class RigidParticleAttachment : public Scene
{
public:

	RigidParticleAttachment()
	{
		float scale = 0.5f;
		float density = 1500.0f;
		float density2 = 100.0f;

		Mesh* boxMesh = ImportMesh("../../data/box.ply");
		boxMesh->Transform(ScaleMatrix(scale));
		g_tetraMaterials.resize(0);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(2.5e+5f, 0.40f, 0));
		int phase = NvFlexMakePhaseWithChannels(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter, eNvFlexPhaseShapeChannel0);
		int NoCollisionphase = NvFlexMakePhaseWithChannels(1, 0, eNvFlexPhaseShapeChannel0 << 1);
		int particleOffset = 0;
		int particleEnd = 0;

		for (int j = 0; j < 3; ++j)
		{
			Vec3 boxPosition = Vec3(0.0f, 0.85f + 2.3f * (float)j, 0.f);
			particleOffset = g_buffers->positions.size();
			//make soft material
			{
				const float radius = 0.1f;
				Vec3 lower = boxPosition + Vec3(-0.2f, 0.22f, -0.2f);

				CreateTetGrid(Transform(lower, Quat()), 4, 12, 4, radius, radius, radius, 110, ConstantMaterial<0>, false, false);
			}
			particleEnd = g_buffers->positions.size();
			g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

			//make lower box
			{
				NvFlexTriangleMeshId shapeId = CreateTriangleMesh(boxMesh, 0.00125f);
				NvFlexRigidShape shape;
				NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
				shape.filter = 0x0;
				shape.material.friction = 1.0f;
				shape.material.torsionFriction = 0.1;
				shape.material.rollingFriction = 0.0f;
				shape.thickness = 0.01f;

				NvFlexRigidBody body;
				NvFlexMakeRigidBody(g_flexLib, &body, boxPosition, Quat(), &shape, &density2, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);
			}

			//make top box
			{
				NvFlexTriangleMeshId shapeId = CreateTriangleMesh(boxMesh, 0.00125f);
				NvFlexRigidShape shape;
				NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
				shape.filter = 0x0;
				shape.material.friction = 1.0f;
				shape.material.torsionFriction = 0.1;
				shape.material.rollingFriction = 0.0f;
				shape.thickness = 0.01f;

				NvFlexRigidBody body;
				//rotate box 45 degrees on yaw
				NvFlexMakeRigidBody(g_flexLib, &body, boxPosition + Vec3(0, 1.65f, 0), Quat(0,0.382683432365090f,0, 0.923879532511287f), &shape, &density2, 1);

				g_buffers->rigidBodies.push_back(body);
				g_buffers->rigidShapes.push_back(shape);
			}

			//add attachments
			for (int x = 0; x < 5; ++x)
			{
				for (int z = 0; z < 5; ++z)
				{
					int particle = particleOffset + x * 13 * 5 + z;
					Vec3 position = Vec3(g_buffers->positions[particle]);
					Vec3 relPosition(position - boxPosition);
					CreateRigidBodyToParticleAttachment(g_buffers->rigidBodies.size() - 2, particle);

					particle = particleOffset + x * 13 * 5 + z + 12 * 5;
					position = Vec3(g_buffers->positions[particle]);
					relPosition = position - (boxPosition + Vec3(0, 1.65f, 0));
					CreateRigidBodyToParticleAttachment(g_buffers->rigidBodies.size() - 1, particle);
				}
			}
		}
		
		int numParticles = particleEnd - particleOffset;	
		
		g_params.gravity[1] = -9.81f;
		g_numSubsteps = 3;
		g_params.numIterations = 4;
		g_params.solverType = eNvFlexSolverPCR;
		g_params.geometricStiffness = 0.08f;

		g_params.dynamicFriction = 1.0f;
		g_params.particleFriction = 1.5f;
		g_params.shapeCollisionMargin = 0.01f;
		g_params.collisionDistance = 0.01f;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		g_drawPoints = true;
		g_pause = true;
	}

	virtual void Draw(int pass)
	{
		BeginLines();
				
		for (int i = 0; i < g_buffers->rigidParticleAttachments.size(); ++i)
		{
			int bodyIndex = g_buffers->rigidParticleAttachments[i].body;
			Vec3 bodyPos(g_buffers->rigidBodies[bodyIndex].com);
			uint32_t p = g_buffers->rigidParticleAttachments[i].particle;
			Quat theta(g_buffers->rigidBodies[bodyIndex].theta);
			Transform T(bodyPos, theta);
			Mat44 transformMat = TransformMatrix(T);

			Vec3 localOffset(g_buffers->rigidParticleAttachments[i].localOffset);
			Mat44 localMat = TransformMatrix(Rotation(), Point3(localOffset));
			Vec3 globalLoc = Vec3((transformMat*localMat).GetTranslation());

			//draw red line on constraint (if it shows is because the constriant was violated)
			DrawLine(g_buffers->rigidBodies[bodyIndex].com, globalLoc , Vec4(1.f, 0.f, 0.f, 0.f));
			//draw green line form body com to particle
			DrawLine(bodyPos, Vec3(g_buffers->positions[p]), Vec4(0.f, 1.f, 0.f, 0.f));
		}

		EndLines();
	}
};