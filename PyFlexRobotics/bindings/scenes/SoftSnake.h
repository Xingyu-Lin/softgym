#include "../mjcf.h"
#include "../urdf.h"
#include "../../core/maths.h"
#include "../deformable.h"
#include <vector>
#include "rlbase.h"

#pragma once
namespace softsnake
{
	const int nParticles = 376;

	//Particles used to set up constraints with rigid bodies
	//Particle coefficients for lower extremity of link offset from first particle in link;
	const int nBorderParticles = 31;
	const int lowerParticles[nBorderParticles] = { 0,1,6,13,16,17,18,19,20,21,22,23,24,25,26,62,63,75,90,154,203,204,211,243,244,284,325,345,367,368,369 };
	//Particle coefficients for upper extremity of link offset from first particle in link
	const int upperParticles[nBorderParticles] = { 2,3,4,5,7,8,9,10,11,12,14,15,40,47,64,76,113,127,167,168,178,200,234,241,255,295,322,335,358,361,366 };

	//particles in the center of the link (for extension constraint)
	//repeat first particle in end to simplify spring generation loop
	const int nCenterParticles = 33;
	const int centerParticles[nCenterParticles] = { 2,366,7,168,293,289,290,291,292,169,283,282,173,174,177,26,369,367,24,154,155,156,158,276,277,160,162,163,164,165,166,167,2 };
	//radial constraint
	//first three columns are outer (in x), last 3 are inner
	const int nChambers = 2;
	const int nPlanes = 13;
	const int nPlanePartricles = 6;

	const int ChamberPlanarParticles[nChambers][nPlanes][nPlanePartricles] = {
		{
			{ 8,	76,	4,		358,	361,	113},
			{ 78,	77,	92,		122,	360,	346 },
			{ 112,	93,	94,		121,	363,	111 },
			{ 79,	95,	120,	119,	347,	362 },
			{ 110,	97,	96,		118,	359,	109 },
			{ 80,	81,	114,	115,	364,	104 },
			{ 82,	98,	116,	126,	365,	105 },
			{ 106,	100,117,	354,	349,	355 },
			{ 83,	84,	99,		353,	348,	107 },
			{ 85,	86,	101,	123,	350,	108 },
			{ 87,	88,	102,	124,	352,	356 },
			{ 89,	91,	103,	125,	351,	357 },
			{ 90,	21,	19,		6,		368,	25 }
		},

		{
			{ 15,	178,	9,		200,	335,	234},
			{ 230,	219,	198,	199,	209,	232 },
			{ 181,	180,	186,	336,	190,	233 },
			{ 229,	220,	185,	197,	339,	228 },
			{ 179,	218,	217,	337,	192,	212 },
			{ 231,	184,	201,	191,	338,	208 },
			{ 240,	216,	202,	341,	194,	227 },
			{ 239,	224,	183,	187,	342,	226 },
			{ 238,	215,	223,	188,	340,	196 },
			{ 237,	214,	205,	189,	343,	236 },
			{ 213,	182,	206,	207,	193,	225 },
			{ 235,	222,	221,	195,	210,	344 },
			{ 16,	17,		203,	204,	211,	345 }
		}
	};
	const int nSpringsL1 = (nPlanes - 1)*nPlanePartricles;
	const int nSpringsL2 = (nPlanes - 2)*nPlanePartricles;

	//spring lengths and scale
	const float chamberSpringLenghtL1 = 0.004557500718182f;
	const float chamberSpringLenghtL2 = chamberSpringLenghtL1 * 2.0f;

	const float SpringScaleInner = 0.578069634550040f;
	const float SpringScaleOuter = 1.0f;

	const float maxPressure = 9.0f;
}


class SoftSnake : public Scene
{
	struct SnakeLink
	{
		DeformableMesh* mDeformable;
		uint32_t mparticlesOffset;
		uint32_t mSpringsL1Offset; //springs for each chamber
		uint32_t mSpringsL2Offset; //srpings L2 for each chamber
		uint32_t mRigidIndex;

		SnakeLink() {}
	};

	Mesh mesh;
	bool mDrawMesh{ false };

public:
	SoftSnake(const char *configfile)
	{				
		mPhase = NvFlexMakePhaseWithChannels(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter, eNvFlexPhaseShapeChannel0);

		int nLinks;

		Vec3 translation(0.f, 0.05f, 0.f);
		float theta = 45.f;
		Rotation Rbase(0.f, 0.f, -90.f);//first make the link be in horizontal instead of Vertical		

		Mat44 Rb = TransformMatrix(Rbase, Point3());
		Rotation Rtheta(theta, 0.f, 0.f);//first make the link be in horizontal instead of Vertical
		Mat44 Rt = TransformMatrix(Rtheta, Point3());
		Mat44 Ft = Rt * Rb;
		Matrix33 Rr(Ft.GetAxis(0), Ft.GetAxis(1), Ft.GetAxis(2));
		Quat rotation(Rr);
		Vec3 linkOffset = Rotate(rotation, Vec3(0.f, 0.055f + 0.022f, 0.f));

		for (int i = 0; i < 4; ++i)
		{
			createLink(translation + linkOffset * float(i), Ft, rotation);
		}

		//add last rigid body to snake
		addRigidBodyComponent(translation + linkOffset * float(mLinks.size()), Ft, rotation);
		connectLinks();

		const float chamber_offset = 0.0175f;
		const float stiffness = 2.5e+4f;
		const float poissons = 0.40f;
		const float damping = 0.00f;

		g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);
		g_tetraMaterials.push_back(IsotropicMaterialCompliance(stiffness, poissons, damping));
		g_pause = true;
		g_drawPoints = false;
		//g_camSpeed = 0.005f;
		g_params.radius = 0.001f;
		g_numSubsteps = 2;
		g_params.numIterations = 4;
		g_params.solverType = eNvFlexSolverPCR;
		g_params.geometricStiffness = 0.08f;
		//
		g_params.dynamicFriction = 1.0f;
		g_params.particleFriction = 1.5f;
		g_params.sleepThreshold = 0.07f;
		//
		//g_params.relaxationFactor = 0.25f;
		g_params.shapeCollisionMargin = 0.001f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);
	}

	void createLink(Vec3 translation,Mat44 Ft, Quat rotation)
	{
		int pointsOffset = g_buffers->positions.size();
		//resolve link rotation
		int trianglesOffset = g_buffers->triangles.size() / 3;
		DeformableMesh *deformable = CreateDeformableMesh("../../data/Snake/snake_soft_link.obj", "../../data/Snake/snake_soft_link.tet", translation, rotation, mRadius, mDensity, 0, mPhase);
		//int trianglesOffset = g_buffers->triangles.size() / 3;
		
		int springOffsetL1;
		int springOffsetL2;

		//
		//create deformation constraints
		//center longitudinal stretch
		for (int i = 0; i < softsnake::nCenterParticles-1; ++i)
		{
			CreateSpring(softsnake::centerParticles[i] + pointsOffset, softsnake::centerParticles[i + 1] + pointsOffset, 1e6f);
		}

		//Upper and lower bases
		for (int i = 0; i < softsnake::nBorderParticles; ++i)
		{
			for (int j = i; j < softsnake::nBorderParticles; ++j)
			{
				if(hasEdge(softsnake::lowerParticles[i] + pointsOffset, softsnake::lowerParticles[j] + pointsOffset, trianglesOffset))
					CreateSpring(softsnake::lowerParticles[i] + pointsOffset, softsnake::lowerParticles[j] + pointsOffset, 1e9f);
				if (hasEdge(softsnake::upperParticles[i] + pointsOffset, softsnake::upperParticles[j] + pointsOffset, trianglesOffset))
					CreateSpring(softsnake::upperParticles[i] + pointsOffset, softsnake::upperParticles[j] + pointsOffset, 1e9f);
			}
				
		}

		//radial constraint on chambers
		for (int ch = 0; ch < softsnake::nChambers; ++ch)
		{
			for (int c = 0; c < softsnake::nPlanes; ++c)
			{
				for (int i = 0; i < softsnake::nPlanePartricles; ++i)
				{
					for (int j = i; j < softsnake::nPlanePartricles; ++j)
					{
						CreateSpring(softsnake::ChamberPlanarParticles[ch][c][i] + pointsOffset, softsnake::ChamberPlanarParticles[ch][c][j] + pointsOffset, 1e6f);
					}
				}						
			}
		}
		
		//L1 constraints (the springs that will be stretched when applying pressure)
		//Only create one side, and update the particles to the other chamber when changing side.				
		{
			springOffsetL1 = g_buffers->springLengths.size();

			for (int j = 0; j < softsnake::nPlanePartricles; ++j)
			{
				for (int i = 0; i <  softsnake::nPlanes-2; ++i)
				{
					int p1 = pointsOffset + softsnake::ChamberPlanarParticles[0][i][j];
					int p2 = pointsOffset + softsnake::ChamberPlanarParticles[0][i+1][j];
					CreateSpring(p1, p2, 1e6f, 0.002f);
					if(0)
					{
						int s = g_buffers->springLengths.size()-1;
						printf("%.10f\n", g_buffers->springLengths[s]);
					}
				}
			}
		}

		{
			springOffsetL2 = g_buffers->springLengths.size();

			for (int j = 0; j < softsnake::nPlanePartricles; ++j)
			{
				for (int i = 0; i < softsnake::nPlanes-2; ++i)
				{
					int p1 = pointsOffset + softsnake::ChamberPlanarParticles[0][i][j];
					int p2 = pointsOffset + softsnake::ChamberPlanarParticles[0][i + 2][j];
					CreateSpring(p1, p2, 1e6f, 0.002f);
					if (0)
					{
						int s = g_buffers->springLengths.size() - 1;
						printf("%.10f\n", g_buffers->springLengths[s]);
					}
				}
			}
		}	

		//Rigid Body				
		addRigidBodyComponent(translation, Ft, rotation, true, pointsOffset);
		SnakeLink sl;
		sl.mDeformable = deformable;
		sl.mparticlesOffset = pointsOffset;						
		sl.mSpringsL1Offset = springOffsetL1;
		sl.mSpringsL2Offset = springOffsetL2;

		sl.mRigidIndex = g_buffers->rigidBodies.size() - 3;
		mLinks.push_back(sl);
	}

	void addRigidBodyComponent(Vec3 translation, Mat44 Ft, Quat rotation,bool connect = false,int pointsOffset = 0)
	{
		{
			float density = 100.0f;
			Mesh* rigidMesh = ImportMesh("../../data/Snake/Snake_rigid.ply");
			rigidMesh->Transform(ScaleMatrix(0.0008f));
			NvFlexTriangleMeshId shapeId = CreateTriangleMesh(rigidMesh, 0.000f);
			NvFlexRigidShape shape;
			NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), shapeId, NvFlexMakeRigidPose(0, 0), 1.f, 1.f, 1.f);
			shape.filter = 0x1<<(mLinks.size()) + 1;
			shape.material.friction = 1.0f;
			shape.material.torsionFriction = 0.1;
			shape.material.rollingFriction = 0.0f;
			shape.thickness = 0.0005f;

			NvFlexRigidBody body;
			//Vec3 Offset(0, -.007f, -0.0455f);
			//Vec3 Offset(0, -.013f, -0.0455f);
			//Vec3 Offset(0, -.0060f, 0.005f);
			Vec3 Offset(0, -.0060f, 0.0115F);
			Mat44 OffsetTransform = Ft*TranslationMatrix(Point3(Offset));
			Offset = Vec3(OffsetTransform.GetTranslation());
			NvFlexMakeRigidBody(g_flexLib, &body, Offset + translation, rotation, &shape, &density, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			int bodyid = g_buffers->rigidBodies.size() - 1;
			if (connect)
			{
				for (int i = 0; i < 31; ++i)
				{
					int p = pointsOffset + softsnake::lowerParticles[i];
					CreateRigidBodyToParticleAttachment(bodyid, p);
				}
			}

			NvFlexRigidPose bodyPose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[bodyid], &bodyPose);
			Transform o_T_b(bodyPose.p, bodyPose.q);
			Transform b_T_o(Inverse(o_T_b));
			for (int i = 0; i < 2; ++i)
			{
				NvFlexRigidShape wheelShape;
				//NvFlexMakeRigidSphereShape(&wheelShape, g_buffers->rigidBodies.size(), 0.005f, NvFlexMakeRigidPose(0, 0));
				NvFlexMakeRigidCapsuleShape(&wheelShape, g_buffers->rigidBodies.size(), 0.005f, 0.003f, NvFlexMakeRigidPose(0, 0));
				wheelShape.filter = 0x1<< mLinks.size() + 1;
				wheelShape.material.friction = 10.0f;
				wheelShape.material.torsionFriction = 100.1f;
				wheelShape.material.rollingFriction = 0.3f;
				wheelShape.thickness = 0.02f;
				Vec3 Offset = Vec3(i ? -0.03f : 0.03f, -0.006f, -0.005f);
				Mat44 OffsetTransform = Ft*TranslationMatrix(Point3(Offset));
				Offset = Vec3(OffsetTransform.GetTranslation());
				NvFlexRigidBody wheelBody;
				NvFlexMakeRigidBody(g_flexLib, &wheelBody, translation + Offset, rotation, &wheelShape, &density, 1);
				g_buffers->rigidBodies.push_back(wheelBody);
				g_buffers->rigidShapes.push_back(wheelShape);
				int wheelid = g_buffers->rigidBodies.size() - 1;
				//Make Revolute joint for wheel
				NvFlexRigidJoint joint;
				//get relative pose from wheel to Rigid body
				NvFlexRigidPose wheelPose;

				NvFlexGetRigidPose(&g_buffers->rigidBodies[wheelid], &wheelPose);
				Transform o_T_w(wheelPose.p, wheelPose.q);
				Transform b_T_w = b_T_o*o_T_w;
				NvFlexRigidPose Pose0;
				memcpy(Pose0.p, b_T_w.p, sizeof(Pose0.p));
				memcpy(Pose0.q, b_T_w.q, sizeof(Pose0.p));
				NvFlexRigidPose Pose1;
				memset(&Pose1, 0, sizeof(float) * 6);
				Pose1.q[3] = 1.0f;
				NvFlexMakeHingeJoint(&joint, bodyid, wheelid, Pose0, Pose1, eNvFlexRigidJointAxisTwist);						
				joint.damping[eNvFlexRigidJointAxisTwist] = 1.e10f;
				joint.lambda[eNvFlexRigidJointAxisTwist] = 1.e3f;						
				g_buffers->rigidJoints.push_back(joint);
			}
		}
	}

	void connectLinks()
	{
		for (int firstLink = 0; firstLink < mLinks.size(); ++firstLink)
		{
			int secondLink = firstLink + 1;
			int bodyid = 0;
			if (secondLink == mLinks.size())
			{
				bodyid = g_buffers->rigidBodies.size() - 3;
			}
			else
			{
				bodyid = mLinks[secondLink].mRigidIndex;
			}
		
			for (int i = 0; i < 31; ++i)
			{
				int p = mLinks[firstLink].mparticlesOffset + softsnake::upperParticles[i];							
				CreateRigidBodyToParticleAttachment(bodyid, p);
			}	
		}
	}

	virtual void Update()
	{
		mAngle += k2Pi * mOmega * g_dt;

		for (int i = 0; i < mLinks.size(); ++i)
		{
			float pressure = Clamp<float>(sin(mAngle + DegToRad(mAlpha)*i) + mOffset, -1.0f, 1.0f) * mAmplitude;
			//float pressure = mPressures[i];
			updateLinkPressure(i, pressure);
			UpdateDeformableMesh(mLinks[i].mDeformable);
		}
		mAngle = fmod(mAngle, k2Pi);
	}

	void updateLinkPressure(int link, float pressure)
	{
		int chamber = pressure > 0.f;
		pressure = abs(pressure);
		int pointsOffset = mLinks[link].mparticlesOffset;
		{
			int springOffsetL1 = mLinks[link].mSpringsL1Offset;

			for (int j = 0; j < softsnake::nPlanePartricles; ++j)
			{
				for (int i = 0; i < softsnake::nPlanes - 1; ++i)
				{
					int p1 = pointsOffset + softsnake::ChamberPlanarParticles[chamber][i][j];
					int p2 = pointsOffset + softsnake::ChamberPlanarParticles[chamber][i + 1][j];
					int springIndex = springOffsetL1 + j*(softsnake::nPlanes-1) + i;

					g_buffers->springIndices[springIndex * 2] = p1;
					g_buffers->springIndices[springIndex * 2 + 1] = p2;
					g_buffers->springStiffness[springIndex] = Lerp(1.e+3f, 1.e+9f, pressure / softsnake::maxPressure);
					if (j < softsnake::nPlanePartricles / 2)
						g_buffers->springLengths[springIndex] = softsnake::chamberSpringLenghtL1 + pressure*softsnake::SpringScaleOuter*0.001f;
					else
						g_buffers->springLengths[springIndex] = softsnake::chamberSpringLenghtL1 + pressure*softsnake::SpringScaleInner*0.001f;
				}
			}

			int springOffsetL2 = mLinks[link].mSpringsL2Offset;

			for (int j = 0; j < softsnake::nPlanePartricles; ++j)
			{
				for (int i = 0; i < softsnake::nPlanes - 2; ++i)
				{
					int p1 = pointsOffset + softsnake::ChamberPlanarParticles[chamber][i][j];
					int p2 = pointsOffset + softsnake::ChamberPlanarParticles[chamber][i + 2][j];
					int springIndex = springOffsetL2 + j*(softsnake::nPlanes-2) + i;
					g_buffers->springIndices[springIndex * 2] = p1;
					g_buffers->springIndices[springIndex * 2 + 1] = p2;
					g_buffers->springStiffness[springIndex] = Lerp(1.e+3f, 1.e+9f, pressure / softsnake::maxPressure);
					if (j < softsnake::nPlanePartricles / 2)
						g_buffers->springLengths[springIndex] = softsnake::chamberSpringLenghtL2 + pressure*softsnake::SpringScaleOuter*0.002f;
					else
						g_buffers->springLengths[springIndex] = softsnake::chamberSpringLenghtL2 + pressure*softsnake::SpringScaleInner*0.002f;
				}
			}
		}
	}

	virtual void PostUpdate()
	{
		NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springLengths.size());				
	}

	virtual void Sync()
	{
		if (g_buffers->tetraIndices.size())
		{
			NvFlexSetFEMGeometry(g_solver, g_buffers->tetraIndices.buffer, g_buffers->tetraRestPoses.buffer, g_buffers->tetraMaterials.buffer, g_buffers->tetraMaterials.size());
		}
	}

	virtual void DoGui()
	{
		//imguiSlider("Correction", &mForceFix, 0, 100.0f, 1);
		//for (int i = 0; i < 4; ++i)
		//{
		//	std::ostringstream title;
		//	title << "Link " << i << " pressure";
		//	imguiSlider(title.str().c_str(), &mPressures[i], -9.00f, 9.0f, 0.001f);
		//	
		//}

		imguiSlider("Omega", &mOmega, 0.00f, 12.0f, 1.00f);
		imguiSlider("Alpha", &mAlpha, 0.00f, 180.0f, 1.00f);
		imguiSlider("Offset", &mOffset, -.5f, .5f, 0.001f);
		imguiSlider("Amplitude", &mAmplitude, 0.0f, 8.0f, 0.001f);

		if (imguiCheck("Draw Mesh", mDrawMesh))
			mDrawMesh = !mDrawMesh;
	}

	virtual void Draw(int pass)
	{
		float sum = 0.0f;
		if (mDrawMesh)
		{
			// visual mesh
			for (auto link : mLinks)
				DrawDeformableMesh(link.mDeformable);
		}
		else
		{
			//if (pass == 1)
			{
				mesh.m_positions.resize(g_buffers->positions.size());
				mesh.m_normals.resize(g_buffers->normals.size());
				mesh.m_colours.resize(g_buffers->positions.size());
				mesh.m_indices.resize(g_buffers->triangles.size());

				for (int i = 0; i < g_buffers->triangles.size(); ++i)
					mesh.m_indices[i] = g_buffers->triangles[i];

				float rangeMin = FLT_MAX;
				float rangeMax = -FLT_MAX;

				vector<Vec2> averageStress(mesh.m_positions.size());

				// calculate average Von-Mises stress on each vertex for visualization
				for (int i = 0; i < g_buffers->tetraIndices.size(); i += 4)
				{
					float vonMises = fabsf(g_buffers->tetraStress[i / 4]);

					sum += vonMises;

					//printf("%f\n", vonMises);

					averageStress[g_buffers->tetraIndices[i + 0]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i + 1]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i + 2]] += Vec2(vonMises, 1.0f);
					averageStress[g_buffers->tetraIndices[i + 3]] += Vec2(vonMises, 1.0f);

					rangeMin = Min(rangeMin, vonMises);
					rangeMax = Max(rangeMax, vonMises);
				}

				//printf("%f %f\n", rangeMin,rangeMax);

				rangeMin = 0.0f; //Min(rangeMin, vonMises);
				rangeMax = 0.5f; //Max(rangeMax, vonMises);

				for (int i = 0; i < g_buffers->positions.size(); ++i)
				{
					mesh.m_normals[i] = Vec3(g_buffers->normals[i]);
					mesh.m_positions[i] = Point3(g_buffers->positions[i]) + Vec3(g_buffers->normals[i])*g_params.collisionDistance*1.5f;

					mesh.m_colours[i] = BourkeColorMap(rangeMin, rangeMax, averageStress[i].x / averageStress[i].y);
				}
			}

			DrawMesh(&mesh, g_renderMaterials[0]);

			/*
			if (pass == 0)
			{

				SetFillMode(true);

				DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[3], 0.001f);

				SetFillMode(false);

			}
			*/
		}
		BeginLines();
		//
		/*const Vec4 Color(1, 0, 0);
		for (int i = 0; i < g_buffers->springLengths.size(); ++i)
		{
			uint32_t p1 = g_buffers->springIndices[i * 2];
			uint32_t p2 = g_buffers->springIndices[i * 2+1];
			DrawLine(Vec3(g_buffers->positions[p1]), Vec3(g_buffers->positions[p2]), Vec4(1, 0, 0, 0));
		}
		*/

		int springOffsetL1 = mLinks[0].mSpringsL1Offset;
		for (int j = 0; j < softsnake::nPlanePartricles; ++j)
		{
			for (int i = 0; i < softsnake::nPlanes - 1; ++i)
			{
				int s = springOffsetL1 + j*(softsnake::nPlanes - 1) + i;
				uint32_t p1 = g_buffers->springIndices[s * 2];
				uint32_t p2 = g_buffers->springIndices[s * 2 + 1];
				DrawLine(Vec3(g_buffers->positions[p1]), Vec3(g_buffers->positions[p2]), Vec4(0.f, 1.f, 0.f, 0.f));
			}
		}

		int springOffsetL2 = mLinks[0].mSpringsL2Offset;
		for (int j = 0; j < softsnake::nPlanePartricles; ++j)
		{
			for (int i = 0; i < softsnake::nPlanes - 2; ++i)
			{
				int s = springOffsetL2 + j*(softsnake::nPlanes - 2) + i;
				uint32_t p1 = g_buffers->springIndices[s * 2];
				uint32_t p2 = g_buffers->springIndices[s * 2 + 1];
				DrawLine(Vec3(g_buffers->positions[p1]), Vec3(g_buffers->positions[p2]), Vec4(1.f, 0.f, 0.f, 0.f));
			}
		}

		//for (uint32_t i = 0; i < g_buffers->rigidParticleAttachments.size(); ++i)
		//{
		//	int bodyIndex = g_buffers->rigidParticleAttachments[i].body;
		//	Vec3 bodyPos(g_buffers->rigidBodies[bodyIndex].com);
		//	Quat theta(g_buffers->rigidBodies[bodyIndex].theta);
		//	Transform T(bodyPos, theta);
		//	Mat44 transformMat = TransformMatrix(T);

		//	Vec3 localOffset(g_buffers->rigidParticleAttachments[i].localOffset);
		//	Mat44 localMat = TransformMatrix(Rotation(), Point3(localOffset));
		//	Vec3 GlobalLoc = (transformMat*localMat).GetTranslation();


		//	//int particle = g_buffers->rigidParticleAttachments[i].particle;
		//	DrawLine(g_buffers->rigidBodies[bodyIndex].com, GlobalLoc, Color);
		//}
		EndLines();
	}

	bool hasEdge(uint32_t lhp, uint32_t rhp, uint32_t start_t = 0)
	{
		bool ret = false;
		for (uint32_t i = start_t; i < (uint32_t)g_buffers->triangles.size(); i += 3)
		{
			if (g_buffers->triangles[i] == lhp || g_buffers->triangles[i + 1] == lhp || g_buffers->triangles[i + 2] == lhp)
			{
				if (g_buffers->triangles[i] == rhp || g_buffers->triangles[i + 1] == rhp || g_buffers->triangles[i + 2] == rhp)
				{
					ret = true;
					break;
				}
			}
		}
		return ret;
	}

private:
	uint32_t mNLinks{ 4 };
	float mRadius{ 0.001f };
	float mDensity{ 110.0f };
	int mPhase;

	vector<SnakeLink> mLinks;
	float mOmega;
	float mAlpha;
	float mOffset;
	float mAmplitude;
	float mAngle{ 0.0f };
	float mPressures[4];
};