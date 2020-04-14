// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2013-2017 NVIDIA Corporation. All rights reserved.

#pragma once

#include <stdarg.h>

// disable some warnings
#if _WIN32
#pragma warning(disable: 4267)  // conversion from 'size_t' to 'int', possible loss of data
#endif

float SampleSDF(const float* sdf, int dim, int x, int y, int z)
{
	assert(x < dim && x >= 0);
	assert(y < dim && y >= 0);
	assert(z < dim && z >= 0);

	return sdf[z*dim*dim + y*dim + x];
}

// return normal of signed distance field
Vec3 SampleSDFGrad(const float* sdf, int dim, int x, int y, int z)
{
	int x0 = max(x-1, 0);
	int x1 = min(x+1, dim-1);

	int y0 = max(y-1, 0);
	int y1 = min(y+1, dim-1);

	int z0 = max(z-1, 0);
	int z1 = min(z+1, dim-1);

	float dx = (SampleSDF(sdf, dim, x1, y, z) - SampleSDF(sdf, dim, x0, y, z))*(dim*0.5f);
	float dy = (SampleSDF(sdf, dim, x, y1, z) - SampleSDF(sdf, dim, x, y0, z))*(dim*0.5f);
	float dz = (SampleSDF(sdf, dim, x, y, z1) - SampleSDF(sdf, dim, x, y, z0))*(dim*0.5f);

	return Vec3(dx, dy, dz);
}

void GetParticleBounds(Vec3& lower, Vec3& upper)
{
	lower = Vec3(FLT_MAX);
	upper = Vec3(-FLT_MAX);

	for (int i=0; i < g_buffers->positions.size(); ++i)
	{
		lower = Min(Vec3(g_buffers->positions[i]), lower);
		upper = Max(Vec3(g_buffers->positions[i]), upper);
	}
}


void CreateParticleGrid(Vec3 lower, int dimx, int dimy, int dimz, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)
{
	if (rigid && g_buffers->shapeMatchingIndices.empty())
		g_buffers->shapeMatchingOffsets.push_back(0);

	for (int x = 0; x < dimx; ++x)
	{
		for (int y = 0; y < dimy; ++y)
		{
			for (int z=0; z < dimz; ++z)
			{
				if (rigid)
					g_buffers->shapeMatchingIndices.push_back(int(g_buffers->positions.size()));

				Vec3 position = lower + Vec3(float(x), float(y), float(z))*radius + RandomUnitVector()*jitter;

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);
			}
		}
	}

	if (rigid)
	{
		g_buffers->shapeMatchingCoefficients.push_back(rigidStiffness);
		g_buffers->shapeMatchingOffsets.push_back(int(g_buffers->shapeMatchingIndices.size()));
	}
}

void CreateParticleSphere(Vec3 center, int dim, float radius, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, float jitter=0.005f)
{
	if (rigid && g_buffers->shapeMatchingIndices.empty())
			g_buffers->shapeMatchingOffsets.push_back(0);

	for (int x=0; x <= dim; ++x)
	{
		for (int y=0; y <= dim; ++y)
		{
			for (int z=0; z <= dim; ++z)
			{
				float sx = x - dim*0.5f;
				float sy = y - dim*0.5f;
				float sz = z - dim*0.5f;

				if (sx*sx + sy*sy + sz*sz <= float(dim*dim)*0.25f)
				{
					if (rigid)
						g_buffers->shapeMatchingIndices.push_back(int(g_buffers->positions.size()));

					Vec3 position = center + radius*Vec3(sx, sy, sz) + RandomUnitVector()*jitter;

					g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
					g_buffers->velocities.push_back(velocity);
					g_buffers->phases.push_back(phase);
				}
			}
		}
	}

	if (rigid)
	{
		g_buffers->shapeMatchingCoefficients.push_back(rigidStiffness);
		g_buffers->shapeMatchingOffsets.push_back(int(g_buffers->shapeMatchingIndices.size()));
	}
}

void CreateSpring(int i, int j, float stiffness, float give=0.0f)
{
	g_buffers->springIndices.push_back(i);
	g_buffers->springIndices.push_back(j);
	g_buffers->springLengths.push_back((1.0f+give)*Length(Vec3(g_buffers->positions[i])-Vec3(g_buffers->positions[j])));
	g_buffers->springStiffness.push_back(stiffness);	
}

void CreateRigidBodyToParticleAttachment(int rigidIndex, int particleIndex, float x, float y, float z)
{
	NvFlexRigidParticleAttachment attachment;
	attachment.body = rigidIndex;
	attachment.particle = particleIndex;
	attachment.localOffset[0] = x;
	attachment.localOffset[1] = y;
	attachment.localOffset[2] = z;
	g_buffers->rigidParticleAttachments.push_back(attachment);
}

void CreateRigidBodyToParticleAttachment(int rigidIndex, int particleIndex)
{
	NvFlexRigidParticleAttachment attachment;
	attachment.body = rigidIndex;
	attachment.particle = particleIndex;
	NvFlexRigidPose pose;
	NvFlexGetRigidPose(&g_buffers->rigidBodies[rigidIndex], &pose);
	Transform o_T_p(Vec3(pose.p), Quat(pose.q));
	Transform b_T_o(Inverse(o_T_p));
	Vec3 offset(Rotate(b_T_o.q, Vec3(g_buffers->positions[particleIndex])) + b_T_o.p);
	attachment.localOffset[0] = offset.x;
	attachment.localOffset[1] = offset.y;
	attachment.localOffset[2] = offset.z;

	g_buffers->rigidParticleAttachments.push_back(attachment);
}
void CreateParticleShape(const Mesh* srcMesh, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
{
	if (rigid && g_buffers->shapeMatchingIndices.empty())
		g_buffers->shapeMatchingOffsets.push_back(0);

	if (!srcMesh)
		return;

	// duplicate mesh
	Mesh mesh;
	mesh.AddMesh(*srcMesh);

	int startIndex = int(g_buffers->positions.size());

	{
		mesh.Transform(RotationMatrix(rotation, Vec3(0.0f, 1.0f, 0.0f)));

		Vec3 meshLower, meshUpper;
		mesh.GetBounds(meshLower, meshUpper);

		Vec3 edges = meshUpper-meshLower;
		float maxEdge = max(max(edges.x, edges.y), edges.z);

		// put mesh at the origin and scale to specified size
		Matrix44 xform = ScaleMatrix(scale/maxEdge)*TranslationMatrix(Point3(-meshLower));

		mesh.Transform(xform);
		mesh.GetBounds(meshLower, meshUpper);

		// recompute expanded edges
		edges = meshUpper-meshLower;
		maxEdge = max(max(edges.x, edges.y), edges.z);

		// tweak spacing to avoid edge cases for particles laying on the boundary
		// just covers the case where an edge is a whole multiple of the spacing.
		float spacingEps = spacing*(1.0f - 1e-4f);

		// make sure to have at least one particle in each dimension
		int dx, dy, dz;
		dx = spacing > edges.x ? 1 : int(edges.x/spacingEps);
		dy = spacing > edges.y ? 1 : int(edges.y/spacingEps);
		dz = spacing > edges.z ? 1 : int(edges.z/spacingEps);

		int maxDim = max(max(dx, dy), dz);

		// expand border by two voxels to ensure adequate sampling at edges
		meshLower -= 2.0f*Vec3(spacing);
		meshUpper += 2.0f*Vec3(spacing);
		maxDim += 4;

		vector<uint32_t> voxels(maxDim*maxDim*maxDim);

		// we shift the voxelization bounds so that the voxel centers
		// lie symmetrically to the center of the object. this reduces the 
		// chance of missing features, and also better aligns the particles
		// with the mesh
		Vec3 meshOffset;
		meshOffset.x = 0.5f * (spacing - (edges.x - (dx-1)*spacing));
		meshOffset.y = 0.5f * (spacing - (edges.y - (dy-1)*spacing));
		meshOffset.z = 0.5f * (spacing - (edges.z - (dz-1)*spacing));
		meshLower -= meshOffset;

		//Voxelize(*mesh, dx, dy, dz, &voxels[0], meshLower - Vec3(spacing*0.05f) , meshLower + Vec3(maxDim*spacing) + Vec3(spacing*0.05f));
		Voxelize((const Vec3*)&mesh.m_positions[0], mesh.m_positions.size(), (const int*)&mesh.m_indices[0], mesh.m_indices.size(), maxDim, maxDim, maxDim, &voxels[0], meshLower, meshLower + Vec3(maxDim*spacing));

		vector<int> indices(maxDim*maxDim*maxDim);
		vector<float> sdf(maxDim*maxDim*maxDim);
		MakeSDF(&voxels[0], maxDim, maxDim, maxDim, &sdf[0]);

		for (int x=0; x < maxDim; ++x)
		{
			for (int y=0; y < maxDim; ++y)
			{
				for (int z=0; z < maxDim; ++z)
				{
					const int index = z*maxDim*maxDim + y*maxDim + x;

					// if voxel is marked as occupied the add a particle
					if (voxels[index])
					{
						if (rigid)
							g_buffers->shapeMatchingIndices.push_back(int(g_buffers->positions.size()));

						Vec3 position = lower + meshLower + spacing*Vec3(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f) + RandomUnitVector()*jitter;

						 // normalize the sdf value and transform to world scale
						Vec3 n = SafeNormalize(SampleSDFGrad(&sdf[0], maxDim, x, y, z));
						float d = sdf[index]*maxEdge;

						if (rigid)
							g_buffers->shapeMatchingLocalNormals.push_back(Vec4(n, d));

						// track which particles are in which cells
						indices[index] = g_buffers->positions.size();

						g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));						
						g_buffers->velocities.push_back(velocity);
						g_buffers->phases.push_back(phase);
					}
				}
			}
		}
		mesh.Transform(ScaleMatrix(1.0f + skinExpand)*TranslationMatrix(Point3(-0.5f*(meshUpper+meshLower))));
		mesh.Transform(TranslationMatrix(Point3(lower + 0.5f*(meshUpper+meshLower))));	
	
	
		if (springStiffness > 0.0f)
		{
			// construct cross link springs to occupied cells
			for (int x=0; x < maxDim; ++x)
			{
				for (int y=0; y < maxDim; ++y)
				{
					for (int z=0; z < maxDim; ++z)
					{
						const int centerCell = z*maxDim*maxDim + y*maxDim + x;

						// if voxel is marked as occupied the add a particle
						if (voxels[centerCell])
						{
							const int width = 1;

							// create springs to all the neighbors within the width
							for (int i=x-width; i <= x+width; ++i)
							{
								for (int j=y-width; j <= y+width; ++j)
								{
									for (int k=z-width; k <= z+width; ++k)
									{
										const int neighborCell = k*maxDim*maxDim + j*maxDim + i;

										if (neighborCell > 0 && neighborCell < int(voxels.size()) && voxels[neighborCell] && neighborCell != centerCell)
										{
											CreateSpring(indices[neighborCell], indices[centerCell], springStiffness);
										}
									}
								}
							}
						}
					}
				}
			}
		}

	}
	

	if (skin)
	{
		g_buffers->shapeMatchingMeshSize.push_back(mesh.GetNumVertices());

		int startVertex = 0;

		if (!g_mesh)
			g_mesh = new Mesh();

		// append to mesh
		startVertex = g_mesh->GetNumVertices();

		g_mesh->Transform(TranslationMatrix(Point3(skinOffset)));
		g_mesh->AddMesh(mesh);

		const Colour colors[7] = 
		{
			Colour(0.0f, 0.5f, 1.0f),
			Colour(0.797f, 0.354f, 0.000f),			
			Colour(0.000f, 0.349f, 0.173f),
			Colour(0.875f, 0.782f, 0.051f),
			Colour(0.01f, 0.170f, 0.453f),
			Colour(0.673f, 0.111f, 0.000f),
			Colour(0.612f, 0.194f, 0.394f) 
		};

		for (uint32_t i=startVertex; i < g_mesh->GetNumVertices(); ++i)
		{
			int indices[g_numSkinWeights] = { -1, -1, -1, -1 };
			float distances[g_numSkinWeights] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
			
			if (LengthSq(color) == 0.0f)
				g_mesh->m_colours[i] = 1.25f*colors[((unsigned int)(phase))%7];
			else
				g_mesh->m_colours[i] = Colour(color);

			// find closest n particles
			for (int j=startIndex; j < g_buffers->positions.size(); ++j)
			{
				float dSq = LengthSq(Vec3(g_mesh->m_positions[i])-Vec3(g_buffers->positions[j]));

				// insertion sort
				int w=0;
				for (; w < 4; ++w)
					if (dSq < distances[w])
						break;
				
				if (w < 4)
				{
					// shuffle down
					for (int s=3; s > w; --s)
					{
						indices[s] = indices[s-1];
						distances[s] = distances[s-1];
					}

					distances[w] = dSq;
					indices[w] = int(j);				
				}
			}

			// weight particles according to distance
			float wSum = 0.0f;

			for (int w=0; w < 4; ++w)
			{				
				// convert to inverse distance
				distances[w] = 1.0f/(0.1f + powf(distances[w], .125f));

				wSum += distances[w];

			}

			float weights[4];
			for (int w=0; w < 4; ++w)
				weights[w] = distances[w]/wSum;

			for (int j=0; j < 4; ++j)
			{
				g_meshSkinIndices.push_back(indices[j]);
				g_meshSkinWeights.push_back(weights[j]);
			}
		}
	}

	if (rigid)
	{
		g_buffers->shapeMatchingCoefficients.push_back(rigidStiffness);
		g_buffers->shapeMatchingOffsets.push_back(int(g_buffers->shapeMatchingIndices.size()));
	}
}

// wrapper to create shape from a filename
void CreateParticleShape(const char* filename, Vec3 lower, Vec3 scale, float rotation, float spacing, Vec3 velocity, float invMass, bool rigid, float rigidStiffness, int phase, bool skin, float jitter=0.005f, Vec3 skinOffset=0.0f, float skinExpand=0.0f, Vec4 color=Vec4(0.0f), float springStiffness=0.0f)
{
	Mesh* mesh = ImportMesh(filename);
	if (mesh)
		CreateParticleShape(mesh, lower, scale, rotation, spacing, velocity, invMass, rigid, rigidStiffness, phase, skin, jitter, skinOffset, skinExpand, color, springStiffness);
	
	delete mesh;
}

void SkinMesh()
{
	if (g_mesh)
	{
		int startVertex = 0;

		for (int r=0; r < g_buffers->shapeMatchingRotations.size(); ++r)
		{
			const Matrix33 rotation = g_buffers->shapeMatchingRotations[r];
			const int numVertices = g_buffers->shapeMatchingMeshSize[r];

			for (int i=startVertex; i < numVertices+startVertex; ++i)
			{
				Vec3 skinPos;

				for (int w=0; w < 4; ++w)
				{
					// small shapes can have < 4 particles
					if (g_meshSkinIndices[i*4+w] > -1)
					{
						assert(g_meshSkinWeights[i*4+w] < FLT_MAX);

						int index = g_meshSkinIndices[i*4+w];
						float weight = g_meshSkinWeights[i*4+w];

						skinPos += (rotation*(g_meshRestPositions[i]-Point3(g_buffers->restPositions[index])) + Vec3(g_buffers->positions[index]))*weight;
					}
				}

				g_mesh->m_positions[i] = Point3(skinPos);
			}

			startVertex += numVertices;
		}

		g_mesh->CalculateNormals();
	}
}

void AddBox(Vec3 halfEdge = Vec3(2.0f), Vec3 center=Vec3(0.0f), Quat quat=Quat(), bool dynamic=false, int channels=eNvFlexPhaseShapeChannelMask)
{
	// transform
	g_buffers->shapePositions.push_back(Vec4(center.x, center.y, center.z, 0.0f));
	g_buffers->shapeRotations.push_back(quat);

	g_buffers->shapePrevPositions.push_back(g_buffers->shapePositions.back());
	g_buffers->shapePrevRotations.push_back(g_buffers->shapeRotations.back());

	NvFlexCollisionGeometry geo;
	geo.box.halfExtents[0] = halfEdge.x;
	geo.box.halfExtents[1] = halfEdge.y;
	geo.box.halfExtents[2] = halfEdge.z;

	g_buffers->shapeGeometry.push_back(geo);
	g_buffers->shapeFlags.push_back(NvFlexMakeShapeFlagsWithChannels(eNvFlexShapeBox, dynamic, channels));
}

void PopBox(int num){
	g_buffers->shapePositions.resize(g_buffers->shapePositions.size() - num);
	g_buffers->shapeRotations.resize(g_buffers->shapeRotations.size() - num);
	g_buffers->shapePrevPositions.resize(g_buffers->shapePrevPositions.size() - num);
	g_buffers->shapePrevRotations.resize(g_buffers->shapePrevRotations.size() - num);
	g_buffers->shapeGeometry.resize(g_buffers->shapeGeometry.size() - num);
	g_buffers->shapeFlags.resize(g_buffers->shapeFlags.size() - num);
}

// helper that creates a plinth whose center matches the particle bounds
void AddPlinth()
{
	Vec3 lower, upper;
	GetParticleBounds(lower, upper);

	Vec3 center = (lower+upper)*0.5f;
	center.y = 0.5f;

	AddBox(Vec3(2.0f, 0.5f, 2.0f), center);
}

void AddSphere(float radius, Vec3 position, Quat rotation)
{
	NvFlexCollisionGeometry geo;
	geo.sphere.radius = radius;
	g_buffers->shapeGeometry.push_back(geo);

	g_buffers->shapePositions.push_back(Vec4(position, 0.0f));
	g_buffers->shapeRotations.push_back(rotation);

	g_buffers->shapePrevPositions.push_back(g_buffers->shapePositions.back());
	g_buffers->shapePrevRotations.push_back(g_buffers->shapeRotations.back());

	int flags = NvFlexMakeShapeFlags(eNvFlexShapeSphere, false);
	g_buffers->shapeFlags.push_back(flags);
}

// creates a capsule aligned to the local x-axis with a given radius
void AddCapsule(float radius, float halfHeight, Vec3 position, Quat rotation)
{
	NvFlexCollisionGeometry geo;
	geo.capsule.radius = radius;
	geo.capsule.halfHeight = halfHeight;

	g_buffers->shapeGeometry.push_back(geo);

	g_buffers->shapePositions.push_back(Vec4(position, 0.0f));
	g_buffers->shapeRotations.push_back(rotation);

	g_buffers->shapePrevPositions.push_back(g_buffers->shapePositions.back());
	g_buffers->shapePrevRotations.push_back(g_buffers->shapeRotations.back());

	int flags = NvFlexMakeShapeFlags(eNvFlexShapeCapsule, false);
	g_buffers->shapeFlags.push_back(flags);
}

void CreateSDF(const Mesh* mesh, uint32_t dim, Vec3 lower, Vec3 upper, float* sdf)
{
	if (mesh)
	{
		printf("Begin mesh voxelization\n");

		double startVoxelize = GetSeconds();

		uint32_t* volume = new uint32_t[dim*dim*dim];
		Voxelize((const Vec3*)&mesh->m_positions[0], mesh->m_positions.size(), (const int*)&mesh->m_indices[0], mesh->m_indices.size(), dim, dim, dim, volume, lower, upper);

		printf("End mesh voxelization (%.2fs)\n", (GetSeconds()-startVoxelize));
	
		printf("Begin SDF gen (fast marching method)\n");

		double startSDF = GetSeconds();

		MakeSDF(volume, dim, dim, dim, sdf);

		printf("End SDF gen (%.2fs)\n", (GetSeconds()-startSDF));
	
		delete[] volume;
	}
}


void AddRandomConvex(int numPlanes, Vec3 position, float minDist, float maxDist, Vec3 axis, float angle)
{
	const int maxPlanes = 12;

	// 12-kdop
	const Vec3 directions[maxPlanes] = { 
		Vec3(1.0f, 0.0f, 0.0f),
		Vec3(0.0f, 1.0f, 0.0f),
		Vec3(0.0f, 0.0f, 1.0f),
		Vec3(-1.0f, 0.0f, 0.0f),
		Vec3(0.0f, -1.0f, 0.0f),
		Vec3(0.0f, 0.0f, -1.0f),
		Vec3(1.0f, 1.0f, 0.0f),
		Vec3(-1.0f, -1.0f, 0.0f),
		Vec3(1.0f, 0.0f, 1.0f),
		Vec3(-1.0f, 0.0f, -1.0f),
		Vec3(0.0f, 1.0f, 1.0f),
		Vec3(0.0f, -1.0f, -1.0f),
	 };

	numPlanes = Clamp(6, numPlanes, maxPlanes);

	int mesh = NvFlexCreateConvexMesh(g_flexLib);

	NvFlexVector<Vec4> planes(g_flexLib);
	planes.map();

	// create a box
	for (int i=0; i < numPlanes; ++i)
	{
		Vec4 plane = Vec4(Normalize(directions[i]), -Randf(minDist, maxDist));
		planes.push_back(plane);
	}

	g_buffers->shapePositions.push_back(Vec4(position.x, position.y, position.z, 0.0f));
	g_buffers->shapeRotations.push_back(QuatFromAxisAngle(axis, angle));

	g_buffers->shapePrevPositions.push_back(g_buffers->shapePositions.back());
	g_buffers->shapePrevRotations.push_back(g_buffers->shapeRotations.back());

	// set aabbs
	ConvexMeshBuilder builder(&planes[0]);
	builder(numPlanes);

	Vec3 lower(FLT_MAX), upper(-FLT_MAX);
	for (size_t v=0; v < builder.mVertices.size(); ++v)
	{
		const Vec3 p =  builder.mVertices[v];

		lower = Min(lower, p);
		upper = Max(upper, p);
	}

	planes.unmap();

	// todo: replace with ConvexMeshBuilderNew
	NvFlexUpdateConvexMesh(g_flexLib, mesh, NULL, NULL, planes.buffer, 0, 0, planes.size(), lower, upper);

	NvFlexCollisionGeometry geo;
	geo.convexMesh.mesh = mesh;
	geo.convexMesh.scale[0] = 1.0f;
	geo.convexMesh.scale[1] = 1.0f;
	geo.convexMesh.scale[2] = 1.0f;

	g_buffers->shapeGeometry.push_back(geo);

	int flags = NvFlexMakeShapeFlags(eNvFlexShapeConvexMesh, false);
	g_buffers->shapeFlags.push_back(flags);


	// create render mesh for convex
	Mesh renderMesh;

	for (uint32_t j = 0; j < builder.mIndices.size(); j += 3)
	{
		uint32_t a = builder.mIndices[j + 0];
		uint32_t b = builder.mIndices[j + 1];
		uint32_t c = builder.mIndices[j + 2];

		Vec3 n = Normalize(Cross(builder.mVertices[b] - builder.mVertices[a], builder.mVertices[c] - builder.mVertices[a]));
		
		int startIndex = renderMesh.m_positions.size();

		renderMesh.m_positions.push_back(Point3(builder.mVertices[a]));
		renderMesh.m_normals.push_back(n);

		renderMesh.m_positions.push_back(Point3(builder.mVertices[b]));
		renderMesh.m_normals.push_back(n);

		renderMesh.m_positions.push_back(Point3(builder.mVertices[c]));
		renderMesh.m_normals.push_back(n);

		renderMesh.m_indices.push_back(startIndex+0);
		renderMesh.m_indices.push_back(startIndex+1);
		renderMesh.m_indices.push_back(startIndex+2);
	}

	// insert into the global mesh list
	RenderMesh* gpuMesh = CreateRenderMesh(&renderMesh);
	g_convexes[mesh] = gpuMesh;
}


int AddRenderMaterial(const Vec3& color, float roughness=0.1f, float metallic=0.0f, bool hidden=false, const char* texture=NULL)
{
	RenderMaterial mat;
	mat.frontColor = color;
	mat.backColor = color;
	mat.roughness = roughness;
	mat.metallic = metallic;
	mat.hidden = hidden;

	if (texture)
		mat.colorTex = CreateRenderTexture(texture);

	g_renderMaterials.push_back(mat);

	return g_renderMaterials.size()-1;
}

int AddRenderMaterial(const RenderMaterial& mat)
{
	g_renderMaterials.push_back(mat);

	return g_renderMaterials.size()-1;
}

void CreateRandomBody(int numPlanes, Vec3 position, float minDist, float maxDist, Vec3 axis, float angle, float invMass, int phase, float stiffness)
{
	// 12-kdop
	const Vec3 directions[] = { 
		Vec3(1.0f, 0.0f, 0.0f),
		Vec3(0.0f, 1.0f, 0.0f),
		Vec3(0.0f, 0.0f, 1.0f),
		Vec3(-1.0f, 0.0f, 0.0f),
		Vec3(0.0f, -1.0f, 0.0f),
		Vec3(0.0f, 0.0f, -1.0f),
		Vec3(1.0f, 1.0f, 0.0f),
		Vec3(-1.0f, -1.0f, 0.0f),
		Vec3(1.0f, 0.0f, 1.0f),
		Vec3(-1.0f, 0.0f, -1.0f),
		Vec3(0.0f, 1.0f, 1.0f),
		Vec3(0.0f, -1.0f, -1.0f),
	 };

	numPlanes = max(4, numPlanes);

	vector<Vec4> planes;

	// create a box
	for (int i=0; i < numPlanes; ++i)
	{
		// pick random dir and distance
		Vec3 dir = Normalize(directions[i]);//RandomUnitVector();
		float dist = Randf(minDist, maxDist);

		planes.push_back(Vec4(dir.x, dir.y, dir.z, -dist));
	}

	// set aabbs
	ConvexMeshBuilder builder(&planes[0]);
	builder(numPlanes);
			
	int startIndex = int(g_buffers->positions.size());

	for (size_t v=0; v < builder.mVertices.size(); ++v)
	{
		Quat q = QuatFromAxisAngle(axis, angle);
		Vec3 p =  rotate(Vec3(q), q.w, builder.mVertices[v]) + position;

		g_buffers->positions.push_back(Vec4(p.x, p.y, p.z, invMass));
		g_buffers->velocities.push_back(0.0f);
		g_buffers->phases.push_back(phase);

		// add spring to all verts with higher index
		for (size_t i=v+1; i < builder.mVertices.size(); ++i)
		{
			int a = startIndex + int(v);
			int b = startIndex + int(i);

			g_buffers->springIndices.push_back(a);
			g_buffers->springIndices.push_back(b);
			g_buffers->springLengths.push_back(Length(builder.mVertices[v]-builder.mVertices[i]));
			g_buffers->springStiffness.push_back(stiffness);

		}
	}	

	for (size_t t=0; t < builder.mIndices.size(); ++t)
		g_buffers->triangles.push_back(startIndex + builder.mIndices[t]);		

	// lazy
	g_buffers->triangleNormals.resize(g_buffers->triangleNormals.size() + builder.mIndices.size()/3, Vec3(0.0f));
}

extern "C"
{

// fwd robust predicates defined in predicates.c, stub for now
void exactinit();
float insphere(float*, float*, float*, float*, float*);
float insphereexact(float*, float*, float*, float*, float*);
float orient3d(float*, float*, float*, float*);

} // anonymous namespace


struct ConvexMeshBuilderNew
{
	struct Edge
	{
		Edge(int i, int j)
		{
			vertices[0] = i;
			vertices[1] = j;
		}

		int vertices[2];

		int64_t GetKey() const
		{
			union
			{
				int sorted[2];
				int64_t key;
			};

			sorted[0] = Min(vertices[0], vertices[1]);
			sorted[1] = Max(vertices[0], vertices[1]);

			return key;
		}

		inline bool operator < (const Edge& e) const
		{
			return GetKey() < e.GetKey();
		}

		inline bool operator == (const Edge& e) const 
		{
			return GetKey() == e.GetKey();
		}
	};

	struct Triangle
	{
		int vertices[3];
	};

	void AddVertex(Vec3 v)
	{
		// ignore coincident vertices
		for (int i=0; i < vertices.size(); ++i)
		{
			const float eps = 1.e-5f;

			if (LengthSq(v-vertices[i]) < eps)
				return;
		}

		std::vector<Edge> openEdges;

		// test silhouette of each triangle
		for (int i=0; i < triangles.size(); )
		{
			Triangle tri = triangles[i];

			Vec3 a = vertices[tri.vertices[0]];
			Vec3 b = vertices[tri.vertices[1]];
			Vec3 c = vertices[tri.vertices[2]];

			if (orient3d(v, a, b, c) > 0.0f)
			{
				for (int e=0; e < 3; ++e)
				{
					Edge edge(tri.vertices[e], tri.vertices[(e+1)%3]);

					// triangle is visible from v, add to open edge set
					auto iter = find(openEdges.begin(), openEdges.end(), edge);
					if (iter == openEdges.end())
					{
						openEdges.push_back(edge);
					}
					else
					{
						openEdges.erase(iter);
					}
				}

				// delete visible triangles (pop from back)
				triangles[i] = triangles.back();
				triangles.resize(triangles.size()-1);
			}
			else
			{
				++i;
			}
		}

		if (openEdges.size())
		{
			// triangulate open edges
			for (int i=0; i < openEdges.size(); ++i)
			{
				Edge e = openEdges[i];
				Triangle tri = { e.vertices[0], e.vertices[1], int(vertices.size()) };

				triangles.push_back(tri);
			}

			// if at least one edge then point does not lie inside current hull
			vertices.push_back(v);
		}
	}


	void BuildFromPlanes()
	{
		// brute force compute vertices from intersection of planes

		// build from points
	}
	
	bool BuildFromPoints(Vec3* points, int numPoints)
	{
		exactinit();

		if (numPoints < 4)
			return false;

		vertices.push_back(points[0]);
		vertices.push_back(points[1]);
		vertices.push_back(points[2]);

		// check initial 3 verts are not colinear, todo: iterate until find a good starting point
		if (Length(Cross(vertices[1]-vertices[0], vertices[2]-vertices[0])) < 1.e-9f)
		{
			return false;
		}

		Triangle t0 = { 0, 1, 2};
		Triangle t1 = { 0, 2, 1};

		triangles.push_back(t0);
		triangles.push_back(t1);

		for (int i=3; i < numPoints; ++i)
		{
			AddVertex(points[i]);
		}

		// compact vertices and triangles
		std::vector<int> remap(vertices.size(), -1);
		std::vector<Vec3> newVertices;
		
		for (int i=0; i < triangles.size(); ++i)
		{
			Triangle tri = triangles[i];

			for (int v=0; v < 3; ++v)
			{
				// if we haven't inserted this vertex yet
				if (remap[tri.vertices[v]] == -1)
				{
					// add to the remap table
					remap[tri.vertices[v]] = newVertices.size();					
					newVertices.push_back(vertices[tri.vertices[v]]);	
				}

				tri.vertices[v] = remap[tri.vertices[v]];
			}

			triangles[i] = tri;
		}

		// build unique edge list
		for (int i=0; i < triangles.size(); ++i)
		{
			Triangle tri = triangles[i];

			for (int v=0; v < 3; ++v)
			{
				Edge e = { tri.vertices[v], tri.vertices[(v+1)%3]  };

				edges.push_back(e);
			}
		}

		// compact edges
		std::sort(edges.begin(), edges.end());
		auto last = std::unique(edges.begin(), edges.end());
		edges.erase(last, edges.end());

		// assign compact vertices
		vertices = newVertices;

		// compute planes
		for (int i=0; i < triangles.size(); ++i)
		{
			Vec3 a = vertices[triangles[i].vertices[0]];
			Vec3 b = vertices[triangles[i].vertices[1]];
			Vec3 c = vertices[triangles[i].vertices[2]];

			Vec3 n = Cross(b-a, c-a);
			Plane p(a, SafeNormalize(n));

			planes.push_back(p);
		}

		// compute bounds
		lower = FLT_MAX;
		upper = -FLT_MAX;

		for (int i=0; i < vertices.size(); ++i)
		{
			lower = Min(lower, vertices[i]);
			upper = Max(upper, vertices[i]);
		}

		return true;
	}

	std::vector<Vec3> vertices;
	std::vector<Vec4> planes;
	std::vector<Triangle> triangles;
	std::vector<Edge> edges;

	Vec3 lower;
	Vec3 upper;
};

NvFlexConvexMeshId CreateConvexMesh(Mesh* m)
{
	if (!m)
		return 0;

	// create convex mesh data from visual mesh vertices
	ConvexMeshBuilderNew builder;
	
	if (builder.BuildFromPoints((Vec3*)&m->m_positions[0], m->GetNumVertices()))
	{
		NvFlexVector<Vec3> vertices(g_flexLib, &builder.vertices[0], builder.vertices.size());
		NvFlexVector<Vec4> planes(g_flexLib, &builder.planes[0], builder.planes.size());
		NvFlexVector<int> edges(g_flexLib, (const int*)&builder.edges[0], builder.edges.size()*2);
	
		NvFlexConvexMeshId flexMesh = NvFlexCreateConvexMesh(g_flexLib);		
		NvFlexUpdateConvexMesh(g_flexLib, flexMesh, vertices.buffer, edges.buffer, planes.buffer, vertices.size(), edges.size()/2, planes.size(), builder.lower, builder.upper);		

		
		if (0)
		{
			// visualize convex mesh
			Mesh mesh;
			for (int i=0; i < builder.vertices.size(); ++i)
			{
				mesh.m_positions.push_back(Point3(builder.vertices[i]));
				mesh.m_normals.push_back(0.0f);
			}

			for (int i=0; i < builder.triangles.size(); ++i)
			{
				mesh.m_indices.push_back(builder.triangles[i].vertices[0]);
				mesh.m_indices.push_back(builder.triangles[i].vertices[1]);
				mesh.m_indices.push_back(builder.triangles[i].vertices[2]);
			}

			mesh.CalculateNormals();
		
			g_convexes[flexMesh] = CreateRenderMesh(&mesh);
		}
		else
		{
			// draw the original visual mesh
			g_convexes[flexMesh] = CreateRenderMesh(m);
		}

		return flexMesh;
	}
	else
	{
		return 0;
	}
}



NvFlexTriangleMeshId CreateTriangleMesh(const Mesh* m, float dilation=0.01f)
{
	if (!m)
		return 0;

	Vec3 lower, upper;
	m->GetBounds(lower, upper);

	// weld input mesh vertices
	std::vector<int> uniqueVertices(m->GetNumVertices());
	std::vector<int> uniqueRemap(m->GetNumVertices());

	const float eps = 1.e-5f;

	int numUnique = NvFlexExtCreateWeldedMeshIndices((const float*)&m->m_positions[0], m->GetNumVertices(), &uniqueVertices[0], &uniqueRemap[0], eps);

	std::vector<Vec3> weldedVertices;
	std::vector<int> weldedIndices;

	// build unique vertex list
	for (int i=0; i < numUnique; ++i)
		weldedVertices.push_back(Vec3(m->m_positions[uniqueVertices[i]]));

	// remap triangle indices
	for (int i=0; i < int(m->GetNumFaces())*3; ++i)
		weldedIndices.push_back(uniqueRemap[m->m_indices[i]]);

	// assign mesh features
	NvFlexExtMeshAdjacency* meshAdj = NvFlexExtCreateMeshAdjacency((float*)&weldedVertices[0], weldedVertices.size(), sizeof(Vec3), &weldedIndices[0], weldedIndices.size()/3, true);

	// dilate welded mesh
	if (dilation != 0.0f)
	{
		std::vector<Vec3> dilatedVertices(weldedVertices.size());

		NvFlexExtDilateMesh((float*)&weldedVertices[0], weldedVertices.size(), &weldedIndices[0], weldedIndices.size()/3, dilation, (float*)&dilatedVertices[0]);

		weldedVertices = dilatedVertices;
	}

	NvFlexVector<Vec3> positions(g_flexLib);
	NvFlexVector<int> indices(g_flexLib);
	NvFlexVector<int> features(g_flexLib);
	NvFlexVector<int> vertexAdj(g_flexLib);
	NvFlexVector<int> vertexAdjOffset(g_flexLib);
	NvFlexVector<int> vertexAdjCount(g_flexLib);
	NvFlexVector<int> edgeAdj(g_flexLib);
	
	// geometry
	positions.assign(&weldedVertices[0], weldedVertices.size());	
	indices.assign(&weldedIndices[0], weldedIndices.size());

	// topology
	features.assign(meshAdj->triFeatures, meshAdj->numTris);
	vertexAdj.assign(meshAdj->vertexAdj, meshAdj->numVertexAdjs);
	vertexAdjOffset.assign(meshAdj->vertexAdjOffset, meshAdj->numVertices);
	vertexAdjCount.assign(meshAdj->vertexAdjCount, meshAdj->numVertices);
	edgeAdj.assign(meshAdj->edgeAdj, meshAdj->numTris*3);

	positions.unmap();
	indices.unmap();
	features.unmap();
	vertexAdj.unmap();
	vertexAdjOffset.unmap();
	vertexAdjCount.unmap();
	edgeAdj.unmap();

	NvFlexExtDestroyMeshAdjacency(meshAdj);

	NvFlexTriangleMeshId flexMesh = NvFlexCreateTriangleMesh(g_flexLib);
	NvFlexUpdateTriangleMesh(g_flexLib, flexMesh, 
		positions.buffer, 
		indices.buffer, 
		features.buffer, 
		vertexAdj.buffer,
		vertexAdjOffset.buffer,
		vertexAdjCount.buffer,
		edgeAdj.buffer,
		weldedVertices.size(),
		weldedIndices.size()/3, 
		(float*)&lower.x, (float*)&upper.x);

/*
	if (0)
	{
		// visualize collision geometry directly, if this is commented out then we visualize the original graphics mesh
		
		// error: no implicit conversion from Vec3 to Point3
		//m->m_positions.assign(&weldedVertices[0], &weldedVertices[0]+weldedVertices.size());

		// CAUTION: assumes that Point3 and Vec3 have the same layout (they should)
		m->m_positions.assign((const Point3*)&weldedVertices[0], (const Point3*)(&weldedVertices[0]+weldedVertices.size()));

		m->m_indices.assign(&weldedIndices[0], &weldedIndices[0]+weldedIndices.size());
		m->CalculateNormals();
	}
*/
	
	printf("Created triangle mesh shape, numVerts: %d (from %d) numTris: %d\n", int(weldedVertices.size()), int(m->GetNumVertices()), int(weldedIndices.size())/3);

	// entry in the collision->render map
	if (g_render)
	{
		g_meshes[flexMesh] = CreateRenderMesh(m);
	}

	return flexMesh;
}


int CreateRigidBodyFromMesh(const char* basePath, const Mesh* rmesh, const Mesh* cmesh, Vec3 position, Quat rotation, float thickness, float density, float friction)
{
	// ---------------------
	// collision mesh

	NvFlexTriangleMeshId meshId = CreateTriangleMesh(cmesh, 0.0f);

	NvFlexRigidShape shape;
	NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), meshId, NvFlexMakeRigidPose(0,0), 1.0f, 1.0f, 1.0f);
	shape.material.friction = friction;
	shape.material.rollingFriction = 0.0f;
	shape.material.torsionFriction = 0.0f;
	shape.user = UnionCast<void*>(AddRenderMaterial(0.5f, 0.5f, 0.0f, true));	// hide collision mesh by default
	shape.filter = 0;	
	shape.thickness = thickness;

	NvFlexRigidBody body;
	NvFlexMakeRigidBody(g_flexLib, &body, position, rotation, &shape, &density, 1);

	g_buffers->rigidBodies.push_back(body);
	g_buffers->rigidShapes.push_back(shape);

	// ---------------------
	// render mesh

	const int parentBody = g_buffers->rigidBodies.size()-1;

	RenderMesh* renderMesh = CreateRenderMesh(rmesh);

	// construct render batches
	for (size_t a=0; a < rmesh->m_materialAssignments.size(); ++a)
	{
	    MaterialAssignment assign = rmesh->m_materialAssignments[a];

	    RenderMaterial renderMaterial;
	    renderMaterial.frontColor = rmesh->m_materials[assign.material].Kd;
	    renderMaterial.backColor = rmesh->m_materials[assign.material].Kd;
	    renderMaterial.specular = rmesh->m_materials[assign.material].Ks.x;
	    renderMaterial.roughness = SpecularExponentToRoughness(rmesh->m_materials[assign.material].Ns);
	    renderMaterial.metallic = rmesh->m_materials[assign.material].metallic;

	    // load texture
	    char texturePath[2048];
	    MakeRelativePath(basePath, rmesh->m_materials[assign.material].mapKd.c_str(), texturePath);

	    renderMaterial.colorTex = CreateRenderTexture(texturePath);

	    // construct render batch for this mesh/material combination
	    RenderAttachment attach;
	    attach.parent = parentBody;
	    attach.material = renderMaterial;
	    attach.mesh = renderMesh;
	    attach.origin =  Transform();
	    attach.startTri = rmesh->m_materialAssignments[a].startTri;
	    attach.endTri = rmesh->m_materialAssignments[a].endTri;

	    g_renderAttachments.push_back(attach);
	}

	// if no material assignment then create a default grey one
	if (rmesh->m_materialAssignments.empty())
	{
	    RenderAttachment attach;
	    attach.parent = parentBody;                                       
	    attach.material.frontColor = Vec3(0.5f);
	    attach.material.backColor = Vec3(0.5f);
	    attach.material.specular = 0.5f;
	    attach.material.metallic = 0.0f;

	    attach.mesh = renderMesh;
	    attach.origin =  Transform();
	    attach.startTri = 0;
	    attach.endTri = 0;

	    g_renderAttachments.push_back(attach);
	}

	return parentBody;
}

// create a rigid body given a visual and collision mesh, returns the index of the body, color only used if there is no .mtl file
int CreateRigidBodyFromMesh(const char* visualMesh, const char* collisionMesh, Vec3 position, Quat rotation, float scale, float thickness, float density, float friction)
{
	Mesh* cmesh = ImportMesh(collisionMesh);
	Mesh* rmesh = ImportMesh(visualMesh);

	if (!cmesh) 
	{
		printf("Could not load collision mesh: %s\n", collisionMesh);
		return -1;
	}

	if (!rmesh)
	{
		printf("Could not load render mesh: %s\n", visualMesh);
		return -1;
	}

	// scale source meshes
	cmesh->Transform(ScaleMatrix(scale));
	rmesh->Transform(ScaleMatrix(scale));

	return CreateRigidBodyFromMesh(visualMesh, rmesh, cmesh, position, rotation, thickness, density, friction);
}

void AddTriangleMesh(NvFlexTriangleMeshId mesh, Vec3 translation, Quat rotation, Vec3 scale)
{
	Vec3 lower, upper;
	NvFlexGetTriangleMeshBounds(g_flexLib, mesh, lower, upper);

	NvFlexCollisionGeometry geo;
	geo.triMesh.mesh = mesh;
	geo.triMesh.scale[0] = scale.x;
	geo.triMesh.scale[1] = scale.y;
	geo.triMesh.scale[2] = scale.z;

	g_buffers->shapePositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapeRotations.push_back(Quat(rotation));
	g_buffers->shapePrevPositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapePrevRotations.push_back(Quat(rotation));
	g_buffers->shapeGeometry.push_back((NvFlexCollisionGeometry&)geo);
	g_buffers->shapeFlags.push_back(NvFlexMakeShapeFlags(eNvFlexShapeTriangleMesh, false));
}

NvFlexDistanceFieldId CreateSDF(const char* meshFile, int dim, float margin = 0.1f, float expand = 0.0f)
{
	Mesh* mesh = ImportMesh(meshFile);

	// include small margin to ensure valid gradients near the boundary
	mesh->Normalize(1.0f - margin);
	mesh->Transform(TranslationMatrix(Point3(margin, margin, margin)*0.5f));

	Vec3 lower(0.0f);
	Vec3 upper(1.0f);

	// try and load the sdf from disc if it exists
	// Begin Add Android Support
#ifdef ANDROID
	string sdfFile = string(meshFile, strlen(meshFile) - strlen(strrchr(meshFile, '.'))) + ".pfm";
#else
	string sdfFile = string(meshFile, strrchr(meshFile, '.')) + ".pfm";
#endif
	// End Add Android Support

	PfmImage pfm;
	if (!PfmLoad(sdfFile.c_str(), pfm))
	{
		pfm.m_width = dim;
		pfm.m_height = dim;
		pfm.m_depth = dim;
		pfm.m_data = new float[dim*dim*dim];

		printf("Cooking SDF: %s - dim: %d^3\n", sdfFile.c_str(), dim);

		CreateSDF(mesh, dim, lower, upper, pfm.m_data);

		PfmSave(sdfFile.c_str(), pfm);
	}

	//printf("Loaded SDF, %d\n", pfm.m_width);

	assert(pfm.m_width == pfm.m_height && pfm.m_width == pfm.m_depth);

	// cheap collision offset
	int numVoxels = int(pfm.m_width*pfm.m_height*pfm.m_depth);
	for (int i = 0; i < numVoxels; ++i)
		pfm.m_data[i] += expand;

	NvFlexVector<float> field(g_flexLib);
	field.assign(pfm.m_data, pfm.m_width*pfm.m_height*pfm.m_depth);
	field.unmap();

	// set up flex collision shape
	NvFlexDistanceFieldId sdf = NvFlexCreateDistanceField(g_flexLib);
	NvFlexUpdateDistanceField(g_flexLib, sdf, dim, dim, dim, field.buffer);

	// entry in the collision->render map
	g_fields[sdf] = CreateRenderMesh(mesh);

	delete mesh;

	return sdf;
}

void AddSDF(NvFlexDistanceFieldId sdf, Vec3 translation, Quat rotation, float width)
{
	NvFlexCollisionGeometry geo;
	geo.sdf.field = sdf;
	geo.sdf.scale = width;

	g_buffers->shapePositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapeRotations.push_back(Quat(rotation));
	g_buffers->shapePrevPositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapePrevRotations.push_back(Quat(rotation));
	g_buffers->shapeGeometry.push_back((NvFlexCollisionGeometry&)geo);
	g_buffers->shapeFlags.push_back(NvFlexMakeShapeFlags(eNvFlexShapeSDF, false));
}

inline int GridIndex(int x, int y, int dx) { return y*dx + x; }

void CreateSpringGrid(Vec3 lower, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass)
{
	int baseIndex = int(g_buffers->positions.size());

	for (int z=0; z < dz; ++z)
	{
		for (int y=0; y < dy; ++y)
		{
			for (int x=0; x < dx; ++x)
			{
				Vec3 position = lower + radius*Vec3(float(x), float(z), float(y));

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);

				if (x > 0 && y > 0)
				{
					g_buffers->triangles.push_back(baseIndex + GridIndex(x-1, y-1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y-1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));
					
					g_buffers->triangles.push_back(baseIndex + GridIndex(x-1, y-1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x-1, y, dx));

					g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
					g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
				}
			}
		}
	}	

	// horizontal
	for (int y=0; y < dy; ++y)
	{
		for (int x=0; x < dx; ++x)
		{
			int index0 = y*dx + x;

			if (x > 0)
			{
				int index1 = y*dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (x > 1 && bendStiffness > 0.0f)
			{
				int index2 = y*dx + x - 2;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}

			if (y > 0 && x < dx-1 && shearStiffness > 0.0f)
			{
				int indexDiag = (y-1)*dx + x + 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}

			if (y > 0 && x > 0 && shearStiffness > 0.0f)
			{
				int indexDiag = (y-1)*dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}
		}
	}

	// vertical
	for (int x=0; x < dx; ++x)
	{
		for (int y=0; y < dy; ++y)
		{
			int index0 = y*dx + x;

			if (y > 0)
			{
				int index1 = (y-1)*dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (y > 1 && bendStiffness > 0.0f)
			{
				int index2 = (y-2)*dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}
		}
	}	
}

void CreateSpringGrid(Vec3 lower, Quat rot, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass)
{
	int baseIndex = int(g_buffers->positions.size());

	Vec3 dirX = Rotate(rot, Vec3(1.f, 0.f, 0.f));
	Vec3 dirY = Rotate(rot, Vec3(0.f, 1.f, 0.f));
	Vec3 dirZ = Rotate(rot, Vec3(0.f, 0.f, 1.f));

	for (int z = 0; z < dz; ++z)
	{
		for (int y = 0; y < dy; ++y)
		{
			for (int x = 0; x < dx; ++x)
			{
				Vec3 position = lower + radius * (float(x) * dirX + float(z) * dirY + float(y) * dirZ);

				g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, invMass));
				g_buffers->velocities.push_back(velocity);
				g_buffers->phases.push_back(phase);

				if (x > 0 && y > 0)
				{
					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));

					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dx));
					g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y, dx));

					g_buffers->triangleNormals.push_back(Rotate(rot, Vec3(0.0f, 1.0f, 0.0f)));
					g_buffers->triangleNormals.push_back(Rotate(rot, Vec3(0.0f, 1.0f, 0.0f)));
				}
			}
		}
	}

	// horizontal
	for (int y = 0; y < dy; ++y)
	{
		for (int x = 0; x < dx; ++x)
		{
			int index0 = y*dx + x;

			if (x > 0)
			{
				int index1 = y*dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (x > 1 && bendStiffness > 0.0f)
			{
				int index2 = y*dx + x - 2;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}

			if (y > 0 && x < dx - 1 && shearStiffness > 0.0f)
			{
				int indexDiag = (y - 1)*dx + x + 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}

			if (y > 0 && x > 0 && shearStiffness > 0.0f)
			{
				int indexDiag = (y - 1)*dx + x - 1;
				CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
			}
		}
	}

	// vertical
	for (int x = 0; x < dx; ++x)
	{
		for (int y = 0; y < dy; ++y)
		{
			int index0 = y*dx + x;

			if (y > 0)
			{
				int index1 = (y - 1)*dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
			}

			if (y > 1 && bendStiffness > 0.0f)
			{
				int index2 = (y - 2)*dx + x;
				CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
			}
		}
	}
}

void CreateRope(Rope& rope, Vec3 start, Vec3 dir, float stretchStiffness, float bendingStiffness, int segments, float length, int phase, float spiralAngle=0.0f, float invmass=1.0f, float give=0.075f)
{
	rope.mIndices.push_back(int(g_buffers->positions.size()));

	g_buffers->positions.push_back(Vec4(start.x, start.y, start.z, invmass));
	g_buffers->velocities.push_back(0.0f);
	g_buffers->phases.push_back(phase);//int(g_buffers->positions.size()));
	
	Vec3 left, right;
	BasisFromVector(dir, &left, &right);

	float segmentLength = length/segments;
	Vec3 spiralAxis = dir;
	float spiralHeight = spiralAngle/(2.0f*kPi)*(length/segments);

	if (spiralAngle > 0.0f)
		dir = left;

	Vec3 p = start;

	for (int i=0; i < segments; ++i)
	{
		int prev = int(g_buffers->positions.size())-1;

		p += dir*segmentLength;

		// rotate 
		if (spiralAngle > 0.0f)
		{
			p += spiralAxis*spiralHeight;

			dir = RotationMatrix(spiralAngle, spiralAxis)*dir;
		}

		rope.mIndices.push_back(int(g_buffers->positions.size()));

		g_buffers->positions.push_back(Vec4(p.x, p.y, p.z, invmass));
		g_buffers->velocities.push_back(0.0f);
		g_buffers->phases.push_back(phase);//int(g_buffers->positions.size()));

		// stretch
		CreateSpring(prev, prev+1, stretchStiffness, give);

		// tether
		//if (i > 0 && i%4 == 0)
			//CreateSpring(prev-3, prev+1, -0.25f);
		
		// bending spring
		if (bendingStiffness > 0.0f)
		{
			if (i > 0)
				CreateSpring(prev-1, prev+1, bendingStiffness*0.5f, give);
		}
	}
}

namespace
{
	struct Tri
	{
		int a;
		int b;
		int c;

		Tri(int a, int b, int c) : a(a), b(b), c(c) {}

		bool operator < (const Tri& rhs)
		{
			if (a != rhs.a)
				return a < rhs.a;
			else if (b != rhs.b)
				return b < rhs.b;
			else
				return c < rhs.c;
		}
	};
}


namespace
{
	struct TriKey
	{
		int orig[3];
		int indices[3];

		TriKey(int a, int b, int c)		
		{
			orig[0] = a;
			orig[1] = b;
			orig[2] = c;

			indices[0] = a;
			indices[1] = b;
			indices[2] = c;

			std::sort(indices, indices+3);
		}			

		bool operator < (const TriKey& rhs) const
		{
			if (indices[0] != rhs.indices[0])
				return indices[0] < rhs.indices[0];
			else if (indices[1] != rhs.indices[1])
				return indices[1] < rhs.indices[1];
			else
				return indices[2] < rhs.indices[2];
		}
	};
}

float CreateTet(int i, int j, int k, int l, int mat)
{
	// calculate rest pose

	Vec4 x0 = g_buffers->positions[i];
	Vec4 x1 = g_buffers->positions[j];
	Vec4 x2 = g_buffers->positions[k];
	Vec4 x3 = g_buffers->positions[l];

	x1 -= Vec4(Vec3(x0), 0.0f);
	x2 -= Vec4(Vec3(x0), 0.0f);
	x3 -= Vec4(Vec3(x0), 0.0f);

	bool success;
	Matrix33 Q = Matrix33(Vec3(x1), Vec3(x2), Vec3(x3));
	Matrix33 rest = Inverse(Q, success);

	const float det = Determinant(Q);

	if (fabsf(det) <= 1.e-9f)
	{
		printf("Degenerate or inverted tet\n");
		//return true;
		
	}

	g_buffers->tetraIndices.push_back(i);
	g_buffers->tetraIndices.push_back(j);
	g_buffers->tetraIndices.push_back(k);
	g_buffers->tetraIndices.push_back(l);

	
	// save rest pose scaled by sqrt of volume for compliance formulation
	g_buffers->tetraRestPoses.push_back(rest);
	g_buffers->tetraMaterials.push_back(mat);
	
	return det/6.0f;

}

struct TetMesh
{
	// vertex positions, volume fraction in w
	std::vector<Vec4> vertices;

	// internal tetrahedra
	std::vector<int> tetraIndices;
	std::vector<Matrix33> tetraRestPoses;

	// surface tris
	std::vector<int> tris;

	Vec3 lower;
	Vec3 upper;

};

TetMesh* ImportTetMesh(const char* filename)
{
	FILE* f = fopen(filename, "r");

	char line[2048];

	if (f)		
	{
		TetMesh* mesh = new TetMesh();

		typedef std::map<TriKey, int> TriMap;
		TriMap triCount;
		
		Vec3 meshLower(FLT_MAX);
		Vec3 meshUpper(-FLT_MAX);

		bool firstTet = true;

		while (!feof(f))
		{
			if (fgets(line, 2048, f))
			{
				switch(line[0])
				{
				case '#':
					break;
				case 'v':
					{
						Vec3 pos;
						sscanf(line, "v %f %f %f", &pos.x, &pos.y, &pos.z);

						mesh->vertices.push_back(Vec4(pos.x, pos.y, pos.z, 0.0f));

						mesh->lower = Min(pos, meshLower);
						mesh->upper = Max(pos, meshUpper);
						break;
					}
				case 't':
					{
						int indices[4];
						sscanf(line, "t %d %d %d %d", &indices[0], &indices[1], &indices[2], &indices[3]);

						TriKey k1(indices[0], indices[2], indices[1]);
						triCount[k1] += 1;

						TriKey k2(indices[1], indices[2], indices[3]);
						triCount[k2] += 1;

						TriKey k3(indices[0], indices[1], indices[3]);
						triCount[k3] += 1;

						TriKey k4(indices[0], indices[3], indices[2]);
						triCount[k4] += 1;

						mesh->tetraIndices.push_back(indices[0]);
						mesh->tetraIndices.push_back(indices[1]);
						mesh->tetraIndices.push_back(indices[2]);
						mesh->tetraIndices.push_back(indices[3]);

						// calculate rest pose
						Vec4 x0 = mesh->vertices[indices[0]];
						Vec4 x1 = mesh->vertices[indices[1]];
						Vec4 x2 = mesh->vertices[indices[2]];
						Vec4 x3 = mesh->vertices[indices[3]];

						x1 -= Vec4(Vec3(x0), 0.0f);
						x2 -= Vec4(Vec3(x0), 0.0f);
						x3 -= Vec4(Vec3(x0), 0.0f);

						bool success;
						Matrix33 Q = Matrix33(Vec3(x1), Vec3(x2), Vec3(x3));
						Matrix33 rest = Inverse(Q, success);

						const float det = Determinant(Q);

						if (fabsf(det) <= 1.e-9f)
						{
							printf("Degenerate or inverted tet\n");
						}

						const float volume = det/6.0f;
						
						// add volume fraction to particles
						mesh->vertices[indices[0]].w += volume*0.25f;
						mesh->vertices[indices[1]].w += volume*0.25f;
						mesh->vertices[indices[2]].w += volume*0.25f;
						mesh->vertices[indices[3]].w += volume*0.25f;

						// save rest pose scaled by sqrt of volume for compliance formulation
						mesh->tetraRestPoses.push_back(rest);
					}
				}
			}
		}

		// add surface triangles
		for (TriMap::iterator iter=triCount.begin(); iter != triCount.end(); ++iter)
		{
			TriKey key = iter->first;

			// only output faces that are referenced by one tet (open faces)
			if (iter->second == 1)
			{
				mesh->tris.push_back(key.orig[0]);
				mesh->tris.push_back(key.orig[1]);
				mesh->tris.push_back(key.orig[2]);
			}
		}


		fclose(f);

		return mesh;
	}

	return NULL;
}


// instance a TetMesh
void CreateTetMeshInstance(const TetMesh* mesh, Vec3 translation, Quat rotation, float scale, float density, int material, int phase)
{
	const int vertOffset = g_buffers->positions.size();

	for (int i=0; i < mesh->vertices.size(); ++i)
	{
		const Vec3 pos = translation + rotation*(Vec3(mesh->vertices[i])*scale);
		float invMass = 1.0f/(mesh->vertices[i].w*density*scale*scale*scale);	// volume adjusted by scale^3

		g_buffers->positions.push_back(Vec4(pos, invMass));
		g_buffers->velocities.push_back(0.0f);
		g_buffers->phases.push_back(phase);
	}

	for (int i=0; i < mesh->tetraRestPoses.size(); ++i)
	{
		g_buffers->tetraIndices.push_back(mesh->tetraIndices[i*4+0] + vertOffset);
		g_buffers->tetraIndices.push_back(mesh->tetraIndices[i*4+1] + vertOffset);
		g_buffers->tetraIndices.push_back(mesh->tetraIndices[i*4+2] + vertOffset);
		g_buffers->tetraIndices.push_back(mesh->tetraIndices[i*4+3] + vertOffset);

		g_buffers->tetraRestPoses.push_back(mesh->tetraRestPoses[i]*(1.0f/scale));
		g_buffers->tetraMaterials.push_back(material);
	}

	for (int i=0; i < mesh->tris.size(); i += 3)
	{
		g_buffers->triangles.push_back(mesh->tris[i+0] + vertOffset);
		g_buffers->triangles.push_back(mesh->tris[i+1] + vertOffset);
		g_buffers->triangles.push_back(mesh->tris[i+2] + vertOffset);

		g_buffers->triangleNormals.push_back(0.0f);
	}
}

void CreateTetMesh(const char* filename, Vec3 lower, float scale, float density, int material, int phase)
{
	TetMesh* mesh = ImportTetMesh(filename);
	CreateTetMeshInstance(mesh, lower, Quat(), scale, density, material, phase);
	delete mesh;		
}

// finds the closest particle to a view ray
int PickParticle(Vec3 origin, Vec3 dir, Vec4* particles, int* phases, int n, float radius, float &outT)
{
	float maxDistSq = radius*radius;
	float minT = FLT_MAX;
	int minIndex = -1;

	for (int i=0; i < n; ++i)
	{
		if (phases[i] & eNvFlexPhaseFluid)
			continue;

		Vec3 delta = Vec3(particles[i])-origin;
		float t = Dot(delta, dir);

		if (t > 0.0f)
		{
			Vec3 perp = delta - t*dir;

			float dSq = LengthSq(perp);

			if (dSq < maxDistSq && t < minT)
			{
				minT = t;
				minIndex = i;
			}
		}
	}

	outT = minT;

	return minIndex;
}

// calculates the center of mass of every rigid given a set of particle positions and rigid indices
void CalculateRigidCentersOfMass(const Vec4* restPositions, int numRestPositions, const int* offsets, Vec3* translations, const int* indices, int numRigids)
{
	// To improve the accuracy of the result, first transform the restPositions to relative coordinates (by finding the mean and subtracting that from all positions)
	// Note: If this is not done, one might see ghost forces if the mean of the restPositions is far from the origin.
	Vec3 shapeOffset(0.0f);

	for (int i = 0; i < numRestPositions; i++)
	{
		shapeOffset += Vec3(restPositions[i]);
	}

	shapeOffset /= float(numRestPositions);

	for (int i=0; i < numRigids; ++i)
	{
		const int startIndex = offsets[i];
		const int endIndex = offsets[i+1];

		const int n = endIndex-startIndex;

		assert(n);

		Vec3 com;
	
		for (int j=startIndex; j < endIndex; ++j)
		{
			const int r = indices[j];

			// By subtracting shapeOffset the calculation is done in relative coordinates
			com += Vec3(restPositions[r]) - shapeOffset;
		}

		com /= float(n);

		// Add the shapeOffset to switch back to absolute coordinates
		com += shapeOffset;

		translations[i] = com;

	}
}

// calculates local space positions given a set of particle positions, rigid indices and centers of mass of the rigids
void CalculateshapeMatchingLocalPositions(const Vec4* restPositions, const int* offsets, const Vec3* translations, const int* indices, int numRigids, Vec3* localPositions)
{
	int count = 0;

	for (int i=0; i < numRigids; ++i)
	{
		const int startIndex = offsets[i];
		const int endIndex = offsets[i+1];

		assert(endIndex-startIndex);

		for (int j=startIndex; j < endIndex; ++j)
		{
			const int r = indices[j];

			localPositions[count++] = Vec3(restPositions[r]) - translations[i];
		}
	}
}

void DrawImguiString(int x, int y, Vec3 color, int align, const char* s, ...)
{
	char buf[2048];

	va_list args;

	va_start(args, s);
	vsnprintf(buf, 2048, s, args);
	va_end(args);

	imguiDrawText(x, y, align, buf, imguiRGBA((unsigned char)(color.x*255), (unsigned char)(color.y*255), (unsigned char)(color.z*255)));
}

enum 
{
	HELPERS_SHADOW_OFFSET = 1,
};

void DrawShadowedText(int x, int y, Vec3 color, int align, const char* s, ...)
{
	char buf[2048];

	va_list args;

	va_start(args, s);
	vsnprintf(buf, 2048, s, args);
	va_end(args);


	imguiDrawText(x + HELPERS_SHADOW_OFFSET, y - HELPERS_SHADOW_OFFSET, align, buf, imguiRGBA(0, 0, 0));
	imguiDrawText(x, y, align, buf, imguiRGBA((unsigned char)(color.x * 255), (unsigned char)(color.y * 255), (unsigned char)(color.z * 255)));
}

void DrawRect(float x, float y, float w, float h, Vec4 color)
{
	imguiDrawRect(x, y, w, h, imguiRGBA((unsigned char)(color.x * 255), (unsigned char)(color.y * 255), (unsigned char)(color.z * 255), (unsigned char)(color.w*255)));
}

void DrawShadowedRect(float x, float y, float w, float h, Vec3 color)
{
	imguiDrawRect(x + HELPERS_SHADOW_OFFSET, y - HELPERS_SHADOW_OFFSET, w, h, imguiRGBA(0, 0, 0));
	imguiDrawRect(x, y, w, h, imguiRGBA((unsigned char)(color.x * 255), (unsigned char)(color.y * 255), (unsigned char)(color.z * 255)));
}

void DrawLine(float x0, float y0, float x1, float y1, float r, Vec3 color)
{
	imguiDrawLine(x0, y0, x1, y1, r, imguiRGBA((unsigned char)(color.x * 255), (unsigned char)(color.y * 255), (unsigned char)(color.z * 255)));
}

void DrawShadowedLine(float x0, float y0, float x1, float y1, float r, Vec3 color)
{
	imguiDrawLine(x0 + HELPERS_SHADOW_OFFSET, y0 - HELPERS_SHADOW_OFFSET, x1 + HELPERS_SHADOW_OFFSET, y1 - HELPERS_SHADOW_OFFSET, r, imguiRGBA(0, 0, 0));
	imguiDrawLine(x0, y0, x1, y1, r, imguiRGBA((unsigned char)(color.x * 255), (unsigned char)(color.y * 255), (unsigned char)(color.z * 255)));
}

void DrawBoundingBox(const Vec3& lower, const Vec3& upper)
{
	/*
	BeginLines();
	
	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, lower.y, lower.z));
	DrawLine(Vec3(lower.x, upper.y, lower.z), Vec3(upper.x, upper.y, lower.z));
	DrawLine(Vec3(lower.x, upper.y, upper.z), Vec3(upper.x, upper.y, upper.z));
	DrawLine(Vec3(lower.x, lower.y, upper.z), Vec3(upper.x, lower.y, upper.z));

	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));
	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));
	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));
	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));
	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));
	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));
	DrawLine(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

	EndLines();
	*/
}

// Soft body support functions

Vec3 CalculateMean(const Vec3* particles, const int* indices, int numIndices)
{
	Vec3 sum;

	for (int i = 0; i < numIndices; ++i)
		sum += Vec3(particles[indices[i]]);

	if (numIndices)
		return sum / float(numIndices);
	else
		return sum;
}

float CalculateRadius(const Vec3* particles, Vec3 center, const int* indices, int numIndices)
{
	float radiusSq = 0.0f;

	for (int i = 0; i < numIndices; ++i)
	{
		float dSq = LengthSq(Vec3(particles[indices[i]]) - center);
		if (dSq > radiusSq)
			radiusSq = dSq;
	}

	return sqrtf(radiusSq);
}

struct Cluster
{
	Vec3 mean;
	float radius;

	// indices of particles belonging to this cluster
	std::vector<int> indices;
};

struct Seed
{
	int index;
	float priority;

	bool operator < (const Seed& rhs) const
	{
		return priority < rhs.priority;
	}
};

int CreateClusters(Vec3* particles, const float* priority, int numParticles, std::vector<int>& outClusterOffsets, std::vector<int>& outClusterIndices, std::vector<Vec3>& outClusterPositions, float radius, float smoothing = 0.0f)
{
	std::vector<Seed> seeds;
	std::vector<Cluster> clusters;

	// flags a particle as belonging to at least one cluster
	std::vector<bool> used(numParticles, false);

	// initialize seeds
	for (int i = 0; i < numParticles; ++i)
	{
		Seed s;
		s.index = i;
		s.priority = priority[i];

		seeds.push_back(s);
	}

	std::stable_sort(seeds.begin(), seeds.end());

	while (seeds.size())
	{
		// pick highest unused particle from the seeds list
		Seed seed = seeds.back();
		seeds.pop_back();

		if (!used[seed.index])
		{
			Cluster c;

			const float radiusSq = sqr(radius);

			// push all neighbors within radius
			for (int p = 0; p < numParticles; ++p)
			{
				float dSq = LengthSq(Vec3(particles[seed.index]) - Vec3(particles[p]));
				if (dSq <= radiusSq)
				{
					c.indices.push_back(p);

					used[p] = true;
				}
			}

			c.mean = CalculateMean(particles, &c.indices[0], c.indices.size());

			clusters.push_back(c);
		}
	}

	if (smoothing > 0.0f)
	{
		// expand clusters by smoothing radius
		float radiusSmoothSq = sqr(smoothing);

		for (int i = 0; i < int(clusters.size()); ++i)
		{
			Cluster& c = clusters[i];

			// clear cluster indices
			c.indices.resize(0);

			// push all neighbors within radius
			for (int p = 0; p < numParticles; ++p)
			{
				float dSq = LengthSq(c.mean - Vec3(particles[p]));
				if (dSq <= radiusSmoothSq)
					c.indices.push_back(p);
			}

			c.mean = CalculateMean(particles, &c.indices[0], c.indices.size());
		}
	}

	// write out cluster indices
	int count = 0;

	//outClusterOffsets.push_back(0);

	for (int c = 0; c < int(clusters.size()); ++c)
	{
		const Cluster& cluster = clusters[c];

		const int clusterSize = int(cluster.indices.size());

		// skip empty clusters
		if (clusterSize)
		{
			// write cluster indices
			for (int i = 0; i < clusterSize; ++i)
				outClusterIndices.push_back(cluster.indices[i]);

			// write cluster offset
			outClusterOffsets.push_back(outClusterIndices.size());

			// write center
			outClusterPositions.push_back(cluster.mean);

			++count;
		}
	}

	return count;
}

// creates distance constraints between particles within some distance
int CreateLinks(const Vec3* particles, int numParticles, std::vector<int>& outSpringIndices, std::vector<float>& outSpringLengths, std::vector<float>& outSpringStiffness, float radius, float stiffness = 1.0f)
{
	const float radiusSq = sqr(radius);

	int count = 0;

	for (int i = 0; i < numParticles; ++i)
	{
		for (int j = i + 1; j < numParticles; ++j)
		{
			float dSq = LengthSq(Vec3(particles[i]) - Vec3(particles[j]));

			if (dSq < radiusSq)
			{
				outSpringIndices.push_back(i);
				outSpringIndices.push_back(j);
				outSpringLengths.push_back(sqrtf(dSq));
				outSpringStiffness.push_back(stiffness);

				++count;
			}
		}
	}

	return count;
}

void CreateSkinning(const Vec3* vertices, int numVertices, const Vec3* clusters, int numClusters, float* outWeights, int* outIndices, float falloff, float maxdist)
{
	const int maxBones = 4;

	// for each vertex, find the closest n clusters
	for (int i = 0; i < numVertices; ++i)
	{
		int indices[4] = { -1, -1, -1, -1 };
		float distances[4] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
		float weights[maxBones];

		for (int c = 0; c < numClusters; ++c)
		{
			float dSq = LengthSq(vertices[i] - clusters[c]);

			// insertion sort
			int w = 0;
			for (; w < maxBones; ++w)
				if (dSq < distances[w])
					break;

			if (w < maxBones)
			{
				// shuffle down
				for (int s = maxBones - 1; s > w; --s)
				{
					indices[s] = indices[s - 1];
					distances[s] = distances[s - 1];
				}

				distances[w] = dSq;
				indices[w] = c;
			}
		}

		// weight particles according to distance
		float wSum = 0.0f;

		for (int w = 0; w < maxBones; ++w)
		{
			if (distances[w] > sqr(maxdist))
			{
				// clamp bones over a given distance to zero
				weights[w] = 0.0f;
			}
			else
			{
				// weight falls off inversely with distance
				weights[w] = 1.0f / (powf(distances[w], falloff) + 0.0001f);
			}

			wSum += weights[w];
		}

		if (wSum == 0.0f)
		{
			// if all weights are zero then just 
			// rigidly skin to the closest bone
			weights[0] = 1.0f;
		}
		else
		{
			// normalize weights
			for (int w = 0; w < maxBones; ++w)
			{
				weights[w] = weights[w] / wSum;
			}
		}

		// output
		for (int j = 0; j < maxBones; ++j)
		{
			outWeights[i*maxBones + j] = weights[j];
			outIndices[i*maxBones + j] = indices[j];
		}
	}
}


void SampleMesh(Mesh* mesh, Vec3 lower, Vec3 scale, float rotation, float radius, float volumeSampling, float surfaceSampling, std::vector<Vec3>& outPositions)
{
	if (!mesh)
		return;

	mesh->Transform(RotationMatrix(rotation, Vec3(0.0f, 1.0f, 0.0f)));

	Vec3 meshLower, meshUpper;
	mesh->GetBounds(meshLower, meshUpper);

	Vec3 edges = meshUpper - meshLower;
	float maxEdge = max(max(edges.x, edges.y), edges.z);

	// put mesh at the origin and scale to specified size
	Matrix44 xform = ScaleMatrix(scale / maxEdge)*TranslationMatrix(Point3(-meshLower));

	mesh->Transform(xform);
	mesh->GetBounds(meshLower, meshUpper);

	std::vector<Vec3> samples;

	if (volumeSampling > 0.0f)
	{
		// recompute expanded edges
		edges = meshUpper - meshLower;
		maxEdge = max(max(edges.x, edges.y), edges.z);

		// use a higher resolution voxelization as a basis for the particle decomposition
		float spacing = radius / volumeSampling;

		// tweak spacing to avoid edge cases for particles laying on the boundary
		// just covers the case where an edge is a whole multiple of the spacing.
		float spacingEps = spacing*(1.0f - 1e-4f);

		// make sure to have at least one particle in each dimension
		int dx, dy, dz;
		dx = spacing > edges.x ? 1 : int(edges.x / spacingEps);
		dy = spacing > edges.y ? 1 : int(edges.y / spacingEps);
		dz = spacing > edges.z ? 1 : int(edges.z / spacingEps);

		int maxDim = max(max(dx, dy), dz);

		// expand border by two voxels to ensure adequate sampling at edges
		meshLower -= 2.0f*Vec3(spacing);
		meshUpper += 2.0f*Vec3(spacing);
		maxDim += 4;

		vector<uint32_t> voxels(maxDim*maxDim*maxDim);

		// we shift the voxelization bounds so that the voxel centers
		// lie symmetrically to the center of the object. this reduces the 
		// chance of missing features, and also better aligns the particles
		// with the mesh
		Vec3 meshOffset;
		meshOffset.x = 0.5f * (spacing - (edges.x - (dx - 1)*spacing));
		meshOffset.y = 0.5f * (spacing - (edges.y - (dy - 1)*spacing));
		meshOffset.z = 0.5f * (spacing - (edges.z - (dz - 1)*spacing));
		meshLower -= meshOffset;

		//Voxelize(*mesh, dx, dy, dz, &voxels[0], meshLower - Vec3(spacing*0.05f) , meshLower + Vec3(maxDim*spacing) + Vec3(spacing*0.05f));
		Voxelize((const Vec3*)&mesh->m_positions[0], mesh->m_positions.size(), (const int*)&mesh->m_indices[0], mesh->m_indices.size(), maxDim, maxDim, maxDim, &voxels[0], meshLower, meshLower + Vec3(maxDim*spacing));

		// sample interior
		for (int x = 0; x < maxDim; ++x)
		{
			for (int y = 0; y < maxDim; ++y)
			{
				for (int z = 0; z < maxDim; ++z)
				{
					const int index = z*maxDim*maxDim + y*maxDim + x;

					// if voxel is marked as occupied the add a particle
					if (voxels[index])
					{
						Vec3 position = lower + meshLower + spacing*Vec3(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);

						// normalize the sdf value and transform to world scale
						samples.push_back(position);
					}
				}
			}
		}
	}

	// move back
	mesh->Transform(ScaleMatrix(1.0f)*TranslationMatrix(Point3(-0.5f*(meshUpper + meshLower))));
	mesh->Transform(TranslationMatrix(Point3(lower + 0.5f*(meshUpper + meshLower))));

	if (surfaceSampling > 0.0f)
	{
		// sample vertices
		for (int i = 0; i < int(mesh->m_positions.size()); ++i)
			samples.push_back(Vec3(mesh->m_positions[i]));

		// random surface sampling
		if (1)
		{
			for (int i = 0; i < 50000; ++i)
			{
				int t = Rand() % mesh->GetNumFaces();
				float u = Randf();
				float v = Randf()*(1.0f - u);
				float w = 1.0f - u - v;

				int a = mesh->m_indices[t * 3 + 0];
				int b = mesh->m_indices[t * 3 + 1];
				int c = mesh->m_indices[t * 3 + 2];
				
				Point3 pt = mesh->m_positions[a] * u + mesh->m_positions[b] * v + mesh->m_positions[c] * w;
				Vec3 p(pt.x,pt.y,pt.z);

				samples.push_back(p);
			}
		}
	}

	std::vector<int> clusterIndices;
	std::vector<int> clusterOffsets;
	std::vector<Vec3> clusterPositions;
	std::vector<float> priority(samples.size());

	CreateClusters(&samples[0], &priority[0], samples.size(), clusterOffsets, clusterIndices, outPositions, radius);

}

void ClearShapes()
{
	g_buffers->shapeGeometry.resize(0);
	g_buffers->shapePositions.resize(0);
	g_buffers->shapeRotations.resize(0);
	g_buffers->shapePrevPositions.resize(0);
	g_buffers->shapePrevRotations.resize(0);
	g_buffers->shapeFlags.resize(0);
}

void UpdateShapes()
{	
	// mark shapes as dirty so they are sent to flex during the next update
	g_shapesChanged = true;
}

void GetGeometryBounds(const NvFlexCollisionGeometry& geo, int type, Vec3& localLower, Vec3& localUpper)
{
	switch(type)
	{
		case eNvFlexShapeBox:
		{
			localLower = -Vec3(geo.box.halfExtents);
			localUpper = Vec3(geo.box.halfExtents);
			break;
		}
		case eNvFlexShapeSphere:
		{
			localLower = -geo.sphere.radius;
			localUpper = geo.sphere.radius;
			break;
		}
		case eNvFlexShapeCapsule:
		{
			localLower = -Vec3(geo.capsule.halfHeight, 0.0f, 0.0f) - Vec3(geo.capsule.radius);
			localUpper = Vec3(geo.capsule.halfHeight, 0.0f, 0.0f) + Vec3(geo.capsule.radius);
			break;
		}
		case eNvFlexShapeConvexMesh:
		{
			NvFlexGetConvexMeshBounds(g_flexLib, geo.convexMesh.mesh, localLower, localUpper);

			// apply instance scaling
			localLower *= geo.convexMesh.scale;
			localUpper *= geo.convexMesh.scale;
			break;
		}
		case eNvFlexShapeTriangleMesh:
		{
			NvFlexGetTriangleMeshBounds(g_flexLib, geo.triMesh.mesh, localLower, localUpper);
				
			// apply instance scaling
			localLower *= Vec3(geo.triMesh.scale);
			localUpper *= Vec3(geo.triMesh.scale);
			break;
		}
		case eNvFlexShapeSDF:
		{
			localLower = 0.0f;
			localUpper = geo.sdf.scale;
			break;
		}
	};
}

// calculates the union bounds of all the collision shapes in the scene
void GetShapeBounds(Vec3& totalLower, Vec3& totalUpper)
{
	Bounds totalBounds;

	for (int i=0; i < g_buffers->shapeFlags.size(); ++i)
	{
		NvFlexCollisionGeometry geo = g_buffers->shapeGeometry[i];

		int type = g_buffers->shapeFlags[i]&eNvFlexShapeFlagTypeMask;

		Vec3 localLower;
		Vec3 localUpper;

		GetGeometryBounds(geo, type, localLower, localUpper);

		// transform local bounds to world space
		Vec3 worldLower, worldUpper;
		TransformBounds(localLower, localUpper, Vec3(g_buffers->shapePositions[i]), g_buffers->shapeRotations[i], 1.0f, worldLower, worldUpper);

		totalBounds = Union(totalBounds, Bounds(worldLower, worldUpper));
	}

	totalLower = totalBounds.lower;
	totalUpper = totalBounds.upper;
}


NvFlexFEMMaterial IsotropicMaterialCompliance(float E, float mu, float lambda = 0.0f)
{
	NvFlexFEMMaterial c;
	memset(&c, 0, sizeof(c));
	
	float invE = 1.0f/E;
	
	c.rows[0][0] =  invE;
	c.rows[0][1] = -invE*mu;
	c.rows[0][2] = -invE*mu;

	c.rows[1][0] = -invE*mu;
	c.rows[1][1] =  invE;
	c.rows[1][2] = -invE*mu;

	c.rows[2][0] = -invE*mu;
	c.rows[2][1] = -invE*mu;
	c.rows[2][2] =  invE;

	c.rows[3][3] = invE*(1.0f+mu);
	c.rows[4][4] = invE*(1.0f+mu);
	c.rows[5][5] = invE*(1.0f+mu);

	c.damping[0] = lambda;
	c.damping[1] = lambda;
	c.damping[2] = lambda;
	c.damping[3] = lambda;
	c.damping[4] = lambda;
	c.damping[5] = lambda;

	return c;


}





int TetVertexIndex(int dimx, int dimy, int dimz, int x, int y, int z)
{
	return (dimx+1)*(dimy+1)*z + (dimx+1)*y + x;
}

typedef int (*FEMMaterialCallback)(Vec3 position);

template<int mat>
int ConstantMaterial(Vec3 position) { return mat; };


void CreateTetGrid(Transform xform, int dimx, int dimy, int dimz, float cellWidth, float cellHeight, float cellDepth, float density, FEMMaterialCallback callback, bool baseFixed=false, bool topFixed=false, bool leftFixed=false, bool rightFixed=false)
{
	const int particleOffset = g_buffers->positions.size();

	float mass = cellWidth*cellHeight*cellDepth*density;
	float invMass = 1.0f/mass;

	for (int z=0; z <= dimz; ++z)
	{
		for (int y=0; y <= dimy; ++y)
		{
			for (int x=0; x <= dimx; ++x)
			{
				Vec3 pos = Vec3(x*cellWidth, y*cellHeight, z*cellDepth);

				float m = invMass;

				if (baseFixed && y == 0)
					m = 0.0f;

				if (topFixed && y == dimy)
					m = 0.0f;

				if (leftFixed && x == 0)
					m = 0.0f;

				if (rightFixed && x == dimx)
					m = 0.0f;

				pos = TransformPoint(xform, pos);

				g_buffers->positions.push_back(Vec4(pos, m));
				g_buffers->velocities.push_back(0.0f);
				g_buffers->phases.push_back(NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter));
			}
		}
	}

	const int tetraBegin = g_buffers->tetraIndices.size();

	typedef std::map<TriKey, int> TriMap;
	TriMap triCount;

	// add tetra indices
	for (int z=0; z < dimz; ++z)
	{
		for (int y=0; y < dimy; ++y)
		{
			for (int x=0; x < dimx; ++x)
			{
				int v0 = TetVertexIndex(dimx, dimy, dimz, x, y, z) + particleOffset;
				int v1 = TetVertexIndex(dimx, dimy, dimz, x+1, y, z) + particleOffset;
				int v2 = TetVertexIndex(dimx, dimy, dimz, x+1, y, z+1) + particleOffset;
				int v3 = TetVertexIndex(dimx, dimy, dimz, x, y, z+1) + particleOffset;
				int v4 = TetVertexIndex(dimx, dimy, dimz, x, y+1, z) + particleOffset;
				int v5 = TetVertexIndex(dimx, dimy, dimz, x+1, y+1, z) + particleOffset;
				int v6 = TetVertexIndex(dimx, dimy, dimz, x+1, y+1, z+1) + particleOffset;
				int v7 = TetVertexIndex(dimx, dimy, dimz, x, y+1, z+1) + particleOffset;

				int mat = callback(Vec3(g_buffers->positions[v0]));

				if (((x & 1) ^ (y & 1) ^ (z & 1)))
				{
					CreateTet(v0, v1, v4, v3, mat);
					CreateTet(v2, v3, v6, v1, mat);
					CreateTet(v5, v4, v1, v6, mat);
					CreateTet(v7, v6, v3, v4, mat);
					CreateTet(v4, v1, v6, v3, mat);
				}
				else
				{
					CreateTet(v1, v2, v5, v0, mat);
					CreateTet(v3, v0, v7, v2, mat);
					CreateTet(v4, v7, v0, v5, mat);
					CreateTet(v6, v5, v2, v7, mat);
					CreateTet(v5, v2, v7, v0, mat);
				}

				/*
				int v0 = VertexIndex(dimx, dimy, dimz, x, y, z);
				int v1 = VertexIndex(dimx, dimy, dimz, x+1, y, z);
				int v2 = VertexIndex(dimx, dimy, dimz, x, y+1, z);
				int v3 = VertexIndex(dimx, dimy, dimz, x, y, z+1);

				int v4 = VertexIndex(dimx, dimy, dimz, x+1, y+1, z);
				int v5 = VertexIndex(dimx, dimy, dimz, x, y+1, z);
				int v6 = VertexIndex(dimx, dimy, dimz, x+1, y, z);
				int v7 = VertexIndex(dimx, dimy, dimz, x+1, y+1, z+1);

				int v8 = VertexIndex(dimx, dimy, dimz, x+1, y, z+1);
				int v9 = VertexIndex(dimx, dimy, dimz, x+1, y+1, z+1);
				int v10 = VertexIndex(dimx, dimy, dimz, x+1, y, z);
				int v11 = VertexIndex(dimx, dimy, dimz, x, y, z+1);

				int v12 = VertexIndex(dimx, dimy, dimz, x, y+1, z+1);
				int v13 = VertexIndex(dimx, dimy, dimz, x, y+1, z);
				int v14 = VertexIndex(dimx, dimy, dimz, x+1, y+1, z+1);
				int v15 = VertexIndex(dimx, dimy, dimz, x, y, z+1);
				
				int mat = callback(Vec3(g_buffers->positions[v0]));

				// 5 tetras per-cube
				CreateTet(v0, v1, v2, v3, mat);
				CreateTet(v4, v5, v6, v7, mat);
				CreateTet(v8, v9, v10, v11, mat);
				CreateTet(v12, v13, v14, v15, mat);

				int v16 = VertexIndex(dimx, dimy, dimz, x, y+1, z);
				int v17 = VertexIndex(dimx, dimy, dimz, x+1, y, z);
				int v18 = VertexIndex(dimx, dimy, dimz, x+1, y+1, z+1);
				int v19 = VertexIndex(dimx, dimy, dimz, x, y, z+1);

				// regular tet in the middle
				CreateTet(v16, v17, v18, v19, mat);
				*/
			}
		}
	}

	const int tetraEnd = g_buffers->tetraIndices.size();

	// generate external faces
	for (int i=tetraBegin; i < tetraEnd; i+=4)
	{
		int v0 = g_buffers->tetraIndices[i+0];
		int v1 = g_buffers->tetraIndices[i+1];
		int v2 = g_buffers->tetraIndices[i+2];
		int v3 = g_buffers->tetraIndices[i+3];

		// tet faces
		TriKey k1(v0, v2, v1);
		triCount[k1] += 1;

		TriKey k2(v1, v2, v3);
		triCount[k2] += 1;
				
		TriKey k3(v0, v1, v3);
		triCount[k3] += 1;

		TriKey k4(v0, v3, v2);
		triCount[k4] += 1;
	}

	for (TriMap::iterator iter=triCount.begin(); iter != triCount.end(); ++iter)
	{
		TriKey key = iter->first;

		// only output faces that are referenced by one tet (open faces)
		if (iter->second == 1)
		{
			g_buffers->triangles.push_back(key.orig[0]);
			g_buffers->triangles.push_back(key.orig[1]);
			g_buffers->triangles.push_back(key.orig[2]);
			g_buffers->triangleNormals.push_back(0.0f);
		}
	}
}


