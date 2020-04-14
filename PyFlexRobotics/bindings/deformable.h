#pragma once

struct DeformableRenderBatch
{
	int startTri;
	int endTri;

	RenderMaterial material;
};

struct DeformableMesh
{
	Mesh* hostMesh;
	TetMesh* tetMesh;

	RenderMesh* renderMesh;

	std::vector<DeformableRenderBatch> renderBatches;

	// skinning weights
	std::vector<int> parentIndices;
	std::vector<Vec3> parentWeights; // a = 1.0-u-v-w

	int particleOffset;
};


void UpdateDeformableMesh(const DeformableMesh* deformable)
{
	static std::vector<Vec3> positions;
	static std::vector<Vec3> normals;

	positions.resize(0);
	normals.resize(0);

	const Vec4* particles = &g_buffers->positions[0] + deformable->particleOffset;

	for (int i=0; i < deformable->hostMesh->m_positions.size(); ++i)
	{
		const int parent = deformable->parentIndices[i];

		Vec3 a = Vec3(particles[deformable->tetMesh->tetraIndices[parent*4+0]]);
		Vec3 b = Vec3(particles[deformable->tetMesh->tetraIndices[parent*4+1]]);
		Vec3 c = Vec3(particles[deformable->tetMesh->tetraIndices[parent*4+2]]);
		Vec3 d = Vec3(particles[deformable->tetMesh->tetraIndices[parent*4+3]]);

		if (0)
		{
			b -= a;
			c -= a;
			d -= a;

			// deformation gradient
			Matrix33 Qinv = deformable->tetMesh->tetraRestPoses[parent];
			Matrix33 F = Matrix33(b, c, d)*Qinv;

			Vec3 p = Vec3(deformable->hostMesh->m_positions[i]);
			Vec3 n = deformable->hostMesh->m_normals[i];

			bool success;

			Vec3 weights = deformable->parentWeights[i];
			p = b*weights.x + c*weights.y + d*weights.z + a;//*(1.0f-weights.x-weights.y-weights.z);//F*p + a;//F*(p-a) + a;
			
			if (Determinant(F) > 0.0f)
				n = Normalize(Transpose(Inverse(F, success))*n);

			positions.push_back(p);
			normals.push_back(n);

		}
		else
		{
			Vec3 weights = deformable->parentWeights[i];

			Vec3 p = b*weights.x + c*weights.y + d*weights.z + a*(1.0f-weights.x-weights.y-weights.z);

			positions.push_back(p);

		}

	}

	//UpdateRenderMesh(deformable->renderMesh, &positions[0], &normals[0], positions.size());

	Mesh copy(*deformable->hostMesh);
	copy.m_positions.assign(positions.begin(), positions.end());
	copy.CalculateNormals();


	UpdateRenderMesh(deformable->renderMesh, (Vec3*)&copy.m_positions[0], &copy.m_normals[0], positions.size());
}

DeformableMesh* CreateDeformableMesh(const char* visualFile, const char* tetFile, Vec3 translation, Quat rotation, float scale, float density, int material, int phase)
{
	DeformableMesh* m = new DeformableMesh();

	m->hostMesh = ImportMesh(visualFile);
	m->tetMesh = ImportTetMesh(tetFile);

	// build skinning weights
	for (int i=0; i < m->hostMesh->m_positions.size(); ++i)
	{
		const Vec3 p = Vec3(m->hostMesh->m_positions[i]);

		int closestIndex;
		float closestLengthSq = FLT_MAX;
		Vec3 closestBary;

		// find closest tetrahedron
		for (int j=0; j < m->tetMesh->tetraRestPoses.size(); j++)
		{
			// calculate bary centric coordiantes
			Vec3 a = Vec3(m->tetMesh->vertices[m->tetMesh->tetraIndices[j*4+0]]);

			Vec3 bary = m->tetMesh->tetraRestPoses[j]*(p-a);
			float baryLengthSq = LengthSq(bary) + Sqr(1.0f-bary.x-bary.y-bary.z);

			if (baryLengthSq < closestLengthSq)
			{
				closestIndex = j;
				closestLengthSq = baryLengthSq;
				closestBary = bary;
			}

			// inside tet
			float w = 1.0f-bary.x-bary.y-bary.z;

			if (bary.x >= 0.0f && bary.x <= 1.0f &&
				bary.y >= 0.0f && bary.y <= 1.0f &&
				bary.z >= 0.0f && bary.z <= 1.0f &&
				w >= 0.0f && w <= 1.0f)
				break;
		}

		m->parentIndices.push_back(closestIndex);
		m->parentWeights.push_back(closestBary);
	}

	// create bound render mesh
    RenderMesh* renderMesh = CreateRenderMesh(m->hostMesh);
    m->renderMesh = renderMesh;

    m->particleOffset = g_buffers->positions.size();

    CreateTetMeshInstance(m->tetMesh, translation, rotation, scale, density, material, phase);

    // update initial render state
    UpdateDeformableMesh(m);

    // create render materials
    for (size_t a=0; a < m->hostMesh->m_materialAssignments.size(); ++a)
    {
        MaterialAssignment assign = m->hostMesh->m_materialAssignments[a];

        RenderMaterial renderMaterial;
        renderMaterial.frontColor = m->hostMesh->m_materials[assign.material].Kd;
        renderMaterial.backColor = m->hostMesh->m_materials[assign.material].Kd;
        renderMaterial.specular = m->hostMesh->m_materials[assign.material].Ks.x;
        renderMaterial.roughness = SpecularExponentToRoughness(m->hostMesh->m_materials[assign.material].Ns);

        // load texture
        char texturePath[2048];
        MakeRelativePath(visualFile, m->hostMesh->m_materials[assign.material].mapKd.c_str(), texturePath);

        renderMaterial.colorTex = CreateRenderTexture(texturePath);

        // construct render batch for this mesh/material combination
        DeformableRenderBatch batch;
        batch.material = renderMaterial;
        batch.startTri = m->hostMesh->m_materialAssignments[a].startTri;
        batch.endTri = m->hostMesh->m_materialAssignments[a].endTri;

        m->renderBatches.push_back(batch);
    }

	return m;
}


void DrawDeformableMesh(DeformableMesh* d)
{
	for (int i=0; i < d->renderBatches.size(); ++i)
	{
		DrawRenderMesh(d->renderMesh, Matrix44::kIdentity, d->renderBatches[i].material, d->renderBatches[i].startTri, d->renderBatches[i].endTri);
	}
}