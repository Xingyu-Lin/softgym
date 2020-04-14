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
// Copyright (c) 20132017 NVIDIA Corporation. All rights reserved.

#include <unordered_set>
#include <set>
#include <vector>

#include "../include/NvFlexExt.h"

#include "../core/core.h"
#include "../core/maths.h"
#include "../core/voxelize.h"

#ifdef _WIN32

#include <windows.h>
#include <commdlg.h>
#include <mmsystem.h>

double GetSeconds()
{
	static LARGE_INTEGER lastTime;
	static LARGE_INTEGER freq;
	static bool first = true;
	
	if (first)
	{	
		QueryPerformanceCounter(&lastTime);
		QueryPerformanceFrequency(&freq);

		first = false;
	}
	
	static double time = 0.0;
	
	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);
	
	__int64 delta = t.QuadPart-lastTime.QuadPart;
	double deltaSeconds = double(delta) / double(freq.QuadPart);
	
	time += deltaSeconds;

	lastTime = t;

	return time;

}

#include <vector>
#include <algorithm>

using namespace std;

// FEM support functions

namespace
{

CUDA_CALLABLE inline int LongestAxis(const Vec3& v)
{
	if (v.x > v.y && v.x > v.z) 
		return 0;
	if (v.y > v.z)
		return 1;
	else
		return 2;
}

struct PartitionPredicateMedian
{
	PartitionPredicateMedian(const Bounds* bounds, int a) : bounds(bounds), axis(a) {}

	bool operator()(int a, int b) const
	{
		return bounds[a].GetCenter()[axis] < bounds[b].GetCenter()[axis];
	}

	const Bounds* bounds;
	int axis;
};


int PartitionObjectsMedian(const Bounds* bounds, int* indices, int start, int end, Bounds rangeBounds)
{
	assert(end-start >= 2);

	Vec3 edges = rangeBounds.GetEdges();

	int longestAxis = LongestAxis(edges);

	const int k = (start+end)/2;

	std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(&bounds[0], longestAxis));

	return k;
}	
	
struct PartitionPredictateMidPoint
{
	PartitionPredictateMidPoint(const Bounds* bounds, int a, float m) : bounds(bounds), axis(a), mid(m) {}

	bool operator()(int index) const 
	{
		return bounds[index].GetCenter()[axis] <= mid;
	}

	const Bounds* bounds;
	int axis;
	float mid;
};


int PartitionObjectsMidPoint(const Bounds* bounds, int* indices, int start, int end, Bounds rangeBounds)
{
	assert(end-start >= 2);

	Vec3 edges = rangeBounds.GetEdges();
	Vec3 center = rangeBounds.GetCenter();

	int longestAxis = LongestAxis(edges);

	float mid = center[longestAxis];


	int* upper = std::partition(indices+start, indices+end, PartitionPredictateMidPoint(&bounds[0], longestAxis, mid));

	int k = int(upper-indices);

	// if we failed to split items then just split in the middle
	if (k == start || k == end)
		k = (start+end)/2;


	return k;
}


	template <typename T>
	struct DynamicBVH
	{
		struct Node
		{
			Bounds bounds;

			int leftChild;
			int rightChild;

			inline bool IsLeaf() const { return rightChild == -1; }
		};

		DynamicBVH()
		{
			mRoot = -1;
			mItemCount = 0;
		}

		int Insert(const Bounds& b)
		{
			// insert into items list
			if (mRoot == -1)
			{
				mRoot = AddNode(mItemCount, b.lower, b.upper);
			}
			else
			{
				InsertRecursive(mRoot, mItemCount, b.lower, b.upper);
			}

			mItemCount++;
			return mItemCount-1;
		}

		int Insert(const Bounds* items, int n)
		{
			std::vector<int> indices(n);
			for (int i=0; i < n; ++i)
				indices[i] = i;

			mRoot = BuildRecursive(items, &indices[0], 0, n);

			int startIndex = mItemCount;
			mItemCount += n;
			return startIndex;
		}

		void Remove(T value, Vec3 lower, Vec3 upper)
		{
		}
		
		void QueryBox(Vec3 lower, Vec3 upper)
		{
		}

		void QueryPoint(const Vec3& point, std::vector<T>& overlaps)
		{
			const int maxDepth = 64;

			int stack[64];
			int count = 1;
			
			stack[0] = mRoot;

			while (count)
			{
				const int i = stack[--count];

				//assert(i != -1);

				const Node& node = mNodes[i];

				if (node.bounds.Overlaps(point))
				{
					if (node.IsLeaf())
					{
						overlaps.push_back(node.leftChild);
					}
					else
					{
						stack[count++] = node.leftChild;
						stack[count++] = node.rightChild;
					}
				}
			}
		}

		int mItemCount;

	private:

		int BuildRecursive(const Bounds* items, int* indices, int start, int end)
		{
			assert(start < end);

			const int n = end-start;

			// calculate bounds of the range
			Bounds b;
			for (int i=start; i < end; ++i)
				b = Union(b, items[indices[i]]);
		
			const int kMaxItemsPerLeaf = 1;

			if (n <= kMaxItemsPerLeaf)
			{
				Node node;
				node.leftChild = indices[start];
				node.rightChild = -1;
				node.bounds = b;

				mNodes.push_back(node);
				return int(mNodes.size())-1;
			}
			else
			{
				//int split = PartitionObjectsMidPoint(bounds, bvh.mIndices, start, end, bvh.mNodeBounds[nodeIndex]);
				int split = PartitionObjectsMedian(items, indices, start, end, b);
		
				int leftChild = BuildRecursive(items, indices, start, split);
				int rightChild = BuildRecursive(items, indices, split, end);
			
				Node node;
				node.leftChild = leftChild;
				node.rightChild = rightChild;
				node.bounds = b;

				mNodes.push_back(node);
				return int(mNodes.size())-1;
			}
		}

		int AddNode(int itemIndex, const Vec3& lower, const Vec3& upper)
		{
			Node n;
			n.leftChild = itemIndex;
			n.rightChild = -1;
			n.bounds = Bounds(lower, upper);

			const int nodeIndex = int(mNodes.size());
			mNodes.push_back(n);

			return nodeIndex;
		}

		int CopyNode(int index)
		{
			const int nodeIndex = int(mNodes.size());
			mNodes.push_back(mNodes[index]);

			return nodeIndex;
		}

		void InsertRecursive(int nodeIndex, int itemIndex, const Vec3& lower, const Vec3& upper)
		{
			// if node is a leaf then split and insert a new internal node
			if (mNodes[nodeIndex].IsLeaf())
			{
				Node internalNode;

				internalNode.leftChild = CopyNode(nodeIndex);
				internalNode.rightChild = AddNode(itemIndex, lower, upper);
				internalNode.bounds = Union(mNodes[nodeIndex].bounds, Bounds(lower, upper));

				mNodes[nodeIndex] = internalNode;

				return;
			}
			else
			{
				// expand node
				mNodes[nodeIndex].bounds = Union(mNodes[nodeIndex].bounds, Bounds(lower, upper));

				// insert on the subtree which minimizes the surface area increase
				Bounds leftBounds = Union(mNodes[mNodes[nodeIndex].leftChild].bounds, Bounds(lower, upper));
				Bounds rightBounds = Union(mNodes[mNodes[nodeIndex].rightChild].bounds, Bounds(lower, upper));

				if (SurfaceArea(leftBounds) < SurfaceArea(rightBounds))
				{
					InsertRecursive(mNodes[nodeIndex].leftChild, itemIndex, lower, upper);
					return;
				}
				else
				{
					InsertRecursive(mNodes[nodeIndex].rightChild, itemIndex, lower, upper);
					return;
				}										
			}
		}
		
		int mRoot;

		std::vector<Node> mNodes;
	};

}


void DynamicBVHTest()
{
	RandInit();

	std::vector<Bounds> items;

	DynamicBVH<int> bvh;
	
	const int numItems = 50000;

	const float extents = 10.0f;

	for (int i=0; i < numItems; ++i)
	{
		Bounds b;
		
		// generate a random bound
		b.lower = RandVec3()*extents;
		b.upper = b.lower + Vec3(0.5f);// Vec3(Randf(), Randf(), Randf())*0.5f;

		items.push_back(b);

		bvh.Insert(b);
	}

	//bvh.Insert(&items[0], int(items.size()));

	const int numTests = 1000;

	std::vector<int> overlaps;
	std::vector<int> overlapsRef;

	for (int i=0; i < numTests; ++i)
	{
		overlaps.resize(0);
		overlapsRef.resize(0);

		// generate a random point
		const Vec3 p = RandVec3()*extents;

		double refBegin = GetSeconds();

		// generate reference overlaps O(N)
		for (int i=0; i < items.size(); ++i)
		{
			if (items[i].Overlaps(p))
			{
				overlapsRef.push_back(i);
			}
		}

		double refEnd = GetSeconds();

		// generate bvh overlaps
		double bvhBegin = GetSeconds();

		bvh.QueryPoint(p, overlaps);

		double bvhEnd = GetSeconds();

		printf("bvh: %d ref: %d bvh time: %f ref time: %f\n", int(overlaps.size()), int(overlapsRef.size()), (bvhEnd-bvhBegin)*1000.0f, (refEnd-refBegin)*1000.0f);
		fflush(stdout);

		// ensure correct counts
		assert(overlaps.size() == overlapsRef.size());

		// ensure correct results
		std::sort(overlaps.begin(), overlaps.end());
		std::sort(overlapsRef.begin(), overlapsRef.end());

		for (int i=0; i < overlaps.size(); ++i)
			assert(overlaps[i] == overlapsRef[i]);


	}
}



extern "C"
{

// fwd robust predicates defined in predicates.c, stub for now
	void exactinit() {}
float insphere(float*, float*, float*, float*, float*) { return 0.0f; }
float insphereexact(float*, float*, float*, float*, float*) { return 0.0f; }
float orient3d(float*, float*, float*, float*) {return 0.0f; }

} // anonymous namespace



float TetVolume(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d)
{
	Vec3 p = b-a;
	Vec3 q = c-a;
	Vec3 r = d-a;

	//float v = Dot(p, Cross(q, r));

	float v = orient3d(Vec3(a), Vec3(b), Vec3(c), Vec3(d));

	assert(v >= 0.0f);

	return v/6.0f;


}

Vec3 TetCentroid(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d)
{
	return 0.25f*(a+b+c+d);
}

Vec3 TetCircumcenter(const Vec3& p, const Vec3& q, const Vec3& r, const Vec3& s)
{
	Vec3 p0 = p;
	Vec3 b = q-p;
	Vec3 c = r-p;
	Vec3 d = s-p;

	float det = b.x * (c.y * d.z - c.z * d.y) - b.y * (c.x * d.z - c.z * d.x) + b.z * (c.x * d.y - c.y * d.x);

	if (det <= 0.0f)
	{
		return TetCentroid(p, q, r, s);
	}
	else
	{
		det *= 2.0f;
		Vec3 v = Cross(c, d)*Dot(b,b) + Cross(d, b)*Dot(c,c) + Cross(b, c)*Dot(d,d);
		v /= det;

		return p0 + v;
	}



	/*
	Vec3 p = b-a;
	Vec3 q = c-a;
	Vec3 r = d-a;

	float v = TetVolume(a, b, c, d);

	if (v < 1.e-8f)
		return TetCentroid(a, b, c, d);
	else
		return a + (Dot(p,p)*Cross(q,r) + Dot(q,q)*Cross(r,p) + Dot(r,r)*Cross(p, q))/(12.0f*v);		
		*/
}

float TetInSphere(Vec3& p, Vec3& q, Vec3& r, Vec3& s, Vec3& x)
{
	return insphere(p, q, r, s, x);
		
}


struct DelaunayTetrahedralization
{
	struct Element
	{
		int vertices[4];

		inline bool operator == (const Element& e) const
		{
			return e.vertices[0] == vertices[0] && e.vertices[1] == vertices[1] && e.vertices[2] == vertices[2] && e.vertices[3] == vertices[3];	
		}
	};

	struct Face
	{
		Face(int i, int j, int k)
		{
			vertices[0] = i;
			vertices[1] = j;
			vertices[2] = k;

			std::sort(vertices, vertices+3);
		}

		int vertices[3];

		inline bool operator == (const Face& f) const 
		{
			return (f.vertices[0] == vertices[0] && f.vertices[1] == vertices[1] && f.vertices[2] == vertices[2]);
		}
		
	};

	struct ElementHasher
	{
		inline size_t operator()(const Element& e) const { return e.vertices[0] ^ e.vertices[1] ^ e.vertices[2] ^ e.vertices[3]; };
	};



	DelaunayTetrahedralization(const Vec3* points, int n)
	{
	//	exactinit();

		// construct an initial bounding tetrahedron
		Vec3 lower(FLT_MAX);
		Vec3 upper(-FLT_MAX);

		for (int i=0; i < n; ++i)
		{
			lower = Min(lower, points[i]);
			upper = Max(upper, points[i]);
		}

		const float expand = 1000.0f;
		Vec3 margin = (upper-lower)*expand;

		lower -= Vec3(10.0f);
		upper += Vec3(10.0f);
		
		Vec3 extents = upper-lower;

		
		mVertices.push_back(lower);
		mVertices.push_back(lower + Vec3(extents.x*2.0f, 0.0f, 0.0f));
		mVertices.push_back(lower + Vec3(0.0f, extents.y*2.0f, 0.0f));
		mVertices.push_back(lower + Vec3(0.0f, 0.0f, extents.z*2.0f));
		
		
		Element e = { 0, 1, 2, 3 };
		AddElement(e);

		for (int i=0; i < n; ++i)
		{
			Insert(points[i]);
		}

#if 1

		// remove any tets connected to the initial boundary
		for (auto iter=mElements.begin(); iter != mElements.end(); )
		{
			const Element& e = *iter;

			if (e.vertices[0] < 4 ||
				e.vertices[1] < 4 ||
				e.vertices[2] < 4 ||
				e.vertices[3] < 4)
			{
				iter = mElements.erase(iter);
			}
			else
			{
				++iter;
			}
		}
#endif 

	}

	void Insert(const Vec3& p)
	{
		for (int i=0; i < mVertices.size(); ++i)
		{
			// remove co-incicent vertices
			if (LengthSq(p-mVertices[i]) == 0.0f)
			{
				printf("removed vert %f %f %f\n", p.x, p.y, p.z);
				return;
			}
		}

		const int index = int(mVertices.size());
		mVertices.push_back(p);

		// remove all elements whose circumcircle this point lies inside and retriangulate, i.e.: Bowyer-Waton
		std::vector<Element> overlaps;
		GetOverlappingElements(p, overlaps);
		
		assert(overlaps.size() > 0);

		// find open faces
		std::vector<Face> openFaces;

		for (int i=0; i < int(overlaps.size()); ++i)
		{
			Element e = overlaps[i];

			for (int f=0; f < 4; ++f)
			{
				Face face(
					e.vertices[(f+0)%4],
					e.vertices[(f+1)%4],
					e.vertices[(f+2)%4]);
				
				const vector<Face>::iterator it = std::find(openFaces.begin(), openFaces.end(), face);
				if (it == openFaces.end())
				{
					openFaces.push_back(face);
				}
				else
				{
					openFaces.erase(it);
				}
			}
		}

		for (int i=0; i < int(overlaps.size()); ++i)
			RemoveElement(overlaps[i]);

		// retriangulate open faces
		for (int i=0; i < int(openFaces.size()); ++i)
		{
			const Face& face = openFaces[i];

			Element e;
			e.vertices[0] = face.vertices[0];
			e.vertices[1] = face.vertices[1];
			e.vertices[2] = face.vertices[2];
			e.vertices[3] = index;

			AddElement(e);
		}

		//assert(Valid());

	}

	void AddElement(Element e)
	{
		// check determinant of element, if negative then flip
		Vec3 a = mVertices[e.vertices[0]];
		Vec3 b = mVertices[e.vertices[1]];
		Vec3 c = mVertices[e.vertices[2]];
		Vec3 d = mVertices[e.vertices[3]];

		float det = orient3d(a, b, c, d);

		if (det < 0.0f)
		{
			swap(e.vertices[1], e.vertices[2]);
			det *= -1.0f;
		}

		{
			Vec3 a = mVertices[e.vertices[0]];
			Vec3 b = mVertices[e.vertices[1]];
			Vec3 c = mVertices[e.vertices[2]];
			Vec3 d = mVertices[e.vertices[3]];

			if (orient3d(a, b, c, d) <= 0.0f)
				printf("Go\n");
		}
		

		mElements.insert(e);
	}

	void RemoveElement(Element e)
	{
		mElements.erase(e);
	}
	
	bool Valid()
	{
		for (int i=0; i < mVertices.size(); ++i)
		{
			for (auto it=mElements.begin(); it != mElements.end(); ++it)
			{
				Element e = *it;

				if (e.vertices[0] == i ||
					e.vertices[1] == i ||
					e.vertices[2] == i ||
					e.vertices[3] == i)
					continue;


				Vec3 a = mVertices[e.vertices[0]];
				Vec3 b = mVertices[e.vertices[1]];
				Vec3 c = mVertices[e.vertices[2]];
				Vec3 d = mVertices[e.vertices[3]];

				if (orient3d(a, b, c, d) < 0.0f)
					return false;

				float det = insphere(a, b, c, d, mVertices[i]);
				
				if (det > 0.0f)
					return false;
			}
		}

		return true;
	}

	void GetOverlappingElements(Vec3 p, std::vector<Element>& elements)
	{
		for (auto it=mElements.begin(); it != mElements.end(); ++it)
		{	
			Element e = *it;

			Vec3 a = mVertices[e.vertices[0]];
			Vec3 b = mVertices[e.vertices[1]];
			Vec3 c = mVertices[e.vertices[2]];
			Vec3 d = mVertices[e.vertices[3]];

			if (TetInSphere(a, b, c, d, p) > 0.0f)
			{
				assert(orient3d(a, b, c, d) > 0.0f);

				elements.push_back(e);
			}
		}
	}

	std::unordered_set<Element, ElementHasher> mElements;
	std::vector<Vec3> mVertices;
};


//void TriangulateVariational(const Vec2* points, uint32_t numPoints, const Vec2* bpoints, uint32_t numBPoints, uint32_t iterations, std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris);

#include "../core/aabbtree.h"

bool TestInside(const AABBTree& bvh, Vec3 point)
{
	int samples = 1000;
	int hits = 0;
	for (int i=0; i < samples; ++i)
	{
		float t, u, v, w, sign;
		uint32_t face;
		if (bvh.TraceRay(point, UniformSampleSphere(), t, u, v, w, sign, face))
		{
			if (sign < 0.0f)
				hits++;
		}
	}

	float r = float(hits)/samples;
	bool inside = r > 0.9f;

	return inside;
}

bool FindVertex(const Vec4* samples, int n, float tol, Vec3 p)
{
	for (int i=0; i < n; ++i)
		if (LengthSq(p-Vec3(samples[i])) < tol)
			return true;

	return false;
}

void SampleMeshBoundary(const Vec3* points, int numPoints, const int* tris, int numTris, int numSamples, float eqtol, std::vector<Vec4>& samples)
{
	const float kSourceWeight = 1.0f;
	const float kInternalWeight = 1.0f;
	
	for (int i=0; i < numPoints; ++i)
	{
		//if (!samples.empty() && FindVertex(&samples[0], samples.size(), eqtol, points[i]) == false)
			samples.push_back(Vec4(points[i], kSourceWeight));
	}

	// sample boundary
	for (int i=0; i < numSamples; ++i)
	{
		// generate a random sample on the boundary (not uniformly distributed, assumes each triangle area is equal)
		const int tri = Rand()%numTris;

		float u, v;
		UniformSampleTriangle(u, v);

		Vec3 p = points[tris[tri*3+0]];
		Vec3 q = points[tris[tri*3+1]];
		Vec3 r = points[tris[tri*3+2]];

		Vec3 s = u*p + v*q + (1.0f-u-v)*r;

		samples.push_back(Vec4(s, kInternalWeight));
	}

}

void SampleMeshInterior(const Vec3* points, int numPoints, const int* tris, int numTris, const AABBTree& bvh, float resolution, std::vector<Vec3>& samples)
{
	Vec3 lower(FLT_MAX);
	Vec3 upper(-FLT_MAX);

	// calculate mesh bounds
	for (int i=0; i < numPoints; ++i)
	{
		lower = Min(lower, ((const Vec3*)(points))[i]);
		upper = Max(upper, ((const Vec3*)(points))[i]);
	}

	// sample interior at regularly spaced intervals (stratified)
	Vec3 edges = upper-lower;

	int dimx = int(edges.x / resolution + 0.5f);
	int dimy = int(edges.y / resolution + 0.5f);
	int dimz = int(edges.z / resolution + 0.5f);

	for (int x=0; x <= dimx; ++x)
	{
		for (int y=0; y <= dimy; ++y)
		{
			for (int z=0; z <= dimz; ++z)
			{
				Vec3 p = lower + Vec3(x*resolution, y*resolution, z*resolution);

				if (TestInside(bvh, p))
				{
					samples.push_back(p);
				}
			}
		}
	}
}


NV_FLEX_API NvFlexExtTetrahedralization* NvFlexExtCreateTetrahedralization(const float* points, int numPoints, const int* tris, int numTris)
{

	RandInit();
	exactinit();


	AABBTree bvh((const Vec3*)points, numPoints, (const uint32_t*)tris, numTris);

	const int kNumBoundarySamples = 16*1024;
	const float kResolution = 0.075f;
	const float kEqualTolerance = kResolution*0.1f;

	std::vector<Vec4> boundarySamples;
	std::vector<Vec3> interiorSamples;

	printf("Sampling mesh boundary\n");
	SampleMeshBoundary((const Vec3*)points, numPoints, tris, numTris, kNumBoundarySamples, kEqualTolerance, boundarySamples);

	printf("Sampling mesh interior\n");
	SampleMeshInterior((const Vec3*)points, numPoints, tris, numTris, bvh, kResolution, interiorSamples);

	//interiorSamples.insert(interiorSamples.end(), (const Vec3*)points, ((const Vec3*)points) + numPoints);

	std::random_shuffle(interiorSamples.begin(), interiorSamples.end());

	const int kNumRelaxations = 6;

	for (int k=0; k < kNumRelaxations; ++k)
	{
		printf("iter %d\n", k);

		DelaunayTetrahedralization tetra(&interiorSamples[0], int(interiorSamples.size()));
		
		interiorSamples.assign(tetra.mVertices.begin() + 4, tetra.mVertices.end());

		const int kInner = 4;
		for (int l=0; l < kInner; ++l)
		{
			std::vector<Vec4> relaxed(int(interiorSamples.size()));

			// optimize sample positions based on boundary
			for (int i=0; i < int(boundarySamples.size()); ++i)
			{
				float closestDistSq = FLT_MAX;
				int closestIndex;

				const Vec3 b = boundarySamples[i];

				// find closed vertex to each boundary sample
				for (int j=0; j < int(interiorSamples.size()); ++j)
				{
					float distSq = LengthSq(interiorSamples[j]-b);

					if (distSq < closestDistSq)
					{
						closestIndex = j;
						closestDistSq = distSq;
					}
				}
			
				relaxed[closestIndex] += Vec4(b*boundarySamples[i].w, -boundarySamples[i].w);
			}
			/*
			// apply boundary constraints to samples
			for (int i=0; i < int(relaxed.size()); ++i)
			{
				if (relaxed[i].w < 0.0f)
				{
					interiorSamples[i] = Vec3(relaxed[i])/fabsf(relaxed[i].w);
				}
			}
			*/

			// optimize sample positions 
			for (auto iter=tetra.mElements.begin(); iter != tetra.mElements.end(); ++iter)
			{
				const DelaunayTetrahedralization::Element& e = *iter;
		
				// ignore elements connected to exterior boundary
				if (e.vertices[0] < 4 || e.vertices[1] < 4 || e.vertices[2] < 4 || e.vertices[3] < 4)
					continue;

				// move each vertex to the average of its 1-ring
				const int a = e.vertices[0]-4;
				const int b = e.vertices[1]-4;
				const int c = e.vertices[2]-4;
				const int d = e.vertices[3]-4;

				Vec3 circumcenter = TetCircumcenter(
									  interiorSamples[a],
									  interiorSamples[b],
									  interiorSamples[c],
									  interiorSamples[d]);

				const float volume = 1.0f;/*fabsf(TetVolume(
									interiorSamples[a],
						   			interiorSamples[b],
									interiorSamples[c],
									interiorSamples[d]));*/

				for (int v=0; v < 4; ++v)
				{
					if (relaxed[e.vertices[v]-4].w >= 0.0f)
						relaxed[e.vertices[v]-4] += Vec4(circumcenter*volume, volume);
				}
			}

		

			// apply boundary constraints to samples
			for (int i=0; i < int(relaxed.size()); ++i)
			{
				//if (relaxed[i].w >  0.0f)
				{
					interiorSamples[i] = Vec3(relaxed[i])/fabsf(relaxed[i].w);
				}
			}
		}
	}


	// final tetrahedralization
	DelaunayTetrahedralization tetra(&interiorSamples[0], int(interiorSamples.size()));
	
	if (!tetra.Valid())
		printf("Not Delaunay\n");

	const int vertOffset = 4;

	Vec3* vertices = new Vec3[tetra.mVertices.size()];
	int* tetrahedra = new int[tetra.mElements.size()*4];

	memcpy(vertices, &tetra.mVertices[vertOffset], (tetra.mVertices.size()-vertOffset)*sizeof(Vec3));

	int index = 0;

	float minVolume = FLT_MAX;

	printf("Removing exterior tetrahedra\n");

	for (auto iter=tetra.mElements.begin(); iter != tetra.mElements.end(); ++iter)
	{
		const DelaunayTetrahedralization::Element& e = *iter;
		
		
		Vec3 a = vertices[e.vertices[0]-vertOffset];
		Vec3 b = vertices[e.vertices[1]-vertOffset];
		Vec3 c = vertices[e.vertices[2]-vertOffset];
		Vec3 d = vertices[e.vertices[3]-vertOffset];

		Vec3 centroid = TetCentroid(a, b, c, d);
		
		if (TestInside(bvh, centroid))
		{
			float det = orient3d(a, b, c, d);
			if (det < minVolume)
				minVolume = det;

			tetrahedra[index*4+0] = e.vertices[0]-vertOffset;
			tetrahedra[index*4+1] = e.vertices[1]-vertOffset;
			tetrahedra[index*4+2] = e.vertices[2]-vertOffset;
			tetrahedra[index*4+3] = e.vertices[3]-vertOffset;

			index++;
		}
	}

	printf("minvol: %f\n", minVolume);

	NvFlexExtTetrahedralization* t = new NvFlexExtTetrahedralization();
	t->vertices = (float*)vertices;
	t->numVertices = int(tetra.mVertices.size()-vertOffset);
	t->tetrahedra = tetrahedra;
	t->numTetrahedra = index;

	return t;
	
}



void NvFlexExtDestroyTetrahedralization(NvFlexExtTetrahedralization* t)
{
	delete[] t->vertices;
	delete[] t->tetrahedra;

	delete t;

}

#else

//Linux stubs

NV_FLEX_API NvFlexExtTetrahedralization* NvFlexExtCreateTetrahedralization(const float*, int , const int* , int )
{
	return NULL;
}

void NvFlexExtDestroyTetrahedralization(NvFlexExtTetrahedralization*)
{

}


#endif // Linux


NV_FLEX_API NvFlexExtMeshAdjacency* NvFlexExtCreateMeshAdjacency(const float* points, int numPoints, int stride, const int* indices, int numTris, bool onesided)
{
	std::vector<Vec3> vertices;
	vertices.reserve(numPoints);

	// feature assignment to triangles 
	std::vector<int> triFeatures;

	// vertex adjacency information (0-n vertices opposite each vertex, the 1-star)
	std::vector<int> vertexAdj;
	std::vector<int> vertexAdjOffset;
	std::vector<int> vertexAdjCount;

	// edge adjacency information (0-1 vertices opposite edge, 3*numTris in length)
	std::vector<int> edgeAdj;

	// de-stride input vertices for convenience
	const char* inputPoints = (const char*)(points);

	for (int i=0; i < numPoints; ++i)
	{
		vertices.push_back(*((const Vec3*)inputPoints));
		inputPoints += stride;
	}

	// build unique feature lists
	triFeatures.resize(numTris, eNvFlexTriFeatureFace);

	std::vector<int> vertexParents(vertices.size(), -1);
	std::vector<int> vertexValence(vertices.size(), 0);		

	// assign vertices to triangles
	for (int i=0; i < numTris; ++i)
	{
		for (int v=0; v < 3; ++v)
		{
			const int index = indices[i*3+v];

			if (vertexParents[index] == -1)
			{
				// take ownership of this vertex
				triFeatures[i] |= eNvFlexTriFeatureVertex0 << v;

				vertexParents[index] = i;
			}
		}
	}

	struct Edge
	{
		int vertices[2];
		mutable int faces[2];
		mutable int features[2];	
		mutable int opposites[2];

		Edge(int a, int b)
		{
			vertices[0] = Min(a, b);
			vertices[1] = Max(a, b);
			faces[0] = -1;
			faces[1] = -1;
			features[0] = -1;
			features[1] = -1;
			opposites[0] = -1;
			opposites[1] = -1;
		}

		bool operator<(const Edge& rhs) const
		{
			return vertices[0] < rhs.vertices[0] || (!(rhs.vertices[0] < vertices[0]) && vertices[1] < rhs.vertices[1]);
		}
	};

	std::set<Edge> edges;
	std::vector<Vec3> normals(numTris);

	// assign edges to triangles
	for (int face=0; face < numTris; ++face)
	{
		Vec3 a = vertices[indices[face*3+0]];
		Vec3 b = vertices[indices[face*3+1]];
		Vec3 c = vertices[indices[face*3+2]];

		normals[face] = SafeNormalize(Cross(b-a, c-a));

		for (int e=0; e < 3; ++e)
		{
			int vertex0 = indices[face*3+e];
			int vertex1 = indices[face*3+(e+1)%3];

			Edge edge(vertex0, vertex1);

			auto iter = edges.find(edge);
			if (iter == edges.end())
			{
				edge.faces[0] = face;
				edge.features[0] = e;
				edge.opposites[1] = indices[face*3+(e+2)%3];	// update edge opposite (only valid if face index is also valid)
					
				edges.insert(edge);

				vertexValence[vertex0]++;
				vertexValence[vertex1]++;
			}
			else
			{
				// update second face
				if (iter->faces[1] != -1)
					printf("Non manifold triangle mesh\n");

				iter->faces[1] = face;
				iter->features[1] = e;
				iter->opposites[0] = indices[face*3+(e+2)%3];
			}
		}
	}

	// remove non-convex edges (if one-sided), and assign to the first face
	for (auto iter=edges.begin(); iter != edges.end(); ++iter)
	{
		const int f0 = iter->faces[0];
		const int f1 = iter->faces[1];

		if (onesided && iter->faces[0] != -1 && iter->faces[1] != -1)
		{
			Vec3 a0 = vertices[indices[f0*3+0]];
			Vec3 b0 = vertices[indices[f0*3+1]];
			Vec3 c0 = vertices[indices[f0*3+2]];

			// test centroid of the opposing face against this one
			Vec3 a1 = vertices[indices[f1*3+0]];
			Vec3 b1 = vertices[indices[f1*3+1]];
			Vec3 c1 = vertices[indices[f1*3+2]];

			Vec3 delta = SafeNormalize(((a1 + b1 + c1) - (a0 + b0 + c0)));

			Vec3 n0 = normals[f0];
			
			const float kConvexityThreshold = 1.e-3f;
			
			// if non-convex then skip
			if (Dot(n0, delta) >= -kConvexityThreshold)
				continue;

		}

		// first face takes ownership of the edge
		triFeatures[iter->faces[0]] |= eNvFlexTriFeatureEdge0 << iter->features[0];
	}

	// build vertex adjacency (exclusive prefix sum to find start offset in the adjacency array)
	vertexAdjOffset.resize(vertices.size());

	int offset = 0;
	for (int i=0; i < int(vertexValence.size()); ++i)
	{
		vertexAdjOffset[i] = offset;
		offset += vertexValence[i];
	}

	vertexAdj.resize(offset);
	vertexAdjCount.resize(vertices.size());	

	edgeAdj.resize(numTris*3);

	for (auto iter=edges.begin(); iter != edges.end(); ++iter)
	{
		// populate adjacency array with vertex 1-star by iterating over edges
		int vertex0 = iter->vertices[0];
		int vertex1 = iter->vertices[1];
			
		vertexAdj[vertexAdjOffset[vertex0] + vertexAdjCount[vertex0]] = vertex1;	// vertex0 adjacent to vertex1 and vice-versa
		vertexAdj[vertexAdjOffset[vertex1] + vertexAdjCount[vertex1]] = vertex0;

		vertexAdjCount[vertex0]++;
		vertexAdjCount[vertex1]++;

		// build edge opposite vertex adjacency information by assigning opposite vertices to faces
		int face0 = iter->faces[0];
		int face1 = iter->faces[1];

		if (face0 != -1)
			edgeAdj[face0*3+iter->features[0]] = iter->opposites[0];

		if (face1 != -1)
			edgeAdj[face1*3+iter->features[1]] = iter->opposites[1];
	}

	// copy to the output struct
	NvFlexExtMeshAdjacency* adj = new NvFlexExtMeshAdjacency();

	adj->numTris = numTris;
	adj->numVertexAdjs = int(vertexAdj.size());
	adj->numVertices = int(vertices.size());

	adj->triFeatures = new int[triFeatures.size()];
	memcpy(adj->triFeatures, &triFeatures[0], sizeof(int)*triFeatures.size());

	adj->vertexAdj = new int[vertexAdj.size()];
	memcpy(adj->vertexAdj, &vertexAdj[0], sizeof(int)*vertexAdj.size());

	adj->vertexAdjCount = new int[vertexAdjCount.size()];
	memcpy(adj->vertexAdjCount, &vertexAdjCount[0], sizeof(int)*vertexAdjCount.size());

	adj->vertexAdjOffset = new int[vertexAdjOffset.size()];
	memcpy(adj->vertexAdjOffset, &vertexAdjOffset[0], sizeof(int)*vertexAdjOffset.size());

	adj->edgeAdj = new int[edgeAdj.size()];
	memcpy(adj->edgeAdj, &edgeAdj[0], sizeof(int)*edgeAdj.size());

	return adj;
			
}

NV_FLEX_API void NvFlexExtDestroyMeshAdjacency(NvFlexExtMeshAdjacency* adj)
{
	delete[] adj->triFeatures;
	delete[] adj->vertexAdj;
	delete[] adj->vertexAdjCount;
	delete[] adj->vertexAdjOffset;
	delete[] adj->edgeAdj;
}

NV_FLEX_API void NvFlexExtDilateMesh(const float* points, int numPoints, const int* indices, int numTris, float dilation, float* output)
{
	// perform dilation
	const int kIterations = 50;

	const Vec3* verticesIn = (const Vec3*)points;
	Vec3* verticesOut = (Vec3*)output;

	// initialize output
	for (int i=0; i < numPoints; ++i)
		verticesOut[i] = verticesIn[i];

	const float kRelaxation = 1.0f;

	for (int k=0; k < kIterations; ++k)
	{
		for (int face=0; face < numTris; ++face)
		{
			Vec3 a = verticesIn[indices[face*3+0]];
			Vec3 b = verticesIn[indices[face*3+1]];
			Vec3 c = verticesIn[indices[face*3+2]];

			Vec3 n = SafeNormalize(Cross(b-a, c-a));

			for (int v=0; v < 3; ++v)
			{
				Vec3 p = verticesOut[indices[face*3+v]];

				// project each vertex onto the offset normal
				float dist = Dot(a-p, n);
				if (dist < dilation)
				{
					float err = dist - dilation;
					verticesOut[indices[face*3+v]] += n*err*kRelaxation;
				}
			}
		}
	}
}