#pragma once

#include "rlbase.h"
#include "rllocomotion.h"

using namespace std;
using namespace tinyxml2;


int createTerrain(float sizex, float sizez, int gridx, int gridz, 
				const Vec3& offset, const Transform& trans, const Vec3& scale = Vec3(25.f, 1.f, 25.f), 
				int octaves = 5, float persistance = 0.4)
{
	Mesh* terrain = CreateTerrain(sizex, sizez, gridx, gridz, offset, scale, octaves, persistance);
	terrain->Transform(TransformMatrix(trans));

	NvFlexTriangleMeshId terrainId = CreateTriangleMesh(terrain);

	Vec3 terrainFrontColour = Vec3(0.17f, 0.19f, 0.27f);
	Vec3 terrainBackColour = Vec3(0.06f, 0.05f, 0.043f);

	RenderMaterial terrainMat;
	terrainMat.frontColor = terrainFrontColour;
	terrainMat.backColor = terrainBackColour;
	terrainMat.gridScale = 80.0f;

	const int renderMaterial = AddRenderMaterial(terrainMat);

	NvFlexRigidShape terrainShape;
	NvFlexMakeRigidTriangleMeshShape(&terrainShape, -1, terrainId, NvFlexMakeRigidPose(0, 0), 1.0f, 1.0f, 1.0f);
	terrainShape.filter = 0;
	terrainShape.group = -1;
	terrainShape.thickness = 0.01f;
	terrainShape.material.friction = 1.f;
	terrainShape.material.torsionFriction = 0.01f;
	terrainShape.material.rollingFriction = 0.005f;
	terrainShape.user = UnionCast<void*>(renderMaterial);

	int terrainIndex = g_buffers->rigidShapes.size();
	g_buffers->rigidShapes.push_back(terrainShape);

	return terrainIndex;
}

void createImmovableBlock(Vec3 position, float angle, float width, float length, float height, float friction = 1.f, Vec3 color = Vec3())
{
	position.y += height; // because position of box shape is in the middle of the box

	NvFlexRigidShape block;
	NvFlexMakeRigidBoxShape(&block, -1, width, height, length,
		NvFlexMakeRigidPose(position, QuatFromAxisAngle(Vec3(0.f, 1.f, 0.f), angle)));
	block.filter = 0;
	block.group = -1;
	block.thickness = 0.01f;
	block.material.friction = friction;
	block.user = UnionCast<void*>(AddRenderMaterial(color));
	g_buffers->rigidShapes.push_back(block);
}

void createStairs(Vec3 position, float angle, float width, float length, 
				float stepLength, float stepHeight, int numSteps, 
				float friction = 1.f, Vec3 color = Vec3())
{
	for (int i = 0; i < numSteps; ++i)
	{
		createImmovableBlock(position + Vec3(0.f, float(i) * stepHeight, 0.f), angle,
			width, length - float(i) * stepLength, stepHeight / 2.f, friction, color);
	}
}

void createPyramid(Vec3 position, float angle, float width, float length,
	float stepLength, float stepHeight, int numSteps,
	float friction = 1.f, Vec3 color = Vec3())
{
	for (int i = 0; i < numSteps; ++i)
	{
		createImmovableBlock(position + Vec3(0.f, float(i) * stepHeight, 0.f), angle,
			width - float(i) * stepLength, length - float(i) * stepLength, stepHeight / 2.f, friction, color);
	}
}

class SpawnableObstacle
{
public:

    SpawnableObstacle() {}

    // Adds the desired obstacle starting in the given startPos
    virtual void Spawn(Vec3 startPos) = 0;

    Vec2 UniformSampleRectangle(float startX, float endX, float startY, float endY)
    {
        float x = Randf(startX, endX);
        float y = Randf(startY, endY);

        return Vec2(x, y);
    }

    Vec2 UniformSampleGrid(float startX, int numCols, float startY, int numRows, float spacing,
        bool randomX = true, bool randomY = true)
    {
        int rowNum = int(Randf() * numCols);
        int colNum = int(Randf() * numRows);

		Vec2 loc = Grid(startX, numCols, startY, numRows, spacing, rowNum, colNum);

        if (randomX) loc.x += Randf(-1.f, 1.f) * spacing / 2.5f;
        if (randomY) loc.y += Randf(-1.f, 1.f) * spacing / 2.5f;

        return loc;
    }

	Vec2 Grid(float startX, int numCols, float startY, int numRows, float spacing, int rowNum, int colNum)
	{
        float x = startX + rowNum * spacing;
        float y = startY + colNum * spacing;

        return Vec2(x, y);
	}
};

class SpawnableTerrain : public SpawnableObstacle
{
public:

    float sizeX, sizeZ;
    int numSubdiv;
    float terrainUpOffset, terrainUpOffsetRange;
    int octavesMin, octavesMax;
    float persistance, persistanceRange;
    float xScale, zScale, yScaleMin, yScaleMax;

    float terrainSizeX, terrainSizeZ;
    vector<int> terrainShapeIds;

    SpawnableTerrain(){}
    SpawnableTerrain(float sizeX, float sizeZ, int numSubdiv, float terrainUpOffset, float terrainUpOffsetRange,
                    int octavesMin, int octavesMax, float persistance, float persistanceRange,
                    float xScale, float zScale, float yScaleMin, float yScaleMax):
                    sizeX(sizeX), sizeZ(sizeZ), numSubdiv(numSubdiv), 
                    terrainUpOffset(terrainUpOffset), terrainUpOffsetRange(terrainUpOffsetRange),
                    octavesMin(octavesMin), octavesMax(octavesMax), persistance(persistance), persistanceRange(persistanceRange),
                    xScale(xScale), zScale(zScale), yScaleMin(yScaleMin), yScaleMax(yScaleMax)
    {
        terrainShapeIds.resize(0);
        
		terrainSizeX = sizeX / float(numSubdiv);
		terrainSizeZ = sizeZ / float(numSubdiv);
    }

    void Spawn(Vec3 startPos)
    {
		for (int i = 0; i < numSubdiv; ++i)
		{
			for (int j = 0; j < numSubdiv; ++j)
			{
				Transform terrainTrans = Transform(
                    startPos + Vec3(terrainSizeX * float(i),
					                terrainUpOffset + Randf(-terrainUpOffsetRange, terrainUpOffsetRange),
					                terrainSizeZ * float(j)),
					Quat());
				
                int terrainShapeId = createTerrain(terrainSizeX, terrainSizeZ, 
                    Rand(100, 110) / numSubdiv, Rand(90, 100) / numSubdiv,
					RandVec3() * 7.5f, terrainTrans, Vec3(xScale, Randf(yScaleMin, yScaleMax), zScale),
					Rand(octavesMin, octavesMax), persistance + Randf(-persistanceRange, persistanceRange));
                
                terrainShapeIds.push_back(terrainShapeId);
			}
		}
    }
};

class SpawnableBigObstacles : public SpawnableObstacle
{
public:

    float areaWidth, areaLength, density;
    int count;

    // block sizes
    float minWidth, maxWidth, minHeight, maxHeight, minLength, maxLength;

	float minHeightOffset, maxHeightOffset;
	Vec3 obstacleColor;

    SpawnableBigObstacles(){}
    SpawnableBigObstacles(float areaWidth, float areaLength, float density, float minWidth, float maxWidth,
                        float minHeight, float maxHeight, float minLength, float maxLength,
                        float minHeightOffset, float maxHeightOffset, Vec3 obstacleColor = Vec3(0.6f, 0.6f, 0.65f)):
        areaWidth(areaWidth), areaLength(areaLength), density(density),
        minWidth(minWidth), maxWidth(maxWidth), minHeight(minHeight), maxHeight(maxHeight),
        minLength(minLength), maxLength(maxLength), minHeightOffset(minHeightOffset), maxHeightOffset(maxHeightOffset),
		obstacleColor(obstacleColor)
    {
        count = int(areaWidth * areaLength * density);
    }

    void makeImmovableBlock(Vec3 position, float angle, Vec3 color)
    {
        float width, length, height, friction;
        width = Randf(minWidth, maxWidth);
        length = Randf(minLength, maxLength);
        height = Randf(minHeight, maxHeight);
        friction = Randf(0.9f, 1.f);

		createImmovableBlock(position, angle, width, length, height, friction, color);
    }
    
    void Spawn(Vec3 startPos)
    {
        for (int i = 0; i < count; i++)
        {
            Vec2 deltaPlanePos = UniformSampleRectangle(0.f, areaWidth, 0.f, areaLength);
            makeImmovableBlock(Vec3(
                startPos.x + deltaPlanePos.x,
                startPos.y + Randf(minHeightOffset, maxHeightOffset),
                startPos.z + deltaPlanePos.y
            ), Randf() * 2 * kPi, obstacleColor);
        }
    }

    void SpawnGrid(Vec3 startPos, float offsetX, int numCols, float offsetY, int numRows, float spacing, 
                    float minAngle, float maxAngle, bool randomX = true, bool randomY = true)
    {
        float angle;
        for (int i = 0; i < count; i++)
        {
            Vec2 deltaPlanePos = UniformSampleGrid(offsetX, numCols, offsetY, numRows, spacing, randomX, randomY);
            angle = Randf(minAngle, maxAngle);
            makeImmovableBlock(Vec3(
                startPos.x + deltaPlanePos.x,
                startPos.y + Randf(minHeightOffset, maxHeightOffset),
                startPos.z + deltaPlanePos.y
            ), angle, obstacleColor);
        }
    }
};

class SpawnableStairs : public SpawnableObstacle
{
public:
	float minWidth, maxWidth, minLength, maxLength;
	float minStepLength, maxStepLength, minStepHeight, maxStepHeight;
	int minNumSteps, maxNumSteps;
	Vec3 stairsColor;

	SpawnableStairs(){}
    SpawnableStairs(float minWidth, float maxWidth, float minLength, float maxLength,
                        float minStepLength, float maxStepLength, float minStepHeight, float maxStepHeight,
						int minNumSteps, int maxNumSteps, Vec3 stairsColor=Vec3(0.2f, 0.4f, 0.95f)):
        minWidth(minWidth), maxWidth(maxWidth), minLength(minLength), maxLength(maxLength),
        minStepLength(minStepLength), maxStepLength(maxStepLength), minStepHeight(minStepHeight), maxStepHeight(maxStepHeight),
        minNumSteps(minNumSteps), maxNumSteps(maxNumSteps), stairsColor(stairsColor){}

	
	void makeStairs(Vec3 position, float angle)
	{
		float width, length, stepLength, stepHeight, friction;
		int numSteps;

        width = Randf(minWidth, maxWidth);
		length = Randf(minLength, maxLength);
        stepLength = Randf(minStepLength, maxStepLength);
		stepHeight = Randf(minStepHeight, maxStepHeight);
        friction = Randf(0.9f, 1.f);

		numSteps = Rand(minNumSteps, maxNumSteps + 1);
		createStairs(position, angle, width, length, stepLength, stepHeight, numSteps, friction, stairsColor);
	}

	void makePyramids(Vec3 position, float angle)
	{
		float width, length, stepLength, stepHeight, friction;
		int numSteps;

		width = Randf(minWidth, maxWidth);
		length = Randf(minLength, maxLength);
		stepLength = Randf(minStepLength, maxStepLength);
		stepHeight = Randf(minStepHeight, maxStepHeight);
		friction = Randf(0.9f, 1.f);

		numSteps = Rand(minNumSteps, maxNumSteps + 1);
		createPyramid(position, angle, width, length, stepLength, stepHeight, numSteps, friction, stairsColor);
	}

	void Spawn(Vec3 startPos) {}

	void SpawnGrid(Vec3 startPos, float offsetX, int numCols, float offsetY, int numRows, float spacing, 
                    float minAngle, float maxAngle)
    {
        float angle;
        for (int i = 0; i < numCols; i++)
        {
			for (int j = 0; j < numRows; j++)
			{
				angle = Randf(minAngle, maxAngle);
				Vec2 loc = Grid(offsetX, numCols, offsetY, numRows, spacing, i, j);
				makeStairs(startPos + Vec3(loc.x, 0.f, loc.y), angle);
			}
        }
    }
};


class SpawnableGapObstacles : public SpawnableObstacle
{
public:

	float areaWidth, areaLength, spacing;
	float minGapDistance, maxGapDistance;

	// block sizes
	float minWidth, maxWidth, minHeight, maxHeight, minLength, maxLength;

	float minHeightOffset, maxHeightOffset;
	Vec3 obstacleColor;

	// TODO(jaliang): implement sampling block sizes
	SpawnableGapObstacles() {}
	SpawnableGapObstacles(float spacing, float minDist, float maxDist,
		float minHeight, float maxHeight, Vec3 obstacleColor = Vec3(0.6f, 0.6f, 0.75f)) :
		areaWidth(areaWidth), areaLength(areaLength), spacing(spacing), minGapDistance(minDist), maxGapDistance(maxDist),
		minHeight(minHeight), maxHeight(maxHeight),
		obstacleColor(obstacleColor)
	{
	}

	void makeImmovableBlock(Vec3 position, float angle, Vec3 color)
	{
		float width, length, height, friction;
		width = 0.5f * (spacing - Randf(minGapDistance, maxGapDistance));
		length = 0.5f * (spacing - Randf(minGapDistance, maxGapDistance));
		height = Randf(minHeight, maxHeight);
		friction = Randf(0.9f, 1.f);

		createImmovableBlock(position, angle, width, length, height, friction, color);
	}

	void Spawn(Vec3 startPos) // Do we need it?
	{
		Vec2 deltaPlanePos = UniformSampleRectangle(0.f, areaWidth, 0.f, areaLength);
		makeImmovableBlock(Vec3(
			startPos.x + deltaPlanePos.x,
			startPos.y + Randf(minHeightOffset, maxHeightOffset),
			startPos.z + deltaPlanePos.y),
			Randf() * 2 * kPi, obstacleColor);
	}

	void SpawnGrid(Vec3 startPos, float offsetX, int numCols, float offsetY, int numRows, float spacing,
		float minAngle, float maxAngle, bool randomX = true, bool randomY = true)
	{
		float angle;
		for (int i = 0; i < numCols; ++i)
		{
			for (int j = 0; j < numRows; ++j)
			{
				angle = Randf(minAngle, maxAngle);
				makeImmovableBlock(Vec3(
					startPos.x + (float)i * spacing,
					startPos.y + Randf(minHeightOffset, maxHeightOffset),
					startPos.z + (float)j * spacing
				), angle, obstacleColor);
			}
		}
	}
};