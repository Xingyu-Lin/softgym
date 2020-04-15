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

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../core/types.h"
#include "../core/maths.h"
#include "../core/platform.h"
#include "../core/mesh.h"
#include "../core/voxelize.h"
#include "../core/sdf.h"
#include "../core/pfm.h"
#include "../core/tga.h"
#include "../core/perlin.h"
#include "../core/convex.h"
#include "../core/cloth.h"

#include "../external/SDL2-2.0.4/include/SDL.h"
#include "../external/json/json_utils.h"

#include "../include/NvFlex.h"
#include "../include/NvFlexExt.h"
#include "../include/NvFlexDevice.h"

#include "vr/vr.h"

#include <iostream>
#include <map>
#include <unordered_map>

#include "shaders.h"
#include "imgui.h"

#if FLEX_DX
#include "d3d/shadersDemoContext.h"
class DemoContext;
extern DemoContext* CreateDemoContextD3D12();
extern DemoContext* CreateDemoContextD3D11();
#endif // FLEX_DX

#include "bindings/utils/utils.h"

using json = nlohmann::json;
using namespace std;


SDL_Window* g_window;			// window handle
unsigned int g_windowId;		// window id

#define SDL_CONTROLLER_BUTTON_LEFT_TRIGGER (SDL_CONTROLLER_BUTTON_MAX + 1)
#define SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER (SDL_CONTROLLER_BUTTON_MAX + 2)

int GetKeyFromGameControllerButton(SDL_GameControllerButton button)
{
    switch (button)
    {
    case SDL_CONTROLLER_BUTTON_DPAD_UP:
    {
        return SDLK_q;		   // -- camera translate up
    }
    case SDL_CONTROLLER_BUTTON_DPAD_DOWN:
    {
        return SDLK_z;		   // -- camera translate down
    }
    case SDL_CONTROLLER_BUTTON_DPAD_LEFT:
    {
        return SDLK_h;		   // -- hide GUI
    }
    case SDL_CONTROLLER_BUTTON_DPAD_RIGHT:
    {
        return -1;			   // -- unassigned
    }
    case SDL_CONTROLLER_BUTTON_START:
    {
        return SDLK_RETURN;	   // -- start selected scene
    }
    case SDL_CONTROLLER_BUTTON_BACK:
    {
        return SDLK_ESCAPE;	   // -- quit
    }
    case SDL_CONTROLLER_BUTTON_LEFTSHOULDER:
    {
        return SDLK_UP;		   // -- select prev scene
    }
    case SDL_CONTROLLER_BUTTON_RIGHTSHOULDER:
    {
        return SDLK_DOWN;	   // -- select next scene
    }
    case SDL_CONTROLLER_BUTTON_A:
    {
        return SDLK_g;		   // -- toggle gravity
    }
    case SDL_CONTROLLER_BUTTON_B:
    {
        return SDLK_p;		   // -- pause
    }
    case SDL_CONTROLLER_BUTTON_X:
    {
        return SDLK_r;		   // -- reset
    }
    case SDL_CONTROLLER_BUTTON_Y:
    {
        return SDLK_o;		   // -- step sim
    }
    case SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER:
    {
        return SDLK_SPACE;	   // -- emit particles
    }
    default:
    {
        return -1;			   // -- nop
    }
    };
};

//
// Gamepad thresholds taken from XINPUT API
//
#define XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  7849
#define XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE 8689
#define XINPUT_GAMEPAD_TRIGGER_THRESHOLD    30

int deadzones[3] = { XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE, XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE, XINPUT_GAMEPAD_TRIGGER_THRESHOLD };

inline float joyAxisFilter(int value, int stick)
{
    //clamp values in deadzone to zero, and remap rest of range so that it linearly rises in value from edge of deadzone toward max value.
    if (value < -deadzones[stick])
    {
        return (value + deadzones[stick]) / (32768.0f - deadzones[stick]);
    }
    else if (value > deadzones[stick])
    {
        return (value - deadzones[stick]) / (32768.0f - deadzones[stick]);
    }
    else
    {
        return 0.0f;
    }
}

SDL_GameController* g_gamecontroller = NULL;

int g_windowWidth = 72;
int g_windowHeight = 720;
int g_screenWidth = g_windowWidth;
int g_screenHeight = g_windowHeight;
int g_msaaSamples = 4; // Xingyu: Different from 1.0
int g_upscaling = 3;

int g_numSubsteps;

// a setting of -1 means Flex will use the device specified in the NVIDIA control panel
int g_device = -1;
int g_rank = 0; // Rank for multi GPU purpose

char g_deviceName[256];
bool g_vsync = true;

bool g_benchmark = false;
bool g_extensions = true;
bool g_teamCity = false;
bool g_interop = false;
bool g_d3d12 = false;
bool g_headless = false;
bool g_render = true;
bool g_useAsyncCompute = true;
bool g_increaseGfxLoadForAsyncComputeTesting = false;

char g_experimentFilter[255];
int g_experimentLength = 30;
bool g_experiment = false;


FluidRenderer* g_fluidRenderer;
FluidRenderBuffers* g_fluidRenderBuffers;
DiffuseRenderBuffers* g_diffuseRenderBuffers;

vector<FluidRenderer*> g_sensorFluidRenderers;

NvFlexSolver* g_solver;
NvFlexSolverDesc g_solverDesc;
NvFlexLibrary* g_flexLib;
NvFlexParams g_params;
NvFlexTimers g_timers;
int g_numDetailTimers;
NvFlexDetailTimer * g_detailTimers;

NvFlexTimers g_timersAvg;
NvFlexTimers g_timersVar;
int g_timersCount = 0;

int g_maxDiffuseParticles;
unsigned char g_maxNeighborsPerParticle;
int g_numExtraParticles;
int g_numExtraMultiplier = 1;

// mesh used for deformable object rendering
Mesh* g_mesh;
vector<int> g_meshSkinIndices;
vector<float> g_meshSkinWeights;
vector<Point3> g_meshRestPositions;
const int g_numSkinWeights = 4;

// mapping of collision mesh to render mesh
std::map<NvFlexConvexMeshId, RenderMesh*> g_convexes;
std::map<NvFlexTriangleMeshId, RenderMesh*> g_meshes;
std::map<NvFlexDistanceFieldId, RenderMesh*> g_fields;

RenderMesh* g_sphereMesh;
RenderMesh* g_cylinderMesh;
RenderMesh* g_boxMesh;

struct RenderAttachment
{
	RenderMesh* mesh;
	RenderMaterial material;

	int startTri;
	int endTri;

    int parent;			// index of the parent body
    Transform origin;	// local offset of this attachment
};

std::vector<RenderAttachment> g_renderAttachments;
std::vector<RenderMaterial> g_renderMaterials;

// FEM materials
std::vector<NvFlexFEMMaterial> g_tetraMaterials;

// flag to request collision shapes be updated
bool g_shapesChanged = false;

/* Note that this array of colors is altered by demo code, and is also read from global by graphics API impls */
Colour g_colors[] =
{
        Colour(0.000f, 0.500f, 1.000f),
        Colour(0.875f, 0.782f, 0.051f),
        Colour(0.800f, 0.100f, 0.100f),
        Colour(0.673f, 0.111f, 0.000f),
        Colour(0.612f, 0.194f, 0.394f),
        Colour(0.0f, 1.f, 0.0f),
        Colour(0.797f, 0.354f, 0.000f),
        Colour(0.092f, 0.465f, 0.820f),
        Colour(1.0f, 1.0f, 1.0f),
        Colour(0.0f, 0.0f, 0.0f),
        Colour(0.0f, 0.0f, 0.0f)
};

struct SimBuffers
{
    // particle data
    NvFlexVector<Vec4> positions;
    NvFlexVector<Vec4> restPositions;
    NvFlexVector<Vec3> velocities;
    NvFlexVector<int> phases;
    NvFlexVector<float> densities;
    NvFlexVector<Vec4> anisotropy1;
    NvFlexVector<Vec4> anisotropy2;
    NvFlexVector<Vec4> anisotropy3;
    NvFlexVector<Vec4> normals;
    NvFlexVector<Vec4> smoothPositions;
    NvFlexVector<Vec4> diffusePositions;
    NvFlexVector<Vec4> diffuseVelocities;
    NvFlexVector<int> diffuseCount;

    NvFlexVector<int> activeIndices;

    // static geometry
    NvFlexVector<NvFlexCollisionGeometry> shapeGeometry;
    NvFlexVector<Vec4> shapePositions;
    NvFlexVector<Quat> shapeRotations;
    NvFlexVector<Vec4> shapePrevPositions;
    NvFlexVector<Quat> shapePrevRotations;
    NvFlexVector<int> shapeFlags;

    // shape matching
    NvFlexVector<int> shapeMatchingOffsets;
    NvFlexVector<int> shapeMatchingIndices;
    NvFlexVector<int> shapeMatchingMeshSize;
    NvFlexVector<float> shapeMatchingCoefficients;
    NvFlexVector<float> shapeMatchingPlasticThresholds;
    NvFlexVector<float> shapeMatchingPlasticCreeps;
    NvFlexVector<Quat> shapeMatchingRotations;
    NvFlexVector<Vec3> shapeMatchingTranslations;
    NvFlexVector<Vec3> shapeMatchingLocalPositions;
    NvFlexVector<Vec4> shapeMatchingLocalNormals;

    // inflatables
    NvFlexVector<int> inflatableTriOffsets;
    NvFlexVector<int> inflatableTriCounts;
    NvFlexVector<float> inflatableVolumes;
    NvFlexVector<float> inflatableCoefficients;
    NvFlexVector<float> inflatablePressures;

    // springs
    NvFlexVector<int> springIndices;
	NvFlexVector<float> springLengths;
	NvFlexVector<float> springStiffness;

	//Rigid to Particle attachment
	NvFlexVector<NvFlexRigidParticleAttachment> rigidParticleAttachments;

    // tetrahedra
    NvFlexVector<int> tetraIndices;
    NvFlexVector<Matrix33> tetraRestPoses;
    NvFlexVector<float> tetraStress;
    NvFlexVector<int> tetraMaterials;

    // rigid bodies
    NvFlexVector<NvFlexRigidBody> rigidBodies;
    NvFlexVector<NvFlexRigidShape> rigidShapes;
    NvFlexVector<NvFlexRigidJoint> rigidJoints;

	// Cables
	NvFlexVector<NvFlexCableLink> cableLinks;

    // cloth mesh
    NvFlexVector<int> triangles;
    NvFlexVector<Vec3> triangleNormals;
    NvFlexVector<int> triangleFeatures;

    NvFlexVector<Vec3> uvs;

    SimBuffers(NvFlexLibrary* l) :
        positions(l), restPositions(l), velocities(l), phases(l), densities(l),
        anisotropy1(l), anisotropy2(l), anisotropy3(l), normals(l), smoothPositions(l),
        diffusePositions(l), diffuseVelocities(l), diffuseCount(l), activeIndices(l),
        shapeGeometry(l), shapePositions(l), shapeRotations(l), shapePrevPositions(l),
        shapePrevRotations(l),	shapeFlags(l), shapeMatchingOffsets(l), shapeMatchingIndices(l), shapeMatchingMeshSize(l),
        shapeMatchingCoefficients(l), shapeMatchingPlasticThresholds(l), shapeMatchingPlasticCreeps(l), shapeMatchingRotations(l), shapeMatchingTranslations(l),
        shapeMatchingLocalPositions(l), shapeMatchingLocalNormals(l), inflatableTriOffsets(l),
        inflatableTriCounts(l), inflatableVolumes(l), inflatableCoefficients(l),
        inflatablePressures(l), springIndices(l), springLengths(l), springStiffness(l),
		rigidParticleAttachments(l),
        tetraIndices(l), tetraStress(l), tetraRestPoses(l), tetraMaterials(l),
        rigidBodies(l), rigidShapes(l), rigidJoints(l), cableLinks(l),
        triangles(l), triangleFeatures(l), triangleNormals(l), uvs(l)
    {}
};

SimBuffers* g_buffers;

void MapBuffers(SimBuffers* buffers)
{
    buffers->positions.map();
    buffers->restPositions.map();
    buffers->velocities.map();
    buffers->phases.map();
    buffers->densities.map();
    buffers->anisotropy1.map();
    buffers->anisotropy2.map();
    buffers->anisotropy3.map();
    buffers->normals.map();
    buffers->diffusePositions.map();
    buffers->diffuseVelocities.map();
    buffers->diffuseCount.map();
    buffers->smoothPositions.map();
    buffers->activeIndices.map();

    buffers->shapeGeometry.map();
    buffers->shapePositions.map();
    buffers->shapeRotations.map();
    buffers->shapePrevPositions.map();
    buffers->shapePrevRotations.map();
    buffers->shapeFlags.map();

    buffers->shapeMatchingOffsets.map();
    buffers->shapeMatchingIndices.map();
    buffers->shapeMatchingMeshSize.map();
    buffers->shapeMatchingCoefficients.map();
    buffers->shapeMatchingPlasticThresholds.map();
    buffers->shapeMatchingPlasticCreeps.map();
    buffers->shapeMatchingRotations.map();
    buffers->shapeMatchingTranslations.map();
    buffers->shapeMatchingLocalPositions.map();
    buffers->shapeMatchingLocalNormals.map();

    buffers->springIndices.map();
    buffers->springLengths.map();
    buffers->springStiffness.map();

    buffers->tetraIndices.map();
    buffers->tetraStress.map();
    buffers->tetraRestPoses.map();
    buffers->tetraMaterials.map();

    buffers->rigidBodies.map();
    buffers->rigidShapes.map();
    buffers->rigidJoints.map();
	buffers->cableLinks.map();

    buffers->inflatableTriOffsets.map();
    buffers->inflatableTriCounts.map();
    buffers->inflatableVolumes.map();
    buffers->inflatableCoefficients.map();
    buffers->inflatablePressures.map();

    buffers->triangles.map();
    buffers->triangleNormals.map();
    buffers->triangleFeatures.map();
    buffers->uvs.map();

	buffers->rigidParticleAttachments.map();
}

void UnmapBuffers(SimBuffers* buffers)
{
    // particles
    buffers->positions.unmap();
    buffers->restPositions.unmap();
    buffers->velocities.unmap();
    buffers->phases.unmap();
    buffers->densities.unmap();
    buffers->anisotropy1.unmap();
    buffers->anisotropy2.unmap();
    buffers->anisotropy3.unmap();
    buffers->normals.unmap();
    buffers->diffusePositions.unmap();
    buffers->diffuseVelocities.unmap();
    buffers->diffuseCount.unmap();
    buffers->smoothPositions.unmap();
    buffers->activeIndices.unmap();

    // convexes
    buffers->shapeGeometry.unmap();
    buffers->shapePositions.unmap();
    buffers->shapeRotations.unmap();
    buffers->shapePrevPositions.unmap();
    buffers->shapePrevRotations.unmap();
    buffers->shapeFlags.unmap();

    // rigids
    buffers->shapeMatchingOffsets.unmap();
    buffers->shapeMatchingIndices.unmap();
    buffers->shapeMatchingMeshSize.unmap();
    buffers->shapeMatchingCoefficients.unmap();
    buffers->shapeMatchingPlasticThresholds.unmap();
    buffers->shapeMatchingPlasticCreeps.unmap();
    buffers->shapeMatchingRotations.unmap();
    buffers->shapeMatchingTranslations.unmap();
    buffers->shapeMatchingLocalPositions.unmap();
    buffers->shapeMatchingLocalNormals.unmap();

    // springs
    buffers->springIndices.unmap();
    buffers->springLengths.unmap();
    buffers->springStiffness.unmap();

    // tetra
    buffers->tetraIndices.unmap();
    buffers->tetraStress.unmap();
    buffers->tetraRestPoses.unmap();
    buffers->tetraMaterials.unmap();

    // rigids
    buffers->rigidBodies.unmap();
    buffers->rigidShapes.unmap();
    buffers->rigidJoints.unmap();
	buffers->cableLinks.unmap();

    // inflatables
    buffers->inflatableTriOffsets.unmap();
    buffers->inflatableTriCounts.unmap();
    buffers->inflatableVolumes.unmap();
    buffers->inflatableCoefficients.unmap();
    buffers->inflatablePressures.unmap();

    // triangles
    buffers->triangles.unmap();
    buffers->triangleNormals.unmap();
    buffers->triangleFeatures.unmap();
    buffers->uvs.unmap();

	buffers->rigidParticleAttachments.unmap();

}

SimBuffers* AllocBuffers(NvFlexLibrary* lib)
{
    return new SimBuffers(lib);
}

void DestroyBuffers(SimBuffers* buffers)
{
    // particles
    buffers->positions.destroy();
    buffers->restPositions.destroy();
    buffers->velocities.destroy();
    buffers->phases.destroy();
    buffers->densities.destroy();
    buffers->anisotropy1.destroy();
    buffers->anisotropy2.destroy();
    buffers->anisotropy3.destroy();
    buffers->normals.destroy();
    buffers->diffusePositions.destroy();
    buffers->diffuseVelocities.destroy();
    buffers->diffuseCount.destroy();
    buffers->smoothPositions.destroy();
    buffers->activeIndices.destroy();

    // convexes
    buffers->shapeGeometry.destroy();
    buffers->shapePositions.destroy();
    buffers->shapeRotations.destroy();
    buffers->shapePrevPositions.destroy();
    buffers->shapePrevRotations.destroy();
    buffers->shapeFlags.destroy();

    // shape matching
    buffers->shapeMatchingOffsets.destroy();
    buffers->shapeMatchingIndices.destroy();
    buffers->shapeMatchingMeshSize.destroy();
    buffers->shapeMatchingCoefficients.destroy();
    buffers->shapeMatchingPlasticThresholds.destroy();
    buffers->shapeMatchingPlasticCreeps.destroy();
    buffers->shapeMatchingRotations.destroy();
    buffers->shapeMatchingTranslations.destroy();
    buffers->shapeMatchingLocalPositions.destroy();
    buffers->shapeMatchingLocalNormals.destroy();

    // springs
    buffers->springIndices.destroy();
    buffers->springLengths.destroy();
    buffers->springStiffness.destroy();

    // tetra
    buffers->tetraIndices.destroy();
    buffers->tetraStress.destroy();
    buffers->tetraRestPoses.destroy();
    buffers->tetraMaterials.destroy();

    // rigids
    buffers->rigidBodies.destroy();
    buffers->rigidShapes.destroy();
    buffers->rigidJoints.destroy();
	buffers->cableLinks.destroy();

    // inflatables
    buffers->inflatableTriOffsets.destroy();
    buffers->inflatableTriCounts.destroy();
    buffers->inflatableVolumes.destroy();
    buffers->inflatableCoefficients.destroy();
    buffers->inflatablePressures.destroy();

    // triangles
    buffers->triangles.destroy();
    buffers->triangleNormals.destroy();
    buffers->triangleFeatures.destroy();
    buffers->uvs.destroy();

	buffers->rigidParticleAttachments.destroy();
    delete buffers;
}

struct RenderSensor
{
	int parent;			// index of the parent body (-1 if free)
	Transform origin;	// local transform of the camera to parent body

	float fov;			// vertical field of view (in radians)

	int width;
	int height;

	float* rgbd;		// rgba32 pixel data
	RenderTexture* target;

	int fluidRendererId = -1;

	DepthRenderProfile depthProfile;
};

DepthRenderProfile defaultDepthProfile = {
	0.f, // minRange
	0.f, // maxRange
};

std::vector<RenderSensor> g_renderSensors;
int g_visSensorId = 0; // id of the one sensor to be visualized

					   // uses the convention from URDF that z-forward, x-right and y-down (opposite to OpenGL which has -z forward)
size_t AddSensor(int width, int height, int parent, Transform origin, float fov, bool renderFluids = false, DepthRenderProfile depthProfile = defaultDepthProfile)
{
	RenderSensor s;
	s.parent = parent;
	s.width = width;
	s.height = height;
	s.origin = origin;
	s.fov = fov;

	// allocate images
	s.rgbd = new float[width*height * 4];

	s.target = CreateRenderTarget(width, height, true);

	// adding fluid renderers
	if (renderFluids) 
	{
		uint32_t numParticles = g_buffers->positions.size();
		uint32_t maxParticles = numParticles + g_numExtraParticles * g_numExtraMultiplier;

		FluidRenderer* fluidRenderer = CreateFluidRenderer(width, height);
		s.fluidRendererId = (int)g_sensorFluidRenderers.size();
		g_sensorFluidRenderers.push_back(fluidRenderer);
	}

	// set depth render profile
	s.depthProfile = depthProfile;

	g_renderSensors.push_back(s);

	// returns id of added sensor
	return g_renderSensors.size() - 1;
}

size_t AddPrimesenseSensor(int parent, Transform origin, float scale = 1.f, bool renderFluids = false)
{
	// TODO(jaliang): Color sensor should have higher res
	DepthRenderProfile primesenseDepthProfile = {
		0.35f, // minRange
		3.f, // maxRange
	};
	return AddSensor(int(640.f * scale), int(480.f * scale), parent, origin, DegToRad(45.f), renderFluids, primesenseDepthProfile);
}

float* ReadSensor(int sensorId)
{
	// TODO(jaliang): Need different modes
	return g_renderSensors[sensorId].rgbd;
}

void SetSensorOrigin(int sensorId, Transform origin)
{
    g_renderSensors[sensorId].origin = origin;
}

Transform GetSensorOrigin(int sensorId)
{
    return g_renderSensors[sensorId].origin;
}

void SetVisSensor(int sensorId)
{
	g_visSensorId = sensorId;
}

Vec3 g_camPos(6.0f, 8.0f, 18.0f);
Vec3 g_camAngle(0.0f, -DegToRad(20.0f), 0.0f);
Vec3 g_camVel(0.0f);
Vec3 g_camSmoothVel(0.0f);

float g_camSpeed;
float g_camNear;
float g_camFar;

Vec3 g_lightPos;
Vec3 g_lightDir;
Vec3 g_lightTarget;

bool g_doLearning = false;
bool g_pause = false;
bool g_step = false;
bool g_capture = false;
bool g_showHelp = true;
bool g_tweakPanel = true;
bool g_fullscreen = false;
bool g_wireframe = false;
bool g_debug = false;

bool g_emit = false;
bool g_warmup = false;

float g_windTime = 0.0f;
float g_windFrequency = 0.1f;
float g_windStrength = 0.0f;

bool g_wavePool = false;
float g_waveTime = 0.0f;
float g_wavePlane;
float g_waveFrequency = 1.5f;
float g_waveAmplitude = 1.0f;
float g_waveFloorTilt = 0.0f;

Vec3 g_sceneLower;
Vec3 g_sceneUpper;
float g_sceneMinBoxSize = 2.0f;

float g_blur;
float g_ior;
bool g_drawEllipsoids;
bool g_drawPoints;
bool g_drawMesh;
bool g_drawCloth;
float g_expandCloth;	// amount to expand cloth along normal (to account for particle radius)

bool g_drawOpaque;
int g_drawSprings;		// 0: no draw, 1: draw stretch 2: draw tether
bool g_drawBases = false;
bool g_drawJoints = false;
bool g_drawContacts = false;
bool g_drawNormals = false;
bool g_drawSensors = false;
bool g_drawVisual = true;
bool g_drawCable = true;
bool g_cableDirty = true;
bool g_drawDiffuse;
bool g_drawShapeGrid = false;
bool g_drawDensity = false;
bool g_drawRopes;
float g_pointScale;
float g_ropeScale;
float g_drawPlaneBias;	// move planes along their normal for rendering

float g_diffuseScale;
float g_diffuseMotionScale;
bool g_diffuseShadow;
float g_diffuseInscatter;
float g_diffuseOutscatter;

float g_dt = 1.0f / 100.0f;	// the time delta used for simulation
float g_realdt;				// the real world time delta between updates

float g_waitTime;		// the CPU time spent waiting for the GPU
float g_updateTime;     // the CPU time spent on Flex
float g_renderTime;		// the CPU time spent calling OpenGL to render the scene
// the above times don't include waiting for vsync
float g_simLatency;     // the time the GPU spent between the first and last NvFlexUpdateSolver() operation. Because some GPUs context switch, this can include graphics time.

int g_frame = 0;
int g_numSolidParticles = 0;

int g_mouseParticle = -1;
int g_mouseJoint = -1;

float g_mouseT = 0.0f;
Vec3 g_mousePos;
float g_mouseMass;
bool g_mousePicked = false;

// mouse
int g_lastx;
int g_lasty;
int g_lastb = -1;

bool g_profile = false;
bool g_outputAllFrameTimes = false;
bool g_asyncComputeBenchmark = false;

ShadowMap* g_shadowMap;

Vec4 g_fluidColor;
Vec4 g_diffuseColor;
Vec3 g_meshColor;
Vec3  g_clearColor(0.065f, 0.065f, 0.073f);
float g_lightDistance;
float g_fogDistance = 0.02f;

FILE* g_ffmpeg;
int g_argc;
char** g_argv;

// For controlling joints with numpads
unordered_map<int, bool> g_numpadPressedState = {
	{ SDLK_KP_0, false },
	{ SDLK_KP_1, false },
	{ SDLK_KP_2, false },
	{ SDLK_KP_3, false },
	{ SDLK_KP_4, false },
	{ SDLK_KP_5, false },
	{ SDLK_KP_6, false },
	{ SDLK_KP_7, false },
	{ SDLK_KP_8, false },
	{ SDLK_KP_9, false },
	{ SDLK_KP_PLUS, false },
	{ SDLK_KP_MINUS, false },	
};

void DrawStaticShapes();
void DrawRigidShapes(bool wire=false, int offset=0);
void DrawCables();
void DrawRigidAttachments();
void DrawJoints();
void DrawSensors(const int numParticles, const int numDiffuse, float radius, Matrix44 lightTransform);
void DrawBasis(const Matrix33& frame, const Vec3& p, bool depthTest=false);

struct Emitter
{
    Emitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f), mWidth(8)   {}

    Vec3 mPos;
    Vec3 mDir;
    Vec3 mRight;
    float mSpeed;
    bool mEnabled;
    float mLeftOver;
    int mWidth;
};

std::vector<Emitter> g_emitters(1);	// first emitter is the camera 'gun'

struct Rope
{
    std::vector<int> mIndices;
};

std::vector<Rope> g_ropes;

inline float sqr(float x)
{
    return x*x;
}

class Scene;
Scene* g_scene;

#ifdef NV_FLEX_GYM
class RLFlexEnv;
RLFlexEnv* g_rlflexenv = nullptr;
#endif

int g_sceneIndex = 0;
int g_selectedScene = g_sceneIndex;
int g_sceneScroll;			// offset for level selection scroll area
bool g_resetScene = false;  //if the user clicks the reset button or presses the reset key this is set to true;
json g_sceneJson;

#include "helpers.h"
#include "scenes.h"
#include "benchmark.h"

std::vector<SceneFactory> g_sceneFactories;

inline Matrix44 GetCameraRotationMatrix(bool getInversed = false)
{
	const float multiplier = (getInversed ? -1.f : 1.f);
	const float camAngleX = multiplier * g_camAngle.x;
	const float camAngleY = multiplier * g_camAngle.y;

	return RotationMatrix(camAngleX, Vec3(0.0f, 1.0f, 0.0f)) * RotationMatrix(camAngleY, Vec3(cosf(camAngleX), 0.0f, sinf(camAngleX)));
}


void InitScene(int scene, py::array_t<float> scene_params, bool centerCamera, int thread_idx)
{
    if (g_sceneFactories[scene].mIsVR && !g_vrSystem)
    {
        printf("Scene \"%s\" requires VR but it wasn't initialized.\nTerminating process.", g_sceneFactories[scene].mName);
        exit(1);
    }

    if (g_buffers)
    {
        // Wait for any running GPU work to finish
        MapBuffers(g_buffers);
        UnmapBuffers(g_buffers);
    }

    if (g_scene)
    {
        delete g_scene;
        g_scene = NULL;
    }

    if (g_solver)
    {
        if (g_buffers)
        {
            DestroyBuffers(g_buffers);
        }

		if (g_render)
		{
			DestroyFluidRenderBuffers(g_fluidRenderBuffers);
			DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);
		}

        for (auto& iter : g_meshes)
        {
            NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
            DestroyRenderMesh(iter.second);
        }

        for (auto& iter : g_fields)
        {
            NvFlexDestroyDistanceField(g_flexLib, iter.first);
            DestroyRenderMesh(iter.second);
        }

        for (auto& iter : g_convexes)
        {
            NvFlexDestroyConvexMesh(g_flexLib, iter.first);
            DestroyRenderMesh(iter.second);
        }


        g_fields.clear();
        g_meshes.clear();
        g_convexes.clear();

        g_renderAttachments.clear();

        g_renderMaterials.clear();

        NvFlexDestroySolver(g_solver);
        g_solver = NULL;
	}

	if (g_render == true)
	{
        // destroy old sensors
        for (int i = 0; (unsigned int)i < g_renderSensors.size(); i++)
        {
            RenderSensor sensor = g_renderSensors[i];
            delete[] sensor.rgbd;
            DestroyRenderTexture(sensor.target);

            if (sensor.fluidRendererId != -1)
                DestroyFluidRenderer(g_sensorFluidRenderers[sensor.fluidRendererId]);
        }

        g_renderSensors.clear();

        RenderMaterial defaultMat;
        defaultMat.frontColor = Vec3(SrgbToLinear(Colour(71.0f/255.0f, 165.0f/255.0f, 1.0f)));
        defaultMat.backColor = defaultMat.frontColor;
        defaultMat.roughness = 1.0f;
        defaultMat.specular = 0.1f;
        defaultMat.metallic = 0.0f;

        g_renderMaterials.push_back(defaultMat);

        // create default render materials
        for (int i=0; i < 8; ++i)
        {
            defaultMat.frontColor = Vec3(g_colors[i]);
            defaultMat.backColor = Vec3(g_colors[i+1]);
            defaultMat.specular = 0.0f;
            
            g_renderMaterials.push_back(defaultMat);
        }
    }

    memset(&g_timersAvg, 0, sizeof(g_timersAvg));
    memset(&g_timersVar, 0, sizeof(g_timersVar));
    g_timersCount = 0;

    // alloc buffers
    g_buffers = AllocBuffers(g_flexLib);

    // map during initialization
    MapBuffers(g_buffers);

    g_buffers->positions.resize(0);
    g_buffers->velocities.resize(0);
    g_buffers->phases.resize(0);

    g_buffers->shapeMatchingOffsets.resize(0);
    g_buffers->shapeMatchingIndices.resize(0);
    g_buffers->shapeMatchingMeshSize.resize(0);
    g_buffers->shapeMatchingRotations.resize(0);
    g_buffers->shapeMatchingTranslations.resize(0);
    g_buffers->shapeMatchingCoefficients.resize(0);
    g_buffers->shapeMatchingPlasticThresholds.resize(0);
    g_buffers->shapeMatchingPlasticCreeps.resize(0);
    g_buffers->shapeMatchingLocalPositions.resize(0);
    g_buffers->shapeMatchingLocalNormals.resize(0);

    g_buffers->tetraIndices.resize(0);
    g_buffers->tetraMaterials.resize(0);
    g_buffers->tetraRestPoses.resize(0);
    g_buffers->tetraStress.resize(0);

    g_buffers->springIndices.resize(0);
    g_buffers->springLengths.resize(0);
    g_buffers->springStiffness.resize(0);
    g_buffers->triangles.resize(0);
    g_buffers->triangleNormals.resize(0);
    g_buffers->uvs.resize(0);

    g_meshSkinIndices.resize(0);
    g_meshSkinWeights.resize(0);

    g_emitters.resize(1);
    g_emitters[0].mEnabled = false;
    g_emitters[0].mSpeed = 1.0f;
    g_emitters[0].mLeftOver = 0.0f;
    g_emitters[0].mWidth = 8;

    g_buffers->shapeGeometry.resize(0);
    g_buffers->shapePositions.resize(0);
    g_buffers->shapeRotations.resize(0);
    g_buffers->shapePrevPositions.resize(0);
    g_buffers->shapePrevRotations.resize(0);
    g_buffers->shapeFlags.resize(0);

    g_ropes.resize(0);

    // remove collision shapes
    delete g_mesh;
    g_mesh = NULL;

    g_frame = 0;
    g_pause = false;

    g_dt = 1.0f / 100.0f;
    g_waveTime = 0.0f;
    g_windTime = 0.0f;
    g_windStrength = 1.0f;

    g_blur = 1.0f;
    g_fluidColor = Vec4(0.1f, 0.4f, 0.8f, 1.0f);
    g_meshColor = Vec3(0.9f, 0.9f, 0.9f);
    g_drawEllipsoids = false;
    g_drawPoints = true;
    g_drawCloth = true;
    g_expandCloth = 0.0f;

    g_drawOpaque = false;
    g_drawSprings = false;
    g_drawDiffuse = false;
    g_drawMesh = true;
    g_drawRopes = true;
    g_drawDensity = false;
    g_drawJoints = false;
    g_drawVisual = true;
    
    g_ior = 1.0f;
    g_lightDistance = 2.0f;

    g_camSpeed = 0.075f;
    g_camNear = 0.01f;
    g_camFar = 1000.0f;

    g_pointScale = 1.0f;
    g_ropeScale = 1.0f;
    g_drawPlaneBias = 0.0f;

	g_numSubsteps = 2;
	
    g_diffuseScale = 0.5f;
    g_diffuseColor = 1.0f;
    g_diffuseMotionScale = 1.0f;
    g_diffuseShadow = false;
    g_diffuseInscatter = 0.8f;
    g_diffuseOutscatter = 0.53f;

    // reset phase 0 particle color to blue
    g_colors[0] = Colour(0.0f, 0.5f, 1.0f);

    g_numSolidParticles = 0;

    g_waveFrequency = 1.5f;
    g_waveAmplitude = 1.5f;
    g_waveFloorTilt = 0.0f;
    g_emit = false;
    g_warmup = false;

    g_mouseParticle = -1;
	g_mouseJoint = -1;

    g_maxDiffuseParticles = 0;	// number of diffuse particles
    g_maxNeighborsPerParticle = 96;
    g_numExtraParticles = 0;	// number of particles allocated but not made active

    g_sceneLower = FLT_MAX;
    g_sceneUpper = -FLT_MAX;

    g_lightDir = Normalize(Vec3(5.0f, 15.0f, 7.5f));

    // init sim params
	NvFlexInitParams(&g_params);
  
    // initialize solver desc
    NvFlexSetSolverDescDefaults(&g_solverDesc);

    //initialize with a ground plane
    g_params.numPlanes = 1;

    // create scene
    StartGpuWork();
    g_scene = g_sceneFactories[scene].mFactory();
    g_scene->PrepareScene();
    EndGpuWork();

    uint32_t numParticles = g_buffers->positions.size();
    uint32_t maxParticles = numParticles + g_numExtraParticles*g_numExtraMultiplier;

    if (g_params.solidRestDistance == 0.0f)
    {
        g_params.solidRestDistance = g_params.radius;
    }

    // if fluid present then we assume solid particles have the same radius
    if (g_params.fluidRestDistance > 0.0f)
    {
        g_params.solidRestDistance = g_params.fluidRestDistance;
    }

    // set collision distance automatically based on rest distance if not alraedy set
    if (g_params.collisionDistance == 0.0f)
    {
        g_params.collisionDistance = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f;
    }

    // default particle friction to 10% of shape friction
    if (g_params.particleFriction == 0.0f)
    {
        g_params.particleFriction = g_params.dynamicFriction*0.1f;
    }

    // add a margin for detecting contacts between particles and shapes
    if (g_params.shapeCollisionMargin == 0.0f)
    {
        g_params.shapeCollisionMargin = g_params.collisionDistance*0.5f;
    }

    // calculate particle bounds
    Vec3 particleLower, particleUpper;
    GetParticleBounds(particleLower, particleUpper);

    // accommodate shapes
    Vec3 shapeLower, shapeUpper;
    GetShapeBounds(shapeLower, shapeUpper);

    // update bounds
    g_sceneLower = Min(Min(g_sceneLower, particleLower), shapeLower);
    g_sceneUpper = Max(Max(g_sceneUpper, particleUpper), shapeUpper);

    g_sceneLower -= g_params.collisionDistance;
    g_sceneUpper += g_params.collisionDistance;

    // update collision planes to match flexs
    Vec3 up = Normalize(Vec3(-g_waveFloorTilt, 1.0f, 0.0f));

    (Vec4&)g_params.planes[0] = Vec4(up.x, up.y, up.z, 0.0f);
    (Vec4&)g_params.planes[1] = Vec4(0.0f, 0.0f, 1.0f, -g_sceneLower.z);
    (Vec4&)g_params.planes[2] = Vec4(1.0f, 0.0f, 0.0f, -g_sceneLower.x);
    (Vec4&)g_params.planes[3] = Vec4(-1.0f, 0.0f, 0.0f, g_sceneUpper.x);
    (Vec4&)g_params.planes[4] = Vec4(0.0f, 0.0f, -1.0f, g_sceneUpper.z);
    (Vec4&)g_params.planes[5] = Vec4(0.0f, -1.0f, 0.0f, g_sceneUpper.y);

    g_wavePlane = g_params.planes[2][3];

    g_buffers->diffusePositions.resize(g_maxDiffuseParticles);
    g_buffers->diffuseVelocities.resize(g_maxDiffuseParticles);
    g_buffers->diffuseCount.resize(1, 0);

    // for fluid rendering these are the Laplacian smoothed positions
    g_buffers->smoothPositions.resize(maxParticles);

    g_buffers->normals.resize(0);
    g_buffers->normals.resize(maxParticles);

    // initialize normals (just for rendering before simulation starts)
    int numTris = g_buffers->triangles.size() / 3;
    for (int i = 0; i < numTris; ++i)
    {
        Vec3 v0 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 0]]);
        Vec3 v1 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 1]]);
        Vec3 v2 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 2]]);

        Vec3 n = Cross(v1 - v0, v2 - v0);

        g_buffers->normals[g_buffers->triangles[i * 3 + 0]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 1]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 2]] += Vec4(n, 0.0f);
    }

    for (int i = 0; i < int(maxParticles); ++i)
    {
        g_buffers->normals[i] = Vec4(SafeNormalize(Vec3(g_buffers->normals[i]), Vec3(0.0f, 1.0f, 0.0f)), 0.0f);
    }


    // save mesh positions for skinning
    if (g_mesh)
    {
        g_meshRestPositions = g_mesh->m_positions;
    }
    else
    {
        g_meshRestPositions.resize(0);
    }

    g_solverDesc.maxParticles = maxParticles;
    g_solverDesc.maxDiffuseParticles = g_maxDiffuseParticles;
    g_solverDesc.maxNeighborsPerParticle = g_maxNeighborsPerParticle;

    // main create method for the Flex solver
    g_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc);

    // give scene a chance to do some post solver initialization
    g_scene->PostInitialize();

    // center camera on particles
    if (centerCamera)
    {
        g_camPos = Vec3((g_sceneLower.x + g_sceneUpper.x)*0.5f, min(g_sceneUpper.y*1.25f, 6.0f), g_sceneUpper.z + min(g_sceneUpper.y, 6.0f)*2.0f);
        g_camAngle = Vec3(0.0f, -DegToRad(15.0f), 0.0f);

        // give scene a chance to modify camera position
        g_scene->CenterCamera();

        if (g_vrSystem)
        {
            g_vrSystem->SetAnchorPos(g_camPos);
            g_vrSystem->SetAnchorRotation(GetCameraRotationMatrix());
        }
    }

    // create active indices (just a contiguous block for the demo)
    g_buffers->activeIndices.resize(g_buffers->positions.size());
    for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
    {
        g_buffers->activeIndices[i] = i;
    }

    // resize particle buffers to fit
    g_buffers->positions.resize(maxParticles);
    g_buffers->velocities.resize(maxParticles);
    g_buffers->phases.resize(maxParticles);

    g_buffers->densities.resize(maxParticles);
    g_buffers->anisotropy1.resize(maxParticles);
    g_buffers->anisotropy2.resize(maxParticles);
    g_buffers->anisotropy3.resize(maxParticles);

    // resize tetra stress buffer
    g_buffers->tetraStress.resize(g_buffers->tetraRestPoses.size(), 0.0f);

    // save rest positions
    g_buffers->restPositions.resize(g_buffers->positions.size());
    for (int i = 0; i < g_buffers->positions.size(); ++i)
    {
        g_buffers->restPositions[i] = g_buffers->positions[i];
    }

    // builds rigids constraints
    if (g_buffers->shapeMatchingOffsets.size())
    {
        assert(g_buffers->shapeMatchingOffsets.size() > 1);

        const int numRigids = g_buffers->shapeMatchingOffsets.size() - 1;

        // If the centers of mass for the rigids are not yet computed, this is done here
        // (If the CreateParticleShape method is used instead of the NvFlexExt methods, the centers of mass will be calculated here)
        if (g_buffers->shapeMatchingTranslations.size() == 0)
        {
            g_buffers->shapeMatchingTranslations.resize(g_buffers->shapeMatchingOffsets.size() - 1, Vec3());
            CalculateRigidCentersOfMass(&g_buffers->positions[0], g_buffers->positions.size(), &g_buffers->shapeMatchingOffsets[0], &g_buffers->shapeMatchingTranslations[0], &g_buffers->shapeMatchingIndices[0], numRigids);
        }

        // calculate local rest space positions
        g_buffers->shapeMatchingLocalPositions.resize(g_buffers->shapeMatchingOffsets.back());
        CalculateshapeMatchingLocalPositions(&g_buffers->positions[0], &g_buffers->shapeMatchingOffsets[0], &g_buffers->shapeMatchingTranslations[0], &g_buffers->shapeMatchingIndices[0], numRigids, &g_buffers->shapeMatchingLocalPositions[0]);

        // set shapeMatchingRotations to correct length, probably NULL up until here
        g_buffers->shapeMatchingRotations.resize(g_buffers->shapeMatchingOffsets.size() - 1, Quat());
    }

    // build feature information for dynamic triangles
    if (g_buffers->triangles.size())
    {
        NvFlexExtMeshAdjacency* adj = NvFlexExtCreateMeshAdjacency(g_buffers->positions[0], g_buffers->positions.size(), sizeof(Vec4), &g_buffers->triangles[0], g_buffers->triangles.size()/3, false);

        g_buffers->triangleFeatures.assign(adj->triFeatures, adj->numTris);

        NvFlexExtDestroyMeshAdjacency(adj);
    }

    // unmap so we can start transferring data to GPU
    UnmapBuffers(g_buffers);

    //-----------------------------
    // Send data to Flex

    NvFlexCopyDesc copyDesc;
    copyDesc.dstOffset = 0;
    copyDesc.srcOffset = 0;
    copyDesc.elementCount = numParticles;

    NvFlexSetParams(g_solver, &g_params);
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, &copyDesc);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, &copyDesc);
    NvFlexSetNormals(g_solver, g_buffers->normals.buffer, &copyDesc);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, &copyDesc);
    NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, &copyDesc);

    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, &copyDesc);
    NvFlexSetActiveCount(g_solver, numParticles);

    // springs
    if (g_buffers->springIndices.size())
    {
        assert((g_buffers->springIndices.size() & 1) == 0);
        assert((g_buffers->springIndices.size() / 2) == g_buffers->springLengths.size());

        NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springLengths.size());
    }

	// attachments
	if (g_buffers->rigidParticleAttachments.size())
	{
		NvFlexSetRigidParticleAttachments(g_solver, g_buffers->rigidParticleAttachments.buffer, g_buffers->rigidParticleAttachments.size());		
	}

    // tetra
    if (g_tetraMaterials.size())
    {
        NvFlexSetFEMMaterials(g_solver, &g_tetraMaterials[0], g_tetraMaterials.size());
    }

    if (g_buffers->tetraIndices.size())
    {
        NvFlexSetFEMGeometry(g_solver, g_buffers->tetraIndices.buffer, g_buffers->tetraRestPoses.buffer, g_buffers->tetraMaterials.buffer, g_buffers->tetraMaterials.size());
    }

    // shape matching
    if (g_buffers->shapeMatchingOffsets.size())
    {
        NvFlexSetRigids(g_solver, g_buffers->shapeMatchingOffsets.buffer, g_buffers->shapeMatchingIndices.buffer, g_buffers->shapeMatchingLocalPositions.buffer, g_buffers->shapeMatchingLocalNormals.buffer, g_buffers->shapeMatchingCoefficients.buffer, g_buffers->shapeMatchingPlasticThresholds.buffer, g_buffers->shapeMatchingPlasticCreeps.buffer, g_buffers->shapeMatchingRotations.buffer, g_buffers->shapeMatchingTranslations.buffer, g_buffers->shapeMatchingOffsets.size() - 1, g_buffers->shapeMatchingIndices.size());
    }

    // rigids
    if (g_buffers->rigidBodies.size())
    {
        NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
    }

    if (g_buffers->rigidShapes.size())
    {
        NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
    }

    if (g_buffers->rigidJoints.size())
    {
        NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());
    }

	if (g_buffers->cableLinks.size())
	{
		NvFlexSetCableLinks(g_solver, g_buffers->cableLinks.buffer, g_buffers->cableLinks.size());
	}

    // inflatables
    if (g_buffers->inflatableTriOffsets.size())
    {
        NvFlexSetInflatables(g_solver, g_buffers->inflatableTriOffsets.buffer, g_buffers->inflatableTriCounts.buffer, g_buffers->inflatableVolumes.buffer, g_buffers->inflatablePressures.buffer, g_buffers->inflatableCoefficients.buffer, g_buffers->inflatableTriOffsets.size());
    }

    // dynamic triangles
    if (g_buffers->triangles.size())
    {
        NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangleFeatures.buffer, g_buffers->triangles.size() / 3);
    }

    // collision shapes
    if (g_buffers->shapeFlags.size())
    {
        NvFlexSetShapes(
            g_solver,
            g_buffers->shapeGeometry.buffer,
            g_buffers->shapePositions.buffer,
            g_buffers->shapeRotations.buffer,
            g_buffers->shapePrevPositions.buffer,
            g_buffers->shapePrevRotations.buffer,
            g_buffers->shapeFlags.buffer,
            int(g_buffers->shapeFlags.size()));
    }

	if (g_render == true)
	{
		// create render buffers
		g_fluidRenderBuffers = CreateFluidRenderBuffers(maxParticles, g_interop);
		g_diffuseRenderBuffers = CreateDiffuseRenderBuffers(g_maxDiffuseParticles, g_interop);
	}


    // perform initial sim warm up
    if (g_warmup)
    {
        printf("Warming up sim..\n");

        // warm it up (relax positions to reach rest density without affecting velocity)
        NvFlexParams copy = g_params;
        copy.numIterations = 4;

        NvFlexSetParams(g_solver, &copy);

        const int kWarmupIterations = 100;

        for (int i = 0; i < kWarmupIterations; ++i)
        {
            NvFlexUpdateSolver(g_solver, 0.0001f, 1, false);
            NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
        }

        // udpate host copy
        NvFlexGetParticles(g_solver, g_buffers->positions.buffer, NULL);
        NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, NULL);
        NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer, g_buffers->anisotropy3.buffer, NULL);
        NvFlexGetNormals(g_solver, g_buffers->normals.buffer, NULL);
        printf("Finished warm up.\n");
    }
}

void InitScene(int scene, bool centerCamera = true)
{
    py::array_t<float> scene_params;
    InitScene(scene, scene_params, centerCamera, 0);
}

void Reset()
{
    assert(0); // xingyu: Modify the arguments below
//    InitScene(g_sceneIndex, false);
}

void Shutdown()
{
    // free buffers
    DestroyBuffers(g_buffers);

    for (auto& iter : g_meshes)
    {
        NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
        DestroyRenderMesh(iter.second);
    }

    for (auto& iter : g_fields)
    {
        NvFlexDestroyDistanceField(g_flexLib, iter.first);
        DestroyRenderMesh(iter.second);
    }

    for (auto& iter : g_convexes)
    {
        NvFlexDestroyConvexMesh(g_flexLib, iter.first);
        DestroyRenderMesh(iter.second);
    }

    g_fields.clear();
    g_meshes.clear();
    g_convexes.clear();

    NvFlexDestroySolver(g_solver);
    NvFlexShutdown(g_flexLib);
    
    g_buffers = NULL;
    g_solver = NULL;
    g_flexLib = NULL;

    if (g_ffmpeg)
    {
#if _WIN32
        _pclose(g_ffmpeg);
#elif __linux__
        pclose(g_ffmpeg);
#endif
    }
}

void UpdateEmitters()
{
    float spin = DegToRad(15.0f);

    const Vec3 forward(-sinf(g_camAngle.x + spin)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x + spin)*cosf(g_camAngle.y));
    const Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

    g_emitters[0].mDir = Normalize(forward + Vec3(0.0, 0.4f, 0.0f));
    g_emitters[0].mRight = right;
    g_emitters[0].mPos = g_camPos + forward*1.f + Vec3(0.0f, 0.2f, 0.0f) + right*0.65f;

    // process emitters
    if (g_emit)
    {
        int activeCount = NvFlexGetActiveCount(g_solver);

        size_t e = 0;

        // skip camera emitter when moving forward or things get messy
        if (g_camSmoothVel.z >= 0.025f)
        {
            e = 1;
        }

        for (; e < g_emitters.size(); ++e)
        {
            if (!g_emitters[e].mEnabled)
            {
                continue;
            }

            Vec3 emitterDir = g_emitters[e].mDir;
            Vec3 emitterRight = g_emitters[e].mRight;
            Vec3 emitterPos = g_emitters[e].mPos;


            float r = g_params.fluidRestDistance;
            int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);

            float numParticles = (g_emitters[e].mSpeed / r)*g_dt;

            // whole number to emit
            int n = int(numParticles + g_emitters[e].mLeftOver);

            if (n)
            {
                g_emitters[e].mLeftOver = (numParticles + g_emitters[e].mLeftOver) - n;
            }
            else
            {
                g_emitters[e].mLeftOver += numParticles;
            }

            // create a grid of particles (n particles thick)
            for (int k = 0; k < n; ++k)
            {
                int emitterWidth = g_emitters[e].mWidth;
                int numParticles = emitterWidth*emitterWidth;
                for (int i = 0; i < numParticles; ++i)
                {
                    float x = float(i%emitterWidth) - float(emitterWidth/2);
                    float y = float((i / emitterWidth) % emitterWidth) - float(emitterWidth/2);

                    if ((sqr(x) + sqr(y)) <= (emitterWidth / 2)*(emitterWidth / 2))
                    {
                        Vec3 up = Normalize(Cross(emitterDir, emitterRight));
                        Vec3 offset = r*(emitterRight*x + up*y) + float(k)*emitterDir*r;

                        if (activeCount < g_buffers->positions.size())
                        {
                            g_buffers->positions[activeCount] = Vec4(emitterPos + offset, 1.0f);
                            g_buffers->velocities[activeCount] = emitterDir*g_emitters[e].mSpeed;
                            g_buffers->phases[activeCount] = phase;

                            g_buffers->activeIndices.push_back(activeCount);

                            activeCount++;
                        }
                    }
                }
            }
        }
    }
}

void UpdateCamera()
{
    const Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
    const Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

    g_camSmoothVel = Lerp(g_camSmoothVel, g_camVel, 0.1f);

    Vec3 camDelta = forward*g_camSmoothVel.z + right*g_camSmoothVel.x + Cross(right, forward)*g_camSmoothVel.y;
    if (g_vrSystem)
    {
        const Matrix44 camMat = GetCameraRotationMatrix();
        camDelta = g_vrSystem->GetHmdRotation() * AffineInverse(camMat) * camDelta;
        g_camPos += camDelta;
        g_vrSystem->SetAnchorPos(g_camPos);
        g_vrSystem->SetAnchorRotation(camMat);
    }
    else
    {
        g_camPos += camDelta;
    }
}

void UpdateMouse()
{
    // mouse button is up release particle
    if (g_lastb == -1)
    {
        if (g_mouseParticle != -1)
        {
            // restore particle mass
            g_buffers->positions[g_mouseParticle].w = g_mouseMass;

            // deselect
            g_mouseParticle = -1;
        }

        if (g_mouseJoint != -1)
        {
            g_buffers->rigidJoints.resize(g_buffers->rigidJoints.size()-1);

            g_mouseJoint = -1;
        }
    }

    // mouse went down, pick new particle
    if (g_mousePicked)
    {
        assert(g_mouseParticle == -1);

        Vec3 origin, dir;
        GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

        const int numActive = NvFlexGetActiveCount(g_solver);

        if (numActive)
        {
            g_mouseParticle = PickParticle(origin, dir, &g_buffers->positions[0], &g_buffers->phases[0], numActive, g_params.radius*0.8f, g_mouseT);

            if (g_mouseParticle != -1)
            {
                printf("picked: %d, mass: %f v: %f %f %f\n", g_mouseParticle, g_buffers->positions[g_mouseParticle].w, g_buffers->velocities[g_mouseParticle].x, g_buffers->velocities[g_mouseParticle].y, g_buffers->velocities[g_mouseParticle].z);

                g_mousePos = origin + dir*g_mouseT;
                g_mouseMass = g_buffers->positions[g_mouseParticle].w;
                g_buffers->positions[g_mouseParticle].w = 0.0f;		// increase picked particle's mass to force it towards the point
            }
        }

        if (g_mouseParticle == -1)
        {
            // try to pick a rigid
            NvFlexRay ray;
            (Vec3&)ray.dir = dir;
            (Vec3&)ray.start = origin;
            ray.maxT = FLT_MAX;
            ray.filter = 0;
            ray.group = -1;

            NvFlexVector<NvFlexRay> rayBuf(g_flexLib, &ray, 1);
            NvFlexVector<NvFlexRayHit> hitBuf(g_flexLib, 1);

            NvFlexRayCast(g_solver, rayBuf.buffer, hitBuf.buffer, 1);

            // map the hit buffer, this synchronizes to ensure results are ready
            hitBuf.map();

            NvFlexRayHit hit = hitBuf[0];

            printf("picked shape: %d element: %d t: %f normal: %f %f %f\n", hit.shape, hit.element, hit.t, hit.n[0], hit.n[1], hit.n[2]);

            if (hit.shape != -1)
            {
                NvFlexRigidShape shape = g_buffers->rigidShapes[hit.shape];

				if (shape.body != -1)
				{
	                g_mouseJoint = g_buffers->rigidJoints.size();
					g_mousePos = origin + dir*hit.t;
					g_mouseT = hit.t;

					NvFlexRigidBody body = g_buffers->rigidBodies[shape.body];

					Transform bodyPose;
					NvFlexGetRigidPose(&body, (NvFlexRigidPose*)&bodyPose);

					Vec3 attachPoint = InverseTransformPoint(bodyPose, g_mousePos);

					NvFlexRigidJoint joint;
					NvFlexMakeSphericalJoint(&joint, -1, shape.body, NvFlexMakeRigidPose(g_mousePos, Quat()), NvFlexMakeRigidPose(attachPoint, Quat()));

					const float mouseCompliance = 1.e-3f;

					joint.compliance[0] = mouseCompliance;
					joint.compliance[1] = mouseCompliance;
					joint.compliance[2] = mouseCompliance;

					g_buffers->rigidJoints.push_back(joint);
				}
            }


            hitBuf.unmap();
        }

        g_mousePicked = false;
    }

    // update picked particle position
    if (g_mouseParticle != -1)
    {
        Vec3 p = Lerp(Vec3(g_buffers->positions[g_mouseParticle]), g_mousePos, 0.8f);
        Vec3 delta = p - Vec3(g_buffers->positions[g_mouseParticle]);

        g_buffers->positions[g_mouseParticle].x = p.x;
        g_buffers->positions[g_mouseParticle].y = p.y;
        g_buffers->positions[g_mouseParticle].z = p.z;

        g_buffers->velocities[g_mouseParticle].x = delta.x / g_dt;
        g_buffers->velocities[g_mouseParticle].y = delta.y / g_dt;
        g_buffers->velocities[g_mouseParticle].z = delta.z / g_dt;
    }

    if (g_mouseJoint != -1)
    {
        g_buffers->rigidJoints[g_mouseJoint].pose0 = NvFlexMakeRigidPose(g_mousePos, Quat());
    }
}

void UpdateWind()
{
    g_windTime += g_dt;

    const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
    const float kNoise = Perlin1D(g_windTime*g_windFrequency, 10, 0.25f);
    Vec3 wind = g_windStrength*kWindDir*Vec3(kNoise, fabsf(kNoise), 0.0f);

    g_params.wind[0] = wind.x;
    g_params.wind[1] = wind.y;
    g_params.wind[2] = wind.z;

    if (g_wavePool)
    {
        g_waveTime += g_dt;

        g_params.planes[2][3] = g_wavePlane + (sinf(float(g_waveTime)*g_waveFrequency - kPi*0.5f)*0.5f + 0.5f)*g_waveAmplitude;
    }
}

void SyncScene()
{
    // let the scene send updates to flex directly
    g_scene->Sync();
}

void UpdateScene()
{
    // give scene a chance to make changes to particle buffers
    if (!g_experiment)
    {
        g_scene->Update();
    }
}

void RenderScene(int eye = 2, Matrix44* usedProj = nullptr, Matrix44* usedView = nullptr)
{
	for (auto& sensor : g_renderSensors)
	{
		// read back sensor data from previous frame to avoid stall, todo: multi-view / tiled sensor rendering / etc
		ReadRenderTarget(sensor.target, sensor.rgbd, 0, 0, sensor.width, sensor.height);
	}

    const int numParticles = NvFlexGetActiveCount(g_solver);
    const int numDiffuse = g_buffers->diffuseCount[0];

    //---------------------------------------------------
    // use VBO buffer wrappers to allow Flex to write directly to the OpenGL buffers
    // Flex will take care of any CUDA interop mapping/unmapping during the get() operations

    if (numParticles)
    {

        if (g_interop)
        {
            // copy data directly from solver to the renderer buffers
            UpdateFluidRenderBuffers(g_fluidRenderBuffers, g_solver, g_drawEllipsoids, g_drawDensity);
        }
        else
        {
            // copy particle data to GPU render device

            if (g_drawEllipsoids)
            {
                // if fluid surface rendering then update with smooth positions and anisotropy
                UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                                         &g_buffers->smoothPositions[0],
                                         (g_drawDensity) ? &g_buffers->densities[0] : (float*)&g_buffers->phases[0],
                                         &g_buffers->anisotropy1[0],
                                         &g_buffers->anisotropy2[0],
                                         &g_buffers->anisotropy3[0],
                                         g_buffers->positions.size(),
                                         &g_buffers->activeIndices[0],
                                         numParticles);
            }
            else
            {
                // otherwise just send regular positions and no anisotropy
                UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                                         &g_buffers->positions[0],
                                         (float*)&g_buffers->phases[0],
                                         NULL, NULL, NULL,
                                         g_buffers->positions.size(),
                                         &g_buffers->activeIndices[0],
                                         numParticles);
            }
        }
    }

    // GPU Render time doesn't include CPU->GPU copy time
    GraphicsTimerBegin();

    if (numDiffuse)
    {
        if (g_interop)
        {
            // copy data directly from solver to the renderer buffers
            UpdateDiffuseRenderBuffers(g_diffuseRenderBuffers, g_solver);
        }
        else
        {
            // copy diffuse particle data from host to GPU render device
            UpdateDiffuseRenderBuffers(g_diffuseRenderBuffers,
                                       &g_buffers->diffusePositions[0],
                                       &g_buffers->diffuseVelocities[0],
                                       numDiffuse);
        }
    }

    //---------------------------------------
    // setup view and state
    Matrix44 proj;
    Matrix44 view;
    float fov = kPi / 4.0f;
    float aspect = float(g_screenWidth) / g_screenHeight;

    if (eye >= 2 || !g_vrSystem)
    {
        view = GetCameraRotationMatrix(true) * TranslationMatrix(-Point3(g_camPos));
        proj = ProjectionMatrix(RadToDeg(fov), aspect, g_camNear, g_camFar);
    }
    else
    {
        Vec3 eyeOffset;
        // todo: calculate fov and aspect or fix parts that use it?? Need to check if something is broken with VR
        g_vrSystem->GetProjectionMatrixAndEyeOffset(eye, g_camNear, g_camFar, proj, eyeOffset);
        view = TranslationMatrix(-Point3(eyeOffset)) * g_vrSystem->GetHmdRotationInverse() * TranslationMatrix(-Point3(g_vrSystem->GetHmdPos()));
    }

    if (usedProj)
    {
        *usedProj = proj;
    }
    if (usedView)
    {
        *usedView = view;
    }

    //------------------------------------
    // lighting pass

    // expand scene bounds to fit most scenes
    g_sceneLower = Min(g_sceneLower, Vec3(-g_sceneMinBoxSize, 0.0f, -g_sceneMinBoxSize));
    g_sceneUpper = Max(g_sceneUpper, Vec3(g_sceneMinBoxSize, g_sceneMinBoxSize, g_sceneMinBoxSize));
	
    Vec3 sceneExtents = g_sceneUpper - g_sceneLower;
    Vec3 sceneCenter = 0.5f*(g_sceneUpper + g_sceneLower);

    g_lightPos = sceneCenter + g_lightDir*Length(sceneExtents)*g_lightDistance;
    g_lightTarget = sceneCenter;

    // calculate tight bounds for shadow frustum
    float lightFov = 2.0f*atanf(Length(g_sceneUpper - sceneCenter) / Length(g_lightPos - sceneCenter));

    // scale and clamp fov for aesthetics
    lightFov = Clamp(lightFov, DegToRad(25.0f), DegToRad(65.0f));

    Matrix44 lightPerspective = ProjectionMatrix(RadToDeg(lightFov), 1.0f, 1.0f, 1000.0f);
    Matrix44 lightView = LookAtMatrix(Point3(g_lightPos), Point3(g_lightTarget));
    Matrix44 lightTransform = lightPerspective*lightView;

    // radius used for drawing
    float radius = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f*g_pointScale;

    //-------------------------------------
    // shadowing pass

    if (g_meshSkinIndices.size())
    {
        SkinMesh();
    }

    // create shadow maps
    ShadowBegin(g_shadowMap);

    SetView(lightView, lightPerspective);
    SetCullMode(false);

    // give scene a chance to do custom drawing
    g_scene->Draw(1);

    if (g_drawMesh)
    {
        DrawMesh(g_mesh, RenderMaterial());
    }

    DrawStaticShapes();
    DrawRigidShapes();
    DrawRigidAttachments();
	if (g_drawCable) 
	{
		DrawCables();
	}

    if (g_drawCloth && g_buffers->triangles.size())
    {
        DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), RenderMaterial(), g_expandCloth);
    }

    if (g_drawRopes)
    {
        for (size_t i = 0; i < g_ropes.size(); ++i)
        {
            DrawRope(&g_buffers->positions[0], &g_ropes[i].mIndices[0], g_ropes[i].mIndices.size(), radius*g_ropeScale, RenderMaterial());
        }
    }

    int shadowParticles = numParticles;
    int shadowParticlesOffset = 0;

    if (!g_drawPoints)
    {
        shadowParticles = 0;

        if (g_drawEllipsoids)
        {
            shadowParticles = numParticles - g_numSolidParticles;
            shadowParticlesOffset = g_numSolidParticles;
        }
    }
    else
    {
        int offset = g_drawMesh ? g_numSolidParticles : 0;

        shadowParticles = numParticles - offset;
        shadowParticlesOffset = offset;
    }

    if (g_buffers->activeIndices.size())
    {
        DrawPoints(g_fluidRenderBuffers, shadowParticles, shadowParticlesOffset, radius, 2048, 1.0f, lightFov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
    }

    ShadowEnd();

    //----------------
    // lighting pass

    BindSolidShader(g_lightPos, g_lightTarget, lightTransform, g_shadowMap, 0.0f, Vec4(g_clearColor, g_fogDistance));

    SetView(view, proj);
    SetCullMode(true);
    SetFillMode(g_wireframe);

    // When the benchmark measures async compute, we need a graphics workload that runs for a whole frame.
    // We do this by rerendering our simple graphics many times.
    int passes = g_increaseGfxLoadForAsyncComputeTesting ? 50 : 1;

    for (int i = 0; i != passes; i++)
    {
        DrawPlanes((Vec4*)g_params.planes, g_params.numPlanes, g_drawPlaneBias);

        if (g_drawMesh)
        {
            DrawMesh(g_mesh, g_renderMaterials[0]);
        }

        DrawStaticShapes();
        DrawRigidShapes();
        DrawRigidAttachments();
		if (g_drawCable)
		{
			DrawCables();
		}

        if (g_drawCloth && g_buffers->triangles.size())
        {
            DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[1], g_expandCloth);
        }

        if (g_drawRopes)
        {
            for (size_t i = 0; i < g_ropes.size(); ++i)
            {
                DrawRope(&g_buffers->positions[0], &g_ropes[i].mIndices[0], g_ropes[i].mIndices.size(), g_params.radius*0.5f*g_ropeScale, g_renderMaterials[i%8]);
            }
        }

        // give scene a chance to do custom drawing
        g_scene->Draw(0);
    }

    UnbindSolidShader();

    // first pass of diffuse particles (behind fluid surface)
    if (g_drawDiffuse)
    {
		RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius*g_diffuseScale, float(g_screenWidth), aspect, fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow, false);
    }

    if (g_drawEllipsoids)
    {
        // draw solid particles separately
        if (g_numSolidParticles && g_drawPoints)
        {
            DrawPoints(g_fluidRenderBuffers, g_numSolidParticles, 0, radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
        }

        // render fluid surface
        RenderEllipsoids(g_fluidRenderer, g_fluidRenderBuffers, numParticles - g_numSolidParticles, g_numSolidParticles, radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_fluidColor, g_blur, g_ior, g_drawOpaque);

        // second pass of diffuse particles for particles in front of fluid surface
        if (g_drawDiffuse)
        {
			RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius*g_diffuseScale, float(g_screenWidth), aspect, fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow, true);
        }
    }
    else
    {
        // draw all particles as spheres
        if (g_drawPoints)
        {
            int offset = g_drawMesh ? g_numSolidParticles : 0;

            if (g_buffers->activeIndices.size())
            {
                DrawPoints(g_fluidRenderBuffers, numParticles - offset, offset, radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
            }
        }
    }

	//------------------------------
	// sensors pass

	DrawSensors(numParticles, numDiffuse, radius, lightTransform);

	// need to reset the view for picking
    SetView(view, proj);

	// end timing
    GraphicsTimerEnd();
}

void RenderDebug()
{
    if (g_mouseParticle != -1)
    {
        // draw mouse spring
        BeginLines();
        DrawLine(g_mousePos, Vec3(g_buffers->positions[g_mouseParticle]), Vec4(1.0f));
        EndLines();
    }

    if (g_mouseJoint != -1)
    {
        BeginLines();

        Transform bodyPose;
        NvFlexGetRigidPose(&g_buffers->rigidBodies[g_buffers->rigidJoints[g_mouseJoint].body1], (NvFlexRigidPose*)&bodyPose);

        DrawLine(g_mousePos, TransformPoint(bodyPose, g_buffers->rigidJoints[g_mouseJoint].pose1.p), Vec4(1.0f));
        EndLines();
    }

    // springs
    if (g_drawSprings)
    {
        Vec4 color;

        if (g_drawSprings == 1)
        {
            // stretch
            color = Vec4(0.0f, 0.0f, 1.0f, 0.8f);
        }
        if (g_drawSprings == 2)
        {
            // tether
            color = Vec4(0.0f, 1.0f, 0.0f, 0.8f);
        }

        BeginLines();

        int start = 0;

        for (int i = start; i < g_buffers->springLengths.size(); ++i)
        {
            if (g_drawSprings == 1 && g_buffers->springStiffness[i] < 0.0f)
            {
                continue;
            }
            if (g_drawSprings == 2 && g_buffers->springStiffness[i] > 0.0f)
            {
                continue;
            }

            int a = g_buffers->springIndices[i * 2];
            int b = g_buffers->springIndices[i * 2 + 1];

            DrawLine(Vec3(g_buffers->positions[a]), Vec3(g_buffers->positions[b]), color);
        }

        EndLines();
    }

    // visualize contacts against the environment
    if (g_drawContacts)
    {
        const int maxContactsPerParticle = 6;

        NvFlexVector<Vec4> contactPlanes(g_flexLib, g_buffers->positions.size()*maxContactsPerParticle);
        NvFlexVector<Vec4> contactVelocities(g_flexLib, g_buffers->positions.size()*maxContactsPerParticle);
        NvFlexVector<int> contactIndices(g_flexLib, g_buffers->positions.size());
        NvFlexVector<unsigned int> contactCounts(g_flexLib, g_buffers->positions.size());

        NvFlexGetContacts(g_solver, contactPlanes.buffer, contactVelocities.buffer, contactIndices.buffer, contactCounts.buffer);

        // ensure transfers have finished
        contactPlanes.map();
        contactVelocities.map();
        contactIndices.map();
        contactCounts.map();

        BeginLines();

        for (int i = 0; i < int(g_buffers->activeIndices.size()); ++i)
        {
            const int contactIndex = contactIndices[g_buffers->activeIndices[i]];
            const unsigned int count = contactCounts[contactIndex];

            const float scale = g_params.radius;

            for (unsigned int c = 0; c < count; ++c)
            {
                Vec4 plane = contactPlanes[contactIndex*maxContactsPerParticle + c];

                DrawLine(Vec3(g_buffers->positions[g_buffers->activeIndices[i]]),
                         Vec3(g_buffers->positions[g_buffers->activeIndices[i]]) + Vec3(plane)*scale,
                         Vec4(0.0f, 1.0f, 0.0f, 0.0f));
            }
        }

        EndLines();

        // rigid body contacts
        {
            NvFlexVector<NvFlexRigidContact> rigidContacts(g_flexLib, g_solverDesc.maxRigidBodyContacts);
            NvFlexVector<int> rigidContactCount(g_flexLib, 1);

            NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);

            rigidContacts.map();
            rigidContactCount.map();

            if (rigidContactCount[0] > g_solverDesc.maxRigidBodyContacts)
            {
                printf("Overflowed rigid body contacts: %d > %d\n", rigidContactCount[0], g_solverDesc.maxRigidBodyContacts);
                rigidContactCount[0] = g_solverDesc.maxRigidBodyContacts;
            }

            BeginLines();

           // printf("rigidContactCount: %d\n", rigidContactCount[0]);

            // convert contact multiplier to impulse (better for visualization)
            const float invDt = 1.0f/g_dt;

            for (int i=0; i < rigidContactCount[0]; ++i)
            {
                const NvFlexRigidContact& contact = rigidContacts[i];

                Vec3 a, b;

                if (contact.body0 != -1)
                {
                    const NvFlexRigidBody& body0 = g_buffers->rigidBodies[contact.body0];

                    Transform xform;
                    NvFlexGetRigidPose(&body0, (NvFlexRigidPose*)&xform);

                    a = TransformPoint(xform, contact.localPos0);
                }
                else
                {
                    a = contact.localPos0;
                }

                if (contact.body1 != -1)
                {
                    const NvFlexRigidBody& body1 = g_buffers->rigidBodies[contact.body1];

                    Transform xform;
                    NvFlexGetRigidPose(&body1, (NvFlexRigidPose*)&xform);

                    b = TransformPoint(xform, contact.localPos1);
                }
                else
                {
                    b = contact.localPos1;
                }

				// convert to impulse for better visualization
				const float scale = g_dt/g_numSubsteps;

                DrawLine(a, a + Vec3(contact.normal)*contact.lambda*scale, Vec4(0.0f, 0.0f, 1.0f, 1.0f));
                DrawLine(a, b, Vec4(0.0f, 1.0f, 0.0f, 1.0f));
            }

            EndLines();
        }

        {

            // rigid->soft contacts
            NvFlexVector<NvFlexSoftContact> rigidContacts(g_flexLib, g_solverDesc.maxRigidSoftContacts);
            NvFlexVector<int> rigidContactCount(g_flexLib, 1);

            NvFlexGetRigidSoftContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);

            rigidContacts.map();
            rigidContactCount.map();

            BeginLines();

            if (rigidContactCount[0] > g_solverDesc.maxRigidSoftContacts)
            {
                printf("Overflowed rigid soft body contacts: %d > %d\n", rigidContactCount[0], g_solverDesc.maxRigidSoftContacts);
                rigidContactCount[0] = g_solverDesc.maxRigidSoftContacts;
            }

            for (int i=0; i < rigidContactCount[0]; ++i)
            {
                const NvFlexSoftContact& contact = rigidContacts[i];

                Vec3 rigidPos;

                if (contact.bodyIndex != -1)
                {
                    const NvFlexRigidBody& body0 = g_buffers->rigidBodies[contact.bodyIndex];

                    Transform xform;
                    NvFlexGetRigidPose(&body0, (NvFlexRigidPose*)&xform);

                    rigidPos = TransformPoint(xform, contact.bodyOffset);
                }
                else
                {
                    rigidPos = contact.bodyOffset;
                }

                Vec3 a = Vec3(g_buffers->positions[contact.particleIndices[0]]);
                Vec3 b = Vec3(g_buffers->positions[contact.particleIndices[1]]);
                Vec3 c = Vec3(g_buffers->positions[contact.particleIndices[2]]);

                Vec3 triPos = Vec3(contact.particleBarys[0]*a + contact.particleBarys[1]*b + contact.particleBarys[2]*c);

                const float scale = g_dt/g_numSubsteps;

                //DrawLine(rigidPos, triPos, Vec4(1.0f, 1.0f, 0.0f, 1.0f));
                DrawLine(triPos, triPos + Vec3(contact.normal)*contact.lambda*scale, Vec4(0.0f, 0.5f, 1.0f, 1.0f));
            }

            EndLines();
        }
    }

    if (g_drawJoints)
    {
        DrawJoints();
    }

    if (g_drawBases)
    {
        for (int i = 0; i < int(g_buffers->shapeMatchingRotations.size()); ++i)
        {
            BeginLines();

            float size = 0.1f;

            Matrix33 frame(g_buffers->shapeMatchingRotations[i]);

            for (int b = 0; b < 3; ++b)
            {
                Vec3 color;
                color[b] = 1.0f;

                DrawLine(Vec3(g_buffers->shapeMatchingTranslations[i]),
                         Vec3(g_buffers->shapeMatchingTranslations[i] + frame.cols[b] * size),
                         Vec4(color, 0.0f));
            }

            EndLines();
        }
    }

    if (g_drawNormals)
    {
        NvFlexGetNormals(g_solver, g_buffers->normals.buffer, NULL);

        BeginLines();

        for (int i = 0; i < g_buffers->normals.size(); ++i)
        {
            DrawLine(Vec3(g_buffers->positions[i]),
                     Vec3(g_buffers->positions[i] - g_buffers->normals[i] * g_buffers->normals[i].w),
                     Vec4(0.0f, 1.0f, 0.0f, 0.0f));
        }

        EndLines();
    }

	if (g_drawSensors && g_renderSensors.size() > 0)
	{
		int y = 0;

		RenderSensor sensor = g_renderSensors[g_visSensorId];

		Transform cameraToWorld;

		if (sensor.parent)
		{
			Transform bodyToWorld;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[sensor.parent], (NvFlexRigidPose*)&bodyToWorld);

			cameraToWorld = bodyToWorld*sensor.origin;
		}
		else
		{
			cameraToWorld = sensor.origin;
		}

		// draw camera frame
		DrawBasis(Matrix33(cameraToWorld.q), cameraToWorld.p, false);

		// visualize depth
		SetDepthRenderProfile(sensor.depthProfile);

        int scaled_width = sensor.width * g_upscaling;
        int scaled_height = sensor.height * g_upscaling;
        
		DrawQuad(g_screenWidth - scaled_width, y, scaled_width, scaled_height, sensor.target, 1);
        y += scaled_height;
		SetDepthRenderProfile(defaultDepthProfile);

		// visualize rgb
        DrawQuad(g_screenWidth - scaled_width, y, scaled_width, scaled_height, sensor.target, 0);
        y += sensor.height*g_upscaling;	
	}
}

void DrawStaticShapes()
{
    RenderMaterial mat;
    mat.frontColor = Vec3(0.8f);
    mat.backColor = Vec3(0.8f);
	mat.specular = 0.0f;
	mat.roughness = 1.0f;

    for (int i = 0; i < g_buffers->shapeFlags.size(); ++i)
    {
        const int flags = g_buffers->shapeFlags[i];

        // unpack flags
        int type = int(flags&eNvFlexShapeFlagTypeMask);
        //bool dynamic = int(flags&eNvFlexShapeFlagDynamic) > 0;

        // render with prev positions to match particle update order
        // can also think of this as current/next
        const Quat rotation = g_buffers->shapePrevRotations[i];
        const Vec3 position = Vec3(g_buffers->shapePrevPositions[i]);

        NvFlexCollisionGeometry geo = g_buffers->shapeGeometry[i];

        if (type == eNvFlexShapeSphere)
        {
            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.sphere.radius);

            DrawRenderMesh(g_sphereMesh, xform, mat);
        }
        else if (type == eNvFlexShapeCapsule)
        {
            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation));

            DrawRenderMesh(g_sphereMesh, xform*TranslationMatrix(Point3(-geo.capsule.halfHeight, 0.0f, 0.0f))*ScaleMatrix(geo.capsule.radius), mat);
            DrawRenderMesh(g_sphereMesh, xform*TranslationMatrix(Point3( geo.capsule.halfHeight, 0.0f, 0.0f))*ScaleMatrix(geo.capsule.radius), mat);
            DrawRenderMesh(g_cylinderMesh, xform*RotationMatrix(DegToRad(-90.0f), Vec3(0.0f, 0.0f, 1.0f))*ScaleMatrix(Vec3(geo.capsule.radius, geo.capsule.halfHeight, geo.capsule.radius)), mat);
        }
        else if (type == eNvFlexShapeBox)
        {
            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(Vec3(geo.box.halfExtents)*2.0f);

            DrawRenderMesh(g_boxMesh, xform, mat);
        }
        else if (type == eNvFlexShapeConvexMesh)
        {
            if (g_convexes.find(geo.convexMesh.mesh) != g_convexes.end())
            {
                RenderMesh* m = g_convexes[geo.convexMesh.mesh];

                if (m)
                {
                    Matrix44 xform = TranslationMatrix(Point3(g_buffers->shapePositions[i]))*RotationMatrix(Quat(g_buffers->shapeRotations[i]))*ScaleMatrix(geo.convexMesh.scale);
                    DrawRenderMesh(m, xform, mat);
                }
            }
        }
        else if (type == eNvFlexShapeTriangleMesh)
        {
            if (g_meshes.find(geo.triMesh.mesh) != g_meshes.end())
            {
                RenderMesh* m = g_meshes[geo.triMesh.mesh];

                if (m)
                {
                    Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.triMesh.scale);
                    DrawRenderMesh(m, xform, mat);
                }
            }
        }
        else if (type == eNvFlexShapeSDF)
        {
            if (g_fields.find(geo.sdf.field) != g_fields.end())
            {
                RenderMesh* m = g_fields[geo.sdf.field];

                if (m)
                {
                    Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.sdf.scale);
                    DrawRenderMesh(m, xform, mat);
                }
            }
        }
    }
}

void DrawRigidAttachments()
{
    if (!g_drawVisual)
        return;

    // extract current frustum for culling
    Matrix44 view, projection;
    GetView(view, projection);

    Plane frustumPlanes[6];
    ExtractFrustumPlanes(projection*view, frustumPlanes);

    for (int i = 0; i < int(g_renderAttachments.size()); ++i)
    {
        const RenderAttachment& attach = g_renderAttachments[i];

        // body transform
        Vec3 bodyPosition;
        Quat bodyRotation;

        if (attach.parent != -1)
        {
            NvFlexRigidBody body = g_buffers->rigidBodies[attach.parent];

            NvFlexRigidPose bodyPose;
            NvFlexGetRigidPose(&body, &bodyPose);

            bodyPosition = bodyPose.p;
            bodyRotation = bodyPose.q;
        }

        // concatenate transforms
        Quat rotation = bodyRotation * attach.origin.q;
        Vec3 position = bodyPosition + bodyRotation * attach.origin.p;

        Vec3 localLower, localUpper;
        Vec3 worldLower, worldUpper;

		GetRenderMeshBounds(attach.mesh, &localLower, &localUpper);
		
        TransformBounds(localLower, localUpper, position, rotation, 1.0f, worldLower, worldUpper);

        Vec3 center = 0.5f*(worldLower + worldUpper);
        float radius = Length(worldUpper-center)*2.0f;	// todo: factor of two here shouldn't be necessary but seeing some overly agressive culling for shapes near camera

        if (!TestSphereAgainstFrustum(frustumPlanes, center, radius))
        {
            continue;
        }


		Matrix44 xform = TranslationMatrix(Point3(bodyPosition))*RotationMatrix(bodyRotation)*TransformMatrix(attach.origin);

		// draw attached meshes
        DrawRenderMesh(attach.mesh, xform, attach.material, attach.startTri, attach.endTri);
    }
}

void DrawRigidShapes(bool wire, int offset)
{
    // extract current frustum for culling
    Matrix44 view, projection;
    GetView(view, projection);

    Plane frustumPlanes[6];
    ExtractFrustumPlanes(projection*view, frustumPlanes);

    for (int i=0; i < g_buffers->rigidShapes.size() - offset; ++i)
    {
        // unpack flags
        NvFlexRigidShape shape = g_buffers->rigidShapes[i];
        int type = shape.geoType;

        Vec3 localLower, localUpper;
        GetGeometryBounds(shape.geo, type, localLower, localUpper);

        RenderMaterial material = g_renderMaterials[UnionCast<int>(shape.user)];

        // if hiding visual meshes then force render of all collision shapes
        if (!g_drawVisual)
        {
            if (material.hidden)
                material = g_renderMaterials[0];
        }

        if (wire)
        {
            material.frontColor = 0.0f;
            material.backColor = 0.0f;
        }

        // body transform
        Vec3 bodyPosition;
        Quat bodyRotation;

        if (shape.body != -1)
        {
            NvFlexRigidBody body = g_buffers->rigidBodies[shape.body];

            NvFlexRigidPose bodyPose;
            NvFlexGetRigidPose(&body, &bodyPose);

            bodyPosition = bodyPose.p;
            bodyRotation = bodyPose.q;
        }
        else
        {
            // static shape, override material
            if (shape.user == NULL)
            {
                material.frontColor = Vec3(0.1f, 0.7f, 0.3f);
                material.backColor = Vec3(0.1f, 0.7f, 0.3f);
            }
        }

        // shape transform
        const Quat shapeRotation(shape.pose.q);
        const Vec3 shapePosition(shape.pose.p);

        // concatenate transforms
        Quat rotation = bodyRotation*shapeRotation;
        Vec3 position = bodyPosition + bodyRotation*shapePosition;

        Vec3 worldLower, worldUpper;
        TransformBounds(localLower, localUpper, position, rotation, 1.0f, worldLower, worldUpper);

        Vec3 center = 0.5f*(worldLower + worldUpper);
        float radius = Length(worldUpper-center)*2.0f;	// todo: factor of two here shouldn't be necessary but seeing some overly agressive culling for shapes near camera

        if (!TestSphereAgainstFrustum(frustumPlanes, center, radius))
        {
            continue;
        }

        NvFlexCollisionGeometry geo = shape.geo;

        if (type == eNvFlexShapeSphere)
        {
            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.sphere.radius);

            DrawRenderMesh(g_sphereMesh, xform, material);
        }
        else if (type == eNvFlexShapeCapsule)
        {
            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation));

            DrawRenderMesh(g_sphereMesh, xform*TranslationMatrix(Point3(-geo.capsule.halfHeight, 0.0f, 0.0f))*ScaleMatrix(geo.capsule.radius), material);
            DrawRenderMesh(g_sphereMesh, xform*TranslationMatrix(Point3( geo.capsule.halfHeight, 0.0f, 0.0f))*ScaleMatrix(geo.capsule.radius), material);
            DrawRenderMesh(g_cylinderMesh, xform*RotationMatrix(DegToRad(-90.0f), Vec3(0.0f, 0.0f, 1.0f))*ScaleMatrix(Vec3(geo.capsule.radius, geo.capsule.halfHeight, geo.capsule.radius)), material);
        }
        else if (type == eNvFlexShapeBox)
        {
            Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(Vec3(geo.box.halfExtents)*2.0f + Vec3(shape.thickness)*2.0f);

            DrawRenderMesh(g_boxMesh, xform, material);
        }
        else if (type == eNvFlexShapeConvexMesh)
        {
            if (g_convexes.find(geo.convexMesh.mesh) != g_convexes.end())
            {
                RenderMesh* m = g_convexes[geo.convexMesh.mesh];

                if (m)
                {
                    Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(Vec3(geo.convexMesh.scale));
                    DrawRenderMesh(m, xform, material);
                }
            }
        }
        else if (type == eNvFlexShapeTriangleMesh)
        {
            if (g_meshes.find(geo.triMesh.mesh) != g_meshes.end())
            {
                RenderMesh* m = g_meshes[geo.triMesh.mesh];

                if (1)
                {
                    if (m)
                    {
                        Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.triMesh.scale);
                        DrawRenderMesh(m, xform, material);
                    }
                }
                else
                {
                    // edge highlight mode, draws assigned convex edges
                    Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.triMesh.scale);

                    int numVertices, numTriangles;

                    // query size
                    NvFlexGetTriangleMesh(g_flexLib, geo.triMesh.mesh, 0,0,0, &numVertices, &numTriangles, 0, 0);

                    NvFlexVector<Vec3> positions(g_flexLib, numVertices);
                    NvFlexVector<int> features(g_flexLib, numTriangles);
                    NvFlexVector<int> indices(g_flexLib, numTriangles*3);

                    // get geometry
                    NvFlexGetTriangleMesh(g_flexLib, geo.triMesh.mesh, positions.buffer, indices.buffer, features.buffer, &numVertices, &numTriangles, 0, 0);

                    positions.map();
                    features.map();
                    indices.map();

                    BeginLines();

                    // draw assigned edges
                    for (int i=0; i < numTriangles; i++)
                    {
                        Vec3 a = Vec3(xform*Point3(positions[indices[i*3+0]]));
                        Vec3 b = Vec3(xform*Point3(positions[indices[i*3+1]]));
                        Vec3 c = Vec3(xform*Point3(positions[indices[i*3+2]]));

                        int mask = features[i];

                        if (mask&eNvFlexTriFeatureEdge0)
                        {
                            DrawLine(a, b, Vec4(1.0f));
                        }
                        if (mask&eNvFlexTriFeatureEdge1)
                        {
                            DrawLine(b, c, Vec4(1.0f));
                        }
                        if (mask&eNvFlexTriFeatureEdge2)
                        {
                            DrawLine(c, a, Vec4(1.0f));
                        }
                    }

                    EndLines();

                    positions.unmap();
                    features.unmap();
                    indices.unmap();
                }

            }
        }
        else if (type == eNvFlexShapeSDF)
        {
            if (g_fields.find(geo.sdf.field) != g_fields.end())
            {
                RenderMesh* m = g_fields[geo.sdf.field];

                if (m)
                {
                    Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.sdf.scale);
                    DrawRenderMesh(m, xform, material);
                }
            }
        }
    }
}

void DrawCables() 
{
	static vector<Vec3> points;
	if (g_cableDirty) 
	{
		g_cableDirty = false;
		points.clear();
		for (int i = 0; i < g_buffers->cableLinks.size(); i++) 
		{
			NvFlexCableLink& link = g_buffers->cableLinks[i];
			if (link.nextLink >= 0) 
			{
				NvFlexCableLink& next = g_buffers->cableLinks[link.nextLink];
				Transform trans;
				Transform ntrans;
				if (link.body >= 0) 
				{
					NvFlexGetRigidPose(&g_buffers->rigidBodies[link.body], (NvFlexRigidPose*)&trans);
				}
				if (next.body >= 0) 
				{
					NvFlexGetRigidPose(&g_buffers->rigidBodies[next.body], (NvFlexRigidPose*)&ntrans);
				}

				Vec3 pos = TransformPoint(trans, link.att1);
				Vec3 npos = TransformPoint(ntrans, next.att0);
				points.push_back(pos);
				points.push_back(npos);
			}
		}
	}
	if (points.size() > 0) 
	{
		DrawLines(&points[0], points.size(), Vec4(1.0f, 0.0f, 0.0f, 1.0f));
	}
	/*
	BeginLines(true);
	for (int i = 0; i < g_buffers->cableLinks.size(); i++) {
		NvFlexCableLink& link = g_buffers->cableLinks[i];
		if (link.nextLink >= 0) {
			NvFlexCableLink& next = g_buffers->cableLinks[link.nextLink];
			Transform trans;
			Transform ntrans;
			if (link.body >= 0) {
				NvFlexGetRigidPose(&g_buffers->rigidBodies[link.body], (NvFlexRigidPose*)&trans);
			}
			if (next.body >= 0) {
				NvFlexGetRigidPose(&g_buffers->rigidBodies[next.body], (NvFlexRigidPose*)&ntrans);
			}

			Vec3 pos = TransformPoint(trans, link.att1);
			Vec3 npos = TransformPoint(ntrans, next.att0);
			DrawLine(pos, npos, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
		}
	}
	EndLines();
	*/
}

void DrawBasis(const Matrix33& frame, const Vec3& p, bool depthTest)
{
    BeginLines(10.0, depthTest);

    const float size = 0.05f;

    for (int b = 0; b < 3; ++b)
    {
        Vec3 color;
        color[b] = 1.0f;

        DrawLine(p, p + frame.cols[b] * size, Vec4(color, 0.0f));
    }

    EndLines();
}

void DrawSensors(const int numParticles, const int numDiffuse, float radius, Matrix44 lightTransform)
{
	for (int i = 0; i < g_renderSensors.size(); i++)
	{
		RenderSensor sensor = g_renderSensors[i];

		// Depth settings
		SetDepthRenderProfile(sensor.depthProfile);

		BindSolidShader(g_lightPos, g_lightTarget, lightTransform, g_shadowMap, 0.f, Vec4(g_clearColor, g_fogDistance));

		SetCullMode(true);
		SetFillMode(false);

		Transform cameraToWorld;

		if (sensor.parent)
		{
			Transform bodyToWorld;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[sensor.parent], (NvFlexRigidPose*)&bodyToWorld);
			
			cameraToWorld = bodyToWorld * sensor.origin;
		}
		else
		{
			cameraToWorld = sensor.origin;
		}

		// convert from URDF (positive z-forward) to OpenGL (negative z-forward)
		Matrix44 conversion = RotationMatrix(kPi, Vec3(1.0f, 0.0f, 0.0f));

		Matrix44 view = AffineInverse(TransformMatrix(cameraToWorld) * conversion);
		Matrix44 proj = ProjectionMatrix(RadToDeg(sensor.fov), 1.0f, g_camNear, g_camFar);

		SetRenderTarget(sensor.target, 0, 0, sensor.width, sensor.height);	
		SetView(view, proj);	

		// draw all rigid attachments, todo: call into the main scene render or just render what we need selectively?		
		DrawRigidAttachments();
		DrawRigidShapes();
		DrawStaticShapes();
		DrawPlanes((Vec4*)g_params.planes, g_params.numPlanes, g_drawPlaneBias);

		if (g_drawMesh)
		{
			DrawMesh(g_mesh, g_renderMaterials[0]);
		}

		if (g_drawCloth && g_buffers->triangles.size())
		{
			DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL, 
					&g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), g_renderMaterials[1], 
					g_expandCloth);
		}

		if (g_drawRopes)
		{
			for (size_t i = 0; i < g_ropes.size(); ++i)
			{
				DrawRope(&g_buffers->positions[0], &g_ropes[i].mIndices[0], g_ropes[i].mIndices.size(), 
					g_params.radius * 0.5f * g_ropeScale, g_renderMaterials[i % 8]);
			}
		}

		// give scene a chance to do custom drawing
		g_scene->Draw(0);

		// draw fluids
		if (sensor.fluidRendererId != -1) {
			float fov = sensor.fov;
			float aspect = float(sensor.width) / sensor.height;
			FluidRenderer* fluidRenderer = g_sensorFluidRenderers[sensor.fluidRendererId];

			// TODO(jaliang): this might not be working
			if (g_drawDiffuse)
			{
				RenderDiffuse(fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius * g_diffuseScale, float(sensor.width), aspect, fov,
					g_diffuseColor, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter,
					g_diffuseOutscatter, g_diffuseShadow, false);
			}

			if (g_drawEllipsoids)
			{
				// draw solid particles separately
				if (g_numSolidParticles && g_drawPoints)
				{
					DrawPoints(g_fluidRenderBuffers, g_numSolidParticles, 0, radius, float(sensor.width), aspect, fov,
						g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
				}

				// render fluid surface
				RenderEllipsoids(fluidRenderer, g_fluidRenderBuffers, numParticles - g_numSolidParticles, g_numSolidParticles, radius,
					float(sensor.width), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_fluidColor,
					g_blur, g_ior, g_drawOpaque, sensor.target);

				// second pass of diffuse particles for particles in front of fluid surface
				if (g_drawDiffuse)
				{
					RenderDiffuse(fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius * g_diffuseScale, float(sensor.width), aspect, fov,
						g_diffuseColor, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter,
						g_diffuseOutscatter, g_diffuseShadow, true);
				}
			}
			else
			{
				// draw all particles as spheres
				if (g_drawPoints)
				{
					int offset = g_drawMesh ? g_numSolidParticles : 0;

					if (g_buffers->activeIndices.size())
					{
						DrawPoints(g_fluidRenderBuffers, numParticles - offset, offset, radius, float(sensor.width), aspect, fov,
							g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
					}
				}
			}
		}
		
		SetDepthRenderProfile(defaultDepthProfile);

		UnbindSolidShader();
	}

	SetRenderTarget(0, 0, 0, 0, 0);
}

void DrawJoints()
{
    for (int i = 0; i < g_buffers->rigidJoints.size(); ++i)
    {
        NvFlexRigidJoint joint = g_buffers->rigidJoints[i];

        Transform body0;
        Transform body1;

        Transform joint0;
        Transform joint1;


        if (joint.body0 != -1)
        {
            const Quat jointRotation(joint.pose0.q);
            const Vec3 jointPosition(joint.pose0.p);

            // body transform
            NvFlexGetRigidPose(&g_buffers->rigidBodies[joint.body0], (NvFlexRigidPose*)&body0);

            joint0 = body0*Transform(joint.pose0.p, joint.pose0.q);;
        }
        else
        {
            joint0 = Transform(joint.pose0.p, joint.pose0.q);
        }

        if (joint.body1 != -1)
        {
            const Quat jointRotation(joint.pose1.q);
            const Vec3 jointPosition(joint.pose1.p);

            // body transform
            NvFlexGetRigidPose(&g_buffers->rigidBodies[joint.body1], (NvFlexRigidPose*)&body1);

            joint1 = body1*Transform(joint.pose1.p, joint.pose1.q);
        }
        else
        {
            joint1 = Transform(joint.pose1.p, joint.pose1.q);
        }

        DrawBasis(Matrix33(joint0.q), joint0.p, true);
        DrawBasis(Matrix33(joint1.q), joint1.p, true);

        BeginLines(true);

        // link from body0 origin to joint0
        if (joint.body0 != -1)
        {
            DrawLine(joint0.p, body0.p, Vec4(1.0f, 1.0f, 0.0f, 1.0f));
        }

        // link from body1 origin to joint1
        if (joint.body1 != -1)
        {
            DrawLine(joint1.p, body1.p, Vec4(1.0f, 1.0f, 0.0f, 1.0f));
        }

        // link between joints
        DrawLine(joint0.p, joint1.p, Vec4(0.0f, 1.0f, 0.0f, 1.0f));

        EndLines();
    }

}

// returns the new scene if one is selected
int DoUI()
{
	// gui may set a new scene
	int newScene = -1;

	if (!g_showHelp) 
	{
		int x = g_screenWidth - 200;
		int y = g_screenHeight - 23;

		// imgui
		unsigned char button = 0;
		if (g_lastb == SDL_BUTTON_LEFT)
		{
			button = IMGUI_MBUT_LEFT;
		}
		else if (g_lastb == SDL_BUTTON_RIGHT)
		{
			button = IMGUI_MBUT_RIGHT;
		}

		imguiBeginFrame(g_lastx, g_screenHeight - g_lasty, button, 0);
		g_scene->DoStats();
		imguiEndFrame();

		// kick render commands
		DrawImguiGraph();
	} else
    {
        const int numParticles = NvFlexGetActiveCount(g_solver);
        const int numDiffuse = g_buffers->diffuseCount[0];

        int x = g_screenWidth - 200;
        int y = g_screenHeight - 23;

        // imgui
        unsigned char button = 0;
        if (g_lastb == SDL_BUTTON_LEFT)
        {
            button = IMGUI_MBUT_LEFT;
        }
        else if (g_lastb == SDL_BUTTON_RIGHT)
        {
            button = IMGUI_MBUT_RIGHT;
        }

        imguiBeginFrame(g_lastx, g_screenHeight - g_lasty, button, 0);

        x += 180;

        int fontHeight = 13;

        if (1)
        {
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Frame: %d", g_frame);
            y -= fontHeight * 2;

            if (!g_ffmpeg)
            {
                DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Frame Time: %.2fms", g_realdt*1000.0f);
                y -= fontHeight * 2;

                // If detailed profiling is enabled, then these timers will contain the overhead of the detail timers, so we won't display them.
                if (!g_profile)
                {
                    DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Sim Time (CPU): %.2fms", g_updateTime*1000.0f);
                    y -= fontHeight;
                    DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Render Time (CPU): %.2fms", g_renderTime*1000.0f);
                    y -= fontHeight;
                    DrawImguiString(x, y, Vec3(0.97f, 0.59f, 0.27f), IMGUI_ALIGN_RIGHT, "Sim Latency (GPU): %.2fms", g_simLatency);
                    y -= fontHeight * 2;

                    BenchmarkUpdateGraph();
                }
                else
                {
                    y -= fontHeight * 3;
                }
            }

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Particle Count: %d", numParticles);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Diffuse Count: %d", numDiffuse);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Shape Match Count: %d", g_buffers->shapeMatchingOffsets.size() > 0 ? g_buffers->shapeMatchingOffsets.size() - 1 : 0);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Rigid Body Count: %d", g_buffers->rigidBodies.size());
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Rigid Shape Count: %d", g_buffers->rigidShapes.size());
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Rigid Joint Count: %d", g_buffers->rigidJoints.size());
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Spring Count: %d", g_buffers->springLengths.size());
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Tetra Count: %d", g_buffers->tetraRestPoses.size());
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Num Substeps: %d", g_numSubsteps);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Num Iterations: %d", g_params.numIterations);
            y -= fontHeight * 2;

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Device: %s", g_deviceName);
            y -= fontHeight * 2;
        }

        if (g_profile)
        {
            DrawImguiString(x, y, Vec3(0.97f, 0.59f, 0.27f), IMGUI_ALIGN_RIGHT, "Total GPU Sim Latency: %.2fms", g_timers.total);
            y -= fontHeight * 2;

            DrawImguiString(x, y, Vec3(0.0f, 1.0f, 0.0f), IMGUI_ALIGN_RIGHT, "GPU Latencies");
            y -= fontHeight;

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Predict: %.2fms", g_timers.predict);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Create Cell Indices: %.2fms", g_timers.createCellIndices);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Sort Cell Indices: %.2fms", g_timers.sortCellIndices);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Reorder: %.2fms", g_timers.reorder);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "CreateGrid: %.2fms", g_timers.createGrid);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Particles: %.2fms", g_timers.collideParticles);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Shapes: %.2fms", g_timers.collideShapes);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Triangles: %.2fms", g_timers.collideTriangles);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Calculate Density: %.2fms", g_timers.calculateDensity);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Densities: %.2fms", g_timers.solveDensities);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Velocities: %.2fms", g_timers.solveVelocities);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Rigids: %.2fms", g_timers.solveShapes);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Springs: %.2fms", g_timers.solveSprings);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Tetra: %.2fms", g_timers.solveTetra);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Inflatables: %.2fms", g_timers.solveInflatables);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Contacts: %.2fms", g_timers.solveContacts);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Apply Deltas: %.2fms", g_timers.applyDeltas);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Finalize: %.2fms", g_timers.finalize);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Triangles: %.2fms", g_timers.updateTriangles);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Normals: %.2fms", g_timers.updateNormals);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Bounds: %.2fms", g_timers.updateBounds);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Calculate Anisotropy: %.2fms", g_timers.calculateAnisotropy);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Diffuse: %.2fms", g_timers.updateDiffuse);
            
            y -= fontHeight * 2;            


            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Build Global: %.2f / %.2f ms", g_timersAvg.buildGlobal, sqrtf(g_timersVar.buildGlobal/g_timersCount));
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Evaluate Global: %.2f / %.2fms", g_timersAvg.evaluateGlobal, sqrtf(g_timersVar.evaluateGlobal/g_timersCount));
            y -= fontHeight;            
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Global: %.2f / %.2f ms", g_timersAvg.solveGlobal, sqrtf(g_timersVar.solveGlobal/g_timersCount));
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Global: %.2f / %.2f ms", g_timersAvg.updateGlobal, sqrtf(g_timersVar.updateGlobal/g_timersCount));
            y -= fontHeight * 2;            
        }

        g_scene->DoStats();

        x -= 180;

        int uiOffset = 250;
        int uiBorder = 20;
        int uiWidth = 200;
        int uiHeight = g_screenHeight - uiOffset - uiBorder * 3;
        int uiLeft = uiBorder;

        if (g_tweakPanel)
        {
            imguiBeginScrollArea("Scene", uiLeft, g_screenHeight - uiBorder - uiOffset, uiWidth, uiOffset, &g_sceneScroll);
        }
        else
        {
            imguiBeginScrollArea("Scene", uiLeft, uiBorder, uiWidth, g_screenHeight - uiBorder - uiBorder, &g_sceneScroll);
        }

        for (int i = 0; i < int(g_sceneFactories.size()); ++i)
        {
            unsigned int color = g_sceneIndex == i ? imguiRGBA(255, 151, 61, 255) : imguiRGBA(255, 255, 255, 200);
            if (imguiItem(g_sceneFactories[i].mName, true, color)) // , i == g_selectedScene))
            {
                newScene = i;
            }
        }

        imguiEndScrollArea();

        if (g_tweakPanel)
        {
            static int scroll = 0;

            imguiBeginScrollArea("Options", uiLeft, g_screenHeight - uiBorder - uiHeight - uiOffset - uiBorder, uiWidth, uiHeight, &scroll);
            imguiSeparatorLine();

            // global options
            imguiLabel("Global");
            if (imguiCheck("Emit particles", g_emit))
            {
                g_emit = !g_emit;
            }

            if (imguiCheck("Pause", g_pause))
            {
                g_pause = !g_pause;
            }

            imguiSeparatorLine();

            if (imguiCheck("Wireframe", g_wireframe))
            {
                g_wireframe = !g_wireframe;
            }

            if (imguiCheck("Draw Points", g_drawPoints))
            {
                g_drawPoints = !g_drawPoints;
            }

            if (imguiCheck("Draw Fluid", g_drawEllipsoids))
            {
                g_drawEllipsoids = !g_drawEllipsoids;
            }

            if (imguiCheck("Draw Mesh", g_drawMesh))
            {
                g_drawMesh = !g_drawMesh;
                g_drawRopes = !g_drawRopes;
            }

            if (imguiCheck("Draw Basis", g_drawBases))
            {
                g_drawBases = !g_drawBases;
            }

            if (imguiCheck("Draw Springs", bool(g_drawSprings != 0)))
            {
                g_drawSprings = (g_drawSprings) ? 0 : 1;
            }

            if (imguiCheck("Draw Contacts", g_drawContacts))
            {
                g_drawContacts = !g_drawContacts;
            }

            if (imguiCheck("Draw Joints", g_drawJoints))
            {
                g_drawJoints = !g_drawJoints;
            }

            if (imguiCheck("Draw Visual", g_drawVisual))
            {
                g_drawVisual = !g_drawVisual;
            }            

			if (imguiCheck("Draw Sensors", g_drawSensors))
			{
				g_drawSensors = !g_drawSensors;
			}

            imguiSeparatorLine();

            bool f = g_params.frictionMode == eNvFlexFrictionModeSingle;            
            if (imguiCheck("Friction Single", f))
                g_params.frictionMode = eNvFlexFrictionModeSingle;

            f = g_params.frictionMode == eNvFlexFrictionModeDouble;
            if (imguiCheck("Friction Double", f))
                g_params.frictionMode = eNvFlexFrictionModeDouble;

            f = g_params.frictionMode == eNvFlexFrictionModeFull;
            if (imguiCheck("Friction Full", f))
                g_params.frictionMode = eNvFlexFrictionModeFull;


            imguiSeparatorLine();

            // scene options
			if (g_renderSensors.size())
			{
				float visSensorId = float(g_visSensorId);
				if (imguiSlider("Draw Sensor Id", &visSensorId, 0, float(g_renderSensors.size() - 1), 1))
				{
					g_visSensorId = int(visSensorId);
				}
			}

            g_scene->DoGui();

            if (imguiButton("Reset Scene"))
            {
                g_resetScene = true;
            }

            imguiSeparatorLine();

            bool b;

            b= g_params.solverType == eNvFlexSolverPBD;
            if (imguiCheck("PBD", b))
            {
                g_params.solverType = eNvFlexSolverPBD;
				g_numSubsteps = 4;
				g_params.numIterations = 30;
            }

            b= g_params.solverType == eNvFlexSolverJacobi;
            if (imguiCheck("Jacobi", b))
            {
                g_params.solverType = eNvFlexSolverJacobi;
				g_numSubsteps = 2;
				g_params.numIterations = 6;
				g_params.numInnerIterations = 20;
				g_params.relaxationFactor = 0.75f;
            }

            b= g_params.solverType == eNvFlexSolverGaussSeidel;
            if (imguiCheck("Gauss-Seidel", b))
            {
                g_params.solverType = eNvFlexSolverGaussSeidel;
                g_numSubsteps = 2;
                g_params.numIterations = 6;
                g_params.numInnerIterations = 20;
                g_params.relaxationFactor = 0.75f;
            }

            b= g_params.solverType == eNvFlexSolverLDLT;
            if (imguiCheck("LDLT", b))
            {
                g_params.solverType = eNvFlexSolverLDLT;
				g_numSubsteps = 2;
				g_params.numIterations = 4;
				g_params.relaxationFactor = 0.75f;
            }

            b= g_params.solverType == eNvFlexSolverPCG1;
            if (imguiCheck("PCG (CPU)", b))
            {
                g_params.solverType = eNvFlexSolverPCG1;
				g_numSubsteps = 2;
				g_params.numIterations = 6;
				g_params.numInnerIterations = 20;
				g_params.relaxationFactor = 0.75f;
            }

            b= g_params.solverType == eNvFlexSolverPCG2;
            if (imguiCheck("PCG (GPU)", b))
            {
                g_params.solverType = eNvFlexSolverPCG2;
				g_numSubsteps = 2;
				g_params.numIterations = 6;
				g_params.numInnerIterations = 20;
				g_params.relaxationFactor = 0.75f;
            }

            b= g_params.solverType == eNvFlexSolverPCR;
            if (imguiCheck("PCR", b))
            {
                g_params.solverType = eNvFlexSolverPCR;
				g_numSubsteps = 2;
				g_params.numIterations = 6;
				g_params.numInnerIterations = 20;
				g_params.relaxationFactor = 0.75f;
            }

			/*
            if (imguiCheck("Substep Mode", g_params.substepMode))
                g_params.substepMode = !g_params.substepMode;


            float scalingMode = float(g_params.scalingMode);
            if (imguiSlider("Scaling Mode", &scalingMode, 0, 3, 1))
            {
                g_params.scalingMode = int(scalingMode);
                printf("scaling mode: %d\n", g_params.scalingMode);                
            }
			*/

            imguiSeparatorLine();

            float n = float(g_numSubsteps);
            if (imguiSlider("Num Substeps", &n, 1, 20, 1))
            {
                g_numSubsteps = int(n);
            }

            n = float(g_params.numIterations);
            if (imguiSlider("Num Outer Iterations", &n, 1, 500, 1))
            {
                g_params.numIterations = int(n);
            }

			if (g_params.solverType != eNvFlexSolverPBD)
			{

				if (g_params.solverType != eNvFlexSolverLDLT)
				{
					n = float(g_params.numInnerIterations);
					if (imguiSlider("Num Inner Iterations", &n, 1, 500, 1))
					{
						g_params.numInnerIterations = int(n);
					}
				}

				n = float(g_params.numLineIterations);
				if (imguiSlider("Num Line Iterations", &n, 0, 50, 1))
				{
					g_params.numLineIterations = int(n);
				}

                imguiSlider("Warm Start", &g_params.warmStart, 0.0f, 1.0f, 0.00001f);

				float contactRegularization = log10f(g_params.contactRegularization);
				if (imguiSlider("Contact Reg. (Log)", &contactRegularization, -10, 0, 0.001f))
				{
					g_params.contactRegularization = powf(10.0f, contactRegularization);
				}
            
                float systemRegularization = log10f(g_params.systemRegularization);
                if (imguiSlider("System Reg. (Log)", &systemRegularization, -10, 3, 0.001f))
                {
                    g_params.systemRegularization = powf(10.0f, systemRegularization);
                }

				float systemTolerance = log10f(g_params.systemTolerance);
				if (imguiSlider("System Tol. (Log)", &systemTolerance, -10, 0, 0.001f))
				{
					g_params.systemTolerance = powf(10.0f, systemTolerance);
				}

                imguiSlider("System Beta", &g_params.systemBeta, 0, 1, 0.001f);
			}

            imguiSeparatorLine();
            imguiSlider("Gravity X", &g_params.gravity[0], -50.0f, 50.0f, 0.1f);
            imguiSlider("Gravity Y", &g_params.gravity[1], -50.0f, 50.0f, 0.1f);
            imguiSlider("Gravity Z", &g_params.gravity[2], -50.0f, 50.0f, 0.1f);

            imguiSeparatorLine();
            imguiSlider("Radius", &g_params.radius, 0.01f, 0.5f, 0.01f);
            imguiSlider("Solid Radius", &g_params.solidRestDistance, 0.0f, 0.5f, 0.001f);
            imguiSlider("Fluid Radius", &g_params.fluidRestDistance, 0.0f, 0.5f, 0.001f);

			imguiSeparatorLine();
			imguiSlider("SOR", &g_params.relaxationFactor, 0.0f, 5.0f, 0.01f);
			imguiSlider("Geometric Stiffness", &g_params.geometricStiffness, 0.0f, 1.0f, 0.001f);

            // common params
            imguiSeparatorLine();
            imguiSlider("Dynamic Friction", &g_params.dynamicFriction, 0.0f, 2.0f, 0.01f);
            imguiSlider("Static Friction", &g_params.staticFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Particle Friction", &g_params.particleFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Restitution", &g_params.restitution, 0.0f, 1.0f, 0.01f);
            imguiSlider("SleepThreshold", &g_params.sleepThreshold, 0.0f, 1.0f, 0.01f);
            imguiSlider("Shock Propagation", &g_params.shockPropagation, 0.0f, 10.0f, 0.01f);
            imguiSlider("Damping", &g_params.damping, 0.0f, 10.0f, 0.01f);
            imguiSlider("Dissipation", &g_params.dissipation, 0.0f, 0.01f, 0.0001f);

            imguiSlider("Collision Distance", &g_params.collisionDistance, 0.0f, 0.5f, 0.001f);
            imguiSlider("Collision Margin", &g_params.shapeCollisionMargin, 0.0f, 5.0f, 0.01f);

            // cloth params
            imguiSeparatorLine();
            imguiSlider("Wind", &g_windStrength, -1.0f, 1.0f, 0.01f);
            imguiSlider("Drag", &g_params.drag, 0.0f, 1.0f, 0.01f);
            imguiSlider("Lift", &g_params.lift, 0.0f, 1.0f, 0.01f);
            imguiSeparatorLine();

            // fluid params
            imguiSlider("Adhesion", &g_params.adhesion, 0.0f, 10.0f, 0.01f);
            imguiSlider("Cohesion", &g_params.cohesion, 0.0f, 0.2f, 0.0001f);
            imguiSlider("Surface Tension", &g_params.surfaceTension, 0.0f, 50.0f, 0.01f);
            imguiSlider("Viscosity", &g_params.viscosity, 0.0f, 120.0f, 0.01f);
            imguiSlider("Vorticicty Confinement", &g_params.vorticityConfinement, 0.0f, 120.0f, 0.1f);
            imguiSlider("Solid Pressure", &g_params.solidPressure, 0.0f, 1.0f, 0.01f);
            imguiSlider("Surface Drag", &g_params.freeSurfaceDrag, 0.0f, 1.0f, 0.01f);
            imguiSlider("Buoyancy", &g_params.buoyancy, -1.0f, 1.0f, 0.01f);

            imguiSeparatorLine();
            imguiSlider("Anisotropy Scale", &g_params.anisotropyScale, 0.0f, 30.0f, 0.01f);
            imguiSlider("Smoothing", &g_params.smoothing, 0.0f, 1.0f, 0.01f);

            // diffuse params
            imguiSeparatorLine();
            imguiSlider("Diffuse Threshold", &g_params.diffuseThreshold, 0.0f, 1000.0f, 1.0f);
            imguiSlider("Diffuse Buoyancy", &g_params.diffuseBuoyancy, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Drag", &g_params.diffuseDrag, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Scale", &g_diffuseScale, 0.0f, 1.5f, 0.01f);
            imguiSlider("Diffuse Alpha", &g_diffuseColor.w, 0.0f, 3.0f, 0.01f);
            imguiSlider("Diffuse Inscatter", &g_diffuseInscatter, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Outscatter", &g_diffuseOutscatter, 0.0f, 2.0f, 0.01f);
            imguiSlider("Diffuse Motion Blur", &g_diffuseMotionScale, 0.0f, 5.0f, 0.1f);

            n = float(g_params.diffuseBallistic);
            if (imguiSlider("Diffuse Ballistic", &n, 1, 40, 1))
            {
                g_params.diffuseBallistic = int(n);
            }

            imguiEndScrollArea();
        }
        imguiEndFrame();

        // kick render commands
        DrawImguiGraph();
    }

    return newScene;
}




void UpdateControllerInput()
{
	static double lastTime;

	// real elapsed frame time
	double frameBeginTime = GetSeconds();

	g_realdt = float(frameBeginTime - lastTime);
	lastTime = frameBeginTime;

	// do gamepad input polling
	double currentTime = frameBeginTime;
	static double lastJoyTime = currentTime;

	if (g_gamecontroller && currentTime - lastJoyTime > g_dt)
	{
		lastJoyTime = currentTime;

		int leftStickX = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_LEFTX);
		int leftStickY = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_LEFTY);
		int rightStickX = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_RIGHTX);
		int rightStickY = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_RIGHTY);
		int leftTrigger = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_TRIGGERLEFT);
		int rightTrigger = SDL_GameControllerGetAxis(g_gamecontroller, SDL_CONTROLLER_AXIS_TRIGGERRIGHT);

		Vec2 leftStick(joyAxisFilter(leftStickX, 0), joyAxisFilter(leftStickY, 0));
		Vec2 rightStick(joyAxisFilter(rightStickX, 1), joyAxisFilter(rightStickY, 1));
		Vec2 trigger(leftTrigger / 32768.0f, rightTrigger / 32768.0f);

		if (leftStick.x != 0.0f || leftStick.y != 0.0f ||
			rightStick.x != 0.0f || rightStick.y != 0.0f)
		{
			// note constant factor to speed up analog control compared to digital because it is more controllable.
			g_camVel.z = -4 * g_camSpeed * leftStick.y;
			g_camVel.x = 4 * g_camSpeed * leftStick.x;

			// cam orientation
			g_camAngle.x -= rightStick.x * 0.05f;
			g_camAngle.y -= rightStick.y * 0.05f;
		}

		// Handle left stick motion
		static bool bLeftStick = false;

		if ((leftStick.x != 0.0f || leftStick.y != 0.0f) && !bLeftStick)
		{
			bLeftStick = true;
		}
		else if ((leftStick.x == 0.0f && leftStick.y == 0.0f) && bLeftStick)
		{
			bLeftStick = false;
			g_camVel.z = -4.f * g_camSpeed * leftStick.y;
			g_camVel.x = 4.f * g_camSpeed * leftStick.x;
		}

		// Handle triggers as controller button events
		void ControllerButtonEvent(SDL_ControllerButtonEvent event);

		static bool bLeftTrigger = false;
		static bool bRightTrigger = false;
		SDL_ControllerButtonEvent e;

		if (!bLeftTrigger && trigger.x > 0.0f)
		{
			e.type = SDL_CONTROLLERBUTTONDOWN;
			e.button = SDL_CONTROLLER_BUTTON_LEFT_TRIGGER;
			ControllerButtonEvent(e);
			bLeftTrigger = true;
		}
		else if (bLeftTrigger && trigger.x == 0.0f)
		{
			e.type = SDL_CONTROLLERBUTTONUP;
			e.button = SDL_CONTROLLER_BUTTON_LEFT_TRIGGER;
			ControllerButtonEvent(e);
			bLeftTrigger = false;
		}

		if (!bRightTrigger && trigger.y > 0.0f)
		{
			e.type = SDL_CONTROLLERBUTTONDOWN;
			e.button = SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER;
			ControllerButtonEvent(e);
			bRightTrigger = true;
		}
		else if (bRightTrigger && trigger.y == 0.0f)
		{
			e.type = SDL_CONTROLLERBUTTONDOWN;
			e.button = SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER;
			ControllerButtonEvent(e);
			bRightTrigger = false;
		}
	}


}


void UpdateFrame(py::array_t<float> update_params)
{
	if (!g_headless)
		UpdateControllerInput();

	//-------------------------------------------------------------------
    // Scene Update

    double waitBeginTime = GetSeconds();

    MapBuffers(g_buffers);

    double waitEndTime = GetSeconds();

	float newSimLatency = 0.0f;
	float newGfxLatency = 0.0f;

	if (!g_headless)
	{
		// Getting timers causes CPU/GPU sync, so we do it after a map
		newSimLatency = NvFlexGetDeviceLatency(g_solver, &g_GpuTimers.computeBegin, &g_GpuTimers.computeEnd, &g_GpuTimers.computeFreq);
		newGfxLatency = RendererGetDeviceTimestamps(&g_GpuTimers.renderBegin, &g_GpuTimers.renderEnd, &g_GpuTimers.renderFreq);
		(void)newGfxLatency;

		UpdateCamera();

		if (!g_pause || g_step)
		{
			UpdateEmitters();
			UpdateMouse();
			UpdateWind();
			UpdateScene();
		}
		
	}
	else
	{
		if (g_render)
			UpdateCamera();

		UpdateEmitters();
		UpdateWind();
		UpdateScene();
	}

    //-------------------------------------------------------------------
    // Render

	double renderBeginTime = 0.0;
	double renderEndTime   = 0.0;

	int newScene = -1;

	if (g_render)
	{
		renderBeginTime = GetSeconds();

		if (g_profile && (!g_pause || g_step) && g_frame > 0)
		{
			if (g_benchmark)
			{
				g_numDetailTimers = NvFlexGetDetailTimers(g_solver, &g_detailTimers);
			}
			else
			{
				memset(&g_timers, 0, sizeof(g_timers));
				NvFlexGetTimers(g_solver, &g_timers);

                g_timersCount++;

                // update stats
                int numStats = sizeof(g_timers)/sizeof(float);
                for (int i=0; i < numStats; ++i)
                {                    
                   
                    float newValue = ((float*)(&g_timers))[i];
                    float mean = ((float*)(&g_timersAvg))[i];
                    float var = ((float*)(&g_timersVar))[i];

                    float delta = newValue-mean;
                    mean += delta/g_timersCount;

                    float deltaVar = newValue-mean;
                    var += delta*deltaVar;

                    ((float*)(&g_timersAvg))[i] = mean;
                    ((float*)(&g_timersVar))[i] = var;
                }
			}
		}

        if (!g_vrSystem)
        {
            StartFrame(Vec4(g_clearColor, 1.0f));

            // main scene render
            RenderScene();
            RenderDebug();

            if (!g_headless)
                newScene = DoUI();

            EndFrame();
        }
        else
        {
            for (unsigned i = 0; i < 2; ++i)
            {
                StartFrame(Vec4(g_clearColor, 1.0f));

                Matrix44 proj;
                Matrix44 view;

                // main scene render
                RenderScene(i, &proj, &view);

                g_vrSystem->RenderControllers(proj * view);

                RenderDebug();

                newScene = DoUI();

                const size_t eyeFb = g_vrSystem->GetEyeFrameBuffer(i);
                CopyFramebufferTo(eyeFb);
            }

            g_vrSystem->UploadGraphicsToDevice();
            EndFrame(g_vrSystem->GetEyeFrameBuffer(1), g_screenWidth, g_screenHeight, g_windowWidth, g_windowHeight);
        }


		// If user has disabled async compute, ensure that no compute can overlap
		// graphics by placing a sync between them
		if (!g_useAsyncCompute)
		{
			NvFlexComputeWaitForGraphics(g_flexLib);
		}
	}

    UnmapBuffers(g_buffers);

	if (!g_headless)
	{
		// move mouse particle (must be done here as GetViewRay() uses the GL projection state)
		if (g_mouseParticle != -1 || g_mouseJoint != -1)
		{
			Vec3 origin, dir;
			GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

            g_mousePos = origin + dir*g_mouseT;
        }
    }

    if (g_render)
    {
        if (g_capture)
        {
            TgaImage img;
            img.m_width = g_screenWidth;
            img.m_height = g_screenHeight;
            img.m_data = new uint32_t[g_screenWidth*g_screenHeight];

            ReadFrame((int*)img.m_data, g_screenWidth, g_screenHeight);

            fwrite(img.m_data, sizeof(uint32_t)*g_screenWidth*g_screenHeight, 1, g_ffmpeg);

            delete[] img.m_data;
        }

        renderEndTime = GetSeconds();
    }
	
    // if user requested a scene reset process it now
    if (g_resetScene)
    {
        assert(0); // xingyu: Reset need to have scene_params
        Reset();
        g_resetScene = false;
    }

    //-------------------------------------------------------------------
    // Flex Update

    double updateBeginTime = GetSeconds();

    // send any particle updates to the solver
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, NULL);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, NULL);
    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, NULL);
	NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());

    // update rigids
    if (g_buffers->rigidBodies.size())
    {
        NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
        NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
    }
    else
    {
        NvFlexSetRigidBodies(g_solver, NULL, 0);
        NvFlexSetRigidShapes(g_solver, NULL, 0);
    }

    // update joints
    if (g_buffers->rigidJoints.size())
    {
        NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());
    }
    else
    {
        NvFlexSetRigidJoints(g_solver, NULL, 0);
    }

    // allow scene to update constraints etc
    SyncScene();
    if (g_shapesChanged)
    {
        NvFlexSetShapes(
            g_solver,
            g_buffers->shapeGeometry.buffer,
            g_buffers->shapePositions.buffer,
            g_buffers->shapeRotations.buffer,
            g_buffers->shapePrevPositions.buffer,
            g_buffers->shapePrevRotations.buffer,
            g_buffers->shapeFlags.buffer,
            int(g_buffers->shapeFlags.size()));

        g_shapesChanged = false;
    }

    g_scene->PreSimulation();
    if (!g_scene->IsSkipSimulation())
    {
        if (!g_pause || g_step)
        {
            // tick solver
            NvFlexSetParams(g_solver, &g_params);
            NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

            g_frame++;
            g_step = false;
        }
    }

    g_scene->PostUpdate();

	g_cableDirty = true;

    // read back base particle data
    // Note that flexGet calls don't wait for the GPU, they just queue a GPU copy
    // to be executed later.
    // When we're ready to read the fetched buffers we'll Map them, and that's when
    // the CPU will wait for the GPU flex update and GPU copy to finish.
    NvFlexGetParticles(g_solver, g_buffers->positions.buffer, NULL);
    NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
    NvFlexGetNormals(g_solver, g_buffers->normals.buffer, NULL);

    // readback triangle normals
    if (g_buffers->triangles.size())
    {
        NvFlexGetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, NULL, g_buffers->triangles.size() / 3);
    }

    // readback rigid transforms
    if (g_buffers->shapeMatchingOffsets.size())
    {
        NvFlexGetRigids(g_solver, NULL, NULL, NULL, NULL, NULL, NULL, NULL, g_buffers->shapeMatchingRotations.buffer, g_buffers->shapeMatchingTranslations.buffer);
    }

    // tetrahedral stress
    if (g_buffers->tetraStress.size())
    {
        NvFlexGetFEMStress(g_solver, g_buffers->tetraStress.buffer);
    }

    // rigid bodies
    if (g_buffers->rigidBodies.size())
    {
        NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
    }

	// cables
	if (g_buffers->cableLinks.size())
	{
		NvFlexGetCableLinks(g_solver, g_buffers->cableLinks.buffer);
	}

	if (!g_interop && g_render)
	{
		// if not using interop then we read back fluid data to host
		if (g_drawEllipsoids)
		{
			NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, NULL);
			NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer, g_buffers->anisotropy3.buffer, NULL);
		}

		// read back diffuse data to host
		if (g_drawDensity)
		{
			NvFlexGetDensities(g_solver, g_buffers->densities.buffer, NULL);
		}

		if (GetNumDiffuseRenderParticles(g_diffuseRenderBuffers))
		{
			NvFlexGetDiffuseParticles(g_solver, g_buffers->diffusePositions.buffer, g_buffers->diffuseVelocities.buffer, g_buffers->diffuseCount.buffer);
		}
	}
	else if (g_render)
	{
		// read back just the new diffuse particle count, render buffers will be updated during rendering
		NvFlexGetDiffuseParticles(g_solver, NULL, NULL, g_buffers->diffuseCount.buffer);
	}



    double updateEndTime = GetSeconds();

    //-------------------------------------------------------
    // Update the on-screen timers

    float newUpdateTime = float(updateEndTime - updateBeginTime);
    float newRenderTime = float(renderEndTime - renderBeginTime);
    float newWaitTime = float(waitBeginTime - waitEndTime);

    // Exponential filter to make the display easier to read
    const float timerSmoothing = 1.0f;

    g_updateTime = (g_updateTime == 0.0f) ? newUpdateTime : Lerp(g_updateTime, newUpdateTime, timerSmoothing);
    g_renderTime = (g_renderTime == 0.0f) ? newRenderTime : Lerp(g_renderTime, newRenderTime, timerSmoothing);
    g_waitTime = (g_waitTime == 0.0f) ? newWaitTime : Lerp(g_waitTime, newWaitTime, timerSmoothing);
    g_simLatency = (g_simLatency == 0.0f) ? newSimLatency : Lerp(g_simLatency, newSimLatency, timerSmoothing);

    if (g_benchmark)
    {
        newScene = BenchmarkUpdate();
    }

    // flush out the last frame before freeing up resources in the event of a scene change
    // this is necessary for d3d12
    if (!g_headless)
	{
		PresentFrame(g_vsync);
	}
	else if (g_render)
	{
		PresentFrameHeadless();
	}

    // if gui or benchmark requested a scene change process it now
    if (newScene != -1)
    {
        g_sceneIndex = newScene;
        InitScene(g_sceneIndex);
    }
}

void UpdateFrame()
{
    py::array_t<float> update_params;
    UpdateFrame(update_params);
}

void ReshapeWindow(int width, int height)
{
    if (!g_benchmark)
    {
        printf("Reshaping\n");
    }

    g_windowWidth = width;
    g_windowHeight = height;

    int oldScreenWidth = g_screenWidth;
    int oldScreenHeight = g_screenWidth;

    if (!g_vrSystem)
    {
        g_screenWidth = g_windowWidth;
        g_screenHeight = g_windowHeight;
    }

    ReshapeRender(g_screenWidth, g_screenHeight);

    if (!g_fluidRenderer || (oldScreenWidth != g_screenWidth || oldScreenHeight != g_screenHeight))
    {
        if (g_fluidRenderer)
        {
            DestroyFluidRenderer(g_fluidRenderer);
        }
        g_fluidRenderer = CreateFluidRenderer(g_screenWidth, g_screenHeight);
    }
}

void InputArrowKeysDown(int key, int x, int y)
{
    switch (key)
    {
    case SDLK_DOWN:
    {
        if (g_selectedScene < int(g_sceneFactories.size()) - 1)
        {
            g_selectedScene++;
        }

        // update scroll UI to center on selected scene
        g_sceneScroll = max((g_selectedScene - 4) * 24, 0);
        break;
    }
    case SDLK_UP:
    {
        if (g_selectedScene > 0)
        {
            g_selectedScene--;
        }

        // update scroll UI to center on selected scene
        g_sceneScroll = max((g_selectedScene - 4) * 24, 0);
        break;
    }
    case SDLK_LEFT:
    {
        if (g_sceneIndex > 0)
        {
            --g_sceneIndex;
        }
        InitScene(g_sceneIndex);

        // update scroll UI to center on selected scene
        g_sceneScroll = max((g_sceneIndex - 4) * 24, 0);
        break;
    }
    case SDLK_RIGHT:
    {
        if (g_sceneIndex < int(g_sceneFactories.size()) - 1)
        {
            ++g_sceneIndex;
        }
        InitScene(g_sceneIndex);

        // update scroll UI to center on selected scene
        g_sceneScroll = max((g_sceneIndex - 4) * 24, 0);
        break;
    }
    }
}

void InputArrowKeysUp(int key, int x, int y)
{
}

void InputNumpadKeysDown(int key) 
{
	if (g_numpadPressedState.find(key) != g_numpadPressedState.end())
	{
		g_numpadPressedState[key] = true;
	}
}

void InputNumpadKeysUp(int key)
{
	if (g_numpadPressedState.find(key) != g_numpadPressedState.end())
	{
		g_numpadPressedState[key] = false;
	}
}

bool InputKeyboardDown(unsigned char key, int x, int y)
{
	/*
    if (key > '0' && key <= '9')
    {
        g_sceneIndex = key - '0' - 1;
        InitScene(g_sceneIndex);
        return false;
    }
	*/
    float kSpeed = g_camSpeed;

    switch (key)
    {
    case 'w':
    {
        g_camVel.z = kSpeed;
        break;
    }
    case 's':
    {
        g_camVel.z = -kSpeed;
        break;
    }
    case 'a':
    {
        g_camVel.x = -kSpeed;
        break;
    }
    case 'd':
    {
        g_camVel.x = kSpeed;
        break;
    }
    case 'q':
    {
		g_camVel.y = kSpeed;
	/*        
		static bool quartile = false;
        
        if (!quartile)
        {
            NvFlexStartExperiment("quartile.m", "quartile");
            NvFlexStartExperimentRun("pile");
            quartile = true;
        }
        else
        {
            NvFlexStopExperimentRun();
            NvFlexStopExperiment();
            quartile = false;
        }
	*/

        break;
    }
    case 'z':
    {
        //g_drawCloth = !g_drawCloth;
        g_camVel.y = -kSpeed;
        break;
    }

    case 'u':
    {
#ifndef ANDROID
        if (g_fullscreen)
        {
            SDL_SetWindowFullscreen(g_window, 0);
            ReshapeWindow(1280, 720);
            g_fullscreen = false;
        }
        else
        {
            SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
            g_fullscreen = true;
        }
#endif
        break;
    }
    case 'r':
    {
        g_resetScene = true;
        break;
    }
    case 'y':
    {
        g_wavePool = !g_wavePool;
        break;
    }
    case 'c':
    {
        if (!g_ffmpeg)
        {
            printf("Start capture at camera pos: %f %f %f, angle: %f %f %f\n", g_camPos.x, g_camPos.y, g_camPos.z, g_camAngle.x, g_camAngle.y, g_camAngle.z);

            // open ffmpeg stream
            int i = 0;
            char buf[255];
            FILE* f = NULL;

            do
            {
                sprintf(buf, "../../movies/output%d.mp4", i);
                f = fopen(buf, "rb");
                if (f)
                {
                    fclose(f);
                }

                ++i;
            }
            while (f);

            const char* str = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s %dx%d -i - "
                              "-threads 0 -preset fast -y -crf 19 -pix_fmt yuv420p -tune animation -vf vflip %s";

            char cmd[1024];
            sprintf(cmd, str, g_screenWidth, g_screenHeight, buf);
#if _WIN32
            g_ffmpeg = _popen(cmd, "wb");
#elif __linux__
            g_ffmpeg = popen(cmd, "w");
#endif
            assert(g_ffmpeg);
        }
        else
        {
#if _WIN32
        	_pclose(g_ffmpeg);
#elif __linux__
        	pclose(g_ffmpeg);
#endif
            g_ffmpeg = NULL;
        }

        g_capture = !g_capture;
        g_frame = 0;
        break;
    }
    case 'p':
    {
        g_pause = !g_pause;
        break;
    }
    case 'o':
    {
        g_step = true;
        break;
    }
    case 'h':
    {
        g_showHelp = !g_showHelp;
        break;
    }
    case 'e':
    {
        g_drawEllipsoids = !g_drawEllipsoids;
        break;
    }
    case 't':
    {
        g_drawOpaque = !g_drawOpaque;
        break;
    }
    case 'v':
    {
        g_drawPoints = !g_drawPoints;
        break;
    }
    case 'f':
    {
        g_drawSprings = (g_drawSprings + 1) % 3;
        break;
    }
    case 'i':
    {
        g_drawDiffuse = !g_drawDiffuse;
        break;
    }
    case 'm':
    {
          g_drawMesh = !g_drawMesh;
          break;
    }
    case 'n':
    {
        g_drawRopes = !g_drawRopes;
        break;
    }
    case 'j':
    {
        g_windTime = 0.0f;
        g_windStrength = 1.5f;
        g_windFrequency = 0.2f;
        break;
    }
    case '.':
    {
        g_profile = !g_profile;
        break;
    }
    case 'g':
    {
        if (g_params.gravity[1] != 0.0f)
        {
            g_params.gravity[1] = 0.0f;
        }
        else
        {
            g_params.gravity[1] = -9.8f;
        }

        break;
    }
    case '-':
    {
        if (g_params.numPlanes)
        {
            g_params.numPlanes--;
        }

        break;
    }
    case ' ':
    {
        g_emit = !g_emit;
        break;
    }
    case ';':
    {
        g_debug = !g_debug;
		if (g_solver)
			NvFlexSetDebug(g_solver, g_debug, 0, 0, 0.0f);

        break;
    }
    case 13:
    {
        g_sceneIndex = g_selectedScene;
        InitScene(g_sceneIndex);
        break;
    }
    case 27:
    {
        // return quit = true
        return true;
    }
    };

    g_scene->KeyDown(key);

    return false;
}

void InputKeyboardUp(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'w':
    case 's':
    {
        g_camVel.z = 0.0f;
        break;
    }
    case 'a':
    case 'd':
    {
        g_camVel.x = 0.0f;
        break;
    }
    case 'q':
    case 'z':
    {
        g_camVel.y = 0.0f;
        break;
    }
    };
}

void MouseFunc(int b, int state, int x, int y)
{
    FixMouseCoordinatesForVr(x, y);

    switch (state)
    {
    case SDL_RELEASED:
    {
        g_lastx = x;
        g_lasty = y;
        g_lastb = -1;

        break;
    }
    case SDL_PRESSED:
    {
        g_lastx = x;
        g_lasty = y;
        g_lastb = b;
#ifdef ANDROID
        extern void setStateLeft(bool bLeftDown);
        setStateLeft(false);
#else
        if ((SDL_GetModState() & KMOD_LSHIFT) && g_lastb == SDL_BUTTON_LEFT)
        {
            // record that we need to update the picked particle
            g_mousePicked = true;
			g_mouseParticle = -1;
        }
#endif
        break;
    }
    };
}

void MousePassiveMotionFunc(int x, int y)
{
    FixMouseCoordinatesForVr(x, y);
    g_lastx = x;
    g_lasty = y;
}

void MouseMotionFunc(unsigned state, int x, int y)
{
    FixMouseCoordinatesForVr(x, y);

    float dx = float(x - g_lastx);
    float dy = float(y - g_lasty);

    g_lastx = x;
    g_lasty = y;

    if (state & SDL_BUTTON_RMASK)
    {
        const float kSensitivity = DegToRad(0.1f);
        const float kMaxDelta = FLT_MAX;

        g_camAngle.x -= Clamp(dx*kSensitivity, -kMaxDelta, kMaxDelta);
        g_camAngle.y -= Clamp(dy*kSensitivity, -kMaxDelta, kMaxDelta);
    }
}

bool g_error = false;

void ErrorCallback(NvFlexErrorSeverity severity, const char* msg, const char* file, int line)
{
    printf("Flex: %s - %s:%d\n", msg, file, line);
    g_error = (severity == eNvFlexLogError);
    //assert(0); asserts are bad for TeamCity
}

void ControllerButtonEvent(SDL_ControllerButtonEvent event)
{
	/*
    // map controller buttons to keyboard keys
    if (event.type == SDL_CONTROLLERBUTTONDOWN)
    {
        InputKeyboardDown(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);
        InputArrowKeysDown(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);

        if (event.button == SDL_CONTROLLER_BUTTON_LEFT_TRIGGER)
        {
            // Handle picking events using the game controller
            g_lastx = g_screenWidth / 2;
            g_lasty = g_screenHeight / 2;
            g_lastb = 1;

            // record that we need to update the picked particle
            g_mousePicked = true;
        }
    }
    else
    {
        InputKeyboardUp(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);
        InputArrowKeysUp(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);

        if (event.button == SDL_CONTROLLER_BUTTON_LEFT_TRIGGER)
        {
            // Handle picking events using the game controller
            g_lastx = g_screenWidth / 2;
            g_lasty = g_screenHeight / 2;
            g_lastb = -1;
        }
    }
	*/
}

void ControllerDeviceUpdate()
{
    if (SDL_NumJoysticks() > 0)
    {
        SDL_JoystickEventState(SDL_ENABLE);
        if (SDL_IsGameController(0))
        {
            g_gamecontroller = SDL_GameControllerOpen(0);
        }
    }
}

void SDLInit(const char* title, bool resizableWindow = true)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0)	// Initialize SDL's Video subsystem and game controllers
    {
        printf("Unable to initialize SDL");
    }

#if FLEX_DX
    unsigned int flags = 0;
#else
    unsigned int flags = SDL_WINDOW_OPENGL;
#endif
    if (resizableWindow)
    {
        flags |= SDL_WINDOW_RESIZABLE;
    }

    g_window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                g_windowWidth, g_windowHeight, flags);

    g_windowId = SDL_GetWindowID(g_window);
}

void SDLMainLoop()
{
    bool quit = false;
    SDL_Event e;
    while (!quit)
    {
        if (g_vrSystem)
        {
            g_vrSystem->Update();
        }

        UpdateFrame();

        while (SDL_PollEvent(&e))
        {
            switch (e.type)
            {
            case SDL_QUIT:
                quit = true;
                break;

            case SDL_KEYDOWN:
                if (e.key.keysym.sym < 256 && (e.key.keysym.mod == KMOD_NONE || (e.key.keysym.mod & KMOD_NUM)))
                {
                    quit = InputKeyboardDown(e.key.keysym.sym, 0, 0);
                }
                InputArrowKeysDown(e.key.keysym.sym, 0, 0);
				InputNumpadKeysDown(e.key.keysym.sym);
                break;

            case SDL_KEYUP:
                if (e.key.keysym.sym < 256 && (e.key.keysym.mod == 0 || (e.key.keysym.mod & KMOD_NUM)))
                {
                    InputKeyboardUp(e.key.keysym.sym, 0, 0);
                }
                InputArrowKeysUp(e.key.keysym.sym, 0, 0);
				InputNumpadKeysUp(e.key.keysym.sym);
                break;

            case SDL_MOUSEMOTION:
                if (e.motion.state)
                {
                    MouseMotionFunc(e.motion.state, e.motion.x, e.motion.y);
                }
                else
                {
                    MousePassiveMotionFunc(e.motion.x, e.motion.y);
                }
                break;

            case SDL_MOUSEBUTTONDOWN:
            case SDL_MOUSEBUTTONUP:
                MouseFunc(e.button.button, e.button.state, e.motion.x, e.motion.y);
                break;

            case SDL_WINDOWEVENT:
                if (e.window.windowID == g_windowId)
                {
                    if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                    {
                        ReshapeWindow(e.window.data1, e.window.data2);
                    }
                }
                break;

            case SDL_WINDOWEVENT_LEAVE:
                g_camVel = Vec3(0.0f, 0.0f, 0.0f);
                break;

            case SDL_CONTROLLERBUTTONUP:
            case SDL_CONTROLLERBUTTONDOWN:
                ControllerButtonEvent(e.cbutton);
                break;

            case SDL_JOYDEVICEADDED:
            case SDL_JOYDEVICEREMOVED:
                ControllerDeviceUpdate();
                break;
            }
        }
    }
}
    
void HeadlessMainLoop()
{
	bool quit = false;
	while (!quit)  
	{
		UpdateFrame();
	}
}


static bool SetSceneIndexByName(const char* sceneName)
{
    for (int i=0; i < int(g_sceneFactories.size()); ++i)
    {
        if (strcmp(g_sceneFactories[i].mName, sceneName) == 0)
        {
            g_sceneIndex = i;
            return true;
        }
    }
    return false;
}

inline void FixPathBuffer(char* buffer, size_t bufferSize)
{
#if ISAAC_PLATFORM_WINDOWS
    constexpr char ReplacedSepartor = '/';
    constexpr char ReplacingSepartor = '\\';
#else
    constexpr char ReplacedSepartor = '\\';
    constexpr char ReplacingSepartor = '/';
#endif

    for (size_t i = 0; i < bufferSize && buffer[i]; ++i)
    {
        if (buffer[i] == ReplacedSepartor)
        {
            buffer[i] = ReplacingSepartor;
        }
    }
}

static void MergeJson(json& target, const json& source)
{
    if (source.is_null())
    {
        return;
    }

    // Adding new parameters from source or overwriting already present ones in the target
    for (auto it = source.begin(); it != source.end(); ++it)
    {
        const string& key = it.key();
        target[key] = it.value();
    }
}

static json LoadJson(const std::string& filePath, const std::string& parentJsonKey) noexcept
{
    static const string jsonExt = ".json";

    try
    {
        const bool endsWithJson = (filePath.size() >= jsonExt.size() && std::equal(jsonExt.rbegin(), jsonExt.rend(), filePath.rbegin()));

        ifstream jsonFile(endsWithJson ? filePath : filePath + jsonExt);
        json resultJson;
        jsonFile >> resultJson;

        if (!parentJsonKey.empty())
        {
            const string parentJsonPath = resultJson.value(parentJsonKey, string());

            if (!parentJsonPath.empty())
            {
                std::array<char, 1024> parentPathBuffer;
                MakeRelativePath(filePath.c_str(), parentJsonPath.c_str(), parentPathBuffer.data());
                FixPathBuffer(parentPathBuffer.data(), parentPathBuffer.size());

                json parentJson = LoadJson(parentPathBuffer.data(), parentJsonKey);
                if (!parentJson.is_null())
                {
                    MergeJson(parentJson, resultJson);
                    resultJson = parentJson;
                }
            }
        }
        return resultJson;
    }
    catch (exception& e)
    {
        printf("%s\n", e.what());
        printf("Error loading JSON file \"%s\".\n", filePath.c_str());
        exit(-1);
    }
}

static json JsonFromString(const std::string& source) noexcept
{
    try
    {
        json result = json::parse(source);
        return result;
    }
    catch (exception& e)
    {
        printf("%s\n", e.what());
        printf("Error pasing JSON from command line.\n");
        exit(-1);
    }
}
void pyflex_init_debug(bool headless=false, bool render=true, int camera_width=720, int camera_height=720) {
    g_screenWidth = camera_width;
    g_screenHeight = camera_height;

    g_headless = headless;
    g_render = render;
    if (g_headless) {
        g_interop = false;
        g_pause = false;
    }

    RandInit();
//	g_argc = argc;
//	g_argv = argv;
    RegisterPhysicsScenes();
	RegisterExperimentScenes();

	#ifndef ANDROID

#if FLEX_DX
    const char* title = "Flex Gym (Direct Compute)";
#else
    const char* title = "Flex Gym (CUDA)";
#endif

#if FLEX_VR
    if (g_sceneFactories[g_sceneIndex].mIsVR)
    {
        g_vrSystem = VrSystem::Create(g_camPos, GetCameraRotationMatrix(), g_vrMoveScale);
        if (!g_vrSystem)
        {
            printf("Error during VR initialization, terminating process");
            exit(1);
        }
    }

    if (g_vrSystem)
    {
        // Setting the same aspect for the window as for VR (not really necessary)
        g_screenWidth = g_vrSystem->GetRecommendedRtWidth();
        g_screenHeight = g_vrSystem->GetRecommendedRtHeight();

        float vrAspect = static_cast<float>(g_screenWidth) / g_screenHeight;
        g_windowWidth = static_cast<int>(vrAspect * g_windowHeight);
    }
    else
#endif // #if FLEX_VR
    {
        g_screenWidth = g_windowWidth;
        g_screenHeight = g_windowHeight;
    }

    if (!g_headless)
    {
        SDLInit(title);
    }

    RenderInitOptions options;
    options.window = g_window;
    options.numMsaaSamples = g_msaaSamples;
    options.asyncComputeBenchmark = g_asyncComputeBenchmark;
    options.defaultFontHeight = -1;
    options.fullscreen = g_fullscreen;

#if FLEX_DX
    {
        DemoContext* demoContext = nullptr;

        if (g_d3d12)
        {
            // workaround for a driver issue with D3D12 with msaa, force it to off
            options.numMsaaSamples = 1;

            demoContext = CreateDemoContextD3D12();
        }
        else
        {
            demoContext = CreateDemoContextD3D11();
        }
        // Set the demo context
        SetDemoContext(demoContext);
    }
#endif
	if (!g_headless)
	{
		InitRender(options);

		if (g_vrSystem)
		{
			g_vrSystem->InitGraphicalResources();
		}

		if (g_fullscreen)
		{
			SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
		}

		ReshapeWindow(g_windowWidth, g_windowHeight);
	}
#endif // ifndef ANDROID

#if !FLEX_DX

#if 0

    // use the PhysX GPU selected from the NVIDIA control panel
    if (g_device == -1)
    {
        g_device = NvFlexDeviceGetSuggestedOrdinal();
    }

    // Create an optimized CUDA context for Flex and set it on the
    // calling thread. This is an optional call, it is fine to use
    // a regular CUDA context, although creating one through this API
    // is recommended for best performance.
    bool success = NvFlexDeviceCreateCudaContext(g_device);

    if (!success)
    {
        printf("Error creating CUDA context.\n");
        exit(-1);
    }

#endif // _WIN32

#endif

    NvFlexInitDesc desc;
    desc.deviceIndex = g_device;
    desc.enableExtensions = g_extensions;
    desc.renderDevice = 0;
    desc.renderContext = 0;
    desc.computeContext = 0;
    desc.computeType = eNvFlexCUDA;

#if FLEX_DX

    if (g_d3d12)
    {
        desc.computeType = eNvFlexD3D12;
    }
    else
    {
        desc.computeType = eNvFlexD3D11;
    }

    bool userSpecifiedGpuToUseForFlex = (g_device != -1);

    if (userSpecifiedGpuToUseForFlex)
    {
        // Flex doesn't currently support interop between different D3DDevices.
        // If the user specifies which physical device to use, then Flex always
        // creates its own D3DDevice, even if graphics is on the same physical device.
        // So specified physical device always means no interop.
        g_interop = false;
    }
    else
    {
        // Ask Flex to run on the same GPU as rendering
        GetRenderDevice(&desc.renderDevice,
                        &desc.renderContext);
    }

    // Shared resources are unimplemented on D3D12,
    // so disable it for now.
    if (g_d3d12)
    {
        g_interop = false;
    }

    // Setting runOnRenderContext = true doesn't prevent async compute, it just
    // makes Flex send compute and graphics to the GPU on the same queue.
    //
    // So to allow the user to toggle async compute, we set runOnRenderContext = false
    // and provide a toggleable sync between compute and graphics in the app.
    //
    // Search for g_useAsyncCompute for details
    desc.runOnRenderContext = false;
#endif

    // Init Flex library, note that no CUDA methods should be called before this
    // point to ensure we get the device context we want
    g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);

    if (g_error || g_flexLib == NULL)
    {
        printf("Could not initialize Flex, exiting.\n");
        exit(-1);
    }

    // store device name
    strcpy(g_deviceName, NvFlexGetDeviceName(g_flexLib));
    printf("Compute Device: %s\n\n", g_deviceName);

    if (g_benchmark)
    {
        g_sceneIndex = BenchmarkInit();
    }

	if (g_render == true)
	{
		// create shadow maps
		g_shadowMap = ShadowCreate();

		// create default render meshes
		Mesh* sphere = CreateSphere(12, 24, 1.0f);
		Mesh* cylinder = CreateCylinder(24, 1.0f, 1.0f);
		Mesh* box = CreateCubeMesh();

		g_sphereMesh = CreateRenderMesh(sphere);
		g_cylinderMesh = CreateRenderMesh(cylinder);
		g_boxMesh = CreateRenderMesh(box);

		delete sphere;
		delete cylinder;
		delete box;
	}

	if (!g_experiment)
	{
		// to ensure D3D context is active
		StartGpuWork();

		// init default scene
		InitScene(g_sceneIndex);

		// release context
		EndGpuWork();
		if (g_headless == true)
		{
			HeadlessMainLoop();
		}
		else
		{
			SDLMainLoop();
		}
	}
	else
	{
		RunExperiments(g_experimentFilter);
		exit(0);
	}

	if (g_render == true)
	{
		DestroyFluidRenderer(g_fluidRenderer);
		DestroyFluidRenderBuffers(g_fluidRenderBuffers);
		DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);

		ShadowDestroy(g_shadowMap);
	}

    Shutdown();

	if (g_headless == false)
	{
		DestroyRender();

		SDL_DestroyWindow(g_window);
		SDL_Quit();
	}

    printf("Pyflex init done!\n");
}

void pyflex_init(bool headless=false, bool render=true, int camera_width=720, int camera_height=720) {
    g_screenWidth = camera_width;
    g_screenHeight = camera_height;

    g_headless = headless;
    g_render = render;
    if (g_headless) {
        g_interop = false;
        g_pause = false;
    }

    RandInit();
//	g_argc = argc;
//	g_argv = argv;
    RegisterPhysicsScenes();
	RegisterExperimentScenes();

    // init gl
#ifndef ANDROID

#if FLEX_DX
    const char* title = "Flex Gym (Direct Compute)";
#else
    const char* title = "Flex Gym (CUDA)";
#endif

#if FLEX_VR
    if (g_sceneFactories[g_sceneIndex].mIsVR)
    {
        g_vrSystem = VrSystem::Create(g_camPos, GetCameraRotationMatrix(), g_vrMoveScale);
        if (!g_vrSystem)
        {
            printf("Error during VR initialization, terminating process");
            exit(1);
        }
    }

    if (g_vrSystem)
    {
        // Setting the same aspect for the window as for VR (not really necessary)
        g_screenWidth = g_vrSystem->GetRecommendedRtWidth();
        g_screenHeight = g_vrSystem->GetRecommendedRtHeight();

        float vrAspect = static_cast<float>(g_screenWidth) / g_screenHeight;
        g_windowWidth = static_cast<int>(vrAspect * g_windowHeight);
    }
    else
#endif // #if FLEX_VR
    {
        g_screenWidth = g_windowWidth;
        g_screenHeight = g_windowHeight;
    }

    if (!g_headless)
    {
        SDLInit(title);
    }

    RenderInitOptions options;
    options.window = g_window;
    options.numMsaaSamples = g_msaaSamples;
    options.asyncComputeBenchmark = g_asyncComputeBenchmark;
    options.defaultFontHeight = -1;
    options.fullscreen = g_fullscreen;

#if FLEX_DX
    {
        DemoContext* demoContext = nullptr;

        if (g_d3d12)
        {
            // workaround for a driver issue with D3D12 with msaa, force it to off
            options.numMsaaSamples = 1;

            demoContext = CreateDemoContextD3D12();
        }
        else
        {
            demoContext = CreateDemoContextD3D11();
        }
        // Set the demo context
        SetDemoContext(demoContext);
    }
#endif
	if (!g_headless)
	{
		InitRender(options);

		if (g_vrSystem)
		{
			g_vrSystem->InitGraphicalResources();
		}

		if (g_fullscreen)
		{
			SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
		}

		ReshapeWindow(g_windowWidth, g_windowHeight);
	}
#endif // ifndef ANDROID

#if !FLEX_DX

#if 0

    // use the PhysX GPU selected from the NVIDIA control panel
    if (g_device == -1)
    {
        g_device = NvFlexDeviceGetSuggestedOrdinal();
    }

    // Create an optimized CUDA context for Flex and set it on the
    // calling thread. This is an optional call, it is fine to use
    // a regular CUDA context, although creating one through this API
    // is recommended for best performance.
    bool success = NvFlexDeviceCreateCudaContext(g_device);

    if (!success)
    {
        printf("Error creating CUDA context.\n");
        exit(-1);
    }

#endif // _WIN32

#endif

    NvFlexInitDesc desc;
    desc.deviceIndex = g_device;
    desc.enableExtensions = g_extensions;
    desc.renderDevice = 0;
    desc.renderContext = 0;
    desc.computeContext = 0;
    desc.computeType = eNvFlexCUDA;

#if FLEX_DX

    if (g_d3d12)
    {
        desc.computeType = eNvFlexD3D12;
    }
    else
    {
        desc.computeType = eNvFlexD3D11;
    }

    bool userSpecifiedGpuToUseForFlex = (g_device != -1);

    if (userSpecifiedGpuToUseForFlex)
    {
        // Flex doesn't currently support interop between different D3DDevices.
        // If the user specifies which physical device to use, then Flex always
        // creates its own D3DDevice, even if graphics is on the same physical device.
        // So specified physical device always means no interop.
        g_interop = false;
    }
    else
    {
        // Ask Flex to run on the same GPU as rendering
        GetRenderDevice(&desc.renderDevice,
                        &desc.renderContext);
    }

    // Shared resources are unimplemented on D3D12,
    // so disable it for now.
    if (g_d3d12)
    {
        g_interop = false;
    }

    // Setting runOnRenderContext = true doesn't prevent async compute, it just
    // makes Flex send compute and graphics to the GPU on the same queue.
    //
    // So to allow the user to toggle async compute, we set runOnRenderContext = false
    // and provide a toggleable sync between compute and graphics in the app.
    //
    // Search for g_useAsyncCompute for details
    desc.runOnRenderContext = false;
#endif

    // Init Flex library, note that no CUDA methods should be called before this
    // point to ensure we get the device context we want
    g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);

    if (g_error || g_flexLib == NULL)
    {
        printf("Could not initialize Flex, exiting.\n");
        exit(-1);
    }

    // store device name
    strcpy(g_deviceName, NvFlexGetDeviceName(g_flexLib));
    printf("Compute Device: %s\n\n", g_deviceName);

    if (g_benchmark)
    {
        g_sceneIndex = BenchmarkInit();
    }

	if (g_render == true)
	{
		// create shadow maps
		g_shadowMap = ShadowCreate();

		// create default render meshes
		Mesh* sphere = CreateSphere(12, 24, 1.0f);
		Mesh* cylinder = CreateCylinder(24, 1.0f, 1.0f);
		Mesh* box = CreateCubeMesh();

		g_sphereMesh = CreateRenderMesh(sphere);
		g_cylinderMesh = CreateRenderMesh(cylinder);
		g_boxMesh = CreateRenderMesh(box);

		delete sphere;
		delete cylinder;
		delete box;
	}

	if (!g_experiment)
	{
		// to ensure D3D context is active
		StartGpuWork();

		// init default scene
		InitScene(g_sceneIndex);

		// release context
		EndGpuWork();
		if (g_headless == true)
		{
			HeadlessMainLoop();
		}
		else
		{
			SDLMainLoop();
		}
	}
	else
	{
		RunExperiments(g_experimentFilter);
		exit(0);
	}

	if (g_render == true)
	{
		DestroyFluidRenderer(g_fluidRenderer);
		DestroyFluidRenderBuffers(g_fluidRenderBuffers);
		DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);

		ShadowDestroy(g_shadowMap);
	}

    Shutdown();

	if (g_headless == false)
	{
		DestroyRender();

		SDL_DestroyWindow(g_window);
		SDL_Quit();
	}

    printf("Pyflex init done!\n");
}

//int main(int argc, char* argv[])
//{
//    RandInit();
//	g_argc = argc;
//	g_argv = argv;
//    RegisterPhysicsScenes();
//	RegisterExperimentScenes();
//    //RegisterLearningScenes();
//
//
//    // process command line args
//    json cmdJson;
//
//    for (int i = 1; i < argc; ++i)
//    {
//        int d;
//        if (sscanf(argv[i], "-device=%d", &d))
//        {
//            g_device = d;
//        }
//
//        char sceneTmpString[1024];
//        bool usedSceneOption = false;
//        if (sscanf(argv[i], "-scene=%[^\n]", sceneTmpString) == 1)
//        {
//            usedSceneOption = true;
//            SetSceneIndexByName(sceneTmpString);
//        }
//
//        // Loading json for the scene
//        if (sscanf(argv[i], "-config=%[^\n]", sceneTmpString) == 1)
//        {
//            if (usedSceneOption)
//            {
//                printf("Warning! Used \"-config\" option overwrites \"-scene\" option\n");
//            }
//
//            g_sceneJson = LoadJson(sceneTmpString, RL_JSON_PARENT_RELATIVE_PATH);
//        }
//
//        if (sscanf(argv[i], "-json=%[^\n]", sceneTmpString) == 1)
//        {
//            cmdJson = JsonFromString(sceneTmpString);
//        }
//
//        if (sscanf(argv[i], "-extensions=%d", &d))
//        {
//            g_extensions = d != 0;
//        }
//
//        if (strcmp(argv[i], "-benchmark") == 0)
//        {
//            g_benchmark = true;
//            g_profile = true;
//            g_outputAllFrameTimes = false;
//            g_vsync = false;
//            g_fullscreen = true;
//        }
//
//        if (strcmp(argv[i], "-d3d12") == 0)
//        {
//            g_d3d12 = true;
//            // Currently interop doesn't work on d3d12
//            g_interop = false;
//        }
//
//		if (strcmp(argv[i], "-headless") == 0)
//		{
//			g_headless = true;
//			// No graphics so interop doesn't make sense
//			g_interop = false;
//			g_pause = false;
//		}
//
//		if (strcmp(argv[i], "-norender") == 0)
//		{
//			g_headless = true;
//			g_render = false;
//			g_interop = false;
//			g_pause = false;
//		}
//
//        if (strcmp(argv[i], "-benchmarkAllFrameTimes") == 0)
//        {
//            g_benchmark = true;
//            g_outputAllFrameTimes = true;
//        }
//
//        if (strcmp(argv[i], "-tc") == 0)
//        {
//            g_teamCity = true;
//        }
//
//        if (sscanf(argv[i], "-msaa=%d", &d))
//        {
//            g_msaaSamples = d;
//        }
//
//        int w = 1280;
//        int h = 720;
//        if (sscanf(argv[i], "-fullscreen=%dx%d", &w, &h) == 2)
//        {
//            g_windowWidth = w;
//            g_windowHeight = h;
//            g_fullscreen = true;
//        }
//        else if (strcmp(argv[i], "-fullscreen") == 0)
//        {
//            g_windowWidth = w;
//            g_windowHeight = h;
//            g_fullscreen = true;
//        }
//
//        if (sscanf(argv[i], "-windowed=%dx%d", &w, &h) == 2)
//        {
//            g_windowWidth = w;
//            g_windowHeight = h;
//            g_fullscreen = false;
//        }
//        else if (strstr(argv[i], "-windowed"))
//        {
//            g_windowWidth = w;
//            g_windowHeight = h;
//            g_fullscreen = false;
//        }
//
//        if (sscanf(argv[i], "-vsync=%d", &d))
//        {
//            g_vsync = d != 0;
//        }
//
//        if (sscanf(argv[i], "-multiplier=%d", &d) == 1)
//        {
//            g_numExtraMultiplier = d;
//        }
//
//        if (strcmp(argv[i], "-disabletweak") == 0)
//        {
//            g_tweakPanel = false;
//        }
//
//        if (strcmp(argv[i], "-disableinterop") == 0)
//        {
//            g_interop = false;
//        }
//        if (sscanf(argv[i], "-asynccompute=%d", &d) == 1)
//        {
//            g_useAsyncCompute = (d != 0);
//        }
//		if (sscanf(argv[i], "-experiment=%s", &g_experimentFilter) == 1)
//		{
//			g_experiment = true;
//		}
//    }
//
//    if (!cmdJson.is_null())
//    {
//        MergeJson(g_sceneJson, cmdJson);
//    }
//
//    if (!g_sceneJson.is_null())
//    {
//        const string& sceneName = JsonGetOrExit<const string>(g_sceneJson, RL_JSON_SCENE_NAME, "Error while parsing specified config file.");
//
//        if (!SetSceneIndexByName(sceneName.c_str()))
//        {
//            printf("Unknown scene name: \"%s\"", sceneName.c_str());
//            exit(-1);
//        }
//    }
//
//    // init gl
//#ifndef ANDROID
//
//#if FLEX_DX
//    const char* title = "Flex Gym (Direct Compute)";
//#else
//    const char* title = "Flex Gym (CUDA)";
//#endif
//
//#if FLEX_VR
//    if (g_sceneFactories[g_sceneIndex].mIsVR)
//    {
//        g_vrSystem = VrSystem::Create(g_camPos, GetCameraRotationMatrix(), g_vrMoveScale);
//        if (!g_vrSystem)
//        {
//            printf("Error during VR initialization, terminating process");
//            exit(1);
//        }
//    }
//
//    if (g_vrSystem)
//    {
//        // Setting the same aspect for the window as for VR (not really necessary)
//        g_screenWidth = g_vrSystem->GetRecommendedRtWidth();
//        g_screenHeight = g_vrSystem->GetRecommendedRtHeight();
//
//        float vrAspect = static_cast<float>(g_screenWidth) / g_screenHeight;
//        g_windowWidth = static_cast<int>(vrAspect * g_windowHeight);
//    }
//    else
//#endif // #if FLEX_VR
//    {
//        g_screenWidth = g_windowWidth;
//        g_screenHeight = g_windowHeight;
//    }
//
//    if (!g_headless)
//    {
//        SDLInit(title);
//    }
//
//    RenderInitOptions options;
//    options.window = g_window;
//    options.numMsaaSamples = g_msaaSamples;
//    options.asyncComputeBenchmark = g_asyncComputeBenchmark;
//    options.defaultFontHeight = -1;
//    options.fullscreen = g_fullscreen;
//
//#if FLEX_DX
//    {
//        DemoContext* demoContext = nullptr;
//
//        if (g_d3d12)
//        {
//            // workaround for a driver issue with D3D12 with msaa, force it to off
//            options.numMsaaSamples = 1;
//
//            demoContext = CreateDemoContextD3D12();
//        }
//        else
//        {
//            demoContext = CreateDemoContextD3D11();
//        }
//        // Set the demo context
//        SetDemoContext(demoContext);
//    }
//#endif
//	if (!g_headless)
//	{
//		InitRender(options);
//
//		if (g_vrSystem)
//		{
//			g_vrSystem->InitGraphicalResources();
//		}
//
//		if (g_fullscreen)
//		{
//			SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
//		}
//
//		ReshapeWindow(g_windowWidth, g_windowHeight);
//	}
//#endif // ifndef ANDROID
//
//#if !FLEX_DX
//
//#if 0
//
//    // use the PhysX GPU selected from the NVIDIA control panel
//    if (g_device == -1)
//    {
//        g_device = NvFlexDeviceGetSuggestedOrdinal();
//    }
//
//    // Create an optimized CUDA context for Flex and set it on the
//    // calling thread. This is an optional call, it is fine to use
//    // a regular CUDA context, although creating one through this API
//    // is recommended for best performance.
//    bool success = NvFlexDeviceCreateCudaContext(g_device);
//
//    if (!success)
//    {
//        printf("Error creating CUDA context.\n");
//        exit(-1);
//    }
//
//#endif // _WIN32
//
//#endif
//
//    NvFlexInitDesc desc;
//    desc.deviceIndex = g_device;
//    desc.enableExtensions = g_extensions;
//    desc.renderDevice = 0;
//    desc.renderContext = 0;
//    desc.computeContext = 0;
//    desc.computeType = eNvFlexCUDA;
//
//#if FLEX_DX
//
//    if (g_d3d12)
//    {
//        desc.computeType = eNvFlexD3D12;
//    }
//    else
//    {
//        desc.computeType = eNvFlexD3D11;
//    }
//
//    bool userSpecifiedGpuToUseForFlex = (g_device != -1);
//
//    if (userSpecifiedGpuToUseForFlex)
//    {
//        // Flex doesn't currently support interop between different D3DDevices.
//        // If the user specifies which physical device to use, then Flex always
//        // creates its own D3DDevice, even if graphics is on the same physical device.
//        // So specified physical device always means no interop.
//        g_interop = false;
//    }
//    else
//    {
//        // Ask Flex to run on the same GPU as rendering
//        GetRenderDevice(&desc.renderDevice,
//                        &desc.renderContext);
//    }
//
//    // Shared resources are unimplemented on D3D12,
//    // so disable it for now.
//    if (g_d3d12)
//    {
//        g_interop = false;
//    }
//
//    // Setting runOnRenderContext = true doesn't prevent async compute, it just
//    // makes Flex send compute and graphics to the GPU on the same queue.
//    //
//    // So to allow the user to toggle async compute, we set runOnRenderContext = false
//    // and provide a toggleable sync between compute and graphics in the app.
//    //
//    // Search for g_useAsyncCompute for details
//    desc.runOnRenderContext = false;
//#endif
//
//    // Init Flex library, note that no CUDA methods should be called before this
//    // point to ensure we get the device context we want
//    g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);
//
//    if (g_error || g_flexLib == NULL)
//    {
//        printf("Could not initialize Flex, exiting.\n");
//        exit(-1);
//    }
//
//    // store device name
//    strcpy(g_deviceName, NvFlexGetDeviceName(g_flexLib));
//    printf("Compute Device: %s\n\n", g_deviceName);
//
//    if (g_benchmark)
//    {
//        g_sceneIndex = BenchmarkInit();
//    }
//
//	if (g_render == true)
//	{
//		// create shadow maps
//		g_shadowMap = ShadowCreate();
//
//		// create default render meshes
//		Mesh* sphere = CreateSphere(12, 24, 1.0f);
//		Mesh* cylinder = CreateCylinder(24, 1.0f, 1.0f);
//		Mesh* box = CreateCubeMesh();
//
//		g_sphereMesh = CreateRenderMesh(sphere);
//		g_cylinderMesh = CreateRenderMesh(cylinder);
//		g_boxMesh = CreateRenderMesh(box);
//
//		delete sphere;
//		delete cylinder;
//		delete box;
//	}
//
//	if (!g_experiment)
//	{
//		// to ensure D3D context is active
//		StartGpuWork();
//
//		// init default scene
//		InitScene(g_sceneIndex);
//
//		// release context
//		EndGpuWork();
//		if (g_headless == true)
//		{
//			HeadlessMainLoop();
//		}
//		else
//		{
//			SDLMainLoop();
//		}
//	}
//	else
//	{
//		RunExperiments(g_experimentFilter);
//		exit(0);
//	}
//
//	if (g_render == true)
//	{
//		DestroyFluidRenderer(g_fluidRenderer);
//		DestroyFluidRenderBuffers(g_fluidRenderBuffers);
//		DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);
//
//		ShadowDestroy(g_shadowMap);
//	}
//
//    Shutdown();
//
//	if (g_headless == false)
//	{
//		DestroyRender();
//
//		SDL_DestroyWindow(g_window);
//		SDL_Quit();
//	}
//    return 0;
//}

// Flex Gym
#ifdef NV_FLEX_GYM
#include "../include/NvFlexGym.h"
#include "gym.h"
#endif


void pyflex_clean() {

    if (g_fluidRenderer)
        DestroyFluidRenderer(g_fluidRenderer);

    DestroyFluidRenderBuffers(g_fluidRenderBuffers);
    DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);

    ShadowDestroy(g_shadowMap);

    Shutdown();
    if (g_headless == false)
	{
		DestroyRender();

		SDL_DestroyWindow(g_window);
		SDL_Quit();
	}
}

void pyflex_step(py::array_t<float> update_params, int capture, char *path) {
    if (capture == 1) {
        g_capture = true;
        g_ffmpeg = fopen(path, "wb");
    }

    UpdateFrame(update_params);
    SDLMainLoop();

    if (capture == 1) {
        g_capture = false;
        fclose(g_ffmpeg);
        g_ffmpeg = nullptr;
    }
}

float rand_float(float LO, float HI) {
    return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

void pyflex_set_scene(int scene_idx, py::array_t<float> scene_params, int thread_idx = 0) {
    g_scene = scene_idx;
    g_selectedScene = g_scene;
    InitScene(g_selectedScene, scene_params, true, thread_idx);
}

void pyflex_MapShapeBuffers(SimBuffers *buffers) {
    buffers->shapeGeometry.map();
    buffers->shapePositions.map();
    buffers->shapeRotations.map();
    buffers->shapePrevPositions.map();
    buffers->shapePrevRotations.map();
    buffers->shapeFlags.map();
}

void pyflex_UnmapShapeBuffers(SimBuffers *buffers) {
    buffers->shapeGeometry.unmap();
    buffers->shapePositions.unmap();
    buffers->shapeRotations.unmap();
    buffers->shapePrevPositions.unmap();
    buffers->shapePrevRotations.unmap();
    buffers->shapeFlags.unmap();
}

void pyflex_add_capsule(py::array_t<float> params, py::array_t<float> lower_pos, py::array_t<float> quat_) {
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_params = (float *) params.request().ptr;
    float capsule_radius = ptr_params[0];
    float halfheight = ptr_params[1];

    auto ptr_lower_pos = (float *) lower_pos.request().ptr;
    Vec3 lower_position = Vec3(ptr_lower_pos[0], ptr_lower_pos[1], ptr_lower_pos[2]);

    auto ptr_quat = (float *) quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddCapsule(capsule_radius, halfheight, lower_position, quat);

    pyflex_UnmapShapeBuffers(g_buffers);
}


void pyflex_add_box(py::array_t<float> halfEdge_, py::array_t<float> center_, py::array_t<float> quat_) {
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_halfEdge = (float *) halfEdge_.request().ptr;
    Vec3 halfEdge = Vec3(ptr_halfEdge[0], ptr_halfEdge[1], ptr_halfEdge[2]);

    auto ptr_center = (float *) center_.request().ptr;
    Vec3 center = Vec3(ptr_center[0], ptr_center[1], ptr_center[2]);

    auto ptr_quat = (float *) quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddBox(halfEdge, center, quat);

    pyflex_UnmapShapeBuffers(g_buffers);
}

void pyflex_pop_box(int num) {
    pyflex_MapShapeBuffers(g_buffers);
    PopBox(num);
    pyflex_UnmapShapeBuffers(g_buffers);
}

void pyflex_add_sphere(float radius, py::array_t<float> position_, py::array_t<float> quat_) {
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_center = (float *) position_.request().ptr;
    Vec3 center = Vec3(ptr_center[0], ptr_center[1], ptr_center[2]);

    auto ptr_quat = (float *) quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddSphere(radius, center, quat);

    pyflex_UnmapShapeBuffers(g_buffers);
}

int pyflex_get_n_particles() {
    g_buffers->positions.map();
    int n_particles = g_buffers->positions.size();
    g_buffers->positions.unmap();
    return n_particles;
}

int pyflex_get_n_shapes() {
    g_buffers->shapePositions.map();
    int n_shapes = g_buffers->shapePositions.size();
    g_buffers->shapePositions.unmap();
    return n_shapes;
}

py::array_t<int> pyflex_get_groups() {
    g_buffers->phases.map();

    auto groups = py::array_t<int>((size_t) g_buffers->phases.size());
    auto ptr = (int *) groups.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->phases.size(); i++) {
        ptr[i] = g_buffers->phases[i] & 0xfffff; // Flex 1.1 manual actually says 24 bits while it is actually 20 bits
    }

    g_buffers->phases.unmap();

    return groups;
}

void pyflex_set_groups(py::array_t<int> groups) {
//    if (not set_color)
//        cout<<"Warning: Overloading GroupMask for colors. Make sure the eFlexPhaseSelfCollide is set!"<<endl;
    g_buffers->phases.map();

    auto buf = groups.request();
    auto ptr = (int *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->phases.size(); i++) {
        g_buffers->phases[i] = (g_buffers->phases[i] & ~0xfffff) | (ptr[i] & 0xfffff);
    }

    g_buffers->phases.unmap();

    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
}

py::array_t<int> pyflex_get_phases() {
    g_buffers->phases.map();

    auto phases = py::array_t<int>((size_t) g_buffers->phases.size());
    auto ptr = (int *) phases.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->phases.size(); i++) {
        ptr[i] = g_buffers->phases[i];
    }

    g_buffers->phases.unmap();

    return phases;
}


void pyflex_set_phases(py::array_t<int> phases) {
//    if (not set_color)
//        cout<<"Warning: Overloading GroupMask for colors. Make sure the eFlexPhaseSelfCollide is set!"<<endl;
    g_buffers->phases.map();

    auto buf = phases.request();
    auto ptr = (int *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->phases.size(); i++) {
        g_buffers->phases[i] = ptr[i];
    }

    g_buffers->phases.unmap();

    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
}

py::array_t<float> pyflex_get_positions() {
    g_buffers->positions.map();
    auto positions = py::array_t<float>((size_t) g_buffers->positions.size() * 4);
    auto ptr = (float *) positions.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->positions.size(); i++) {
        ptr[i * 4] = g_buffers->positions[i].x;
        ptr[i * 4 + 1] = g_buffers->positions[i].y;
        ptr[i * 4 + 2] = g_buffers->positions[i].z;
        ptr[i * 4 + 3] = g_buffers->positions[i].w;
    }

    g_buffers->positions.unmap();

    return positions;
}

void pyflex_set_positions(py::array_t<float> positions) {
    g_buffers->positions.map();

    auto buf = positions.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->positions.size(); i++) {
        g_buffers->positions[i].x = ptr[i * 4];
        g_buffers->positions[i].y = ptr[i * 4 + 1];
        g_buffers->positions[i].z = ptr[i * 4 + 2];
        g_buffers->positions[i].w = ptr[i * 4 + 3];
    }

    g_buffers->positions.unmap();

    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
}

void pyflex_add_rigid_body(py::array_t<float> positions, py::array_t<float> velocities, int num, py::array_t<float> lower) {
    auto bufp = positions.request();
    auto position_ptr = (float *) bufp.ptr;

    auto bufv = velocities.request();
    auto velocity_ptr = (float *) bufv.ptr;

    auto bufl = lower.request();
    auto lower_ptr = (float *) bufl.ptr;

    MapBuffers(g_buffers);

    // if (g_buffers->rigidIndices.empty())
	// 	g_buffers->rigidOffsets.push_back(0);

    int phase = NvFlexMakePhase(5, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);
    for (size_t i = 0; i < (size_t)num; i++) {
        g_buffers->activeIndices.push_back(int(g_buffers->activeIndices.size()));
        // g_buffers->rigidIndices.push_back(int(g_buffers->positions.size()));
        Vec3 position = Vec3(lower_ptr[0], lower_ptr[1], lower_ptr[2]) + Vec3(position_ptr[i*4], position_ptr[i*4+1], position_ptr[i*4+2]);
        g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, position_ptr[i*4 + 3]));
        Vec3 velocity = Vec3(velocity_ptr[i*3], velocity_ptr[i*3 + 1], velocity_ptr[i*3 + 2]);
        g_buffers->velocities.push_back(velocity);
        g_buffers->phases.push_back(phase);
    }

    // g_buffers->rigidCoefficients.push_back(1.0);
    // g_buffers->rigidOffsets.push_back(int(g_buffers->rigidIndices.size()));

    // g_buffers->activeIndices.resize(g_buffers->positions.size());
    // for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
    //     printf("active particle idx: %d %d \n", i, g_buffers->activeIndices[i]);

    // builds rigids constraints
    // if (g_buffers->rigidOffsets.size()) {
    //     assert(g_buffers->rigidOffsets.size() > 1);

    //     const int numRigids = g_buffers->rigidOffsets.size() - 1;

    //     // If the centers of mass for the rigids are not yet computed, this is done here
    //     // (If the CreateParticleShape method is used instead of the NvFlexExt methods, the centers of mass will be calculated here)
    //     if (g_buffers->rigidTranslations.size() == 0) {
    //         g_buffers->rigidTranslations.resize(g_buffers->rigidOffsets.size() - 1, Vec3());
    //         CalculateRigidCentersOfMass(&g_buffers->positions[0], g_buffers->positions.size(), &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids);
    //     }

    //     // calculate local rest space positions
    //     g_buffers->rigidLocalPositions.resize(g_buffers->rigidOffsets.back());
    //     CalculateRigidLocalPositions(&g_buffers->positions[0], &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids, &g_buffers->rigidLocalPositions[0]);

    //     // set rigidRotations to correct length, probably NULL up until here
    //     g_buffers->rigidRotations.resize(g_buffers->rigidOffsets.size() - 1, Quat());
    // }
    uint32_t numParticles = g_buffers->positions.size();

    UnmapBuffers(g_buffers);

    // reset pyflex solvers
    // NvFlexSetParams(g_solver, &g_params);
    // NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    // NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    // NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    // NvFlexSetNormals(g_solver, g_buffers->normals.buffer, nullptr);
    // NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, nullptr);

    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);
    // printf("ok till here\n");
    NvFlexSetActiveCount(g_solver, numParticles);
    // NvFlexSetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer,
    //     g_buffers->rigidLocalPositions.buffer, g_buffers->rigidLocalNormals.buffer,
    //     g_buffers->rigidCoefficients.buffer, g_buffers->rigidPlasticThresholds.buffer,
    //     g_buffers->rigidPlasticCreeps.buffer, g_buffers->rigidRotations.buffer,
    //     g_buffers->rigidTranslations.buffer, g_buffers->rigidOffsets.size() - 1, g_buffers->rigidIndices.size());
    // printf("also ok here\n");
}

py::array_t<float> pyflex_get_restPositions() {
    g_buffers->restPositions.map();

    auto restPositions = py::array_t<float>((size_t) g_buffers->restPositions.size() * 4);
    auto ptr = (float *) restPositions.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->restPositions.size(); i++) {
        ptr[i * 4] = g_buffers->restPositions[i].x;
        ptr[i * 4 + 1] = g_buffers->restPositions[i].y;
        ptr[i * 4 + 2] = g_buffers->restPositions[i].z;
        ptr[i * 4 + 3] = g_buffers->restPositions[i].w;
    }

    g_buffers->restPositions.unmap();

    return restPositions;
}

//py::array_t<int> pyflex_get_rigidOffsets() {
//    g_buffers->rigidOffsets.map();
//
//    auto rigidOffsets = py::array_t<int>((size_t) g_buffers->rigidOffsets.size());
//    auto ptr = (int *) rigidOffsets.request().ptr;
//
//    for (size_t i = 0; i < (size_t) g_buffers->rigidOffsets.size(); i++) {
//        ptr[i] = g_buffers->rigidOffsets[i];
//    }
//
//    g_buffers->rigidOffsets.unmap();
//
//    return rigidOffsets;
//}
//
//py::array_t<int> pyflex_get_rigidIndices() {
//    g_buffers->rigidIndices.map();
//
//    auto rigidIndices = py::array_t<int>((size_t) g_buffers->rigidIndices.size());
//    auto ptr = (int *) rigidIndices.request().ptr;
//
//    for (size_t i = 0; i < (size_t) g_buffers->rigidIndices.size(); i++) {
//        ptr[i] = g_buffers->rigidIndices[i];
//    }
//
//    g_buffers->rigidIndices.unmap();
//
//    return rigidIndices;
//}

//int pyflex_get_n_rigidPositions() {
//    g_buffers->rigidLocalPositions.map();
//    int n_rigidPositions = g_buffers->rigidLocalPositions.size();
//    g_buffers->rigidLocalPositions.unmap();
//    return n_rigidPositions;
//}
//
//py::array_t<float> pyflex_get_rigidLocalPositions() {
//    g_buffers->rigidLocalPositions.map();
//
//    auto rigidLocalPositions = py::array_t<float>((size_t) g_buffers->rigidLocalPositions.size() * 3);
//    auto ptr = (float *) rigidLocalPositions.request().ptr;
//
//    for (size_t i = 0; i < (size_t) g_buffers->rigidLocalPositions.size(); i++) {
//        ptr[i * 3] = g_buffers->rigidLocalPositions[i].x;
//        ptr[i * 3 + 1] = g_buffers->rigidLocalPositions[i].y;
//        ptr[i * 3 + 2] = g_buffers->rigidLocalPositions[i].z;
//    }
//
//    g_buffers->rigidLocalPositions.unmap();
//
//    return rigidLocalPositions;
//}
//
//py::array_t<float> pyflex_get_rigidGlobalPositions() {
//    g_buffers->rigidOffsets.map();
//    g_buffers->rigidIndices.map();
//    g_buffers->rigidLocalPositions.map();
//    g_buffers->rigidTranslations.map();
//    g_buffers->rigidRotations.map();
//
//    auto rigidGlobalPositions = py::array_t<float>((size_t) g_buffers->positions.size() * 3);
//    auto ptr = (float *) rigidGlobalPositions.request().ptr;
//
//    int count = 0;
//    int numRigids = g_buffers->rigidOffsets.size() - 1;
//    float n_clusters[g_buffers->positions.size()] = {0};
//
//    for (int i = 0; i < numRigids; i++) {
//        const int st = g_buffers->rigidOffsets[i];
//        const int ed = g_buffers->rigidOffsets[i + 1];
//
//        assert(ed - st);
//
//        for (int j = st; j < ed; j++) {
//            const int r = g_buffers->rigidIndices[j];
//            Vec3 p = Rotate(g_buffers->rigidRotations[i], g_buffers->rigidLocalPositions[count++]) +
//                     g_buffers->rigidTranslations[i];
//
//            if (n_clusters[r] == 0) {
//                ptr[r * 3] = p.x;
//                ptr[r * 3 + 1] = p.y;
//                ptr[r * 3 + 2] = p.z;
//            } else {
//                ptr[r * 3] += p.x;
//                ptr[r * 3 + 1] += p.y;
//                ptr[r * 3 + 2] += p.z;
//            }
//            n_clusters[r] += 1;
//        }
//    }
//
//    for (int i = 0; i < g_buffers->positions.size(); i++) {
//        if (n_clusters[i] > 0) {
//            ptr[i * 3] /= n_clusters[i];
//            ptr[i * 3 + 1] /= n_clusters[i];
//            ptr[i * 3 + 2] /= n_clusters[i];
//        }
//    }
//
//    g_buffers->rigidOffsets.unmap();
//    g_buffers->rigidIndices.unmap();
//    g_buffers->rigidLocalPositions.unmap();
//    g_buffers->rigidTranslations.unmap();
//    g_buffers->rigidRotations.unmap();
//
//    return rigidGlobalPositions;
//}
//
//int pyflex_get_n_rigids() {
//    g_buffers->rigidRotations.map();
//    int n_rigids = g_buffers->rigidRotations.size();
//    g_buffers->rigidRotations.unmap();
//    return n_rigids;
//}
//
//py::array_t<float> pyflex_get_rigidRotations() {
//    g_buffers->rigidRotations.map();
//
//    auto rigidRotations = py::array_t<float>((size_t) g_buffers->rigidRotations.size() * 4);
//    auto ptr = (float *) rigidRotations.request().ptr;
//
//    for (size_t i = 0; i < (size_t) g_buffers->rigidRotations.size(); i++) {
//        ptr[i * 4] = g_buffers->rigidRotations[i].x;
//        ptr[i * 4 + 1] = g_buffers->rigidRotations[i].y;
//        ptr[i * 4 + 2] = g_buffers->rigidRotations[i].z;
//        ptr[i * 4 + 3] = g_buffers->rigidRotations[i].w;
//    }
//
//    g_buffers->rigidRotations.unmap();
//
//    return rigidRotations;
//}
//
//py::array_t<float> pyflex_get_rigidTranslations() {
//    g_buffers->rigidTranslations.map();
//
//    auto rigidTranslations = py::array_t<float>((size_t) g_buffers->rigidTranslations.size() * 3);
//    auto ptr = (float *) rigidTranslations.request().ptr;
//
//    for (size_t i = 0; i < (size_t) g_buffers->rigidTranslations.size(); i++) {
//        ptr[i * 3] = g_buffers->rigidTranslations[i].x;
//        ptr[i * 3 + 1] = g_buffers->rigidTranslations[i].y;
//        ptr[i * 3 + 2] = g_buffers->rigidTranslations[i].z;
//    }
//
//    g_buffers->rigidTranslations.unmap();
//
//    return rigidTranslations;
//}

py::array_t<float> pyflex_get_velocities() {
    g_buffers->velocities.map();

    auto velocities = py::array_t<float>((size_t) g_buffers->velocities.size() * 3);
    auto ptr = (float *) velocities.request().ptr;

    for (size_t i = 0; i < (size_t) g_buffers->velocities.size(); i++) {
        ptr[i * 3] = g_buffers->velocities[i].x;
        ptr[i * 3 + 1] = g_buffers->velocities[i].y;
        ptr[i * 3 + 2] = g_buffers->velocities[i].z;
    }

    g_buffers->velocities.unmap();

    return velocities;
}

void pyflex_set_velocities(py::array_t<float> velocities) {
    g_buffers->velocities.map();

    auto buf = velocities.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->velocities.size(); i++) {
        g_buffers->velocities[i].x = ptr[i * 3];
        g_buffers->velocities[i].y = ptr[i * 3 + 1];
        g_buffers->velocities[i].z = ptr[i * 3 + 2];
    }

    g_buffers->velocities.unmap();
}

py::array_t<float> pyflex_get_shape_states() {
    pyflex_MapShapeBuffers(g_buffers);

    // position + prev_position + rotation + prev_rotation
    auto states = py::array_t<float>((size_t) g_buffers->shapePositions.size() * (3 + 3 + 4 + 4));
    auto buf = states.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->shapePositions.size(); i++) {
        ptr[i * 14] = g_buffers->shapePositions[i].x;
        ptr[i * 14 + 1] = g_buffers->shapePositions[i].y;
        ptr[i * 14 + 2] = g_buffers->shapePositions[i].z;

        ptr[i * 14 + 3] = g_buffers->shapePrevPositions[i].x;
        ptr[i * 14 + 4] = g_buffers->shapePrevPositions[i].y;
        ptr[i * 14 + 5] = g_buffers->shapePrevPositions[i].z;

        ptr[i * 14 + 6] = g_buffers->shapeRotations[i].x;
        ptr[i * 14 + 7] = g_buffers->shapeRotations[i].y;
        ptr[i * 14 + 8] = g_buffers->shapeRotations[i].z;
        ptr[i * 14 + 9] = g_buffers->shapeRotations[i].w;

        ptr[i * 14 + 10] = g_buffers->shapePrevRotations[i].x;
        ptr[i * 14 + 11] = g_buffers->shapePrevRotations[i].y;
        ptr[i * 14 + 12] = g_buffers->shapePrevRotations[i].z;
        ptr[i * 14 + 13] = g_buffers->shapePrevRotations[i].w;
    }

    pyflex_UnmapShapeBuffers(g_buffers);

    return states;
}

void pyflex_set_shape_states(py::array_t<float> states) {
    pyflex_MapShapeBuffers(g_buffers);

    auto buf = states.request();
    auto ptr = (float *) buf.ptr;

    for (size_t i = 0; i < (size_t) g_buffers->shapePositions.size(); i++) {
        g_buffers->shapePositions[i].x = ptr[i * 14];
        g_buffers->shapePositions[i].y = ptr[i * 14 + 1];
        g_buffers->shapePositions[i].z = ptr[i * 14 + 2];

        g_buffers->shapePrevPositions[i].x = ptr[i * 14 + 3];
        g_buffers->shapePrevPositions[i].y = ptr[i * 14 + 4];
        g_buffers->shapePrevPositions[i].z = ptr[i * 14 + 5];

        g_buffers->shapeRotations[i].x = ptr[i * 14 + 6];
        g_buffers->shapeRotations[i].y = ptr[i * 14 + 7];
        g_buffers->shapeRotations[i].z = ptr[i * 14 + 8];
        g_buffers->shapeRotations[i].w = ptr[i * 14 + 9];

        g_buffers->shapePrevRotations[i].x = ptr[i * 14 + 10];
        g_buffers->shapePrevRotations[i].y = ptr[i * 14 + 11];
        g_buffers->shapePrevRotations[i].z = ptr[i * 14 + 12];
        g_buffers->shapePrevRotations[i].w = ptr[i * 14 + 13];
    }

    UpdateShapes();

    pyflex_UnmapShapeBuffers(g_buffers);
}

// void pyflex_add_fluid_particle_grid(py::array_t<float> params) {
//     auto buf = params.request();
//     auto ptr = (float*) buf.ptr;

//     Vec3 lower = Vec3(ptr[0], ptr[1], ptr[2]);
//     int dimx = int(ptr[3]);
//     int dimy = int(ptr[4]);
//     int dimz = int(ptr[5]);
//     float radius = ptr[6];
//     Vec3 velocity = Vec3(0.0); // Vec3(ptr[7], ptr[8], ptr[9]);
//     float invMass = ptr[10];
//     float rigidStiffness = ptr[12];
//     int phase, float jitter=0.005f
// }

//py::array_t<float> pyflex_get_sceneParams() {
//    if (g_scene == 5) {
//        auto params = py::array_t<float>(3);
//        auto ptr = (float *) params.request().ptr;
//
//        ptr[0] = ((yz_SoftBody *) g_scenes[g_scene])->mInstances[0].mClusterStiffness;
//        ptr[1] = ((yz_SoftBody *) g_scenes[g_scene])->mInstances[0].mClusterPlasticThreshold * 1e4f;
//        ptr[2] = ((yz_SoftBody *) g_scenes[g_scene])->mInstances[0].mClusterPlasticCreep;
//
//        return params;
//    } else {
//        printf("Unsupprted scene_idx %d\n", g_scene);
//
//        auto params = py::array_t<float>(1);
//        auto ptr = (float *) params.request().ptr;
//        ptr[0] = 0.0f;
//
//        return params;
//    }
//}

py::array_t<float> pyflex_get_sceneUpper() {
    auto scene_upper = py::array_t<float>(3);
    auto buf = scene_upper.request();
    auto ptr = (float *) buf.ptr;

    ptr[0] = g_sceneUpper.x;
    ptr[1] = g_sceneUpper.y;
    ptr[2] = g_sceneUpper.z;

    return scene_upper;
}

py::array_t<float> pyflex_get_sceneLower() {
    auto scene_lower = py::array_t<float>(3);
    auto buf = scene_lower.request();
    auto ptr = (float *) buf.ptr;

    ptr[0] = g_sceneLower.x;
    ptr[1] = g_sceneLower.y;
    ptr[2] = g_sceneLower.z;

    return scene_lower;
}

py::array_t<int> pyflex_get_camera_params() {
    // Right now only returns width and height for the default screen camera
    // yf: add the camera position and camera angle
    auto default_camera_param = py::array_t<float>(8);
    auto default_camera_param_ptr = (float *) default_camera_param.request().ptr;
    default_camera_param_ptr[0] = g_screenWidth;
    default_camera_param_ptr[1] = g_screenHeight;
    default_camera_param_ptr[2] = g_camPos.x;
    default_camera_param_ptr[3] = g_camPos.y;
    default_camera_param_ptr[4] = g_camPos.z;
    default_camera_param_ptr[5] = g_camAngle.x;
    default_camera_param_ptr[6] = g_camAngle.y;
    default_camera_param_ptr[7] = g_camAngle.z;
    return default_camera_param;
}

void pyflex_set_camera_params(py::array_t<float> update_camera_param) {
    auto camera_param_ptr = (float *) update_camera_param.request().ptr;
    if (g_render){
        g_camPos.x = camera_param_ptr[0];
        g_camPos.y = camera_param_ptr[1];
        g_camPos.z = camera_param_ptr[2];
        g_camAngle.x =  camera_param_ptr[3];
        g_camAngle.y =  camera_param_ptr[4];
        g_camAngle.z =  camera_param_ptr[5];
        g_screenWidth = camera_param_ptr[6];
        g_screenHeight = camera_param_ptr[7];}
}

py::array_t<int> pyflex_render(int capture, char *path) {
    // TODO: Turn off the GUI menu for rendering
    static double lastTime;

    // real elapsed frame time
    double frameBeginTime = GetSeconds();

    g_realdt = float(frameBeginTime - lastTime);
    lastTime = frameBeginTime;

    if (capture == 1) {
        g_capture = true;
        g_ffmpeg = fopen(path, "wb");
    }

    //-------------------------------------------------------------------
    // Scene Update

    double waitBeginTime = GetSeconds();

    MapBuffers(g_buffers);

    double waitEndTime = GetSeconds();

    // Getting timers causes CPU/GPU sync, so we do it after a map
    float newSimLatency = NvFlexGetDeviceLatency(g_solver, &g_GpuTimers.computeBegin, &g_GpuTimers.computeEnd,
                                                 &g_GpuTimers.computeFreq);
    float newGfxLatency = RendererGetDeviceTimestamps(&g_GpuTimers.renderBegin, &g_GpuTimers.renderEnd,
                                                      &g_GpuTimers.renderFreq);
    (void) newGfxLatency;

    UpdateCamera();

    if (!g_pause || g_step) {
        UpdateEmitters();
        UpdateMouse();
        UpdateWind();
        // UpdateScene();
    }

    //-------------------------------------------------------------------
    // Render

    double renderBeginTime = GetSeconds();

    if (g_profile && (!g_pause || g_step)) {
        if (g_benchmark) {
            g_numDetailTimers = NvFlexGetDetailTimers(g_solver, &g_detailTimers);
        } else {
            memset(&g_timers, 0, sizeof(g_timers));
            NvFlexGetTimers(g_solver, &g_timers);
        }
    }

    StartFrame(Vec4(g_clearColor, 1.0f));

    // main scene render
    RenderScene();
    RenderDebug();

    int newScene = DoUI();

    EndFrame();

    // If user has disabled async compute, ensure that no compute can overlap
    // graphics by placing a sync between them
    if (!g_useAsyncCompute)
        NvFlexComputeWaitForGraphics(g_flexLib);

    UnmapBuffers(g_buffers);

    // move mouse particle (must be done here as GetViewRay() uses the GL projection state)
    if (g_mouseParticle != -1) {
        Vec3 origin, dir;
        GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

        g_mousePos = origin + dir * g_mouseT;
    }

    // Original function for rendering and saving to disk
    if (g_capture) {
        TgaImage img;
        img.m_width = g_screenWidth;
        img.m_height = g_screenHeight;
        img.m_data = new uint32_t[g_screenWidth*g_screenHeight];

        ReadFrame((int*)img.m_data, g_screenWidth, g_screenHeight);
        TgaSave(g_ffmpeg, img, false);

        // fwrite(img.m_data, sizeof(uint32_t)*g_screenWidth*g_screenHeight, 1, g_ffmpeg);

        delete[] img.m_data;
    }

//    auto rendered_img = py::array_t<uint32_t>((uint32_t) g_screenWidth*g_screenHeight);
    auto rendered_img = py::array_t<uint8_t>((int) g_screenWidth * g_screenHeight * 4);
    auto rendered_img_ptr = (uint8_t *) rendered_img.request().ptr;

    int rendered_img_int32_ptr[g_screenWidth * g_screenHeight];
    ReadFrame(rendered_img_int32_ptr, g_screenWidth, g_screenHeight);

    for (int i = 0; i < g_screenWidth * g_screenHeight; ++i) {
        int32_abgr_to_int8_rgba((uint32_t) rendered_img_int32_ptr[i],
                                rendered_img_ptr[4 * i],
                                rendered_img_ptr[4 * i + 1],
                                rendered_img_ptr[4 * i + 2],
                                rendered_img_ptr[4 * i + 3]);
    }
    // Should be able to return the image here, instead of at the end

//    delete[] img.m_data;

    double renderEndTime = GetSeconds();

    // if user requested a scene reset process it now
    if (g_resetScene) {
        // Reset();
        g_resetScene = false;
    }

    //-------------------------------------------------------------------
    // Flex Update

    double updateBeginTime = GetSeconds();

    // send any particle updates to the solver
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);

    NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());

    if (!g_pause || g_step) {
        // tick solver
        // NvFlexSetParams(g_solver, &g_params);
        // NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

        g_frame++;
        g_step = false;
    }

    // read back base particle data
    // Note that flexGet calls don't wait for the GPU, they just queue a GPU copy
    // to be executed later.
    // When we're ready to read the fetched buffers we'll Map them, and that's when
    // the CPU will wait for the GPU flex update and GPU copy to finish.
    NvFlexGetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexGetNormals(g_solver, g_buffers->normals.buffer, nullptr);

    // readback rigid transforms
    // if (g_buffers->rigidOffsets.size())
    //    NvFlexGetRigids(g_solver, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer);

    double updateEndTime = GetSeconds();

    //-------------------------------------------------------
    // Update the on-screen timers

    auto newUpdateTime = float(updateEndTime - updateBeginTime);
    auto newRenderTime = float(renderEndTime - renderBeginTime);
    auto newWaitTime = float(waitBeginTime - waitEndTime);

    // Exponential filter to make the display easier to read
    const float timerSmoothing = 0.05f;

    g_updateTime = (g_updateTime == 0.0f) ? newUpdateTime : Lerp(g_updateTime, newUpdateTime, timerSmoothing);
    g_renderTime = (g_renderTime == 0.0f) ? newRenderTime : Lerp(g_renderTime, newRenderTime, timerSmoothing);
    g_waitTime = (g_waitTime == 0.0f) ? newWaitTime : Lerp(g_waitTime, newWaitTime, timerSmoothing);
    g_simLatency = (g_simLatency == 0.0f) ? newSimLatency : Lerp(g_simLatency, newSimLatency, timerSmoothing);

    if (g_benchmark) newScene = BenchmarkUpdate();

    // flush out the last frame before freeing up resources in the event of a scene change
    // this is necessary for d3d12
    PresentFrame(g_vsync);

    // if gui or benchmark requested a scene change process it now
    if (newScene != -1) {
        g_scene = newScene;
        // Init(g_scene);
    }

    SDLMainLoop();

    if (capture == 1) {
        g_capture = false;
        fclose(g_ffmpeg);
        g_ffmpeg = nullptr;
    }

    return rendered_img;
}

int main() {
    cout<<"PyFlexRobotics loaded" <<endl;
    pyflex_init();
    pyflex_clean();

    return 0;
}

PYBIND11_MODULE(pyflex, m) {
//    m.def("main", &main);
    m.def("init_debug", &pyflex_init_debug);
//    m.def("init", &pyflex_init);
//    m.def("set_scene", &pyflex_set_scene);
//    m.def("clean", &pyflex_clean);
//    m.def("step", &pyflex_step,
//          py::arg("update_params") = nullptr,
//          py::arg("capture") = 0,
//          py::arg("path") = nullptr);
//
//    m.def("render", &pyflex_render,
//          py::arg("capture") = 0,
//          py::arg("path") = nullptr
//        );
//
//    m.def("get_camera_params", &pyflex_get_camera_params, "Get camera parameters");
//    m.def("set_camera_params", &pyflex_set_camera_params, "Set camera parameters");
//
//    m.def("add_box", &pyflex_add_box, "Add box to the scene");
//    m.def("add_sphere", &pyflex_add_sphere, "Add sphere to the scene");
//    m.def("add_capsule", &pyflex_add_capsule, "Add capsule to the scene");
//
//    m.def("pop_box", &pyflex_pop_box, "remove box from the scene");
//
//    m.def("get_n_particles", &pyflex_get_n_particles, "Get the number of particles");
//    m.def("get_n_shapes", &pyflex_get_n_shapes, "Get the number of shapes");
////    m.def("get_n_rigids", &pyflex_get_n_rigids, "Get the number of rigids");
////    m.def("get_n_rigidPositions", &pyflex_get_n_rigidPositions, "Get the number of rigid positions");
//
//    m.def("get_phases", &pyflex_get_phases, "Get particle phases");
//    m.def("set_phases", &pyflex_set_phases, "Set particle phases");
//    m.def("get_groups", &pyflex_get_groups, "Get particle groups");
//    m.def("set_groups", &pyflex_set_groups, "Set particle groups");
//    // TODO: Add keyword set_color for set_phases function and also in python code
//    m.def("get_positions", &pyflex_get_positions, "Get particle positions");
//    m.def("set_positions", &pyflex_set_positions, "Set particle positions");
//    m.def("get_restPositions", &pyflex_get_restPositions, "Get particle restPositions");
////    m.def("get_rigidOffsets", &pyflex_get_rigidOffsets, "Get rigid offsets");
////    m.def("get_rigidIndices", &pyflex_get_rigidIndices, "Get rigid indices");
////    m.def("get_rigidLocalPositions", &pyflex_get_rigidLocalPositions, "Get rigid local positions");
////    m.def("get_rigidGlobalPositions", &pyflex_get_rigidGlobalPositions, "Get rigid global positions");
////    m.def("get_rigidRotations", &pyflex_get_rigidRotations, "Get rigid rotations");
////    m.def("get_rigidTranslations", &pyflex_get_rigidTranslations, "Get rigid translations");
//
////    m.def("get_sceneParams", &pyflex_get_sceneParams, "Get scene parameters");
//
//    m.def("get_velocities", &pyflex_get_velocities, "Get particle velocities");
//    m.def("set_velocities", &pyflex_set_velocities, "Set particle velocities");
//
//    m.def("get_shape_states", &pyflex_get_shape_states, "Get shape states");
//    m.def("set_shape_states", &pyflex_set_shape_states, "Set shape states");
//    m.def("clear_shapes", &ClearShapes, "Clear shapes");
//
//    m.def("get_scene_upper", &pyflex_get_sceneUpper);
//    m.def("get_scene_lower", &pyflex_get_sceneLower);
//
//    m.def("add_rigid_body", &pyflex_add_rigid_body);
}