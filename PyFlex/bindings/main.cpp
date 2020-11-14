#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "include/NvFlex.h"
#include "include/NvFlexExt.h"
#include "include/NvFlexDevice.h"

#include "core/maths.h"
#include "core/types.h"
#include "core/platform.h"
#include "core/mesh.h"
#include "core/voxelize.h"
#include "core/sdf.h"
#include "core/pfm.h"
#include "core/tga.h"
#include "core/perlin.h"
#include "core/convex.h"
#include "core/cloth.h"

#include "external/SDL2-2.0.4/include/SDL.h"

#include "shaders.h"
#include "imgui.h"
#include "shadersDemoContext.h"

#include "bindings/utils/utils.h"


void InitRenderHeadless(const RenderInitOptions& options, int width, int height);

SDL_Window *g_window;           // window handle
unsigned int g_windowId;        // window id

#define SDL_CONTROLLER_BUTTON_LEFT_TRIGGER (SDL_CONTROLLER_BUTTON_MAX + 1)
#define SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER (SDL_CONTROLLER_BUTTON_MAX + 2)

int GetKeyFromGameControllerButton(SDL_GameControllerButton button) {
    switch (button) {
        case SDL_CONTROLLER_BUTTON_DPAD_UP: {
            return SDLK_q;
        }    // -- camera translate up
        case SDL_CONTROLLER_BUTTON_DPAD_DOWN: {
            return SDLK_z;
        }    // -- camera translate down
        case SDL_CONTROLLER_BUTTON_DPAD_LEFT: {
            return SDLK_h;
        }    // -- hide GUI
        case SDL_CONTROLLER_BUTTON_DPAD_RIGHT: {
            return -1;
        }    // -- unassigned
        case SDL_CONTROLLER_BUTTON_START: {
            return SDLK_RETURN;
        }   // -- start selected scene
        case SDL_CONTROLLER_BUTTON_BACK: {
            return SDLK_ESCAPE;
        }   // -- quit
        case SDL_CONTROLLER_BUTTON_LEFTSHOULDER: {
            return SDLK_UP;
        }   // -- select prev scene
        case SDL_CONTROLLER_BUTTON_RIGHTSHOULDER: {
            return SDLK_DOWN;
        }     // -- select next scene
        case SDL_CONTROLLER_BUTTON_A: {
            return SDLK_g;
        }    // -- toggle gravity
        case SDL_CONTROLLER_BUTTON_B: {
            return SDLK_p;
        }    // -- pause
        case SDL_CONTROLLER_BUTTON_X: {
            return SDLK_r;
        }    // -- reset
        case SDL_CONTROLLER_BUTTON_Y: {
            return SDLK_o;
        }    // -- step sim
        case SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER: {
            return SDLK_SPACE;
        }    // -- emit particles
        default: {
            return -1;
        }    // -- nop
    };
};

//
// Gamepad thresholds taken from XINPUT API
//
#define XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE  7849
#define XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE 8689
#define XINPUT_GAMEPAD_TRIGGER_THRESHOLD    30

int deadzones[3] = {
        XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
        XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE,
        XINPUT_GAMEPAD_TRIGGER_THRESHOLD};

inline float joyAxisFilter(int value, int stick) {
    //clamp values in deadzone to zero, and remap rest of range so that it linearly rises in value from edge of deadzone toward max value.
    if (value < -deadzones[stick])
        return (value + deadzones[stick]) / (32768.0f - deadzones[stick]);
    else if (value > deadzones[stick])
        return (value - deadzones[stick]) / (32768.0f - deadzones[stick]);
    else
        return 0.0f;
}

SDL_GameController *g_gamecontroller = nullptr;

using namespace std;

int g_screenWidth = 720;
int g_screenHeight = 720;
int g_msaaSamples = 8;

int g_numSubsteps;

// a setting of -1 means Flex will use the device specified in the NVIDIA control panel
int g_device = -1;
char g_deviceName[256];
bool g_vsync = true;

// these two are migrated from Flex 2.0
bool g_headless = true;
bool g_render = true;

bool g_benchmark = false;
bool g_extensions = true;
bool g_teamCity = false;
bool g_interop = true;
bool g_d3d12 = false;
bool g_useAsyncCompute = true;
bool g_increaseGfxLoadForAsyncComputeTesting = false;
int g_graphics = 0; // 0=ogl, 1=DX11, 2=DX12

FluidRenderer *g_fluidRenderer;
FluidRenderBuffers *g_fluidRenderBuffers;
DiffuseRenderBuffers *g_diffuseRenderBuffers;

NvFlexSolver *g_solver;
NvFlexSolverDesc g_solverDesc;
NvFlexLibrary *g_flexLib;
NvFlexParams g_params;
NvFlexTimers g_timers;
int g_numDetailTimers;
NvFlexDetailTimer *g_detailTimers;

int g_maxDiffuseParticles;
int g_maxNeighborsPerParticle;
int g_numExtraParticles;
int g_numExtraMultiplier = 1;
int g_maxContactsPerParticle;

// mesh used for deformable object rendering
Mesh *g_mesh;
vector<int> g_meshSkinIndices;
vector<float> g_meshSkinWeights;
vector<Point3> g_meshRestPositions;
const int g_numSkinWeights = 4;

// mapping of collision mesh to render mesh
std::map<NvFlexConvexMeshId, GpuMesh *> g_convexes;
std::map<NvFlexTriangleMeshId, GpuMesh *> g_meshes;
std::map<NvFlexDistanceFieldId, GpuMesh *> g_fields;

// flag to request collision shapes be updated
bool g_shapesChanged = false;

/* Note that this array of colors is altered by demo code, and is also read from global by graphics API impls */
Colour g_colors[] = {
        Colour(0.000f, 0.500f, 1.000f),
        Colour(0.875f, 0.782f, 0.051f),
        Colour(0.800f, 0.100f, 0.100f),
        Colour(0.673f, 0.111f, 0.000f),
        Colour(0.612f, 0.194f, 0.394f),
        Colour(0.0f, 1.f, 0.0f),
        Colour(0.797f, 0.354f, 0.000f),
        Colour(0.092f, 0.465f, 0.820f)
};

//Colour g_colors[] = {
//        Colour(0.0f, 0.0f, 0.0f),
//        Colour(0.05f, 0.05f, 0.05f),
//        Colour(0.1f, 0.1f, 0.1f),
//        Colour(0.15f, 0.15f, 0.15f),
//        Colour(0.2f, 0.2f, 0.2f),
//        Colour(0.25f, 0.25f, 0.25f),
//        Colour(0.3f, 0.3f, 0.3f),
//        Colour(0.35f, 0.35f, 0.35f),
//        Colour(0.4f, 0.4f, 0.4f),
//        Colour(0.45f, 0.45f, 0.45f),
//        Colour(0.5f, 0.5f, 0.5f),
//        Colour(0.55f, 0.55f, 0.55f),
//        Colour(0.6f, 0.6f, 0.6f),
//        Colour(0.65f, 0.65f, 0.65f),
//        Colour(0.7f, 0.7f, 0.7f),
//        Colour(0.75f, 0.75f, 0.75f),
//        Colour(0.8f, 0.8f, 0.8f),
//        Colour(0.85f, 0.85f, 0.85f),
//        Colour(0.9f, 0.9f, 0.9f),
//        Colour(0.95f, 0.95f, 0.95f),
//};

struct SimBuffers {
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

    // convexes
    NvFlexVector<NvFlexCollisionGeometry> shapeGeometry;
    NvFlexVector<Vec4> shapePositions;
    NvFlexVector<Quat> shapeRotations;
    NvFlexVector<Vec4> shapePrevPositions;
    NvFlexVector<Quat> shapePrevRotations;
    NvFlexVector<int> shapeFlags;

    // rigids
    NvFlexVector<int> rigidOffsets;
    NvFlexVector<int> rigidIndices;
    NvFlexVector<int> rigidMeshSize;
    NvFlexVector<float> rigidCoefficients;
    NvFlexVector<float> rigidPlasticThresholds;
    NvFlexVector<float> rigidPlasticCreeps;
    NvFlexVector<Quat> rigidRotations;
    NvFlexVector<Vec3> rigidTranslations;
    NvFlexVector<Vec3> rigidLocalPositions;
    NvFlexVector<Vec4> rigidLocalNormals;

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

    NvFlexVector<int> triangles;
    NvFlexVector<Vec3> triangleNormals;
    NvFlexVector<Vec3> uvs;

    SimBuffers(NvFlexLibrary *l) :
            positions(l), restPositions(l), velocities(l), phases(l), densities(l),
            anisotropy1(l), anisotropy2(l), anisotropy3(l), normals(l), smoothPositions(l),
            diffusePositions(l), diffuseVelocities(l), diffuseCount(l), activeIndices(l),
            shapeGeometry(l), shapePositions(l), shapeRotations(l), shapePrevPositions(l),
            shapePrevRotations(l), shapeFlags(l), rigidOffsets(l), rigidIndices(l), rigidMeshSize(l),
            rigidCoefficients(l), rigidPlasticThresholds(l), rigidPlasticCreeps(l), rigidRotations(l),
            rigidTranslations(l),
            rigidLocalPositions(l), rigidLocalNormals(l), inflatableTriOffsets(l),
            inflatableTriCounts(l), inflatableVolumes(l), inflatableCoefficients(l),
            inflatablePressures(l), springIndices(l), springLengths(l),
            springStiffness(l), triangles(l), triangleNormals(l), uvs(l) {}
};

SimBuffers *g_buffers;

void MapBuffers(SimBuffers *buffers) {
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

    // convexes
    buffers->shapeGeometry.map();
    buffers->shapePositions.map();
    buffers->shapeRotations.map();
    buffers->shapePrevPositions.map();
    buffers->shapePrevRotations.map();
    buffers->shapeFlags.map();

    buffers->rigidOffsets.map();
    buffers->rigidIndices.map();
    buffers->rigidMeshSize.map();
    buffers->rigidCoefficients.map();
    buffers->rigidPlasticThresholds.map();
    buffers->rigidPlasticCreeps.map();
    buffers->rigidRotations.map();
    buffers->rigidTranslations.map();
    buffers->rigidLocalPositions.map();
    buffers->rigidLocalNormals.map();

    buffers->springIndices.map();
    buffers->springLengths.map();
    buffers->springStiffness.map();

    // inflatables
    buffers->inflatableTriOffsets.map();
    buffers->inflatableTriCounts.map();
    buffers->inflatableVolumes.map();
    buffers->inflatableCoefficients.map();
    buffers->inflatablePressures.map();

    buffers->triangles.map();
    buffers->triangleNormals.map();
    buffers->uvs.map();
}

void UnmapBuffers(SimBuffers *buffers) {
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
    buffers->rigidOffsets.unmap();
    buffers->rigidIndices.unmap();
    buffers->rigidMeshSize.unmap();
    buffers->rigidCoefficients.unmap();
    buffers->rigidPlasticThresholds.unmap();
    buffers->rigidPlasticCreeps.unmap();
    buffers->rigidRotations.unmap();
    buffers->rigidTranslations.unmap();
    buffers->rigidLocalPositions.unmap();
    buffers->rigidLocalNormals.unmap();

    // springs
    buffers->springIndices.unmap();
    buffers->springLengths.unmap();
    buffers->springStiffness.unmap();

    // inflatables
    buffers->inflatableTriOffsets.unmap();
    buffers->inflatableTriCounts.unmap();
    buffers->inflatableVolumes.unmap();
    buffers->inflatableCoefficients.unmap();
    buffers->inflatablePressures.unmap();

    // triangles
    buffers->triangles.unmap();
    buffers->triangleNormals.unmap();
    buffers->uvs.unmap();

}

SimBuffers *AllocBuffers(NvFlexLibrary *lib) {
    return new SimBuffers(lib);
}

void DestroyBuffers(SimBuffers *buffers) {
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

    // rigids
    buffers->rigidOffsets.destroy();
    buffers->rigidIndices.destroy();
    buffers->rigidMeshSize.destroy();
    buffers->rigidCoefficients.destroy();
    buffers->rigidPlasticThresholds.destroy();
    buffers->rigidPlasticCreeps.destroy();
    buffers->rigidRotations.destroy();
    buffers->rigidTranslations.destroy();
    buffers->rigidLocalPositions.destroy();
    buffers->rigidLocalNormals.destroy();

    // springs
    buffers->springIndices.destroy();
    buffers->springLengths.destroy();
    buffers->springStiffness.destroy();

    // inflatables
    buffers->inflatableTriOffsets.destroy();
    buffers->inflatableTriCounts.destroy();
    buffers->inflatableVolumes.destroy();
    buffers->inflatableCoefficients.destroy();
    buffers->inflatablePressures.destroy();

    // triangles
    buffers->triangles.destroy();
    buffers->triangleNormals.destroy();
    buffers->uvs.destroy();

    delete buffers;
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

bool g_pause = false;
bool g_step = false;
bool g_capture = false;
bool g_showHelp = true;
bool g_tweakPanel = false;
bool g_fullscreen = false;
bool g_wireframe = false;
bool g_debug = false;

bool g_emit = false;
bool g_warmup = false;

float g_windTime = 0.0f;
float g_windFrequency = 0.0f;
float g_windStrength = 0.0f;

bool g_wavePool = false;
float g_waveTime = 0.0f;
float g_wavePlane;
float g_waveFrequency = 1.5f;
float g_waveAmplitude = 1.0f;
float g_waveFloorTilt = 0.0f;

Vec3 g_shape_color=Vec3(0.9);
Vec3 g_sceneLower;
Vec3 g_sceneUpper;

float g_blur;
float g_ior;
bool g_drawEllipsoids;
bool g_drawPoints;
bool g_drawMesh;
bool g_drawCloth;
float g_expandCloth;    // amount to expand cloth along normal (to account for particle radius)

bool g_drawOpaque;
int g_drawSprings;        // 0: no draw, 1: draw stretch 2: draw tether
bool g_drawBases = false;
bool g_drawContacts = false;
bool g_drawNormals = false;
bool g_drawDiffuse;
bool g_drawShapeGrid = false;
bool g_drawDensity = false;
bool g_drawRopes;
float g_pointScale;
float g_ropeScale;
float g_drawPlaneBias;    // move planes along their normal for rendering

float g_diffuseScale;
float g_diffuseMotionScale;
bool g_diffuseShadow;
float g_diffuseInscatter;
float g_diffuseOutscatter;

float g_dt = 1.0f / 240.0f;    // the time delta used for simulation
float g_realdt;               // the real world time delta between updates

float g_waitTime;       // the CPU time spent waiting for the GPU
float g_updateTime;     // the CPU time spent on Flex
float g_renderTime;     // the CPU time spent calling OpenGL to render the scene
// the above times don't include waiting for vsync
float g_simLatency;     // the time the GPU spent between the first and last NvFlexUpdateSolver() operation. Because some GPUs context switch, this can include graphics time.

int g_scene = 0;
int g_selectedScene = g_scene;
int g_levelScroll;          // offset for level selection scroll area
bool g_resetScene = false;  //if the user clicks the reset button or presses the reset key this is set to true;

int g_frame = 0;
int g_numSolidParticles = 0;

int g_mouseParticle = -1;
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

ShadowMap *g_shadowMap;

Vec4 g_fluidColor;
Vec4 g_diffuseColor;
Vec3 g_meshColor;
Vec3 g_clearColor;
float g_lightDistance;
float g_fogDistance;

FILE *g_ffmpeg;

void DrawShapes();

class Scene;

vector<Scene *> g_scenes;

struct Emitter {
    Emitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f), mWidth(8) {}

    Vec3 mPos;
    Vec3 mDir;
    Vec3 mRight;
    float mSpeed;
    bool mEnabled;
    float mLeftOver;
    int mWidth;
};

vector<Emitter> g_emitters(1);    // first emitter is the camera 'gun'

struct Rope {
    std::vector<int> mIndices;
};

vector<Rope> g_ropes;

inline float sqr(float x) { return x * x; }

#include "helpers.h"
#include "scenes.h"
#include "benchmark.h"

#include <iostream>
using namespace std;

void Init(int scene, py::array_t<float> scene_params, bool centerCamera = true, int thread_idx = 0) {
    RandInit();
    if (g_solver) {
        if (g_buffers)
            DestroyBuffers(g_buffers);

        if (g_render) {
            DestroyFluidRenderBuffers(g_fluidRenderBuffers);
            DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);
        }

        for (auto &iter : g_meshes) {
            NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        // std::cout << "mesh destroyed" << endl;

        for (auto &iter : g_fields) {
            NvFlexDestroyDistanceField(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        for (auto &iter : g_convexes) {
            NvFlexDestroyConvexMesh(g_flexLib, iter.first);
            DestroyGpuMesh(iter.second);
        }

        g_fields.clear();
        g_meshes.clear();
        g_convexes.clear();

        NvFlexDestroySolver(g_solver);
        g_solver = nullptr;
    }

    // alloc buffers
    g_buffers = AllocBuffers(g_flexLib);

    // map during initialization
    MapBuffers(g_buffers);

    // std::cout << "buffers mapped" << endl;


    g_buffers->positions.resize(0);
    g_buffers->velocities.resize(0);
    g_buffers->phases.resize(0);

    g_buffers->rigidOffsets.resize(0);
    g_buffers->rigidIndices.resize(0);
    g_buffers->rigidMeshSize.resize(0);
    g_buffers->rigidRotations.resize(0);
    g_buffers->rigidTranslations.resize(0);
    g_buffers->rigidCoefficients.resize(0);
    g_buffers->rigidPlasticThresholds.resize(0);
    g_buffers->rigidPlasticCreeps.resize(0);
    g_buffers->rigidLocalPositions.resize(0);
    g_buffers->rigidLocalNormals.resize(0);

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
    g_fluidColor = Vec4(0.1f, 0.4f, 0.8f, 1.0f); // we can change fluid color here
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
    g_ior = 1.0f;
    g_lightDistance = 10.0f;
    g_fogDistance = 0.005f;

    g_camSpeed = 0.075f;
    g_camNear = 0.01f;
    g_camFar = 1000.0f;

    g_pointScale = 1.0f;
    g_ropeScale = 1.0f;
    g_drawPlaneBias = 0.0f;

    // sim params
    g_params.gravity[0] = 0.0f;
    g_params.gravity[1] = -9.8f;
    g_params.gravity[2] = 0.0f;

    g_params.wind[0] = 0.0f;
    g_params.wind[1] = 0.0f;
    g_params.wind[2] = 0.0f;

    g_params.radius = 0.15f;
    g_params.viscosity = 0.0f;
    g_params.dynamicFriction = 0.0f;
    g_params.staticFriction = 0.0f;
    g_params.particleFriction = 0.0f; // scale friction between particles by default
    g_params.freeSurfaceDrag = 0.0f;
    g_params.drag = 0.0f;
    g_params.lift = 0.0f;
    g_params.numIterations = 3;
    g_params.fluidRestDistance = 0.0f;
    g_params.solidRestDistance = 0.0f;

    g_params.anisotropyScale = 1.0f;
    g_params.anisotropyMin = 0.1f;
    g_params.anisotropyMax = 2.0f;
    g_params.smoothing = 1.0f;

    g_params.dissipation = 0.0f;
    g_params.damping = 0.0f;
    g_params.particleCollisionMargin = 0.0f;
    g_params.shapeCollisionMargin = 0.0f;
    g_params.collisionDistance = 0.0f;
    g_params.sleepThreshold = 0.0f;
    g_params.shockPropagation = 0.0f;
    g_params.restitution = 0.0f;

    g_params.maxSpeed = FLT_MAX;
    g_params.maxAcceleration = 100.0f;    // approximately 10x gravity

    g_params.relaxationMode = eNvFlexRelaxationLocal;
    g_params.relaxationFactor = 1.0f;
    g_params.solidPressure = 1.0f;
    g_params.adhesion = 0.0f;
    g_params.cohesion = 0.025f;
    g_params.surfaceTension = 0.0f;
    g_params.vorticityConfinement = 0.0f;
    g_params.buoyancy = 1.0f;
    g_params.diffuseThreshold = 100.0f;
    g_params.diffuseBuoyancy = 1.0f;
    g_params.diffuseDrag = 0.8f;
    g_params.diffuseBallistic = 16;
    g_params.diffuseLifetime = 2.0f;

    g_numSubsteps = 20;

    // planes created after particles
    g_params.numPlanes = 1;

    g_diffuseScale = 0.5f;
    g_diffuseColor = 1.0f;
    g_diffuseMotionScale = 1.0f;
    g_diffuseShadow = false;
    g_diffuseInscatter = 0.8f;
    g_diffuseOutscatter = 0.53f;

    // reset phase 0 particle color to blue
//    g_colors[0] = Colour(0.0f, 0.5f, 1.0f);

    g_numSolidParticles = 0;

    g_waveFrequency = 1.5f;
    g_waveAmplitude = 1.5f;
    g_waveFloorTilt = 0.0f;
    g_emit = false;
    g_warmup = false;

    g_mouseParticle = -1;

    g_maxDiffuseParticles = 0;    // number of diffuse particles
    g_maxNeighborsPerParticle = 96;
    g_numExtraParticles = 0;    // number of particles allocated but not made active
    g_maxContactsPerParticle = 6;

    g_sceneLower = FLT_MAX;
    g_sceneUpper = -FLT_MAX;

    // initialize solver desc
    NvFlexSetSolverDescDefaults(&g_solverDesc);
    // printf("sovler initialized\n");

    // create scene
    StartGpuWork();
    // printf("Gpu started. \n");
//    cout<<thread_idx<<endl;
    g_scenes[g_scene]->Initialize(scene_params, thread_idx);
    EndGpuWork();

    uint32_t numParticles = g_buffers->positions.size();
    uint32_t maxParticles = numParticles + g_numExtraParticles * g_numExtraMultiplier;

    if (g_params.solidRestDistance == 0.0f)
        g_params.solidRestDistance = g_params.radius;

    // if fluid present then we assume solid particles have the same radius
    if (g_params.fluidRestDistance > 0.0f)
        g_params.solidRestDistance = g_params.fluidRestDistance;

    // set collision distance automatically based on rest distance if not already set
    if (g_params.collisionDistance == 0.0f)
        g_params.collisionDistance = Max(g_params.solidRestDistance, g_params.fluidRestDistance) * 0.5f;

    // default particle friction to 10% of shape friction
    if (g_params.particleFriction == 0.0f)
        g_params.particleFriction = g_params.dynamicFriction * 0.1f;

    // add a margin for detecting contacts between particles and shapes
    if (g_params.shapeCollisionMargin == 0.0f)
        g_params.shapeCollisionMargin = g_params.collisionDistance * 0.5f;

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

    (Vec4 &) g_params.planes[0] = Vec4(up.x, up.y, up.z, 0.0f);
    (Vec4 &) g_params.planes[1] = Vec4(0.0f, 0.0f, 1.0f, -g_sceneLower.z);
    (Vec4 &) g_params.planes[2] = Vec4(1.0f, 0.0f, 0.0f, -g_sceneLower.x);
    (Vec4 &) g_params.planes[3] = Vec4(-1.0f, 0.0f, 0.0f, g_sceneUpper.x);
    (Vec4 &) g_params.planes[4] = Vec4(0.0f, 0.0f, -1.0f, g_sceneUpper.z);
    (Vec4 &) g_params.planes[5] = Vec4(0.0f, -1.0f, 0.0f, g_sceneUpper.y);

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
    for (int i = 0; i < numTris; ++i) {
        Vec3 v0 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 0]]);
        Vec3 v1 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 1]]);
        Vec3 v2 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 2]]);

        Vec3 n = Cross(v1 - v0, v2 - v0);

        g_buffers->normals[g_buffers->triangles[i * 3 + 0]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 1]] += Vec4(n, 0.0f);
        g_buffers->normals[g_buffers->triangles[i * 3 + 2]] += Vec4(n, 0.0f);
    }

    for (int i = 0; i < int(maxParticles); ++i)
        g_buffers->normals[i] = Vec4(SafeNormalize(Vec3(g_buffers->normals[i]), Vec3(0.0f, 1.0f, 0.0f)), 0.0f);


    // std::cout << "normals initialized" << endl;


    // save mesh positions for skinning
    if (g_mesh) {
        g_meshRestPositions = g_mesh->m_positions;
    } else {
        g_meshRestPositions.resize(0);
    }

    g_solverDesc.maxParticles = maxParticles;
    g_solverDesc.maxDiffuseParticles = g_maxDiffuseParticles;
    g_solverDesc.maxNeighborsPerParticle = g_maxNeighborsPerParticle;
    g_solverDesc.maxContactsPerParticle = g_maxContactsPerParticle;

    // main create method for the Flex solver
    g_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc);

    // give scene a chance to do some post solver initialization
    g_scenes[g_scene]->PostInitialize();

    // center camera on particles
    if (centerCamera) {
        g_camPos = Vec3((g_sceneLower.x + g_sceneUpper.x) * 0.5f, min(g_sceneUpper.y * 1.25f, 6.0f),
                        g_sceneUpper.z + min(g_sceneUpper.y, 6.0f) * 2.0f);
        g_camAngle = Vec3(0.0f, -DegToRad(15.0f), 0.0f);

        // give scene a chance to modify camera position
        g_scenes[g_scene]->CenterCamera();
    }

    // create active indices (just a contiguous block for the demo)
    g_buffers->activeIndices.resize(g_buffers->positions.size());
    for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
        g_buffers->activeIndices[i] = i;

    // printf("set active indices done.\n");

    // resize particle buffers to fit
    g_buffers->positions.resize(maxParticles);
    g_buffers->velocities.resize(maxParticles);
    g_buffers->phases.resize(maxParticles);

    g_buffers->densities.resize(maxParticles);
    g_buffers->anisotropy1.resize(maxParticles);
    g_buffers->anisotropy2.resize(maxParticles);
    g_buffers->anisotropy3.resize(maxParticles);

    // save rest positions
    g_buffers->restPositions.resize(g_buffers->positions.size());
    for (int i = 0; i < g_buffers->positions.size(); ++i)
        g_buffers->restPositions[i] = g_buffers->positions[i];

    // printf("save rest positions done.\n");

    // builds rigids constraints
    if (g_buffers->rigidOffsets.size()) {
        assert(g_buffers->rigidOffsets.size() > 1);

        const int numRigids = g_buffers->rigidOffsets.size() - 1;

        /*
        printf("rigidOffsets\n");
        for (size_t i = 0; i < (size_t) g_buffers->rigidOffsets.size(); i++) {
            printf("%d %d\n", i, g_buffers->rigidOffsets[i]);
        }

        printf("rigidIndices\n");
        for (size_t i = 0; i < (size_t) g_buffers->rigidIndices.size(); i++) {
            printf("%d %d\n", i, g_buffers->rigidIndices[i]);
        }
         */

        // If the centers of mass for the rigids are not yet computed, this is done here
        // (If the CreateParticleShape method is used instead of the NvFlexExt methods, the centers of mass will be calculated here)
        if (g_buffers->rigidTranslations.size() == 0) {
            g_buffers->rigidTranslations.resize(g_buffers->rigidOffsets.size() - 1, Vec3());
            CalculateRigidCentersOfMass(&g_buffers->positions[0], g_buffers->positions.size(),
                                        &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0],
                                        &g_buffers->rigidIndices[0], numRigids);
        }
        // printf("rigid mass center computation done.\n");

        // calculate local rest space positions
        g_buffers->rigidLocalPositions.resize(g_buffers->rigidOffsets.back());
        CalculateRigidLocalPositions(&g_buffers->positions[0], &g_buffers->rigidOffsets[0],
                                     &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids,
                                     &g_buffers->rigidLocalPositions[0]);

        // set rigidRotations to correct length, probably NULL up until here
        g_buffers->rigidRotations.resize(g_buffers->rigidOffsets.size() - 1, Quat());
    }

    // printf("rigid constraints build done.\n");

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
    if (g_buffers->springIndices.size()) {
        assert((g_buffers->springIndices.size() & 1) == 0);
        assert((g_buffers->springIndices.size() / 2) == g_buffers->springLengths.size());

        NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer,
                         g_buffers->springStiffness.buffer, g_buffers->springLengths.size());
    }

    // rigids
    if (g_buffers->rigidOffsets.size()) {
        NvFlexSetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer,
                        g_buffers->rigidLocalPositions.buffer, g_buffers->rigidLocalNormals.buffer,
                        g_buffers->rigidCoefficients.buffer, g_buffers->rigidPlasticThresholds.buffer,
                        g_buffers->rigidPlasticCreeps.buffer, g_buffers->rigidRotations.buffer,
                        g_buffers->rigidTranslations.buffer, g_buffers->rigidOffsets.size() - 1,
                        g_buffers->rigidIndices.size());
    }

    // std::cout << "rigids setup done" << endl;


    // inflatables
    if (g_buffers->inflatableTriOffsets.size()) {
        NvFlexSetInflatables(g_solver, g_buffers->inflatableTriOffsets.buffer, g_buffers->inflatableTriCounts.buffer,
                             g_buffers->inflatableVolumes.buffer, g_buffers->inflatablePressures.buffer,
                             g_buffers->inflatableCoefficients.buffer, g_buffers->inflatableTriOffsets.size());
    }

    // dynamic triangles
    if (g_buffers->triangles.size()) {
        NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer,
                                  g_buffers->triangles.size() / 3);
    }

    // collision shapes
    if (g_buffers->shapeFlags.size()) {
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

    // create render buffers
    if (g_render) {
        g_fluidRenderBuffers = CreateFluidRenderBuffers(maxParticles, g_interop);
        g_diffuseRenderBuffers = CreateDiffuseRenderBuffers(g_maxDiffuseParticles, g_interop);
    }

    // perform initial sim warm up
    if (g_warmup) {
        printf("Warming up sim..\n");

        // warm it up (relax positions to reach rest density without affecting velocity)
        NvFlexParams copy = g_params;
        copy.numIterations = 4;

        NvFlexSetParams(g_solver, &copy);

        const int kWarmupIterations = 100;

        for (int i = 0; i < kWarmupIterations; ++i) {
            NvFlexUpdateSolver(g_solver, 0.0001f, 1, false);
            NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, NULL);
        }

        // udpate host copy
        NvFlexGetParticles(g_solver, g_buffers->positions.buffer, NULL);
        NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, NULL);
        NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer,
                            g_buffers->anisotropy3.buffer, NULL);

        printf("Finished warm up.\n");
    }
//    printf("init scene done.\n");
}

/*
void Reset() {
    Init(g_scene, false);
}
*/

void Shutdown() {
    // free buffers
    DestroyBuffers(g_buffers);

    for (auto &iter : g_meshes) {
        NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    for (auto &iter : g_fields) {
        NvFlexDestroyDistanceField(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    for (auto &iter : g_convexes) {
        NvFlexDestroyConvexMesh(g_flexLib, iter.first);
        DestroyGpuMesh(iter.second);
    }

    g_fields.clear();
    g_meshes.clear();

    NvFlexDestroySolver(g_solver);
    NvFlexShutdown(g_flexLib);
}

void UpdateEmitters() {
    float spin = DegToRad(15.0f);

    const Vec3 forward(-sinf(g_camAngle.x + spin) * cosf(g_camAngle.y), sinf(g_camAngle.y),
                       -cosf(g_camAngle.x + spin) * cosf(g_camAngle.y));
    const Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

    g_emitters[0].mDir = Normalize(forward + Vec3(0.0, 0.4f, 0.0f));
    g_emitters[0].mRight = right;
    g_emitters[0].mPos = g_camPos + forward * 1.f + Vec3(0.0f, 0.2f, 0.0f) + right * 0.65f;

    // process emitters
    if (g_emit) {
        int activeCount = NvFlexGetActiveCount(g_solver);

        size_t e = 0;

        // skip camera emitter when moving forward or things get messy
        if (g_camSmoothVel.z >= 0.025f)
            e = 1;

        for (; e < g_emitters.size(); ++e) {
            if (!g_emitters[e].mEnabled)
                continue;

            Vec3 emitterDir = g_emitters[e].mDir;
            Vec3 emitterRight = g_emitters[e].mRight;
            Vec3 emitterPos = g_emitters[e].mPos;

            float r = g_params.fluidRestDistance;
            int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);

            float numParticles = (g_emitters[e].mSpeed / r) * g_dt;

            // whole number to emit
            auto n = int(numParticles + g_emitters[e].mLeftOver);

            if (n)
                g_emitters[e].mLeftOver = (numParticles + g_emitters[e].mLeftOver) - n;
            else
                g_emitters[e].mLeftOver += numParticles;

            // create a grid of particles (n particles thick)
            for (int k = 0; k < n; ++k) {
                int emitterWidth = g_emitters[e].mWidth;
                int numParticles = emitterWidth * emitterWidth;
                for (int i = 0; i < numParticles; ++i) {
                    float x = float(i % emitterWidth) - float(emitterWidth / 2);
                    float y = float((i / emitterWidth) % emitterWidth) - float(emitterWidth / 2);

                    if ((sqr(x) + sqr(y)) <= (emitterWidth / 2) * (emitterWidth / 2)) {
                        Vec3 up = Normalize(Cross(emitterDir, emitterRight));
                        Vec3 offset = r * (emitterRight * x + up * y) + float(k) * emitterDir * r;

                        if (activeCount < g_buffers->positions.size()) {
                            g_buffers->positions[activeCount] = Vec4(emitterPos + offset, 1.0f);
                            g_buffers->velocities[activeCount] = emitterDir * g_emitters[e].mSpeed;
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

void UpdateCamera() {
    Vec3 forward(-sinf(g_camAngle.x) * cosf(g_camAngle.y), sinf(g_camAngle.y),
                 -cosf(g_camAngle.x) * cosf(g_camAngle.y));
    Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));

    g_camSmoothVel = Lerp(g_camSmoothVel, g_camVel, 0.1f);
    g_camPos += (forward * g_camSmoothVel.z + right * g_camSmoothVel.x + Cross(right, forward) * g_camSmoothVel.y);
//    cout<<"g_camPos"<<g_camPos[0] << " " << g_camPos[1] << " " << g_camPos[2]<<endl;
//    cout<<"g_camAngle"<<g_camAngle[0] << " "<< g_camAngle[1]<<" "<<g_camAngle[2]<<endl;
}

void UpdateMouse() {
    // mouse button is up release particle
    if (g_lastb == -1) {
        if (g_mouseParticle != -1) {
            // restore particle mass
            g_buffers->positions[g_mouseParticle].w = g_mouseMass;

            // deselect
            g_mouseParticle = -1;
        }
    }

    // mouse went down, pick new particle
    if (g_mousePicked) {
        assert(g_mouseParticle == -1);

        Vec3 origin, dir;
        GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

        const int numActive = NvFlexGetActiveCount(g_solver);

        g_mouseParticle = PickParticle(origin, dir, &g_buffers->positions[0], &g_buffers->phases[0], numActive,
                                       g_params.radius * 0.8f, g_mouseT);

        if (g_mouseParticle != -1) {
            printf("picked: %d, mass: %f v: %f %f %f\n", g_mouseParticle, g_buffers->positions[g_mouseParticle].w,
                   g_buffers->velocities[g_mouseParticle].x, g_buffers->velocities[g_mouseParticle].y,
                   g_buffers->velocities[g_mouseParticle].z);

            g_mousePos = origin + dir * g_mouseT;
            g_mouseMass = g_buffers->positions[g_mouseParticle].w;
            g_buffers->positions[g_mouseParticle].w = 0.0f;        // increase picked particle's mass to force it towards the point
        }

        g_mousePicked = false;
    }

    // update picked particle position
    if (g_mouseParticle != -1) {
        Vec3 p = Lerp(Vec3(g_buffers->positions[g_mouseParticle]), g_mousePos, 0.8f);
        Vec3 delta = p - Vec3(g_buffers->positions[g_mouseParticle]);

        g_buffers->positions[g_mouseParticle].x = p.x;
        g_buffers->positions[g_mouseParticle].y = p.y;
        g_buffers->positions[g_mouseParticle].z = p.z;

        g_buffers->velocities[g_mouseParticle].x = delta.x / g_dt;
        g_buffers->velocities[g_mouseParticle].y = delta.y / g_dt;
        g_buffers->velocities[g_mouseParticle].z = delta.z / g_dt;
    }
}

void UpdateWind() {
    g_windTime += g_dt;

    const Vec3 kWindDir = Vec3(3.0f, 15.0f, 0.0f);
    const float kNoise = Perlin1D(g_windTime * g_windFrequency, 10, 0.25f);
    Vec3 wind = g_windStrength * kWindDir * Vec3(kNoise, fabsf(kNoise), 0.0f);

    g_params.wind[0] = wind.x;
    g_params.wind[1] = wind.y;
    g_params.wind[2] = wind.z;

    if (g_wavePool) {
        g_waveTime += g_dt;
        g_params.planes[2][3] =
                g_wavePlane + (sinf(float(g_waveTime) * g_waveFrequency - kPi * 0.5f) * 0.5f + 0.5f) * g_waveAmplitude;
    }
}

void SyncScene() {
    // let the scene send updates to flex directly
    g_scenes[g_scene]->Sync();
}

void UpdateScene(py::array_t<float> update_params) {
    // give scene a chance to make changes to particle buffers
    g_scenes[g_scene]->Update(update_params);
}

void RenderScene() {
    const int numParticles = NvFlexGetActiveCount(g_solver);
    const int numDiffuse = g_buffers->diffuseCount[0];

    //---------------------------------------------------
    // use VBO buffer wrappers to allow Flex to write directly to the OpenGL buffers
    // Flex will take care of any CUDA interop mapping/unmapping during the get() operations

    if (numParticles) {
        if (g_interop) {
            // copy data directly from solver to the renderer buffers
            UpdateFluidRenderBuffers(g_fluidRenderBuffers, g_solver, g_drawEllipsoids, g_drawDensity);
            // printf("pass UpdateFluidRenderBuffers\n");
        } else {
            // copy particle data to GPU render device

            if (g_drawEllipsoids) {
                // if fluid surface rendering then update with smooth positions and anisotropy
                UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                                         &g_buffers->smoothPositions[0],
                                         (g_drawDensity) ? &g_buffers->densities[0] : (float *) &g_buffers->phases[0],
                                         &g_buffers->anisotropy1[0],
                                         &g_buffers->anisotropy2[0],
                                         &g_buffers->anisotropy3[0],
                                         g_buffers->positions.size(),
                                         &g_buffers->activeIndices[0],
                                         numParticles);
            } else {
                // otherwise just send regular positions and no anisotropy
                UpdateFluidRenderBuffers(g_fluidRenderBuffers,
                                         &g_buffers->positions[0],
                                         (float *) &g_buffers->phases[0],
                                         nullptr, nullptr, nullptr,
                                         g_buffers->positions.size(),
                                         &g_buffers->activeIndices[0],
                                         numParticles);
            }
        }
    }

    // GPU Render time doesn't include CPU->GPU copy time
    GraphicsTimerBegin();

    if (numDiffuse) {
        if (g_interop) {
            // copy data directly from solver to the renderer buffers
            UpdateDiffuseRenderBuffers(g_diffuseRenderBuffers, g_solver);
        } else {
            // copy diffuse particle data from host to GPU render device
            UpdateDiffuseRenderBuffers(g_diffuseRenderBuffers,
                                       &g_buffers->diffusePositions[0],
                                       &g_buffers->diffuseVelocities[0],
                                       numDiffuse);
        }
    }

    //---------------------------------------
    // setup view and state

    float fov = kPi / 4.0f;
    float aspect = float(g_screenWidth) / g_screenHeight;

    Matrix44 proj = ProjectionMatrix(RadToDeg(fov), aspect, g_camNear, g_camFar);
    Matrix44 view = RotationMatrix(-g_camAngle.x, Vec3(0.0f, 1.0f, 0.0f)) *
                    RotationMatrix(-g_camAngle.y, Vec3(cosf(-g_camAngle.x), 0.0f, sinf(-g_camAngle.x))) *
                    TranslationMatrix(-Point3(g_camPos));

    //------------------------------------
    // lighting pass

    // expand scene bounds to fit most scenes
    g_sceneLower = Min(g_sceneLower, Vec3(-2.0f, 0.0f, -2.0f));
    g_sceneUpper = Max(g_sceneUpper, Vec3(2.0f, 2.0f, 2.0f));

    Vec3 sceneExtents = g_sceneUpper - g_sceneLower;
    Vec3 sceneCenter = 0.5f * (g_sceneUpper + g_sceneLower);

    g_lightDir = Normalize(Vec3(5.0f, 15.0f, 7.5f));
    g_lightPos = sceneCenter + g_lightDir * Length(sceneExtents) * g_lightDistance;
    g_lightTarget = sceneCenter;

    // calculate tight bounds for shadow frustum
    float lightFov = 2.0f * atanf(Length(g_sceneUpper - sceneCenter) / Length(g_lightPos - sceneCenter));

    // scale and clamp fov for aesthetics
    lightFov = Clamp(lightFov, DegToRad(25.0f), DegToRad(65.0f));

    Matrix44 lightPerspective = ProjectionMatrix(RadToDeg(lightFov), 1.0f, 1.0f, 1000.0f);
    Matrix44 lightView = LookAtMatrix(Point3(g_lightPos), Point3(g_lightTarget));
    Matrix44 lightTransform = lightPerspective * lightView;

    // radius used for drawing
    float radius = Max(g_params.solidRestDistance, g_params.fluidRestDistance) * 0.5f * g_pointScale;

    //-------------------------------------
    // shadowing pass

    if (g_meshSkinIndices.size())
        SkinMesh();

    // create shadow maps
    ShadowBegin(g_shadowMap);
    // printf("pass ShadowBegin\n");


    SetView(lightView, lightPerspective);
    SetCullMode(false);

    // give scene a chance to do custom drawing
    g_scenes[g_scene]->Draw(1);

    if (g_drawMesh)
        DrawMesh(g_mesh, g_meshColor);
    
    // printf("pass DrawMesh\n");

    DrawShapes();
    // printf("pass DrawShapes\n");

    if (g_drawCloth && g_buffers->triangles.size()) {
        DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0], g_buffers->uvs.size() ? &g_buffers->uvs[0].x : NULL,
                  &g_buffers->triangles[0], g_buffers->triangles.size() / 3, g_buffers->positions.size(), 3,
                  g_expandCloth);
    }

    if (g_drawRopes) {
        for (size_t i = 0; i < g_ropes.size(); ++i)
            DrawRope(&g_buffers->positions[0], &g_ropes[i].mIndices[0], g_ropes[i].mIndices.size(),
                     radius * g_ropeScale, i);
    }
    // printf("pass DrawRope\n");

    int shadowParticles = numParticles;
    int shadowParticlesOffset = 0;

    if (!g_drawPoints) {
        shadowParticles = 0;

        if (g_drawEllipsoids) {
            shadowParticles = numParticles - g_numSolidParticles;
            shadowParticlesOffset = g_numSolidParticles;
        }
    } else {
        int offset = g_drawMesh ? g_numSolidParticles : 0;

        shadowParticles = numParticles - offset;
        shadowParticlesOffset = offset;
    }

    if (g_buffers->activeIndices.size())
        DrawPoints(g_fluidRenderBuffers, shadowParticles, shadowParticlesOffset, radius, 2048, 1.0f, lightFov,
                   g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);

    ShadowEnd();
    // printf("pass ShadowEnd\n");

    //----------------
    // lighting pass

    BindSolidShader(g_lightPos, g_lightTarget, lightTransform, g_shadowMap, 0.0f, Vec4(g_clearColor, g_fogDistance));

    SetView(view, proj);
    SetCullMode(true);

    // When the benchmark measures async compute, we need a graphics workload that runs for a whole frame.
    // We do this by rerendering our simple graphics many times.
    int passes = g_increaseGfxLoadForAsyncComputeTesting ? 50 : 1;

    for (int i = 0; i != passes; i++) {
        DrawPlanes((Vec4 *) g_params.planes, g_params.numPlanes, g_drawPlaneBias);

        if (g_drawMesh)
            DrawMesh(g_mesh, g_meshColor);


        DrawShapes();

        if (g_drawCloth && g_buffers->triangles.size())
            DrawCloth(&g_buffers->positions[0], &g_buffers->normals[0],
                      g_buffers->uvs.size() ? &g_buffers->uvs[0].x : nullptr, &g_buffers->triangles[0],
                      g_buffers->triangles.size() / 3, g_buffers->positions.size(), 3, g_expandCloth);

        if (g_drawRopes) {
            for (size_t i = 0; i < g_ropes.size(); ++i)
                DrawRope(&g_buffers->positions[0], &g_ropes[i].mIndices[0], g_ropes[i].mIndices.size(),
                         g_params.radius * 0.5f * g_ropeScale, i);
        }

        // give scene a chance to do custom drawing
        g_scenes[g_scene]->Draw(0);
    }
    UnbindSolidShader();
    // printf("pass UnbindSolidShader\n");

    // first pass of diffuse particles (behind fluid surface)
    if (g_drawDiffuse)
        RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius * g_diffuseScale,
                      float(g_screenWidth), aspect, fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform,
                      g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow,
                      false);
    // printf("pass RenderDiffuse\n");

    if (g_drawEllipsoids) {
        // draw solid particles separately
        if (g_numSolidParticles && g_drawPoints)
            DrawPoints(g_fluidRenderBuffers, g_numSolidParticles, 0, radius, float(g_screenWidth), aspect, fov,
                       g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);

        // printf("pass DrawPoints\n");
        // render fluid surface
        RenderEllipsoids(g_fluidRenderer, g_fluidRenderBuffers, numParticles - g_numSolidParticles, g_numSolidParticles,
                         radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform,
                         g_shadowMap, g_fluidColor, g_blur, g_ior, g_drawOpaque);

        // printf("pass RenderEllipsoids\n");
        // second pass of diffuse particles for particles in front of fluid surface
        if (g_drawDiffuse)
            RenderDiffuse(g_fluidRenderer, g_diffuseRenderBuffers, numDiffuse, radius * g_diffuseScale,
                          float(g_screenWidth), aspect, fov, g_diffuseColor, g_lightPos, g_lightTarget, lightTransform,
                          g_shadowMap, g_diffuseMotionScale, g_diffuseInscatter, g_diffuseOutscatter, g_diffuseShadow,
                          true);
        // printf("pass RenderDiffuse\n");
        
    } else {
        // draw all particles as spheres
        if (g_drawPoints) {
            int offset = g_drawMesh ? g_numSolidParticles : 0;

            if (g_buffers->activeIndices.size())
                DrawPoints(g_fluidRenderBuffers, numParticles - offset, offset, radius, float(g_screenWidth), aspect,
                           fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_drawDensity);
        }
        // printf("pass DrawPoints\n");
    }

    GraphicsTimerEnd();
    // printf("pass GraphicsTimerEnd\n");
}

void RenderDebug() {
    if (g_mouseParticle != -1) {
        // draw mouse spring
        BeginLines();
        DrawLine(g_mousePos, Vec3(g_buffers->positions[g_mouseParticle]), Vec4(1.0f));
        EndLines();
    }

    // springs
    if (g_drawSprings) {
        Vec4 color;

        if (g_drawSprings == 1) {
            // stretch
            color = Vec4(0.0f, 0.0f, 1.0f, 0.8f);
        }
        if (g_drawSprings == 2) {
            // tether
            color = Vec4(0.0f, 1.0f, 0.0f, 0.8f);
        }

        BeginLines();

        int start = 0;

        for (int i = start; i < g_buffers->springLengths.size(); ++i) {
            if (g_drawSprings == 1 && g_buffers->springStiffness[i] < 0.0f)
                continue;
            if (g_drawSprings == 2 && g_buffers->springStiffness[i] > 0.0f)
                continue;

            int a = g_buffers->springIndices[i * 2];
            int b = g_buffers->springIndices[i * 2 + 1];

            DrawLine(Vec3(g_buffers->positions[a]), Vec3(g_buffers->positions[b]), color);
        }

        EndLines();
    }

    // visualize contacts against the environment
    if (g_drawContacts) {
        const int maxContactsPerParticle = 6;

        NvFlexVector<Vec4> contactPlanes(g_flexLib, g_buffers->positions.size() * maxContactsPerParticle);
        NvFlexVector<Vec4> contactVelocities(g_flexLib, g_buffers->positions.size() * maxContactsPerParticle);
        NvFlexVector<int> contactIndices(g_flexLib, g_buffers->positions.size());
        NvFlexVector<unsigned int> contactCounts(g_flexLib, g_buffers->positions.size());

        NvFlexGetContacts(g_solver, contactPlanes.buffer, contactVelocities.buffer, contactIndices.buffer,
                          contactCounts.buffer);

        // ensure transfers have finished
        contactPlanes.map();
        contactVelocities.map();
        contactIndices.map();
        contactCounts.map();

        BeginLines();

        for (int i = 0; i < int(g_buffers->activeIndices.size()); ++i) {
            const int contactIndex = contactIndices[g_buffers->activeIndices[i]];
            const unsigned int count = contactCounts[contactIndex];

            const float scale = 0.1f;

            for (unsigned int c = 0; c < count; ++c) {
                Vec4 plane = contactPlanes[contactIndex * maxContactsPerParticle + c];

                DrawLine(Vec3(g_buffers->positions[g_buffers->activeIndices[i]]),
                         Vec3(g_buffers->positions[g_buffers->activeIndices[i]]) + Vec3(plane) * scale,
                         Vec4(0.0f, 1.0f, 0.0f, 0.0f));
            }
        }

        EndLines();
    }

    if (g_drawBases) {
        for (int i = 0; i < int(g_buffers->rigidRotations.size()); ++i) {
            BeginLines();

            float size = 0.1f;

            for (int b = 0; b < 3; ++b) {
                Vec3 color(0.0f, 0.0f, 0.0f);

                if (b == 0)
                    color.x = 1.0f;
                else if (b == 1)
                    color.y = 1.0f;
                else
                    color.z = 1.0f;

                Matrix33 frame(g_buffers->rigidRotations[i]);

                DrawLine(Vec3(g_buffers->rigidTranslations[i]),
                         Vec3(g_buffers->rigidTranslations[i] + frame.cols[b] * size),
                         Vec4(color, 0.0f));
            }

            EndLines();
        }
    }

    if (g_drawNormals) {
        NvFlexGetNormals(g_solver, g_buffers->normals.buffer, nullptr);

        BeginLines();

        for (int i = 0; i < g_buffers->normals.size(); ++i) {
            DrawLine(Vec3(g_buffers->positions[i]),
                     Vec3(g_buffers->positions[i] - g_buffers->normals[i] * g_buffers->normals[i].w),
                     Vec4(0.0f, 1.0f, 0.0f, 0.0f));
        }

        EndLines();
    }
}

void DrawShapes() {
    for (int i = 0; i < g_buffers->shapeFlags.size(); ++i) {

        const int flags = g_buffers->shapeFlags[i];

        // unpack flags
        auto type = int(flags & eNvFlexShapeFlagTypeMask);
        //bool dynamic = int(flags&eNvFlexShapeFlagDynamic) > 0;

        Vec3 color = g_shape_color;

        if (flags & eNvFlexShapeFlagTrigger) {
            // printf("there is a trigger shape! \n");
            color = Vec3(1.0f, 0.0f, 0.0f);
            // SetFillMode(true);
        }

        // render with prev positions to match particle update order
        // can also think of this as current/next
        const Quat rotation = g_buffers->shapePrevRotations[i];
        const Vec3 position = Vec3(g_buffers->shapePrevPositions[i]);

        NvFlexCollisionGeometry geo = g_buffers->shapeGeometry[i];

        if (type == eNvFlexShapeSphere) {
            Mesh *sphere = CreateSphere(20, 20, geo.sphere.radius);

            Matrix44 xform = TranslationMatrix(Point3(position)) * RotationMatrix(Quat(rotation));
            sphere->Transform(xform);

            DrawMesh(sphere, Vec3(color));

            delete sphere;
        } else if (type == eNvFlexShapeCapsule) {
            Mesh *capsule = CreateCapsule(10, 20, geo.capsule.radius, geo.capsule.halfHeight);

            // transform to world space
            Matrix44 xform = TranslationMatrix(Point3(position)) * RotationMatrix(Quat(rotation)) *
                             RotationMatrix(DegToRad(-90.0f), Vec3(0.0f, 0.0f, 1.0f));
            capsule->Transform(xform);

            DrawMesh(capsule, Vec3(color));

            delete capsule;
        } else if (type == eNvFlexShapeBox) {
            Mesh *box = CreateCubeMesh();

            Matrix44 xform = TranslationMatrix(Point3(position)) * RotationMatrix(Quat(rotation)) *
                             ScaleMatrix(Vec3(geo.box.halfExtents) * 2.0f);
            box->Transform(xform);

            DrawMesh(box, Vec3(color));
            delete box;
        } else if (type == eNvFlexShapeConvexMesh) {
            if (g_convexes.find(geo.convexMesh.mesh) != g_convexes.end()) {
                GpuMesh *m = g_convexes[geo.convexMesh.mesh];

                if (m) {
                    Matrix44 xform = TranslationMatrix(Point3(g_buffers->shapePositions[i])) *
                                     RotationMatrix(Quat(g_buffers->shapeRotations[i])) *
                                     ScaleMatrix(geo.convexMesh.scale);
                    DrawGpuMesh(m, xform, Vec3(color));
                }
            }
        } else if (type == eNvFlexShapeTriangleMesh) {
            if (g_meshes.find(geo.triMesh.mesh) != g_meshes.end()) {
                GpuMesh *m = g_meshes[geo.triMesh.mesh];

                if (m) {
                    Matrix44 xform = TranslationMatrix(Point3(position)) * RotationMatrix(Quat(rotation)) *
                                     ScaleMatrix(geo.triMesh.scale);
                    DrawGpuMesh(m, xform, Vec3(color));
                }
            }
        } else if (type == eNvFlexShapeSDF) {
            if (g_fields.find(geo.sdf.field) != g_fields.end()) {
                GpuMesh *m = g_fields[geo.sdf.field];

                if (m) {
                    Matrix44 xform = TranslationMatrix(Point3(position)) * RotationMatrix(Quat(rotation)) *
                                     ScaleMatrix(geo.sdf.scale);
                    DrawGpuMesh(m, xform, Vec3(color));
                }
            }
        }
    }

    SetFillMode(g_wireframe);
}


// returns the new scene if one is selected
int DoUI() {
    // gui may set a new scene
    int newScene = -1;

    if (g_showHelp) {
        const int numParticles = NvFlexGetActiveCount(g_solver);
        const int numDiffuse = g_buffers->diffuseCount[0];

        int x = g_screenWidth - 200;
        int y = g_screenHeight - 23;

        // imgui
        unsigned char button = 0;
        if (g_lastb == SDL_BUTTON_LEFT)
            button = IMGUI_MBUT_LEFT;
        else if (g_lastb == SDL_BUTTON_RIGHT)
            button = IMGUI_MBUT_RIGHT;

        imguiBeginFrame(g_lastx, g_screenHeight - g_lasty, button, 0);

        x += 180;

        int fontHeight = 13;

        if (g_profile) {
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Frame: %d", g_frame);
            y -= fontHeight * 2;

            if (!g_ffmpeg) {
                DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Frame Time: %.2fms", g_realdt * 1000.0f);
                y -= fontHeight * 2;

                // If detailed profiling is enabled, then these timers will contain the overhead of the detail timers, so we won't display them.
                if (!g_profile) {
                    DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Sim Time (CPU): %.2fms",
                                    g_updateTime * 1000.0f);
                    y -= fontHeight;
                    DrawImguiString(x, y, Vec3(0.97f, 0.59f, 0.27f), IMGUI_ALIGN_RIGHT, "Sim Latency (GPU): %.2fms",
                                    g_simLatency);
                    y -= fontHeight * 2;

                    BenchmarkUpdateGraph();
                } else {
                    y -= fontHeight * 3;
                }
            }

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Particle Count: %d", numParticles);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Diffuse Count: %d", numDiffuse);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Rigid Count: %d",
                            g_buffers->rigidOffsets.size() > 0 ? g_buffers->rigidOffsets.size() - 1 : 0);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Spring Count: %d", g_buffers->springLengths.size());
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Num Substeps: %d", g_numSubsteps);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Num Iterations: %d", g_params.numIterations);
            y -= fontHeight * 2;

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Device: %s", g_deviceName);
            y -= fontHeight * 2;
        }

        if (g_profile) {
            DrawImguiString(x, y, Vec3(0.97f, 0.59f, 0.27f), IMGUI_ALIGN_RIGHT, "Total GPU Sim Latency: %.2fms",
                            g_timers.total);
            y -= fontHeight * 2;

            DrawImguiString(x, y, Vec3(0.0f, 1.0f, 0.0f), IMGUI_ALIGN_RIGHT, "GPU Latencies");
            y -= fontHeight;

            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Predict: %.2fms", g_timers.predict);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Create Cell Indices: %.2fms",
                            g_timers.createCellIndices);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Sort Cell Indices: %.2fms", g_timers.sortCellIndices);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Reorder: %.2fms", g_timers.reorder);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "CreateGrid: %.2fms", g_timers.createGrid);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Particles: %.2fms",
                            g_timers.collideParticles);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Shapes: %.2fms", g_timers.collideShapes);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Collide Triangles: %.2fms",
                            g_timers.collideTriangles);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Calculate Density: %.2fms",
                            g_timers.calculateDensity);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Densities: %.2fms", g_timers.solveDensities);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Velocities: %.2fms", g_timers.solveVelocities);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Rigids: %.2fms", g_timers.solveShapes);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Springs: %.2fms", g_timers.solveSprings);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Solve Inflatables: %.2fms",
                            g_timers.solveInflatables);
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
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Calculate Anisotropy: %.2fms",
                            g_timers.calculateAnisotropy);
            y -= fontHeight;
            DrawImguiString(x, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Update Diffuse: %.2fms", g_timers.updateDiffuse);
            y -= fontHeight * 2;
        }

        x -= 180;

        int uiOffset = 250;
        int uiBorder = 20;
        int uiWidth = 200;
        int uiHeight = g_screenHeight - uiOffset - uiBorder * 3;
        int uiLeft = uiBorder;

        if (g_tweakPanel)
            imguiBeginScrollArea("Scene", uiLeft, g_screenHeight - uiBorder - uiOffset, uiWidth, uiOffset,
                                 &g_levelScroll);
        else
            imguiBeginScrollArea("Scene", uiLeft, uiBorder, uiWidth, g_screenHeight - uiBorder - uiBorder,
                                 &g_levelScroll);

        for (int i = 0; i < int(g_scenes.size()); ++i) {
            unsigned int color = g_scene == i ? imguiRGBA(255, 151, 61, 255) : imguiRGBA(255, 255, 255, 200);
            if (imguiItem(g_scenes[i]->GetName(), true, color)) {
                newScene = i;
            }
        }

        imguiEndScrollArea();

        if (g_tweakPanel) {
            static int scroll = 0;

            imguiBeginScrollArea("Options", uiLeft, g_screenHeight - uiBorder - uiHeight - uiOffset - uiBorder, uiWidth,
                                 uiHeight, &scroll);
            imguiSeparatorLine();

            // global options
            imguiLabel("Global");
            if (imguiCheck("Emit particles", g_emit))
                g_emit = !g_emit;

            if (imguiCheck("Pause", g_pause))
                g_pause = !g_pause;

            imguiSeparatorLine();

            if (imguiCheck("Wireframe", g_wireframe))
                g_wireframe = !g_wireframe;

            if (imguiCheck("Draw Points", g_drawPoints))
                g_drawPoints = !g_drawPoints;

            if (imguiCheck("Draw Fluid", g_drawEllipsoids))
                g_drawEllipsoids = !g_drawEllipsoids;

            if (imguiCheck("Draw Mesh", g_drawMesh)) {
                g_drawMesh = !g_drawMesh;
                g_drawRopes = !g_drawRopes;
            }

            if (imguiCheck("Draw Basis", g_drawBases))
                g_drawBases = !g_drawBases;

            if (imguiCheck("Draw Springs", bool(g_drawSprings != 0)))
                g_drawSprings = (g_drawSprings) ? 0 : 1;

            if (imguiCheck("Draw Contacts", g_drawContacts))
                g_drawContacts = !g_drawContacts;

            imguiSeparatorLine();

            // scene options
            g_scenes[g_scene]->DoGui();

            if (imguiButton("Reset Scene"))
                g_resetScene = true;

            imguiSeparatorLine();

            auto n = float(g_numSubsteps);
            if (imguiSlider("Num Substeps", &n, 1, 10, 1))
                g_numSubsteps = int(n);

            n = float(g_params.numIterations);
            if (imguiSlider("Num Iterations", &n, 1, 20, 1))
                g_params.numIterations = int(n);

            imguiSeparatorLine();
            imguiSlider("Gravity X", &g_params.gravity[0], -50.0f, 50.0f, 1.0f);
            imguiSlider("Gravity Y", &g_params.gravity[1], -50.0f, 50.0f, 1.0f);
            imguiSlider("Gravity Z", &g_params.gravity[2], -50.0f, 50.0f, 1.0f);

            imguiSeparatorLine();
            imguiSlider("Radius", &g_params.radius, 0.01f, 0.5f, 0.01f);
            imguiSlider("Solid Radius", &g_params.solidRestDistance, 0.0f, 0.5f, 0.001f);
            imguiSlider("Fluid Radius", &g_params.fluidRestDistance, 0.0f, 0.5f, 0.001f);

            // common params
            imguiSeparatorLine();
            imguiSlider("Dynamic Friction", &g_params.dynamicFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Static Friction", &g_params.staticFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Particle Friction", &g_params.particleFriction, 0.0f, 1.0f, 0.01f);
            imguiSlider("Restitution", &g_params.restitution, 0.0f, 1.0f, 0.01f);
            imguiSlider("SleepThreshold", &g_params.sleepThreshold, 0.0f, 1.0f, 0.01f);
            imguiSlider("Shock Propagation", &g_params.shockPropagation, 0.0f, 10.0f, 0.01f);
            imguiSlider("Damping", &g_params.damping, 0.0f, 10.0f, 0.01f);
            imguiSlider("Dissipation", &g_params.dissipation, 0.0f, 0.01f, 0.0001f);
            imguiSlider("SOR", &g_params.relaxationFactor, 0.0f, 5.0f, 0.01f);

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
                g_params.diffuseBallistic = int(n);

            imguiEndScrollArea();
        }

        imguiEndFrame();

        // kick render commands
//        DrawImguiGraph();
    }

    return newScene;
}

void UpdateFrame(py::array_t<float> update_params) {
    if (!g_headless) {
        static double lastTime;

        // real elapsed frame time
        double frameBeginTime = GetSeconds();

        g_realdt = float(frameBeginTime - lastTime);
        lastTime = frameBeginTime;

        // do gamepad input polling
        double currentTime = frameBeginTime;
        static double lastJoyTime = currentTime;

        if (g_gamecontroller && currentTime - lastJoyTime > g_dt) {
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
                rightStick.x != 0.0f || rightStick.y != 0.0f) {
                // note constant factor to speed up analog control compared to digital because it is more controllable.
                g_camVel.z = -4 * g_camSpeed * leftStick.y;
                g_camVel.x = 4 * g_camSpeed * leftStick.x;

                // cam orientation
                g_camAngle.x -= rightStick.x * 0.05f;
                g_camAngle.y -= rightStick.y * 0.05f;
            }

            // Handle left stick motion
            static bool bLeftStick = false;

            if ((leftStick.x != 0.0f || leftStick.y != 0.0f) && !bLeftStick) {
                bLeftStick = true;
            } else if ((leftStick.x == 0.0f && leftStick.y == 0.0f) && bLeftStick) {
                bLeftStick = false;
                g_camVel.z = -4 * g_camSpeed * leftStick.y;
                g_camVel.x = 4 * g_camSpeed * leftStick.x;
            }

            // Handle triggers as controller button events
            void ControllerButtonEvent(SDL_ControllerButtonEvent event);

            static bool bLeftTrigger = false;
            static bool bRightTrigger = false;
            SDL_ControllerButtonEvent e;

            if (!bLeftTrigger && trigger.x > 0.0f) {
                e.type = SDL_CONTROLLERBUTTONDOWN;
                e.button = SDL_CONTROLLER_BUTTON_LEFT_TRIGGER;
                ControllerButtonEvent(e);
                bLeftTrigger = true;
            } else if (bLeftTrigger && trigger.x == 0.0f) {
                e.type = SDL_CONTROLLERBUTTONUP;
                e.button = SDL_CONTROLLER_BUTTON_LEFT_TRIGGER;
                ControllerButtonEvent(e);
                bLeftTrigger = false;
            }

            if (!bRightTrigger && trigger.y > 0.0f) {
                e.type = SDL_CONTROLLERBUTTONDOWN;
                e.button = SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER;
                ControllerButtonEvent(e);
                bRightTrigger = true;
            } else if (bRightTrigger && trigger.y == 0.0f) {
                e.type = SDL_CONTROLLERBUTTONDOWN;
                e.button = SDL_CONTROLLER_BUTTON_RIGHT_TRIGGER;
                ControllerButtonEvent(e);
                bRightTrigger = false;
            }
        }
    }

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
			UpdateScene(update_params);
		}
		
	}
	else
	{
        if (g_render) {
            UpdateCamera();
        }
        // printf("updatecamera done\n");
		UpdateEmitters();
        // printf("updateemitters done\n");
		UpdateWind();
        // printf("updatewind done\n");
		UpdateScene(update_params);
        // printf("updatescene done\n");
	}

    //-------------------------------------------------------------------
    // Render
    int newScene = -1;
    double renderBeginTime = GetSeconds();

    if (g_render) {
        

        if (g_profile && (!g_pause || g_step)) {
            if (g_benchmark) {
                g_numDetailTimers = NvFlexGetDetailTimers(g_solver, &g_detailTimers);
            } else {
                memset(&g_timers, 0, sizeof(g_timers));
                NvFlexGetTimers(g_solver, &g_timers);
            }
        }

        StartFrame(Vec4(g_clearColor, 1.0f));
        // printf("start frame done\n");

        // main scene render
        RenderScene();
        RenderDebug();

        // printf("render scene & debug done\n");

        if (!g_headless) {
            newScene = DoUI();
        }

        EndFrame();
        // printf("endframe done\n");

        // If user has disabled async compute, ensure that no compute can overlap
        // graphics by placing a sync between them
        if (!g_useAsyncCompute)
            NvFlexComputeWaitForGraphics(g_flexLib);
    }

    UnmapBuffers(g_buffers);
    // printf("unmap buffers done\n");

    if (!g_headless) {
        // move mouse particle (must be done here as GetViewRay() uses the GL projection state)
        if (g_mouseParticle != -1) {
            Vec3 origin, dir;
            GetViewRay(g_lastx, g_screenHeight - g_lasty, origin, dir);

            g_mousePos = origin + dir*g_mouseT;
        }
    }

    if (g_render) {
        if (g_capture) {
            TgaImage img;
            img.m_width = g_screenWidth;
            img.m_height = g_screenHeight;
            img.m_data = new uint32_t[g_screenWidth * g_screenHeight];

            ReadFrame((int *) img.m_data, g_screenWidth, g_screenHeight);
            TgaSave(g_ffmpeg, img, false);

            // printf("readframe done\n");
            // fwrite(img.m_data, sizeof(uint32_t)*g_screenWidth*g_screenHeight, 1, g_ffmpeg);

            delete[] img.m_data;
        }
    }

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

    // allow scene to update constraints etc
    SyncScene();

    if (g_shapesChanged) {
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

    if (!g_pause || g_step) {
        // tick solver
        NvFlexSetParams(g_solver, &g_params);
        NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

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

    // readback triangle normals
    if (g_buffers->triangles.size())
        NvFlexGetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer,
                                  g_buffers->triangles.size() / 3);

    // readback rigid transforms
    if (g_buffers->rigidOffsets.size())
        NvFlexGetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer,
                        g_buffers->rigidLocalPositions.buffer, nullptr, nullptr, nullptr, nullptr,
                        g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer);

    if (!g_interop && g_render) {
        // if not using interop then we read back fluid data to host
        if (g_drawEllipsoids) {
            NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, nullptr);
            NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer,
                                g_buffers->anisotropy3.buffer, NULL);
        }

        // read back diffuse data to host
        if (g_drawDensity)
            NvFlexGetDensities(g_solver, g_buffers->densities.buffer, nullptr);

        if (GetNumDiffuseRenderParticles(g_diffuseRenderBuffers)) {
            NvFlexGetDiffuseParticles(g_solver, g_buffers->diffusePositions.buffer, g_buffers->diffuseVelocities.buffer,
                                      g_buffers->diffuseCount.buffer);
        }
    } else if (g_render) {
        // read back just the new diffuse particle count, render buffers will be updated during rendering
        NvFlexGetDiffuseParticles(g_solver, nullptr, nullptr, g_buffers->diffuseCount.buffer);
    }

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
    if (!g_headless)
	{
		PresentFrame(g_vsync);
	}
	// else if (g_render) 
	// {
	// 	// PresentFrameHeadless();
    //     // printf("haha, you want to render on a cluster, but we do not have a device here for render!\n");
	// }


    // if gui or benchmark requested a scene change process it now
    if (newScene != -1) {
        g_scene = newScene;
        // Init(g_scene);
    }
}

void ReshapeWindow(int width, int height) {
    if (!g_benchmark)
        printf("Reshaping\n");

    ReshapeRender(g_window);

    if (!g_fluidRenderer || (width != g_screenWidth || height != g_screenHeight)) {
        if (g_fluidRenderer)
            DestroyFluidRenderer(g_fluidRenderer);
        g_fluidRenderer = CreateFluidRenderer(width, height);
    }

    g_screenWidth = width;
    g_screenHeight = height;
}

void InputArrowKeysDown(int key, int x, int y) {
    switch (key) {
        case SDLK_DOWN: {
            if (g_selectedScene < int(g_scenes.size()) - 1)
                g_selectedScene++;

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_selectedScene - 4) * 24, 0);
            break;
        }
        case SDLK_UP: {
            if (g_selectedScene > 0)
                g_selectedScene--;

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_selectedScene - 4) * 24, 0);
            break;
        }
        case SDLK_LEFT: {
            if (g_scene > 0)
                --g_scene;
            // Init(g_scene);

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_scene - 4) * 24, 0);
            break;
        }
        case SDLK_RIGHT: {
            if (g_scene < int(g_scenes.size()) - 1)
                ++g_scene;
            // Init(g_scene);

            // update scroll UI to center on selected scene
            g_levelScroll = max((g_scene - 4) * 24, 0);
            break;
        }
    }
}

void InputArrowKeysUp(int key, int x, int y) {
}

bool InputKeyboardDown(unsigned char key, int x, int y) {
    if (key > '0' && key <= '9') {
        g_scene = key - '0' - 1;
        // Init(g_scene);
        return false;
    }

    float kSpeed = g_camSpeed;

    switch (key) {
        case 'w': {
            g_camVel.z = kSpeed;
            break;
        }
        case 's': {
            g_camVel.z = -kSpeed;
            break;
        }
        case 'a': {
            g_camVel.x = -kSpeed;
            break;
        }
        case 'd': {
            g_camVel.x = kSpeed;
            break;
        }
        case 'q': {
            g_camVel.y = kSpeed;
            break;
        }
        case 'z': {
            //g_drawCloth = !g_drawCloth;
            g_camVel.y = -kSpeed;
            break;
        }
        case 'r': {
            g_resetScene = true;
            break;
        }
        case 'y': {
            g_wavePool = !g_wavePool;
            break;
        }
        case 'p': {
            g_pause = !g_pause;
            break;
        }
        case 'o': {
            g_step = true;
            break;
        }
        case 'h': {
            g_showHelp = !g_showHelp;
            break;
        }
        case 'e': {
            g_drawEllipsoids = !g_drawEllipsoids;
            break;
        }
        case 't': {
            g_drawOpaque = !g_drawOpaque;
            break;
        }
        case 'v': {
            g_drawPoints = !g_drawPoints;
            break;
        }
        case 'f': {
            g_drawSprings = (g_drawSprings + 1) % 3;
            break;
        }
        case 'i': {
            g_drawDiffuse = !g_drawDiffuse;
            break;
        }
        case 'm': {
            g_drawMesh = !g_drawMesh;
            break;
        }
        case 'n': {
            g_drawRopes = !g_drawRopes;
            break;
        }
        case 'j': {
            g_windTime = 0.0f;
            g_windStrength = 1.5f;
            g_windFrequency = 0.2f;
            break;
        }
        case '.': {
            g_profile = !g_profile;
            break;
        }
        case 'g': {
            if (g_params.gravity[1] != 0.0f)
                g_params.gravity[1] = 0.0f;
            else
                g_params.gravity[1] = -9.8f;

            break;
        }
        case '-': {
            if (g_params.numPlanes)
                g_params.numPlanes--;

            break;
        }
        case ' ': {
            g_emit = !g_emit;
            break;
        }
        case ';': {
            g_debug = !g_debug;
            break;
        }
        case 13: {
            g_scene = g_selectedScene;
            // Init(g_scene);
            break;
        }
        case 27: {
            // return quit = true
            return true;
        }
    };

    g_scenes[g_scene]->KeyDown(key);

    return false;
}

void InputKeyboardUp(unsigned char key, int x, int y) {
    switch (key) {
        case 'w':
        case 's': {
            g_camVel.z = 0.0f;
            break;
        }
        case 'a':
        case 'd': {
            g_camVel.x = 0.0f;
            break;
        }
        case 'q':
        case 'z': {
            g_camVel.y = 0.0f;
            break;
        }
    };
}

void MouseFunc(int b, int state, int x, int y) {
    switch (state) {
        case SDL_RELEASED: {
            g_lastx = x;
            g_lasty = y;
            g_lastb = -1;

            break;
        }
        case SDL_PRESSED: {
            g_lastx = x;
            g_lasty = y;
            g_lastb = b;
            if ((SDL_GetModState() & KMOD_LSHIFT) && g_lastb == SDL_BUTTON_LEFT) {
                // record that we need to update the picked particle
                g_mousePicked = true;
            }
            break;
        }
    };
}

void MousePassiveMotionFunc(int x, int y) {
    g_lastx = x;
    g_lasty = y;
}

void MouseMotionFunc(unsigned state, int x, int y) {
    auto dx = float(x - g_lastx);
    auto dy = float(y - g_lasty);

    g_lastx = x;
    g_lasty = y;

    if (state & SDL_BUTTON_RMASK) {
        const float kSensitivity = DegToRad(0.1f);
        const float kMaxDelta = FLT_MAX;

        g_camAngle.x -= Clamp(dx * kSensitivity, -kMaxDelta, kMaxDelta);
        g_camAngle.y -= Clamp(dy * kSensitivity, -kMaxDelta, kMaxDelta);
    }
}

bool g_Error = false;

void ErrorCallback(NvFlexErrorSeverity severity, const char *msg, const char *file, int line) {
    printf("Flex: %s - %s:%d\n", msg, file, line);
    g_Error = (severity == eNvFlexLogError);
    //assert(0); asserts are bad for TeamCity
}

void ControllerButtonEvent(SDL_ControllerButtonEvent event) {
    // map controller buttons to keyboard keys
    if (event.type == SDL_CONTROLLERBUTTONDOWN) {
        InputKeyboardDown(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);
        InputArrowKeysDown(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);

        if (event.button == SDL_CONTROLLER_BUTTON_LEFT_TRIGGER) {
            // Handle picking events using the game controller
            g_lastx = g_screenWidth / 2;
            g_lasty = g_screenHeight / 2;
            g_lastb = 1;

            // record that we need to update the picked particle
            g_mousePicked = true;
        }
    } else {
        InputKeyboardUp(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);
        InputArrowKeysUp(GetKeyFromGameControllerButton(SDL_GameControllerButton(event.button)), 0, 0);

        if (event.button == SDL_CONTROLLER_BUTTON_LEFT_TRIGGER) {
            // Handle picking events using the game controller
            g_lastx = g_screenWidth / 2;
            g_lasty = g_screenHeight / 2;
            g_lastb = -1;
        }
    }
}

void ControllerDeviceUpdate() {
    if (SDL_NumJoysticks() > 0) {
        SDL_JoystickEventState(SDL_ENABLE);
        if (SDL_IsGameController(0)) {
            g_gamecontroller = SDL_GameControllerOpen(0);
        }
    }
}

void SDLInit(const char *title) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) <
        0)    // Initialize SDL's Video subsystem and game controllers
        printf("Unable to initialize SDL");

    unsigned int flags = SDL_WINDOW_RESIZABLE;

    if (g_graphics == 0) {
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL;
    }

    g_window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                g_screenWidth, g_screenHeight, flags);

    g_windowId = SDL_GetWindowID(g_window);
}

