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

#ifndef NV_FLEX_H
#define NV_FLEX_H

//! \cond HIDDEN_SYMBOLS
#if _WIN32
#define NV_FLEX_API __declspec(dllexport)
#else
#define NV_FLEX_API
#endif

// least 2 significant digits define minor version, eg: 10 -> version 0.10
#define NV_FLEX_VERSION 200

//! \endcond

/** \file NvFlex.h
 * The main include file for the core Flex solver.
 */

#include "float.h"

extern "C" {

/**
 * Opaque type representing a library that can create FlexSolvers, FlexTriangleMeshes, and NvFlexBuffers
 */
typedef struct NvFlexLibrary NvFlexLibrary;

/**
 * Opaque type representing a collection of particles and constraints
 */
typedef struct NvFlexSolver NvFlexSolver;

/**
 * Opaque type representing a data buffer, type and contents depends on usage, see NvFlexAllocBuffer()
 */
typedef struct NvFlexBuffer NvFlexBuffer;

/**
 * Controls behavior of NvFlexMap()
 */
enum NvFlexMapFlags
{
   	eNvFlexMapWait		= 0,	//!< Calling thread will be blocked until buffer is ready for access, default
   	eNvFlexMapDoNotWait = 1,	//!< Calling thread will check if buffer is ready for access, if not ready then the method will return NULL immediately
};

/**
 * Controls memory space of a NvFlexBuffer, see NvFlexAllocBuffer()
 */
enum NvFlexBufferType
{
   	eNvFlexBufferHost	= 0,	//!< A host mappable buffer, pinned memory on CUDA, staging buffer on DX
   	eNvFlexBufferDevice	= 1,	//!< A device memory buffer, mapping this on CUDA will return a device memory pointer, and will return a buffer pointer on DX
};

/**
 * Controls the relaxation method used by the solver to ensure convergence
 */
enum NvFlexRelaxationMode
{
	eNvFlexRelaxationGlobal = 0,	//!< The relaxation factor is a fixed multiplier on each constraint's position delta
	eNvFlexRelaxationLocal  = 1		//!< The relaxation factor is a fixed multiplier on each constraint's delta divided by the particle's constraint count, convergence will be slower but more reliable
};


enum NvFlexSolverType
{
	eNvFlexSolverPBD,			// XPBD (GPU)
	eNvFlexSolverJacobi,		// Newton - Jacobi (GPU)
	eNvFlexSolverLDLT,			// Newton - LDLT (CPU)
	eNvFlexSolverPCG1,			// Newton - PCG (CPU)
	eNvFlexSolverPCG2,			// Newton - PCG (GPU)
	eNvFlexSolverPCR,			// Newton - PCR (GPU)
	eNvFlexSolverGaussSeidel	// Newton - Gauss Seidel (CPU)
};

enum NvFlexFrictionMode
{
	eNvFlexFrictionModeSingle = 0,	// single friction dir, non-linear cone projection, but can't change direction during linear solve
	eNvFlexFrictionModeDouble = 1,	// two friction dirs, non-linear cone projection, can change direction during linear solve
	eNvFlexFrictionModeFull   = 2	// same as above plus torsional (spinning) friction
};


/**
 * Simulation parameters for a solver
 */
struct NvFlexParams
{
	NvFlexSolverType solverType;		//!< Type of solver to use

	int numIterations;					//!< Number of solver iterations to perform per-substep
	int numInnerIterations;				//!< Number of linear solver iterations
	int numLineIterations;				//!< Number of line search iterations
	int numPostCollisionIterations;		//!< Number of iterations to run after main solve to ensure collisions are satisfied

	float contactRegularization;		//!< Controls diagonal regularization for contact constraints, default: 10^-5
	float systemRegularization;			//!< Controls diagonal regularization for the system applied to all constraints, default: 10^-5
	float systemTolerance;				//!< Controls target error for residual, default: 10^-4
	float systemBeta;					//!< Controls amount of steepest descent to use during PCG / PCR solves, can improve robustness at cost of convergence, default: 0.0
	float warmStart;					//!< Warm start factor [0, 1], default 0.0
	NvFlexFrictionMode frictionMode;	//!< Enable or disable friction features, only for Newton solver

	NvFlexRelaxationMode relaxationMode;//!< How the relaxation is applied inside the solver, only affects the Jacobi PBD solver
	float relaxationFactor;				//!< Control the convergence rate of the parallel solver, default: 1, values greater than 1 may lead to instability

	float geometricStiffness;			//!< Control the amount of geometric stiffness is considered by the system
	
	// scene options
	float gravity[3];					//!< Constant acceleration applied to all particles
	float radius;						//!< The maximum interaction radius for particles
	float solidRestDistance;			//!< The distance non-fluid particles attempt to maintain from each other, must be in the range (0, radius]
	float fluidRestDistance;			//!< The distance fluid particles are spaced at the rest density, must be in the range (0, radius], for fluids this should generally be 50-70% of mRadius, for rigids this can simply be the same as the particle radius

	// common params
	float dynamicFriction;				//!< Coefficient of friction used when colliding against shapes
	float staticFriction;				//!< Coefficient of static friction used when colliding against shapes
	float particleFriction;				//!< Coefficient of friction used when colliding particles
	float restitution;					//!< Coefficient of restitution used when colliding against shapes, particle collisions are always inelastic
	float adhesion;						//!< Controls how strongly particles stick to surfaces they hit, default 0.0, range [0.0, +inf]
	float sleepThreshold;				//!< Particles with a velocity magnitude < this threshold will be considered fixed
	
	float maxSpeed;						//!< The magnitude of particle velocity will be clamped to this value at the end of each step
	float maxAcceleration;				//!< The magnitude of particle acceleration will be clamped to this value at the end of each step (limits max velocity change per-second), useful to avoid popping due to large interpenetrations
	
	float shockPropagation;				//!< Artificially decrease the mass of particles based on height from a fixed reference point, this makes stacks and piles converge faster
	float dissipation;					//!< Damps particle velocity based on how many particle contacts it has
	float damping;						//!< Viscous drag force, applies a force proportional, and opposite to the particle velocity

	// cloth params
	float wind[3];						//!< Constant acceleration applied to particles that belong to dynamic triangles, drag needs to be > 0 for wind to affect triangles
	float drag;							//!< Drag force applied to particles belonging to dynamic triangles, proportional to velocity^2*area in the negative velocity direction
	float lift;							//!< Lift force applied to particles belonging to dynamic triangles, proportional to velocity^2*area in the direction perpendicular to velocity and (if possible), parallel to the plane normal

	// fluid params
	float cohesion;						//!< Control how strongly particles hold each other together, default: 0.025, range [0.0, +inf]
	float surfaceTension;				//!< Controls how strongly particles attempt to minimize surface area, default: 0.0, range: [0.0, +inf]    
	float viscosity;					//!< Smoothes particle velocities using XSPH viscosity
	float vorticityConfinement;			//!< Increases vorticity by applying rotational forces to particles
	float anisotropyScale;				//!< Control how much anisotropy is present in resulting ellipsoids for rendering, if zero then anisotropy will not be calculated, see NvFlexGetAnisotropy()
	float anisotropyMin;				//!< Clamp the anisotropy scale to this fraction of the radius
	float anisotropyMax;				//!< Clamp the anisotropy scale to this fraction of the radius
	float smoothing;					//!< Control the strength of Laplacian smoothing in particles for rendering, if zero then smoothed positions will not be calculated, see NvFlexGetSmoothParticles()
	float solidPressure;				//!< Add pressure from solid surfaces to particles
	float freeSurfaceDrag;				//!< Drag force applied to boundary fluid particles
	float buoyancy;						//!< Gravity is scaled by this value for fluid particles

	// diffuse params
	float diffuseThreshold;				//!< Particles with kinetic energy + divergence above this threshold will spawn new diffuse particles
	float diffuseBuoyancy;				//!< Scales force opposing gravity that diffuse particles receive
	float diffuseDrag;					//!< Scales force diffuse particles receive in direction of neighbor fluid particles
	int diffuseBallistic;				//!< The number of neighbors below which a diffuse particle is considered ballistic
	float diffuseLifetime;				//!< Time in seconds that a diffuse particle will live for after being spawned, particles will be spawned with a random lifetime in the range [0, diffuseLifetime]

	// collision params
	float collisionDistance;			//!< Distance particles maintain against shapes, note that for robust collision against triangle meshes this distance should be greater than zero
	float particleCollisionMargin;		//!< Increases the radius used during neighbor finding, this is useful if particles are expected to move significantly during a single step to ensure contacts aren't missed on subsequent iterations
	float shapeCollisionMargin;			//!< Increases the radius used during contact finding against kinematic shapes

	float planes[8][4];					//!< Collision planes in the form ax + by + cz + d = 0
	int numPlanes;						//!< Num collision planes

	int scalingMode;					//!< Solver debug option
	bool substepMode;					//!< Experimental mode

};


/**
 * Flags that control a particle's behavior and grouping, use NvFlexMakePhase() to construct a valid 32bit phase identifier
 */
enum NvFlexPhase
{
	eNvFlexPhaseGroupMask			= 0x000fffff,	//!< Bits [ 0, 19] represent the particle group for controlling collisions
	eNvFlexPhaseFlagsMask			= 0x00f00000,	//!< Bits [20, 23] hold flags about how the particle behave 
	eNvFlexPhaseShapeChannelMask	= 0x7f000000,	//!< Bits [24, 30] hold flags representing what shape collision channels particles will collide with, see NvFlexMakeShapeFlags() (highest bit reserved for now)
	
	eNvFlexPhaseSelfCollide			= 1 << 20,		//!< If set this particle will interact with particles of the same group
	eNvFlexPhaseSelfCollideFilter	= 1 << 21,		//!< If set this particle will ignore collisions with particles closer than the radius in the rest pose, this flag should not be specified unless valid rest positions have been specified using NvFlexSetRestParticles()
	eNvFlexPhaseFluid				= 1 << 22,		//!< If set this particle will generate fluid density constraints for its overlapping neighbors
	eNvFlexPhaseUnused				= 1 << 23,		//!< Reserved
	
	eNvFlexPhaseShapeChannel0		= 1 << 24,		//!< Particle will collide with shapes with channel 0 set (see NvFlexMakeShapeFlags())
	eNvFlexPhaseShapeChannel1		= 1 << 25,		//!< Particle will collide with shapes with channel 1 set (see NvFlexMakeShapeFlags())
	eNvFlexPhaseShapeChannel2		= 1 << 26,		//!< Particle will collide with shapes with channel 2 set (see NvFlexMakeShapeFlags())
	eNvFlexPhaseShapeChannel3		= 1 << 27,		//!< Particle will collide with shapes with channel 3 set (see NvFlexMakeShapeFlags())
	eNvFlexPhaseShapeChannel4		= 1 << 28,		//!< Particle will collide with shapes with channel 4 set (see NvFlexMakeShapeFlags())
	eNvFlexPhaseShapeChannel5		= 1 << 29,		//!< Particle will collide with shapes with channel 5 set (see NvFlexMakeShapeFlags())
	eNvFlexPhaseShapeChannel6		= 1 << 30,		//!< Particle will collide with shapes with channel 6 set (see NvFlexMakeShapeFlags())	
};


/**
 * Generate a bit set for the particle phase, this is a helper method to simply combine the
 * group id and bit flags into a single integer.
 *
 * @param[in] group The index of the group for this particle, should be an integer < 2^20
 * @param[in] particleFlags A combination of the phase flags which should be a combination of eNvFlexPhaseSelfCollide, eNvFlexPhaseSelfCollideFilter, and eNvFlexPhaseFluid
 * @param[in] shapeChannels A combination of eNvFlexPhaseShapeChannel* flags that control which shapes will be collided against, particles will only collide against shapes that share at least one set channel, see NvFlexMakeShapeFlagsWithChannels()
 */
NV_FLEX_API inline int NvFlexMakePhaseWithChannels(int group, int particleFlags, int shapeChannels) { return (group & eNvFlexPhaseGroupMask) | (particleFlags & eNvFlexPhaseFlagsMask) | (shapeChannels & eNvFlexPhaseShapeChannelMask); }

/**
 * Deprecated helper method to generates a phase with all shape channels set
 */
NV_FLEX_API inline int NvFlexMakePhase(int group, int particleFlags) { return NvFlexMakePhaseWithChannels(group, particleFlags, eNvFlexPhaseShapeChannelMask); }


/**
 * Time spent in each section of the solver update, times in GPU seconds, see NvFlexUpdateSolver()
 */
struct NvFlexTimers
{
	float predict;				//!< Time spent in prediction
	float createCellIndices;	//!< Time spent creating grid indices
	float sortCellIndices;		//!< Time spent sorting grid indices
	float createGrid;			//!< Time spent creating grid
	float reorder;				//!< Time spent reordering particles
	float collideParticles;		//!< Time spent finding particle neighbors
	float collideShapes;		//!< Time spent colliding convex shapes
	float collideTriangles;		//!< Time spent colliding triangle shapes
	float collideFields;		//!< Time spent colliding signed distance field shapes
	float calculateDensity;		//!< Time spent calculating fluid density
	float solveDensities;		//!< Time spent solving density constraints
	float solveVelocities;		//!< Time spent solving velocity constraints
	float solveShapes;			//!< Time spent solving rigid body constraints
	float solveSprings;			//!< Time spent solving distance constraints
	float solveTetra;			//!< Time spent solving tetrahedral constraints
	float solveContacts;		//!< Time spent solving contact constraints
	float solveInflatables;		//!< Time spent solving pressure constraints
	float applyDeltas;	        //!< Time spent adding position deltas to particles
	float calculateAnisotropy;	//!< Time spent calculating particle anisotropy for fluid
	float updateDiffuse;		//!< Time spent updating diffuse particles
	float updateTriangles;		//!< Time spent updating dynamic triangles
	float updateNormals;		//!< Time spent updating vertex normals
	float finalize;				//!< Time spent finalizing state
	float updateBounds;			//!< Time spent updating particle bounds
	float buildGlobal;			//!< Time spent build system matrices
	float evaluateGlobal;		//!< Time spent evaluate system matrices
	float solveGlobal;			//!< Time spent solving system matrices
	float updateGlobal;			//!< Time spent updating system after a solve

	float total;				//!< Sum of all timers above
};

/**
 * Flex error return codes
 */
enum NvFlexErrorSeverity
{
	eNvFlexLogError		=  0,	//!< Error messages
	eNvFlexLogInfo		=  1,	//!< Information messages
	eNvFlexLogWarning	=  2,	//!< Warning messages
	eNvFlexLogDebug		=  4,	//!< Used only in debug version of dll
	eNvFlexLogAll		= -1,	//!< All log types
};

 
/** Defines the set of stages at which callbacks may be registered 
 */
enum NvFlexSolverCallbackStage
{
	eNvFlexStageIterationStart,	//!< Called at the beginning of each constraint iteration
	eNvFlexStageIterationEnd,	//!< Called at the end of each constraint iteration
	eNvFlexStageSubstepBegin,	//!< Called at the beginning of each substep after the prediction step has been completed
	eNvFlexStageSubstepEnd,		//!< Called at the end of each substep after the velocity has been updated by the constraints
	eNvFlexStageUpdateEnd,		//!< Called at the end of solver update after the final substep has completed
	eNvFlexStageCount,			//!< Number of stages
};


/** Structure containing pointers to the internal solver data that is passed to each registered solver callback
 *
 *  @remarks Pointers to internal data are only valid for the lifetime of the callback and should not be stored.
 *  However, it is safe to launch kernels and memory transfers using the device pointers.
 *
 *  @remarks Because Flex re-orders particle data internally for performance, the particle data in the callback is not
 *  in the same order as it was provided to the API. The callback provides arrays which map original particle indices
 *  to sorted positions and vice-versa.
 *
 *  @remarks Particle positions may be modified during any callback, but velocity modifications should only occur during 
 *  the eNvFlexStageUpdateEnd stage, otherwise any velocity changes will be discarded.
 */
struct NvFlexSolverCallbackParams
{
	NvFlexSolver* solver;				//!< Pointer to the solver that the callback is registered to
	void* userData;						//!< Pointer to the user data provided to NvFlexRegisterSolverCallback()

	float* particles;					//!< Device pointer to the active particle basic data in the form x,y,z,1/m
	float* velocities;					//!< Device pointer to the active particle velocity data in the form x,y,z,w (last component is not used)
	int* phases;						//!< Device pointer to the active particle phase data

	int numActive;						//!< The number of active particles returned, the callback data only return pointers to active particle data, this is the same as NvFlexGetActiveCount()
	
	float dt;							//!< The per-update time-step, this is the value passed to NvFlexUpdateSolver()

	const int* originalToSortedMap;		//!< Device pointer that maps the sorted callback data to the original position given by SetParticles()
	const int* sortedToOriginalMap;		//!< Device pointer that maps the original particle index to the index in the callback data structure
};

/** Solver callback definition, see NvFlexRegisterSolverCallback()
 */
struct NvFlexSolverCallback
{
	/** User data passed to the callback*/
	void* userData;
	
	/** Function pointer to a callback method */
	void (*function)(NvFlexSolverCallbackParams params);
};

/**
 * Function pointer type for error reporting callbacks
 */
typedef void (*NvFlexErrorCallback)(NvFlexErrorSeverity type, const char* msg, const char* file, int line);




/** Defines the different compute backends that Flex can use
*/
enum NvFlexComputeType
{
	eNvFlexCUDA,		//!< Use CUDA compute for Flex, the application must link against the CUDA libraries
	eNvFlexD3D11,		//!< Use DirectX 11 compute for Flex, the application must link against the D3D libraries
	eNvFlexD3D12,		//!< Use DirectX 12 compute for Flex, the application must link against the D3D libraries
};


/** Descriptor used to initialize Flex
*/
struct NvFlexInitDesc
{
	int deviceIndex;				//!< The GPU device index that should be used, if there is already a CUDA context on the calling thread then this parameter will be ignored and the active CUDA context used. Otherwise a new context will be created using the suggested device ordinal.
	bool enableExtensions;			//!< Enable or disable NVIDIA/AMD extensions in DirectX, can lead to improved performance.
	void* renderDevice;				//!< Direct3D device to use for simulation, if none is specified a new device and context will be created.
	void* renderContext;			//!< Direct3D context that the app is using for rendering. In DirectX 12 this should be a ID3D12CommandQueue pointer.
	void* computeContext;           //!< Direct3D context to use for simulation, if none is specified a new context will be created, in DirectX 12 this should be a pointer to the ID3D12CommandQueue where compute operations will take place. 
	bool runOnRenderContext;		//!< If true, run Flex on D3D11 render context, or D3D12 direct queue. If false, run on a D3D12 compute queue, or vendor specific D3D11 compute queue, allowing compute and graphics to run in parallel on some GPUs.

	NvFlexComputeType computeType;	//!< Set to eNvFlexD3D11 if DirectX 11 should be used, eNvFlexD3D12 for DirectX 12, this must match the libraries used to link the application
};

/**
* Initialize library, should be called before any other API function.
*
*
* @param[in] version The version number the app is expecting, should almost always be NV_FLEX_VERSION
* @param[in] errorFunc The callback used for reporting errors.
* @param[in] desc The NvFlexInitDesc struct defining the device ordinal, D3D device/context and the type of D3D compute being used
* @return A pointer to a library instance that can be used to allocate shared object such as triangle meshes, buffers, etc
*/
NV_FLEX_API NvFlexLibrary* NvFlexInit(int version = NV_FLEX_VERSION, NvFlexErrorCallback errorFunc = 0, NvFlexInitDesc* desc = 0);

/**
 * Shutdown library, users should manually destroy any previously created   
 * solvers to ensure memory is freed before calling this method. If a new CUDA context was created during NvFlexInit() then it will be destroyed.
 *
 * @param[in] lib The library intance to use
 */
NV_FLEX_API void NvFlexShutdown(NvFlexLibrary* lib);

/**
 * Get library version number
 */
NV_FLEX_API int NvFlexGetVersion();

/** 
 * Controls which features are enabled, choosing a simple option will disable features and can lead to better performance and reduced memory usage
 */
enum NvFlexFeatureMode
{
	eNvFlexFeatureModeDefault			= 0,	//!< All features enabled
	eNvFlexFeatureModeSimpleSolids		= 1,	//!< Simple per-particle collision (no per-particle SDF normals, no fluids)
	eNvFlexFeatureModeSimpleFluids		= 2,	//!< Simple single phase fluid-only particles (no solids)
};

/**
 * Describes the creation time parameters for the solver
 */
struct NvFlexSolverDesc
{
	NvFlexFeatureMode featureMode;	//!< Control which features are enabled

	int maxParticles;				//!< Maximum number of regular particles in the solver
	int maxDiffuseParticles;		//!< Maximum number of diffuse particles in the solver
	int maxNeighborsPerParticle;	//!< Maximum number of neighbors per-particle, for solids this can be around 32, for fluids up to 128 may be necessary depending on smoothing radius
	
	int maxRigidShapePairs;			//!< Maximum rigid broad phase pairs, default: 128*1024
	int maxRigidBodyContacts;		//!< Maximum rigid narrow phase contacts, default: 256*1024	
	int maxRigidSoftContacts;		//!< Maximum rigid soft narrow phase contacts, default: 64*1024

};

/**
 * Initialize the solver desc to its default values
 * @param[in] desc Pointer to a description structure that will be initialized to default values
 */
NV_FLEX_API void NvFlexSetSolverDescDefaults(NvFlexSolverDesc* desc);

/**
 * Create a new particle solver
 *
 * @param[in] lib The library instance to use
 * @param[in] desc Pointer to a solver description structure used to create the solver
 */
NV_FLEX_API NvFlexSolver* NvFlexCreateSolver(NvFlexLibrary* lib, const NvFlexSolverDesc* desc);
/**
 * Delete a particle solver
 *
 * @param[in] solver A valid solver pointer created from NvFlexCreateSolver()
 */
NV_FLEX_API void NvFlexDestroySolver(NvFlexSolver* solver);

/**
 * Return the library associated with a solver
 *
 * @param[in] solver A valid solver created with NvFlexCreateSolver()
 * @return A library pointer
 */
NV_FLEX_API NvFlexLibrary* NvFlexGetSolverLibrary(NvFlexSolver* solver);

/**
 * Return the solver desc that was used to create a solver
 *
 * @param[in] solver Pointer to a valid Flex solver
 * @param[in] desc Pointer to a desc structure
 */
NV_FLEX_API void NvFlexGetSolverDesc(NvFlexSolver* solver, NvFlexSolverDesc* desc);

/** Registers a callback for a solver stage, the callback will be invoked from the same thread that calls NvFlexUpdateSolver().
 *
 * @param[in] solver A valid solver
 * @param[in] function A pointer to a function that will be called during the solver update
 * @param[in] stage The stage of the update at which the callback function will be called
 *
 * @return The previously registered callback for this slot, this allows multiple users to chain callbacks together
 */
NV_FLEX_API NvFlexSolverCallback NvFlexRegisterSolverCallback(NvFlexSolver* solver, NvFlexSolverCallback function, NvFlexSolverCallbackStage stage);

/**
 * Integrate particle solver forward in time. Below is an example of how to step Flex in the context of a simple game loop:
 *
 \code{.c}
	
	NvFlexLibrary* library = NvFlexInit();
	NvFlexSolver* solver = NvFlexCreateSolver(library);

	NvFlexBuffer* particleBuffer = NvFlexAllocBuffer(library, n, sizeof(Vec4), eNvFlexBufferHost);
	NvFlexBuffer* velocityBuffer = NvFlexAllocBuffer(library, n, sizeof(Vec4), eNvFlexBufferHost);
	NvFlexBuffer* phaseBuffer = NvFlexAllocBuffer(library, n, sizeof(int), eNvFlexBufferHost);

	while(!done)
	{
		// map buffers for reading / writing
		float4* particles = (float4*)NvFlexMap(particles, eNvFlexMapWait);
		float3* velocities  = (float3*)NvFlexMap(velocities, eNvFlexMapWait);
		int* phases = (int*)NvFlexMap(phases, eNvFlexMapWait);

		// spawn (user method)
		SpawnParticles(particles, velocities, phases);

		// render (user method)
		RenderParticles(particles, velocities, phases);

		// unmap buffers
		NvFlexUnmap(particleBuffer);
		NvFlexUnmap(velocityBuffer);
		NvFlexUnmap(phaseBuffer);

		// write to device (async)
		NvFlexSetParticles(particleBuffer, n);
		NvFlexSetVelocities(velocityBuffer, n);
		NvFlexSetPhases(phaseBuffer, n);

		// tick
		NvFlexUpdateSolver(solver, dt, 1, NULL);

		// read back (async)
		NvFlexGetParticles(particleBuffer, n);
		NvFlexGetVelocities(velocityBuffer, n);
		NvFlexGetPhases(phaseBuffer, n);
	}

	NvFlexFreeBuffer(particleBuffer);
	NvFlexFreeBuffer(velocityBuffer);
	NvFlexFreeBuffer(phaseBuffer);

	NvFlexDestroySolver(solver);
	NvFlexShutdown(library);


 \endcode
 *
 * @param[in] solver A valid solver
 * @param[in] dt Time to integrate the solver forward in time by
 * @param[in] substeps The time dt will be divided into the number of sub-steps given by this parameter
 * @param[in] enableTimers Whether to enable per-kernel timers for profiling. Note that profiling can substantially slow down overall performance so this param should only be true in non-release builds
 */
NV_FLEX_API void NvFlexUpdateSolver(NvFlexSolver* solver, float dt, int substeps, bool enableTimers);


/**
 * Initialize solver parameter defaults
 *
 * @param[in] params Parameters structure in host memory, see NvFlexParams
 */
NV_FLEX_API void NvFlexInitParams(NvFlexParams* params);

/**
 * Update solver paramters
 *
 * @param[in] solver A valid solver
 * @param[in] params Parameters structure in host memory, see NvFlexParams
 */
NV_FLEX_API void NvFlexSetParams(NvFlexSolver* solver, const NvFlexParams* params);

/**
 * Retrieve solver paramters, default values will be set at solver creation time
 *
 * @param[in] solver A valid solver
 * @param[out] params Parameters structure in host memory, see NvFlexParams
 */

NV_FLEX_API void NvFlexGetParams(NvFlexSolver* solver, NvFlexParams* params);

/**
 * Describes a source and destination buffer region for performing a copy operation.
 */
struct NvFlexCopyDesc
{
	int srcOffset;			//<! Offset in elements from the start of the source buffer to begin reading from
	int dstOffset;			//<! Offset in elements from the start of the destination buffer to being writing to
	int elementCount;		//<! Number of elements to copy
};

/**
 * Set the active particles indices in the solver
 * 
 * @param[in] solver A valid solver
 * @param[in] indices Holds the indices of particles that have been made active
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexSetActive(NvFlexSolver* solver, NvFlexBuffer* indices, const NvFlexCopyDesc* desc);

/**
 * Return the active particle indices
 * 
 * @param[in] solver A valid solver
 * @param[out] indices a buffer of indices at least activeCount in length
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexGetActive(NvFlexSolver* solver, NvFlexBuffer* indices, const NvFlexCopyDesc* desc);

/**
 * Set the total number of active particles
 * 
 * @param[in] solver A valid solver
 * @param[in] n The number of active particles, the first n indices in the active particles array will be used as the active count
 */
NV_FLEX_API void NvFlexSetActiveCount(NvFlexSolver* solver, int n);

/**
 * Return the number of active particles in the solver
 * 
 * @param[in] solver A valid solver
 * @return The number of active particles in the solver
 */
NV_FLEX_API int NvFlexGetActiveCount(NvFlexSolver* solver);

/**
 * Set the particles state of the solver, a particle consists of 4 floating point numbers, its x,y,z position followed by its inverse mass (1/m)
 * 
 * @param[in] solver A valid solver
 * @param[in] p Pointer to a buffer of particle data, should be 4*n in length
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 *
 */
NV_FLEX_API void NvFlexSetParticles(NvFlexSolver* solver, NvFlexBuffer* p, const NvFlexCopyDesc* desc);

/**
 * Get the particles state of the solver, a particle consists of 4 floating point numbers, its x,y,z position followed by its inverse mass (1/m)
 * 
 * @param[in] solver A valid solver
 * @param[out] p Pointer to a buffer of 4*n floats that will be filled out with the particle data, can be either a host or device pointer
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexGetParticles(NvFlexSolver* solver, NvFlexBuffer* p, const NvFlexCopyDesc* desc);

/**
 * Set the particle positions in their rest state, if eNvFlexPhaseSelfCollideFilter is set on the particle's
 * phase attribute then particles that overlap in the rest state will not generate collisions with each other
 * 
 * @param[in] solver A valid solver
 * @param[in] p Pointer to a buffer of particle data, should be 4*n in length
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 *
 */
NV_FLEX_API void NvFlexSetRestParticles(NvFlexSolver* solver, NvFlexBuffer* p, const NvFlexCopyDesc* desc);

/**
 * Get the particle positions in their rest state
 * 
 * @param[in] solver A valid solver
 * @param[in] p Pointer to a buffer of particle data, should be 4*n in length
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 *
 */
NV_FLEX_API void NvFlexGetRestParticles(NvFlexSolver* solver, NvFlexBuffer* p, const NvFlexCopyDesc* desc);


/**
 * Get the Laplacian smoothed particle positions for rendering, see NvFlexParams::smoothing
 * 
 * @param[in] solver A valid solver
 * @param[out] p Pointer to a buffer of 4*n floats that will be filled out with the data, can be either a host or device pointer
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexGetSmoothParticles(NvFlexSolver* solver, NvFlexBuffer* p, const NvFlexCopyDesc* desc);

/**
 * Set the particle velocities, each velocity is a 3-tuple of x,y,z floating point values
 * 
 * @param[in] solver A valid solver
 * @param[in] v Pointer to a buffer of 3*n floats
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 *
 */
NV_FLEX_API void NvFlexSetVelocities(NvFlexSolver* solver, NvFlexBuffer* v, const NvFlexCopyDesc* desc);
/**
 * Get the particle velocities, each velocity is a 3-tuple of x,y,z floating point values
 * 
 * @param[in] solver A valid solver
 * @param[out] v Pointer to a buffer of 3*n floats that will be filled out with the data, can be either a host or device pointer
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexGetVelocities(NvFlexSolver* solver, NvFlexBuffer* v, const NvFlexCopyDesc* desc);

/**
 * Set the particles phase id array, each particle has an associated phase id which 
 * controls how it interacts with other particles. Particles with phase 0 interact with all
 * other phase types.
 *
 * Particles with a non-zero phase id only interact with particles whose phase differs 
 * from theirs. This is useful, for example, to stop particles belonging to a single
 * rigid shape from interacting with each other.
 * 
 * Phase 0 is used to indicate fluid particles when NvFlexParams::mFluid is set.
 * 
 * @param[in] solver A valid solver
 * @param[in] phases Pointer to a buffer of n integers containing the phases
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 *
 */
NV_FLEX_API void NvFlexSetPhases(NvFlexSolver* solver, NvFlexBuffer* phases, const NvFlexCopyDesc* desc);
/**
 * Get the particle phase ids
 * 
 * @param[in] solver A valid solver
 * @param[out] phases Pointer to a buffer of n integers that will be filled with the phase data, can be either a host or device pointer
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexGetPhases(NvFlexSolver* solver, NvFlexBuffer* phases, const NvFlexCopyDesc* desc);

/**
 * Set per-particle normals to the solver, these will be overwritten after each simulation step, but can be used to initialize the normals to valid values
 * 
 * @param[in] solver A valid solver
 * @param[in] normals Pointer to a buffer of normals, should be 4*n in length
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexSetNormals(NvFlexSolver* solver, NvFlexBuffer* normals, const NvFlexCopyDesc* desc);

/**
 * Get per-particle normals from the solver, these are the world-space normals computed during surface tension, cloth, and rigid body calculations
 * 
 * @param[in] solver A valid solver
 * @param[out] normals Pointer to a buffer of normals, should be 4*n in length
 * @param[in] desc Describes the copy region, if NULL the solver will try to access the entire buffer (maxParticles length)
 */
NV_FLEX_API void NvFlexGetNormals(NvFlexSolver* solver, NvFlexBuffer* normals, const NvFlexCopyDesc* desc);


/**
 * Set distance constraints for the solver. Each distance constraint consists of two particle indices
 * stored consecutively, a rest-length, and a stiffness value. These are not springs in the traditional
 * sense, but behave somewhat like a traditional spring when lowering the stiffness coefficient.
 * 
 * @param[in] solver A valid solver
 * @param[in] indices Pointer to the spring indices array, should be 2*numSprings length, 2 indices per-spring
 * @param[in] restLengths Pointer to a buffer of rest lengths, should be numSprings length
 * @param[in] stiffness Pointer to the spring stiffness coefficents, should be numSprings in length, a negative stiffness value represents a tether constraint
 * @param[in] numSprings The number of springs to set
 *
 */
NV_FLEX_API void NvFlexSetSprings(NvFlexSolver* solver, NvFlexBuffer* indices, NvFlexBuffer* restLengths, NvFlexBuffer* stiffness, int numSprings);


/**
* Set rigid body to particle constraints for the solver. Each attachment constraint consists of one rigid body and one particle indices stored 
* in one NVFlexRigidParticleAttachment struct, and a set of distances in x,y and z in the rigid body frame.
* @param[in] solver A valid solver
* @param[in] attachments Pointer to the NVFlexRigidParticleAttachment struct
* @param[in] numAttachments The number of springs to set
*
*/
NV_FLEX_API void NvFlexSetRigidParticleAttachments(NvFlexSolver* s, NvFlexBuffer* attachments, int numAttachments);

/**
 * Get the distance constraints from the solver
 * 
 * @param[in] solver A valid solver
 * @param[out] indices Pointer to the spring indices array, should be 2*numSprings length, 2 indices per-spring
 * @param[out] restLengths Pointer to a buffer of rest lengths, should be numSprings length
 * @param[out] stiffness Pointer to the spring stiffness coefficents, should be numSprings in length, a negative stiffness value represents a unilateral tether constraint (only resists stretching, not compression), valid range [-1, 1]
 * @param[in] numSprings The number of springs to get
 */
NV_FLEX_API void NvFlexGetSprings(NvFlexSolver* solver, NvFlexBuffer* indices, NvFlexBuffer* restLengths, NvFlexBuffer* stiffness, int numSprings);


struct NvFlexFEMMaterial
{
	float rows[6][6];
	float damping[6];
};

NV_FLEX_API void NvFlexSetFEMMaterials(NvFlexSolver* solver, NvFlexFEMMaterial* materials, int numMaterials);
NV_FLEX_API void NvFlexSetFEMGeometry(NvFlexSolver* solver, NvFlexBuffer* tetrahedraIndices, NvFlexBuffer* tetrahedraRestPoses, NvFlexBuffer* tetrahedraMaterials, int numTetrahedra);
NV_FLEX_API void NvFlexGetFEMStress(NvFlexSolver* solver, NvFlexBuffer* stressTensors);

/**
 * Set rigid body constraints for the solver. 
 * @note A particle should not belong to more than one rigid body at a time.
 * 
 * @param[in] solver A valid solver
 * @param[in] offsets Pointer to a buffer of start offsets for a rigid in the indices array, should be numRigids+1 in length, the first entry must be 0
 * @param[in] indices Pointer to a buffer of indices for the rigid bodies, the indices for the jth rigid body start at indices[offsets[j]] and run to indices[offsets[j+1]] exclusive
 * @param[in] restPositions Pointer to a buffer of local space positions relative to the rigid's center of mass (average position), this should be at least 3*numIndices in length in the format x,y,z
 * @param[in] restNormals Pointer to a buffer of local space normals, this should be at least 4*numIndices in length in the format x,y,z,w where w is the (negative) signed distance of the particle inside its shape
 * @param[in] stiffness Pointer to a buffer of rigid stiffness coefficents, should be numRigids in length, valid values in range [0, 1]
 * @param[in] thresholds Pointer to a buffer of plastic deformation threshold coefficients, should be numRigids in length
 * @param[in] creeps Pointer to a buffer of plastic deformation creep coefficients, should be numRigids in length, valid values in range [0, 1]
 * @param[in] rotations Pointer to a buffer of quaternions (4*numRigids in length)
 * @param[in] translations Pointer to a buffer of translations of the center of mass (3*numRigids in length)
 * @param[in] numRigids The number of rigid bodies to set
 * @param[in] numIndices The number of indices in the indices array
 *
 */
NV_FLEX_API void NvFlexSetRigids(NvFlexSolver* solver, NvFlexBuffer* offsets, NvFlexBuffer* indices, NvFlexBuffer* restPositions, NvFlexBuffer* restNormals, NvFlexBuffer* stiffness, NvFlexBuffer* thresholds, NvFlexBuffer* creeps, NvFlexBuffer* rotations, NvFlexBuffer* translations, int numRigids, int numIndices);

/**
 * Retrive the rigid body shape matching constraints and transforms, if any buffer pointers are NULL then they will be ignored
 * This method supersedes the previous NvFlexGetRigidTransforms method and can be used to retrieve modified rest positions from plastic deformation.
 * 
 * @param[in] solver A valid solver
 * @param[in] offsets Pointer to a buffer of start offsets for a rigid in the indices array, should be numRigids+1 in length, the first entry must be 0
 * @param[in] indices Pointer to a buffer of indices for the rigid bodies, the indices for the jth rigid body start at indices[offsets[j]] and run to indices[offsets[j+1]] exclusive
 * @param[in] restPositions Pointer to a buffer of local space positions relative to the rigid's center of mass (average position), this should be at least 3*numIndices in length in the format x,y,z
 * @param[in] restNormals Pointer to a buffer of local space normals, this should be at least 4*numIndices in length in the format x,y,z,w where w is the (negative) signed distance of the particle inside its shape
 * @param[in] stiffness Pointer to a buffer of rigid stiffness coefficents, should be numRigids in length, valid values in range [0, 1]
 * @param[in] thresholds Pointer to a buffer of plastic deformation threshold coefficients, should be numRigids in length
 * @param[in] creeps Pointer to a buffer of plastic deformation creep coefficients, should be numRigids in length, valid values in range [0, 1]
 * @param[in] rotations Pointer to a buffer of quaternions (4*numRigids in length with the imaginary elements in the x,y,z components)
 * @param[in] translations Pointer to a buffer of translations of the center of mass (3*numRigids in length)
 */
NV_FLEX_API void NvFlexGetRigids(NvFlexSolver* solver, NvFlexBuffer* offsets, NvFlexBuffer* indices, NvFlexBuffer* restPositions, NvFlexBuffer* restNormals, NvFlexBuffer* stiffness, NvFlexBuffer* thresholds, NvFlexBuffer* creeps, NvFlexBuffer* rotations, NvFlexBuffer* translations);

/**
 * An opaque type representing a static triangle mesh in the solver
 */
typedef unsigned int NvFlexTriangleMeshId;

/**
* An opaque type representing a static curve in the solver
*/

typedef unsigned int NvFlexCurveId;

/**
 * An opaque type representing a signed distance field collision shape in the solver.
 */
typedef unsigned int NvFlexDistanceFieldId;

/**
 * An opaque type representing a convex mesh collision shape in the solver.
 * Convex mesh shapes may consist of up to 64 planes of the form a*x + b*y + c*z + d = 0,
 * particles will be constrained to the outside of the shape.
 */
typedef unsigned int NvFlexConvexMeshId;

enum NvFlexTriFeature
{
	eNvFlexTriFeatureFace	= 1 << 0,
	eNvFlexTriFeatureEdge0	= 1 << 1,
	eNvFlexTriFeatureEdge1	= 1 << 2,
	eNvFlexTriFeatureEdge2	= 1 << 3,
	eNvFlexTriFeatureVertex0 = 1 << 4,
	eNvFlexTriFeatureVertex1 = 1 << 5,
	eNvFlexTriFeatureVertex2 = 1 << 6
};

/**
 * Create triangle mesh geometry, note that meshes may be used by multiple solvers if desired
 * 
 * @param[in] lib The library instance to use
 * @return A pointer to a triangle mesh object
 */
NV_FLEX_API NvFlexTriangleMeshId NvFlexCreateTriangleMesh(NvFlexLibrary* lib);

/**
 * Destroy a triangle mesh created with NvFlexCreateTriangleMesh()
 *
 * @param[in] lib The library instance to use
 * @param[in] mesh A triangle mesh created with NvFlexCreateTriangleMesh()
 */
NV_FLEX_API void NvFlexDestroyTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId mesh);

/**
 * Specifies the triangle mesh geometry (vertices and indices), this method will cause any internal
 * data structures (e.g.: bounding volume hierarchies) to be rebuilt.
 * 
 * @param[in] lib The library instance to use
 * @param[in] mesh A triangle mesh created with NvFlexCreateTriangleMesh()
 * @param[in] vertices Pointer to a buffer of float4 vertex positions
 * @param[in] indices Pointer to a buffer of triangle indices, should be length numTriangles*3
 * @param[in] numVertices The number of vertices in the vertices array
 * @param[in] numTriangles The number of triangles in the mesh
 * @param[in] lower A pointer to a float3 vector holding the lower spatial bounds of the mesh
 * @param[in] upper A pointer to a float3 vector holding the upper spatial bounds of the mesh
 */
NV_FLEX_API void NvFlexUpdateTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId mesh, NvFlexBuffer* vertices, NvFlexBuffer* indices, NvFlexBuffer* features, NvFlexBuffer* vertexAdj, NvFlexBuffer* vertexAdjOffset,  NvFlexBuffer* vertexAdjCount, NvFlexBuffer* edgeAdj, int numVertices, int numTriangles, const float* lower, const float* upper);

/**
 * Retrieve the mesh geometry, size, and bounds
 */
NV_FLEX_API void NvFlexGetTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId mesh, NvFlexBuffer* vertices, NvFlexBuffer* indices, NvFlexBuffer* features, int* numVertices, int* numTriangles, float* lower, float* upper);

/**
 * Retrieve the local space bounds of the mesh, these are the same values specified to NvFlexUpdateTriangleMesh()
 * 
 * @param[in] lib The library instance to use
 * @param[in] mesh Pointer to a triangle mesh object
 * @param[out] lower Pointer to a buffer of 3 floats that the lower mesh bounds will be written to
 * @param[out] upper Pointer to a buffer of 3 floats that the upper mesh bounds will be written to
 */
NV_FLEX_API void NvFlexGetTriangleMeshBounds(NvFlexLibrary* lib, const NvFlexTriangleMeshId mesh, float* lower, float* upper);

/**
* Create curve, note that curves may be used by multiple solvers if desired
*
* @param[in] lib The library instance to use
* @return A pointer to a curve object
*/
NV_FLEX_API NvFlexCurveId NvFlexCreateCurve(NvFlexLibrary* lib);

/**
* Destroy a curve created with NvFlexCreateCurve()
*
* @param[in] lib The library instance to use
* @param[in] curveIndex A curve created with NvFlexCreateCurve()
*/
NV_FLEX_API void NvFlexDestroyCurve(NvFlexLibrary* lib, NvFlexCurveId curveIndex);

/**
* Specifies the curve (vertices)
*
* @param[in] lib The library instance to use
* @param[in] curve A curve created with NvFlexCurve()
* @param[in] vertices Pointer to a buffer of float3 vertex positions
* @param[in] numVertices The number of vertices in the vertices array
*/
void NvFlexUpdateCurve(NvFlexLibrary* lib, NvFlexCurveId curveIndex, NvFlexBuffer* positions, int numVerts);

/**
* Retrieve the mesh geometry, size, and bounds
*/
NV_FLEX_API void NvFlexGetCurve(NvFlexLibrary* lib, NvFlexCurveId curveIndex, NvFlexBuffer* vertices, int* numVertices);

/**
 * Create a signed distance field collision shape, see NvFlexDistanceFieldId for details.
 * 
 * @param[in] lib The library instance to use
 * @return A pointer to a signed distance field object
 */
NV_FLEX_API NvFlexDistanceFieldId NvFlexCreateDistanceField(NvFlexLibrary* lib);

/**
 * Destroy a signed distance field
 * 
 * @param[in] lib The library instance to use
 * @param[in] sdf A signed distance field created with NvFlexCreateDistanceField()
 */
NV_FLEX_API void NvFlexDestroyDistanceField(NvFlexLibrary* lib, NvFlexDistanceFieldId sdf);

/**
 * Update the signed distance field volume data, this method will upload
 * the field data to a 3D texture on the GPU
 * 
 * @param[in] lib The library instance to use
 * @param[in] sdf A signed distance field created with NvFlexCreateDistanceField()
 * @param[in] dimx The x-dimension of the volume data in voxels
 * @param[in] dimy The y-dimension of the volume data in voxels
 * @param[in] dimz The z-dimension of the volume data in voxels
 * @param[in] field The volume data stored such that the voxel at the x,y,z coordinate is addressed as field[z*dimx*dimy + y*dimx + x]
 */
NV_FLEX_API void NvFlexUpdateDistanceField(NvFlexLibrary* lib, NvFlexDistanceFieldId sdf, int dimx, int dimy, int dimz, NvFlexBuffer* field);

/**
 * Create a convex mesh collision shape, see NvFlexConvexMeshId for details.
 * 
 * @param[in] lib The library instance to use
 * @return A pointer to a signed distance field object
 */
NV_FLEX_API NvFlexConvexMeshId NvFlexCreateConvexMesh(NvFlexLibrary* lib);

/**
 * Destroy a convex mesh
 * 
 * @param[in] lib The library instance to use
 * @param[in] convex A a convex mesh created with NvFlexCreateConvexMesh()
 */
NV_FLEX_API void NvFlexDestroyConvexMesh(NvFlexLibrary* lib, NvFlexConvexMeshId convex);

/**
 * Update the convex mesh geometry
 * 
 * @param[in] lib The library instance to use
 * @param[in] convex A valid convex mesh shape created from NvFlexCreateConvexMesh()
 * @param[in] planes An array of planes, each plane consists of 4 floats in the form a*x + b*y + c*z + d = 0
 * @param[in] numPlanes The number of planes in the convex, must be less than 64 planes per-convex
 * @param[in] lower The local space lower bound of the convex shape
 * @param[in] upper The local space upper bound of the convex shape
  */
NV_FLEX_API void NvFlexUpdateConvexMesh(NvFlexLibrary* lib, NvFlexConvexMeshId convex, NvFlexBuffer* vertices, NvFlexBuffer* edges, NvFlexBuffer* planes, int numVertices, int numEdges, int numPlanes, const float* lower, const float* upper);

/**
 * Retrieve the local space bounds of the mesh, these are the same values specified to NvFlexUpdateConvexMesh()
 * 
 * @param[in] lib The library instance to use
 * @param[in] mesh Pointer to a convex mesh object
 * @param[out] lower Pointer to a buffer of 3 floats that the lower mesh bounds will be written to
 * @param[out] upper Pointer to a buffer of 3 floats that the upper mesh bounds will be written to
 */
NV_FLEX_API void NvFlexGetConvexMeshBounds(NvFlexLibrary* lib, NvFlexConvexMeshId mesh, float* lower, float* upper);

/**
 * A basic sphere shape with origin at the center of the sphere and radius
 */
struct NvFlexSphereGeometry
{
	float radius;
};

/**
 * A collision capsule extends along the x-axis with its local origin at the center of the capsule 
 */
struct NvFlexCapsuleGeometry
{
	float radius;
	float halfHeight;
};

/**
 * A simple box with interior [-halfHeight, +halfHeight] along each dimension 
 */
struct NvFlexBoxGeometry
{
	float halfExtents[3];
};

/**
 * A convex mesh instance with non-uniform scale
 */
struct NvFlexConvexMeshGeometry
{
	float scale[3];
	NvFlexConvexMeshId mesh;
};

/**
 * A scaled triangle mesh instance with non-uniform scale
 */
struct NvFlexTriangleMeshGeometry
{
	float scale[3];			//!< The scale of the object from local space to world space
	NvFlexTriangleMeshId mesh;	//!< A triangle mesh pointer created by NvFlexCreateTriangleMesh()
};

/**
 * A scaled signed distance field instance, the local origin of the SDF is at corner of the field corresponding to the first voxel.
 * The field is mapped to the local space volume [0, 1] in each dimension.
 */
struct NvFlexSDFGeometry
{
	float scale;				 //!< Uniform scale of SDF, this corresponds to the world space width of the shape
	NvFlexDistanceFieldId field;	 //!< A signed distance field pointer created by NvFlexCreateDistanceField()
};


/**
 * This union allows collision geometry to be sent to Flex as a flat array of 16-byte data structures,
 * the shape flags array specifies the type for each shape, see NvFlexSetShapes().
 */
union NvFlexCollisionGeometry
{
	NvFlexSphereGeometry sphere;
	NvFlexCapsuleGeometry capsule;
	NvFlexBoxGeometry box;
	NvFlexConvexMeshGeometry convexMesh;
	NvFlexTriangleMeshGeometry triMesh;
	NvFlexSDFGeometry sdf;
};

enum NvFlexCollisionShapeType
{
	eNvFlexShapeSphere			= 0,		//!< A sphere shape, see FlexSphereGeometry
	eNvFlexShapeCapsule			= 1,		//!< A capsule shape, see FlexCapsuleGeometry
	eNvFlexShapeBox				= 2,		//!< A box shape, see FlexBoxGeometry
	eNvFlexShapeConvexMesh		= 3,		//!< A convex mesh shape, see FlexConvexMeshGeometry
	eNvFlexShapeTriangleMesh	= 4,		//!< A triangle mesh shape, see FlexTriangleMeshGeometry
	eNvFlexShapeSDF				= 5,		//!< A signed distance field shape, see FlexSDFGeometry
};

enum NvFlexCollisionShapeFlags
{
	eNvFlexShapeFlagTypeMask	= 0x7,		//!< Lower 3 bits holds the type of the collision shape given by the NvFlexCollisionShapeType enum
	eNvFlexShapeFlagDynamic		= 0x8,		//!< Indicates the shape is dynamic and should have lower priority over static collision shapes
	eNvFlexShapeFlagTrigger		= 0x10,		//!< Indicates that the shape is a trigger volume, this means it will not perform any collision response, but will be reported in the contacts array (see NvFlexGetContacts())
	
	eNvFlexShapeFlagReserved	= 0xffffff00
};

/** 
 * Helper function to combine shape type, flags, and phase/shape collision channels into a 32bit value
 * 
 * @param[in] type The type of the shape, see NvFlexCollisionShapeType
 * @param[in] dynamic See eNvFlexShapeFlagDynamic
 * @param[in] shapeChannels A combination of the eNvFlexPhaseShapeChannel* flags, collisions will only be processed between a particle and a shape if a channel is set on both the particle and shape, see NvFlexMakePhaseWithChannels()
 */
NV_FLEX_API inline int NvFlexMakeShapeFlagsWithChannels(NvFlexCollisionShapeType type, bool dynamic, int shapeChannels) { return type | (dynamic?eNvFlexShapeFlagDynamic:0) | shapeChannels; }

/** 
 * Deprecrated helper method that creates shape flags that by default have all collision channels enabled
 */
NV_FLEX_API inline int NvFlexMakeShapeFlags(NvFlexCollisionShapeType type, bool dynamic) { return NvFlexMakeShapeFlagsWithChannels(type, dynamic, eNvFlexPhaseShapeChannelMask); }


/**
 * Set the collision shapes for the solver
 * 
 * @param[in] solver A valid solver
 * @param[in] geometry Pointer to a buffer of NvFlexCollisionGeometry entries, the type of each shape determines how many entries it has in the array
 * @param[in] shapePositions Pointer to a buffer of translations for each shape in world space, should be 4*numShapes in length
 * @param[in] shapeRotations Pointer to an a buffer of rotations for each shape stored as quaternion, should be 4*numShapes in length
 * @param[in] shapePrevPositions Pointer to a buffer of translations for each shape at the start of the time step, should be 4*numShapes in length
 * @param[in] shapePrevRotations Pointer to an a buffer of rotations for each shape stored as a quaternion at the start of the time step, should be 4*numShapees in length
 * @param[in] shapeFlags The type and behavior of the shape, NvFlexCollisionShapeFlags for more detail
 * @param[in] numShapes The number of shapes
 *
 */
NV_FLEX_API void NvFlexSetShapes(NvFlexSolver* solver, NvFlexBuffer* geometry, NvFlexBuffer* shapePositions, NvFlexBuffer* shapeRotations, NvFlexBuffer* shapePrevPositions, NvFlexBuffer* shapePrevRotations, NvFlexBuffer* shapeFlags, int numShapes);


struct NvFlexRigidMaterial
{
	float friction;
	float rollingFriction;
	float torsionFriction;

	float restitution;
	float compliance;
};

struct NvFlexRigidPose
{
	float p[3];
	float q[4];
};

enum NvFlexRigidBodyFlags
{
	eNvFlexRigidBodyFlagsDisableSimulation = (1 << 0),		//!< Disable simulation for this rigid body
};

struct NvFlexRigidBody
{
	float com[3];			//!< Vector representing the center of mass of the body in world space, to retrieve the pose of the body's origin use NvFlexGetRigidPose()
	float theta[4];			//!< Quaternion representing the orientation of the body in world space

	float linearVel[3];		//!< Linear velocity of the center of mass in world space
	float angularVel[3];	//!< Angular velocity of the body in world space
	
	float force[3];			//!< Linear force
	float torque[3];		//!< Angular torque

	float origin[3];		//!< Body space offset of the origin relative to the center of mass, the world space origin of a body can be calculated as worldOrigin = com + theta*origin, or use NvFlexGetRigidBodyPose()

	float mass;				//!< Total body force
	float inertia[9];		//!< Inertia of the body in body space

	float invMass;
	float invInertia[9];

	float linearDamping;
	float angularDamping;

	float maxLinearVelocity;
	float maxAngularVelocity;

	int flags;				//!< Rigid body flags
};

enum NvFlexRigidShapeFlags
{
	eNvFlexRigidShapeFlagsDisableSimulation = (1 << 0),		//!< Disable simulation for this rigid shape
};

struct NvFlexRigidShape
{
	int body;						//!< Rigid body index
	
	int filter;						//!< Contacts will only be ignored for any two shapes that have a shared bit set
	int group;						//!< Only shapes of the same group will collide. Set -1 to collide with all other groups

	NvFlexCollisionGeometry geo;	//!< Geometry
	int geoType;					//!< Geometry type
	
	NvFlexRigidPose pose;			//!< Transform of the shape in the parent body's local space
	NvFlexRigidMaterial material;	//!< Physical material

	float thickness;				//!< How far objects should come to rest from the surface of this body

	void* user;						//!< User data

	int flags;						//!< Shape flags
};

enum NvFlexRigidJointMode
{
	eNvFlexRigidJointModeFree,			//!< Constraint will be disabled
	eNvFlexRigidJointModePosition,		//!< Constraint will target translation or rotation given by targets array along corresponding axis
	eNvFlexRigidJointModeVelocity,		//!< Constraint will target translational or angular velocity given by targets array along corresponding axis
	eNvFlexRigidJointModeLimit,			//!< Constraint will be enforced between limits given by limits array
	eNvFlexRigidJointModeLimitSpring,	//!< Constraint will be enforced between limits given by limits array and includes an explicit spring with force f = -k1*error -k2*errorRate, where k1 is the inverse compliance param, and k2 is the axis damping param
	eNvFlexRigidJointModeMimic,			//!< Constraint will set targets based on another joint (see NvFlexRigidJoint::mimicIndex), with scale and bias mimicScale*mimicJoint.targets + mimicBias. 
};

enum NvFlexRigidJointAxis
{
	eNvFlexRigidJointAxisX,			//!< Corresponds to translation around the body0 x-axis
	eNvFlexRigidJointAxisY,			//!< Corresponds to translation around the body0 y-axis
	eNvFlexRigidJointAxisZ,			//!< Corresponds to translation around the body0 z-axis
	eNvFlexRigidJointAxisTwist,		//!< Corresponds to rotation around the body0 x-axis
	eNvFlexRigidJointAxisSwing1,	//!< Corresponds to rotation around the body0 y-axis
	eNvFlexRigidJointAxisSwing2,	//!< Corresponds to rotation around the body0 z-axis
	
};

enum NvFlexRigidJointFlags
{
	eNvFlexRigidJointFlagsDisableCollision = (1 << 0),		//!< Disable collision between connected bodies
	eNvFlexRigidJointFlagsDisableSimulation = (1 << 1),		//!< Disable simulation for this rigid joint
};

struct NvFlexRigidJoint
{
	int body0;					//!< Index of first body, use -1 to indicate world body
	int body1;					//!< Index of second body, use -1 to indicate world body

	NvFlexRigidPose pose0;		//!< Transform of the joint frame in body0 local space
	NvFlexRigidPose pose1;		//!< Transform of the joint frame in body1 local space
	
	int modes[6];				//!< See NvFlexRigidJointMode

	float targets[6];			//! For translational axes this is the displacement of body1 with respect to body0's joint frame, for rotational axes this is the angle in radians of body1 with respect to body0's joint frame

	float lowerLimits[6];		//! For translational axes this is a lower bound on the displacement of body1 with respect to body0's joint frame, for rotational axes this is the lower bound angle in radians of body1 with respect to body0's joint frame
	float upperLimits[6];		//! For translational axes this is an upper bound on the displacement of body1 with respect to body0's joint frame, for rotational axes this is the upper bound angle in radians of body1 with respect to body0's joint frame

	float compliance[6];
	float damping[6];

	float motorLimit[6];

	float lambda[6];

	int mimicIndex;				//!< Index of joint to mimic
	float mimicScale[3];		//!< Scale of parent joint 
	float mimicOffset[3];		//!< Offset from parent joint

	int maxIterations;			//!< Solver iterations limit
	
	int flags;					//!< Joint flags
};

struct NvFlexRigidContact
{
	int body0;
	int body1;
	
	float localPos0[3];	//!< Local space position of the body0 contact feature excluding thickness, normal forces applied here
	float localPos1[3];	//!< Local space position of the body1 contact feature excluding thickness normal forces applied here

	float minDist;		//!< Minimum distance to try and maintain along the contact normal between the two points
	float normal[3];	//!< Contact normal from body0->body1 in world space

	float offset0[3];	//!< The local space offset from the feature localPos0 to the surface, friction will be applied here
	float offset1[3];	//!< The local space offset from the feature localPos1 to the surface, friction will be applied here

	float lambda;
	
	float friction;
	float torsionFriction;	
	float rollingFriction;
};

struct NvFlexSoftContact
{
	int particleIndices[3];	//!< Index of the particles involved in the contact 
	float particleBarys[3];	//!< Barycentric coordinates of the contact point

	int bodyIndex;			//!< Index of the rigid body involved in the contact (see NvFlexSetRigidBodies())
	float bodyOffset[3];	//!< Position of the contact in the body's local space

	float normal[3];		//!< Contact normal in world space pointing from the dynamic triangle to the rigid body

	float lambda;			//!< Total Lagrange multiplier for the constraint

	float minDist;			//!< Distance constraint tries to maintain along normal direction between contact points
};

struct NvFlexRigidParticleAttachment
{
	int body;				//!< Index of the rigid body to attach
	int particle;			//!< Index of the particle to attach to the rigid body

	float localOffset[3];	//!< The local space offset from the particle to the center of mass of the rigid body
};

// rigids
NV_FLEX_API void NvFlexGetRigidPose(const NvFlexRigidBody* b, NvFlexRigidPose* pose);
NV_FLEX_API void NvFlexSetRigidPose(NvFlexRigidBody* b, const NvFlexRigidPose* pose);
NV_FLEX_API NvFlexRigidPose NvFlexMakeRigidPose(const float* pos, const float* rot);

NV_FLEX_API void NvFlexMakeRigidBody(NvFlexLibrary* lib, NvFlexRigidBody* body, const float* pos, const float* rot, NvFlexRigidShape* shapes, const float* densities, int numShapes);

// sets the velocity necessary for a body to achieve a target pose over the course of a time-step dt
NV_FLEX_API void NvFlexSetRigidTarget(NvFlexRigidBody* body, NvFlexRigidPose* pose, float dt);

NV_FLEX_API void NvFlexSetRigidBodies(NvFlexSolver* solver, NvFlexBuffer* rigidBodies, int numBodies);
NV_FLEX_API void NvFlexGetRigidBodies(NvFlexSolver* solver, NvFlexBuffer* rigidBodies);

// shapes
NV_FLEX_API void NvFlexMakeRigidBoxShape(NvFlexRigidShape* shape, int body, float hx, float hy, float hz, NvFlexRigidPose pose);
NV_FLEX_API void NvFlexMakeRigidSphereShape(NvFlexRigidShape* shape, int body, float radius, NvFlexRigidPose pose);
NV_FLEX_API void NvFlexMakeRigidCapsuleShape(NvFlexRigidShape* shape, int body, float radius, float halfHeight, NvFlexRigidPose pose);
NV_FLEX_API void NvFlexMakeRigidConvexMeshShape(NvFlexRigidShape* shape, int body, NvFlexConvexMeshId mesh, NvFlexRigidPose pose, float sx, float sy, float sz);
NV_FLEX_API void NvFlexMakeRigidTriangleMeshShape(NvFlexRigidShape* shape, int body, NvFlexTriangleMeshId mesh, NvFlexRigidPose pose, float sx, float sy, float sz);

NV_FLEX_API void NvFlexSetRigidShapes(NvFlexSolver* solver, NvFlexBuffer* rigidShapes, int numShapes);
NV_FLEX_API void NvFlexGetRigidShapes(NvFlexSolver* solver, NvFlexBuffer* rigidShapes);

// joints
NV_FLEX_API void NvFlexMakeFixedJoint(NvFlexRigidJoint* joint, int body0, int body1, NvFlexRigidPose pose0, NvFlexRigidPose pose1);
NV_FLEX_API void NvFlexMakeSphericalJoint(NvFlexRigidJoint* joint, int body0, int body1, NvFlexRigidPose pose0, NvFlexRigidPose pose1);
NV_FLEX_API void NvFlexMakeHingeJoint(NvFlexRigidJoint* joint, int body0, int body1, NvFlexRigidPose pose0, NvFlexRigidPose pose1, int axis, float lower=-FLT_MAX, float upper=FLT_MAX);
NV_FLEX_API void NvFlexMakePrismaticJoint(NvFlexRigidJoint* joint, int body0, int body1, NvFlexRigidPose pose0, NvFlexRigidPose pose1, int axis, float lower=-FLT_MAX, float upper=FLT_MAX);

NV_FLEX_API void NvFlexSetRigidJoints(NvFlexSolver* solver, NvFlexBuffer* rigidJoints, int numJoints);
NV_FLEX_API void NvFlexGetRigidJoints(NvFlexSolver* solver, NvFlexBuffer* rigidJoints);

//retrieve the joint translational and angular error values and error rates for each axis, caller must supply the body's explicitly (since Flex does not have CPU copies of this data)
NV_FLEX_API void NvFlexGetRigidJointState(const NvFlexRigidJoint* joint, const NvFlexRigidBody* body0, const NvFlexRigidBody* body1, float linPos[3], float linRate[3], float angPos[3], float angRate[3]);

// cables
NV_FLEX_API void NvFlexSetCableLinks(NvFlexSolver* s, NvFlexBuffer* cableLinks, int numCableLinks);
NV_FLEX_API void NvFlexGetCableLinks(NvFlexSolver* s, NvFlexBuffer* cableLinks);

// contacts
NV_FLEX_API void NvFlexGetRigidContacts(NvFlexSolver* solver, NvFlexBuffer* rigidContacts, NvFlexBuffer* rigidContactCount);
NV_FLEX_API void NvFlexGetRigidSoftContacts(NvFlexSolver* s, NvFlexBuffer* softContacts, NvFlexBuffer* softContactCount);

// ray casts
struct NvFlexRay
{
	float start[3];			//!< Ray start position in world space	
	float dir[3];			//!< Ray dir in world space, should be normalized
	
	int filter;				//!< Filter mask, will not report hits with any shape that shares a bit with this mask
	int group;				//!< Group ID, will report shapes with matching group ID. Set -1 for any group ID

	float maxT;				//!< Will not consider hits past this point
};

struct NvFlexRayHit
{	
	int shape;				//!< Shape index, will be -1 if no hit
	int element;			//!< Shape's element index, e.g.: triangle index

	float t;				//!< Distance along ray hit occured
	float n[3];				//!< Hit normal in world space
};

NV_FLEX_API void NvFlexRayCast(NvFlexSolver* solver, const NvFlexBuffer* rays, NvFlexBuffer* hits, int numRays);



/**
 * Set dynamic triangles mesh indices, typically used for cloth. Flex will calculate normals and 
 * apply wind and drag effects to connected particles. See NvFlexParams::drag, NvFlexParams::wind.
 * 
 * @param[in] solver A valid solver
 * @param[in] indices Pointer to a buffer of triangle indices into the particles array, should be 3*numTris in length
 * @param[in] normals Pointer to a buffer of triangle normals, should be 3*numTris in length, can be NULL
 * @param[in] numTris The number of dynamic triangles
 *s
 */
NV_FLEX_API void NvFlexSetDynamicTriangles(NvFlexSolver* solver, NvFlexBuffer* indices, NvFlexBuffer* normals, NvFlexBuffer* features, int numTris);
/**
 * Get the dynamic triangle indices and normals.
 * 
 * @param[in] solver A valid solver
 * @param[out] indices Pointer to a buffer of triangle indices into the particles array, should be 3*numTris in length, if NULL indices will not be returned
 * @param[out] normals Pointer to a buffer of triangle normals, should be 3*numTris in length, if NULL normals will be not be returned
 * @param[in] numTris The number of dynamic triangles
 */
NV_FLEX_API void NvFlexGetDynamicTriangles(NvFlexSolver* solver, NvFlexBuffer* indices, NvFlexBuffer* normals, NvFlexBuffer* features, int numTris);

/**
 * Set inflatable shapes, an inflatable is a range of dynamic triangles (wound CCW) that represent a closed mesh.
 * Each inflatable has a given rest volume, constraint scale (roughly equivalent to stiffness), and "over pressure"
 * that controls how much the shape is inflated.
 * 
 * @param[in] solver A valid solver
 * @param[in] startTris Pointer to a buffer of offsets into the solver's dynamic triangles for each inflatable, should be numInflatables in length
 * @param[in] numTris Pointer to a buffer of triangle counts for each inflatable, should be numInflatablesin length
 * @param[in] restVolumes Pointer to a buffer of rest volumes for the inflatables, should be numInflatables in length
 * @param[in] overPressures Pointer to a buffer of floats specifying the pressures for each inflatable, a value of 1.0 means the rest volume, > 1.0 means over-inflated, and < 1.0 means under-inflated, should be numInflatables in length
 * @param[in] constraintScales Pointer to a buffer of scaling factors for the constraint, this is roughly equivalent to stiffness but includes a constraint scaling factor from position-based dynamics, see helper code for details, should be numInflatables in length
 * @param[in] numInflatables Number of inflatables to set
 *
 */
NV_FLEX_API void NvFlexSetInflatables(NvFlexSolver* solver, NvFlexBuffer* startTris, NvFlexBuffer* numTris, NvFlexBuffer* restVolumes, NvFlexBuffer* overPressures, NvFlexBuffer* constraintScales, int numInflatables);

/**
 * Get the density values for fluid particles
 *
 * @param[in] solver A valid solver
 * @param[out] densities Pointer to a buffer of floats, should be maxParticles in length, density values are normalized between [0, 1] where 1 represents the rest density
 * @param[in] desc Pointer to a descriptor specifying the contents to read back
 */
NV_FLEX_API void NvFlexGetDensities(NvFlexSolver* solver, NvFlexBuffer* densities, const NvFlexCopyDesc* desc);

/**
 * Get the anisotropy of fluid particles, the particle distribution for a particle is represented
 * by 3 orthogonal vectors. Each 3-vector has unit length with the variance along that axis
 * packed into the w component, i.e.: x,y,z,lambda.
*
 * The anisotropy defines an oriented ellipsoid in worldspace that can be used for rendering
 * or surface extraction.
 *
 * @param[in] solver A valid solver
 * @param[out] q1 Pointer to a buffer of floats that receive the first basis vector and scale, should be 4*maxParticles in length
 * @param[out] q2 Pointer to a buffer of floats that receive the second basis vector and scale, should be 4*maxParticles in length
 * @param[out] q3 Pointer to a buffer of floats that receive the third basis vector and scale, should be 4*maxParticles in length
 * @param[in] desc Pointer to a descriptor specifying the contents to read back
 */
NV_FLEX_API void NvFlexGetAnisotropy(NvFlexSolver* solver, NvFlexBuffer* q1, NvFlexBuffer* q2, NvFlexBuffer* q3, const NvFlexCopyDesc* desc);
/**
 * Get the state of the diffuse particles. Diffuse particles are passively advected by the fluid
 * velocity field.
 *
 * @param[in] solver A valid solver
 * @param[out] p Pointer to a buffer of floats, should be 4*maxParticles in length, the w component represents the particles lifetime with 1 representing a new particle, and 0 representing an inactive particle
 * @param[out] v Pointer to a buffer of floats, should be 4*maxParticles in length, the w component is not used
 * @param[out] count Pointer to a buffer of a single int that holds the current particle count (this may be updated by the GPU which is why it is passed back in a buffer)
 */
NV_FLEX_API void NvFlexGetDiffuseParticles(NvFlexSolver* solver, NvFlexBuffer* p, NvFlexBuffer* v, NvFlexBuffer* count);

/**
 * Set the state of the diffuse particles. Diffuse particles are passively advected by the fluid
 * velocity field.
 *
 * @param[in] solver A valid solver
 * @param[in] p Pointer to a buffer of floats, should be 4*n in length, the w component represents the particles lifetime with 1 representing a new particle, and 0 representing an inactive particle
 * @param[in] v Pointer to a buffer of floats, should be 4*n in length, the w component is not used
 * @param[in] n The number of active diffuse particles
 *
 */
NV_FLEX_API void NvFlexSetDiffuseParticles(NvFlexSolver* solver, NvFlexBuffer* p, NvFlexBuffer* v, int n);

/**
 * Get the particle contact planes. Note this will only include contacts that were active on the last substep of an update, and will include all contact planes generated within NvFlexParams::shapeCollisionMargin.
 *
 * @param[in] solver A valid solver
 * @param[out] planes Pointer to a destination buffer containing the contact planes for the particle, each particle can have up to 6 contact planes so this buffer should be 4*6*maxParticles floats in length
 * @param[out] velocities Pointer to a destination buffer containing the velocity of the contact point on the shape in world space, the index of the shape (corresponding to the shape in NvFlexSetShapes() is stored in the w component), each particle can have up to 6 contact planes so this buffer should be 4*6*maxParticles floats in length
 * @param[out] indices Pointer to a buffer of indices into the contacts buffer, the first contact plane for the i'th particle is given by planes[indices[i]*sizeof(float)*4*6] and subsequent contact planes for that particle are stored sequentially, this array should be maxParticles in length
 * @param[out] counts Pointer to a buffer of contact counts for each particle (will be <= 6), this buffer should be maxParticles in length
 */
NV_FLEX_API void NvFlexGetContacts(NvFlexSolver* solver, NvFlexBuffer* planes, NvFlexBuffer* velocities, NvFlexBuffer* indices, NvFlexBuffer* counts);

/**
 * Get the particle neighbor lists, these are stored in a strided format, and can be iterated in the following manner:
 *
\code{.c}

	NvFlexGetNeighbors(solver, neighborsBuffer, countsBuffer, indicesBuffer);

	int* neighbors = (int*)NvFlexMap(neighborsBuffer, 0);
	int* counts = (int*)NvFlexMap(countsBuffer, 0);
	int* remap = (int*)NvFlexMap(remapBuffer, 0);

	// neighbors are stored in a strided format so that the first neighbor
	// of each particle is stored sequentially, then the second, and so on
	
	int stride = maxParticles;

	for (int i=0; i < maxParticles; ++i)
	{
		// find offset in the neighbors buffer
		int offset = remap[i];
		int count = counts[offset];

		for (int c=0; c < count; ++c)
		{
			int neighbor = remap[neighbors[c*stride + offset]];

			printf("Particle %d's neighbor %d is particle %d\n", i, c, neighbor);
		}
	}

	NvFlexUnmap(neighborsBuffer);
	NvFlexUnmap(countsBuffer);
	NvFlexUnmap(remapBuffer);

\endcode
 *
 * @param[in] solver A valid solver
 * @param[out] neighbors Pointer to a destination buffer containing the the neighbors for all particles, this should be maxParticles*maxParticleNeighbors ints (passed to NvFlexInit() in length)
 * @param[out] counts Pointer to a buffer of neighbor counts per-particle, should be maxParticles ints in length
 * @param[out] remap Pointer to a buffer of indices, because Flex internally re-orders particles these are used to map from an API particle index to it internal index
 *
 * @note Neighbors are only valid after a call to NvFlexUpdateSolver() has completed, the returned neighbors correspond to the last substep of the last update
 */
NV_FLEX_API void NvFlexGetNeighbors(NvFlexSolver* solver, NvFlexBuffer* neighbors, NvFlexBuffer* counts, NvFlexBuffer* remap);

/**
 * Get the world space AABB of all particles in the solver, note that the bounds are calculated during the update (see NvFlexUpdateSolver()) so only become valid after an update has been performed.
 * The returned bounds represent bounds of the particles in their predicted positions *before* the constraint solve.
 * 
 * @param[in] solver A valid solver
 * @param[out] lower Pointer to a buffer of 3 floats to receive the lower bounds
 * @param[out] upper Pointer to a buffer of 3 floats to receive the upper bounds
 */
NV_FLEX_API void NvFlexGetBounds(NvFlexSolver* solver, NvFlexBuffer* lower, NvFlexBuffer* upper);

/**
 *
 * @param[in] solver A valid solver
 * @param[out] begin Optional pointer to a 64 bit unsigned to receive the value of the GPU clock when Flex update began (in cycles)
 * @param[out] end Optional pointer to a 64 bit unsigned to receive the value of the GPU clock when Flex update ended (in cycles)
 * @param[out] frequency Optional pointer to a 64 bit unsigned to receive the frequency of the clock used to measure begin and end
 * @return The time in seconds between the first and last GPU operations executed by the last NvFlexUpdateSolver.
 *
 * @note This method causes the CPU to wait until the GPU has finished any outstanding work. 
 *		 To avoid blocking the calling thread it should be called after work has completed, e.g.: directly after a NvFlexMap().
 */
NV_FLEX_API float NvFlexGetDeviceLatency(NvFlexSolver* solver, unsigned long long* begin, unsigned long long* end, unsigned long long* frequency);

/**
 * Fetch high-level GPU timers.
 *
 * @param[in] solver The solver instance to use
 * @param[out] timers A struct containing the GPU latency of each stage in the physics pipeline.
 *
 * @note This method causes the CPU to wait until the GPU has finished any outstanding work.
 *		 To avoid blocking the calling thread it should be called after work has completed, e.g.: directly after a NvFlexMap().
 *       To capture there timers you must pass true for enableTimers in NvFlexUpdateSolver()
 */
NV_FLEX_API void NvFlexGetTimers(NvFlexSolver* solver, NvFlexTimers* timers);

/**
* Holds the execution time for a specfic shader
*/
struct NvFlexDetailTimer
{ 
	char* name; 
	float time;
};

/**
* Fetch per-shader GPU timers.
*
* @param[in] solver The solver instance to use
* @param[out] timers An array of NvFlexDetailTimer structures, each representing a unique shader.
* @return The number of detail timers in the timers array
*
* @note This method causes the CPU to wait until the GPU has finished any outstanding work.
*		To avoid blocking the calling thread it should be called after work has completed, e.g.: directly after a NvFlexMap().
*       To capture there timers you must pass true for enableTimers in NvFlexUpdateSolver()
*		Timers are valid until the next call to NvFlexGetDetailTimers
*/
NV_FLEX_API int NvFlexGetDetailTimers(NvFlexSolver* solver, NvFlexDetailTimer** timers);

/**
 * Allocate a Flex buffer. Buffers are used to pass data to the API in an efficient manner.
 *
 * @param[in] lib The library instance to use
 * @param[in] elementCount The number of elements in the buffer
 * @param[in] elementByteStride The size of each element in bytes
 * @param[in] type The type of buffer to allocate, can be either host memory or device memory
 * @return A pointer to a NvFlexBuffer
 */
NV_FLEX_API NvFlexBuffer* NvFlexAllocBuffer(NvFlexLibrary* lib, int elementCount, int elementByteStride, NvFlexBufferType type);

/**
 * Free a Flex buffer
 *
 * @param[in] buf A buffer to free, must be allocated with NvFlexAllocBuffer()
 */
NV_FLEX_API void NvFlexFreeBuffer(NvFlexBuffer* buf);

/**
 * Maps a buffer for reading and writing. When the buffer is created with NvFlexBufferType::eHost, then the returned pointer will be a host memory address
 * that can be read/written.
 * Mapping a buffer implicitly synchronizes with the GPU to ensure that any reads or writes from the buffer (e.g.: from the NvFlexGet*() or NvFlexSet*() methods) have completed.
 *
 * @param[in] buffer A buffer allocated with NvFlexAllocBuffer()
 * @param[in] flags Hints to Flex how the buffer is to be accessed, typically this should be eNvFlexMapWait (0)
 * @return A pointer to the mapped memory
 */
NV_FLEX_API void* NvFlexMap(NvFlexBuffer* buffer, int flags);

/**
 * Unmaps a buffer that was mapped through NvFlexMap(), note that buffers must be unmapped before they can be passed to a NvFlexGet*() or NvFlexSet*() method
 *
 * @param[in] buffer A valid buffer allocated through NvFlexAllocBuffer()
 */
NV_FLEX_API void NvFlexUnmap(NvFlexBuffer* buffer);


/**
 * A struct of raw device pointers directly to the solver copy of the data
 */
struct NvFlexDeviceBuffers
{
	float* particlePositions;				//!< array of vec4 positions
	float* particleVelocities;				//!< array of vec3 velocities
	int* particlePhases;					//!< array of int phases 

	NvFlexRigidBody* rigidBodies;
	NvFlexRigidShape* rigidShapes;
	NvFlexRigidJoint* rigidJoints;
	NvFlexRigidContact* rigidContacts;
	int* rigidContactCounter;
};

/**
 * Returns a struct of device pointers to the GPU solver data directly. The pointers in this
 * struct are only valid until the next call to NvFlexUnmapDeviceBuffers(). It is illegal to
 * call any other solver methods while the device buffers are mapped.
 */
NV_FLEX_API void NvFlexMapDeviceBuffers(NvFlexSolver* solver, NvFlexDeviceBuffers* buffers);

/**
 * Unmap device buffers
 */ 
NV_FLEX_API void NvFlexUnmapDeviceBuffers(NvFlexSolver* solver, NvFlexDeviceBuffers* buffers);
 

/**
 * Registers an OpenGL buffer to Flex which can be used to copy directly into a graphics resource. Example usage is below
 *
 \code{.c}

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW)

	NvFlexBuffer* vboBuffer = NvFlexRegisterOGLBuffer(lib, vbo, n, sizeof(float)*4);

	// simulate 
	...

	// copy directly from Flex into render buffer
	NvFlexGetParticles(vboBuffer, n);

	// render
	...

 \endcode
 *
 * @param[in] lib The library instance to use
 * @param[in] buf An OpenGL buffer identifier
 * @param[in] elementCount The number of elements in the buffer
 * @param[in] elementByteStride the size of each element in bytes
 * @return A valid NvFlexBuffer pointer that may be used with NvFlexGet*() methods to populate the render buffer using direct GPU-GPU copies
 */
NV_FLEX_API NvFlexBuffer* NvFlexRegisterOGLBuffer(NvFlexLibrary* lib, int buf, int elementCount, int elementByteStride);

/**
 * Unregister a NvFlexBuffer allocated through NvFlexRegisterOGLBuffer()
 *
 * @param[in] buf A valid buffer
 */
NV_FLEX_API void NvFlexUnregisterOGLBuffer(NvFlexBuffer* buf);

/**
* Registers a Direct3D buffer to Flex which can be used to copy directly into a graphics resource
*
* @param[in] lib The library instance to use
* @param[in] buffer A pointer to either an ID3D11Buffer or ID3D12Resource object
* @param[in] elementCount The number of elements in the buffer
* @param[in] elementByteStride the size of each element in bytes
* @return A valid NvFlexBuffer pointer that may be used with NvFlexGet*() methods to populate the render buffer using direct GPU-GPU copies
*/
NV_FLEX_API NvFlexBuffer* NvFlexRegisterD3DBuffer(NvFlexLibrary* lib, void* buffer, int elementCount, int elementByteStride);

/**
* Unregister a NvFlexBuffer allocated through NvFlexRegistereD3DBuffer()
*
* @param[in] buf A valid buffer
*/
NV_FLEX_API void NvFlexUnregisterD3DBuffer(NvFlexBuffer* buf);

/** 
 * Ensures that the CUDA context the library was initialized with is present on the current thread
 *
 * @param[in] lib The library instance to use
 */
NV_FLEX_API void NvFlexAcquireContext(NvFlexLibrary* lib);

/** 
 * Restores the CUDA context (if any) that was present on the last call to NvFlexAcquireContext()
 * Note: the acquire/restore pair of calls must come from the same thread
 */
NV_FLEX_API void NvFlexRestoreContext(NvFlexLibrary* lib);

/** 
 * Returns a null-terminated string with the compute device name
 *
 * @param[in] lib The library instance to use
 */
NV_FLEX_API const char* NvFlexGetDeviceName(NvFlexLibrary* lib);

/** 
 * Retrieve the device and context for the the library.
 * On CUDA the context pointer will be filled with a pointer to a CUcontext structure
 * On D3D the device and context pointers will be filled with pointers to a NvFlex::Device, and NvFlex::Context wrapper
 *
 * @param[in] lib Pointer to a valid library returned from NvFlexInit()
 * @param[out] device Pointer to a device pointer, see description
 * @param[out] context Pointer to a context pointer, see description
 */
NV_FLEX_API void NvFlexGetDeviceAndContext(NvFlexLibrary* lib, void** device, void** context);
 

/**
 * Force a pipeline flush to ensure any queued work is submitted to the GPU
 *
 * @param[in] lib The library instance to use
 */
NV_FLEX_API void NvFlexFlush(NvFlexLibrary* lib);

NV_FLEX_API void NvFlexWait(NvFlexLibrary* lib);

//! \cond HIDDEN_SYMBOLS

/**
 * Debug methods (unsupported)
 */

NV_FLEX_API void NvFlexStartExperiment(const char* file, const char* title);	// e.g.: "tower_solver", "Tower Solver Comparision"
NV_FLEX_API void NvFlexStopExperiment();

NV_FLEX_API void NvFlexStartExperimentRun(const char* name);	// e.g.: "Jacobi"
NV_FLEX_API void NvFlexStopExperimentRun();


NV_FLEX_API void NvFlexSetDebug(NvFlexSolver* solver, bool enable, const char* name, int flag, float param);
NV_FLEX_API void NvFlexGetShapeBVH(NvFlexSolver* solver, void* bvh);
NV_FLEX_API void NvFlexCopySolver(NvFlexSolver* dst, NvFlexSolver* src);
NV_FLEX_API void NvFlexCopyDeviceToHost(NvFlexSolver* solver, NvFlexBuffer* pDevice, void* pHost, int size, int stride);
NV_FLEX_API void NvFlexComputeWaitForGraphics(NvFlexLibrary* lib);

//! \endcond

enum NVFlexCableLinkType
{
	eNvFlexCableLinkTypeFixed,		// Fixed link to body
	eNvFlexCableLinkTypeRolling,	// Allow body to spin around an axis
	eNvFlexCableLinkTypePinhole,	// Pin hole link, cable is allowed to slide through the holes
};

struct NvFlexCableLink
{
	NVFlexCableLinkType type;								
	int index; 
	int prevLink;
	int nextLink;
	int body;
	float att0[3];
	float att1[3];
	NvFlexRigidPose profilePose;
	NvFlexCurveId profileVerts;
	float stretchingLambda, compressionLambda;
	float segLength, linkLength, currentSegLength;
	float update;
	bool inLimit;
	bool flipped;
	float stretchingCompliance, compressionCompliance;
	float stretchingDamping, compressionDamping;
	float radius;
	bool secondTolastLink;
	bool firstLink;
};

/**
* Initialize a fixed cable link
*
* @param[in] lib The library instance to use
* @param[out] link Pointer to link structure
* @param[in] linkIndex Global index of the link
* @param[in] body Pointer to the rigid body this link belongs to, NULL for world
* @param[in] shapes Pointer to the shapes of the rigid body, NULL for world 
* @param[in] numShapes Number of the shapes of the rigid body, 0 for world
* @param[in] globalPos Global position of the fixed link 
*/
NV_FLEX_API void NvFlexInitFixedCableLink(NvFlexLibrary* lib, NvFlexCableLink* link, const int linkIndex, const NvFlexRigidBody* body, const NvFlexRigidShape* shapes, const int numShapes, const float* globalPos);

/**
* Initialize a pinhole cable link
*
* @param[in] lib The library instance to use
* @param[out] link Pointer to link structure
* @param[in] linkIndex Global index of the link
* @param[in] body Pointer to the rigid body this link belongs to, NULL for world
* @param[in] shapes Pointer to the shapes of the rigid body, NULL for world
* @param[in] numShapes Number of the shapes of the rigid body, 0 for world
* @param[in] inGlobalPos Global position of the inward pinhole
* @param[in] outGlobalPos Global position of the outward pinhole
*/
NV_FLEX_API void NvFlexInitPinholeCableLink(NvFlexLibrary* lib, NvFlexCableLink* link, const int linkIndex, const NvFlexRigidBody* body, const NvFlexRigidShape* shapes, const int numShapes, const float* inGlobalPos, const float* outGlobalPos);

/**
* Initialize a rolling cable link
*
* @param[in] lib The library instance to use
* @param[out] link Pointer to link structure
* @param[in] linkIndex Global index of the link
* @param[in] body Pointer to the rigid body this link belongs to, NULL for world
* @param[in] shapes Pointer to the shapes of the rigid body, NULL for world
* @param[in] numShapes Number of the shapes of the rigid body, 0 for world
* @param[in] globalPlane Global plane of the rolling link, plane equation is globalPlane[0]*x + globalPlane[1]*y + globalPlane[2]*z + globalPlane[3] = 0, the plane should be oriented such that the right hand rule around the plane normal is the direction the cable wraps around the rigid body
* @param[in] extraLength Extra length of the cable before it's stretch
*/
NV_FLEX_API void NvFlexInitRollingCableLink(NvFlexLibrary* lib, NvFlexCableLink* link, const int linkIndex, const NvFlexRigidBody* body, const NvFlexRigidShape* shapes, const int numShapes, const float* globalPlane, const float extraLength = 0.0f, const NvFlexCurveId curveId = 0);

/**
* Initialize a rolling cable link
*
* @param[in] lib The library instance to use
* @param[in/out] link Pointer to links that was previously initialized
* @param[in] bodies Array of pointers to the rigid body the links belongs to, NULL for world
* @param[in] numLinks Number of links 
* @param[in] cyclic Whether the cable is cyclic
* @param[in] stretchingCompliance Stretching compliance
* @param[in] stretchingDamping Stretching damping
* @param[in] compressionCompliance Compression compliance
* @param[in] compressionDamping Compression damping
*/
NV_FLEX_API void NvFlexMakeCable(NvFlexLibrary* lib, NvFlexCableLink* links, const NvFlexRigidBody** bodies, const int numLinks, const bool cyclic, const float stretchingCompliance = 0.0f, const float stretchingDamping = 0.0f, const float compressionCompliance = -1.0f, const float compressionDamping = 0.0f);

} // extern "C"

#endif // NV_FLEX_H
