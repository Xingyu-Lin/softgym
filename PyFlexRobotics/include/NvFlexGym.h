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

#ifndef NV_FLEX_GYM_H
#define NV_FLEX_GYM_H

#include "NvFlex.h"

extern "C" {

enum NvFlexGymRenderBackend
{
	eNvFlexNoRendering,	// Headless and no rendering
	eNvFlexOpenGL,		// OpenGl rendering
	eNvFlexHeadless,    // Headless rendering EGL
};

/**
* Parameters for Flex Gym initialization
*/
struct NvFlexGymInitParams
{
	NvFlexGymRenderBackend renderBackend;
	int screenWidth;
	int screenHeight;
	int msaaSamples;
	int device;
	int rank; 
	uint32_t seed;
	bool vsync;
};

/**
 * Initialize Felx Gym environment
 * @param[in] initParams Parameters structure in host memory, see NvFlexGymInitParams
 */
NV_FLEX_API void NvFlexGymInit(const NvFlexGymInitParams* initParams);

/**
 * Update Felx Gym environment
 */
NV_FLEX_API int NvFlexGymUpdate();

/**
 * Shutdown Felx Gym environment
 */
NV_FLEX_API void NvFlexGymShutdown();

/**
 * Load Felx Gym scene
 *
 * @param[in] sceneName The name of a pre-compiled scene
 * @param[in] jsonParams Optional json string with scene specific parameters
 */
NV_FLEX_API void NvFlexGymLoadScene(const wchar_t* sceneName, const wchar_t* jsonParams);

/**
 * Reset current Felx Gym scene
 */
NV_FLEX_API void NvFlexGymResetScene();

/**
 * Set agents actions
 *
 * @param[in] actions The array of actions
 * @param[in] firstAction First action index
 * @param[in] numActions Number of actions to set
 */
NV_FLEX_API void NvFlexGymSetActions(const float* actions, int firstAction, int numActions);

/**
 * Get agents observations
 *
 * @param[in] observations The array of observations
 * @param[in] firstObservation First observation index
 * @param[in] numObservations Number of observations to get
 */
NV_FLEX_API void NvFlexGymGetObservations(float* observations, int firstObservation, int numObservations);

/**
 * Get agents extras
 *
 * @param[in] extras The array of extras
 * @param[in] firstExtra First extra index
 * @param[in] numExtras Number of extras to get
 */
NV_FLEX_API void NvFlexGymGetExtras(float* extras, int firstExtra, int numExtras);

/**
 * Get agents rewards
 *
 * @param[in] rewards The array of rewards
 * @param[in] deaths The array of deaths
 * @param[in] firstAgent First agent index
 * @param[in] numAgents Number of agents
 */
NV_FLEX_API void NvFlexGymGetRewards(float* rewards, char* deaths, int firstAgent, int numAgents);

/**
 * Reset agent
 *
 * @param[in] agent Agent index
 */
NV_FLEX_API void NvFlexGymResetAgent(int agent);

/**
 * Reset all agents
 *
 */
NV_FLEX_API void NvFlexGymResetAllAgents();

/**
 * Get the index of a rigid body by its agent index and name
 *
 * @param[in] agentIndex Agent index
 * @param[in] bodyName Rigid body name
 */
NV_FLEX_API int NvFlexGymGetRigidBodyIndex(int agentIndex, const wchar_t* bodyName);

/**
 * Map rigid body array
 *
 */
NV_FLEX_API void NvFlexGymMapRigidBodyArray();

/**
 * Unmap rigid body array
 *
 */
NV_FLEX_API void NvFlexGymUnmapRigidBodyArray();

/**
 * Get the offset of a rigid body structure field by its name
 *
 * @param[in] fieldName Field name
 */
NV_FLEX_API int NvFlexGymGetRigidBodyFieldOffset(const wchar_t* fieldName);

/**
 * Get the size of a rigid body structure field by its name
 *
 * @param[in] fieldName Field name
 */
NV_FLEX_API int NvFlexGymGetRigidBodyFieldSize(const wchar_t* fieldName);

/**
 * Get the pointer to a rigid body structure field in the rigid body array
 *
 * @param[in] bodyIndex rigid body index
 * @param[in] fieldOffset Field offset
 */
NV_FLEX_API void* NvFlexGymGetRigidBodyField(int bodyIndex, int fieldOffset);

/**
 * Get the index of a rigid joint by its agent index and name
 *
 * @param[in] agentIndex Agent index
 * @param[in] jointName Rigid joint name
 */
NV_FLEX_API int NvFlexGymGetRigidJointIndex(int agentIndex, const wchar_t* jointName);

/**
 * Map rigid joint array
 *
 */
NV_FLEX_API void NvFlexGymMapRigidJointArray();

/**
 * Unmap rigid body array
 *
 */
NV_FLEX_API void NvFlexGymUnmapRigidJointArray();

/**
 * Get the offset of a rigid joint structure field by its name
 *
 * @param[in] fieldName Field name
 */
NV_FLEX_API int NvFlexGymGetRigidJointFieldOffset(const wchar_t* fieldName);

/**
 * Get the size of a rigid joint structure field by its name
 *
 * @param[in] fieldName Field name
 */
NV_FLEX_API int NvFlexGymGetRigidJointFieldSize(const wchar_t* fieldName);

/**
 * Get the pointer to a rigid joint structure field in the rigid body array
 *
 * @param[in] jointIndex rigid joint index
 * @param[in] fieldOffset Field offset
 */
NV_FLEX_API void* NvFlexGymGetRigidJointField(int jointIndex, int fieldOffset);

/**
 * Sets the random seed.
 * 
 * @param[in] seed Random Seed
 */
NV_FLEX_API void NvFlexGymSetSeed(uint32_t seed);

/**
 * Create a ray batch for raycast
 *
 * @return The index of a new ray batch
 */
NV_FLEX_API int NvFlexGymCreateRayBatch();

/**
 * Destroy a ray batch
 *
 * @param[in] batchIndex ray batch index
 */
NV_FLEX_API void NvFlexGymDestroyRayBatch(int batchIndex);

/**
 * Add a ray to a ray batch
 *
 * @param[in] batchIndex ray batch index
 * @return The index of a new ray
 */
NV_FLEX_API int NvFlexGymRayBatchAddRay(int batchIndex);

/**
 * Set ray parameters
 *
 * @param[in] batchIndex ray batch index
 * @param[in] rayIndex ray batch index
 * @param[in] rayStart ray start. Ray start position in world space
 * @param[in] rayDirection ray direction. Ray dir in world space, should be normalized
 * @param[in] rayFilter ray filter. Will not report hits with any shape that shares a bit with this mask
 * @param[in] rayGroup ray group. Will report shapes with matching group ID. Set -1 for any group ID
 * @param[in] rayMaxDistance ray max distance. Will not consider hits past this point
 */
NV_FLEX_API void NvFlexGymRayBatchSetRay(int batchIndex, int rayIndex, const float rayStart[3], const float rayDirection[3], int rayFilter, int rayGroup, float rayMaxDistance);

/**
 * Cast all rays in a ray batch
 *
 * @param[in] batchIndex ray batch index
 */
NV_FLEX_API void NvFlexGymRayBatchCastRays(int batchIndex);

/**
 * Get ray hit information
 *
 * @param[in] batchIndex ray batch index
 * @param[in] rayIndex ray batch index
 * @param[out] shapeIndex ray hit shape index. Will be -1 if no hit
 * @param[out] shapeElementIndex ray hit shape element index. E.g.: triangle index
 * @param[out] hitDistance ray hit distance
 * @param[out] hitNormal ray hit normal in world space
 */
NV_FLEX_API void NvFlexGymRayBatchGetHit(int batchIndex, int rayIndex, int shapeIndex[1], int shapeElementIndex[1], float hitDistance[1], float hitNormal[3]);


} // extern "C"

#endif // NV_FLEX_GYM_H
