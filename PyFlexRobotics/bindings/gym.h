// This file is #include'd in the end of main.cpp
#ifdef NV_FLEX_GYM

#include <codecvt>

#include "../external/rl/RLFlexEnv.h"


namespace convert {

std::string w2s(const std::wstring &var)
{
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
	return conv.to_bytes(var);
}

std::wstring s2w(const std::string &var)
{
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
	return conv.from_bytes(var);
}

} // convert

namespace gym {

	struct GymRayBatch
	{
		GymRayBatch() : rays(g_flexLib), hits(g_flexLib) {}

		int AddRay()
		{
			int rayIndex = (int)rays.size();
			if (!rays.mappedPtr) rays.map();
			rays.push_back(NvFlexRay());
			if (!hits.mappedPtr) hits.map();
			hits.push_back(NvFlexRayHit());
			return rayIndex;
		}
		void SetRay(int rayIndex, const Vec3& rayStart, const Vec3& rayDirection, int rayFilter, int rayGroup, float rayMaxDistance)
		{
			if (rayIndex < (int)rays.size())
			{
				NvFlexRay ray;
				(Vec3&)ray.start = rayStart;
				(Vec3&)ray.dir = rayDirection;
				ray.filter = rayFilter;
				ray.group = rayGroup;
				ray.maxT = rayMaxDistance;
				if (!rays.mappedPtr) rays.map();
				rays[rayIndex] = ray;
			}
		}
		void CastRays()
		{
			if (rays.mappedPtr) rays.unmap();
			if (hits.mappedPtr) hits.unmap();
			NvFlexRayCast(g_solver, rays.buffer, hits.buffer, (int)rays.size());
		}
		void GetHit(int rayIndex, int& shapeIndex, int& shapeElementIndex, float& hitDistance, Vec3& hitNormal)
		{
			if (rayIndex < (int)hits.size())
			{
				if (!hits.mappedPtr) hits.map();
				NvFlexRayHit hit = hits[rayIndex];
				shapeIndex = hit.shape;
				shapeElementIndex = hit.element;
				hitDistance = hit.t;
				hitNormal = (Vec3&)hit.n;
			}
			else
				shapeIndex = -1;
		}

		static int Create()
		{
			int batchIndex = -1;
			if (sm_freeRayBatches.empty())
			{
				batchIndex = (int)sm_gymRayBatches.size();
				sm_gymRayBatches.push_back(nullptr);
			}
			else
			{
				batchIndex = sm_freeRayBatches.back();
				sm_freeRayBatches.pop_back();
			}
			sm_gymRayBatches[batchIndex] = new GymRayBatch();
			return batchIndex;
		}
		static void Destroy(int batchIndex)
		{
			if (batchIndex < (int)sm_gymRayBatches.size() && sm_gymRayBatches[batchIndex] != nullptr)
			{
				delete sm_gymRayBatches[batchIndex];
				sm_gymRayBatches[batchIndex] = nullptr;
				sm_freeRayBatches.push_back(batchIndex);
			}
		}
		static int AddRay(int batchIndex)
		{
			if (batchIndex < (int)sm_gymRayBatches.size() && sm_gymRayBatches[batchIndex] != nullptr)
			{
				return sm_gymRayBatches[batchIndex]->AddRay();
			}
			return -1;
		}
		static void SetRay(int batchIndex, int rayIndex, const Vec3& rayStart, const Vec3& rayDirection, int rayFilter, int rayGroup, float rayMaxDistance)
		{
			if (batchIndex < (int)sm_gymRayBatches.size() && sm_gymRayBatches[batchIndex] != nullptr)
			{
				sm_gymRayBatches[batchIndex]->SetRay(rayIndex, rayStart, rayDirection, rayFilter, rayGroup, rayMaxDistance);
			}
		}
		static void CastRays(int batchIndex)
		{
			if (batchIndex < (int)sm_gymRayBatches.size() && sm_gymRayBatches[batchIndex] != nullptr)
			{
				sm_gymRayBatches[batchIndex]->CastRays();
			}
		}
		static void GetHit(int batchIndex, int rayIndex, int& shapeIndex, int& shapeElementIndex, float& hitDistance, Vec3& hitNormal)
		{
			if (batchIndex < (int)sm_gymRayBatches.size() && sm_gymRayBatches[batchIndex] != nullptr)
			{
				sm_gymRayBatches[batchIndex]->GetHit(rayIndex, shapeIndex, shapeElementIndex, hitDistance, hitNormal);
			}
		}
		static void Clear()
		{
			for (auto e : sm_gymRayBatches) delete e;
			sm_gymRayBatches.clear();
			sm_freeRayBatches.clear();
		}

	private:
		NvFlexVector<NvFlexRay> rays;
		NvFlexVector<NvFlexRayHit> hits;
		static std::vector<GymRayBatch*> sm_gymRayBatches;
		static std::vector<int> sm_freeRayBatches;
	};
	std::vector<GymRayBatch*> GymRayBatch::sm_gymRayBatches;
	std::vector<int> GymRayBatch::sm_freeRayBatches;

} // gym

extern "C" {

NV_FLEX_API void NvFlexGymInit(const NvFlexGymInitParams* initParams)
{
	NvFlexGymSetSeed(initParams->seed);	

	if (initParams && initParams->renderBackend == eNvFlexNoRendering)
	{
		g_headless = true;
		g_render = false;
		g_interop = false;
		g_pause = false;
	}
	else if (initParams && initParams->renderBackend == eNvFlexHeadless)
	{
		g_headless = true;
		g_render = true;
		g_interop = false;
		g_pause = false;
	}
	else
	{
		g_headless = false;
		g_render = true;
	}

	g_screenWidth = initParams->screenWidth;
	g_screenHeight = initParams->screenHeight;
	g_msaaSamples = initParams->msaaSamples;
	g_device = initParams->device;
	g_rank = initParams->rank;
	g_vsync = initParams->vsync;

	// Add empty default scene
	RegisterScene("Empty", []() { return new Scene(); });

	RegisterPhysicsScenes();

	if (g_headless == false)
	{
		SDLInit("Flex Gym");
	
		RenderInitOptions options;
		options.window = g_window;
		options.numMsaaSamples = g_msaaSamples;
		options.asyncComputeBenchmark = g_asyncComputeBenchmark;
		options.defaultFontHeight = -1;
		options.fullscreen = g_fullscreen;
		
		InitRender(options);
		ReshapeWindow(g_screenWidth, g_screenHeight);
	}
	else if (g_render == true)
	{
		RenderInitOptions options;
		options.numMsaaSamples = g_msaaSamples;

		InitRenderHeadless(options, g_screenWidth, g_screenHeight);
	}

	NvFlexInitDesc desc;
	desc.deviceIndex = g_device;
	desc.enableExtensions = false;
	desc.renderDevice = 0;
	desc.renderContext = 0;
	desc.computeContext = 0;
	desc.computeType = eNvFlexCUDA;

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

	if (g_render == true) {
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

	// init default scene
	InitScene(g_sceneIndex);
}

NV_FLEX_API void NvFlexGymAcquireContext()
{
	// ensure CUDA context is active
	NvFlexAcquireContext(g_flexLib);

	// ensure OpenGL context is active
	AcquireRenderContext();

}

NV_FLEX_API void NvFlexGymRestoreContext()
{
	// clears any active OpenGL context
	ClearRenderContext();

	// reset previous CUDA context
	NvFlexRestoreContext(g_flexLib);
}

NV_FLEX_API int NvFlexGymUpdate()
{
	UpdateFrame();
	
	if (g_headless==true)
		return 0;


	bool quit = false;
	SDL_Event e;
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
	return quit ? 1 : 0;
}

NV_FLEX_API void NvFlexGymShutdown()
{
	gym::GymRayBatch::Clear();

	g_sceneFactories.clear();

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
}

NV_FLEX_API void NvFlexGymLoadScene(const wchar_t* sceneName, const wchar_t* jsonParams)
{
	//assert(0);
	if (SetSceneIndexByName(convert::w2s(sceneName).c_str()))
	{
		g_sceneJson.clear();
		if (jsonParams)
		{
			json cmdJson = JsonFromString(convert::w2s(jsonParams).c_str());
			if (!cmdJson.is_null())
			{
				MergeJson(g_sceneJson, cmdJson);
			}
		}
		InitScene(g_sceneIndex);
	}
}

NV_FLEX_API void NvFlexGymResetScene()
{
	InitScene(g_sceneIndex, false);
}

NV_FLEX_API void NvFlexGymSetActions(const float* actions, int firstAction, int numActions)
{	
	if (g_rlflexenv)
		g_rlflexenv->SetActions(actions, firstAction, numActions);
}

NV_FLEX_API void NvFlexGymSetPyToC(const float* pytoc)
{	
	if (g_rlflexenv)
		g_rlflexenv->SetPyToC(pytoc);
}

NV_FLEX_API void NvFlexGymPrepareHER(int x)
{	
	if (g_rlflexenv)
		g_rlflexenv->PrepareHER(x);
}

NV_FLEX_API int NvFlexGymGetNumActions()
{
	if (g_rlflexenv)
		return g_rlflexenv->GetNumActions();
	return -1;
}

NV_FLEX_API int NvFlexGymGetNumObservations()
{
	if (g_rlflexenv)
		return g_rlflexenv->GetNumObservations();
	return -1;
}

NV_FLEX_API int NvFlexGymGetNumExtras()
{
	if (g_rlflexenv)
		return g_rlflexenv->GetNumExtras();
	return -1;
}

NV_FLEX_API int NvFlexGymGetNumPyToC()
{
	if (g_rlflexenv)
		return g_rlflexenv->GetNumPyToC();
	return -1;
}

NV_FLEX_API void NvFlexGymGetObservations(float* observations, int firstObservation, int numObservations)
{
	if (g_rlflexenv)
		std::memcpy(observations, g_rlflexenv->GetObservations() + firstObservation, sizeof(float) * std::max(0, std::min(numObservations, g_rlflexenv->GetNumAgents() * g_rlflexenv->GetNumObservations() - firstObservation)));
}

NV_FLEX_API void NvFlexGymGetExtras(float* extras, int firstExtra, int numExtras)
{
	if (g_rlflexenv)
		std::memcpy(extras, g_rlflexenv->GetExtras() + firstExtra, sizeof(float) * std::max(0, std::min(numExtras, g_rlflexenv->GetNumAgents() * g_rlflexenv->GetNumExtras() - firstExtra)));
}

NV_FLEX_API void NvFlexGymGetRewards(float* rewards, char* deaths, int firstAgent, int numAgents)
{
	if (g_rlflexenv)
	{
		std::memcpy(rewards, g_rlflexenv->GetRewards() + firstAgent, sizeof(float) * std::max(0, std::min(numAgents, g_rlflexenv->GetNumAgents() - firstAgent)));
		std::memcpy(deaths, g_rlflexenv->GetDeaths() + firstAgent, std::max(0, std::min(numAgents, g_rlflexenv->GetNumAgents() - firstAgent)));
	}
}

NV_FLEX_API void NvFlexGymResetAgent(int agent)
{
	if (g_rlflexenv)
	{
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		g_buffers->rigidBodies.map();
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
		g_buffers->rigidJoints.map();

		g_rlflexenv->ResetAgent(agent);

		g_rlflexenv->PopulateState(agent, &(g_rlflexenv->GetObservations()[agent * g_rlflexenv->GetNumObservations()]));

		g_buffers->rigidBodies.unmap();
		g_buffers->rigidJoints.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
	}
}

NV_FLEX_API void NvFlexGymResetAllAgents()
{
	if (g_rlflexenv)
	{
		NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
		g_buffers->rigidBodies.map();
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
		g_buffers->rigidJoints.map();

		g_rlflexenv->ResetAllAgents();

		for (int agent = 0; agent < g_rlflexenv->GetNumAgents(); ++agent)
			g_rlflexenv->PopulateState(agent, &(g_rlflexenv->GetObservations()[agent * g_rlflexenv->GetNumObservations()]));

		g_buffers->rigidBodies.unmap();
		g_buffers->rigidJoints.unmap();
		NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size());
	}
}

NV_FLEX_API void NvFlexPopulateExtras()
{
	if (g_rlflexenv)
	{
		int numExtras = g_rlflexenv->GetNumExtras();
		if (numExtras > 0)
		{
			g_buffers->rigidBodies.map();
			float* extras = g_rlflexenv->GetExtras();
			for (int ai = 0; ai < g_rlflexenv->GetNumAgents(); ai++)
			{
				float *a = &extras[ai * numExtras];
				g_rlflexenv->PopulateExtra(ai, a);
			}
			g_buffers->rigidBodies.unmap();
		}
	}
}

NV_FLEX_API int NvFlexGymGetRigidBodyIndex(int agentIndex, const wchar_t* bodyName)
{
	if (g_rlflexenv)
		g_rlflexenv->GetRigidBodyIndex(agentIndex, bodyName);
	return -1;
}

NV_FLEX_API void NvFlexGymMapRigidBodyArray()
{
	if (g_buffers->rigidBodies.mappedPtr == nullptr)
		g_buffers->rigidBodies.map();
}

NV_FLEX_API void NvFlexGymUnmapRigidBodyArray()
{
	if (g_buffers->rigidBodies.mappedPtr != nullptr)
		g_buffers->rigidBodies.unmap();
}

NV_FLEX_API void NvFlexGymSetSeed(uint32_t seed)
{
	srand(seed);
	RandInit(seed);
}

NV_FLEX_API int NvFlexGymGetRigidBodyFieldOffset(const wchar_t* fieldName)
{
#define LSTR(X) L ## X
#define CHECK_CASE(NAME) if (std::wstring(LSTR(#NAME)) == fieldName) return offsetof(NvFlexRigidBody, NAME)

	CHECK_CASE(com);
	CHECK_CASE(theta);
	CHECK_CASE(linearVel);
	CHECK_CASE(angularVel);
	CHECK_CASE(force);
	CHECK_CASE(torque);
	CHECK_CASE(origin);
	CHECK_CASE(mass);
	CHECK_CASE(inertia);
	CHECK_CASE(invMass);
	CHECK_CASE(invInertia);
	CHECK_CASE(linearDamping);
	CHECK_CASE(angularDamping);
	CHECK_CASE(maxLinearVelocity);
	CHECK_CASE(maxAngularVelocity);
	CHECK_CASE(flags);

#undef CHECK_CASE
#undef LSTR

	return -1;
}

NV_FLEX_API int NvFlexGymGetRigidBodyFieldSize(const wchar_t* fieldName)
{
#define LSTR(X) L ## X
#define CHECK_CASE(NAME) if (std::wstring(LSTR(#NAME)) == fieldName) return sizeof(NvFlexRigidBody::NAME)

	CHECK_CASE(com);
	CHECK_CASE(theta);
	CHECK_CASE(linearVel);
	CHECK_CASE(angularVel);
	CHECK_CASE(force);
	CHECK_CASE(torque);
	CHECK_CASE(origin);
	CHECK_CASE(mass);
	CHECK_CASE(inertia);
	CHECK_CASE(invMass);
	CHECK_CASE(invInertia);
	CHECK_CASE(linearDamping);
	CHECK_CASE(angularDamping);
	CHECK_CASE(maxLinearVelocity);
	CHECK_CASE(maxAngularVelocity);
	CHECK_CASE(flags);

#undef CHECK_CASE
#undef LSTR

	return -1;
}

NV_FLEX_API void* NvFlexGymGetRigidBodyField(int bodyIndex, int fieldOffset)
{
	if (g_buffers->rigidBodies.mappedPtr == nullptr) return nullptr;
	return (char*)(&g_buffers->rigidBodies[bodyIndex]) + fieldOffset;
}

NV_FLEX_API int NvFlexGymGetRigidJointIndex(int agentIndex, const wchar_t* jointName)
{
	if (g_rlflexenv)
		g_rlflexenv->GetRigidJointIndex(agentIndex, jointName);
	return -1;
}

NV_FLEX_API void NvFlexGymMapRigidJointArray()
{
	if (g_buffers->rigidJoints.mappedPtr == nullptr)
		g_buffers->rigidJoints.map();
}

NV_FLEX_API void NvFlexGymUnmapRigidJointArray()
{
	if (g_buffers->rigidJoints.mappedPtr != nullptr)
		g_buffers->rigidJoints.unmap();
}

NV_FLEX_API int NvFlexGymGetRigidJointFieldOffset(const wchar_t* fieldName)
{
#define LSTR(X) L ## X
#define CHECK_CASE(NAME) if (std::wstring(LSTR(#NAME)) == fieldName) return offsetof(NvFlexRigidJoint, NAME)

	CHECK_CASE(body0);
	CHECK_CASE(body1);
	CHECK_CASE(pose0);
	CHECK_CASE(pose1);
	CHECK_CASE(modes);
	CHECK_CASE(targets);
	CHECK_CASE(lowerLimits);
	CHECK_CASE(upperLimits);
	CHECK_CASE(compliance);
	CHECK_CASE(damping);
	CHECK_CASE(motorLimit);
	CHECK_CASE(lambda);
	CHECK_CASE(mimicIndex);
	CHECK_CASE(mimicScale);
	CHECK_CASE(mimicOffset);
	CHECK_CASE(maxIterations);
	CHECK_CASE(flags);

#undef CHECK_CASE
#undef LSTR

	return -1;
}

NV_FLEX_API int NvFlexGymGetRigidJointFieldSize(const wchar_t* fieldName)
{
#define LSTR(X) L ## X
#define CHECK_CASE(NAME) if (std::wstring(LSTR(#NAME)) == fieldName) return sizeof(NvFlexRigidJoint::NAME)

	CHECK_CASE(body0);
	CHECK_CASE(body1);
	CHECK_CASE(pose0);
	CHECK_CASE(pose1);
	CHECK_CASE(modes);
	CHECK_CASE(targets);
	CHECK_CASE(lowerLimits);
	CHECK_CASE(upperLimits);
	CHECK_CASE(compliance);
	CHECK_CASE(damping);
	CHECK_CASE(motorLimit);
	CHECK_CASE(lambda);
	CHECK_CASE(mimicIndex);
	CHECK_CASE(mimicScale);
	CHECK_CASE(mimicOffset);
	CHECK_CASE(maxIterations);
	CHECK_CASE(flags);

#undef CHECK_CASE
#undef LSTR

	return -1;
}

NV_FLEX_API void* NvFlexGymGetRigidJointField(int jointIndex, int fieldOffset)
{
	if (g_buffers->rigidJoints.mappedPtr == nullptr) return nullptr;
	return (char*)(&g_buffers->rigidJoints[jointIndex]) + fieldOffset;
}

////////

NV_FLEX_API int NvFlexGymCreateRayBatch()
{
	return gym::GymRayBatch::Create();
}

NV_FLEX_API void NvFlexGymDestroyRayBatch(int batchIndex)
{
	gym::GymRayBatch::Destroy(batchIndex);
}

NV_FLEX_API int NvFlexGymRayBatchAddRay(int batchIndex)
{
	return gym::GymRayBatch::AddRay(batchIndex);
}

NV_FLEX_API void NvFlexGymRayBatchSetRay(int batchIndex, int rayIndex, const float rayStart[3], const float rayDirection[3], int rayFilter, int rayGroup, float rayMaxDistance)
{
	gym::GymRayBatch::SetRay(batchIndex, rayIndex, *(const Vec3*)rayStart, *(const Vec3*)rayDirection, rayFilter, rayGroup, rayMaxDistance);
}

NV_FLEX_API void NvFlexGymRayBatchCastRays(int batchIndex)
{
	gym::GymRayBatch::CastRays(batchIndex);
}

NV_FLEX_API void NvFlexGymRayBatchGetHit(int batchIndex, int rayIndex, int shapeIndex[1], int shapeElementIndex[1], float hitDistance[1], float hitNormal[3])
{
	gym::GymRayBatch::GetHit(batchIndex, rayIndex, *shapeIndex, *shapeElementIndex, *hitDistance, *(Vec3*)hitDistance);
}

} // extern "C"

#endif
