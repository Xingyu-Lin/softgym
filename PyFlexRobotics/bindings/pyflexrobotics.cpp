#include <bindings/main.cpp>

void pyflex_init(bool headless=false, bool render=true, int camera_width=720, int camera_height=720) {
    g_screenWidth = g_windowWidth = camera_width;
    g_screenHeight = g_windowHeight = camera_height;

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

    printf("PyFlexRobotics init done!\n");
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

void pyflex_step(py::array_t<float> update_params, int capture, char *path, int render) {
    int temp_render = g_render;
    g_render = render;

    if (capture == 1) {
        g_capture = true;
        g_ffmpeg = fopen(path, "wb");
    }

    UpdateFrame(update_params);
    SDL_EventFunc();

    if (capture == 1) {
        g_capture = false;
        fclose(g_ffmpeg);
        g_ffmpeg = nullptr;
    }
    g_render = temp_render;
}

void pyflex_loop() { SDLMainLoop();}

float rand_float(float LO, float HI) {
    return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

void pyflex_set_scene(int scene_idx, py::array_t<float> scene_params, int thread_idx = 0, py::array_t<float> robot_params = py::array_t<float>()) {
    int g_sceneIdx = scene_idx;
    g_selectedScene = g_sceneIdx;
    InitScene(g_selectedScene, scene_params, true, thread_idx, robot_params);
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

py::array_t<float> pyflex_render_sensor(int sensor_id) {
    RenderSensor s = g_renderSensors[sensor_id];
    auto rendered_img = py::array_t<float>((int) s.width * s.height * 4);
    auto rendered_img_ptr = (float *) rendered_img.request().ptr;
    float* rgbd = ReadSensor(sensor_id);
    for (int i=0; i< s.width * s.height *4; ++i)
        rendered_img_ptr[i] = rgbd[i];
    return rendered_img;
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

    SDL_EventFunc();

    if (capture == 1) {
        g_capture = false;
        fclose(g_ffmpeg);
        g_ffmpeg = nullptr;
    }

    return rendered_img;
}

void pyflex_set_sensor_segment(bool flag) {g_sensor_segment=flag;}

py::array_t<float> pyflex_get_robot_state(){return g_scene->GetRobotState();}
void pyflex_set_robot_state(py::array_t<float> robot_state){g_scene->SetRobotState(robot_state);}

int main() {
    cout<<"PyFlexRobotics loaded" <<endl;
    pyflex_init();
    pyflex_clean();

    return 0;
}

PYBIND11_MODULE(pyflex, m) {
    m.def("main", &main);
    m.def("init", &pyflex_init);
    m.def("set_scene", &pyflex_set_scene,
          py::arg("scene_idx"),
          py::arg("scene_params") = nullptr,
          py::arg("thread_idx") = 0,
          py::arg("robot_params") = py::array_t<float>());
    m.def("clean", &pyflex_clean);
    m.def("step", &pyflex_step,
          py::arg("update_params") = nullptr,
          py::arg("capture") = 0,
          py::arg("path") = nullptr,
          py::arg("render") = 0);
    m.def("loop", &pyflex_loop);
    m.def("render", &pyflex_render,
          py::arg("capture") = 0,
          py::arg("path") = nullptr
        );
    m.def("render_sensor", &pyflex_render_sensor, py::arg("sensor_id")= 0);
    m.def("set_sensor_segment", &pyflex_set_sensor_segment, py::arg("flag"));
    m.def("get_camera_params", &pyflex_get_camera_params, "Get camera parameters");
    m.def("set_camera_params", &pyflex_set_camera_params, "Set camera parameters");

    m.def("add_box", &pyflex_add_box, "Add box to the scene");
    m.def("add_sphere", &pyflex_add_sphere, "Add sphere to the scene");
    m.def("add_capsule", &pyflex_add_capsule, "Add capsule to the scene");

    m.def("pop_box", &pyflex_pop_box, "remove box from the scene");

    m.def("get_n_particles", &pyflex_get_n_particles, "Get the number of particles");
    m.def("get_n_shapes", &pyflex_get_n_shapes, "Get the number of shapes");
//    m.def("get_n_rigids", &pyflex_get_n_rigids, "Get the number of rigids");
//    m.def("get_n_rigidPositions", &pyflex_get_n_rigidPositions, "Get the number of rigid positions");

    m.def("get_phases", &pyflex_get_phases, "Get particle phases");
    m.def("set_phases", &pyflex_set_phases, "Set particle phases");
    m.def("get_groups", &pyflex_get_groups, "Get particle groups");
    m.def("set_groups", &pyflex_set_groups, "Set particle groups");
//    // TODO: Add keyword set_color for set_phases function and also in python code
    m.def("get_positions", &pyflex_get_positions, "Get particle positions");
    m.def("set_positions", &pyflex_set_positions, "Set particle positions");
    m.def("get_restPositions", &pyflex_get_restPositions, "Get particle restPositions");
////    m.def("get_rigidOffsets", &pyflex_get_rigidOffsets, "Get rigid offsets");
////    m.def("get_rigidIndices", &pyflex_get_rigidIndices, "Get rigid indices");
////    m.def("get_rigidLocalPositions", &pyflex_get_rigidLocalPositions, "Get rigid local positions");
////    m.def("get_rigidGlobalPositions", &pyflex_get_rigidGlobalPositions, "Get rigid global positions");
////    m.def("get_rigidRotations", &pyflex_get_rigidRotations, "Get rigid rotations");
////    m.def("get_rigidTranslations", &pyflex_get_rigidTranslations, "Get rigid translations");
//
//    m.def("get_sceneParams", &pyflex_get_sceneParams, "Get scene parameters");
//
    m.def("get_velocities", &pyflex_get_velocities, "Get particle velocities");
    m.def("set_velocities", &pyflex_set_velocities, "Set particle velocities");

    m.def("get_shape_states", &pyflex_get_shape_states, "Get shape states");
    m.def("set_shape_states", &pyflex_set_shape_states, "Set shape states");
    m.def("clear_shapes", &ClearShapes, "Clear shapes");

    m.def("get_scene_upper", &pyflex_get_sceneUpper);
    m.def("get_scene_lower", &pyflex_get_sceneLower);
    m.def("get_robot_state", &pyflex_get_robot_state);
    m.def("set_robot_state", &pyflex_set_robot_state);


//    m.def("add_rigid_body", &pyflex_add_rigid_body);
}