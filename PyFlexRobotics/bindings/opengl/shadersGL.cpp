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

#include "../shaders.h"

#include "../../core/mesh.h"
#include "../../core/tga.h"	
#include "../../core/png.h"
#include "../../core/platform.h"
#include "../../core/extrude.h"

#include "../../external/SDL2-2.0.4/include/SDL.h"
#include "../../external/glad/src/glad.c"

#ifdef __linux__
#include "../../external/glad/src/glad_egl.c"
#endif

#include <map>

#include "imguiRenderGL.h"
#include "utilsGL.h"

#include "shader.h"

#ifdef ANDROID
#include "android/Log.h"
#include "android/AndroidDefine.h"
#include "android/AndroidMatrixTool.h"
#endif


#define CudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }

typedef unsigned int VertexBuffer;
typedef unsigned int IndexBuffer;
typedef unsigned int Texture;

struct FluidRenderBuffersGL
{
	FluidRenderBuffersGL(int numParticles = 0):
		mPositionVBO(0),
		mDensityVBO(0),
		mIndices(0),
		mPositionBuf(nullptr),
		mDensitiesBuf(nullptr),
		mIndicesBuf(nullptr)
	{
		mNumParticles = numParticles;
		for (int i = 0; i < 3; i++)
		{ 
			mAnisotropyVBO[i] = 0;
			mAnisotropyBuf[i] = nullptr;
		}
	}
	~FluidRenderBuffersGL()
	{
		glDeleteBuffers(1, &mPositionVBO);
		glDeleteBuffers(3, mAnisotropyVBO);
		glDeleteBuffers(1, &mDensityVBO);
		glDeleteBuffers(1, &mIndices);

		NvFlexUnregisterOGLBuffer(mPositionBuf);
		NvFlexUnregisterOGLBuffer(mDensitiesBuf);
		NvFlexUnregisterOGLBuffer(mIndicesBuf);

		NvFlexUnregisterOGLBuffer(mAnisotropyBuf[0]);
		NvFlexUnregisterOGLBuffer(mAnisotropyBuf[1]);
		NvFlexUnregisterOGLBuffer(mAnisotropyBuf[2]);
	}

	int mNumParticles;
	VertexBuffer mPositionVBO;
	VertexBuffer mDensityVBO;
	VertexBuffer mAnisotropyVBO[3];
	IndexBuffer mIndices;

	// wrapper buffers that allow Flex to write directly to VBOs
	NvFlexBuffer* mPositionBuf;
	NvFlexBuffer* mDensitiesBuf;
	NvFlexBuffer* mAnisotropyBuf[3];
	NvFlexBuffer* mIndicesBuf;
};

// vertex buffers for diffuse particles
struct DiffuseRenderBuffersGL
{
	DiffuseRenderBuffersGL(int numParticles = 0):
		mDiffusePositionVBO(0),
		mDiffuseVelocityVBO(0),
		mDiffuseIndicesIBO(0),
		mDiffuseIndicesBuf(nullptr),
		mDiffusePositionsBuf(nullptr),
		mDiffuseVelocitiesBuf(nullptr)
	{
		mNumParticles = numParticles;
	}
	~DiffuseRenderBuffersGL()
	{
		if (mNumParticles > 0)
		{
			glDeleteBuffers(1, &mDiffusePositionVBO);
			glDeleteBuffers(1, &mDiffuseVelocityVBO);
			glDeleteBuffers(1, &mDiffuseIndicesIBO);

			NvFlexUnregisterOGLBuffer(mDiffuseIndicesBuf);
			NvFlexUnregisterOGLBuffer(mDiffusePositionsBuf);
			NvFlexUnregisterOGLBuffer(mDiffuseVelocitiesBuf);
		}
	}

	int mNumParticles;
	VertexBuffer mDiffusePositionVBO;
	VertexBuffer mDiffuseVelocityVBO;
	IndexBuffer mDiffuseIndicesIBO;

	NvFlexBuffer* mDiffuseIndicesBuf;
	NvFlexBuffer* mDiffusePositionsBuf;
	NvFlexBuffer* mDiffuseVelocitiesBuf;
};

namespace
{

int g_msaaSamples;
GLuint g_msaaFbo;
GLuint g_msaaColorBuf;
GLuint g_msaaDepthBuf;

int g_screenWidth;
int g_screenHeight;

static SDL_Window* g_window;
static SDL_GLContext g_context;

#ifdef __linux__
EGLDisplay* g_eglDisplay;
EGLConfig*  g_eglConfig;
EGLContext* g_eglContext;
EGLSurface* g_eglSurface;
#endif

static float g_spotMin = 0.5f;
static float g_spotMax = 1.0f;
float g_shadowBias = 0.05f;

DepthRenderProfile currentDepthProfile;

static GLuint s_lightingProgram = GLuint(-1);
static GLuint s_shadowProgram = GLuint(-1);
static GLuint s_pointsProgram = GLuint(-1);
static GLuint s_diffuseProgram = GLuint(-1);
static GLuint s_quadProgram = GLuint(-1);

static const int kShadowResolution = 2048;

std::map<std::string, GLuint> g_textures;


} // anonymous namespace

extern Colour g_colors[];

struct ShadowMap
{
	GLuint texture;
	GLuint framebuffer;
};

char font_path[100];

char* make_path(char* full_path, std::string path) {
    strcpy(full_path, getenv("PYFLEXROOT"));
    strcat(full_path, path.c_str());
    return full_path;
}

void InitRender(const RenderInitOptions& options)
{
	SDL_Window* window = options.window;
	int msaaSamples = options.numMsaaSamples;

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);

	// Turn on double buffering with a 24bit Z buffer.
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	g_context = SDL_GL_CreateContext(window);

	// This makes our buffer swap syncronized with the monitor's vertical refresh
	SDL_GL_SetSwapInterval(1);

	if (!gladLoadGLLoader(SDL_GL_GetProcAddress))
	{
		printf("Could not initialize GL extensions\n");
	}

	imguiRenderGLInit(GetFilePathByPlatform(make_path(font_path, "/data/DroidSans.ttf")).c_str());

	g_msaaSamples = msaaSamples;
	g_window = window;
}


void AcquireRenderContext()
{
	if (g_window)
	{	
		// windows mode
		int err = SDL_GL_MakeCurrent(g_window, g_context);
		if (err < 0)
		{
			printf("Could not make OpenGL context current, error: %s\n", SDL_GetError());
		}

		return;
	}
	
#ifdef __linux__

	if (g_eglDisplay)
	{
		// headless mode
		eglMakeCurrent(g_eglDisplay, g_eglSurface, g_eglSurface, g_eglContext);		
		return;
	}
#endif

}

void ClearRenderContext()
{
	int err = SDL_GL_MakeCurrent(NULL, NULL);
	if (err < 0)
	{
		printf("Could not clear OpenGL context, error: %s\n", SDL_GetError());
	}
#ifdef __linux__
	if (g_eglDisplay)
	{
		// headless mode
		eglMakeCurrent(g_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		return;
	}
#endif
}


void InitRenderHeadless(const RenderInitOptions& options, int width, int height)
{
#ifdef __linux__
	int msaaSamples = options.numMsaaSamples;

	EGLint ignore;
	EGLBoolean ok;

    EGLNativeDisplayType native_display = EGL_DEFAULT_DISPLAY;

    const char* s = getenv("EGL_GPU");
    EGLDisplay selectedDisplay;
    if (s != NULL)
    {
        int egl_index = atoi(s);
        
        static const int MAX_DEVICES = 8;
        EGLDeviceEXT eglDevs[MAX_DEVICES];
        EGLint numDevices;

        PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
            (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");

        eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);

        printf("Detected %d devices\n", numDevices);

        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
            (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");

        selectedDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[egl_index], 0);
    }
    else
    {
        selectedDisplay = eglGetDisplay(native_display);
    }

	EGLint eglSurfaceType = EGL_PBUFFER_BIT;


	EGLint eglConfigAttribs[] = {
		EGL_SURFACE_TYPE,          eglSurfaceType,
		EGL_RENDERABLE_TYPE,       EGL_OPENGL_BIT,

		EGL_COLOR_BUFFER_TYPE,     EGL_RGB_BUFFER,
		EGL_BUFFER_SIZE,           32,
		EGL_RED_SIZE,              8,
		EGL_GREEN_SIZE,            8,
		EGL_BLUE_SIZE,             8,
		EGL_ALPHA_SIZE,            8,

		EGL_DEPTH_SIZE,            24,
		EGL_STENCIL_SIZE,          8,

		EGL_SAMPLE_BUFFERS,        0,
		EGL_SAMPLES,               0,

		EGL_NONE,
	};

	static const EGLint eglPBAttribs[] = {
		EGL_WIDTH, width,
		EGL_HEIGHT, height,
		EGL_NONE,
	};


  
	ok = eglBindAPI(EGL_OPENGL_API);
	if (!ok)
		printf("eglBindAPI(0x%x) failed", EGL_OPENGL_API);

	EGLDisplay g_eglDisplay = selectedDisplay;
    
	if (g_eglDisplay == EGL_NO_DISPLAY)
		printf("eglGetDisplay() failed");

	ok = eglInitialize(g_eglDisplay, &ignore, &ignore);
	if (!ok)
		printf("eglInitialize() failed");

	EGLint configs_size = 256;
	EGLConfig* configs = new EGLConfig[configs_size];
	EGLint num_configs;
	ok = eglChooseConfig(
						 g_eglDisplay,
						 eglConfigAttribs,
						 configs,
						 configs_size, // num requested configs
						 &num_configs); // num returned configs

	if (!ok)
		printf("eglChooseConfig() failed");
	if (num_configs == 0)
		printf("failed to find suitable EGLConfig");

	g_eglConfig = configs[0];
	delete [] configs;

  
	g_eglContext = eglCreateContext(
							   g_eglDisplay,
							   g_eglConfig,
							   EGL_NO_CONTEXT,
							   NULL);
	if (!g_eglContext)
		printf("eglCreateContext() failed");
  
	g_eglSurface = eglCreatePbufferSurface(g_eglDisplay, g_eglConfig,
									  eglPBAttribs);
	if (!g_eglSurface)
		printf("eglCreatePbufferSurface() failed");


	
	ok = eglMakeCurrent(g_eglDisplay, g_eglSurface, g_eglSurface, g_eglContext);
	if (!ok)
		printf("eglMakeCurrent() failed");

	// Check if surface is double buffered.
	EGLint render_buffer;
	ok = eglQueryContext(
						 g_eglDisplay,
						 g_eglContext,
						 EGL_RENDER_BUFFER,
						 &render_buffer);
	if (!ok)
		printf("eglQueyContext(EGL_RENDER_BUFFER) failed");
	if (render_buffer == EGL_SINGLE_BUFFER)
		printf("warn: EGL surface is single buffered\n");




	
	if (!gladLoadEGL())
	{
		printf("Could not initialize EGL extensions\n");
	}

	if (!gladLoadGL())
	{
		printf("Could not initialize GL extensions\n");
	}

	g_msaaSamples = msaaSamples;





	if (g_msaaSamples)
	{
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));

		if (g_msaaFbo)
		{
			glVerify(glDeleteFramebuffers(1, &g_msaaFbo));
			glVerify(glDeleteRenderbuffers(1, &g_msaaColorBuf));
			glVerify(glDeleteRenderbuffers(1, &g_msaaDepthBuf));
		}

		int samples;
		glGetIntegerv(GL_MAX_SAMPLES_EXT, &samples);

		// clamp samples to 4 to avoid problems with point sprite scaling
		//samples = Min(samples, Min(g_msaaSamples, 4));
		samples = g_msaaSamples;

		glVerify(glGenFramebuffers(1, &g_msaaFbo));
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));

		glVerify(glGenRenderbuffers(1, &g_msaaColorBuf));
		glVerify(glBindRenderbuffer(GL_RENDERBUFFER, g_msaaColorBuf));
		glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, width, height));

		glVerify(glGenRenderbuffers(1, &g_msaaDepthBuf));
		glVerify(glBindRenderbuffer(GL_RENDERBUFFER, g_msaaDepthBuf));
		glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT, width, height));
		glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_msaaDepthBuf));

		glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, g_msaaColorBuf));

		glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

		glEnable(GL_MULTISAMPLE);
	}

	g_screenWidth = width;
	g_screenHeight = height;

#endif

}


void DestroyRender()
{

	s_lightingProgram = GLuint(-1);
	s_shadowProgram = GLuint(-1);
	s_pointsProgram = GLuint(-1);
	s_diffuseProgram = GLuint(-1);
	s_quadProgram = GLuint(-1);

	g_textures.clear();
}

void StartFrame(Vec4 clearColor)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);

	glPointSize(5.0f);

	glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER_EXT, g_msaaFbo));
	glVerify(glClearColor(powf(clearColor.x, 1.0f / 2.2f), powf(clearColor.y, 1.0f / 2.2f), powf(clearColor.z, 1.0f / 2.2f), 0.0f));
	glVerify(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));


}

void EndFrame()
{
	if (g_msaaFbo)
	{
		// blit the msaa buffer to the window
		glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER, g_msaaFbo));
		glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
		glVerify(glBlitFramebuffer(0, 0, g_screenWidth, g_screenHeight, 0, 0, g_screenWidth, g_screenHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
	}

	// render help to back buffer
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	glVerify(glClear(GL_DEPTH_BUFFER_BIT));

}

void EndFrame(size_t sourceFbHandle, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
{
	glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(sourceFbHandle)));
	glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
	glVerify(glBlitFramebuffer(0, 0, sourceWidth, sourceHeight, 0, 0, targetWidth, targetHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR));

	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	glVerify(glClear(GL_DEPTH_BUFFER_BIT));
}

void CopyFramebufferTo(size_t targetFbHandle)
{
	glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER, g_msaaFbo));
	glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(targetFbHandle)));
	glVerify(glBlitFramebuffer(0, 0, g_screenWidth, g_screenHeight, 0, 0, g_screenWidth, g_screenHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR));
	glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_msaaFbo));
}

void SetView(Matrix44 view, Matrix44 proj)
{
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(proj);

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(view);
}

void GetView(Matrix44& view, Matrix44& proj)
{
	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);

}


void SetFillMode(bool wireframe)
{

	glPolygonMode(GL_FRONT_AND_BACK, wireframe?GL_LINE:GL_FILL);	
	glLineWidth(0.5f);

	// offset solid to avoid z-fighting when drawing wireframe overlay
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.f, 1.f);
}

void SetCullMode(bool enabled)
{
	if (enabled)
		glEnable(GL_CULL_FACE);		
	else
		glDisable(GL_CULL_FACE);		
}


void imguiGraphDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);
	glActiveTexture(GL_TEXTURE1);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE2);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE3);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE4);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_TEXTURE_CUBE_MAP);
	glActiveTexture(GL_TEXTURE5);
	glDisable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE0);

	glDisable(GL_BLEND);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);
	glDisable(GL_POINT_SPRITE);

	// save scene camera transform
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	const Matrix44 ortho = OrthographicMatrix(0.0f, float(g_screenWidth), 0.0f, float(g_screenHeight), -1.0f, 1.0f);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadMatrixf(ortho);

	glUseProgram(0);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_TEXTURE_2D);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	imguiRenderGLDraw();

	// restore camera transform (for picking)
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

void ReshapeRender(int width, int height)
{
	if (g_msaaSamples)
	{
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));

		if (g_msaaFbo)
		{
			glVerify(glDeleteFramebuffers(1, &g_msaaFbo));
			glVerify(glDeleteRenderbuffers(1, &g_msaaColorBuf));
			glVerify(glDeleteRenderbuffers(1, &g_msaaDepthBuf));
		}

		int samples;
		glGetIntegerv(GL_MAX_SAMPLES_EXT, &samples);

		// clamp samples to 4 to avoid problems with point sprite scaling
		//samples = Min(samples, Min(g_msaaSamples, 4));
		samples = g_msaaSamples;

		glVerify(glGenFramebuffers(1, &g_msaaFbo));
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));

		glVerify(glGenRenderbuffers(1, &g_msaaColorBuf));
		glVerify(glBindRenderbuffer(GL_RENDERBUFFER, g_msaaColorBuf));
		glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, width, height));

		glVerify(glGenRenderbuffers(1, &g_msaaDepthBuf));
		glVerify(glBindRenderbuffer(GL_RENDERBUFFER, g_msaaDepthBuf));
		glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT, width, height));
		glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_msaaDepthBuf));

		glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, g_msaaColorBuf));

		glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

		glEnable(GL_MULTISAMPLE);
	}

	g_screenWidth = width;
	g_screenHeight = height;
}

void GetViewRay(int x, int y, Vec3& origin, Vec3& dir)
{
	float modelview[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview);

	float projection[16];
	glGetFloatv(GL_PROJECTION_MATRIX, projection);

	int viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);

	float nearPos[3];
	UnProjectf(float(x), float(y), 0.0f, modelview, projection, viewport, nearPos);

	float farPos[3];
	UnProjectf(float(x), float(y), 1.0f, modelview, projection, viewport, farPos);

	origin = Vec3(float(nearPos[0]), float(nearPos[1]), float(nearPos[2]));
	dir = Normalize(Vec3(float(farPos[0]-nearPos[0]), float(farPos[1]-nearPos[1]), float(farPos[2]-nearPos[2])));
}

Vec3 GetScreenCoord(Vec3& pos) {
	float modelview[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview);

	float projection[16];
	glGetFloatv(GL_PROJECTION_MATRIX, projection);

	int viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);

	float screen[3];
	Projectf(pos.x, pos.y, pos.z, modelview, projection, viewport, screen);

	return Vec3((float)screen[0], (float)screen[1], (float)screen[2]);
}

void ReadFrame(int* backbuffer, int width, int height)
{
	glVerify(glReadBuffer(GL_BACK));
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, backbuffer);
}

void PresentFrame(bool fullsync)
{
#ifndef ANDROID
	SDL_GL_SetSwapInterval(fullsync);
	glFinish();
	SDL_GL_SwapWindow(g_window);
#endif
}

void PresentFrameHeadless()
{
#ifdef __linux__
	eglSwapBuffers(g_eglDisplay, g_eglSurface);
#endif
}


GLuint LoadTexture(const char* filename)
{
	PngImage img;
	if (PngLoad(filename, img))
	{
		GLuint tex;

		glVerify(glGenTextures(1, &tex));
		glVerify(glActiveTexture(GL_TEXTURE0));
		glVerify(glBindTexture(GL_TEXTURE_2D, tex));

		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)); 
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE));
		glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.m_width, img.m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.m_data));

		PngFree(img);

		return tex;
	}
	else
	{
		return NULL;
	}
}

GLuint LoadProbe(const char* filename)
{
	HdrImage img;
	if (HdrLoad(filename, img))
	{
		GLuint tex;

		glVerify(glGenTextures(1, &tex));
		glVerify(glActiveTexture(GL_TEXTURE0));
		glVerify(glBindTexture(GL_TEXTURE_2D, tex));

		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)); 
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE));
		glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, img.m_width, img.m_height, 0, GL_RGBA, GL_FLOAT, img.m_data));

		HdrFree(img);

		return tex;
	}
	else
	{
		return NULL;
	}
}

GLuint FindTexture(const char* filename)
{
	auto iter = g_textures.find(filename);
	if (iter == g_textures.end())
	{
		GLuint tex = LoadTexture(filename);
		g_textures[filename] =  tex;

		return tex;
	}
	else
	{
		return iter->second;
	}
}

struct RenderTexture
{
	GLuint colorTex;
	GLuint colorFrameBuffer;

	GLuint depthTex;
	GLuint depthFrameBuffer;

	RenderTexture()
	{
		memset(this, 0, sizeof(*this));
	}
};

RenderTexture* CreateRenderTexture(const char* filename)
{
	GLuint tex = FindTexture(filename);

	if (tex)
	{
		RenderTexture* t = new RenderTexture();
		t->colorTex = tex;

		return t;
	}
	else
	{
		return NULL;
	}
}

RenderTexture* CreateRenderTarget(int width, int height, bool depth)
{
	RenderTexture* t = new RenderTexture();

	// color rgba16f
	glVerify(glGenTextures(1, &t->colorTex));
	glVerify(glBindTexture(GL_TEXTURE_2D, t->colorTex));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); 	 
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL));

	glVerify(glGenFramebuffers(1, &t->colorFrameBuffer));
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, t->colorFrameBuffer));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->colorTex, 0));

	// depth buffer
	glVerify(glGenRenderbuffers(1, &t->depthFrameBuffer));
	glVerify(glBindRenderbuffer(GL_RENDERBUFFER, t->depthFrameBuffer));
	glVerify(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height));
	glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, t->depthFrameBuffer));

	// check success
	glCheckFramebufferStatus(GL_FRAMEBUFFER);
	
	// reset read/write buffers
	glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));
	glVerify(glReadBuffer(GL_COLOR_ATTACHMENT0));
		
	// reset to default buffer
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));

	return t;
}

void SetRenderTarget(const RenderTexture* target, int x, int y, int width, int height)
{
	if (target)
	{
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, target->colorFrameBuffer));
		glVerify(glViewport(x, y, width, height));

		glVerify(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
	}
	else
	{
		glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));
		glVerify(glReadBuffer(GL_COLOR_ATTACHMENT0));

		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));
		glVerify(glViewport(0, 0, g_screenWidth, g_screenHeight));
	}
}

void ReadRenderTarget(const RenderTexture* target, float* rgbd, int x, int y, int width, int height)
{
	if (target)
	{
		// bind target
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, target->colorFrameBuffer));
		glVerify(glReadBuffer(GL_COLOR_ATTACHMENT0));

		// read back, todo: slow, use PBO?
		glVerify(glReadPixels(x, y, width, height, GL_RGBA, GL_FLOAT, rgbd));

		// reset read buffer to back buffer
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));
		glVerify(glViewport(0, 0, g_screenWidth, g_screenHeight));
	}
}


void DestroyRenderTexture(RenderTexture* t)
{
	if (t)
	{
		if (t->colorTex)
			glVerify(glDeleteTextures(1, &t->colorTex));

		if (t->colorFrameBuffer)
			glVerify(glDeleteFramebuffers(1, &t->colorFrameBuffer));

		if (t->depthTex)
			glVerify(glDeleteTextures(1, &t->depthTex));

		if (t->depthFrameBuffer)
			glVerify(glDeleteRenderbuffers(1, &t->depthFrameBuffer));

		delete t;
	}
}


void SetDepthRenderProfile(DepthRenderProfile profile) {
	currentDepthProfile.minRange = profile.minRange;
	currentDepthProfile.maxRange = profile.maxRange;
}

// fixes some banding artifacts with repeated blending during thickness and diffuse rendering
#define USE_HDR_DIFFUSE_BLEND 0

// vertex shader
const char *vertexPointShader = "#version 130\n" STRINGIFY(

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

uniform mat4 lightTransform; 
uniform vec3 lightDir;
uniform vec3 lightDirView;

uniform vec4 colors[8];

uniform vec4 transmission;
uniform int mode;

//in int density;
in float density;
in int phase;
in vec4 velocity;

void main()
{
    // calculate window-space point size
	vec4 viewPos = gl_ModelViewMatrix*vec4(gl_Vertex.xyz, 1.0);

	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
	gl_PointSize = -pointScale * (pointRadius / viewPos.z);

	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_TexCoord[1] = lightTransform*vec4(gl_Vertex.xyz-lightDir*pointRadius*2.0, 1.0);
	gl_TexCoord[2] = gl_ModelViewMatrix*vec4(lightDir, 0.0);

	if (mode == 1)
	{
		// density visualization
		if (density < 0.0f)
			gl_TexCoord[3].xyz = mix(vec3(0.1, 0.1, 1.0), vec3(0.1, 1.0, 1.0), -density);
		else
			gl_TexCoord[3].xyz = mix(vec3(1.0, 1.0, 1.0), vec3(0.1, 0.2, 1.0), density);
	}
	else if (mode == 2)
	{
		gl_PointSize *= clamp(gl_Vertex.w*0.25, 0.0f, 1.0);

		gl_TexCoord[3].xyzw = vec4(clamp(gl_Vertex.w*0.05, 0.0f, 1.0));
	}
	else
	{
		gl_TexCoord[3].xyz = mix(colors[phase % 8].xyz*2.0, vec3(1.0), 0.1);
	}

	gl_TexCoord[4].xyz = gl_Vertex.xyz;
	gl_TexCoord[5].xyz = viewPos.xyz;
}
);

// pixel shader for rendering points as shaded spheres
const char *fragmentPointShader = STRINGIFY(

uniform vec3 lightDir;
uniform vec3 lightPos;
uniform float spotMin;
uniform float spotMax;
uniform int mode;

uniform sampler2DShadow shadowTex;
uniform vec2 shadowTaps[12];
uniform float pointRadius;  // point size in world space

// sample shadow map
float shadowSample()
{
	vec3 pos = vec3(gl_TexCoord[1].xyz/gl_TexCoord[1].w);
	vec3 uvw = (pos.xyz*0.5)+vec3(0.5);

	// user clip
	if (uvw.x  < 0.0 || uvw.x > 1.0)
		return 1.0;
	if (uvw.y < 0.0 || uvw.y > 1.0)
		return 1.0;
	
	float s = 0.0;
	float radius = 0.002;

	for (int i=0; i < 8; i++)
	{
		s += shadow2D(shadowTex, vec3(uvw.xy + shadowTaps[i]*radius, uvw.z)).r;
	}

	s /= 8.0;
	return s;
}

float sqr(float x) { return x*x; }

uniform float depthMinRange;
uniform float depthMaxRange;

float processDepthValue(float depth) {
	depth = max(depth, depthMinRange);
	if (depthMaxRange > 0.f)
		depth = min(depth, depthMaxRange);
	return depth;
}

void main()
{
    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
   	normal.z = sqrt(1.0-mag);

	if (mode == 2)
	{
		float alpha  = normal.z*gl_TexCoord[3].w;
		gl_FragColor.xyz = gl_TexCoord[3].xyz*alpha;
		gl_FragColor.w = alpha;
		return;
	}

    // calculate lighting
	float shadow = shadowSample();
	
	vec3 lVec = normalize(gl_TexCoord[4].xyz-(lightPos));
	vec3 lPos = vec3(gl_TexCoord[1].xyz/gl_TexCoord[1].w);
	float attenuation = max(smoothstep(spotMax, spotMin, dot(lPos.xy, lPos.xy)), 0.05);

	vec3 diffuse = vec3(0.9, 0.9, 0.9);
	vec3 reflectance =  gl_TexCoord[3].xyz;
	
	vec3 Lo = diffuse*reflectance*max(0.0, sqr(-dot(gl_TexCoord[2].xyz, normal)*0.5 + 0.5))*max(0.2,shadow)*attenuation;

	gl_FragColor = vec4(pow(Lo, vec3(1.0/2.2)), 1.0);

	vec3 eyePos = gl_TexCoord[5].xyz + normal*pointRadius;//*2.0;
	vec4 ndcPos = gl_ProjectionMatrix * vec4(eyePos, 1.0);
	ndcPos.z /= ndcPos.w;
	gl_FragDepth = ndcPos.z*0.5 + 0.5;

	gl_FragColor.w = processDepthValue(ndcPos.w);
}
);

// vertex shader
const char *vertexShader = "#version 130\n" STRINGIFY(

uniform mat4 lightTransform; 
uniform vec3 lightDir;
uniform float bias;
uniform vec4 clipPlane;
uniform float expand;

uniform mat4 objectTransform;

void main()
{
	vec3 n = normalize((objectTransform*vec4(gl_Normal, 0.0)).xyz);
	vec3 p = (objectTransform*vec4(gl_Vertex.xyz, 1.0)).xyz;

    // calculate window-space point size
	gl_Position = gl_ModelViewProjectionMatrix * vec4(p + expand*n, 1.0);

	gl_TexCoord[0].xyz = n;
	gl_TexCoord[1] = lightTransform*vec4(p + n*bias, 1.0);
	gl_TexCoord[2] = gl_ModelViewMatrix*vec4(lightDir, 0.0);
	gl_TexCoord[3].xyz = p;
	gl_TexCoord[4] = gl_Color;
	gl_TexCoord[5] = gl_MultiTexCoord0;
	gl_TexCoord[6] = gl_SecondaryColor;
	gl_TexCoord[7] = gl_ModelViewMatrix*vec4(p, 1.0);

	gl_ClipDistance[0] = dot(clipPlane,vec4(p, 1.0));
}
);

const char *passThroughShader = STRINGIFY(

void main()
{
	gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);

}
);

// pixel shader for rendering points as shaded spheres
const char *fragmentShader = STRINGIFY(

uniform vec3 lightDir;
uniform vec3 lightPos;
uniform float spotMin;
uniform float spotMax;
uniform vec3 color;
uniform vec4 fogColor;

uniform sampler2DShadow shadowTex;
uniform vec2 shadowTaps[12];

uniform sampler2D tex;
uniform sampler2D probe;

uniform float gridScale;

uniform bool texture;

uniform float specularPower;
uniform vec3 specularColor;
uniform float metallic;
uniform float roughness;

float sqr(float x) { return x*x; }

vec2 sampleProbe(vec3 dir)
{
	float u = 0.5*(1.0 + atan2(dir.x,-dir.z) / 3.1415926);
	float v = acos(-dir.y) / 3.1415926;

	return vec2(u, 1.0-v);	
}


// sample shadow map
float shadowSample()
{
	vec3 pos = vec3(gl_TexCoord[1].xyz/gl_TexCoord[1].w);
	vec3 uvw = (pos.xyz*0.5)+vec3(0.5);

	// user clip
	if (uvw.x  < 0.0 || uvw.x > 1.0)
		return 1.0;
	if (uvw.y < 0.0 || uvw.y > 1.0)
		return 1.0;
	
	float s = 0.0;
	float radius = 0.001;

	const int numTaps = 12;

	for (int i=0; i < numTaps; i++)
	{
		s += shadow2D(shadowTex, vec3(uvw.xy + shadowTaps[i]*radius, uvw.z)).r;
	}

	s /= numTaps;
	return s;
}

float filterwidth(vec2 v)
{
  vec2 fw = max(abs(dFdx(v)), abs(dFdy(v)));
  return max(fw.x, fw.y);
}

vec2 bump(vec2 x) 
{
	return (floor((x)/2) + 2.f * max(((x)/2) - floor((x)/2) - .5f, 0.f)); 
}

float checker(vec2 uv)
{
  float width = filterwidth(uv);
  vec2 p0 = uv - 0.5 * width;
  vec2 p1 = uv + 0.5 * width;
  
  vec2 i = (bump(p1) - bump(p0)) / width;
  return i.x * i.y + (1 - i.x) * (1 - i.y);
}

float fresnel(float u)
{
    float m = clamp(1-u, 0.0f, 1.0f);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

uniform float depthMinRange;
uniform float depthMaxRange;

float processDepthValue(float depth) {
	depth = max(depth, depthMinRange);
	if (depthMaxRange > 0.f)
		depth = min(depth, depthMaxRange);
	return depth;
}

void main()
{
	float lIntensity = 1.5f;

    // calculate lighting
	float shadowRaw = shadowSample();
	float shadow = max(shadowRaw, 0.5);

	vec3 lVec = lightDir;//normalize(gl_TexCoord[3].xyz-(lightPos));
	vec3 lPos = vec3(gl_TexCoord[1].xyz/gl_TexCoord[1].w);
	float attenuation = 1.0f;//max(smoothstep(spotMax, spotMin, dot(lPos.xy, lPos.xy)), 0.05);
		
	vec3 n = gl_TexCoord[0].xyz;
	vec3 color = gl_TexCoord[4].xyz;

	if (!gl_FrontFacing)
	{
		color = gl_TexCoord[6].xyz;
		n *= -1.0f;
	}

	if (gridScale > 0.0)
	{
		color = mix(gl_TexCoord[4].xyz, gl_TexCoord[6].xyz, checker(gl_TexCoord[5].xy*gridScale));
	}
	else if (texture)
	{
		vec2 uv = gl_TexCoord[5].xy;
		uv.y = 1.0f-uv.y;

		color = pow(texture2D(tex, uv).xyz, vec3(2.2));
	}
	

	float nl = dot(-lVec, n);

	// direct light term
	float wrap = 0.0;
	vec3 diffuse = color*vec3(1.0, 1.0, 1.0)*max(0.0, (nl+wrap)/(1.0+wrap)*shadow)*attenuation*lIntensity;
	
	// wrap ambient term aligned with light dir
	vec3 light = vec3(0.03, 0.025, 0.025)*1.5;
	vec3 dark = vec3(0.025, 0.025, 0.03);
	vec3 ambient = 4.0*color*mix(dark, light, nl*0.5 + 0.5)*attenuation;

	diffuse += ambient;

	// specular
	vec3 eye = normalize(gl_ModelViewMatrixInverse[3].xyz-gl_TexCoord[3].xyz);
	vec3 h = normalize(eye - lVec);

	float fh = fresnel(dot(-lVec, h));
	vec3 schlick = mix(specularColor, vec3(1.0), fh);

	// attenuate specular power on distant fragments to reduce geometric aliasing
	float spower = mix(0.0f, specularPower, exp(gl_TexCoord[7].z*0.04));

	// IBL
	vec3 env = 0.0f;//2.0f*texture2DLod(probe, sampleProbe(reflect(-eye, n)), roughness*10.0f).xyz;
	vec3 point = pow(max(0.0f, dot(h, n)), spower)*(spower + 2.0f)/8.0*max(nl, 0.0)*4.0f*lIntensity;

	vec3 specular = schlick*(point + env)*shadow;
	
	// combine and fog
	vec3 fog = mix(vec3(fogColor), diffuse*(1.0f-metallic) + specular, exp(gl_TexCoord[7].z*fogColor.w));

	// depth component in alpha (for sensor rendering), scale by a constant amount
	float depth = -gl_TexCoord[7].z;

	depth = processDepthValue(depth);
	gl_FragColor = vec4(pow(fog, vec3(1.0/2.2)), depth);
}
);

void ShadowApply(GLint sprogram, Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, GLuint shadowTex)
{
	GLint uLightTransform = glGetUniformLocation(sprogram, "lightTransform");
	glUniformMatrix4fv(uLightTransform, 1, false, lightTransform);

	GLint uLightPos = glGetUniformLocation(sprogram, "lightPos");
	glUniform3fv(uLightPos, 1, lightPos);
	
	GLint uLightDir = glGetUniformLocation(sprogram, "lightDir");
	glUniform3fv(uLightDir, 1, Normalize(lightTarget-lightPos));

	GLint uBias = glGetUniformLocation(sprogram, "bias");
	glUniform1f(uBias, g_shadowBias);

	const Vec2 taps[] = 
	{ 
		Vec2(-0.326212f,-0.40581f),Vec2(-0.840144f,-0.07358f),
		Vec2(-0.695914f,0.457137f),Vec2(-0.203345f,0.620716f),
		Vec2(0.96234f,-0.194983f),Vec2(0.473434f,-0.480026f),
		Vec2(0.519456f,0.767022f),Vec2(0.185461f,-0.893124f),
		Vec2(0.507431f,0.064425f),Vec2(0.89642f,0.412458f),
		Vec2(-0.32194f,-0.932615f),Vec2(-0.791559f,-0.59771f) 
	};
	
	GLint uShadowTaps = glGetUniformLocation(sprogram, "shadowTaps");
	glUniform2fv(uShadowTaps, 12, &taps[0].x);
	
	glEnable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, shadowTex);

}

void DrawPoints(FluidRenderBuffers* buffersIn, int n, int offset, float radius, float screenWidth, float screenAspect, float fov, Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, ShadowMap* shadowMap, bool showDensity)
{
	FluidRenderBuffersGL* buffers = reinterpret_cast<FluidRenderBuffersGL*>(buffersIn);
	GLuint positions = buffers->mPositionVBO;
	GLuint colors = buffers->mDensityVBO;
	GLuint indices = buffers->mIndices;

	if (s_pointsProgram == GLuint(-1))
		s_pointsProgram = CompileProgram(vertexPointShader, fragmentPointShader);

	if (s_pointsProgram)
	{
		glEnable(GL_POINT_SPRITE);
		glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		//glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
	
		int mode = 0;
		if (showDensity)
			mode = 1;
		if (shadowMap == NULL)
			mode = 2;

		glVerify(glUseProgram(s_pointsProgram));
		glVerify(glUniform1f( glGetUniformLocation(s_pointsProgram, "pointRadius"), radius));
		glVerify(glUniform1f( glGetUniformLocation(s_pointsProgram, "pointScale"), screenWidth/screenAspect * (1.0f / (tanf(fov*0.5f)))));
		glVerify(glUniform1f( glGetUniformLocation(s_pointsProgram, "spotMin"), g_spotMin));
		glVerify(glUniform1f( glGetUniformLocation(s_pointsProgram, "spotMax"), g_spotMax));
		glVerify(glUniform1i( glGetUniformLocation(s_pointsProgram, "mode"), mode));
		glVerify(glUniform4fv( glGetUniformLocation(s_pointsProgram, "colors"), 8, (float*)&g_colors[0].r));

		// set depth parameters
		glVerify(glUniform1f( glGetUniformLocation(s_pointsProgram, "depthMinRange"), currentDepthProfile.minRange));
		glVerify(glUniform1f( glGetUniformLocation(s_pointsProgram, "depthMaxRange"), currentDepthProfile.maxRange));

		// set shadow parameters
		ShadowApply(s_pointsProgram, lightPos, lightTarget, lightTransform, shadowMap->texture);

		glEnableClientState(GL_VERTEX_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, positions);
		glVertexPointer(4, GL_FLOAT, 0, 0);

		int d = glGetAttribLocation(s_pointsProgram, "density");
		int p = glGetAttribLocation(s_pointsProgram, "phase");

		if (d != -1)
		{
			glVerify(glEnableVertexAttribArray(d));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, colors));
			glVerify(glVertexAttribPointer(d, 1,  GL_FLOAT, GL_FALSE, 0, 0));	// densities
		}

		if (p != -1)
		{
			glVerify(glEnableVertexAttribArray(p));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, colors));
			glVerify(glVertexAttribIPointer(p, 1,  GL_INT, 0, 0));			// phases
		}

		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices));

		glVerify(glDrawElements(GL_POINTS, n, GL_UNSIGNED_INT, (const void*)(offset*sizeof(int))));

		glVerify(glUseProgram(0));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
		glVerify(glDisableClientState(GL_VERTEX_ARRAY));	
		
		if (d != -1)
			glVerify(glDisableVertexAttribArray(d));
		if (p != -1)
			glVerify(glDisableVertexAttribArray(p));
		
		glDisable(GL_POINT_SPRITE);
		glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);		
	}
}
void DrawPlane(const Vec4& p);


ShadowMap* ShadowCreate()
{
	GLuint texture;
	GLuint framebuffer;

	glVerify(glGenFramebuffers(1, &framebuffer));
	glVerify(glGenTextures(1, &texture));
	glVerify(glBindTexture(GL_TEXTURE_2D, texture));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

	// This is to allow usage of shadow2DProj function in the shader 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY));

	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, kShadowResolution, kShadowResolution, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL));

	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0));

	ShadowMap* map = new ShadowMap();
	map->texture = texture;
	map->framebuffer = framebuffer;

	return map;

}

void ShadowDestroy(ShadowMap* map)
{
	glVerify(glDeleteTextures(1, &map->texture));
	glVerify(glDeleteFramebuffers(1, &map->framebuffer));

	delete map;
}

void ShadowBegin(ShadowMap* map)
{
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(8.f, 8.f);

	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, map->framebuffer));

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, kShadowResolution, kShadowResolution);

	// draw back faces (for teapot)
	glDisable(GL_CULL_FACE);

	// bind shadow shader
	if (s_shadowProgram == GLuint(-1))
		s_shadowProgram = CompileProgram(vertexShader, passThroughShader);

	glUseProgram(s_shadowProgram);
	glVerify(glUniformMatrix4fv(glGetUniformLocation(s_shadowProgram, "objectTransform"), 1, false, Matrix44::kIdentity));
}

void ShadowEnd()
{
	glDisable(GL_POLYGON_OFFSET_FILL);

	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));

	glEnable(GL_CULL_FACE);
	glUseProgram(0);
}

GLuint s_probeTex = -1;
void SetShadowBias(float bias)
{
	g_shadowBias = bias;
}

void BindSolidShader(Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, ShadowMap* shadowMap, float bias, Vec4 fogColor)
{
	glVerify(glViewport(0, 0, g_screenWidth, g_screenHeight));

//	if (s_probeTex == GLuint(-1))
//		s_probeTex = LoadProbe("../../data/scales/env.hdr");

	if (s_lightingProgram == GLuint(-1))
		s_lightingProgram = CompileProgram(vertexShader, fragmentShader);

	if (s_lightingProgram)
	{
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);		

		glVerify(glUseProgram(s_lightingProgram));
		glVerify(glUniform1f(glGetUniformLocation(s_lightingProgram, "gridScale"), 0.0f));
		glVerify(glUniform1f( glGetUniformLocation(s_lightingProgram, "spotMin"), g_spotMin));
		glVerify(glUniform1f( glGetUniformLocation(s_lightingProgram, "spotMax"), g_spotMax));
		glVerify(glUniform4fv( glGetUniformLocation(s_lightingProgram, "fogColor"), 1, fogColor));

		glVerify(glUniformMatrix4fv( glGetUniformLocation(s_lightingProgram, "objectTransform"), 1, false, Matrix44::kIdentity));

		// set depth parameters
		glVerify(glUniform1f(glGetUniformLocation(s_lightingProgram, "depthMinRange"), currentDepthProfile.minRange));
		glVerify(glUniform1f(glGetUniformLocation(s_lightingProgram, "depthMaxRange"), currentDepthProfile.maxRange));

		// set shadow parameters
		ShadowApply(s_lightingProgram, lightPos, lightTarget, lightTransform, shadowMap->texture);
	}
}

void UnbindSolidShader()
{
	glActiveTexture(GL_TEXTURE1);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_2D);

	glUseProgram(0);
}


void SetMaterial(const Matrix44& xform, const RenderMaterial& mat)
{
	GLint program;
	glGetIntegerv(GL_CURRENT_PROGRAM, &program);

	// manual shadow shader bindings
	if (s_shadowProgram != GLuint(-1) && program == GLuint(s_shadowProgram))
	{
		static GLuint uObjectTransform = glGetUniformLocation(program, "objectTransform");
		glUniformMatrix4fv( uObjectTransform, 1, false, xform);
	}

	// manual lighting shader bindings
	if (s_lightingProgram != GLuint(-1) && program == GLuint(s_lightingProgram))
	{
		static GLuint uObjectTransform = glGetUniformLocation(program, "objectTransform");
		static GLuint uSpecularPower = glGetUniformLocation(program, "specularPower");
		static GLuint uSpecularColor = glGetUniformLocation(program, "specularColor");
		static GLuint uRoughness = glGetUniformLocation(program, "roughness");
		static GLuint uMetallic = glGetUniformLocation(program, "metallic");
		static GLuint uGridScale = glGetUniformLocation(program, "gridScale");
		static GLuint uTex = glGetUniformLocation(program, "tex");
		static GLuint uTexture = glGetUniformLocation(program, "texture");
		static GLuint uProbe = glGetUniformLocation(program, "probe");

		glUniformMatrix4fv( uObjectTransform, 1, false, xform);

		glVerify(glUniform1f(uSpecularPower, SpecularRoughnessToExponent(mat.roughness)));
		glVerify(glUniform3fv(uSpecularColor, 1, Lerp(Vec3(mat.specular*0.08f), mat.frontColor, mat.metallic)));
		glVerify(glUniform1f(uRoughness, mat.roughness));
		glVerify(glUniform1f(uMetallic, mat.metallic));

		glVerify(glUniform1f(uGridScale, mat.gridScale));
	
		if (uProbe && s_probeTex != GLuint(-1))
		{
			glActiveTexture(GL_TEXTURE2);
			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, s_probeTex);
			glVerify(glUniform1i(uProbe, 2)); // enable tex sampling
		}

		// set material properties
		if (mat.colorTex)
		{
			GLuint tex = mat.colorTex->colorTex;

			glActiveTexture(GL_TEXTURE1);
			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, tex);

			glVerify(glUniform1i(uTex, 1));		// use slot one
			glVerify(glUniform1i(uTexture, 1)); // enable tex sampling
		}
		else
		{
			glVerify(glUniform1i(uTex, 1));		// use slot one
			glVerify(glUniform1i(uTexture, 0)); // disable tex sampling}
		}

		glVerify(glColor3fv(mat.frontColor));
		glVerify(glSecondaryColor3fv(mat.backColor));
	}
}


void DrawPlanes(Vec4* planes, int n, float bias)
{
	RenderMaterial gridMaterial;
	gridMaterial.frontColor = Vec3(0.6f, 0.6f, 0.6f);
	gridMaterial.backColor = Vec3(0.5f, 0.5f, 0.5f);
	gridMaterial.gridScale = 1.0f;
	gridMaterial.specular = 0.0f;

	SetMaterial(Matrix44::kIdentity, gridMaterial);

	for (int i=0; i < n; ++i)
	{
		Vec4 p = planes[i];
		p.w -= bias;

		DrawPlane(p, false);
	}
}

void DrawMesh(const Mesh* m, const RenderMaterial& mat)
{
	if (m)
	{
		glDisable(GL_CULL_FACE);

		SetMaterial(Matrix44::kIdentity, mat);

		glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

		glVerify(glEnableClientState(GL_NORMAL_ARRAY));
		glVerify(glEnableClientState(GL_VERTEX_ARRAY));

		glVerify(glNormalPointer(GL_FLOAT, sizeof(float) * 3, &m->m_normals[0]));
		glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, &m->m_positions[0]));

		if (m->m_colours.size())
		{
			glVerify(glEnableClientState(GL_COLOR_ARRAY));
			glVerify(glColorPointer(4, GL_FLOAT, 0, &m->m_colours[0]));
		}

		glVerify(glDrawElements(GL_TRIANGLES, m->GetNumFaces() * 3, GL_UNSIGNED_INT, &m->m_indices[0]));

		glVerify(glDisableClientState(GL_VERTEX_ARRAY));
		glVerify(glDisableClientState(GL_NORMAL_ARRAY));

		if (m->m_colours.size())
			glVerify(glDisableClientState(GL_COLOR_ARRAY));
	}
}


void DrawPlane(const Vec4& p, bool color)
{
	Vec3 u, v;
	BasisFromVector(Vec3(p.x, p.y, p.z), &u, &v);

	Vec3 c = Vec3(p.x, p.y, p.z)*-p.w;

	glBegin(GL_QUADS);

	if (color)
		glColor3fv(p*0.5f + Vec4(0.5f, 0.5f, 0.5f, 0.5f));

	float kSize = 200.0f;

	// draw a grid of quads, otherwise z precision suffers
	for (int x = -3; x <= 3; ++x)
	{
		for (int y = -3; y <= 3; ++y)
		{
			Vec3 coff = c + u*float(x)*kSize*2.0f + v*float(y)*kSize*2.0f;

			glTexCoord2f(kSize*2.0f, kSize*2.0f);
			glNormal3f(p.x, p.y, p.z);
			glVertex3fv(coff + u*kSize + v*kSize);

			glTexCoord2f(0.0f, kSize*2.0f);
			glNormal3f(p.x, p.y, p.z);
			glVertex3fv(coff - u*kSize + v*kSize);

			glTexCoord2f(0.0f, 0.0f);
			glNormal3f(p.x, p.y, p.z);
			glVertex3fv(coff - u*kSize - v*kSize);

			glTexCoord2f(kSize*2.0f, 0.0f);
			glNormal3f(p.x, p.y, p.z);
			glVertex3fv(coff + u*kSize - v*kSize);
		}
	}

	glEnd();
}

void DrawCloth(const Vec4* positions, const Vec4* normals, const float* uvs, const int* indices, int numTris, int numPositions, const RenderMaterial& mat, float expand, bool twosided, bool smooth)
{ 
	if (!numTris)
		return;

	if (twosided)
		glDisable(GL_CULL_FACE);

#if 1
	GLint program;
	glGetIntegerv(GL_CURRENT_PROGRAM, &program);

	if (program == GLint(s_lightingProgram))
	{
		GLint uBias = glGetUniformLocation(s_lightingProgram, "bias");
		glUniform1f(uBias, 0.0f);

		GLint uExpand = glGetUniformLocation(s_lightingProgram, "expand");
		glUniform1f(uExpand, expand);
	}
#endif

	SetMaterial(Matrix44::kIdentity, mat);

	glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
	glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

	glVerify(glEnableClientState(GL_VERTEX_ARRAY));
	glVerify(glEnableClientState(GL_NORMAL_ARRAY));

	glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float)*4, positions));
	glVerify(glNormalPointer(GL_FLOAT, sizeof(float)*4, normals));
	
	glVerify(glDrawElements(GL_TRIANGLES, numTris*3, GL_UNSIGNED_INT, indices));

	glVerify(glDisableClientState(GL_VERTEX_ARRAY));
	glVerify(glDisableClientState(GL_NORMAL_ARRAY));

	if (twosided)
		glEnable(GL_CULL_FACE);

#if 1
	if (program == GLint(s_lightingProgram))
	{
		GLint uBias = glGetUniformLocation(s_lightingProgram, "bias");
		glUniform1f(uBias, g_shadowBias);

		GLint uExpand = glGetUniformLocation(s_lightingProgram, "expand");
		glUniform1f(uExpand, 0.0f);
	}
#endif
}

void DrawRope(Vec4* positions, int* indices, int numIndices, float radius, const RenderMaterial& mat)
{
	if (numIndices < 2)
		return;

	std::vector<Vec3> vertices;
	std::vector<Vec3> normals;
	std::vector<int> triangles;
	
	// flatten curve
	std::vector<Vec3> curve(numIndices);
	for (int i=0; i < numIndices; ++i)
		curve[i] = Vec3(positions[indices[i]]);

	const int resolution = 8;
	const int smoothing = 3;

	vertices.reserve(resolution*numIndices*smoothing);
	normals.reserve(resolution*numIndices*smoothing);
	triangles.reserve(numIndices*resolution*6*smoothing);

	Extrude(&curve[0], int(curve.size()), vertices, normals, triangles, radius, resolution, smoothing);
		
	glVerify(glDisable(GL_CULL_FACE));

	SetMaterial(Matrix44::kIdentity, mat);

	glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

	glVerify(glEnableClientState(GL_VERTEX_ARRAY));
	glVerify(glEnableClientState(GL_NORMAL_ARRAY));

	glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float)*3, &vertices[0]));
	glVerify(glNormalPointer(GL_FLOAT, sizeof(float)*3, &normals[0]));

	glVerify(glDrawElements(GL_TRIANGLES, GLsizei(triangles.size()), GL_UNSIGNED_INT, &triangles[0]));

	glVerify(glDisableClientState(GL_VERTEX_ARRAY));
	glVerify(glDisableClientState(GL_NORMAL_ARRAY));
	glVerify(glEnable(GL_CULL_FACE));

}


struct ReflectMap
{
	GLuint texture;

	int width;
	int height;
};

ReflectMap* ReflectCreate(int width, int height)
{
	GLuint texture;

	// copy frame buffer to texture
	glVerify(glActiveTexture(GL_TEXTURE0));
	glVerify(glEnable(GL_TEXTURE_2D));

	glVerify(glGenTextures(1, &texture));
	glVerify(glBindTexture(GL_TEXTURE_2D, texture));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); 	 
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
		
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));

	ReflectMap* map = new ReflectMap();
	map->texture = texture;
	map->width = width;
	map->height = height;

	return map;
}

void ReflectDestroy(ReflectMap* map)
{
	glVerify(glDeleteTextures(1, &map->texture));

	delete map;
}

void ReflectBegin(ReflectMap* map, Vec4 plane, int width, int height)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, width, height);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	Matrix44 scale = Matrix44::kIdentity;
	scale.columns[0][0] *= -2.0f;
	scale.columns[1][1] *= -2.0f;
	scale.columns[2][2] *= -2.0f;
	scale.columns[3][3] *= -2.0f;

	Matrix44 reflect = (scale*Outer(Vec4(plane.x, plane.y, plane.z, 0.0f), plane));
	reflect.columns[0][0] += 1.0f;
	reflect.columns[1][1] += 1.0f;
	reflect.columns[2][2] += 1.0f;
	reflect.columns[3][3] += 1.0f;

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(reflect);

	glVerify(glFrontFace(GL_CW));
	glVerify(glEnable(GL_CLIP_PLANE0));

	glVerify(glUniform4fv( glGetUniformLocation(s_lightingProgram, "clipPlane"), 1, plane));
}

void ReflectEnd(ReflectMap* map, int width, int height)
{
	// copy frame buffer to texture
	glVerify(glActiveTexture(GL_TEXTURE0));
	glVerify(glEnable(GL_TEXTURE_2D));
	glVerify(glBindTexture(GL_TEXTURE_2D, map->texture));

	glVerify(glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height));
	
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glVerify(glDisable(GL_CLIP_PLANE0));
	glVerify(glFrontFace(GL_CCW));

	glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo);

	glViewport(0, 0, g_screenWidth, g_screenHeight);
}


//-----------------------------------------------------------------------------------------------------
// vertex shader

const char *vertexPointDepthShader = STRINGIFY(

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

void main()
{
    // calculate window-space point size
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
	gl_PointSize = pointScale * (pointRadius / gl_Position.w);

	gl_TexCoord[0] = gl_MultiTexCoord0;    
	gl_TexCoord[1] = gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0);
}
);

// pixel shader for rendering points as shaded spheres
const char *fragmentPointDepthShader = STRINGIFY(

uniform float pointRadius;  // point size in world space

void main()
{
    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
   	normal.z = sqrt(1.0-mag);

	vec3 eyePos = gl_TexCoord[1].xyz + normal*pointRadius*2.0;
	vec4 ndcPos = gl_ProjectionMatrix * vec4(eyePos, 1.0);
	ndcPos.z /= ndcPos.w;

	gl_FragColor = vec4(eyePos.z, 1.0, 1.0, 1.0);
	gl_FragDepth = ndcPos.z*0.5 + 0.5;
}
);


// pixel shader for rendering points density
const char *fragmentPointThicknessShader = STRINGIFY(

void main()
{
    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
   	normal.z = sqrt(1.0-mag);

	gl_FragColor = vec4(normal.z*0.005);	
}
);

//--------------------------------------------------------
// Ellipsoid shaders
//
const char *vertexEllipsoidDepthShader = "#version 120\n" STRINGIFY(

// rotation matrix in xyz, scale in w
attribute vec4 q1;
attribute vec4 q2;
attribute vec4 q3;

// returns 1.0 for x==0.0 (unlike glsl)
float Sign(float x) { return x < 0.0 ? -1.0: 1.0; }

bool solveQuadratic(float a, float b, float c, out float minT, out float maxT)
{
	if (a == 0.0 && b == 0.0)
	{
		minT = maxT = 0.0;
		return false;
	}

	float discriminant = b*b - 4.0*a*c;

	if (discriminant < 0.0)
	{
		return false;
	}

	float t = -0.5*(b + Sign(b)*sqrt(discriminant));
	minT = t / a;
	maxT = c / t;

	if (minT > maxT)
	{
		float tmp = minT;
		minT = maxT;
		maxT = tmp;
	}

	return true;
}

float DotInvW(vec4 a, vec4 b) {	return a.x*b.x + a.y*b.y + a.z*b.z - a.w*b.w; }

void main()
{	
	vec3 worldPos = gl_Vertex.xyz;// - vec3(0.0, 0.1*0.25, 0.0);	// hack move towards ground to account for anisotropy

	// construct quadric matrix
	mat4 q;
	q[0] = vec4(q1.xyz*q1.w, 0.0);
	q[1] = vec4(q2.xyz*q2.w, 0.0);
	q[2] = vec4(q3.xyz*q3.w, 0.0);
	q[3] = vec4(worldPos, 1.0);

	// transforms a normal to parameter space (inverse transpose of (q*modelview)^-T)
	mat4 invClip = transpose(gl_ModelViewProjectionMatrix*q);

	// solve for the right hand bounds in homogenous clip space
	float a1 = DotInvW(invClip[3], invClip[3]);
	float b1 = -2.0f*DotInvW(invClip[0], invClip[3]);
	float c1 = DotInvW(invClip[0], invClip[0]);

	float xmin;
	float xmax;
 	solveQuadratic(a1, b1, c1, xmin, xmax);	

	// solve for the right hand bounds in homogenous clip space
	float a2 = DotInvW(invClip[3], invClip[3]);
	float b2 = -2.0f*DotInvW(invClip[1], invClip[3]);
	float c2 = DotInvW(invClip[1], invClip[1]); 

	float ymin;
	float ymax;
 	solveQuadratic(a2, b2, c2, ymin, ymax);

	gl_Position = vec4(worldPos.xyz, 1.0);
	gl_TexCoord[0] = vec4(xmin, xmax, ymin, ymax);

	// construct inverse quadric matrix (used for ray-casting in parameter space)
	mat4 invq;
	invq[0] = vec4(q1.xyz/q1.w, 0.0);
	invq[1] = vec4(q2.xyz/q2.w, 0.0);
	invq[2] = vec4(q3.xyz/q3.w, 0.0);
	invq[3] = vec4(0.0, 0.0, 0.0, 1.0);

	invq = transpose(invq);
	invq[3] = -(invq*gl_Position);

	// transform a point from view space to parameter space
	invq = invq*gl_ModelViewMatrixInverse;

	// pass down
	gl_TexCoord[1] = invq[0];
	gl_TexCoord[2] = invq[1];
	gl_TexCoord[3] = invq[2];
	gl_TexCoord[4] = invq[3];

	// compute ndc pos for frustrum culling in GS
	vec4 ndcPos = gl_ModelViewProjectionMatrix * vec4(worldPos.xyz, 1.0);
	gl_TexCoord[5] = ndcPos / ndcPos.w;
}
);

const char* geometryEllipsoidDepthShader = 
"#version 120\n"
"#extension GL_EXT_geometry_shader4 : enable\n"
STRINGIFY(
void main()
{
	vec3 pos = gl_PositionIn[0].xyz;
	vec4 bounds = gl_TexCoordIn[0][0];
	vec4 ndcPos = gl_TexCoordIn[0][5];

	// frustrum culling
	const float ndcBound = 1.0;
	if (ndcPos.x < -ndcBound) return;
	if (ndcPos.x > ndcBound) return;
	if (ndcPos.y < -ndcBound) return;
	if (ndcPos.y > ndcBound) return;

	float xmin = bounds.x;
	float xmax = bounds.y;
	float ymin = bounds.z;
	float ymax = bounds.w;

	// inv quadric transform
	gl_TexCoord[0] = gl_TexCoordIn[0][1];
	gl_TexCoord[1] = gl_TexCoordIn[0][2];
	gl_TexCoord[2] = gl_TexCoordIn[0][3];
	gl_TexCoord[3] = gl_TexCoordIn[0][4];

	gl_Position = vec4(xmin, ymax, 0.0, 1.0);
	EmitVertex();

	gl_Position = vec4(xmin, ymin, 0.0, 1.0);
	EmitVertex();

	gl_Position = vec4(xmax, ymax, 0.0, 1.0);
	EmitVertex();

	gl_Position = vec4(xmax, ymin, 0.0, 1.0);
	EmitVertex();
}
);

// pixel shader for rendering points as shaded spheres
const char *fragmentEllipsoidDepthShader = "#version 120\n" STRINGIFY(

uniform vec3 invViewport;
uniform vec3 invProjection;

float Sign(float x) { return x < 0.0 ? -1.0: 1.0; }

bool solveQuadratic(float a, float b, float c, out float minT, out float maxT)
{
	if (a == 0.0 && b == 0.0)
	{
		minT = maxT = 0.0;
		return true;
	}

	float discriminant = b*b - 4.0*a*c;

	if (discriminant < 0.0)
	{
		return false;
	}

	float t = -0.5*(b + Sign(b)*sqrt(discriminant));
	minT = t / a;
	maxT = c / t;

	if (minT > maxT)
	{
		float tmp = minT;
		minT = maxT;
		maxT = tmp;
	}

	return true;
}

float sqr(float x) { return x*x; }

void main()
{
	// transform from view space to parameter space
	mat4 invQuadric;
	invQuadric[0] = gl_TexCoord[0];
	invQuadric[1] = gl_TexCoord[1];
	invQuadric[2] = gl_TexCoord[2];
	invQuadric[3] = gl_TexCoord[3];

	vec4 ndcPos = vec4(gl_FragCoord.xy*invViewport.xy*vec2(2.0, 2.0) - vec2(1.0, 1.0), -1.0, 1.0);
	vec4 viewDir = gl_ProjectionMatrixInverse*ndcPos; 

	// ray to parameter space
	vec4 dir = invQuadric*vec4(viewDir.xyz, 0.0);
	vec4 origin = invQuadric[3];

	// set up quadratric equation
	float a = sqr(dir.x) + sqr(dir.y) + sqr(dir.z);// - sqr(dir.w);
	float b = dir.x*origin.x + dir.y*origin.y + dir.z*origin.z - dir.w*origin.w;
	float c = sqr(origin.x) + sqr(origin.y) + sqr(origin.z) - sqr(origin.w);

	float minT;
	float maxT;

	if (solveQuadratic(a, 2.0*b, c, minT, maxT))
	{
		vec3 eyePos = viewDir.xyz*minT;
		vec4 ndcPos = gl_ProjectionMatrix * vec4(eyePos, 1.0);
		ndcPos.z /= ndcPos.w;

		gl_FragColor = vec4(eyePos.z, 1.0, 1.0, 1.0);
		gl_FragDepth = ndcPos.z*0.5 + 0.5;

		return;
	}
	else
		discard;	

	gl_FragColor = vec4(0.5, 0.0, 0.0, 1.0);
}
);

//--------------------------------------------------------------------------------
// Composite shaders

const char* vertexPassThroughShader = STRINGIFY(

void main()
{
	gl_Position = vec4(gl_Vertex.xyz, 1.0);
	gl_TexCoord[0] = gl_MultiTexCoord0; 
}
);

const char* fragmentBlurDepthShader = 
"#extension GL_ARB_texture_rectangle : enable\n"
STRINGIFY(

uniform sampler2DRect depthTex;
uniform sampler2D thicknessTex;
uniform float blurRadiusWorld;
uniform float blurScale;
uniform float blurFalloff;
uniform vec2 invTexScale;

uniform bool debug;

float sqr(float x) { return x*x; }

void main()
{
    // eye-space depth of center sample
    float depth = texture2DRect(depthTex, gl_FragCoord.xy).x;
	float thickness = texture2D(thicknessTex, gl_TexCoord[0].xy).x;

	// hack: ENABLE_SIMPLE_FLUID
	//thickness = 0.0f;

	if (debug)
	{
		// do not blur
		gl_FragColor.x = depth;
		return;
	}

	// threshold on thickness to create nice smooth silhouettes
	if (depth == 0.0)//|| thickness < 0.02f)
	{
		gl_FragColor.x = 0.0;
		return;
	}

	/*
	float dzdx = dFdx(depth);
	float dzdy = dFdy(depth);

	// handle edge case
	if (max(abs(dzdx), abs(dzdy)) > 0.05)
	{
		dzdx = 0.0;
		dzdy = 0.0;

		gl_FragColor.x = depth;
		return;
	}
	*/

	float blurDepthFalloff = 5.5;//blurFalloff*mix(4.0, 1.0, thickness)/blurRadiusWorld*0.0375;	// these constants are just a re-scaling from some known good values

	float maxBlurRadius = 5.0;
	//float taps = min(maxBlurRadius, blurScale * (blurRadiusWorld / -depth));
	//vec2 blurRadius = min(mix(0.25, 2.0/blurFalloff, thickness) * blurScale * (blurRadiusWorld / -depth) / taps, 0.15)*invTexScale;
	
	//discontinuities between different tap counts are visible. to avoid this we 
	//use fractional contributions between #taps = ceil(radius) and floor(radius) 
	float radius = min(maxBlurRadius, blurScale * (blurRadiusWorld / -depth));
	float radiusInv = 1.0/radius;
	float taps = ceil(radius);
	float frac = taps - radius;

	float sum = 0.0;
    float wsum = 0.0;
	float count = 0.0;

    for(float y=-taps; y <= taps; y += 1.0)
	{
        for(float x=-taps; x <= taps; x += 1.0)
		{
			vec2 offset = vec2(x, y);

            float sample = texture2DRect(depthTex, gl_FragCoord.xy + offset).x;

			if (sample < -10000.0*0.5)
				continue;

            // spatial domain
            float r1 = length(vec2(x, y))*radiusInv;
			float w = exp(-(r1*r1));

			//float expectedDepth = depth + dot(vec2(dzdx, dzdy), offset);

            // range domain (based on depth difference)
            float r2 = (sample - depth) * blurDepthFalloff;
            float g = exp(-(r2*r2));

			//fractional radius contributions
			float wBoundary = step(radius, max(abs(x), abs(y)));
			float wFrac = 1.0 - wBoundary*frac;

			sum += sample * w * g * wFrac;
			wsum += w * g * wFrac;
			count += g * wFrac;
        }
    }

    if (wsum > 0.0) {
        sum /= wsum;
    }

	float blend = count/sqr(2.0*radius+1.0);
	gl_FragColor.x = mix(depth, sum, blend);
}
);

const char* fragmentCompositeShader = STRINGIFY(

uniform sampler2D tex;
uniform vec2 invTexScale;
uniform vec3 lightPos;
uniform vec3 lightDir;
uniform float spotMin;
uniform float spotMax;
uniform vec4 color;
uniform float ior;

uniform vec2 clipPosToEye;

uniform sampler2D reflectTex;
uniform sampler2DShadow shadowTex;
uniform vec2 shadowTaps[12];
uniform mat4 lightTransform;

uniform sampler2D thicknessTex;
uniform sampler2D sceneTex;

uniform bool debug;

// sample shadow map
float shadowSample(vec3 worldPos, out float attenuation)
{
	// hack: ENABLE_SIMPLE_FLUID
	//attenuation = 0.0f;
	//return 0.5;

	vec4 pos = lightTransform*vec4(worldPos+lightDir*0.15, 1.0);
	pos /= pos.w;
	vec3 uvw = (pos.xyz*0.5)+vec3(0.5);

	attenuation = max(smoothstep(spotMax, spotMin, dot(pos.xy, pos.xy)), 0.05);

	// user clip
	if (uvw.x  < 0.0 || uvw.x > 1.0)
		return 1.0;
	if (uvw.y < 0.0 || uvw.y > 1.0)
		return 1.0;
	
	float s = 0.0;
	float radius = 0.002;

	for (int i=0; i < 8; i++)
	{
		s += shadow2D(shadowTex, vec3(uvw.xy + shadowTaps[i]*radius, uvw.z)).r;
	}

	s /= 8.0;
	return s;
}

vec3 viewportToEyeSpace(vec2 coord, float eyeZ)
{
	// find position at z=1 plane
	vec2 uv = (coord*2.0 - vec2(1.0))*clipPosToEye;

	return vec3(-uv*eyeZ, eyeZ);
}

vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linearToSrgb(vec3 c) { return pow(c, vec3(1.0/2.2)); }

float sqr(float x) { return x*x; }
float cube(float x) { return x*x*x; }

uniform float depthMinRange;
uniform float depthMaxRange;

float processDepthValue(float depth) {
	depth = max(depth, depthMinRange);
	if (depthMaxRange > 0.f)
		depth = min(depth, depthMaxRange);
	return depth;
}

void main()
{
	float eyeZ = texture2D(tex, gl_TexCoord[0].xy).x;

	if (eyeZ == 0.0)
		discard;

	// reconstruct eye space pos from depth
	vec3 eyePos = viewportToEyeSpace(gl_TexCoord[0].xy, eyeZ);

	// finite difference approx for normals, can't take dFdx because
	// the one-sided difference is incorrect at shape boundaries
	vec3 zl = eyePos - viewportToEyeSpace(gl_TexCoord[0].xy - vec2(invTexScale.x, 0.0), texture2D(tex, gl_TexCoord[0].xy - vec2(invTexScale.x, 0.0)).x);
	vec3 zr = viewportToEyeSpace(gl_TexCoord[0].xy + vec2(invTexScale.x, 0.0), texture2D(tex, gl_TexCoord[0].xy + vec2(invTexScale.x, 0.0)).x) - eyePos;
	vec3 zt = viewportToEyeSpace(gl_TexCoord[0].xy + vec2(0.0, invTexScale.y), texture2D(tex, gl_TexCoord[0].xy + vec2(0.0, invTexScale.y)).x) - eyePos;
	vec3 zb = eyePos - viewportToEyeSpace(gl_TexCoord[0].xy - vec2(0.0, invTexScale.y), texture2D(tex, gl_TexCoord[0].xy - vec2(0.0, invTexScale.y)).x);
	
	vec3 dx = zl;
	vec3 dy = zt;

	if (abs(zr.z) < abs(zl.z))
		dx = zr;

	if (abs(zb.z) < abs(zt.z))
		dy = zb;

	//vec3 dx = dFdx(eyePos.xyz);
	//vec3 dy = dFdy(eyePos.xyz);

	vec4 worldPos = gl_ModelViewMatrixInverse*vec4(eyePos, 1.0);
	
	float attenuation;
	float shadow = shadowSample(worldPos.xyz, attenuation);

	vec3 l = (gl_ModelViewMatrix*vec4(lightDir, 0.0)).xyz;
	vec3 v = -normalize(eyePos);
	
	vec3 n = normalize(cross(dx, dy));
	vec3 h = normalize(v + l);

	vec3 skyColor = vec3(0.1, 0.2, 0.4)*1.2;
	vec3 groundColor = vec3(0.1, 0.1, 0.2);
	
	float fresnel = 0.1 + (1.0 - 0.1)*cube(1.0-max(dot(n, v), 0.0));

	vec3 lVec = normalize(worldPos.xyz-lightPos);

	float ln = dot(l, n)*attenuation;

	vec3 rEye = reflect(-v, n).xyz;
	vec3 rWorld = (gl_ModelViewMatrixInverse*vec4(rEye, 0.0)).xyz;

	vec2 texScale = vec2(0.75, 1.0);	// to account for backbuffer aspect ratio (todo: pass in)

	float refractScale = ior*0.025;
	float reflectScale = ior*0.1;

	// attenuate refraction near ground (hack)
	refractScale *= smoothstep(0.1, 0.4, worldPos.y);
	
	vec2 refractCoord = gl_TexCoord[0].xy + n.xy*refractScale*texScale;	
	//vec2 refractCoord = gl_TexCoord[0].xy + refract(-v, n, 1.0/1.33)*refractScale*texScale;	

	// read thickness from refracted coordinate otherwise we get halos around objectsw
	float thickness = max(texture2D(thicknessTex, refractCoord).x, 0.3);

	//vec3 transmission = exp(-(vec3(1.0)-color.xyz)*thickness);
	vec3 transmission = (1.0-(1.0-color.xyz)*thickness*0.8)*color.w; 
	vec3 refract = texture2D(sceneTex, refractCoord).xyz*transmission;
	
	vec2 sceneReflectCoord = gl_TexCoord[0].xy - rEye.xy*texScale*reflectScale/eyePos.z;
	vec3 sceneReflect = (texture2D(sceneTex, sceneReflectCoord).xyz)*shadow;

	vec3 planarReflect = texture2D(reflectTex, gl_TexCoord[0].xy).xyz;
	planarReflect = vec3(0.0);

	// fade out planar reflections above the ground
	vec3 reflect = mix(planarReflect, sceneReflect, smoothstep(0.05, 0.3, worldPos.y)) + mix(groundColor, skyColor, smoothstep(0.15, 0.25, rWorld.y)*shadow);
	
	// lighting
	vec3 diffuse = color.xyz*mix(vec3(0.29, 0.379, 0.59), vec3(1.0), (ln*0.5 + 0.5)*max(shadow, 0.4))*(1.0-color.w);
	vec3 specular = vec3(1.2*pow(max(dot(h, n), 0.0), 400.0));

	gl_FragColor.xyz = diffuse + (mix(refract, reflect, fresnel) + specular)*color.w;

	if (debug)
		gl_FragColor = vec4(n*0.5 + vec3(0.5), 1.0);

	// write valid z
	vec4 clipPos = gl_ProjectionMatrix*vec4(0.0, 0.0, eyeZ, 1.0);
	clipPos.z /= clipPos.w;

	gl_FragDepth = clipPos.z*0.5 + 0.5;

	gl_FragColor.w = processDepthValue(clipPos.w);
}
);


struct FluidRenderer
{
	GLuint mDepthFbo;
	GLuint mDepthTex;
	GLuint mDepthSmoothTex;
	GLuint mSceneFbo;
	GLuint mSceneTex;
	GLuint mReflectTex;

	GLuint mThicknessFbo;
	GLuint mThicknessTex;

	GLuint mPointThicknessProgram;
	//GLuint mPointDepthProgram;

	GLuint mEllipsoidThicknessProgram;
	GLuint mEllipsoidDepthProgram;

	GLuint mCompositeProgram;
	GLuint mDepthBlurProgram;

	int mSceneWidth;
	int mSceneHeight;
};

FluidRenderer* CreateFluidRenderer(uint32_t width, uint32_t height)
{
	FluidRenderer* renderer = new FluidRenderer();

	renderer->mSceneWidth = width;
	renderer->mSceneHeight = height;

	// scene depth texture
	glVerify(glGenTextures(1, &renderer->mDepthTex));
	glVerify(glBindTexture(GL_TEXTURE_RECTANGLE_ARB, renderer->mDepthTex));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)); 	 
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	glVerify(glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_LUMINANCE32F_ARB, width, height, 0, GL_LUMINANCE, GL_FLOAT, NULL));

	// smoothed depth texture
	glVerify(glGenTextures(1, &renderer->mDepthSmoothTex));
	glVerify(glBindTexture(GL_TEXTURE_2D, renderer->mDepthSmoothTex));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)); 	 
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, GL_LUMINANCE, GL_FLOAT, NULL));

	// scene copy
	glVerify(glGenTextures(1, &renderer->mSceneTex));
	glVerify(glBindTexture(GL_TEXTURE_2D, renderer->mSceneTex));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); 	 
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));

	glVerify(glGenFramebuffers(1, &renderer->mSceneFbo));
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, renderer->mSceneFbo));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderer->mSceneTex, 0));

	// frame buffer
	glVerify(glGenFramebuffers(1, &renderer->mDepthFbo));
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, renderer->mDepthFbo));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, renderer->mDepthTex, 0));
	
	GLuint zbuffer;
	glVerify(glGenRenderbuffers(1, &zbuffer));
	glVerify(glBindRenderbuffer(GL_RENDERBUFFER, zbuffer));
	glVerify(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height));
	glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, zbuffer));

	glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));
	glVerify(glReadBuffer(GL_COLOR_ATTACHMENT0));

	glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo);
	
	// reflect texture
	glVerify(glGenTextures(1, &renderer->mReflectTex));
	glVerify(glBindTexture(GL_TEXTURE_2D, renderer->mReflectTex));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); 	 
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));

	// thickness texture
	const int thicknessWidth = width;
	const int thicknessHeight = height;

	glVerify(glGenTextures(1, &renderer->mThicknessTex));
	glVerify(glBindTexture(GL_TEXTURE_2D, renderer->mThicknessTex));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)); 
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); 	 
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

#if USE_HDR_DIFFUSE_BLEND	
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, thicknessWidth, thicknessHeight, 0, GL_RGBA, GL_FLOAT, NULL));
#else
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, thicknessWidth, thicknessHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
#endif
		
	// thickness buffer
	glVerify(glGenFramebuffers(1, &renderer->mThicknessFbo));
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, renderer->mThicknessFbo));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderer->mThicknessTex, 0));
	
	GLuint thickz;
	glVerify(glGenRenderbuffers(1, &thickz));
	glVerify(glBindRenderbuffer(GL_RENDERBUFFER, thickz));
	glVerify(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, thicknessWidth, thicknessHeight));
	glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, thickz));
	
	glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo);

	// compile shaders
	//renderer->mPointDepthProgram = CompileProgram(vertexPointDepthShader, fragmentPointDepthShader);
	renderer->mPointThicknessProgram = CompileProgram(vertexPointDepthShader, fragmentPointThicknessShader);

	//renderer->mEllipsoidThicknessProgram = CompileProgram(vertexEllipsoidDepthShader, fragmentEllipsoidThicknessShader);
	renderer->mEllipsoidDepthProgram = CompileProgram(vertexEllipsoidDepthShader, fragmentEllipsoidDepthShader, geometryEllipsoidDepthShader);

	renderer->mCompositeProgram = CompileProgram(vertexPassThroughShader, fragmentCompositeShader);
	renderer->mDepthBlurProgram = CompileProgram(vertexPassThroughShader, fragmentBlurDepthShader);

	return renderer;
}

void DestroyFluidRenderer(FluidRenderer* renderer)
{
	glVerify(glDeleteFramebuffers(1, &renderer->mSceneFbo));
	glVerify(glDeleteFramebuffers(1, &renderer->mDepthFbo));
	glVerify(glDeleteTextures(1, &renderer->mDepthTex));
	glVerify(glDeleteTextures(1, &renderer->mDepthSmoothTex));
	glVerify(glDeleteTextures(1, &renderer->mSceneTex));

	glVerify(glDeleteFramebuffers(1, &renderer->mThicknessFbo));
	glVerify(glDeleteTextures(1, &renderer->mThicknessTex));
}

FluidRenderBuffers* CreateFluidRenderBuffers(int numFluidParticles, bool enableInterop)
{
	FluidRenderBuffersGL* buffers = new FluidRenderBuffersGL(numFluidParticles);
	
	// vbos
	glVerify(glGenBuffers(1, &buffers->mPositionVBO));
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mPositionVBO));
	glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numFluidParticles, 0, GL_DYNAMIC_DRAW));

	// density
	glVerify(glGenBuffers(1, &buffers->mDensityVBO));
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDensityVBO));
	glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(int)*numFluidParticles, 0, GL_DYNAMIC_DRAW));

	for (int i = 0; i < 3; ++i)
	{
		glVerify(glGenBuffers(1, &buffers->mAnisotropyVBO[i]));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[i]));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numFluidParticles, 0, GL_DYNAMIC_DRAW));
	}

	glVerify(glGenBuffers(1, &buffers->mIndices));
	glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers->mIndices));
	glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*numFluidParticles, 0, GL_DYNAMIC_DRAW));

	if (enableInterop)
	{
		extern NvFlexLibrary* g_flexLib;

		buffers->mPositionBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mPositionVBO, numFluidParticles, sizeof(Vec4));
		buffers->mDensitiesBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mDensityVBO, numFluidParticles, sizeof(float));
		buffers->mIndicesBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mIndices, numFluidParticles, sizeof(int));

		buffers->mAnisotropyBuf[0] = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mAnisotropyVBO[0], numFluidParticles, sizeof(Vec4));
		buffers->mAnisotropyBuf[1] = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mAnisotropyVBO[1], numFluidParticles, sizeof(Vec4));
		buffers->mAnisotropyBuf[2] = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mAnisotropyVBO[2], numFluidParticles, sizeof(Vec4));
	}

	return reinterpret_cast<FluidRenderBuffers*>(buffers);
}

void DestroyFluidRenderBuffers(FluidRenderBuffers* buffers)
{
	delete reinterpret_cast<FluidRenderBuffersGL*>(buffers);
}

void UpdateFluidRenderBuffers(FluidRenderBuffers* buffersIn, NvFlexSolver* solver, bool anisotropy, bool density)
{
	FluidRenderBuffersGL* buffers = reinterpret_cast<FluidRenderBuffersGL*>(buffersIn);
	// use VBO buffer wrappers to allow Flex to write directly to the OpenGL buffers
	// Flex will take care of any CUDA interop mapping/unmapping during the get() operations
	if (!anisotropy)
	{
		// regular particles
		NvFlexGetParticles(solver, buffers->mPositionBuf, NULL);
	}
	else
	{
		// fluid buffers
		NvFlexGetSmoothParticles(solver, buffers->mPositionBuf, NULL);
		NvFlexGetAnisotropy(solver, buffers->mAnisotropyBuf[0], buffers->mAnisotropyBuf[1], buffers->mAnisotropyBuf[2], NULL);
	}

	if (density)
	{
		NvFlexGetDensities(solver, buffers->mDensitiesBuf, NULL);
	}
	else
	{
		NvFlexGetPhases(solver, buffers->mDensitiesBuf, NULL);
	}

	NvFlexGetActive(solver, buffers->mIndicesBuf, NULL);
}

void UpdateFluidRenderBuffers(FluidRenderBuffers* buffersIn, Vec4* particles, float* densities, Vec4* anisotropy1, Vec4* anisotropy2, Vec4* anisotropy3, int numParticles, int* indices, int numIndices)
{
	FluidRenderBuffersGL* buffers = reinterpret_cast<FluidRenderBuffersGL*>(buffersIn);
	// regular particles
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mPositionVBO));
	glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles*sizeof(Vec4), particles));

	Vec4*const anisotropies[] = 
	{
		anisotropy1,
		anisotropy2, 
		anisotropy3,
	};

	for (int i = 0; i < 3; i++)
	{
		Vec4* anisotropy = anisotropies[i];
		if (anisotropy)
		{
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[i]));
			glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles * sizeof(Vec4), anisotropy));
		}
	}

	// density /phase buffer
	if (densities)
	{
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDensityVBO));
		glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles*sizeof(float), densities));
	}

	if (indices)
	{
		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers->mIndices));
		glVerify(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, numIndices*sizeof(int), indices));
	}

	// reset
	glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

DiffuseRenderBuffers* CreateDiffuseRenderBuffers(int numDiffuseParticles, bool& enableInterop)
{
	DiffuseRenderBuffersGL* buffers = new DiffuseRenderBuffersGL(numDiffuseParticles);
	
	if (numDiffuseParticles > 0)
	{
		glVerify(glGenBuffers(1, &buffers->mDiffusePositionVBO));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffusePositionVBO));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numDiffuseParticles, 0, GL_DYNAMIC_DRAW));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

		glVerify(glGenBuffers(1, &buffers->mDiffuseVelocityVBO));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffuseVelocityVBO));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numDiffuseParticles, 0, GL_DYNAMIC_DRAW));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

		if (enableInterop)
		{
			extern NvFlexLibrary* g_flexLib;

			buffers->mDiffusePositionsBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mDiffusePositionVBO, numDiffuseParticles, sizeof(Vec4));
			buffers->mDiffuseVelocitiesBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mDiffuseVelocityVBO, numDiffuseParticles, sizeof(Vec4));
		}
	}

	return reinterpret_cast<DiffuseRenderBuffers*>(buffers);
}

void DestroyDiffuseRenderBuffers(DiffuseRenderBuffers* buffersIn)
{
	DiffuseRenderBuffersGL* buffers = reinterpret_cast<DiffuseRenderBuffersGL*>(buffersIn);
	if (buffers->mNumParticles > 0)
	{
		glDeleteBuffers(1, &buffers->mDiffusePositionVBO);
		glDeleteBuffers(1, &buffers->mDiffuseVelocityVBO);

		NvFlexUnregisterOGLBuffer(buffers->mDiffusePositionsBuf);
		NvFlexUnregisterOGLBuffer(buffers->mDiffuseVelocitiesBuf);
	}
}

void UpdateDiffuseRenderBuffers(DiffuseRenderBuffers* buffersIn, NvFlexSolver* solver)
{
	DiffuseRenderBuffersGL* buffers = reinterpret_cast<DiffuseRenderBuffersGL*>(buffersIn);
	// diffuse particles
	if (buffers->mNumParticles)
	{
		NvFlexGetDiffuseParticles(solver, buffers->mDiffusePositionsBuf, buffers->mDiffuseVelocitiesBuf, NULL);
	}
}

void UpdateDiffuseRenderBuffers(DiffuseRenderBuffers* buffersIn, Vec4* diffusePositions, Vec4* diffuseVelocities, int numDiffuseParticles)
{
	DiffuseRenderBuffersGL* buffers = reinterpret_cast<DiffuseRenderBuffersGL*>(buffersIn);
	// diffuse particles
	if (buffers->mNumParticles)
	{
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffusePositionVBO));
		glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles*sizeof(Vec4), diffusePositions));

		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffuseVelocityVBO));
		glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles*sizeof(Vec4), diffuseVelocities));
	}
}

void RenderFullscreenQuad()
{
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);

	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);

	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);

	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);

	glEnd();
}

extern Mesh* g_mesh;
extern std::vector<RenderMaterial> g_renderMaterials;
void DrawStaticShapes();

void RenderEllipsoids(FluidRenderer* render, FluidRenderBuffers* buffersIn, int n, int offset, float radius, float screenWidth, float screenAspect, float fov, Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, ShadowMap* shadowMap, Vec4 color, float blur, float ior, bool debug, const RenderTexture* customRenderTarget)
{
	FluidRenderBuffersGL* buffers = reinterpret_cast<FluidRenderBuffersGL*>(buffersIn);

#if !ENABLE_SIMPLE_FLUID
	// resolve msaa back buffer to texture
	// TODO(jaliang): can use glGet to get current framebuffer?
	if (customRenderTarget)
	{
		glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, customRenderTarget->colorFrameBuffer));
	}
	else
	{
		glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, g_msaaFbo));
	}
	glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER_EXT, render->mSceneFbo));
	glVerify(glBlitFramebuffer(0, 0, GLsizei(screenWidth), GLsizei(screenWidth/screenAspect), 0, 0, GLsizei(screenWidth), GLsizei(screenWidth/screenAspect), GL_COLOR_BUFFER_BIT, GL_LINEAR));

	//thickness texture
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, render->mThicknessFbo));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render->mThicknessTex, 0));
	glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));

	glViewport(0, 0, GLsizei(screenWidth), GLsizei(screenWidth/screenAspect));
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	glDepthMask(GL_TRUE);
	glDisable(GL_CULL_FACE);

	
	if (g_mesh)
		DrawMesh(g_mesh, g_renderMaterials[0]);

	DrawStaticShapes();


	glClear(GL_COLOR_BUFFER_BIT);

	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glDepthMask(GL_FALSE);

	// make sprites larger to get smoother thickness texture
	const float thicknessScale = 4.0f; 

	glUseProgram(render->mPointThicknessProgram);
	glUniform1f( glGetUniformLocation(render->mPointThicknessProgram, "pointRadius"), thicknessScale*radius);
	glUniform1f( glGetUniformLocation(render->mPointThicknessProgram, "pointScale"), screenWidth/screenAspect * (1.0f / (tanf(fov*0.5f))));

	glEnableClientState(GL_VERTEX_ARRAY);			
	glBindBuffer(GL_ARRAY_BUFFER, buffers->mPositionVBO);
	glVertexPointer(3, GL_FLOAT, sizeof(float)*4, (void*)(offset*sizeof(float)*4));

	glDrawArrays(GL_POINTS, 0, n);

	glUseProgram(0);
	glDisableClientState(GL_VERTEX_ARRAY);	
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND);
#endif

	// depth texture
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, render->mDepthFbo));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_ARB, render->mDepthTex, 0));
	glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));

	// draw points
	//glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glViewport(0, 0, int(screenWidth), int(screenWidth/screenAspect));
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	const float viewHeight = tanf(fov/2.0f);

	glUseProgram(render->mEllipsoidDepthProgram);
	glUniform3fv( glGetUniformLocation(render->mEllipsoidDepthProgram, "invViewport"), 1, Vec3(1.0f/screenWidth, screenAspect/screenWidth, 1.0f));
	glUniform3fv( glGetUniformLocation(render->mEllipsoidDepthProgram, "invProjection"), 1, Vec3(screenAspect*viewHeight, viewHeight, 1.0f));

	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, buffers->mPositionVBO);
	glVertexPointer(3, GL_FLOAT, sizeof(float)*4, 0);//(void*)(offset*sizeof(float)*4));

	// ellipsoid eigenvectors
	int s1 = glGetAttribLocation(render->mEllipsoidDepthProgram, "q1");
	glEnableVertexAttribArray(s1);
	glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[0]);
	glVertexAttribPointer(s1, 4, GL_FLOAT, GL_FALSE, 0, 0);// (void*)(offset*sizeof(float)*4));

	int s2 = glGetAttribLocation(render->mEllipsoidDepthProgram, "q2");
	glEnableVertexAttribArray(s2);
	glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[1]);
	glVertexAttribPointer(s2, 4, GL_FLOAT, GL_FALSE, 0, 0);//(void*)(offset*sizeof(float)*4));

	int s3 = glGetAttribLocation(render->mEllipsoidDepthProgram, "q3");
	glEnableVertexAttribArray(s3);
	glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[2]);
	glVertexAttribPointer(s3, 4, GL_FLOAT, GL_FALSE, 0, 0);// (void*)(offset*sizeof(float)*4));
	
	glVerify(glDrawArrays(GL_POINTS, offset, n));

	glUseProgram(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableVertexAttribArray(s1);
	glDisableVertexAttribArray(s2);
	glDisableVertexAttribArray(s3);

	glDisable(GL_POINT_SPRITE);

	//---------------------------------------------------------------
	// blur

	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render->mDepthSmoothTex, 0));
	glUseProgram(render->mDepthBlurProgram);
	
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_RECTANGLE_ARB);	
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, render->mDepthTex);

	glActiveTexture(GL_TEXTURE1);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, render->mThicknessTex);

	glVerify(glUniform1f( glGetUniformLocation(render->mDepthBlurProgram, "blurRadiusWorld"), radius*0.5f));	// blur half the radius by default
	glVerify(glUniform1f( glGetUniformLocation(render->mDepthBlurProgram, "blurScale"), screenWidth/screenAspect * (1.0f / (tanf(fov*0.5f)))));
	glVerify(glUniform2fv( glGetUniformLocation(render->mDepthBlurProgram, "invTexScale"), 1, Vec2(1.0f/screenAspect, 1.0f)));
	glVerify(glUniform1f( glGetUniformLocation(render->mDepthBlurProgram, "blurFalloff"),  blur));
	glVerify(glUniform1i( glGetUniformLocation(render->mDepthBlurProgram, "depthTex"), 0));
	glVerify(glUniform1i( glGetUniformLocation(render->mDepthBlurProgram, "thicknessTex"), 1));
	glVerify(glUniform1i(glGetUniformLocation(render->mDepthBlurProgram, "debug"), debug));

	glVerify(RenderFullscreenQuad());

	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);	

	//---------------------------------------------------------------
	// composite with scene
	// TODO(jaliang): can use glGet to get current framebuffer?
	if (customRenderTarget)
	{
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, customRenderTarget->colorFrameBuffer));
	} 
	else
	{
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));
	}
	glVerify(glEnable(GL_DEPTH_TEST));
	glVerify(glDepthMask(GL_TRUE));
	glVerify(glDisable(GL_BLEND));
	glVerify(glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

	glVerify(glUseProgram(render->mCompositeProgram));	

	glVerify(glUniform2fv(glGetUniformLocation(render->mCompositeProgram, "invTexScale"), 1, Vec2(1.0f/screenWidth, screenAspect/screenWidth)));
	glVerify(glUniform2fv(glGetUniformLocation(render->mCompositeProgram, "clipPosToEye"), 1, Vec2(tanf(fov*0.5f)*screenAspect, tanf(fov*0.5f))));
	glVerify(glUniform4fv(glGetUniformLocation(render->mCompositeProgram, "color"), 1, color));
	glVerify(glUniform1f(glGetUniformLocation(render->mCompositeProgram, "ior"),  ior));
	glVerify(glUniform1f(glGetUniformLocation(render->mCompositeProgram, "spotMin"), g_spotMin));
	glVerify(glUniform1f(glGetUniformLocation(render->mCompositeProgram, "spotMax"), g_spotMax));
	glVerify(glUniform1i(glGetUniformLocation(render->mCompositeProgram, "debug"), debug));

	glVerify(glUniform3fv(glGetUniformLocation(render->mCompositeProgram, "lightPos"), 1, lightPos));
	glVerify(glUniform3fv(glGetUniformLocation(render->mCompositeProgram, "lightDir"), 1, -Normalize(lightTarget-lightPos)));
	glVerify(glUniformMatrix4fv(glGetUniformLocation(render->mCompositeProgram, "lightTransform"), 1, false, lightTransform));
	
	glVerify(glUniform1f(glGetUniformLocation(render->mCompositeProgram, "depthMinRange"), currentDepthProfile.minRange));
	glVerify(glUniform1f(glGetUniformLocation(render->mCompositeProgram, "depthMaxRange"), currentDepthProfile.maxRange));

	const Vec2 taps[] = 
	{ 
		Vec2(-0.326212f,-0.40581f),Vec2(-0.840144f,-0.07358f),
		Vec2(-0.695914f,0.457137f),Vec2(-0.203345f,0.620716f),
		Vec2(0.96234f,-0.194983f),Vec2(0.473434f,-0.480026f),
		Vec2(0.519456f,0.767022f),Vec2(0.185461f,-0.893124f),
		Vec2(0.507431f,0.064425f),Vec2(0.89642f,0.412458f),
		Vec2(-0.32194f,-0.932615f),Vec2(-0.791559f,-0.59771f) 
	};
	
	glVerify(glUniform2fv(glGetUniformLocation(render->mCompositeProgram, "shadowTaps"), 12, &taps[0].x));

	// smoothed depth tex
	glVerify(glActiveTexture(GL_TEXTURE0));
	glVerify(glEnable(GL_TEXTURE_2D));
	glVerify(glBindTexture(GL_TEXTURE_2D, render->mDepthSmoothTex));

	// shadow tex
	glVerify(glActiveTexture(GL_TEXTURE1));
	glVerify(glEnable(GL_TEXTURE_2D));
	glVerify(glBindTexture(GL_TEXTURE_2D, shadowMap->texture));

	// thickness tex
	glVerify(glActiveTexture(GL_TEXTURE2));
	glVerify(glEnable(GL_TEXTURE_2D));
	glVerify(glBindTexture(GL_TEXTURE_2D, render->mThicknessTex));

	// scene tex
	glVerify(glActiveTexture(GL_TEXTURE3));
	glVerify(glEnable(GL_TEXTURE_2D));
	glVerify(glBindTexture(GL_TEXTURE_2D, render->mSceneTex));

	/*
	// reflection tex
	glVerify(glActiveTexture(GL_TEXTURE5));
	glVerify(glEnable(GL_TEXTURE_2D));
	glVerify(glBindTexture(GL_TEXTURE_2D, reflectMap->texture));	
	*/

	glVerify(glUniform1i(glGetUniformLocation(render->mCompositeProgram, "tex"), 0));
	glVerify(glUniform1i(glGetUniformLocation(render->mCompositeProgram, "shadowTex"), 1));
	glVerify(glUniform1i(glGetUniformLocation(render->mCompositeProgram, "thicknessTex"), 2));
	glVerify(glUniform1i(glGetUniformLocation(render->mCompositeProgram, "sceneTex"), 3));
	glVerify(glUniform1i(glGetUniformLocation(render->mCompositeProgram, "reflectTex"), 5));

	// -- end shadowing
	
	// ignores projection matrices
	glVerify(RenderFullscreenQuad());

	// reset state
	glActiveTexture(GL_TEXTURE5);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE3);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE2);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE1);
	glDisable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_TEXTURE_2D);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);
}

//------------------------------------------------------------------------------
// Diffuse Shading

const char *vertexDiffuseShader = STRINGIFY(

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform vec3 lightPos;
uniform vec3 lightDir;
uniform mat4 lightTransform;
uniform float spotMin;
uniform float spotMax;
uniform vec4 color;


void main()
{
	vec3 worldPos = gl_Vertex.xyz;// - vec3(0.0, 0.1*0.25, 0.0);	// hack move towards ground to account for anisotropy;
	vec4 eyePos = gl_ModelViewMatrix * vec4(worldPos, 1.0);

	gl_Position = gl_ProjectionMatrix * eyePos;
	//gl_Position.z -= 0.0025;	// bias above fluid surface

    // calculate window-space point size
	gl_PointSize = pointRadius * (pointScale / gl_Position.w);

	gl_TexCoord[0] = gl_MultiTexCoord0;    
	gl_TexCoord[1] = vec4(worldPos, gl_Vertex.w);
	gl_TexCoord[2] = eyePos;

	gl_TexCoord[3].xyz = gl_ModelViewMatrix*vec4(gl_MultiTexCoord1.xyz, 0.0);
	gl_TexCoord[4].xyzw = color;

	// hack to color different emitters 
	if (gl_MultiTexCoord1.w == 2.0)
		gl_TexCoord[4].xyzw = vec4(0.85, 0.65, 0.65, color.w);
	else if (gl_MultiTexCoord1.w == 1.0)
		gl_TexCoord[4].xyzw = vec4(0.65, 0.85, 0.65, color.w);

	// compute ndc pos for frustrum culling in GS
	vec4 ndcPos = gl_ModelViewProjectionMatrix * vec4(worldPos.xyz, 1.0);
	gl_TexCoord[5] = ndcPos / ndcPos.w;
}
);




const char *geometryDiffuseShader = 
"#version 120\n"
"#extension GL_EXT_geometry_shader4 : enable\n"
STRINGIFY(

uniform float pointScale;  // point size in world space
uniform float motionBlurScale;
uniform float diffusion;
uniform vec3 lightDir;

void main()
{
	vec4 ndcPos = gl_TexCoordIn[0][5];

	// frustrum culling
	const float ndcBound = 1.0;
	if (ndcPos.x < -ndcBound) return;
	if (ndcPos.x > ndcBound) return;
	if (ndcPos.y < -ndcBound) return;
	if (ndcPos.y > ndcBound) return;

	float velocityScale = 1.0;

	vec3 v = gl_TexCoordIn[0][3].xyz*velocityScale;
	vec3 p = gl_TexCoordIn[0][2].xyz;
		
	// billboard in eye space
	vec3 u = vec3(0.0, pointScale, 0.0);
	vec3 l = vec3(pointScale, 0.0, 0.0);
	
	// increase size based on life
	float lifeFade = mix(1.0f+diffusion, 1.0, min(1.0, gl_TexCoordIn[0][1].w*0.25f));
	u *= lifeFade;
	l *= lifeFade;

	//lifeFade = 1.0;

	float fade = 1.0/(lifeFade*lifeFade);
	float vlen = length(v)*motionBlurScale;

	if (vlen > 0.5)
	{
		float len = max(pointScale, vlen*0.016);
		fade = min(1.0, 2.0/(len/pointScale));

		u = normalize(v)*max(pointScale, vlen*0.016);	// assume 60hz
		l = normalize(cross(u, vec3(0.0, 0.0, -1.0)))*pointScale;
	}	
	
	{
		
		gl_TexCoord[1] = gl_TexCoordIn[0][1];	// vertex world pos (life in w)
		gl_TexCoord[2] = gl_TexCoordIn[0][2];	// vertex eye pos
		gl_TexCoord[3] = gl_TexCoordIn[0][3];	// vertex velocity in view space
		gl_TexCoord[3].w = fade;
		gl_TexCoord[4] = gl_ModelViewMatrix*vec4(lightDir, 0.0);
		gl_TexCoord[4].w = gl_TexCoordIn[0][3].w; // attenuation
		gl_TexCoord[5].xyzw = gl_TexCoordIn[0][4].xyzw;	// color

		float zbias = 0.0f;//0.00125*2.0;

        gl_TexCoord[0] = vec4(0.0, 1.0, 0.0, 0.0);
        gl_Position = gl_ProjectionMatrix * vec4(p + u - l, 1.0);
		gl_Position.z -= zbias;
        EmitVertex();
		
		gl_TexCoord[0] = vec4(0.0, 0.0, 0.0, 0.0);
        gl_Position = gl_ProjectionMatrix * vec4(p - u - l, 1.0);
		gl_Position.z -= zbias;
        EmitVertex();

		gl_TexCoord[0] = vec4(1.0, 1.0, 0.0, 0.0);
        gl_Position = gl_ProjectionMatrix * vec4(p + u + l, 1.0);
		gl_Position.z -= zbias;
        EmitVertex();

		gl_TexCoord[0] = vec4(1.0, 0.0, 0.0, 0.0);
        gl_Position = gl_ProjectionMatrix * vec4(p - u + l, 1.0);
		gl_Position.z -= zbias;
        EmitVertex();
    }
}
);

const char *fragmentDiffuseShader = STRINGIFY(

float sqr(float x) { return x*x; }
float cube(float x) { return x*x*x; }

uniform sampler2D depthTex;
uniform sampler2D noiseTex;
uniform vec2 invViewport;
uniform vec4 color;
uniform bool front;
uniform bool shadow;

//uniform sampler2DShadow shadowTex;
uniform sampler2D shadowTex;
uniform vec2 shadowTaps[12];
uniform mat4 lightTransform;
uniform vec3 lightDir;
uniform float inscatterCoefficient;
uniform float outscatterCoefficient;

void main()
{
	float attenuation = gl_TexCoord[4].w;
	float lifeFade = min(1.0, gl_TexCoord[1].w*0.125);

    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_TexCoord[0].xy*vec2(2.0, 2.0) + vec2(-1.0, -1.0);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
   	normal.z = 1.0-mag;

	float velocityFade = gl_TexCoord[3].w;
	float alpha = lifeFade*velocityFade*sqr(normal.z);

	gl_FragColor = alpha;
}
);

int GetNumDiffuseRenderParticles(DiffuseRenderBuffers* buffers)
{
	return reinterpret_cast<DiffuseRenderBuffersGL*>(buffers)->mNumParticles;
}

void RenderDiffuse(FluidRenderer* render, DiffuseRenderBuffers* buffersIn, int n, float radius, float screenWidth, float screenAspect, float fov, Vec4 color, Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, ShadowMap* shadowMap, float motionBlur, float inscatter, float outscatter, bool shadow, bool front)
{
	DiffuseRenderBuffersGL* buffers = reinterpret_cast<DiffuseRenderBuffersGL*>(buffersIn);

	if (s_diffuseProgram == GLuint(-1))
		s_diffuseProgram = CompileProgram(vertexDiffuseShader, fragmentDiffuseShader, geometryDiffuseShader);

	int thicknessScale = 1;

	if (s_diffuseProgram)
	{
#if USE_HDR_DIFFUSE_BLEND
	
		{
			glVerify(glBindFramebuffer(GL_READ_FRAMEBUFFER, g_msaaFbo));
			glVerify(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, render->mThicknessFbo));
			glVerify(glBlitFramebuffer(0, 0, render->mSceneWidth, render->mSceneHeight, 0, 0, render->mSceneWidth/thicknessScale, render->mSceneHeight/thicknessScale, GL_DEPTH_BUFFER_BIT, GL_NEAREST));

			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);		
		}
#endif

		glEnable(GL_POINT_SPRITE);
		glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDepthMask(GL_FALSE);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

		
		glUseProgram(s_diffuseProgram);
		glUniform1f( glGetUniformLocation(s_diffuseProgram, "motionBlurScale"), motionBlur);
		glUniform1f( glGetUniformLocation(s_diffuseProgram, "diffusion"), 1.0f);
		glUniform1f( glGetUniformLocation(s_diffuseProgram, "pointScale"), radius*1.0f);
		glUniform1f( glGetUniformLocation(s_diffuseProgram, "pointRadius"), screenWidth / float(thicknessScale) / (2.0f*screenAspect*tanf(fov*0.5f)));		
		glUniform2fv( glGetUniformLocation(s_diffuseProgram, "invViewport"), 1, Vec2(1.0f/screenWidth, screenAspect/screenWidth));
		glUniform4fv( glGetUniformLocation(s_diffuseProgram, "color"), 1, color);
		glUniform1i( glGetUniformLocation(s_diffuseProgram, "tex"), 0);
		glUniform1f( glGetUniformLocation(s_diffuseProgram, "inscatterCoefficient"), inscatter);
		glUniform1f( glGetUniformLocation(s_diffuseProgram, "outscatterCoefficient"), outscatter);

		GLint uLightTransform = glGetUniformLocation(s_diffuseProgram, "lightTransform");
		glUniformMatrix4fv(uLightTransform, 1, false, lightTransform);

		GLint uLightPos = glGetUniformLocation(s_diffuseProgram, "lightPos");
		glUniform3fv(uLightPos, 1, lightPos);
	
		GLint uLightDir = glGetUniformLocation(s_diffuseProgram, "lightDir");
		glUniform3fv(uLightDir, 1, Normalize(lightTarget-lightPos));

		glUniform1f( glGetUniformLocation(s_diffuseProgram, "spotMin"), g_spotMin);
		glUniform1f( glGetUniformLocation(s_diffuseProgram, "spotMax"), g_spotMax);

		const Vec2 taps[] = 
		{ 
			Vec2(-0.326212f,-0.40581f),Vec2(-0.840144f,-0.07358f),
			Vec2(-0.695914f,0.457137f),Vec2(-0.203345f,0.620716f),
			Vec2(0.96234f,-0.194983f),Vec2(0.473434f,-0.480026f),
			Vec2(0.519456f,0.767022f),Vec2(0.185461f,-0.893124f),
			Vec2(0.507431f,0.064425f),Vec2(0.89642f,0.412458f),
			Vec2(-0.32194f,-0.932615f),Vec2(-0.791559f,-0.59771f) 
		};
	
		glVerify(glUniform2fv(glGetUniformLocation(s_diffuseProgram, "shadowTaps"), 12, &taps[0].x));
		glVerify(glUniform1i(glGetUniformLocation(s_diffuseProgram, "noiseTex"), 2));
		glVerify(glUniform1i(glGetUniformLocation(s_diffuseProgram, "shadowTex"), 1));
		glVerify(glUniform1i(glGetUniformLocation(s_diffuseProgram, "depthTex"), 0));
		glVerify(glUniform1i(glGetUniformLocation(s_diffuseProgram, "front"), front));
		glVerify(glUniform1i(glGetUniformLocation(s_diffuseProgram, "shadow"), shadow));

		// noise tex
		//glActiveTexture(GL_TEXTURE2);
		//glEnable(GL_TEXTURE_2D);
		//glBindTexture(GL_TEXTURE_2D, noiseTex);

		// shadow tex
		glActiveTexture(GL_TEXTURE1);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, shadowMap->texture);
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)); 
		//glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)); 


		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, render->mDepthSmoothTex);

		glClientActiveTexture(GL_TEXTURE1);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffuseVelocityVBO));
		glTexCoordPointer(4, GL_FLOAT, sizeof(float)*4, 0);

		glEnableClientState(GL_VERTEX_ARRAY);			
		glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffusePositionVBO);
		glVertexPointer(4, GL_FLOAT, sizeof(float)*4, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		glDrawArrays(GL_POINTS, 0, n);

		glUseProgram(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glDisableClientState(GL_VERTEX_ARRAY);	
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);	
		glDisable(GL_POINT_SPRITE);
		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);

		glVerify(glActiveTexture(GL_TEXTURE2));
		glVerify(glDisable(GL_TEXTURE_2D));
		glVerify(glActiveTexture(GL_TEXTURE1));
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)); 
		glVerify(glDisable(GL_TEXTURE_2D));
		glVerify(glActiveTexture(GL_TEXTURE0));
		glVerify(glDisable(GL_TEXTURE_2D));

#if USE_HDR_DIFFUSE_BLEND
		
			{
			glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaaFbo));
			glVerify(glViewport(0, 0, int(screenWidth), int(screenWidth/screenAspect)));

			//glClear(GL_COLOR_BUFFER_BIT);
			glUseProgram(0);
			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			glDepthMask(GL_FALSE);
			glDisable(GL_CULL_FACE);
			
			glVerify(glActiveTexture(GL_TEXTURE0));
			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, render->mThicknessTex);

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
		
			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
			glLoadIdentity();
			gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0);

			RenderFullscreenQuad();

			glMatrixMode(GL_MODELVIEW);
			glPopMatrix();
		
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();

			glDepthMask(GL_TRUE);
		}
#endif

	}
}




struct RenderMesh
{
	GLuint mPositionsVBO;
	GLuint mNormalsVBO;
	GLuint mTexCoordsVBO;

	GLuint mIndicesIBO;

	int mNumVertices;
	int mNumFaces;

	Vec3 localLower;
	Vec3 localUpper;

};

RenderMesh* CreateRenderMesh(const Mesh* m)
{
	RenderMesh* mesh = new RenderMesh();
	memset(mesh, 0, sizeof(RenderMesh));

	mesh->mNumVertices = m->GetNumVertices();
	mesh->mNumFaces = m->GetNumFaces();

	m->GetBounds(mesh->localLower, mesh->localUpper);

	// vbos
	glVerify(glGenBuffers(1, &mesh->mPositionsVBO));
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, mesh->mPositionsVBO));
	glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*m->m_positions.size(), &m->m_positions[0], GL_STATIC_DRAW));

	glVerify(glGenBuffers(1, &mesh->mNormalsVBO));
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, mesh->mNormalsVBO));
	glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*m->m_normals.size(), &m->m_normals[0], GL_STATIC_DRAW));

	if (m->m_texcoords.size())
	{
		glVerify(glGenBuffers(1, &mesh->mTexCoordsVBO));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, mesh->mTexCoordsVBO));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*m->m_texcoords.size(), &m->m_texcoords[0], GL_STATIC_DRAW));
	}


	glVerify(glGenBuffers(1, &mesh->mIndicesIBO));
	glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->mIndicesIBO));
	glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*m->m_indices.size(), &m->m_indices[0], GL_STATIC_DRAW));
	glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));


	return mesh;
}

void UpdateRenderMesh(RenderMesh* mesh,  const Vec3* positions, const Vec3* normals, int n)
{
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, mesh->mPositionsVBO));
	glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*n, positions, GL_STATIC_DRAW));

	glVerify(glBindBuffer(GL_ARRAY_BUFFER, mesh->mNormalsVBO));
	glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*n, normals, GL_STATIC_DRAW));

}

void DestroyRenderMesh(RenderMesh* m)
{
	glVerify(glDeleteBuffers(1, &m->mPositionsVBO));
	glVerify(glDeleteBuffers(1, &m->mNormalsVBO));
	glVerify(glDeleteBuffers(1, &m->mTexCoordsVBO));
	glVerify(glDeleteBuffers(1, &m->mIndicesIBO));
}

void DrawRenderMesh(RenderMesh* m, const Matrix44& xform, const RenderMaterial& mat, int startTri, int endTri)
{
	
	if (m && !mat.hidden)
	{
		SetMaterial(xform, mat);

		glVerify(glEnableClientState(GL_VERTEX_ARRAY));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, m->mPositionsVBO));
		glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float)*3, 0));	

		glVerify(glEnableClientState(GL_NORMAL_ARRAY));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, m->mNormalsVBO));
		glVerify(glNormalPointer(GL_FLOAT, sizeof(float)*3, 0));		

		if (m->mTexCoordsVBO)
		{
			glVerify(glClientActiveTexture(GL_TEXTURE0));
			glVerify(glEnableClientState(GL_TEXTURE_COORD_ARRAY));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, m->mTexCoordsVBO));
			glVerify(glTexCoordPointer(2, GL_FLOAT, sizeof(float)*2, 0));
		}

		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->mIndicesIBO));

		// num indices to draw
		int n;
		if (endTri == 0)
			n = m->mNumFaces*3;
		else
			n = (endTri-startTri)*3;

		glVerify(glDrawElements(GL_TRIANGLES, n, GL_UNSIGNED_INT, (const void*)(startTri*sizeof(int)*3)));

		glVerify(glDisableClientState(GL_VERTEX_ARRAY));
		glVerify(glDisableClientState(GL_NORMAL_ARRAY));
		glVerify(glDisableClientState(GL_TEXTURE_COORD_ARRAY));

		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));	
	}
}

void DrawRenderMeshInstances(RenderMesh* m, const Matrix44* xforms, int n, const RenderMaterial& mat)
{
	if (m)
	{
		GLint program;
		glGetIntegerv(GL_CURRENT_PROGRAM, &program);

		GLint param = glGetUniformLocation(program, "objectTransform");

		glVerify(glColor3fv(mat.frontColor));
		glVerify(glSecondaryColor3fv(mat.backColor));

		glVerify(glEnableClientState(GL_VERTEX_ARRAY));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, m->mPositionsVBO));
		glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float)*3, 0));	

		glVerify(glEnableClientState(GL_NORMAL_ARRAY));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, m->mNormalsVBO));
		glVerify(glNormalPointer(GL_FLOAT, sizeof(float)*3, 0));
		
		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->mIndicesIBO));

		for (int i=0; i < n; ++i)
		{
			if (program)
				glUniformMatrix4fv( param, 1, false, xforms[i]);

			glVerify(glDrawElements(GL_TRIANGLES, m->mNumFaces*3, GL_UNSIGNED_INT, 0));
		}

		glVerify(glDisableClientState(GL_VERTEX_ARRAY));
		glVerify(glDisableClientState(GL_NORMAL_ARRAY));

		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
	}
}



void GetRenderMeshBounds(RenderMesh* m, Vec3* localLower, Vec3* localUpper)
{
	*localLower = m->localLower;
	*localUpper = m->localUpper;
}

const char *quadVertexShader = STRINGIFY(

void main()
{
	// calculate window-space point size
	gl_Position = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xyz, 1.0);
	gl_TexCoord[0] = gl_MultiTexCoord0;

}
);

const char *quadFragmentShader = STRINGIFY(

uniform sampler2D tex;
uniform int mode;

uniform float depthMinRange;
uniform float depthMaxRange;

void main()
{
	vec4 p = texture2D(tex, gl_TexCoord[0].xy);

	if (mode == 0)
	{
		// rgb
		gl_FragColor = p;
	}
	else
	{
		// depth
		float depth = p.w;
		
		// Normalizes depth vals for display purposes. Vals at minRange should goto 0, while maxRange should goto 1.
		if (depthMaxRange > 0)
			depth = (depth - depthMinRange) / (depthMaxRange - depthMinRange);
		else
			depth -= depthMinRange;

		gl_FragColor = vec4(depth);
	}

}
);



void DrawQuad(int x, int y, int width, int height, RenderTexture* texture, int mode)
{
	if (s_quadProgram == GLuint(-1))
		s_quadProgram = CompileProgram(quadVertexShader, quadFragmentShader);

	glUseProgram(s_quadProgram);
	glUniform1i( glGetUniformLocation(s_quadProgram, "mode"), mode);
	glUniform1f(glGetUniformLocation(s_quadProgram, "depthMinRange"), currentDepthProfile.minRange);
	glUniform1f(glGetUniformLocation(s_quadProgram, "depthMaxRange"), currentDepthProfile.maxRange);
	
	if (texture)
	{
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);		
		glBindTexture(GL_TEXTURE_2D, texture->colorTex);
	}
	else
	{
		glActiveTexture(GL_TEXTURE0);
		glDisable(GL_TEXTURE_2D);
	}

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	
	Matrix44 ortho = OrthographicMatrix(0.0f, float(g_screenWidth), 0.0f, float(g_screenHeight), 0.0f, 1.0f);
	glLoadMatrixf(ortho);

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	glColor3f(1.0f, 1.0f, 1.0f);
		
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(float(x), float(y));
	
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(float(x+width), float(y));

	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(float(x+width), float(y+height));
	
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(float(x), float(y+height));

	glEnd();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}


void BeginLines(float lineWidth, bool depthTest)
{
	glUseProgram(0);
	
	if (depthTest)
		glEnable(GL_DEPTH_TEST);
	else
		glDisable(GL_DEPTH_TEST);

	glDisable(GL_BLEND);

	glLineWidth(lineWidth);

	for (int i = 0; i < 8; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glDisable(GL_TEXTURE_2D);
	}

	glBegin(GL_LINES);
}

void DrawLine(const Vec3& p, const Vec3& q, const Vec4& color)
{
	glColor4fv(color);
	glVertex3fv(p);
	glVertex3fv(q);
}

void EndLines()
{
	glEnd();
}

void BeginPoints(float size)
{
	glPointSize(size);

	glUseProgram(0);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

	for (int i = 0; i < 8; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glDisable(GL_TEXTURE_2D);
	}

	glBegin(GL_POINTS);
}

void DrawPoint(const Vec3& p, const Vec4& color)
{
	glColor3fv(color);
	glVertex3fv(p);
}

void EndPoints()
{
	glEnd();
}



float SyncAndGetRenderTime(unsigned long long* begin, unsigned long long* end, unsigned long long* freq)
{
	*begin = 0;
	*end = 0;
	*freq = 1;
	return 0.0f;
}

float RendererGetDeviceTimestamps(unsigned long long* begin, unsigned long long* end, unsigned long long* freq) { return 0.0f; }
void* GetGraphicsCommandQueue() { return nullptr; } 
void GraphicsTimerBegin() { }
void GraphicsTimerEnd() { }

void StartGpuWork() { }
void EndGpuWork() { }

void GetRenderDevice(void** deviceOut, void** contextOut) 
{ 
	*deviceOut = nullptr;
	*contextOut = nullptr;
}

void DrawImguiGraph()
{
	imguiGraphDraw();
}

void DrawLines(Vec3* pos, int numPos, Vec4 color, bool depthTest, float lineWidth)
{
	glUseProgram(0);

	if (depthTest)
		glEnable(GL_DEPTH_TEST);
	else
		glDisable(GL_DEPTH_TEST);

	glDisable(GL_BLEND);

	glLineWidth(lineWidth);

	for (int i = 0; i < 8; ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glDisable(GL_TEXTURE_2D);
	}

	glColor4fv(&color.x);

	glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(3, GL_FLOAT, sizeof(Vec3), pos);
	glDrawArrays(GL_LINES, 0, numPos);		

	glDisableClientState(GL_VERTEX_ARRAY);
}
