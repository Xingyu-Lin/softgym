#include "vr.h"

#if FLEX_VR

#include <memory>
#include "../opengl/shader.h"
#include "../../demo/shaders.h"
#include "../../core/platform.h"

VrSystem* g_vrSystem;

static void CleanAllGlErrors()
{
	int i = 0;
	const int MaxGlErrors = 100;
	for (; glGetError() != GL_NO_ERROR && i < MaxGlErrors; ++i);
	assert(i < MaxGlErrors);
}

std::string GetTrackedDeviceString(vr::IVRSystem *vrSystem, vr::TrackedDeviceIndex_t deviceInd, vr::TrackedDeviceProperty prop, vr::TrackedPropertyError *peError = nullptr)
{
	uint32_t bufferLen = vrSystem->GetStringTrackedDeviceProperty(deviceInd, prop, nullptr, 0, peError);
	if (bufferLen == 0)
		return std::string();

	std::vector<char> buffer(bufferLen);
	vrSystem->GetStringTrackedDeviceProperty(deviceInd, prop, buffer.data(), bufferLen, peError);
	return std::string(buffer.data());
}

void VrSystem::Update()
{
	vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
	const vr::EVRCompositorError compErr = vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
	assert(compErr == vr::VRCompositorError_None || compErr == vr::VRCompositorError_DoNotHaveFocus);

	if (GetTransformation(poses + mHmdId, mHmdPos, mHmdRotation, false) && !mValidInitialHmdPos)
	{
		mInitialHmdPos = mHmdPos;
		SetInverseInitialHmdRotation(InverseRotation(mHmdRotation));
		mValidInitialHmdPos = true;
	}
	mHmdRotation = ApplyHmdRotationCorrection(mHmdRotation);

	for (size_t i = 0; i < mControllers.size(); ++i)
	{
		Controller& curController = mControllers[i];
		curController.valid = GetTransformation(poses + curController.id, curController.state.pos, curController.state.orientation);
		curController.state.pos = GetPosInScene(curController.state.pos);
		// Note: for now will'll process only the "button pressed" event
		curController.state.buttonPressed = false;

		if (curController.valid && curController.triggerId >= 0)
		{
			vr::VRControllerState_t controllerState;
			const bool result = mOpenVR->GetControllerState(curController.id, &controllerState, sizeof(controllerState));
			if (result)
			{
				curController.state.triggerValue = controllerState.rAxis[curController.triggerId].x;
			}
			else
			{
				curController.valid = false;
			}
		}
	}

	vr::VREvent_t vrEvent;
	while (mOpenVR->PollNextEvent(&vrEvent, sizeof(vrEvent)))
	{
		if (vrEvent.eventType == vr::EVREventType::VREvent_ButtonPress && vrEvent.data.controller.button == vr::k_EButton_Grip)
		{
			for (size_t i = 0; i < mControllers.size(); ++i)
			{
				Controller& curController = mControllers[i];
				if (curController.id == vrEvent.trackedDeviceIndex)
				{
					curController.state.buttonPressed = true;
				}
			}
		}
	}
}

bool VrSystem::GetControllerState(size_t controllerIndex, VrControllerState& resultState)
{
	if (controllerIndex >= mControllers.size())
	{
		return false;
	}

	const Controller& curController = mControllers[controllerIndex];
	if (!curController.valid)
	{
		return false;
	}

	resultState = curController.state;

	return true;
}

void VrSystem::UploadGraphicsToDevice() const
{
	for (int eye = 0; eye < 2; ++eye)
	{
		const uintptr_t textureHandle = mEyeResources[eye].textureId;
		const vr::Texture_t eyeTexture = { reinterpret_cast<void*>(textureHandle), vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
		const vr::EVRCompositorError compErr = vr::VRCompositor()->Submit(static_cast<vr::EVREye>(eye), &eyeTexture);
		assert(compErr == vr::VRCompositorError_None || compErr == vr::VRCompositorError_DoNotHaveFocus);
	}
	// Clean up after OpenVR as they don't fully check for errors
	CleanAllGlErrors();
}

void VrSystem::InitGraphicalResources()
{
	for (int i = 0; i < 2; ++i)
	{
		CreateVrRenderTarget(static_cast<vr::EVREye>(i));
	}

	const char* vertexShader = "#version 410\n" STRINGIFY(
		uniform mat4 matrix;
		layout(location = 0) in vec4 position;
		layout(location = 1) in vec3 v3NormalIn;
		layout(location = 2) in vec2 modelTexCoords;
		out vec2 texCoords;
		void main()
		{
			texCoords = modelTexCoords;
			gl_Position = matrix * vec4(position.xyz, 1);
		}
	);

	const char* fragmentShader = "#version 410 core\n" STRINGIFY(
		uniform sampler2D modelTex;
		in vec2 texCoords;
		out vec4 outColor;
		void main()
		{
			outColor = texture(modelTex, texCoords);
		}
	);

	glVerify(mControllersShader = CompileProgram(vertexShader, fragmentShader));
	if (!mControllersShader)
	{
		printf("Couldn't create shader to draw controllers\n");
		exit(1);
	}
	glVerify(mShaderMatrixLocation = glGetUniformLocation(mControllersShader, "matrix"));
	if (mShaderMatrixLocation == -1)
	{
		printf("Couldn't get required uniform to draw controllers\n");
		exit(1);
	}

	for (size_t i = 0; i < mControllers.size(); ++i)
	{
		Controller& curController = mControllers[i];

		std::string modelName = GetTrackedDeviceString(mOpenVR, curController.id, vr::Prop_RenderModelName_String);
		if (modelName.empty())
		{
			continue;
		}

		ControllerModel* model = GetOrCreateControllerModel(modelName);
		if (!model)
		{
			std::string deviceName = GetTrackedDeviceString(mOpenVR, curController.id, vr::Prop_TrackingSystemName_String);
			printf("Couldn't load model for tracked device %d, \"%s\"\n", curController.id, deviceName.c_str());
			continue;;
		}

		curController.model = model;
	}
}

void VrSystem::CreateVrRenderTarget(vr::EVREye eye)
{
	glVerify(glGenTextures(1, &mEyeResources[eye].textureId));
	glVerify(glBindTexture(GL_TEXTURE_2D, mEyeResources[eye].textureId));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mRecommendedRtWidth, mRecommendedRtHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));

	glVerify(glGenFramebuffers(1, &mEyeResources[eye].framebufferId));
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, mEyeResources[eye].framebufferId));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mEyeResources[eye].textureId, 0));

	glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

	glVerify(glDrawBuffer(GL_COLOR_ATTACHMENT0));
	glVerify(glReadBuffer(GL_COLOR_ATTACHMENT0));
}

void VrSystem::RenderControllers(const Matrix44& projView)
{
	glVerify(glUseProgram(mControllersShader));

	for (size_t i = 0; i < mControllers.size(); ++i)
	{
		Controller& curController = mControllers[i];
		if (!curController.valid)
		{
			continue;
		}
		
		const Matrix44 transformMatrix = projView  * TranslationMatrix(Point3(curController.state.pos)) * curController.state.orientation;

		glVerify(glUniformMatrix4fv(mShaderMatrixLocation, 1, GL_FALSE, transformMatrix));
		curController.model->Render();
	}
	glVerify(glUseProgram(0));
}

VrSystem* VrSystem::Create(Vec3 anchorPos, const Matrix44& anchorRotation, float moveScale)
{
	std::unique_ptr<VrSystem> result(new (std::nothrow) VrSystem(anchorPos, anchorRotation, moveScale));

	if (!result.get())
	{
		return nullptr;
	}

	if (!result->InitSystem())
	{
		return nullptr;
	}

	return result.release();
}

void VrSystem::GetProjectionMatrixAndEyeOffset(unsigned eye, float zNear, float zFar, Matrix44& projMatrix, Vec3& eyeOffset)
{
	const vr::EVREye vrEye = static_cast<vr::EVREye>(eye);
	vr::HmdMatrix44_t mat = mOpenVR->GetProjectionMatrix(vrEye, zNear, zFar);

	// transposing the result so that it works for us
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			projMatrix.columns[i][j] = mat.m[j][i];
		}
	}
	eyeOffset = mEyeOffset[vrEye];
}

VrSystem::VrSystem(Vec3 anchorPos, const Matrix44& anchorRotation, float moveScale)
	: mOpenVR(nullptr)
	, mRecommendedRtWidth(0)
	, mRecommendedRtHeight(0)
	, mHmdId(vr::k_unTrackedDeviceIndexInvalid)
	, mMoveScale(moveScale)
	, mAnchorPos(anchorPos)
	, mValidInitialHmdPos(false)
	, mControllersShader(0)
	, mShaderMatrixLocation(-1)
{
	SetInverseInitialHmdRotation(Matrix44::kIdentity);
	SetAnchorRotation(anchorRotation);
	mHmdRotation = Matrix44::kIdentity;
	mControllers.reserve(3);
}

void VrSystem::DestroyVrRenderTarget(vr::EVREye eye)
{
	if (mEyeResources[eye].framebufferId)
	{
		glVerify(glDeleteFramebuffers(1, &mEyeResources[eye].framebufferId));
		mEyeResources[eye].framebufferId = 0;
	}

	if (mEyeResources[eye].textureId)
	{
		glVerify(glDeleteTextures(1, &mEyeResources[eye].textureId));
		mEyeResources[eye].textureId = 0;
	}
}

void VrSystem::Destroy()
{
	mControllers.clear();

	for (auto& pair : mControllerModels)
	{
		delete pair.second;
	}
	mControllerModels.clear();

	for (int i = 0; i < 2; ++i)
	{
		DestroyVrRenderTarget(static_cast<vr::EVREye>(i));
	}

	if (mOpenVR)
	{
		vr::VR_Shutdown();
		mOpenVR = nullptr;
	}
}

VrSystem::~VrSystem()
{
	Destroy();
}

void VrSystem::InitEyeOffset(vr::EVREye eye)
{
	const vr::HmdMatrix34_t mat = mOpenVR->GetEyeToHeadTransform(eye);
	mEyeOffset[eye] = Vec3(mat.m[0][3], mat.m[1][3], mat.m[2][3]);
}

VrSystem::ControllerModel* VrSystem::GetOrCreateControllerModel(const std::string& modelName)
{
	const auto requiredPair = mControllerModels.find(modelName);
	if (requiredPair != mControllerModels.end())
	{
		return requiredPair->second;
	}

	std::unique_ptr<ControllerModel> model(new (std::nothrow) ControllerModel());
	if (model.get())
	{
		std::unique_ptr<vr::RenderModel_t, void(*)(vr::RenderModel_t*)> renderModel(nullptr, [](vr::RenderModel_t* ptr) { if (ptr) vr::VRRenderModels()->FreeRenderModel(ptr); });
		{
			vr::RenderModel_t *tmpModel = nullptr;
			while (true)
			{
				vr::EVRRenderModelError err = vr::VRRenderModels()->LoadRenderModel_Async(modelName.c_str(), &tmpModel);
				if (err == vr::VRRenderModelError_None)
				{
					break;
				}
				if (err != vr::VRRenderModelError_Loading)
				{
					printf("Couldn't get model \"%s\" from OpenVR\n", modelName.c_str());
					return nullptr;
				}
				Sleep(0.001);
			}
			renderModel.reset(tmpModel);
		}

		std::unique_ptr<vr::RenderModel_TextureMap_t, void(*)(vr::RenderModel_TextureMap_t*)> renderTexture(nullptr, [](vr::RenderModel_TextureMap_t* ptr) { if (ptr) vr::VRRenderModels()->FreeTexture(ptr); });
		{
			vr::RenderModel_TextureMap_t *tmpTexture = nullptr;
			while (true)
			{
				vr::EVRRenderModelError err = vr::VRRenderModels()->LoadTexture_Async(renderModel->diffuseTextureId, &tmpTexture);
				if (err == vr::VRRenderModelError_None)
				{
					break;
				}
				if (err != vr::VRRenderModelError_Loading)
				{
					printf("Couldn't get texture for model \"%s\" from OpenVR\n", modelName.c_str());
					return nullptr;
				}
				Sleep(0.001);
			}
			renderTexture.reset(tmpTexture);
		}
		if (!model->CreateModelResources(*renderModel, *renderTexture))
		{
			return nullptr;
		}
		// Note: unique ptr releases ownership in the return
		mControllerModels[modelName] = model.get();
	}
	return model.release();
}

bool VrSystem::InitSystem()
{
	vr::EVRInitError error = vr::VRInitError_None;

	mOpenVR = VR_Init(&error, vr::VRApplication_Scene);
	if (error != vr::VRInitError_None)
	{
		printf("Couldn't init OpenVR runtime: %s\n", VR_GetVRInitErrorAsEnglishDescription(error));
		return false;
	}

	if (!vr::VRCompositor())
	{
		printf("Couldn't init VRCompositor\n");
		return false;
	}

	vr::VRCompositor()->SetTrackingSpace(vr::TrackingUniverseStanding);

	mOpenVR->GetRecommendedRenderTargetSize(&mRecommendedRtWidth, &mRecommendedRtHeight);

	for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i)
	{
		if (!mOpenVR->IsTrackedDeviceConnected(i))
		{
			continue;
		}

		vr::ETrackedDeviceClass deviceClass = mOpenVR->GetTrackedDeviceClass(i);

		if (deviceClass == vr::TrackedDeviceClass_HMD)
		{
			mHmdId = i;
		}

		if (deviceClass == vr::TrackedDeviceClass_Controller)
		{
			Controller ctrl;
			ctrl.id = i;
			for (uint32_t j = 0; j < vr::k_unControllerStateAxisCount; ++j)
			{
				const int32_t axisType = mOpenVR->GetInt32TrackedDeviceProperty(i, static_cast<vr::ETrackedDeviceProperty>(vr::Prop_Axis0Type_Int32 + j));
				if (axisType == vr::k_eControllerAxis_Trigger)
				{
					ctrl.triggerId = j;
				}
			}
			mControllers.push_back(ctrl);
		}
	}

	if (mHmdId == vr::k_unTrackedDeviceIndexInvalid)
	{
		printf("No HMD found\n");
		return false;
	}

	for (int i = 0; i < 2; ++i)
	{
		InitEyeOffset(static_cast<vr::EVREye>(i));
	}

	return true;
}

bool VrSystem::GetTransformation(vr::TrackedDevicePose_t* devicePose, Vec3& pos, Matrix44& rotation, bool applyHmdRotationCorrection)
{
	if (!(devicePose->bDeviceIsConnected && devicePose->bPoseIsValid))
	{
		return false;
	}

	const vr::HmdMatrix34_t& mat = devicePose->mDeviceToAbsoluteTracking;

	pos[0] = mat.m[0][3];
	pos[1] = mat.m[1][3];
	pos[2] = mat.m[2][3];

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			rotation.columns[i][j] = mat.m[j][i];
		}
	}

	rotation.columns[0][3] = 0;
	rotation.columns[1][3] = 0;
	rotation.columns[2][3] = 0;
	rotation.columns[3][3] = 1;

	if (applyHmdRotationCorrection)
	{
		rotation = ApplyHmdRotationCorrection(rotation);
	}

	return true;
}

VrSystem::ControllerModel::~ControllerModel()
{
	Destroy();
}

bool VrSystem::ControllerModel::CreateModelResources(const vr::RenderModel_t& model, const vr::RenderModel_TextureMap_t& texture)
{
	glVerify(glGenVertexArrays(1, &mVertexArray));
	glVerify(glBindVertexArray(mVertexArray));

	glVerify(glGenBuffers(1, &mVertexBuffer));
	glVerify(glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer));
	glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(vr::RenderModel_Vertex_t) * model.unVertexCount, model.rVertexData, GL_STATIC_DRAW));

	glVerify(glEnableVertexAttribArray(0));
	glVerify(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t), reinterpret_cast<void *>(offsetof(vr::RenderModel_Vertex_t, vPosition))));
	glVerify(glEnableVertexAttribArray(1));
	glVerify(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t), reinterpret_cast<void *>(offsetof(vr::RenderModel_Vertex_t, vNormal))));
	glVerify(glEnableVertexAttribArray(2));
	glVerify(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vr::RenderModel_Vertex_t), reinterpret_cast<void *>(offsetof(vr::RenderModel_Vertex_t, rfTextureCoord))));

	glVerify(glGenBuffers(1, &mIndexBuffer));
	glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer));
	glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint16_t) * model.unTriangleCount * 3, model.rIndexData, GL_STATIC_DRAW));

	glVerify(glBindVertexArray(0));

	glVerify(glGenTextures(1, &mTexture));
	glVerify(glBindTexture(GL_TEXTURE_2D, mTexture));
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.unWidth, texture.unHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture.rubTextureMapData));

	glVerify(glGenerateMipmap(GL_TEXTURE_2D));

	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));

	GLfloat maxAnisotropy;
	glVerify(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy));
	glVerify(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAnisotropy));

	glVerify(glBindTexture(GL_TEXTURE_2D, 0));

	if (glGetError() != GL_NO_ERROR)
	{
		printf("Error during creation of GPU resources for the controller model\n");
		return false;
	}

	mNumIndices = model.unTriangleCount * 3;

	return true;
}

void VrSystem::ControllerModel::Destroy()
{
	if (mTexture)
	{
		glVerify(glDeleteTextures(1, &mTexture));
		mTexture = 0;
	}
	if (mIndexBuffer)
	{
		glVerify(glDeleteBuffers(1, &mIndexBuffer));
		mIndexBuffer = 0;
	}
	if (mVertexBuffer)
	{
		glVerify(glDeleteBuffers(1, &mVertexBuffer));
		mVertexBuffer = 0;
	}
	if (mVertexArray)
	{
		glVerify(glDeleteVertexArrays(1, &mVertexArray));
		mVertexArray = 0;
	}
}

void VrSystem::ControllerModel::Render() const
{
	glVerify(glBindVertexArray(mVertexArray));

	glVerify(glActiveTexture(GL_TEXTURE0));
	glVerify(glBindTexture(GL_TEXTURE_2D, mTexture));

	glVerify(glDrawElements(GL_TRIANGLES, mNumIndices, GL_UNSIGNED_SHORT, 0));

	glVerify(glBindVertexArray(0));
}

#endif // #if FLEX_VR
