#pragma once

#define FLEX_VR (defined _WIN32 && _WIN32)
//#define FLEX_VR 0

#include "../../core/maths.h"

struct VrControllerState
{
	Vec3 pos;
	Matrix44 orientation;
	float triggerValue;
	bool buttonPressed;
};



#if !FLEX_VR

#define FixMouseCoordinatesForVr(x, y)

class VrSystem
{
public:
	void InitGraphicalResources() {}

	// Update must be called prior frame uploading to the VR device
	void Update() {}
	void UploadGraphicsToDevice() const {}

	Vec3 GetHmdPos() const { return Vec3(); }
	void SetAnchorPos(Vec3 newCamPos) {}

	Matrix44 GetHmdRotation() const { return Matrix44(); }
	Matrix44 GetHmdRotationInverse() const { return Matrix44(); }
	void SetAnchorRotation(const Matrix44& anchorRotation) {}

	void RenderControllers(const Matrix44& projView) {}

	size_t GetNumControllers() const { return 0; }
	bool GetControllerState(size_t controllerIndex, VrControllerState& resultState) { return false; }

	uint32_t GetRecommendedRtWidth() const { return 0; }
	uint32_t GetRecommendedRtHeight() const { return 0; }
	size_t GetEyeFrameBuffer(unsigned eye) const { return 0; }
	void GetProjectionMatrixAndEyeOffset(unsigned eye, float zNear, float zFar, Matrix44& projMatrix, Vec3& eyeOffset) {}
};

VrSystem* const g_vrSystem = nullptr;

#else // FLEX_VR

#include "../../external/openvr/headers/openvr.h"
#include "../../external/glad/include/glad/glad.h"
#include <vector>
#include <map>

class VrSystem
{
public:
	static const size_t InvalidControllerIndex = vr::k_unTrackedDeviceIndexInvalid;

	static VrSystem* Create(Vec3 anchorPos, const Matrix44& anchorRotation, float moveScale);

	virtual ~VrSystem();
	void InitGraphicalResources();

	// Update must be called prior frame uploading to the VR device
	void Update();
	void UploadGraphicsToDevice() const;

	Vec3 GetHmdPos() const { return GetPosInScene(mHmdPos); }
	void SetAnchorPos(Vec3 newCamPos) { mAnchorPos = newCamPos; }
	
	Matrix44 GetHmdRotation() const { return mHmdRotation; }
	Matrix44 GetHmdRotationInverse() const { return InverseRotation(mHmdRotation); }
	void SetAnchorRotation(const Matrix44& anchorRotation)
	{
		mAnchorRotation = anchorRotation;
		mHmdRotationCorrection = mAnchorRotation * mInverseInitialHmdRotation;
	}

	void RenderControllers(const Matrix44& projView);

	size_t GetNumControllers() const { return mControllers.size(); }
	bool GetControllerState(size_t controllerIndex, VrControllerState& resultState);

	uint32_t GetRecommendedRtWidth() const { return mRecommendedRtWidth; }
	uint32_t GetRecommendedRtHeight() const { return mRecommendedRtHeight; }
	size_t GetEyeFrameBuffer(unsigned eye) const { return mEyeResources[eye].framebufferId; }
	void GetProjectionMatrixAndEyeOffset(unsigned eye, float zNear, float zFar, Matrix44& projMatrix, Vec3& eyeOffset);

private:
	class ControllerModel
	{
	public:
		ControllerModel()
			: mVertexArray(0)
			, mVertexBuffer(0)
			, mIndexBuffer(0)
			, mTexture(0)
			, mNumIndices(0)
		{}

		~ControllerModel();

		bool CreateModelResources(const vr::RenderModel_t& model, const vr::RenderModel_TextureMap_t& texture);
		void Destroy();
		void Render() const;

	private:
		GLuint mVertexArray;
		GLuint mVertexBuffer;
		GLuint mIndexBuffer;
		GLuint mTexture;
		uint32_t mNumIndices;
	};

	struct GlEyeResources
	{
		GLuint textureId;
		GLuint framebufferId;

		GlEyeResources()
			: textureId(0)
			, framebufferId(0)
		{}
	};

	struct Controller
	{
		uint32_t id;
		ControllerModel* model;
		bool valid;
		VrControllerState state;
		int triggerId;

		Controller()
			: id(vr::k_unTrackedDeviceIndexInvalid)
			, model(nullptr)
			, valid(false)
			, triggerId(-1)
		{
			memset(&state, 0, sizeof(state));
		}
	};

	VrSystem(Vec3 anchorPos, const Matrix44& anchorRotation, float moveScale);
	void DestroyVrRenderTarget(vr::EVREye eye);
	void Destroy();
	void InitEyeOffset(vr::EVREye eye);
	void CreateVrRenderTarget(vr::EVREye eye);
	bool InitSystem();
	bool GetTransformation(vr::TrackedDevicePose_t* devicePose, Vec3& pos, Matrix44& rotation, bool applyHmdRotationCorrection = true);
	ControllerModel* GetOrCreateControllerModel(const std::string& modelName);

	void SetInverseInitialHmdRotation(const Matrix44& rotation)
	{
		mInverseInitialHmdRotation = rotation;
		mHmdRotationCorrection = mAnchorRotation * mInverseInitialHmdRotation;
	}

	Matrix44 ApplyHmdRotationCorrection(const Matrix44& rotation) const { return mHmdRotationCorrection * rotation; }
	
	Vec3 GetPosInScene(Vec3 vrPos) const
	{
		return mValidInitialHmdPos ? mAnchorPos + mHmdRotationCorrection * (vrPos - mInitialHmdPos) * mMoveScale : mAnchorPos;
	}

	static Matrix44 InverseRotation(const Matrix44& rotation)
	{
		Matrix44 result;
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				result.columns[i][j] = rotation.columns[j][i];
			}
			result.columns[i][3] = 0;
			result.columns[3][i] = 0;
		}
		result.columns[3][3] = 1;
		return result;
	}

	vr::IVRSystem* mOpenVR;
	uint32_t mRecommendedRtWidth;
	uint32_t mRecommendedRtHeight;
	uint32_t mHmdId;

	Vec3 mEyeOffset[2];

	float mMoveScale;

	Vec3 mAnchorPos; // Use SetAnchorPos to assign value
	Matrix44 mAnchorRotation;

	Vec3 mInitialHmdPos;
	Matrix44 mInverseInitialHmdRotation; // Use SetInverseInitialHmdRotation to assign value
	bool mValidInitialHmdPos;
	Vec3 mHmdPos;
	Matrix44 mHmdRotation;
	GlEyeResources mEyeResources[2];

	Matrix44 mHmdRotationCorrection;

	std::vector<Controller> mControllers;
	std::map<std::string, ControllerModel*> mControllerModels;

	GLuint mControllersShader;
	GLint mShaderMatrixLocation;
};

extern VrSystem* g_vrSystem;
const float g_vrMoveScale = 1.f;

extern int g_windowWidth;
extern int g_windowHeight;
extern int g_screenWidth;
extern int g_screenHeight;

inline void FixMouseCoordinatesForVr(int& mouseX, int& mouseY)
{
	if (!g_vrSystem)
	{
		return;
	}
	mouseX = mouseX * g_screenWidth / g_windowWidth;
	mouseY = mouseY * g_screenHeight / g_windowHeight;
}

#endif // #if FLEX_VR
