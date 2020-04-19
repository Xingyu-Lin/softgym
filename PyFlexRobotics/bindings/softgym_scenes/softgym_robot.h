#pragma once

class SoftgymRobotBase
{
public:

	SoftgymRobotBase() {}
	virtual ~SoftgymRobotBase() {}

	virtual void Initialize(py::array_t<float> robot_params = py::array_t<float>()) = 0;

//	virtual void PostInitialize() {}

	// Called immediately after scene constructor (used by RL scenes to load parameters and launch Python scripts)
//	virtual void PrepareScene() {}

	// update any buffers (all guaranteed to be mapped here)
	virtual void Update() {}

	// called after update, can be used to read back any additional information that is not read back by the main application
	virtual void PostUpdate() {}

	// send any changes to flex (all buffers guaranteed to be unmapped here)
	virtual void Sync() {}

	virtual void Draw(int pass) {}
	virtual void KeyDown(int key) {}
	virtual void DoGui() {}
	virtual void DoStats() {} // draw on-screen graphs, text, etc

	virtual void CenterCamera() {}

	virtual Matrix44 GetBasis()
	{
		return Matrix44::kIdentity;
	}

	virtual bool IsSkipSimulation()
	{
		return false;
	}
};