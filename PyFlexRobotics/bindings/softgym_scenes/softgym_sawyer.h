#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"
#include "softgym_robot.h"


char* make_path(char* full_path, std::string path);

class SoftgymSawyer: public SoftgymRobotBase
{
public:
    URDFImporter* urdf;

    vector<Transform> rigidTrans;
    map<string, int> jointMap;
    map<string, int> activeJointMap;
    int effectorJoint;

    int fingerLeft;
    int fingerRight;
    float fingerWidth = 0.03f;
    float fingerWidthMin = 0.0f;
    float fingerWidthMax = 0.05f;
    float roll, pitch, yaw;
    std::vector<float*> ptrRotation{&roll, &pitch, &yaw};

    float numpadJointTraSpeed = 0.1f / 60.f; // 10cm/s under 60 fps
    float numpadJointRotSpeed = 10 / 60.f; // 10 deg/s under 60 fps
    float numpadJointRotDir = 1.f; // direction of rotation
    float numpadFingerSpeed = 0.02f / 60.f; // 2cm/s under 60 fps
    char urdfPath[100];

    std::vector<float> forceLeft;
    std::vector<float> forceRight;

    SoftgymSawyer(){}

    void Initialize(py::array_t<float> robot_params = py::array_t<float>())
    {
        cout<<"InitializeRobot"<<endl;
        auto ptr = (float *) robot_params.request().ptr;
        g_numSubsteps = 4;
		g_params.numIterations = 30;
		g_params.numPostCollisionIterations = 10;

		g_params.shapeCollisionMargin = 0.04f;

		roll = 0.0f;
		pitch = 0.0f;
		yaw = 180.0f;
        rigidTrans.clear();

        urdf = new URDFImporter(make_path(urdfPath, "/data/sawyer"), "/sawyer_description/urdf/sawyer_with_gripper.urdf", false,  0.005f, 0.005f, true, 20, false); // sawyer_with_gripper.urdf

		Transform gt(Vec3(0.0f, 0.925f, -2.7f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) // was -0.7
					* QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

        urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 1000.0f, 0.0f, 1e1f, 0.01f, 20.7f, 7.0f, false, 1e-7f, 10.0f, 0);

		for (int i = 0; i < (int)urdf->joints.size(); i++)
        {
            URDFJoint* j = urdf->joints[i];
            NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[j->name]];
            if (j->type == URDFJoint::REVOLUTE)
            {
                joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-8f;	// 10^6 N/m
                joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+3f;	// 5*10^5 N/m/s
            }
            else if (j->type == URDFJoint::PRISMATIC)
            {
                joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
                joint.targets[eNvFlexRigidJointAxisX] = 0.02f;
                joint.compliance[eNvFlexRigidJointAxisX] = 1.e-8f;
                joint.damping[eNvFlexRigidJointAxisX] = 0.0f;
            }
        }
        // fix base in place, todo: add a kinematic body flag?
        g_buffers->rigidBodies[0].invMass = 0.0f;
        (Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();
        fingerLeft = urdf->jointNameMap["r_gripper_l_finger_joint"];
        fingerRight = urdf->jointNameMap["r_gripper_r_finger_joint"];
		fingerWidthMin = 0.002f;

        NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["right_endpoint"]];

        // set up end effector targets
        NvFlexRigidJoint effectorJoint0;
        Quat q = rpy2quat(roll*kPi / 180.0f, pitch*kPi / 180.0f, yaw*kPi / 180.0f);
        NvFlexMakeFixedJoint(&effectorJoint0, -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(-0.2f, 0.8f, 0.0f), q), NvFlexMakeRigidPose(0,0));
        for (int i = 0; i < 6; ++i)
        {
            effectorJoint0.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
            effectorJoint0.damping[i] = 1.e+3f;
            //effectorJoint0.maxIterations = 30;
        }

        effectorJoint = g_buffers->rigidJoints.size();
        g_buffers->rigidJoints.push_back(effectorJoint0);

		forceLeft.resize(0);
		forceRight.resize(0);

		g_numSubsteps = 4;
		g_params.numIterations = 30;

		g_params.dynamicFriction = 1.25f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
//		g_params.sleepThreshold = 0.02f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);
//		g_drawPoints = false;

        const int headLink = urdf->rigidNameMap["head_tilt_link"];
        bool hasFluids = false;
        DepthRenderProfile p = {
			0.f, // minRange
			5.f // maxRange
		};

        AddSensor(g_screenWidth, g_screenHeight,  0,  Transform(Vec3(0.0f, 3.f, 3.5f), rpy2quat(2.7415926f, 0.0f, 0.0f)),  DegToRad(60.f), hasFluids, p);
    }

    virtual py::array_t<float> GetState()
    {
        g_buffers->rigidJoints.map();
        auto robot_state = py::array_t<float>((size_t) (int)g_buffers->rigidJoints.size() * 7);
        auto ptr = (float *) robot_state.request().ptr;
        for (int i = 0; i < (int) g_buffers->rigidJoints.size(); i++)
        {
            NvFlexRigidJoint& joint = g_buffers->rigidJoints[i];
            ptr[i * 7] = joint.pose0.p[0];
            ptr[i * 7 + 1] = joint.pose0.p[1];
            ptr[i * 7 + 2] = joint.pose0.p[2];
            ptr[i * 7 + 3] = joint.pose0.q[0];
            ptr[i * 7 + 4] = joint.pose0.q[1];
            ptr[i * 7 + 5] = joint.pose0.q[2];
            ptr[i * 7 + 6] = joint.pose0.q[3];
        }
        g_buffers->rigidJoints.unmap();

        return robot_state;
    }

    virtual void SetState(py::array_t<float> robot_state = py::array_t<float>())
    {
        g_buffers->rigidJoints.map();
        auto ptr = (float *) robot_state.request().ptr;
        for (int i = 0; i < (int) g_buffers->rigidJoints.size(); i++)
        {
            NvFlexRigidJoint& joint = g_buffers->rigidJoints[i];
            joint.pose0.p[0] = ptr[i * 7];
            joint.pose0.p[1] = ptr[i * 7 + 1];
            joint.pose0.p[2] = ptr[i * 7 + 2];
            joint.pose0.q[0] = ptr[i * 7 + 3];
            joint.pose0.q[1] = ptr[i * 7 + 4];
            joint.pose0.q[2] = ptr[i * 7 + 5];
            joint.pose0.q[3] = ptr[i * 7 + 6];
        }

        g_buffers->rigidJoints.unmap();
        NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());

        int resetNumSteps = 100; // TODO: XY: Right now this step does not work as expected. Not sure what is the issue.
        for (int s = 0; s < resetNumSteps; s++) { NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);}

        return robot_state;
    }

    virtual void DoGui()
    {
        NvFlexRigidJoint effector0 = g_buffers->rigidJoints[effectorJoint];

        float targetx = effector0.pose0.p[0];
        float targety = effector0.pose0.p[1];
        float targetz = effector0.pose0.p[2];

		float oroll = roll;
		float opitch = pitch;
		float oyaw = yaw;
		imguiSlider("Gripper X", &targetx, -0.5f, 0.5f, 0.0001f);
        imguiSlider("Gripper Y", &targety, 0.0f, 1.5f, 0.0001f);
        imguiSlider("Gripper Z", &targetz, -0.5f, 1.2f, 0.0001f);
		imguiSlider("Roll", &roll, -180.0f, 180.0f, 0.01f);
		imguiSlider("Pitch", &pitch, -180.0f, 180.0f, 0.01f);
		imguiSlider("Yaw", &yaw, -180.0f, 180.0f, 0.01f);
		float f = 0.1f;

		roll = Lerp(oroll, roll, f);
		pitch = Lerp(opitch, pitch, f);
		yaw = Lerp(oyaw, yaw, f);

        const float smoothing = 1.f;

        // low-pass filter controls otherwise it is too jerky
        float newx = Lerp(effector0.pose0.p[0], targetx, smoothing);
        float newy = Lerp(effector0.pose0.p[1], targety, smoothing);
        float newz = Lerp(effector0.pose0.p[2], targetz, smoothing);

        effector0.pose0.p[0] = newx;
        effector0.pose0.p[1] = newy;
        effector0.pose0.p[2] = newz;

		Quat q = rpy2quat(roll*kPi / 180.0f, pitch*kPi / 180.0f, yaw*kPi / 180.0f);
		effector0.pose0.q[0] = q.x;
		effector0.pose0.q[1] = q.y;
		effector0.pose0.q[2] = q.z;
		effector0.pose0.q[3] = q.w;

        g_buffers->rigidJoints[effectorJoint] = effector0;

        float newWidth = fingerWidth;
        imguiSlider("Finger Width", &newWidth, fingerWidthMin, fingerWidthMax, 0.001f);

        fingerWidth = Lerp(fingerWidth, newWidth, smoothing);

        g_buffers->rigidJoints[fingerLeft].targets[eNvFlexRigidJointAxisX] = fingerWidth;
        g_buffers->rigidJoints[fingerRight].targets[eNvFlexRigidJointAxisX] = fingerWidth;
    }

//    virtual void DoStats()
//    {
//        int numSamples = 200;
//
//        int start = Max(int(forceLeft.size())-numSamples, 0);
//        int end = Min(start + numSamples, int(forceLeft.size()));
//
//        // convert from position changes to forces
//        float units = -1.0f/Sqr(g_dt/g_numSubsteps);
//
//        float height = 50.0f;
//        float maxForce = 10.0f;
//
//        float dx = 1.0f;
//        float sy = height/maxForce;
//
//        float lineHeight = 10.0f;
//
//        float rectMargin = 10.0f;
//        float rectWidth = dx * float(numSamples) + rectMargin * 4.0f;
//
//        float x = float(g_screenWidth) - rectWidth - 20.0f;
//        float y = 300.0f;
//
//        DrawRect(x, y - height - rectMargin, rectWidth, 2.0f * height + rectMargin * 3.0f, Vec4(0.0f, 0.0f, 0.0f, 0.5f));
//
//        x += rectMargin * 3.0f;
//
//        DrawImguiString(int(x + dx * float(numSamples)), int(y + 55.0f), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");
//
//        DrawLine(x, y, x + float(numSamples) * dx, y, 1.0f, Vec3(1.0f));
//        DrawLine(x, y -50.0f, x, y + 50.0f, 1.0f, Vec3(1.0f));
//
//        float margin = 5.0f;
//
//        DrawImguiString(int(x - margin), int(y), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
//        DrawImguiString(int(x - margin), int(y + height - lineHeight), Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
//        DrawImguiString(int(x - margin), int(y - height), Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);
//
//        for (int i = start; i < end - 1; ++i)
//        {
//        	float fl0 = Clamp(forceLeft[i]*units, -maxForce, maxForce)*sy;
//        	float fr0 = Clamp(forceRight[i]*units, -maxForce, maxForce)*sy;
//
//        	float fl1 = Clamp(forceLeft[i+1]*units, -maxForce, maxForce)*sy;
//        	float fr1 = Clamp(forceRight[i+1]*units, -maxForce, maxForce)*sy;
//
//        	DrawLine(x, y + fl0, x + dx, y + fl1, 1.0f, Vec3(1.0f, 0.0f, 0.0f));
//        	DrawLine(x, y + fr0, x + dx, y + fr1, 1.0f, Vec3(0.0f, 1.0f, 0.0f));
//
//        	x += dx;
//        }
//    }


    virtual void Step(py::array_t<float> control_params = py::array_t<float>())
    {
        if (control_params.size()==1) return; // The default nullptr. Not sure why.
        // Control parameters: dx, dy, dz in cartisian space and roll, pitch, yaw, finally the open/close of the gripper
        auto ptr = (float *) control_params.request().ptr;
        // record force on finger joints
        forceLeft.push_back(g_buffers->rigidJoints[fingerLeft].lambda[eNvFlexRigidJointAxisX]);
        forceRight.push_back(g_buffers->rigidJoints[fingerRight].lambda[eNvFlexRigidJointAxisX]);

		// move end-effector given the control parameters
		NvFlexRigidJoint joint = g_buffers->rigidJoints[effectorJoint];
		for (int i=0; i<3; ++i) joint.pose0.p[i] += ptr[i]; // dx, dy, dz
//        std::cout<<"joint pose in Step: " <<joint.pose0.p[1]<<std::endl;
		Quat deltaRot;
		Quat currentRot = Quat(joint.pose0.q);
		for (int i=0; i<3; ++i) // roll, pitch, yaw
		{
		    Vec3 axis(0.f);
		    axis[i] = 1;
		    deltaRot = QuatFromAxisAngle(axis, ptr[i + 3]);
		    currentRot = deltaRot * currentRot;
            *(ptrRotation[i]) += ptr[i+3];
		}
		joint.pose0.q[0] = currentRot.x;
		joint.pose0.q[1] = currentRot.y;
		joint.pose0.q[2] = currentRot.z;
		joint.pose0.q[3] = currentRot.w;

		fingerWidth += ptr[6];
		fingerWidth = max(min(fingerWidth, fingerWidthMax), fingerWidthMin);
		g_buffers->rigidJoints[fingerLeft].targets[eNvFlexRigidJointAxisX] = fingerWidth;
        g_buffers->rigidJoints[fingerRight].targets[eNvFlexRigidJointAxisX] = fingerWidth;

		g_buffers->rigidJoints[effectorJoint] = joint;
//        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }

    virtual void Update()
    {
        // record force on finger joints
        forceLeft.push_back(g_buffers->rigidJoints[fingerLeft].lambda[eNvFlexRigidJointAxisX]);
        forceRight.push_back(g_buffers->rigidJoints[fingerRight].lambda[eNvFlexRigidJointAxisX]);

		// move end-effector via numpad
		NvFlexRigidJoint joint = g_buffers->rigidJoints[effectorJoint];
		// x
		if (g_numpadPressedState[SDLK_KP_4])
		{
			joint.pose0.p[0] -= numpadJointTraSpeed;
		}
		if (g_numpadPressedState[SDLK_KP_6])
		{
			joint.pose0.p[0] += numpadJointTraSpeed;
		}
		// y
		if (g_numpadPressedState[SDLK_KP_9])
		{
			joint.pose0.p[1] += numpadJointTraSpeed;
		}
		if (g_numpadPressedState[SDLK_KP_7])
		{
			joint.pose0.p[1] -= numpadJointTraSpeed;
		}
		// z
		if (g_numpadPressedState[SDLK_KP_5])
		{
			joint.pose0.p[2] += numpadJointTraSpeed;
		}
		if (g_numpadPressedState[SDLK_KP_8])
		{
			joint.pose0.p[2] -= numpadJointTraSpeed;
		}
		// rpy
		Quat currentRot = Quat(joint.pose0.q);
		if (g_numpadPressedState[SDLK_KP_0])
		{
			numpadJointRotDir *= -1;
		}
		if (g_numpadPressedState[SDLK_KP_1])
		{
			float deltaRollAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaRoll = QuatFromAxisAngle(Vec3(1, 0, 0), deltaRollAngle);
			currentRot = deltaRoll * currentRot;
			roll += deltaRollAngle;
		}
		if (g_numpadPressedState[SDLK_KP_2])
		{
			float deltaYawAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaYaw = QuatFromAxisAngle(Vec3(0, 1, 0), deltaYawAngle);
			currentRot = deltaYaw * currentRot;
			yaw += deltaYawAngle;
		}
		if (g_numpadPressedState[SDLK_KP_3])
		{
			float deltaPitchAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaPitch = QuatFromAxisAngle(Vec3(0, 0, 1), deltaPitchAngle);
			currentRot = deltaPitch * currentRot;
			pitch += deltaPitchAngle;
		}
		if (g_numpadPressedState[SDLK_KP_PLUS])
		{
			fingerWidth = min(fingerWidth + numpadFingerSpeed, fingerWidthMax);
		}
		if (g_numpadPressedState[SDLK_KP_MINUS])
		{
			fingerWidth  = max(fingerWidth - numpadFingerSpeed, fingerWidthMin);
		}
		joint.pose0.q[0] = currentRot.x;
		joint.pose0.q[1] = currentRot.y;
		joint.pose0.q[2] = currentRot.z;
		joint.pose0.q[3] = currentRot.w;
		g_buffers->rigidJoints[effectorJoint] = joint;
    }

    virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }
};