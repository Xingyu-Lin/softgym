#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"

float InToM(float x){
	return x * 0.0254f;
}

class RigidBaxter : public Scene
{
public:

	enum Mode
	{
		eConnect4
	};

	Mode mode;

	URDFImporter* urdf;

	vector<Transform> rigidTrans;
	map<string, int> jointMap;
	map<string, int> activeJointMap;
	int left_effectorJoint;
	int right_effectorJoint;

	int left_fingerLeft;
	int left_fingerRight;
	int right_fingerLeft;
	int right_fingerRight;
	float left_fingerWidth = 0.025f;
	float right_fingerWidth = 0.025f;
	float fingerWidthMin = -.004f;
	float fingerWidthMax = 0.05f;

	//float roll, pitch, yaw;
	Vec3 left_rpy, right_rpy;
	bool hasFluids = false;
	float numpadJointTraSpeed = 0.05f / 60.f; // 5cm/s under 60 fps
	float numpadJointRotSpeed = 10 / 60.f; // 10 deg/s under 60 fps
	float numpadJointRotDir = 1.f; // direction of rotation
	float numpadFingerSpeed = 0.01f / 60.f; // 1cm/s under 60 fps
	float maxForce = 100.0f;

	RigidBaxter(Mode mode) : mode(mode)
	{
		left_rpy.x = 0.0f;
		left_rpy.y = 0.0f;
		left_rpy.z = -180.0f;

		right_rpy.x = 0.0f;
		right_rpy.y = 0.0f;
		right_rpy.z = -180.0f;
		rigidTrans.clear();

		urdf = new URDFImporter("../../data/baxter_common-master/", "baxter_description/urdf/baxter_grippers.urdf", true, 0, 0);

		Transform gt(Vec3(0.0f, 0.94f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi * 0.5f) 
											* QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi * 0.5f));

		// hide collision shapes
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);

		urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 1000.0f, 0, 1e1f, 0.01f, 20.7f, 7.0f, false, 1e-7f, 10.0f, 0);

		for (int i = 0; i < (int)urdf->joints.size(); i++)
		{
			URDFJoint* j = urdf->joints[i];
			NvFlexRigidJoint& joint = g_buffers->rigidJoints[urdf->jointNameMap[j->name]];
			if (j->type == URDFJoint::REVOLUTE)
			{
				joint.compliance[eNvFlexRigidJointAxisTwist] = 1.e-7f;	// 10^6 N/m
				joint.damping[eNvFlexRigidJointAxisTwist] = 1.e+1f;	// 5*10^5 N/m/s
			}
			else if (j->type == URDFJoint::PRISMATIC)
			{
				joint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModePosition;
				//joint.targets[eNvFlexRigidJointAxisX] = 0.02f;
				joint.compliance[eNvFlexRigidJointAxisX] = 1.e-7f;
				joint.damping[eNvFlexRigidJointAxisX] = 0.0f;

			}
		}
		// fix base in place, todo: add a kinematic body flag?
		g_buffers->rigidBodies[0].invMass = 0.0f;
		(Matrix33&)g_buffers->rigidBodies[0].invInertia = Matrix33();

		left_fingerLeft = urdf->jointNameMap["left_gripper_l_finger_joint"];
		left_fingerRight = urdf->jointNameMap["left_gripper_r_finger_joint"];

		NvFlexRigidJoint handLeft = g_buffers->rigidJoints[urdf->jointNameMap["left_endpoint"]];

		// set up end effector targets
		NvFlexRigidJoint left_effectorJoint0;
		NvFlexMakeFixedJoint(&left_effectorJoint0, -1, handLeft.body0, NvFlexMakeRigidPose(Vec3(0.25f, 1.05f, 0.6f), 
							QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi)), NvFlexMakeRigidPose(0, 0));
		for (int i = 0; i < 6; ++i)
		{
			left_effectorJoint0.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
			left_effectorJoint0.damping[i] = 1.e+1f;
			//left_effectorJoint0.maxIterations = 30;
		}

		left_effectorJoint = g_buffers->rigidJoints.size();
		g_buffers->rigidJoints.push_back(left_effectorJoint0);

		right_fingerLeft = urdf->jointNameMap["right_gripper_l_finger_joint"];
		right_fingerRight = urdf->jointNameMap["right_gripper_r_finger_joint"];

		NvFlexRigidJoint handRight = g_buffers->rigidJoints[urdf->jointNameMap["right_endpoint"]];

		int finger_indices[8] = { 24, 25, 26, 27, 42, 43, 44, 45 };
		for (int i = 0; i < 8; i++) {
			g_buffers->rigidShapes[finger_indices[i]].material.friction = 1.0;
			g_buffers->rigidShapes[finger_indices[i]].material.rollingFriction = 0.1;
			g_buffers->rigidShapes[finger_indices[i]].material.torsionFriction = 0.1;
		}

		// set up end effector targets
		NvFlexRigidJoint right_effectorJoint0;
		NvFlexMakeFixedJoint(&right_effectorJoint0, -1, handRight.body0, NvFlexMakeRigidPose(Vec3(-0.25f, 1.f, 0.6f), 
							QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi)), NvFlexMakeRigidPose(0, 0));
		for (int i = 0; i < 6; ++i)
		{
			right_effectorJoint0.compliance[i] = 1.e-4f;	// end effector compliance must be less than the joint compliance!
			right_effectorJoint0.damping[i] = 1.e+1f;
			//right_effectorJoint0.maxIterations = 30;
		}

		right_effectorJoint = g_buffers->rigidJoints.size();
		g_buffers->rigidJoints.push_back(right_effectorJoint0);

		// Make fingers heavier when ID control
		float mul = 4.0f;
		float imul = 1.0f / mul;

		g_buffers->rigidBodies[left_fingerLeft].mass *= mul;
		g_buffers->rigidBodies[left_fingerLeft].invMass *= imul;
		for (int k = 0; k < 9; k++)
		{
			g_buffers->rigidBodies[left_fingerLeft].inertia[k] *= mul;
			g_buffers->rigidBodies[left_fingerLeft].invInertia[k] *= imul;
		}

		g_buffers->rigidBodies[left_fingerRight].mass *= mul;
		g_buffers->rigidBodies[left_fingerRight].invMass *= imul;
		for (int k = 0; k < 9; k++)
		{
			g_buffers->rigidBodies[left_fingerRight].inertia[k] *= mul;
			g_buffers->rigidBodies[left_fingerRight].invInertia[k] *= imul;
		}

		g_buffers->rigidBodies[right_fingerLeft].mass *= mul;
		g_buffers->rigidBodies[right_fingerLeft].invMass *= imul;
		for (int k = 0; k < 9; k++)
		{
			g_buffers->rigidBodies[right_fingerLeft].inertia[k] *= mul;
			g_buffers->rigidBodies[right_fingerLeft].invInertia[k] *= imul;
		}

		g_buffers->rigidBodies[right_fingerRight].mass *= mul;
		g_buffers->rigidBodies[right_fingerRight].invMass *= imul;
		for (int k = 0; k < 9; k++)
		{
			g_buffers->rigidBodies[right_fingerRight].inertia[k] *= mul;
			g_buffers->rigidBodies[right_fingerRight].invInertia[k] *= imul;
		}

		// Table
		NvFlexRigidShape table;
		int table_index = g_buffers->rigidBodies.size();
		NvFlexMakeRigidBoxShape(&table, table_index, 0.4f, 0.01f, 0.2f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));
		table.filter = 0;
		//table.thickness = 0;// 0.0125f;
		table.material.friction = 0.7f;
		table.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.6f, 0.6f, 0.65f)));
		g_buffers->rigidShapes.push_back(table);
		float tableDensity = 1000.0f;
		NvFlexRigidBody tableBody;
		NvFlexMakeRigidBody(g_flexLib, &tableBody, Vec3(0.0f, .75f, .6f), Quat(), &table, &tableDensity, 1);

		g_buffers->rigidBodies.push_back(tableBody);

		//fix table in place, todo: add a kinematic body flag?
		g_buffers->rigidBodies[table_index].invMass = 0.0f;
		(Matrix33&)g_buffers->rigidBodies[table_index].invInertia = Matrix33();

		// Connect4 board
		std::vector<NvFlexRigidShape> board(16);
		int index = g_buffers->rigidBodies.size();
		//main board section
		//dimensions
		float b_l = 6.0f;
		float b_h = 5.0f;
		float b_d = 15.0f/16.0f;
		float b_th = 1.0f/8.0f;
		float b_hdepth = 3.0f+15.0f/16.0f;
		//float b_sl = 0.15748f;
		float ball_d = 5.0/8.0;
		//offsets
		float o_f = b_d / 2.0f - b_th/2.0f;

		float ball_shrink = .95; //clearance multiplier for ball
		//front back
		NvFlexMakeRigidBoxShape(&board[0], index, InToM(b_l)/2.0f, InToM(b_h)/2.0f, InToM(b_th)/2.0f, NvFlexMakeRigidPose(Vec3(0.0f, InToM(b_h)*.5f, InToM(o_f)), Quat()));
		NvFlexMakeRigidBoxShape(&board[1], index, InToM(b_l)/2.0f, InToM(b_h)/2.0f, InToM(b_th)/2.0f, NvFlexMakeRigidPose(Vec3(0.0f, InToM(b_h)*.5f, -InToM(o_f)), Quat()));
		//left slices
		NvFlexMakeRigidBoxShape(&board[2], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3(-(InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 0), InToM(b_h)*.5f, 0.0f), Quat()));
		NvFlexMakeRigidBoxShape(&board[3], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3(-(InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 1), InToM(b_h)*.5f, 0.0f), Quat()));
		NvFlexMakeRigidBoxShape(&board[4], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3(-(InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 2), InToM(b_h)*.5f, 0.0f), Quat()));
		NvFlexMakeRigidBoxShape(&board[5], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3(-(InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 3), InToM(b_h)*.5f, 0.0f), Quat()));
		//right slices
		NvFlexMakeRigidBoxShape(&board[6], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3((InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 0), InToM(b_h)*.5f, 0.0f), Quat()));
		NvFlexMakeRigidBoxShape(&board[7], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3((InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 1), InToM(b_h)*.5f, 0.0f), Quat()));
		NvFlexMakeRigidBoxShape(&board[8], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3((InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 2), InToM(b_h)*.5f, 0.0f), Quat()));
		NvFlexMakeRigidBoxShape(&board[9], index, InToM(b_th)/2.0f, InToM(b_h)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3((InToM(ball_d+ b_th)* .5f  + InToM(ball_d+ b_th) * 3), InToM(b_h)*.5f, 0.0f), Quat()));
		//bottom stop
		NvFlexMakeRigidBoxShape(&board[10], index, InToM(b_l)/2.0f, InToM(b_h-b_hdepth)/2.0f, InToM(b_d)/2.0f, NvFlexMakeRigidPose(Vec3(0.0f, InToM(b_h* 0.5f-b_hdepth/2.0f ), 0.0f), Quat()));

		float cb_l = 7.5f;
		float cb_d = 6.5f;
		float cb_h = 1.0f;
		float cb_th = 1.0f/8.0f;
		float cb_wth = 5.0/8.0f;
		//containing box
		NvFlexMakeRigidBoxShape(&board[11], index, InToM(cb_l)/2.0f, InToM(cb_th)/2.0f, InToM(cb_d)/2.0f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));

		NvFlexMakeRigidBoxShape(&board[12], index, InToM(cb_l)/2.0f, InToM(cb_h)/2.0f, InToM(cb_wth)/2.0f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, InToM(cb_d * .5f )), Quat()));
		NvFlexMakeRigidBoxShape(&board[13], index, InToM(cb_l)/2.0f, InToM(cb_h)/2.0f, InToM(cb_wth)/2.0f, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, -InToM(cb_d * .5f )), Quat()));
		
		NvFlexMakeRigidBoxShape(&board[14], index, InToM(cb_wth)/2.0f, InToM(cb_h)/2.0f, InToM(cb_l)/2.0f, NvFlexMakeRigidPose(Vec3( InToM(cb_d * 0.5f ), 0.0f, 0.0f), Quat()));
		NvFlexMakeRigidBoxShape(&board[15], index, InToM(cb_wth)/2.0f, InToM(cb_h)/2.0f, InToM(cb_l)/2.0f, NvFlexMakeRigidPose(Vec3( -InToM(cb_d * 0.5f ), 0.0f, 0.0f), Quat()));

		for(unsigned int i=0; i<board.size(); i++){
			board[i].thickness = 0.0001;
			board[i].filter = 0x0;
			g_buffers->rigidShapes.push_back(board[i]);
		}
		
		std::vector<float> boxDensity;
		boxDensity.resize(board.size());
		std::fill(boxDensity.begin(), boxDensity.end(), 1000.0f);

		NvFlexRigidBody boxBody;
		NvFlexMakeRigidBody(g_flexLib, &boxBody, Vec3(0.0f, 0.8f, .6f), Quat(), &board[0], &boxDensity[0], board.size());
		g_buffers->rigidBodies.push_back(boxBody);

		NvFlexRigidShape ball;
		NvFlexMakeRigidSphereShape(&ball, g_buffers->rigidBodies.size(), InToM(ball_d)/2.0f * ball_shrink, NvFlexMakeRigidPose(Vec3(0.0f, 0.0f, 0.0f), Quat()));
		ball.filter = 0x0;
		ball.thickness = 0.001f;
		ball.material.friction = 1.0f;
		ball.material.rollingFriction = 0.1f;
		ball.material.torsionFriction = 0.1f;
		ball.user = UnionCast<void*>(AddRenderMaterial(Vec3(0.1f, 0.55f, 0.05f)));

		g_buffers->rigidShapes.push_back(ball);
		float ballDensity = 1000.0f;
		NvFlexRigidBody ballBody;
		NvFlexMakeRigidBody(g_flexLib, &ballBody, Vec3(0.2f, 1.2f, .6f), Quat(), &ball, &ballDensity, 1);
		g_buffers->rigidBodies.push_back(ballBody);

		forceLeft.resize(0);
		forceRight.resize(0);

		g_numSubsteps = 4;
		g_params.numIterations = 40;
		g_params.numPostCollisionIterations = 15;

		g_params.dynamicFriction = 1.25f;	// yes, this is a phsyically plausible friction coefficient, e.g.: velcro, or for rubber on rubber mu is often > 1.0, the solver handles this implicitly and does not violate Coloumb's model
		g_params.particleFriction = 1.0f;
		g_params.damping = 1.0f;
		g_params.sleepThreshold = 0.02f;

		g_params.relaxationFactor = 1.0f;
		g_params.shapeCollisionMargin = 0.01f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);
		g_drawPoints = false;

		/*
		// add camera, todo: read correct links etc from URDF, right now these are thrown away
		const int headLink = urdf->rigidNameMap["head_tilt_link"];
		AddPrimesenseSensor(headLink, Transform(Vec3(-0.1f, 0.2f, 0.0f), rpy2quat(-1.57079632679f, 0.0f, -1.57079632679f)), 1.f, hasFluids);
		*/

		g_pause = true;
	}

	virtual void DoGui()
	{
		float f = 0.008f;
		const float smoothing = 0.01f;
		{
			NvFlexRigidJoint effector0 = g_buffers->rigidJoints[left_effectorJoint];

			float targetx = effector0.pose0.p[0];
			float targety = effector0.pose0.p[1];
			float targetz = effector0.pose0.p[2];

			float oroll = left_rpy.x;
			float opitch = left_rpy.y;
			float oyaw = left_rpy.z;
			imguiSlider("Left Gripper X", &targetx, -1.5f, 1.5f, 0.001f);
			imguiSlider("Left Gripper Y", &targety, 0.0f, 5.0f, 0.001f);
			imguiSlider("Left Gripper Z", &targetz, -2.0f, 2.0f, 0.001f);
			imguiSlider("Left Roll", &left_rpy.x, -180.0f, 180.0f, 0.01f);
			imguiSlider("Left Pitch", &left_rpy.y, -180.0f, 180.0f, 0.01f);
			imguiSlider("Left Yaw", &left_rpy.z, -180.0f, 180.0f, 0.01f);

			left_rpy.x = Lerp(oroll, left_rpy.x, f);
			left_rpy.y = Lerp(opitch, left_rpy.y, f);
			left_rpy.z = Lerp(oyaw, left_rpy.z, f);

			// low-pass filter controls otherwise it is too jerky
			float newx = Lerp(effector0.pose0.p[0], targetx, smoothing);
			float newy = Lerp(effector0.pose0.p[1], targety, smoothing);
			float newz = Lerp(effector0.pose0.p[2], targetz, smoothing);

			effector0.pose0.p[0] = newx;
			effector0.pose0.p[1] = newy;
			effector0.pose0.p[2] = newz;

			Quat q = rpy2quat(left_rpy.x * kPi / 180.0f, left_rpy.y * kPi / 180.0f, left_rpy.z * kPi / 180.0f);
			effector0.pose0.q[0] = q.x;
			effector0.pose0.q[1] = q.y;
			effector0.pose0.q[2] = q.z;
			effector0.pose0.q[3] = q.w;

			g_buffers->rigidJoints[left_effectorJoint] = effector0;

			float newWidth = left_fingerWidth;
			imguiSlider("Finger Width", &newWidth, fingerWidthMin, fingerWidthMax, 0.001f);

			left_fingerWidth = Lerp(left_fingerWidth, newWidth, smoothing);

			g_buffers->rigidJoints[left_fingerLeft].targets[eNvFlexRigidJointAxisX] = left_fingerWidth;
			g_buffers->rigidJoints[left_fingerRight].targets[eNvFlexRigidJointAxisX] = left_fingerWidth;
		}

		{
			NvFlexRigidJoint effector0 = g_buffers->rigidJoints[right_effectorJoint];

			float targetx = effector0.pose0.p[0];
			float targety = effector0.pose0.p[1];
			float targetz = effector0.pose0.p[2];

			float oroll = right_rpy.x;
			float opitch = right_rpy.y;
			float oyaw = right_rpy.z;
			imguiSlider("Right Gripper X", &targetx, -1.5f, 1.5f, 0.001f);
			imguiSlider("Right Gripper Y", &targety, 0.0f, 5.0f, 0.001f);
			imguiSlider("Right Gripper Z", &targetz, -2.0f, 2.0f, 0.001f);
			imguiSlider("Right Roll", &right_rpy.x, -180.0f, 180.0f, 0.01f);
			imguiSlider("Right Pitch", &right_rpy.y, -180.0f, 180.0f, 0.01f);
			imguiSlider("Right Yaw", &right_rpy.z, -180.0f, 180.0f, 0.01f);

			right_rpy.x = Lerp(oroll, right_rpy.x, f);
			right_rpy.y = Lerp(opitch, right_rpy.y, f);
			right_rpy.z = Lerp(oyaw, right_rpy.z, f);

			// low-pass filter controls otherwise it is too jerky
			float newx = Lerp(effector0.pose0.p[0], targetx, smoothing);
			float newy = Lerp(effector0.pose0.p[1], targety, smoothing);
			float newz = Lerp(effector0.pose0.p[2], targetz, smoothing);

			effector0.pose0.p[0] = newx;
			effector0.pose0.p[1] = newy;
			effector0.pose0.p[2] = newz;

			Quat q = rpy2quat(right_rpy.x * kPi / 180.0f, right_rpy.y*  kPi / 180.0f, right_rpy.z * kPi / 180.0f);
			effector0.pose0.q[0] = q.x;
			effector0.pose0.q[1] = q.y;
			effector0.pose0.q[2] = q.z;
			effector0.pose0.q[3] = q.w;

			g_buffers->rigidJoints[right_effectorJoint] = effector0;

			float newWidth = right_fingerWidth;
			imguiSlider("Finger Width", &newWidth, fingerWidthMin, fingerWidthMax, 0.002f);

			right_fingerWidth = Lerp(right_fingerWidth, newWidth, smoothing);

			g_buffers->rigidJoints[right_fingerLeft].targets[eNvFlexRigidJointAxisX] = right_fingerWidth;
			g_buffers->rigidJoints[right_fingerRight].targets[eNvFlexRigidJointAxisX] = right_fingerWidth;
		}
	}

	virtual void DoStats()
	{
		int numSamples = 200;

		int start = Max(int(forceLeft.size()) - numSamples, 0);
		int end = Min(start + numSamples, int(forceLeft.size()));

		// convert from position changes to forces
		float units = -1.0f / Sqr(g_dt / g_numSubsteps);

		int height = 50;

		int dx = 1;
		int sy = int(height / maxForce);

		int lineHeight = 10;

		int rectMargin = 10;
		int rectWidth = dx*numSamples + rectMargin*4;

		int x = g_screenWidth - rectWidth - 20;
		int y = 300;

		DrawRect(float(x), float(y - height - rectMargin), float(rectWidth), float(2.0f*height + rectMargin*3.0f), Vec4(0.0f, 0.0f, 0.0f, 0.5f));

		x += rectMargin*3;

		DrawImguiString(x + dx*numSamples, y + 55, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "Gripper Force (N)");

		DrawLine(float(x), float(y), float(x + numSamples*dx), float(y), 1.0f, Vec3(1.0f));
		DrawLine(float(x), float(y - 50.0f), float(x), float(y + 50.0f), 1.0f, Vec3(1.0f));

		int margin = 5;

		DrawImguiString(x - margin, y, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "0");
		DrawImguiString(x - margin, y + height - lineHeight, Vec3(1.0f), IMGUI_ALIGN_RIGHT, " %.0f", maxForce);
		DrawImguiString(x - margin, y - height, Vec3(1.0f), IMGUI_ALIGN_RIGHT, "-%.0f", maxForce);

		for (int i = start; i < end - 1; ++i)
		{
			float fl0 = Clamp(forceLeft[i] * units, -maxForce, maxForce) * sy;
			float fr0 = Clamp(forceRight[i] * units, -maxForce, maxForce) * sy;

			float fl1 = Clamp(forceLeft[i + 1] * units, -maxForce, maxForce) * sy;
			float fr1 = Clamp(forceRight[i + 1] * units, -maxForce, maxForce) * sy;

			DrawLine(float(x), float(y + fl0), float(x + dx), float(y + fl1), 1.0f, Vec3(1.0f, 0.0f, 0.0f));
			DrawLine(float(x), float(y + fr0), float(x + dx), float(y + fr1), 1.0f, Vec3(0.0f, 1.0f, 0.0f));

			x += dx;
		}
	}

	std::vector<float> forceLeft;
	std::vector<float> forceRight;

	virtual void Update()
	{
		// record force on finger joints
		forceLeft.push_back(g_buffers->rigidJoints[left_fingerLeft].lambda[eNvFlexRigidJointAxisX]);
		forceRight.push_back(g_buffers->rigidJoints[left_fingerRight].lambda[eNvFlexRigidJointAxisX]);

		// move end-effector via numpad
		NvFlexRigidJoint joint = g_buffers->rigidJoints[left_effectorJoint];
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
			left_rpy.x += deltaRollAngle;
		}
		if (g_numpadPressedState[SDLK_KP_2])
		{
			float deltaYawAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaYaw = QuatFromAxisAngle(Vec3(0, 1, 0), deltaYawAngle);
			currentRot = deltaYaw * currentRot;
			left_rpy.z += deltaYawAngle;
		}
		if (g_numpadPressedState[SDLK_KP_3])
		{
			float deltaPitchAngle = numpadJointRotDir * numpadJointRotSpeed;
			Quat deltaPitch = QuatFromAxisAngle(Vec3(0, 0, 1), deltaPitchAngle);
			currentRot = deltaPitch * currentRot;
			left_rpy.y += deltaPitchAngle;
		}
		if (g_numpadPressedState[SDLK_KP_PLUS])
		{
			left_fingerWidth = min(left_fingerWidth + numpadFingerSpeed, fingerWidthMax);
		}
		if (g_numpadPressedState[SDLK_KP_MINUS])
		{
			left_fingerWidth = max(left_fingerWidth - numpadFingerSpeed, fingerWidthMin);
		}
		joint.pose0.q[0] = currentRot.x;
		joint.pose0.q[1] = currentRot.y;
		joint.pose0.q[2] = currentRot.z;
		joint.pose0.q[3] = currentRot.w;
		g_buffers->rigidJoints[left_effectorJoint] = joint;
	}

	virtual void PostUpdate()
	{
		// joints are not read back by default
		NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
	}

	Mesh mesh;

	virtual void Draw(int pass)
	{

	}

};


