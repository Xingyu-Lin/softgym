
class RigidGrasp : public Scene
{
public:

	float angle = 0.0f;
	float lift = 0.0f;

	int padJoint[2];

	RigidGrasp()
	{

		const float linkLength = 0.20f;
		const float linkWidth = 0.05f;

		const float density = 1000.0f;
		const float height = 0.15f;

		NvFlexRigidPose prevJoint;

		for (int i=0; i < 7; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidCapsuleShape(&shape, bodyIndex, linkWidth, linkLength, NvFlexMakeRigidPose(0,0));

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(i*linkLength*2.0f + linkLength, height, 0.0f), Quat(), &shape, &density, 1);

			if (i > 0)
			{
				NvFlexRigidJoint joint;				
				NvFlexMakeFixedJoint(&joint, i-1, bodyIndex, prevJoint, NvFlexMakeRigidPose(Vec3(-linkLength, 0.0f, 0.0f), Quat()));

				g_buffers->rigidJoints.push_back(joint);
			}

			prevJoint = NvFlexMakeRigidPose(Vec3(linkLength, 0.0f, 0.0f), Quat());

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			

		}

		float padWidth = 0.025f;
		float padLength = 0.15f;
		float padCompliance = 1.e-3f;
		float padFriction = 0.7f;
		float padTorsionFriction = 0.0f;

		// attach pad 0
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, bodyIndex, padWidth, padLength, padLength, NvFlexMakeRigidPose(0,0));
			shape.material.friction = padFriction;
			shape.material.torsionFriction = padTorsionFriction;

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(-linkWidth, height, 0.0f), Quat(), &shape, &density, 1);

			NvFlexRigidJoint joint;				
			NvFlexMakeFixedJoint(&joint, 0, bodyIndex, NvFlexMakeRigidPose(Vec3(-linkLength - linkWidth, 0.0f, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(), Quat()));
			joint.compliance[eNvFlexRigidJointAxisSwing1] = padCompliance;

			padJoint[0] = g_buffers->rigidJoints.size();

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);

		}

		// attach pad 1
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, bodyIndex, padWidth, padLength, padLength, NvFlexMakeRigidPose(0,0));
			shape.material.friction = padFriction;
			shape.material.torsionFriction = padTorsionFriction;

			NvFlexRigidPose p;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[6], &p);

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(p.p) + Vec3(linkLength + linkWidth, 0.0f, 0.0f), Quat(), &shape, &density, 1);

			NvFlexRigidJoint joint;				
			NvFlexMakeFixedJoint(&joint, 6, bodyIndex, NvFlexMakeRigidPose(Vec3(linkLength + linkWidth, 0.0f, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(), Quat()));
			joint.compliance[eNvFlexRigidJointAxisSwing1] = padCompliance;

			padJoint[1] = g_buffers->rigidJoints.size();

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);
			g_buffers->rigidJoints.push_back(joint);

		}

		// create manipulation objects
		const int numBoxes = 3;
		float boxWidths[numBoxes] = { 0.05f, 0.1f, 0.05f };
		float boxLength = padLength*0.9f;
		float boxDensity = density*0.1f;
		float boxStart = linkLength*6.0f;

		for (int i=0; i < numBoxes; ++i)
		{
			int bodyIndex = g_buffers->rigidBodies.size();

			NvFlexRigidShape shape;
			NvFlexMakeRigidBoxShape(&shape, bodyIndex, boxWidths[i], boxLength, boxLength*3.0f, NvFlexMakeRigidPose(0,0));
			shape.filter = 0;
			shape.user = UnionCast<void*>(2 + i);	// different render material

			NvFlexRigidBody body;
			NvFlexMakeRigidBody(g_flexLib, &body, Vec3(boxStart + boxWidths[i], boxLength + 0.1f, -boxLength -linkLength*5.0f), Quat(), &shape, &boxDensity, 1);

			g_buffers->rigidBodies.push_back(body);
			g_buffers->rigidShapes.push_back(shape);

			boxStart += boxWidths[i]*2.0f + 0.02f;
		}

		// add fixed joint to middle section
		NvFlexRigidPose p;
		NvFlexGetRigidPose(&g_buffers->rigidBodies[3], &p);

		NvFlexRigidJoint joint;				
		NvFlexMakeFixedJoint(&joint, -1, 3, p, NvFlexMakeRigidPose(Vec3(), Quat()));

		g_buffers->rigidJoints.push_back(joint);


		// set joint damping globally
		for (int i=0; i < g_buffers->rigidJoints.size(); ++i)
		{
			g_buffers->rigidJoints[i].damping[eNvFlexRigidJointAxisTwist] = 10.0f;
			g_buffers->rigidJoints[i].damping[eNvFlexRigidJointAxisSwing1] = 10.0f;
			g_buffers->rigidJoints[i].damping[eNvFlexRigidJointAxisSwing2] = 10.0f;
		}

		g_numSubsteps = 4;
		g_params.numIterations = 50;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);

		// adjust scene scale as needed
		g_params.gravity[1] = -4.0f;

		g_pause = true;
	}

	virtual void Update()
	{
		
	}

	virtual void DoGui()
	{
		float newAngle = angle;
		float newLift = lift;
		imguiSlider("Grasp", &newAngle, 0.0f, 0.76f, 0.001f);
		imguiSlider("Lift", &newLift, 0.0f, kPi, 0.001f);

		// smoothing
		angle = Lerp(angle, newAngle, 0.05f);
		lift = Lerp(lift, newLift, 0.05f);

		for (int i=0; i < 3; ++i)
		{
			g_buffers->rigidJoints[i].targets[eNvFlexRigidJointAxisSwing1] = angle;
		}

		for (int i=3; i < 6; ++i)
		{
			g_buffers->rigidJoints[i].targets[eNvFlexRigidJointAxisSwing1] = angle;
		}

		g_buffers->rigidJoints[padJoint[0]].targets[eNvFlexRigidJointAxisSwing1] = -angle;
		g_buffers->rigidJoints[padJoint[1]].targets[eNvFlexRigidJointAxisSwing1] = angle;

		g_buffers->rigidJoints.back().targets[eNvFlexRigidJointAxisTwist] = lift;

	}
};


class RigidGraspSimple : public Scene
{
public:

	const float boxWidth = 0.05f;
	const float boxDensity = 1000.0f;

	const float gripperRadius = 0.025f;
	const float gripperDensity = 1000.0f;

	float fingerWidth = 0.0f;
	float fingerHeight = 0.0f;

	float motorLimit = 50.0f;


	RigidGraspSimple()
	{

		NvFlexRigidShape boxShape;
		NvFlexMakeRigidBoxShape(&boxShape, 0, boxWidth, boxWidth, boxWidth, NvFlexMakeRigidPose(0,0));
		boxShape.filter = 0;
		boxShape.material.friction = 0.5f;
		boxShape.material.torsionFriction = 0.1f;

		NvFlexRigidBody boxBody;
		NvFlexMakeRigidBody(g_flexLib, &boxBody, Vec3(0.0f, boxWidth, 0.0f), Quat(), &boxShape, &boxDensity, 1);

		g_buffers->rigidShapes.push_back(boxShape);
		g_buffers->rigidBodies.push_back(boxBody);

		// grippers
		{
			
			NvFlexRigidShape gripperShape;
			NvFlexMakeRigidSphereShape(&gripperShape, 1, gripperRadius, NvFlexMakeRigidPose(0,0));
			gripperShape.filter = 0;

			NvFlexRigidPose pose = NvFlexMakeRigidPose(Vec3(-boxWidth - gripperRadius - 0.1f, boxWidth, 0.0f), Quat());			

			NvFlexRigidBody gripperBody;
			NvFlexMakeRigidBody(g_flexLib, &gripperBody, pose.p, pose.q, &gripperShape, &gripperDensity, 1);

			g_buffers->rigidShapes.push_back(gripperShape);
			g_buffers->rigidBodies.push_back(gripperBody);

			NvFlexRigidJoint gripperJoint;
			NvFlexMakeFixedJoint(&gripperJoint, -1, 1, pose, NvFlexMakeRigidPose(0,0));
			gripperJoint.motorLimit[eNvFlexRigidJointAxisX] = motorLimit;
			gripperJoint.compliance[eNvFlexRigidJointAxisX] = 1.e-5f;

			g_buffers->rigidJoints.push_back(gripperJoint);
		}

		// grippers
		{
			
			NvFlexRigidShape gripperShape;
			NvFlexMakeRigidSphereShape(&gripperShape, 2, gripperRadius, NvFlexMakeRigidPose(0,0));
			gripperShape.filter = 0;
			
			NvFlexRigidPose pose = NvFlexMakeRigidPose(Vec3(boxWidth + gripperRadius + 0.1f, boxWidth, 0.0f), Quat());
			
			NvFlexRigidBody gripperBody;
			NvFlexMakeRigidBody(g_flexLib, &gripperBody, pose.p, pose.q, &gripperShape, &gripperDensity, 1);

			g_buffers->rigidShapes.push_back(gripperShape);
			g_buffers->rigidBodies.push_back(gripperBody);

			NvFlexRigidJoint gripperJoint;
			NvFlexMakeFixedJoint(&gripperJoint, -1, 2, pose, NvFlexMakeRigidPose(0,0));
			gripperJoint.motorLimit[eNvFlexRigidJointAxisX] = motorLimit;
			gripperJoint.compliance[eNvFlexRigidJointAxisX] = 1.e-5f;

			g_buffers->rigidJoints.push_back(gripperJoint);
		}


		g_params.numIterations = 32;
		g_params.numPostCollisionIterations = 16;
		//g_params.relaxationMode = eNvFlexRelaxationGlobal;
		//g_params.relaxationFactor = 0.25f;

		g_pause = true;

		g_sceneLower = 0.0f;
		g_sceneUpper = 1.0f;
	}

	float forceLeft = 0.0f;
	float forceRight = 0.0f;

	void DoGui()
	{
		imguiSlider("Gripper", &fingerWidth, -boxWidth*4.0f - gripperRadius*2.0f, boxWidth*2.0f, 0.001f);

		float newHeight = fingerHeight;
		imguiSlider("Gripper Height", &newHeight, 0.0f, 1.0f, 0.001f);
		fingerHeight = Lerp(fingerHeight, newHeight, 0.1f);

		imguiSlider("Motor Limit", &motorLimit, 0.0f, 100.0f, 0.0001f);
		g_buffers->rigidJoints[0].motorLimit[eNvFlexRigidJointAxisX] = motorLimit;
		g_buffers->rigidJoints[1].motorLimit[eNvFlexRigidJointAxisX] = motorLimit;

		g_buffers->rigidJoints[0].targets[eNvFlexRigidJointAxisX] = -fingerWidth*0.5f;
		g_buffers->rigidJoints[1].targets[eNvFlexRigidJointAxisX] = fingerWidth*0.5f;

		g_buffers->rigidJoints[0].targets[eNvFlexRigidJointAxisY] = fingerHeight;
		g_buffers->rigidJoints[1].targets[eNvFlexRigidJointAxisY] = fingerHeight;

	}

	virtual void DoStats()
	{

        int x = g_screenWidth - 200;
        int y = 100;

		forceLeft = g_buffers->rigidJoints[0].lambda[0];
		forceRight = g_buffers->rigidJoints[1].lambda[0];

		DrawImguiString(x, y, Vec3(0.0f, 0.5f, 0.5f), IMGUI_ALIGN_LEFT, "Force Left: %f", forceLeft); y -= 13;
		DrawImguiString(x, y, Vec3(0.0f, 0.5f, 0.5f), IMGUI_ALIGN_LEFT, "Force Right: %f", forceRight); y -= 13;

	}

	virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }
};



class RigidMimic : public Scene
{
public:

	const float boxWidth = 0.05f;
	const float boxDensity = 1000.0f;

	const float gripperRadius = 0.025f;
	const float gripperDensity = 1000.0f;

	float fingerWidth = 0.0f;
	float fingerHeight = 0.0f;

	RigidMimic()
	{

		NvFlexRigidShape boxShape;
		NvFlexMakeRigidBoxShape(&boxShape, 0, boxWidth, boxWidth, boxWidth, NvFlexMakeRigidPose(0,0));
		boxShape.filter = 0;
		boxShape.material.friction = 0.5f;
		boxShape.material.torsionFriction = 0.1f;

		NvFlexRigidBody boxBody;
		NvFlexMakeRigidBody(g_flexLib, &boxBody, Vec3(0.0f, boxWidth, 0.0f), Quat(), &boxShape, &boxDensity, 1);

		g_buffers->rigidShapes.push_back(boxShape);
		g_buffers->rigidBodies.push_back(boxBody);

		// grippers
		{
			
			NvFlexRigidShape gripperShape;
			NvFlexMakeRigidSphereShape(&gripperShape, 1, gripperRadius, NvFlexMakeRigidPose(0,0));
			gripperShape.filter = 0;

			NvFlexRigidPose pose = NvFlexMakeRigidPose(Vec3(-boxWidth - gripperRadius - 0.1f, boxWidth, 0.0f), Quat());			

			NvFlexRigidBody gripperBody;
			NvFlexMakeRigidBody(g_flexLib, &gripperBody, pose.p, pose.q, &gripperShape, &gripperDensity, 1);

			g_buffers->rigidShapes.push_back(gripperShape);
			g_buffers->rigidBodies.push_back(gripperBody);

			NvFlexRigidJoint gripperJoint;
			NvFlexMakeFixedJoint(&gripperJoint, -1, 1, pose, NvFlexMakeRigidPose(0,0));

			g_buffers->rigidJoints.push_back(gripperJoint);
		}

		// grippers
		{
			
			NvFlexRigidShape gripperShape;
			NvFlexMakeRigidSphereShape(&gripperShape, 2, gripperRadius, NvFlexMakeRigidPose(0,0));
			gripperShape.filter = 0;
			
			NvFlexRigidPose pose = NvFlexMakeRigidPose(Vec3(boxWidth + gripperRadius + 0.1f, boxWidth, 0.0f), Quat());
			
			NvFlexRigidBody gripperBody;
			NvFlexMakeRigidBody(g_flexLib, &gripperBody, pose.p, pose.q, &gripperShape, &gripperDensity, 1);

			g_buffers->rigidShapes.push_back(gripperShape);
			g_buffers->rigidBodies.push_back(gripperBody);

			NvFlexRigidJoint gripperJoint;
			
			// set up joint to mimic the first gripper joint
			NvFlexMakeFixedJoint(&gripperJoint, -1, 2, pose, NvFlexMakeRigidPose(0,0));
			gripperJoint.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeMimic;
			gripperJoint.mimicIndex = 0;
			(Vec3&)gripperJoint.mimicScale = Vec3(-1.0f, 0.0f, 0.0f);
			(Vec3&)gripperJoint.mimicOffset = Vec3(0.0f);

			g_buffers->rigidJoints.push_back(gripperJoint);
		}


		g_params.numIterations = 32;
		g_params.numPostCollisionIterations = 16;
		//g_params.relaxationMode = eNvFlexRelaxationGlobal;
		//g_params.relaxationFactor = 0.25f;

		g_pause = true;

		g_sceneLower = 0.0f;
		g_sceneUpper = 1.0f;
	}

	void DoGui()
	{
		imguiSlider("Gripper", &fingerWidth, -boxWidth*4.0f - gripperRadius*2.0f, boxWidth*2.0f, 0.001f);

		float newHeight = fingerHeight;
		imguiSlider("Gripper Height", &newHeight, 0.0f, 1.0f, 0.001f);
		fingerHeight = Lerp(fingerHeight, newHeight, 0.1f);


		g_buffers->rigidJoints[0].targets[eNvFlexRigidJointAxisX] = -fingerWidth*0.5f;
//		g_buffers->rigidJoints[1].targets[eNvFlexRigidJointAxisX] = fingerWidth*0.5f;

		g_buffers->rigidJoints[0].targets[eNvFlexRigidJointAxisY] = fingerHeight;
		g_buffers->rigidJoints[1].targets[eNvFlexRigidJointAxisY] = fingerHeight;


	}

	void Update()
	{
        NvFlexVector<NvFlexRigidContact> rigidContacts(g_flexLib, g_solverDesc.maxRigidBodyContacts);
        NvFlexVector<int> rigidContactCount(g_flexLib, 1);

        NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);

        rigidContacts.map();
        rigidContactCount.map();
	
		for (int i=0; i < rigidContactCount[0]; ++i)
		{
			const NvFlexRigidContact& contact = rigidContacts[i];

			if (contact.lambda > 0.0f)
			{
				printf("%d: %f\n", i, contact.lambda);
			}
		}

		rigidContacts.unmap();
		rigidContactCount.unmap();

		printf("joint0: %f\n", g_buffers->rigidJoints[0].lambda[0]);
		printf("joint1: %f\n", g_buffers->rigidJoints[1].lambda[0]);
	}	

	virtual void PostUpdate()
    {
        // joints are not read back by default
        NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
    }
};
