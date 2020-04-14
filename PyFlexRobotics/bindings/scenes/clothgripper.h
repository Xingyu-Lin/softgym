//--------------

class ClothGripper : public Scene
{
public:

	const char* file;
	Mesh mesh;

	Vec3 handPos;
	float handWidth;
	float handMinWidth;
	float handMaxWidth;
	float handRot;
	float handRadius;

	float clothThickness;

	int beamx;
	int beamy;
	int beamz;
	bool beamFixed;

	ClothGripper()
	{
		const float invMass = 1.0f;

		clothThickness = 0.0025f;

		const float radius = 0.00625f;

		float stretchStiffness = 0.8f;
		float bendStiffness = 0.8f;
		float shearStiffness = 0.7f;

		CreateSpringGrid(Vec3(0.0f, 0.5f, 0.0f), 80, 80, 1, radius, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f);

		g_params.dynamicFriction = 0.9f;
		g_params.staticFriction = 0.95f;
		g_params.particleFriction = 0.1f;

		g_params.radius = radius*2.0f;
		g_params.collisionDistance = clothThickness;
		g_params.shapeCollisionMargin = 0.1f;

		g_params.relaxationMode = NvFlexRelaxationMode::eNvFlexRelaxationGlobal;
		g_params.relaxationFactor = 0.25f;
		
		//g_params.relaxationFactor = 0.0f;

		g_params.numIterations = 50;
		g_numSubsteps = 4;

		// draw options		
		g_drawPoints = false;
		g_drawMesh = false;
		g_drawCloth = true;
		g_pause = true;


		Vec3 lower, upper;
		GetParticleBounds(lower, upper);

		handRot = 0.0f;
		handRadius = 0.01f;

		handPos = upper + Vec3(0.0f, 0.5f, 0.0f);//Vec3(0.0f, 1.0f, 0.0f);
		handPos.y = upper.y + 0.5f;

		handMaxWidth = 0.1f; // based on Fetch
		handMinWidth = handRadius*2.0f + clothThickness*2.0f*0.7f;
		handWidth = handMaxWidth;

	

		Update();
	}


	virtual void DoGui()
	{
		imguiSlider("Hand Pos (x)", &handPos.x, -0.5f, 0.5f, 0.001f);
		
		// for capsule
		//imguiSlider("Hand Pos (y)", &handPos.y, 0.05f + handRadius + g_params.collisionDistance*2.0f*0.5f, 0.5f, 0.0001f);

		// for box
		imguiSlider("Hand Pos (y)", &handPos.y, 0.05f + g_params.collisionDistance*2.0f*0.6f, 1.0f, 0.0001f);

		imguiSlider("Hand Pos (z)", &handPos.z, -0.5f, 0.5f, 0.001f);
		imguiSlider("Hand Rot", &handRot, -kPi*2.0f, kPi*2.0f, 0.001f);

		imguiSlider("Hand Width", &handWidth, handMinWidth, handMaxWidth, 0.001f);
	}

	virtual void Update()
	{
		std::vector<Vec4> prevPositions;
		std::vector<Quat> prevRotations;

		for (int i=0; i < g_buffers->shapePositions.size(); ++i)
		{
			prevPositions.push_back(g_buffers->shapePositions[i]);
			prevRotations.push_back(g_buffers->shapeRotations[i]);
		}

		ClearShapes();

		float radius = handRadius;
		float halfHeight = 0.05f;

		Quat rot = QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), handRot);

		//AddCapsule(radius, halfHeight, handPos - rot*Vec3(+0.5f*handWidth, 0.0f, 0.0f), rot*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f));
		//AddCapsule(radius, halfHeight, handPos - rot*Vec3(-0.5f*handWidth, 0.0f, 0.0f), rot*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f));
		AddBox(Vec3(halfHeight, radius, radius), handPos - rot*Vec3(+0.5f*handWidth, 0.0f, 0.0f), rot*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f));
		AddBox(Vec3(halfHeight, radius, radius), handPos - rot*Vec3(-0.5f*handWidth, 0.0f, 0.0f), rot*QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), kPi*0.5f));
		
		//AddCapsule(radius, handWidth*0.5f, handPos + Vec3(0.0f, halfHeight, 0.0f), rot*Quat());

		// set previous for friction
		for (int i=0; i < prevPositions.size(); ++i)
		{
			g_buffers->shapePrevPositions[i] = prevPositions[i];
			g_buffers->shapePrevRotations[i] = prevRotations[i];
		}

		UpdateShapes();
	}

};

