#include <algorithm>

class RigidCable : public Scene
{
public:

	RigidCable()
	{
		float height = 1.0f;
		NvFlexRigidBody bodies[5];
		NvFlexRigidShape box0;
		float density = 1000.0f;
		NvFlexMakeRigidBoxShape(&box0, 0, 0.15f,0.1f,0.15f, NvFlexMakeRigidPose(Vec3(), Quat()));
		NvFlexMakeRigidBody(g_flexLib, &bodies[0], Vec3(-0.5f, height,0.0f), Quat(), &box0, &density, 1);

		NvFlexRigidShape cap1;
		Mesh* m = ImportMesh("../../data/cylinder.obj");
		m->Normalize();
		float cs = 0.2f;
		NvFlexTriangleMeshId mesh = CreateTriangleMesh(m);
		NvFlexMakeRigidTriangleMeshShape(&cap1, 1, mesh, NvFlexMakeRigidPose(Vec3(-cs*0.5f, -cs*0.5f, -cs*0.5f), Quat()), cs, cs, cs);
		NvFlexMakeRigidBody(g_flexLib, &bodies[1], Vec3(-0.4f, height + 0.5f, 0.0f), Quat(), &cap1, &density, 1);


		NvFlexRigidShape cap2;
		NvFlexMakeRigidCapsuleShape(&cap2, 2, 0.1f, 0.1f, NvFlexMakeRigidPose(Vec3(), Quat()));
		NvFlexMakeRigidBody(g_flexLib, &bodies[2], Vec3(0.0f, height+0.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f), &cap2, &density, 1);


		NvFlexRigidShape cap3;
		NvFlexMakeRigidCapsuleShape(&cap3, 3, 0.1f, 0.1f, NvFlexMakeRigidPose(Vec3(), Quat()));
		NvFlexMakeRigidBody(g_flexLib, &bodies[3], Vec3(0.4f, height + 0.5f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f), &cap3, &density, 1);

		NvFlexRigidShape box4;
		NvFlexMakeRigidBoxShape(&box4, 4, 0.1f, 0.1f, 0.1f, NvFlexMakeRigidPose(Vec3(), Quat()));
		NvFlexMakeRigidBody(g_flexLib, &bodies[4], Vec3(0.5f, height+0.0f, 0.0f), Quat(), &box4, &density, 1);

		box0.filter = 0;
		cap1.filter = 0;
		cap2.filter = 0;
		cap3.filter = 0;
		box4.filter = 0;
		for (int i = 0; i < 5; i++)
		{
			bodies[i].angularDamping = 1.0f;
			bodies[i].linearDamping = 0.001f;
		}

		g_buffers->rigidShapes.push_back(box0);
		g_buffers->rigidBodies.push_back(bodies[0]);
		g_buffers->rigidShapes.push_back(cap1);
		g_buffers->rigidBodies.push_back(bodies[1]);
		g_buffers->rigidShapes.push_back(cap2);
		g_buffers->rigidBodies.push_back(bodies[2]);
		g_buffers->rigidShapes.push_back(cap3);
		g_buffers->rigidBodies.push_back(bodies[3]);
		g_buffers->rigidShapes.push_back(box4);
		g_buffers->rigidBodies.push_back(bodies[4]);


		NvFlexCableLink links[5];
		NvFlexInitFixedCableLink(g_flexLib, &links[0], 0, &bodies[0], &box0, 1, Vec3(-0.5f, height + 0.1f, 0.0f));
		NvFlexInitRollingCableLink(g_flexLib, &links[1], 1, &bodies[1], &cap1, 1, Vec3(0.0f, 0.0f, -1.0f), 0.0f);
		NvFlexInitRollingCableLink(g_flexLib, &links[2], 2, &bodies[2], &cap2, 1, Vec3(0.0f, 0.0f, 1.0f), 0.0f);
		NvFlexInitRollingCableLink(g_flexLib, &links[3], 3, &bodies[3], &cap3, 1, Vec3(0.0f, 0.0f, -1.0f), 0.0f);
		NvFlexInitFixedCableLink(g_flexLib, &links[4], 4, &bodies[4], &box4, 1, Vec3(0.5f, height + 0.1f, 0.0f));

		float stretchingCompliance = 0.0f;
		float stretchingDamping = 0.0f;

		float compressionCompliance = -1.0f; // No resistance to compression
		float compressionDamping = 0.0f;

		const NvFlexRigidBody* pbodies[5] = { &bodies[0], &bodies[1], &bodies[2], &bodies[3], &bodies[4] };
		NvFlexMakeCable(g_flexLib, links, pbodies, 5, false, stretchingCompliance, stretchingDamping, compressionCompliance, compressionDamping);

		g_buffers->cableLinks.push_back(links[0]);
		g_buffers->cableLinks.push_back(links[1]);
		g_buffers->cableLinks.push_back(links[2]);
		g_buffers->cableLinks.push_back(links[3]);
		g_buffers->cableLinks.push_back(links[4]);
		NvFlexRigidJoint c1j;
		NvFlexMakeHingeJoint(&c1j, -1, 1, NvFlexMakeRigidPose(Vec3(-0.4f, height + 0.5f, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(), Quat()), eNvFlexRigidJointAxisSwing2);

		NvFlexRigidJoint c2j;
		NvFlexMakeHingeJoint(&c2j, -1, 2, NvFlexMakeRigidPose(Vec3(0.0f, height + 0.0f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)), NvFlexMakeRigidPose(Vec3(), Quat()), eNvFlexRigidJointAxisTwist);

		NvFlexRigidJoint c3j;
		NvFlexMakeHingeJoint(&c3j, -1, 3, NvFlexMakeRigidPose(Vec3(0.4f, height + 0.5f, 0.0f), QuatFromAxisAngle(Vec3(0.0f, 1.0f, 0.0f), -kPi*0.5f)), NvFlexMakeRigidPose(Vec3(), Quat()), eNvFlexRigidJointAxisTwist);

		g_buffers->rigidJoints.push_back(c1j);
		g_buffers->rigidJoints.push_back(c2j);
		g_buffers->rigidJoints.push_back(c3j);
		g_params.gravity[1] = -9.8f;

		g_sceneLower = Vec3(-2.0f);
		g_sceneUpper = Vec3(2.0f);
		g_numSubsteps = 1;
		g_params.numIterations = 20;
		g_params.relaxationFactor = 1.0;
		g_pause = true;
	}
};
int* g_okJoint = new int [500];

class RigidURDFCable : public Scene
{
public:
	struct Params
	{
		void setDefaults()
		{
			visThickness = 0.02f;
			visSamplingDist = 0.05f;
			visSubdiv = 6;
			visualBendingType = STIFF;
			constant = false;
			stretchingCompliance = 0.0f;
			compressionCompliance = -1.0f;	// none
			alterColor = false;
		}
		float visThickness;
		float visSamplingDist;
		int visSubdiv;
		bool constant;
		float stretchingCompliance, compressionCompliance;
		bool alterColor;

		enum VisualBendingType
		{
			NONE,
			LOOSE,
			STIFF,
		};
		VisualBendingType visualBendingType;
	};
	Params params;

	vector<vector<Transform> > segmentSamples;
	vector<vector<Transform> > linkSamples;

	vector<vector<Vec3> > profileCurves;
	URDFImporter* p_urdf;
	Transform gt;
	vector<int> controlIndices;
	vector<float> phases;

	vector<int> jointIndices;
	vector<float> jphases;
	vector<int> staticIndices;
	vector<Transform> staticPoses;

	vector<int> gearIndices;
	vector<Transform> gearPoses;	
	vector<Vec3> gearNormal;

	void SwingTwistDecomposition(const Quat& rot, const Vec3& dir, Quat& swing, Quat& twist)
	{
		Vec3 v = rot.GetAxis();
		Vec3 p = Dot(v, dir)*dir;

		twist.x = p.x;
		twist.y = p.y;
		twist.z = p.z;
		twist.w = rot.w;
		twist = Normalize(twist);

		Quat twistC(-twist.x, -twist.y, -twist.z, twist.w);
		
		swing = rot * twistC;
	}

	RigidURDFCable()
	{
		memset(g_okJoint, 0, sizeof(int) * 500);
		ifstream inf("okj.txt");
		int i = 0;
		int v;
		while (inf >> v)
		{
			g_okJoint[i] = v;
			i++;
		}


		//p_urdf = new URDFImporter("../../data", "robot_hand.urdf");
		//p_urdf = new URDFImporter("../../data", "cablelimits.urdf");
		//p_urdf = new URDFImporter("../../data", "cableHilbertSmall.urdf");
		//p_urdf = new URDFImporter("../../data", "cablePully.urdf");
		//p_urdf = new URDFImporter("C:\\p4sw\\physx\\research\\SampleViewer\\bin\\resources", "cablemachine.urdf");
		p_urdf = new URDFImporter("../../data/matt_newer", "hand.urdf", false);

		//const int whiteMaterial = AddRenderMaterial(Vec3(1.0f, 1.0f, 1.0f), 0.0f, 0.0f, false);
		const int hiddenMaterial = AddRenderMaterial(0.0f, 0.0f, 0.0f, true);
		for (int a = 0; a < 5; a++)
		{
			for (int b = 0; b < 5; b++)
			{
				gt = Transform(Vec3(-a*2.0f, 0.0f, -b*2.0f), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), 0.0f));
				//p_urdf->AddPhysicsEntities(gt, whiteMaterial, false, 1.0f, 0.0f, 1.0f, 0.01f, 20.0f, 100.0f, false, 1e10, 1.0f);
				p_urdf->AddPhysicsEntities(gt, hiddenMaterial, true, 1000.0f, 1e-5f, 1.0f, 0.0f, 0.0f, 100.0f, true, 1e10, 1.0f);
				
				for (int i = 0; i < p_urdf->cables.size(); i++) 
				{
					URDFCable* c = p_urdf->cables[i];
					for (int l = 0; l < c->links.size(); l++) 
					{
						if (c->links[l].type == URDFCableLink::ROLLING)
						{
							gearIndices.push_back(p_urdf->rigidNameMap[c->links[l].body->name]);
							Transform tt;
							NvFlexGetRigidPose(&g_buffers->rigidBodies[gearIndices.back()], (NvFlexRigidPose*)&tt);
							gearPoses.push_back(tt);
							gearNormal.push_back(Normalize(c->links[l].normal));
							float d0 = fabs(Dot(GetBasisVector0(tt.q), gearNormal.back()));
							float d1 = fabs(Dot(GetBasisVector1(tt.q), gearNormal.back()));
							float d2 = fabs(Dot(GetBasisVector2(tt.q), gearNormal.back()));
							/*
							if ((d0 >= d1) && (d0 >= d2)) cout << "Axis " << l << " is 0" << endl;
							if ((d1 >= d0) && (d1 >= d2)) cout << "Axis " << l << " is 1" << endl;
							if ((d2 >= d1) && (d2 >= d0)) cout << "Axis " << l << " is 2" << endl;
							cout << "Gear normal = " << gearNormal.back().x << " " << gearNormal.back().y << " " << gearNormal.back().z << endl;
							*/
						}
					}

				}
			
				Vec3 gn(-0.552114f, -0.152923f, 0.819625f);
				for (auto rb : p_urdf->rigidNameMap)
				{
					if (rb.first.find("motor") != string::npos) 
					{
						bool found = false;
						for (int j = 0; j < gearIndices.size(); j++) 
						{
							if (gearIndices[j] == rb.second) found = true;
						} 
						if (!found) 
						{
							gearIndices.push_back(rb.second);
							Transform tt;
							NvFlexGetRigidPose(&g_buffers->rigidBodies[gearIndices.back()], (NvFlexRigidPose*)&tt);
							gearPoses.push_back(tt);
							gearNormal.push_back(gn);
						}
					}
				}
				
				for (auto aj : p_urdf->activeJointNameMap)
				{
					for (int k = 0; k < 6; ++k)
					{
						g_buffers->rigidJoints[aj.second].modes[k] = eNvFlexRigidJointModeFree;
					}
				}
				for (int i = 0; i < p_urdf->targets.size(); i++)
				{
					g_buffers->rigidJoints[p_urdf->activeJointNameMap[p_urdf->targets[i].jointName]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
					g_buffers->rigidJoints[p_urdf->activeJointNameMap[p_urdf->targets[i].jointName]].targets[eNvFlexRigidJointAxisTwist] = p_urdf->targets[i].angle;
					g_buffers->rigidJoints[p_urdf->activeJointNameMap[p_urdf->targets[i].jointName]].compliance[eNvFlexRigidJointAxisTwist] = p_urdf->targets[i].compliance;
				}

				jointIndices.push_back(p_urdf->activeJointNameMap["finger1-con"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["finger1-ext"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["finger2-con"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["finger2-ext"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["finger3-con"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["finger3-ext"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["finger4-con"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["finger4-ext"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["thumb-close"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["thumb-con"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["thumb-ext"]);
				jointIndices.push_back(p_urdf->activeJointNameMap["thumb-out"]);
				//jointIndices.push_back(p_urdf->activeJointNameMap["wrist-in"]);
				//jointIndices.push_back(p_urdf->activeJointNameMap["wrist-up-left"]);
				//jointIndices.push_back(p_urdf->activeJointNameMap["wrist-up-right"]);

				staticIndices.push_back(p_urdf->rigidNameMap["arm"]);
				staticIndices.push_back(p_urdf->rigidNameMap["wrist1"]);
				staticIndices.push_back(p_urdf->rigidNameMap["wrist2"]);
				staticIndices.push_back(p_urdf->rigidNameMap["palm"]);
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control1"]].linearVel[1] = -Randf()*100.0f;
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control2"]].linearVel[1] = -Randf()*100.0f;
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control3"]].linearVel[1] = -Randf()*100.0f;
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control4"]].linearVel[1] = -Randf()*100.0f;
				/*
				controlIndices.push_back(p_urdf->rigidNameMap["control1"]);
				controlIndices.push_back(p_urdf->rigidNameMap["control2"]);
				controlIndices.push_back(p_urdf->rigidNameMap["control3"]);
				controlIndices.push_back(p_urdf->rigidNameMap["control4"]);
				phases.push_back(Randf() * k2Pi);
				phases.push_back(Randf() * k2Pi);
				phases.push_back(Randf() * k2Pi);
				phases.push_back(Randf() * k2Pi);
				*/
			}
		}
		for (int i = 0; i < staticIndices.size(); i++)
		{
			Transform tt;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[staticIndices[i]], (NvFlexRigidPose*)&tt);
			staticPoses.push_back(tt);
		}
		for (int i = 0; i <jointIndices.size(); i++)
		{
			g_buffers->rigidJoints[jointIndices[i]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
			g_buffers->rigidJoints[jointIndices[i]].targets[eNvFlexRigidJointAxisTwist] = 0.0f;
			g_buffers->rigidJoints[jointIndices[i]].compliance[eNvFlexRigidJointAxisTwist] = 1e-1f;
		}
		for (int i = 0; i < jointIndices.size(); i++)
		{
			jphases.push_back(Randf() * k2Pi);
		}
		for (int i = 0; i < g_buffers->rigidShapes.size(); i++)
		{
			g_buffers->rigidShapes[i].filter = 1;
		}

		// Lowered gravity
		g_params.gravity[1] = 0.0f;

		g_numSubsteps = 10;
		g_params.numIterations = 5;
		g_params.numPostCollisionIterations = 5;
		g_params.dynamicFriction = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;
		g_params.collisionDistance = 0.01f;
		g_params.relaxationFactor = 0.75f;

		g_sceneLower = Vec3(-1.0f);
		g_sceneUpper = Vec3(1.0f);

		g_pause = true;
		g_drawPoints = false;
		g_drawCloth = false;
		g_drawCable = true;
		/*
		for (int i = 0; i < g_buffers->cableLinks.size(); i++) {
			g_buffers->cableLinks[i].stretchingDamping = 10.0f;
			g_buffers->cableLinks[i].compressionDamping = 10.0f;
		}
		for (int i = 0; i < 4; i++) {
			maxB[i] = g_buffers->cableLinks[3 + 5 * i].segLength;
			dd[i] = 0.0f;
		}
		*/



		if (!g_drawCable)
		{
			unsigned int maxCurveID = 0;
			for (int i = 0; i < g_buffers->cableLinks.size(); i++)
			{
				maxCurveID = std::max(maxCurveID, g_buffers->cableLinks[i].profileVerts);
			}
			maxCurveID++;
			profileCurves.resize(maxCurveID);
			NvFlexVector<Vec3> pos(g_flexLib, 100000);
			for (int i = 1; i < (int)maxCurveID; i++)
			{
				int numV = 0;
				NvFlexGetCurve(g_flexLib, i, pos.buffer, &numV);
				if (numV > 0)
				{
					profileCurves[i].resize(numV);
					pos.map();
					memcpy(&profileCurves[i][0], pos.mappedPtr, sizeof(Vec3)*numV);
					pos.unmap();
				}
			}

			ComputeCableSamples();
		}
		//di = 0;
	}
	float dd[4];
	float maxB[4];
	//float maxS[4];

	virtual void CenterCamera(void)
	{
		g_camPos = Vec3(0.0658046, 1.88832, 5.32831);
		g_camAngle = Vec3(0.00174534, -0.221657, 0);

		g_camPos = Vec3(0.0722092, 0.927709, 0.703847);
		g_camAngle = Vec3(-0.00174532, -0.336849, 0);
	}

	virtual void Update()
	{
		for (int i = 0; i < staticIndices.size(); i++)
		{

			Transform tt = staticPoses[i];
			NvFlexSetRigidPose(&g_buffers->rigidBodies[staticIndices[i]], (NvFlexRigidPose*)&tt);
			//staticPoses.push_back(tt);
		}

		for (int i = 0; i < gearIndices.size(); i++)
		{

			Transform tt;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[gearIndices[i]], (NvFlexRigidPose*)&tt);

			tt.p = gearPoses[i].p;

			/*
			Quat rotFromRest = Inverse(gearPoses[i].q)*tt.q;
			Quat swing, twist;
			SwingTwistDecomposition(rotFromRest, Vec3(0.0f, 0.0f, 1.0f), swing, twist);
			tt.q = gearPoses[i].q*twist;*/
			
			
			Quat swing, twist;
			SwingTwistDecomposition(tt.q, gearNormal[i], swing, twist);
			tt.q = twist;
			

			//tt = gearPoses[i];
			NvFlexSetRigidPose(&g_buffers->rigidBodies[gearIndices[i]], (NvFlexRigidPose*)&tt);
			//staticPoses.push_back(tt);
		}
		/*
		for (int i = 0; i < 4; i++) {
				g_buffers->cableLinks[3 + 5 * i].segLength += dd[i];
				//g_buffers->cableLinks[26 + 4 * i].segLength -= dd[i];


			if (g_buffers->cableLinks[3 + 5 * i].segLength < 0.1f) g_buffers->cableLinks[3 + 5 * i].segLength = 0.1f;
			if (g_buffers->cableLinks[3 + 5 * i].segLength > maxB[i]) g_buffers->cableLinks[3 + 5 * i].segLength = maxB[i];
			cout << g_buffers->cableLinks[3 + 5 * i].segLength << " ";
		}*/
		//cout << "g_camPos = Vec3(" << g_camPos.x << ", " << g_camPos.y << ", " << g_camPos.z << ");" << endl;
		//cout << "g_camAngle = Vec3(" << g_camAngle.x << ", " << g_camAngle.y << ", " << g_camAngle.z << ");" << endl;
		static float time = 0.0f;
		time += g_dt;
		for (int i = 0; i < controlIndices.size(); i++)
		{
			g_buffers->rigidBodies[controlIndices[i]].linearVel[1] = sin(phases[i] + time)*2.0f;
		}
		for (int i = 0; i < jointIndices.size(); i++)
		{
			g_buffers->rigidJoints[jointIndices[i]].targets[eNvFlexRigidJointAxisTwist] = sin(jphases[i] + time*1.0f*( ((i)*32132101 + (i+1)*(i+1)*33211311 + (i+3)) % 15 + 1))*kPi;
		}
		for (int i = 0; i < jointIndices.size(); i++)
		{
			jphases.push_back(Randf() * k2Pi);
		}

		if (!g_drawCable)
		{
			ComputeCableSamples();
		}
	}

	virtual void Sync()
	{
		if (!g_drawCable)
		{
			if (g_buffers->cableLinks.size())
			{
				NvFlexSetCableLinks(g_solver, g_buffers->cableLinks.buffer, g_buffers->cableLinks.size());
			}
		}
	}
	//int di;
	virtual void KeyDown(int key)
	{

		float sp = 0.01f;
		for (int i = 0; i < 4; i++)
		{
			if (key == '1' + i)
			{
				if (dd[i] > 0.0f)
				{
					dd[i] = -sp;
				}
				else if (dd[i] < 0.0f)
				{
					dd[i] = 0.0f;
				}
				else
				{
					dd[i] = sp;
				}
			}
		}
		Scene::KeyDown(key);
	}

	virtual void Draw(int pass)
	{
		/*
		BeginLines(true);
		for (int i = 0; i < gearIndices.size(); i++)
		{
			Transform tt;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[gearIndices[i]], (NvFlexRigidPose*)&tt);
			Vec3 pos = tt.p;
			Vec3 pos2 = tt.p + gearNormal[i] * 0.1f;
			DrawLine(pos, pos2, Vec4(1.0f, 0.0f, 1.0f, 1.0f));
		}
		EndLines();
		*/
		if (!g_drawCable)
		{
			/*
			Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
			Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));
			Vec3 up = Cross(right, forward);
			float size = 0.05f;
			BeginLines(true);
			for (int i = 0; i < p_urdf->debugPoints.size(); i++)
			{
				Vec3 pos = p_urdf->debugPoints[i];
				DrawLine(pos - up*size, pos + up*size, p_urdf->debugPointsCols[i]);
				DrawLine(pos - right*size, pos + right*size, p_urdf->debugPointsCols[i]);
			}
			EndLines();
			*/
			//DrawPlanes(&p_urdf->debugPlanes[di], 1, 0.0f);
			BeginLines(true);
			Vec3 x = GetBasisVector0(gt.q);
			Vec3 y = GetBasisVector1(gt.q);
			Vec3 z = GetBasisVector2(gt.q);
			float len = 1.0f;
			DrawLine(gt.p, gt.p + len*x, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
			DrawLine(gt.p, gt.p + len*y, Vec4(0.0f, 1.0f, 0.0f, 1.0f));
			DrawLine(gt.p, gt.p + len*z, Vec4(0.0f, 0.0f, 1.0f, 1.0f));
			EndLines();

			//vector<vector<Transform> > segmentSamples;
			//vector<vector<Transform> > linkSamples;
			BeginLines(true);
			for (int i = 0; i < segmentSamples.size(); i++)
			{
				if (segmentSamples[i].size() > 1)
				{
					for (int j = 0; j < ((int)segmentSamples[i].size()) - 1; j++)
					{
						DrawLine(segmentSamples[i][j].p, segmentSamples[i][j + 1].p, Vec4(1.0f, 1.0f, 0.0f, 1.0f));
					}
				}
			}
			for (int i = 0; i < linkSamples.size(); i++)
			{
				if (linkSamples[i].size() > 1)
				{
					for (int j = 0; j < ((int)linkSamples[i].size()) - 1; j++)
					{
						//cout << linkSamples[i][j].p.x << " " << linkSamples[i][j].p.y << " " << linkSamples[i][j].p.z << endl;
						DrawLine(linkSamples[i][j].p, linkSamples[i][j + 1].p, Vec4(0.0f, 1.0f, 0.0f, 1.0f));
					}
				}
			}
			EndLines();
		}
	}

	virtual void DoStats()
	{
		/*
		Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
		Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));
		Vec3 up = Cross(right, forward);
		for (int i = 0; i < p_urdf->debugLabelPos.size(); i++)
		{
			Vec3 pos = p_urdf->debugLabelPos[i] + up*0.1f;
			Vec3 sc = GetScreenCoord(pos);
			if (sc.z < 1.0f)
			{
				DrawImguiString(int(sc.x + 5.f), int(sc.y - 5.f), Vec3(1.f, 1.f, 0.f), 0, "%s", p_urdf->debugLabelString[i].c_str());
			}
		}	*/
	}

	bool isZero(const Vec3 v)
	{
		return Dot(v, v) < 1e-10f;
	}

	void createFrame(const Vec3 d, Matrix33 &R, Matrix33* prevR = NULL)
	{
		R.cols[2] = Normalize(d);
		if (prevR && !isZero(Cross(R.cols[2], prevR->cols[0])))
		{
			R.cols[0] = prevR->cols[0];
		}
		else
		{
			if (fabs(d.x) < fabs(d.y) && fabs(d.x) < fabs(d.z))
			{
				R.cols[0] = Vec3(1.0f, 0.0f, 0.0f);
			}
			else if (fabs(d.y) < fabs(d.z))
			{
				R.cols[0] = Vec3(0.0f, 1.0f, 0.0f);
			}
			else
			{
				R.cols[0] = Vec3(0.0f, 0.0f, 1.0f);
			}
		}
		R.cols[1] = Normalize(Cross(R.cols[2], R.cols[0]));
		R.cols[0] = Cross(R.cols[1], R.cols[2]);
	}

	void ComputeCableSamples()
	{
		// segment sampling
		int numLinks = g_buffers->cableLinks.size();
		segmentSamples.resize(numLinks);
		linkSamples.resize(numLinks);

		for (int i = 0; i < numLinks; i++)
		{
			NvFlexCableLink &link = g_buffers->cableLinks[i];
			if (link.nextLink < 0)
			{
				continue;
			}

			NvFlexCableLink &next = g_buffers->cableLinks[link.nextLink];
			Vec3 p0 = link.att1;
			Vec3 p1 = next.att0;
			Transform btrans;
			Transform ntrans;
			if (link.body >= 0)
			{
				NvFlexGetRigidPose(&g_buffers->rigidBodies[link.body], (NvFlexRigidPose*)&btrans);
				p0 = TransformPoint(btrans, p0);
			}
			if (next.body >= 0)
			{
				NvFlexGetRigidPose(&g_buffers->rigidBodies[next.body], (NvFlexRigidPose*)&ntrans);
				p1 = TransformPoint(ntrans, p1);
			}

			Vec3 n = p1 - p0;
			float l = Length(n);
			n /= l;

			float d = link.segLength;

			if (params.visualBendingType == Params::NONE)
			{
				segmentSamples[i].resize(2);
				segmentSamples[i][0].p = p0;
				segmentSamples[i][1].p = p1;
			}
			else if (d <= l)
			{
				int num = std::max((int)ceil(link.segLength / params.visSamplingDist), 2);
				segmentSamples[i].resize(num + 1);
				float dx = l / (float)num;

				for (int j = 0; j <= num; j++)
				{
					segmentSamples[i][j].p = p0 + j * dx * n;
				}
			}
			else
			{
				int num = (int)ceil(l / params.visSamplingDist);
				segmentSamples[i].resize(num + 1);
				float h = std::min(0.5f * sqrtf(d * d - l * l), 0.2f * l);
				float dx = 1.0f / (float)num;
				for (int j = 0; j <= num; j++)
				{
					Vec3 p = p0 + j * dx * l * n;
					if (params.visualBendingType == Params::STIFF)
					{
						float x = 2 * j < num ? j * dx : 1.0f - j * dx;
						float y = 0.3f * h * (64.0f * x * x * x - 32.0f * x * x);
						if (2 * j < num)
						{
							y = -y;
						}
						Vec3 t(-n.y, n.x, 0.0f);
						p += t * y;
					}
					else
					{
						float x = j * dx;
						float y = p.y - h * (-4.0f * x * x + 4.0f * x);
						if (y < 0.0f)
						{
							y = 0.0f;
						}
						/*
						for (int l = 0; l < 2; l++) {
							Body *b = l ? link.body : next.body;
							float t;
							if (b && b->rayCast(p, Vec3(0.0f, -1.0f, 0.0f), t)) {
								float cy = p.y - t;
								if (t >= 0.0f && cy > y)
									y = cy;
							}
						}*/
						p.y = y;
					}
					segmentSamples[i][j].p = p;
				}
				for (int iter = 0; iter < 5; iter++)
				{
					for (int j = 1; j < num - 1; j++)
					{
						Vec3 avg = (segmentSamples[i][j - 1].p + segmentSamples[i][j + 1].p) * 0.5f;
						segmentSamples[i][j].p += (avg - segmentSamples[i][j].p) * 0.5f;
					}
				}
			}
		}

		// link sampling

		for (int i = 0; i < numLinks; i++)
		{
			NvFlexCableLink &link = g_buffers->cableLinks[i];
			if (!link.body || link.type != eNvFlexCableLinkTypeRolling)
			{
				continue;
			}
			Transform profilePose = (Transform&)link.profilePose;
			Transform iprofilePose = Inverse(profilePose);
			Vec3 profilePos0 = TransformPoint(iprofilePose, link.att0);
			Vec3 profilePos1 = TransformPoint(iprofilePose, link.att1);
			Transform bpose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[link.body], (NvFlexRigidPose*)&bpose);
			Transform pose = bpose * profilePose;

			Vec3 test0 = TransformPoint(pose, profilePos0);
			Vec3 test1 = TransformPoint(pose, profilePos1);

			linkSamples[i].clear();
			if (link.profileVerts == 0)
			{
				Vec3 n;
				n = profilePos0;
				if (link.prevLink < 0)
				{
					n = profilePos1;
				}

				n = Normalize(n);

				float phi = acosf(n.x);
				if (n.y < 0.0f)
				{
					phi = 2.0f*kPi - phi;
				}
				float dphi = link.linkLength / link.radius;
				if (link.flipped)
				{
					dphi = -dphi;
				}
				if (link.prevLink < 0)
				{
					dphi = -dphi;
				}
				float l = fabs(dphi * link.radius);
				int num = (int)ceil(l / params.visSamplingDist);
				dphi /= num;
				for (int j = 0; j <= num; j++)
				{
					Vec3 vr = link.radius * Vec3(cosf(phi + j * dphi), sinf(phi + j * dphi), 0.0f);
					linkSamples[i].push_back(Transform(TransformPoint(pose, vr)));
				}
			}
			else
			{
				int nr0 = -1;
				int nr1 = -1;
				float min0 = FLT_MAX;
				float min1 = FLT_MAX;
				vector<Vec3>& profileVerts = profileCurves[link.profileVerts];
				for (int i = 0; i < (int)profileVerts.size(); i++)
				{
					float d0 = LengthSq(profileVerts[i] - profilePos0);
					float d1 = LengthSq(profileVerts[i] - profilePos1);
					if (d0 < min0)
					{
						min0 = d0;
						nr0 = i;
					}
					if (d1 < min1)
					{
						min1 = d1;
						nr1 = i;
					}
				}
				float len = 0.0f;
				int prev = -1;
				int nr = nr0;
				while (prev != nr1)
				{
					linkSamples[i].push_back(Transform(TransformPoint(pose, profileVerts[nr])));
					prev = nr;
					nr = (nr + 1) % profileVerts.size();
				}
			}
		}

		// frames

		for (int i = 0; i < numLinks; i++)
		{
			NvFlexCableLink &link = g_buffers->cableLinks[i];
			for (int j = 0; j < 2; j++)
			{
				std::vector<Transform> &samples = j == 0 ? linkSamples[i] : segmentSamples[i];
				if (samples.size() < 2)
				{
					continue;
				}

				Matrix33 R = Matrix33::Identity();
				for (int k = 0; k < (int)samples.size() - 1; k++)
				{
					createFrame(samples[k + 1].p - samples[k].p, R, &R);
					samples[k].q = Quat(R);
				}
				samples.back().q = samples[samples.size() - 2].q;
			}
		}

		//for (int i = 0; i < (int)links.size(); i++) {
		//	Link &link = links[i];

		//	if (!link.segmentSampleNormals.empty() && !link.linkSampleNormals.empty()) {
		//		Vec3 &nl = link.linkSampleNormals.back();
		//		Vec3 &ns = link.segmentSampleNormals.front();
		//		Vec3 n = (nl + ns);
		//		n.normalize();
		//		nl = n;
		//		ns = n;
		//	}
		//	if (i < (int)links.size() - 1 || cyclic) {
		//		Link &next = links[(i + 1) % links.size()];
		//		if (!next.linkSampleNormals.empty() && !next.linkSampleNormals.empty()) {
		//			Vec3 &nl = next.linkSampleNormals.front();
		//			Vec3 &ns = link.segmentSampleNormals.back();
		//			Vec3 n = (nl + ns);
		//			n.normalize();
		//			nl = n;
		//			ns = n;
		//		}
		//	}
		//}
	}
};



class RigidURDFCableHand : public Scene
{
public:
	struct Params {
		void setDefaults() {
			visThickness = 0.02f;
			visSamplingDist = 0.05f;
			visSubdiv = 6;
			visualBendingType = STIFF;
			constant = false;
			stretchingCompliance = 0.0f;
			compressionCompliance = -1.0f;	// none
			alterColor = false;
		}
		float visThickness;
		float visSamplingDist;
		int visSubdiv;
		bool constant;
		float stretchingCompliance, compressionCompliance;
		bool alterColor;

		enum VisualBendingType {
			NONE,
			LOOSE,
			STIFF,
		};
		VisualBendingType visualBendingType;
	};
	Params params;

	vector<vector<Transform> > segmentSamples;
	vector<vector<Transform> > linkSamples;

	vector<vector<Vec3> > profileCurves;
	URDFImporter* p_urdf;
	Transform gt;
	vector<int> controlIndices;
	vector<float> phases;
	RigidURDFCableHand()
	{
		p_urdf = new URDFImporter("../../data", "robot_hand.urdf");
		//p_urdf = new URDFImporter("../../data", "cablelimits.urdf");
		//p_urdf = new URDFImporter("../../data", "cableHilbertSmall.urdf");
		//p_urdf = new URDFImporter("../../data", "cablePully.urdf");
		//p_urdf = new URDFImporter("C:\\p4sw\\physx\\research\\SampleViewer\\bin\\resources", "cablemachine.urdf");
		//p_urdf = new URDFImporter("../../data/matt", "hand.urdf", false);

		const int whiteMaterial = AddRenderMaterial(Vec3(1.0f, 1.0f, 1.0f), 0.0f, 0.0f, false);
		for (int a = 0; a < 33; a++) {
			for (int b = 0; b < 33; b++) {
				gt = Transform(Vec3(-a*5.0f, 0.0f, -b*5.0f), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), 0.0f));
				p_urdf->AddPhysicsEntities(gt, whiteMaterial, true, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100.0f, true, 1e10, 1.0f);

				for (auto aj : p_urdf->activeJointNameMap)
				{
					for (int k = 0; k < 6; ++k)
					{
						g_buffers->rigidJoints[aj.second].modes[k] = eNvFlexRigidJointModeFree;
					}
				}
				for (int i = 0; i < p_urdf->targets.size(); i++)
				{
					g_buffers->rigidJoints[p_urdf->activeJointNameMap[p_urdf->targets[i].jointName]].modes[eNvFlexRigidJointAxisTwist] = eNvFlexRigidJointModePosition;
					g_buffers->rigidJoints[p_urdf->activeJointNameMap[p_urdf->targets[i].jointName]].targets[eNvFlexRigidJointAxisTwist] = p_urdf->targets[i].angle;
					g_buffers->rigidJoints[p_urdf->activeJointNameMap[p_urdf->targets[i].jointName]].compliance[eNvFlexRigidJointAxisTwist] = p_urdf->targets[i].compliance;
				}
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control1"]].linearVel[1] = -Randf()*100.0f;
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control2"]].linearVel[1] = -Randf()*100.0f;
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control3"]].linearVel[1] = -Randf()*100.0f;
				//g_buffers->rigidBodies[p_urdf->rigidNameMap["control4"]].linearVel[1] = -Randf()*100.0f;
				controlIndices.push_back(p_urdf->rigidNameMap["control1"]);
				controlIndices.push_back(p_urdf->rigidNameMap["control2"]);
				controlIndices.push_back(p_urdf->rigidNameMap["control3"]);
				controlIndices.push_back(p_urdf->rigidNameMap["control4"]);
				phases.push_back(Randf() * k2Pi);
				phases.push_back(Randf() * k2Pi);
				phases.push_back(Randf() * k2Pi);
				phases.push_back(Randf() * k2Pi);

			}
		}
		for (int i = 0; i < g_buffers->rigidShapes.size(); i++)
		{
			g_buffers->rigidShapes[i].filter = 1;
		}

		// Lowered gravity
		g_params.gravity[1] = -1.0f;

		g_numSubsteps = 1;
		g_params.numIterations = 20;
		g_params.dynamicFriction = 1.0f;
		g_params.shapeCollisionMargin = 0.04f;
		g_params.collisionDistance = 0.01f;
		g_params.relaxationFactor = 1.0f;

		g_sceneLower = Vec3(-5.0f);
		g_sceneUpper = Vec3(5.0f);

		g_pause = true;
		g_drawPoints = false;
		g_drawCloth = false;
		g_drawCable = true;
		/*
		for (int i = 0; i < g_buffers->cableLinks.size(); i++) {
		g_buffers->cableLinks[i].stretchingDamping = 10.0f;
		g_buffers->cableLinks[i].compressionDamping = 10.0f;
		}
		for (int i = 0; i < 4; i++) {
		maxB[i] = g_buffers->cableLinks[3 + 5 * i].segLength;
		dd[i] = 0.0f;
		}
		*/

		if (!g_drawCable)
		{
			unsigned int maxCurveID = 0;
			for (int i = 0; i < g_buffers->cableLinks.size(); i++)
			{
				maxCurveID = std::max(maxCurveID, g_buffers->cableLinks[i].profileVerts);
			}
			maxCurveID++;
			profileCurves.resize(maxCurveID);
			NvFlexVector<Vec3> pos(g_flexLib, 100000);
			for (int i = 1; i < (int)maxCurveID; i++)
			{
				int numV = 0;
				NvFlexGetCurve(g_flexLib, i, pos.buffer, &numV);
				if (numV > 0)
				{
					profileCurves[i].resize(numV);
					pos.map();
					memcpy(&profileCurves[i][0], pos.mappedPtr, sizeof(Vec3)*numV);
					pos.unmap();
				}
			}

			ComputeCableSamples();
		}
		//di = 0;
	}
	float dd[4];
	float maxB[4];
	//float maxS[4];

	virtual void CenterCamera(void)
	{
		g_camPos = Vec3(0.0658046, 1.88832, 5.32831);
		g_camAngle = Vec3(0.00174534, -0.221657, 0);
	}

	virtual void Update()
	{
		/*
		for (int i = 0; i < 4; i++) {
		g_buffers->cableLinks[3 + 5 * i].segLength += dd[i];
		//g_buffers->cableLinks[26 + 4 * i].segLength -= dd[i];


		if (g_buffers->cableLinks[3 + 5 * i].segLength < 0.1f) g_buffers->cableLinks[3 + 5 * i].segLength = 0.1f;
		if (g_buffers->cableLinks[3 + 5 * i].segLength > maxB[i]) g_buffers->cableLinks[3 + 5 * i].segLength = maxB[i];
		cout << g_buffers->cableLinks[3 + 5 * i].segLength << " ";
		}*/
		//cout << "g_camPos = Vec3(" << g_camPos.x << ", " << g_camPos.y << ", " << g_camPos.z << ");" << endl;
		//cout << "g_camAngle = Vec3(" << g_camAngle.x << ", " << g_camAngle.y << ", " << g_camAngle.z << ");" << endl;
		static float time = 0.0f;
		time += g_dt;
		for (int i = 0; i < controlIndices.size(); i++)
		{
			g_buffers->rigidBodies[controlIndices[i]].linearVel[1] = sin(phases[i] + time)*10.0f;
		}

		if (!g_drawCable)
		{
			ComputeCableSamples();
		}
	}

	virtual void Sync()
	{
		if (!g_drawCable)
		{
			if (g_buffers->cableLinks.size())
			{
				NvFlexSetCableLinks(g_solver, g_buffers->cableLinks.buffer, g_buffers->cableLinks.size());
			}
		}
	}
	//int di;
	virtual void KeyDown(int key)
	{

		float sp = 0.01f;
		for (int i = 0; i < 4; i++) {
			if (key == '1' + i)
			{
				if (dd[i] > 0.0f) dd[i] = -sp; else
					if (dd[i] < 0.0f) dd[i] = 0.0f; else
						dd[i] = sp;
			}
		}
		Scene::KeyDown(key);
	}

	virtual void Draw(int pass)
	{
		if (!g_drawCable)
		{
			/*
			Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
			Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));
			Vec3 up = Cross(right, forward);
			float size = 0.05f;
			BeginLines(true);
			for (int i = 0; i < p_urdf->debugPoints.size(); i++)
			{
			Vec3 pos = p_urdf->debugPoints[i];
			DrawLine(pos - up*size, pos + up*size, p_urdf->debugPointsCols[i]);
			DrawLine(pos - right*size, pos + right*size, p_urdf->debugPointsCols[i]);
			}
			EndLines();
			*/
			//DrawPlanes(&p_urdf->debugPlanes[di], 1, 0.0f);
			BeginLines(true);
			Vec3 x = GetBasisVector0(gt.q);
			Vec3 y = GetBasisVector1(gt.q);
			Vec3 z = GetBasisVector2(gt.q);
			float len = 1.0f;
			DrawLine(gt.p, gt.p + len*x, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
			DrawLine(gt.p, gt.p + len*y, Vec4(0.0f, 1.0f, 0.0f, 1.0f));
			DrawLine(gt.p, gt.p + len*z, Vec4(0.0f, 0.0f, 1.0f, 1.0f));
			EndLines();

			//vector<vector<Transform> > segmentSamples;
			//vector<vector<Transform> > linkSamples;
			BeginLines(true);
			for (int i = 0; i < segmentSamples.size(); i++)
			{
				if (segmentSamples[i].size() > 1)
				{
					for (int j = 0; j < ((int)segmentSamples[i].size()) - 1; j++)
					{
						DrawLine(segmentSamples[i][j].p, segmentSamples[i][j + 1].p, Vec4(1.0f, 1.0f, 0.0f, 1.0f));
					}
				}
			}

			for (int i = 0; i < linkSamples.size(); i++)
			{
				if (linkSamples[i].size() > 1)
				{
					for (int j = 0; j < ((int)linkSamples[i].size()) - 1; j++)
					{
						//cout << linkSamples[i][j].p.x << " " << linkSamples[i][j].p.y << " " << linkSamples[i][j].p.z << endl;
						DrawLine(linkSamples[i][j].p, linkSamples[i][j + 1].p, Vec4(0.0f, 1.0f, 0.0f, 1.0f));
					}
				}
			}
			EndLines();
		}
	}

	virtual void DoStats()
	{
		/*
		Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
		Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));
		Vec3 up = Cross(right, forward);
		for (int i = 0; i < p_urdf->debugLabelPos.size(); i++)
		{
		Vec3 pos = p_urdf->debugLabelPos[i] + up*0.1f;
		Vec3 sc = GetScreenCoord(pos);
		if (sc.z < 1.0f)
		{
		DrawImguiString(int(sc.x + 5.f), int(sc.y - 5.f), Vec3(1.f, 1.f, 0.f), 0, "%s", p_urdf->debugLabelString[i].c_str());
		}
		}	*/
	}

	bool isZero(const Vec3 v)
	{
		return Dot(v, v) < 1e-10f;
	}

	void createFrame(const Vec3 d, Matrix33 &R, Matrix33* prevR = NULL)
	{
		R.cols[2] = Normalize(d);
		if (prevR && !isZero(Cross(R.cols[2], prevR->cols[0])))
		{
			R.cols[0] = prevR->cols[0];
		}
		else
		{
			if (fabs(d.x) < fabs(d.y) && fabs(d.x) < fabs(d.z))
			{
				R.cols[0] = Vec3(1.0f, 0.0f, 0.0f);
			}
			else if (fabs(d.y) < fabs(d.z))
			{
				R.cols[0] = Vec3(0.0f, 1.0f, 0.0f);
			}
			else
			{
				R.cols[0] = Vec3(0.0f, 0.0f, 1.0f);
			}
		}
		R.cols[1] = Normalize(Cross(R.cols[2], R.cols[0]));
		R.cols[0] = Cross(R.cols[1], R.cols[2]);
	}

	void ComputeCableSamples()
	{
		// segment sampling
		int numLinks = g_buffers->cableLinks.size();
		segmentSamples.resize(numLinks);
		linkSamples.resize(numLinks);

		for (int i = 0; i < numLinks; i++)
		{
			NvFlexCableLink &link = g_buffers->cableLinks[i];
			if (link.nextLink < 0) continue;

			NvFlexCableLink &next = g_buffers->cableLinks[link.nextLink];
			Vec3 p0 = link.att1;
			Vec3 p1 = next.att0;
			Transform btrans;
			Transform ntrans;

			if (link.body >= 0)
			{
				NvFlexGetRigidPose(&g_buffers->rigidBodies[link.body], (NvFlexRigidPose*)&btrans);
				p0 = TransformPoint(btrans, p0);
			}

			if (next.body >= 0)
			{
				NvFlexGetRigidPose(&g_buffers->rigidBodies[next.body], (NvFlexRigidPose*)&ntrans);
				p1 = TransformPoint(ntrans, p1);
			}

			Vec3 n = p1 - p0;
			float l = Length(n);
			n /= l;

			float d = link.segLength;

			if (params.visualBendingType == Params::NONE)
			{
				segmentSamples[i].resize(2);
				segmentSamples[i][0].p = p0;
				segmentSamples[i][1].p = p1;
			}
			else if (d <= l)
			{
				int num = std::max((int)ceil(link.segLength / params.visSamplingDist), 2);
				segmentSamples[i].resize(num + 1);
				float dx = l / (float)num;

				for (int j = 0; j <= num; j++)
					segmentSamples[i][j].p = p0 + j * dx * n;
			}
			else
			{
				int num = (int)ceil(l / params.visSamplingDist);
				segmentSamples[i].resize(num + 1);
				float h = std::min(0.5f * sqrtf(d * d - l * l), 0.2f * l);
				float dx = 1.0f / (float)num;
				for (int j = 0; j <= num; j++)
				{
					Vec3 p = p0 + j * dx * l * n;
					if (params.visualBendingType == Params::STIFF)
					{
						float x = 2 * j < num ? j * dx : 1.0f - j * dx;
						float y = 0.3f * h * (64.0f * x * x * x - 32.0f * x * x);
						if (2 * j < num) y = -y;
						Vec3 t(-n.y, n.x, 0.0f);
						p += t * y;
					}
					else
					{
						float x = j * dx;
						float y = p.y - h * (-4.0f * x * x + 4.0f * x);
						if (y < 0.0f)
							y = 0.0f;
						/*
						for (int l = 0; l < 2; l++) {
						Body *b = l ? link.body : next.body;
						float t;
						if (b && b->rayCast(p, Vec3(0.0f, -1.0f, 0.0f), t)) {
						float cy = p.y - t;
						if (t >= 0.0f && cy > y)
						y = cy;
						}
						}*/
						p.y = y;
					}
					segmentSamples[i][j].p = p;
				}
				for (int iter = 0; iter < 5; iter++)
				{
					for (int j = 1; j < num - 1; j++)
					{
						Vec3 avg = (segmentSamples[i][j - 1].p + segmentSamples[i][j + 1].p) * 0.5f;
						segmentSamples[i][j].p += (avg - segmentSamples[i][j].p) * 0.5f;
					}
				}
			}
		}

		// link sampling

		for (int i = 0; i < numLinks; i++)
		{
			NvFlexCableLink &link = g_buffers->cableLinks[i];
			if (!link.body || link.type != eNvFlexCableLinkTypeRolling)
				continue;
			Transform profilePose = (Transform&)link.profilePose;
			Transform iprofilePose = Inverse(profilePose);
			Vec3 profilePos0 = TransformPoint(iprofilePose, link.att0);
			Vec3 profilePos1 = TransformPoint(iprofilePose, link.att1);
			Transform bpose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[link.body], (NvFlexRigidPose*)&bpose);
			Transform pose = bpose * profilePose;

			Vec3 test0 = TransformPoint(pose, profilePos0);
			Vec3 test1 = TransformPoint(pose, profilePos1);

			linkSamples[i].clear();
			if (link.profileVerts == 0)
			{
				Vec3 n;
				n = profilePos0;
				if (link.prevLink < 0)
					n = profilePos1;

				n = Normalize(n);

				float phi = acosf(n.x); if (n.y < 0.0f) phi = 2.0f*kPi - phi;
				float dphi = link.linkLength / link.radius;
				if (link.flipped)
					dphi = -dphi;
				if (link.prevLink < 0)
					dphi = -dphi;
				float l = fabs(dphi * link.radius);
				int num = (int)ceil(l / params.visSamplingDist);
				dphi /= num;
				for (int j = 0; j <= num; j++)
				{
					Vec3 vr = link.radius * Vec3(cosf(phi + j * dphi), sinf(phi + j * dphi), 0.0f);
					linkSamples[i].push_back(Transform(TransformPoint(pose, vr)));
				}
			}
			else
			{
				int nr0 = -1;
				int nr1 = -1;
				float min0 = FLT_MAX;
				float min1 = FLT_MAX;
				vector<Vec3>& profileVerts = profileCurves[link.profileVerts];
				for (int i = 0; i < (int)profileVerts.size(); i++) {
					float d0 = LengthSq(profileVerts[i] - profilePos0);
					float d1 = LengthSq(profileVerts[i] - profilePos1);
					if (d0 < min0) { min0 = d0; nr0 = i; }
					if (d1 < min1) { min1 = d1; nr1 = i; }
				}
				float len = 0.0f;
				int prev = -1;
				int nr = nr0;
				while (prev != nr1)
				{
					linkSamples[i].push_back(Transform(TransformPoint(pose, profileVerts[nr])));
					prev = nr;
					nr = (nr + 1) % profileVerts.size();
				}
			}
		}

		// frames
		for (int i = 0; i < numLinks; i++)
		{
			NvFlexCableLink &link = g_buffers->cableLinks[i];
			for (int j = 0; j < 2; j++)
			{
				std::vector<Transform> &samples = j == 0 ? linkSamples[i] : segmentSamples[i];
				if (samples.size() < 2)
					continue;

				Matrix33 R = Matrix33::Identity();
				for (int k = 0; k < (int)samples.size() - 1; k++)
				{
					createFrame(samples[k + 1].p - samples[k].p, R, &R);
					samples[k].q = Quat(R);
				}
				samples.back().q = samples[samples.size() - 2].q;
			}
		}

		//for (int i = 0; i < (int)links.size(); i++) {
		//	Link &link = links[i];

		//	if (!link.segmentSampleNormals.empty() && !link.linkSampleNormals.empty()) {
		//		Vec3 &nl = link.linkSampleNormals.back();
		//		Vec3 &ns = link.segmentSampleNormals.front();
		//		Vec3 n = (nl + ns);
		//		n.normalize();
		//		nl = n;
		//		ns = n;
		//	}
		//	if (i < (int)links.size() - 1 || cyclic) {
		//		Link &next = links[(i + 1) % links.size()];
		//		if (!next.linkSampleNormals.empty() && !next.linkSampleNormals.empty()) {
		//			Vec3 &nl = next.linkSampleNormals.front();
		//			Vec3 &ns = link.segmentSampleNormals.back();
		//			Vec3 n = (nl + ns);
		//			n.normalize();
		//			nl = n;
		//			ns = n;
		//		}
		//	}
		//}
	}
};