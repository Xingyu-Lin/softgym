#pragma once
#include <iostream>
#include <vector>
#include "../urdf.h"
#include "../../external/tinyxml2/tinyxml2.h"


class Grasping : public Scene
{
public:
	bool useVisualMesh;
	// For running many copies of dexnet in parallel
	int numExpIncrement; // Number to increment once done (so basically, will run experiments, for al i, i*numExpIncrement + firstExperimentIndex
	
	enum MOTION_PHASE { APPROACH, GRIP, LIFT, SHAKE_ROT, SHAKE_TRA, WAIT, END };

	float offLen;
	URDFImporter* urdf;

	NvFlexVector<NvFlexRigidContact> rigidContacts;
	NvFlexVector<int> rigidContactCount;
	bool anyShapeUpdate;
	int numSimulationStepsPerRenderFrame;
	
	float gripperInitialWidth;
	float forceLimitInNewton;
	float gripperSqueezeSpeed;
	float gripThresholdInNewton;

	const float gripThreshold = 0.02f; // Threshold in which to consider gripper is closed

	const float dilation = 0.001f;
	const float thickness = 0.001f;
	vector<int> bkFilters;

	const float binSuccessThrshldDexnet = 0.002f; // Threshold for Dex-Net to classify a grasp as success for its robustness score
	const float binSuccessThrshldFlex = 0.5; // Threshold for Flex to classify a grasp as success for its proportion of successful lifts

	const int fps = 30;
	
	json grasps_json;

	class Grasp
	{
	public:
		Quat q;
		Vec3 p;
		float width;
		float robustness;
		int id;
		bool collides;
	};

	class StablePose
	{
	public:
		Quat q;
		float prob;
		vector<Grasp> grasps;
	};
	map<string, map<string, StablePose> > dexnets;

	class GraspExperiment
	{
	public:
		Transform offset;
		int objIndex;
		int effectorIndex;
		Transform objectInitPose;
		Transform graspInitPose;
		int lfingerIndex;
		int rfingerIndex;
		float lImpulse;
		float rImpulse;
		float cImpulse;
		int fingerJointIndex;
		int fingermJointIndex;

		int fingerLJointIndex;
		int fingermLJointIndex;
		int baseIndex;
		Vec3 liftDir;

		MOTION_PHASE motionPhase;
		int shakeStartFrame;
		int waitStartFrame;
		bool robotHitFloor;
		bool robotHitObject;
		float gripperWidth;
		string objName;
		string stpId;
		int graspNum;
		int gripperShapeBegin;
		int gripperShapeEnd;
		int objectShapeBegin;
		int objectShapeEnd;
		int frameCounter;
		float moveSpeed;
		bool badExperiment;
		float remainingDist;
		Grasp* grasp;

		int bad;
	};

	struct GraspExperimentTarget 
	{
		string objName;
		string stpId;
		int graspId;
	};
	vector<GraspExperimentTarget> graspTargets;
	bool useGraspTargets;

	// Shaking params
	bool doShake;
	int numShakes;
	float shakeAngle;
	float shakeTra;
	float shakeTime;
	vector<float> shakeDeltaAngles;
	vector<float> shakeDeltaTra;

	// For locating contacts
	vector<int> lfingerMap;
	vector<int> rfingerMap;
	vector<int> objMap;

	vector<GraspExperiment> experiments;
	void* floorCollideColorMaterial;
	void* shapeCollideColorMaterial;
	int frameSinceStart;
	int firstExperimentIndex;
	int currentExperimentIndex;
	bool foundExperiment; 
	int numExperiments;
	int numCurrentExperiments;
	int numPosOrientationPurturb;
	float posSigma;
	float rotSigma;
	float objComSigma;
	int numCols;		// Number of columns per row
	float spacing;   // Spacing of each gripper experiment in m
	int waitNumFrames; // Number of frames to hold obj before ending an experiment
	int maxExperimentFrames; // Max number of frames (length of time) an experiment can take. 
	int e;

	Vec3 jawCenterOffset;
	Transform flipUp;
	ofstream csv;

	float random_normal()
	{
		static int i = 1;
		static double u[2] = { 0.0, 0.0 };
		double r[2];

		if (i == 1)
		{
			float x = Randf();
			float y = Randf();
			if (x < 1e-19f)
			{
				x = 1.0f;
			}
			if (y < 1e-19f)
			{
				y = 1.0f;
			}
			r[0] = sqrt(-2 * log((double)x));
			r[1] = 2.0f * kPi*(double)y;
			u[0] = r[0] * sin(r[1]);
			u[1] = r[0] * cos(r[1]);
			i = 0;
		}
		else
		{
			i = 1;
		}

		return (float)u[i];
	};

	Vec3 random_normal_3d_isotropic()
	{
		return Vec3(random_normal(), random_normal(), random_normal());
	}

	Matrix33 expmSkew33(Matrix33 A)
	{
		/*
		Using https://math.stackexchange.com/questions/879351/matrix-exponential-of-a-skew-symmetric-matrix
		*/
		float a1 = A(2, 1);
		float a2 = A(0, 2);
		float a3 = A(1, 0);
		
		float x2 = pow(a1, 2) + pow(a2, 2) + pow(a3, 2);
		float x = sqrt(x2);

		return Matrix33::Identity() + sin(x) / x * A + (1 - cos(x))/x2 * A * A;
	}

	Matrix33 skew(Vec3 xi)
	{
		/*
		Skew of xi is:
		0    -xi[2]  xi[1]
		xi[2]   0   -xi[0]
		-xi[1] xi[0]   0
		Each argument of Matrix33 is a col, not a row, so construction appears transposed.
		*/
		return Matrix33(
			Vec3(.0f, xi[2], -xi[1]),
			Vec3(-xi[2], .0f, xi[0]),
			Vec3(xi[1], -xi[0], .0f)
		);
	}

	void perturbTransform(Transform &T, float traSigma, float rotSigma)
	{
		T.p += random_normal_3d_isotropic() * traSigma;
		
		Vec3 xi = random_normal_3d_isotropic() * rotSigma;
		Matrix33 R = expmSkew33(skew(xi));
		Quat rQaut = Quat(R);
		T.q = rQaut * T.q;
	}

	void perturbObjectCom(int shapeId, float comSigma)
	{
		NvFlexRigidShape shape = g_buffers->rigidShapes[shapeId];
		NvFlexRigidBody body = g_buffers->rigidBodies[shape.body];
		for (int i = 0; i < 3; i++)
			body.com[i] += random_normal() * comSigma;
	}

	float cosineInterpDelta(float T, float x0, float xT, float t)
	{
		/*
		Interpolates a cosine curve between x0 and xT within time length T. 
		Returns the time-derivative of point on curve at time t.
		*/
		return kPi / T * sin(kPi / T * t) * (xT - x0);
	}

	void ExtractJointCoordinates(NvFlexRigidJoint& joint, Vec3& pos, Vec3& apos)
	{
		NvFlexRigidBody& b0 = g_buffers->rigidBodies[joint.body0];
		NvFlexRigidBody& b1 = g_buffers->rigidBodies[joint.body1];

		Transform body0Pose;
		NvFlexGetRigidPose(&b0, (NvFlexRigidPose*)&body0Pose);
		Transform body1Pose;
		NvFlexGetRigidPose(&b1, (NvFlexRigidPose*)&body1Pose);

		Transform pose0 = body0Pose * Transform(joint.pose0.p, joint.pose0.q);
		Transform pose1 = body1Pose * Transform(joint.pose1.p, joint.pose1.q);
		Transform relPose = Inverse(pose0)*pose1;

		pos = relPose.p;
		Quat qd = relPose.q;

		Quat qtwist = Normalize(Quat(qd.x, 0.0f, 0.0f, qd.w));
		Quat qswing = qd*Inverse(qtwist);
		apos.x = asin(qtwist.x)*2.0f;
		apos.y = asin(qswing.y)*2.0f;
		apos.z = asin(qswing.z)*2.0f;
	}

	void Reset()
	{

	}
	
	void ReadExperimentsFromJson()
	{
		ifstream jsonFile("../../data/grasping/mug.json");
        jsonFile >> grasps_json;
		cout << "Object: " << grasps_json["object"] << endl;
		cout << "Object scale: " << grasps_json["object_scale"] << endl;
		cout << "Number of grasps: " << grasps_json["transforms"].size() << endl;
		
		map<string, StablePose> stablePoses;
		StablePose stablePose;
		for (int i = 0; i < 500; ++i)
		{
			Grasp grasp;
			grasp.collides = false;
			//grasp.width = (float)atof(gr->FirstChildElement("open_width")->FirstChild()->Value());
			//grasp.id = atoi(gr->FirstChildElement("id")->FirstChild()->Value());
			//grasp.robustness = (float)atof(gr->FirstChildElement("robustness_metric")->FirstChild()->Value());
			
			// convert transformation matrix to quaternion + translation vector
			std::vector<std::vector<float> > jm = grasps_json["transforms"][i];
			Matrix33 rot(Vec3(jm[0][0], jm[1][0], jm[2][0]), Vec3(jm[0][1], jm[1][1], jm[2][1]), Vec3(jm[0][2], jm[1][2], jm[2][2]));
			grasp.q = Quat(rot);
			grasp.p = Vec3(jm[0][3], jm[1][3], jm[2][3]);
			
			stablePose.grasps.push_back(grasp);
		}
		stablePoses["stable_pose_0"] = stablePose;
		dexnets[grasps_json["object"]] = stablePoses;
	}

	Grasping() : rigidContacts(g_flexLib, g_solverDesc.maxRigidBodyContacts), rigidContactCount(g_flexLib, 1)
	{
		useVisualMesh = true;

		if (useVisualMesh) 
		{
			renderMat = AddRenderMaterial(Vec3(.7f, .7f, .5f),0.1,0.0,true);
		}
		else
		{
			renderMat = AddRenderMaterial(Vec3(.7f, .7f, .5f));
		}
		// Command line
		numExpIncrement = 1;
		firstExperimentIndex = 0;
		for (int i = 1; i < g_argc; ++i)
		{
			int inc, idx;
			if (sscanf(g_argv[i], "-dexnet=%d,%d", &inc,&idx))
			{				
				numExpIncrement = inc;
				firstExperimentIndex = idx;
			}
		}

		// Setup simulation parameters
		frameSinceStart = 0;
		g_solverDesc.maxRigidBodyContacts = 256 * 1024 * 16;
		rigidContacts.map();
		rigidContacts.resize(g_solverDesc.maxRigidBodyContacts);
		rigidContacts.unmap();
		g_numSubsteps = 1;
		
		//g_params.numIterations = 200;
	//	g_params.numPostCollisionIterations = 50;
		
		// solver params
		g_numSubsteps = 2;
		g_params.numIterations = 6;
		g_params.numInnerIterations = 50;
		g_params.relaxationFactor = 0.75f;
		g_params.solverType = eNvFlexSolverPCR;
		g_params.frictionMode = eNvFlexFrictionModeFull;
		g_params.warmStart = 0.0f;
		
		g_params.gravity[0] = g_params.gravity[1] = g_params.gravity[2] = 0.0f;
		g_params.numPlanes = 0;
			
		g_params.dynamicFriction = 1.0f;
		g_params.staticFriction = 1.0f;
		g_params.shapeCollisionMargin = 0.001f;

		//g_params.relaxationFactor = 0.5f;
		//g_params.relaxationMode = eNvFlexRelaxationGlobal;
		//g_params.relaxationFactor = 0.25f;
		g_sceneLower = Vec3(-0.25f);
		g_sceneUpper = Vec3(0.25f);
		g_camSpeed = 0.005f;

		// Number of experiments
		//firstExperimentIndex = 10311+10133+321+218+2946+1254+1471+1786;		
		currentExperimentIndex = firstExperimentIndex;
		numExperiments = 1;
		numCurrentExperiments = 0;
		numPosOrientationPurturb = 500;
		posSigma = .0025f;
		rotSigma = .01f;
		objComSigma = .01f;
		//const float radius = 0.5f;

		numSimulationStepsPerRenderFrame = 1; // Number of simulation steps per rendering
		floorCollideColorMaterial = UnionCast<void*>(AddRenderMaterial(Vec3(1.0f,0.0f,0.0f), 0.1f, 0.6f));
		shapeCollideColorMaterial = UnionCast<void*>(AddRenderMaterial(Vec3(1.0f, 0.5f, 0.0f), 0.1f, 0.6f));
		// Setup gripping parameters
		offLen = 0.164f;
		gripperInitialWidth = 0.0307;
		gripperInitialWidth += thickness;
		forceLimitInNewton = 10.0f; // N
		gripperSqueezeSpeed = 0.02f; // m/s
		gripThresholdInNewton = 20.0f; // N

		// Experiment parameters
		numCols = 20;		// Number of columns per row
		spacing = 1.0f;   // Spacing of each gripper experiment in m
		waitNumFrames = 5 * fps; // n seconds * fps
		maxExperimentFrames = 30 * fps; // n seconds * fps

		jawCenterOffset = Vec3(0.0f, 0.0f, offLen);
		e = 0;

		anyShapeUpdate = true;
		experiments.clear();
		
		ReadExperimentsFromJson();
		
		/*******************************
		tinyxml2::XMLDocument doc;
		//doc.LoadFile(GetFilePathByPlatform("../../data/dex-net/mini_dexnet/mini_dexnet_all_grasps_new.xml").c_str());
		//doc.LoadFile(GetFilePathByPlatform("../../data/dex-net/mini_dexnet/adversarial_all_grapsps.xml").c_str());
		//doc.LoadFile(GetFilePathByPlatform("../../data/dex-net/mini_dexnet/adversarial_all_grapsps_prune.xml").c_str());
		//doc.LoadFile(GetFilePathByPlatform("../../data/dex-net/mini_dexnet/adversarial_all_grasps_july_19_2018.xml").c_str());
		doc.LoadFile(GetFilePathByPlatform("../../data/grasping/test.xml").c_str());
		
		XMLElement* root = doc.RootElement();
		XMLElement* ele = root->FirstChildElement();
		while (ele)
		{
			map<string, StablePose> stablePoses;
			cout << ele->Name() << endl;
			XMLElement* stb = ele->FirstChildElement();
			while (stb)
			{
				cout << "    " << stb->Name() << endl;
				StablePose stablePose;
				XMLElement* gr = stb->FirstChildElement("grasps");
				while (gr)
				{
					Grasp grasp;
					grasp.collides = strcmp(gr->FirstChildElement("collides")->FirstChild()->Value(), "true") == 0;
					grasp.width = (float)atof(gr->FirstChildElement("open_width")->FirstChild()->Value());
					grasp.id = atoi(gr->FirstChildElement("id")->FirstChild()->Value());
					grasp.robustness = (float)atof(gr->FirstChildElement("robustness_metric")->FirstChild()->Value());
					Quat gq;
					XMLElement* q = gr->FirstChildElement("orientation");
					gq.w = (float)atof(q->FirstChild()->Value());
					q = q->NextSiblingElement("orientation");
					gq.x = (float)atof(q->FirstChild()->Value());
					q = q->NextSiblingElement("orientation");
					gq.y = (float)atof(q->FirstChild()->Value());
					q = q->NextSiblingElement("orientation");
					gq.z = (float)atof(q->FirstChild()->Value());

					Vec3 gp;
					XMLElement* p = gr->FirstChildElement("translation");
					gp.x = (float)atof(p->FirstChild()->Value());
					p = p->NextSiblingElement("translation");
					gp.y = (float)atof(p->FirstChild()->Value());
					p = p->NextSiblingElement("translation");
					gp.z = (float)atof(p->FirstChild()->Value());
					grasp.p = gp;
					grasp.q = gq;
					//if (!grasp.collides)
					//{
					stablePose.grasps.push_back(grasp);
					//}

					gr = gr->NextSiblingElement("grasps");
				}

				XMLElement* q = stb->FirstChildElement("orientation");
				Quat mq;
				mq.w = (float)atof(q->FirstChild()->Value());
				q = q->NextSiblingElement("orientation");
				mq.x = (float)atof(q->FirstChild()->Value());
				q = q->NextSiblingElement("orientation");
				mq.y = (float)atof(q->FirstChild()->Value());
				q = q->NextSiblingElement("orientation");
				mq.z = (float)atof(q->FirstChild()->Value());
				stablePose.q = mq;
				XMLElement* p = stb->FirstChildElement("probability");
				stablePose.prob = (float)atof(p->FirstChild()->Value());
				stablePoses[stb->Name()] = stablePose;
				stb = stb->NextSiblingElement();
			}
			
			dexnets[ele->Name()] = stablePoses;

			ele = ele->NextSiblingElement();
		}
		*******************************/

		//urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi_gripper_nov17_mod.urdf", true, dilation, thickness);
		//urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi_gripper_nov17_with_visual.urdf", true, dilation, thickness);
		urdf = new URDFImporter("../../data/", "yumi_description/urdf/yumi_gripper.urdf", true, dilation, thickness);
		flipUp = Transform(Vec3(), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), -kPi*0.5f));
		//flipUp = Transform(Vec3(), Quat());
		int num = 0;
		char fname[1000];
		while (1)
		{
			sprintf(fname, "/flex_gym_io/newton_grasp_idx_%02d_num_%04d.csv", firstExperimentIndex, num);
			//sprintf(fname, "grasp_idx_%02d_num_%04d.csv", firstExperimentIndex, num);
			FILE* f = fopen(fname, "rt");
			if (!f)
			{
				break;
			}
			else
			{
				fclose(f);
			}
			num++;
		}
		csv.open(fname);
		string csv_header = "obj_name,stp_id,grasp_id,grasp_num,num_trials,num_success,num_failure,robustness,collision,bin_robustness,bin_25,bin_50,bin_75";
		csv << csv_header << endl;
		csv.flush();

		// Add specific grasps to evaluate

		/* sanity */
		/*
		graspTargets = {
			{"bar_clamp", "stable_pose_0", 181},
			{"bar_clamp", "stable_pose_1", 248},
			{"bar_clamp", "stable_pose_3", 73}
		};
		*/
		/*
		// Good
		graspTargets = {
			{ "bar_clamp","stable_pose_0",	181},
		{"endstop_holder","stable_pose_0",	218 },
		{"endstop_holder","stable_pose_4",	212 },
		{"gearbox","stable_pose_0",	12 },
		{"gearbox","stable_pose_0",	121 },
		{"gearbox","stable_pose_2",	520 },
		{"mount1","stable_pose_3",	67 },
		{"mount2","stable_pose_0",	597 },
		{"mount2","stable_pose_2",	655 },
		{"mount2","stable_pose_9",	727 }
		};
		*/

		/*
		// Borderline
		graspTargets = {
			{ "bar_clamp","stable_pose_3",	73 },
		{ "endstop_holder","stable_pose_1",	41},
		{ "endstop_holder","stable_pose_5",	173},
		{ "gearbox","stable_pose_3",	4},
		{ "gearbox","stable_pose_3",	673},
		{ "gearbox","stable_pose_5",	298},
		{ "mount1","stable_pose_0",	28},
		{ "mount1","stable_pose_0",	326},
		{ "mount1","stable_pose_1",	225},
			{ "mount1","stable_pose_6",	642},
		{ "mount2","stable_pose_1",	875},
		{ "mount2","stable_pose_1",	255},
		{ "mount2","stable_pose_8",	162}
		};

		// Failure
		graspTargets = {
			{"bar_clamp","stable_pose_1",	248 },
		{"bar_clamp","stable_pose_2",	151},
		{"endstop_holder","stable_pose_2",	126},
		{"gearbox","stable_pose_1",	50},
		{"gearbox","stable_pose_4",	938},
		{"mount1","stable_pose_0",	154},
		{"mount1","stable_pose_4",	85},
		{"mount1","stable_pose_5",	154},
		{"mount1","stable_pose_5",	310},
		{"mount2","stable_pose_0",	881},
		{"mount2","stable_pose_4",	915}
		};*/


		/*
		graspTargets = {
			{ "/home/clemens/data/shapenet/models_selected/03797390/1bc5d303ff4d6e7e1113901b72a68e7c/model.obj", "stable_pose_0", 0 },
			{ "bar_clamp","stable_pose_0",	181 },
			{ "endstop_holder","stable_pose_0",	218 },
			{ "endstop_holder","stable_pose_4",	212 },
			{ "gearbox","stable_pose_0",	12 },
			{ "gearbox","stable_pose_0",	121 },
			{ "gearbox","stable_pose_2",	520 },
			{ "mount1","stable_pose_3",	67 },
			{ "mount2","stable_pose_0",	597 },
			{ "mount2","stable_pose_2",	655 },
			{ "mount2","stable_pose_9",	727 },
			{ "bar_clamp","stable_pose_3",	73 },
			{ "endstop_holder","stable_pose_1",	41 },
			{ "endstop_holder","stable_pose_5",	173 },
			{ "gearbox","stable_pose_3",	4 },
			{ "gearbox","stable_pose_3",	673 },
			{ "gearbox","stable_pose_5",	298 },
			{ "mount1","stable_pose_0",	28 },
			{ "mount1","stable_pose_0",	326 },
			{ "mount1","stable_pose_1",	225 },
			{ "mount1","stable_pose_6",	642 },
			{ "mount2","stable_pose_1",	875 },
			{ "mount2","stable_pose_1",	255 },
			{ "mount2","stable_pose_8",	162 },
			{ "bar_clamp","stable_pose_1",	248 },
			{ "bar_clamp","stable_pose_2",	151 },
			{ "endstop_holder","stable_pose_2",	126 },
			{ "gearbox","stable_pose_1",	50 },
			{ "gearbox","stable_pose_4",	938 },
			{ "mount1","stable_pose_0",	154 },
			{ "mount1","stable_pose_4",	85 },
			{ "mount1","stable_pose_5",	154 },
			{ "mount1","stable_pose_5",	310 },
			{ "mount2","stable_pose_0",	881 },
			{ "mount2","stable_pose_4",	915 }
		};
		*/
		
		graspTargets = {
		{"bar_clamp","stable_pose_0",	181},
		{"bar_clamp","stable_pose_1",	151 },
		{"bar_clamp","stable_pose_2",	372 },
		{"bar_clamp","stable_pose_3",	60 },
		{"bar_clamp","stable_pose_4",	91 } };
		
		/*
		graspTargets = {
		{"bar_clamp","stable_pose_4",	223 },
		{"gearbox","stable_pose_0",	48},
		{"gearbox","stable_pose_0",	396},
		{"gearbox","stable_pose_2",	620},
		{"gearbox","stable_pose_8",	105},
		{"mount1","stable_pose_2",	83},
		{"mount2","stable_pose_0",	65},
		{"mount2","stable_pose_0",	177},
		{"mount2","stable_pose_2",	130},
		{"mount2","stable_pose_3",	76},
		{"nozzle","stable_pose_5",	103},
		{"part1","stable_pose_2",	259},
		{"part1","stable_pose_6",	410},
		{"part1","stable_pose_7",	289},
		{"part3","stable_pose_0",	295},
		{"part3","stable_pose_8",	497},
		{"pawn","stable_pose_2",	43},
		{"pipe_connector","stable_pose_0",	350},
		{"pipe_connector","stable_pose_0",	793},
		{"turbine_housing","stable_pose_1",	660}
		};
		*/
		/*graspTargets = {
			{ "gearbox", "stable_pose_8", 105 },
			{ "mount1", "stable_pose_2", 130 }
		};
		*/
		/*
		graspTargets = {				
			{ "pawn", "stable_pose_0", 368 },	
			{ "bar_clamp", "stable_pose_0", 114 },
			{ "mount2", "stable_pose_4", 240 },
			{ "nozzle", "stable_pose_2", 94 },
			{ "part3", "stable_pose_2", 167 },
			// Objs w/ large moment arm not rotating after grasp
			{ "endstop_holder", "stable_pose_6", 0 },
			{ "part1", "stable_pose_2", 72 },
			// Gripper pushing into object
			{ "endstop_holder", "stable_pose_5", 27 },
			// Grasps w/ small to no contact areas that still lifts/moves objects
			{ "endstop_holder", "stable_pose_1", 3 },
			// Objs moving at beginning
			{ "part1", "stable_pose_2", 172 },
			{ "pipe_connector", "stable_pose_2", 0 },
			{ "pipe_connector", "stable_pose_9", 3 },			
		};*/
		useGraspTargets = true;
		
		/* A shake is rotating the obj around the grasp axis:
		first forward by shakeAngle/2,
		then backwards by shakeAngle,
		then forward again for shakeAngle/2 to return object to original grasp pose.
		*/
		numShakes = 3;
		shakeAngle = kPi / 16; // total angle to rotate around the grasp axis. this should be positive
		shakeTra = 0.01f;
		shakeTime = 2.0f; // number of seconds to complete a shake
		doShake = true;

		// Generate shake delta angles if needed. Angles follow a cosine curve for smooth motion. 
		if (doShake)
		{
			int singleShakeHalfTime = int(shakeTime / 4 * fps);
			float halfShakeAngle = shakeAngle / 2;
			
			// First forward by half
			for (int t = 0; t < singleShakeHalfTime; t++)
			{
				shakeDeltaAngles.push_back(cosineInterpDelta(float(singleShakeHalfTime), 0.f, halfShakeAngle, float(t)));
				shakeDeltaTra.push_back(cosineInterpDelta(float(singleShakeHalfTime), 0.f, shakeTra, float(t)));
			}
				

			// Intermediate "full swings"
			for (int i = 0; i < 2 * numShakes - 1; i++)
			{
				int dir = int(pow(-1, i + 1)); // flip direction
				for (int t = 0; t < 2 * singleShakeHalfTime; t++)
				{
					shakeDeltaAngles.push_back(cosineInterpDelta(2.f * singleShakeHalfTime, -dir * halfShakeAngle, dir * halfShakeAngle, float(t)));
					shakeDeltaTra.push_back(cosineInterpDelta(2.f * singleShakeHalfTime, -dir * shakeTra, dir * shakeTra, float(t)));
				}
					
			}

			// Last forward by half
			for (int t = 0; t < singleShakeHalfTime; t++)
			{
				shakeDeltaAngles.push_back(cosineInterpDelta(float(singleShakeHalfTime), -halfShakeAngle, 0.f, float(t)));
				shakeDeltaTra.push_back(cosineInterpDelta(float(singleShakeHalfTime), -shakeTra, 0.f, float(t)));
			}
				
		}

		ResetScene();
	}

	void ResetScene()
	{
		experiments.clear();
		e = 0;
		frameSinceStart = 0;
		//CHECK IF GROUND PENETRATION OCCURS, IF SO, MARK AS BAD RIGHT AWAY
		g_buffers->rigidBodies.resize(0);
		g_buffers->rigidShapes.resize(0);
		g_buffers->rigidJoints.resize(0);
		g_renderAttachments.resize(0);
		int num = 0;
		cout << "Current experiment = " << currentExperimentIndex << endl;

		string objName;
		string stpId;
		size_t graspIndex;
		foundExperiment = false;
		if (useGraspTargets)
		{
			if (currentExperimentIndex < (int)graspTargets.size())
			{
				GraspExperimentTarget currentTarget = graspTargets[currentExperimentIndex];
				
				objName = currentTarget.objName;
				stpId = currentTarget.stpId;
				for (int i = 0; i < (int)dexnets[objName][stpId].grasps.size(); i++)
				{
					if (currentTarget.graspId == dexnets[objName][stpId].grasps[i].id)
					{
						cout << "Found " << currentTarget.objName << " " << currentTarget.stpId << " " << currentTarget.graspId << endl;
						graspIndex = i;
						foundExperiment = true;
						break;
					}
				}
				if (!foundExperiment) {
					cout << "Not found " << currentTarget.objName << " " << currentTarget.stpId << " " << currentTarget.graspId << endl;
					cout << "Number of grasps for stable pose id and object: " << dexnets[objName][stpId].grasps.size() << endl;
				}
			}
		}
		else
		{
			for (auto objI : dexnets)
			{
				for (auto stablePose : objI.second)
				{
					for (size_t g = 0; g < stablePose.second.grasps.size(); g++)
					{
						if (e == currentExperimentIndex)
						{
							objName = objI.first;
							stpId = stablePose.first.c_str();
							graspIndex = g;
							foundExperiment = true;
							break;
						}
						e++;
					}
					if (foundExperiment) break;
				} 
				if (foundExperiment) break;
			} 
		}

		if (!foundExperiment)
		{
			// Done!
			exit(0);
		}

		if (foundExperiment)
		{
			/*
			const char * names[] =
			{
				"bar_clamp.obj",
				"climbing_hold.obj",
				"endstop_holder.obj",
				"gearbox.obj",
				"mount1.obj",
				"mount2.obj",
				"nozzle.obj",
				"part1.obj",
				"part3.obj",
				"pawn.obj",
				"pipe_connector.obj",
				"turbine_housing.obj",
				"turbine_housing_original.obj",
				"vase.obj" };
			*/
			char objPath[1000];
			//sprintf(objPath, "../../data/dex-net/mini_dexnet/%s.obj", objName.c_str());
			//sprintf(objPath, "/home/clemens/data/shapenet/models_selected/03797390/1f035aa5fc6da0983ecac81e09b15ea9/%s.obj", objName.c_str());
			//sprintf(objPath, "%s", grasps_json["object"].get<std::string>().c_str());
			sprintf(objPath, "../../data/mug.obj");
			cout << "Object mesh file: " << objPath << endl;
			
			for (int i = 0; i < numPosOrientationPurturb; i++)
			{
				//sprintf(objPath, "../../data/dex-net/mini_dexnet/%s", names[i % 14]);
				GraspExperiment ge;
				ge.bad = 0;
				ge.robotHitFloor = false;
				ge.robotHitObject = false;
				ge.badExperiment = false;
				ge.moveSpeed = 0.0f;
				ge.offset = Transform(Vec3((num % numCols)*spacing, 2.0f, (num / numCols)*spacing), Quat());// * flipUp;
				ge.grasp = &dexnets[objName][stpId].grasps[graspIndex];
				ge.objName = objName;
				ge.stpId = stpId;
				ge.graspNum = graspIndex;
				ge.gripperWidth = gripperInitialWidth;

				// Sample grasp pose
				Transform graspRelPose = Transform(dexnets[objName][stpId].grasps[graspIndex].p, dexnets[objName][stpId].grasps[graspIndex].q);
				//Transform graspRelPose = Transform(Vec3(0, 0, -0.0f), dexnets[objName][stpId].grasps[graspIndex].q);
				//Transform graspRelPose = Transform(Vec3());
				//perturbTransform(graspRelPose, posSigma, rotSigma);

				// Sample obj pose
				Transform objectInitPose;
				Transform objectInitPoseLocal;
				//objectInitPoseLocal = Transform(Vec3(), dexnets[objName][stpId].q);
				Transform rotateUp(Vec3(), QuatFromAxisAngle(Vec3(1.0f, 0.0f, 0.0f), +kPi*0.5f));
				objectInitPoseLocal = rotateUp * Inverse(Transform(dexnets[objName][stpId].grasps[graspIndex].p, dexnets[objName][stpId].grasps[graspIndex].q));
				objectInitPose = ge.offset*objectInitPoseLocal;
				//perturbTransform(objectInitPose, posSigma, rotSigma);

				//float x = 0.0f;
				//float dx = 0.2f;

				Vec3 mins, maxs;
				ge.objectShapeBegin = g_buffers->rigidShapes.size();

				//cout << "Loading " << objPath << endl;						

				LoadSimpleOBJ(objPath, objectInitPose, &mins, &maxs);
				// Sample obj com
				// perturbObjectCom(g_buffers->rigidShapes.size() - 1, objComSigma);

				ge.objectShapeEnd = g_buffers->rigidShapes.size();

				ge.objectInitPose = objectInitPose;
				ge.graspInitPose = ge.offset*objectInitPoseLocal*graspRelPose;
				ge.objIndex = g_buffers->rigidBodies.size() - 1;
				g_buffers->rigidBodies[ge.objIndex].com[1] -= mins.y;
				ge.objectInitPose.p.y -= mins.y;
				ge.graspInitPose.p.y -= mins.y;
				ge.motionPhase = APPROACH;
				ge.liftDir = Rotate(ge.graspInitPose.q, Vec3(0.0f, 0.0f, -1.0f));

				//g_buffers->rigidBodies.back().invMass = 0.0;
				int rbegin = g_buffers->rigidBodies.size();
				ge.gripperShapeBegin = g_buffers->rigidShapes.size();
				//ge.graspInitPose.p += TransformVector(ge.graspInitPose, -jawCenterOffset);
				//ge.graspInitPose.p += ge.liftDir*0.12f;
				//ge.remainingDist = 0.12f;
				ge.remainingDist = 0.0f;

				urdf->AddPhysicsEntities(ge.graspInitPose, renderMat, true, 1000.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f, true, 0.0f, 0.f, 0.001f);
				ge.gripperShapeEnd = g_buffers->rigidShapes.size();
				int rend = g_buffers->rigidBodies.size();

				for (int i = rbegin; i < rend; ++i)
				{
					g_buffers->rigidBodies[i].invMass *= 0.1f;

					(Matrix33&)g_buffers->rigidBodies[i].invInertia = Matrix33::Identity() * 0.1f;
				}

				ge.frameCounter = 0;
				ge.baseIndex = urdf->rigidNameMap["gripper_r_base"];

				NvFlexRigidBody& gripperBase = g_buffers->rigidBodies[ge.baseIndex];
				float factor = 1.0f;
				gripperBase.mass *= factor;
				gripperBase.invMass /= factor;
				(Matrix33&)gripperBase.inertia *= factor;
				(Matrix33&)gripperBase.invInertia *= (1.0f / factor);

				//NvFlexRigidJoint handRight = g_buffers->rigidJoints[urdf->jointNameMap["gripper_r_joint"]];
				// set up end effector targets
				NvFlexRigidJoint effectorJoint0;
				//NvFlexMakeFixedJoint(&effectorJoint0, -1, handRight.body0, NvFlexMakeRigidPose(Vec3(0.0f, 0.3f, 0.0f), Quat()), NvFlexMakeRigidPose(Vec3(0.0f), Quat()));
				NvFlexMakeFixedJoint(&effectorJoint0, -1, ge.baseIndex, NvFlexMakeRigidPose(ge.graspInitPose.p, ge.graspInitPose.q), NvFlexMakeRigidPose(Vec3(), Quat()));

				for (int i = 0; i < 6; ++i)
				{
					effectorJoint0.compliance[i] = 0.f; //1.0f / (50000.0f); zeroed out so shaking motion is predictable
					effectorJoint0.damping[i] = 10.0f;
				}

				ge.effectorIndex = g_buffers->rigidJoints.size();
				g_buffers->rigidJoints.push_back(effectorJoint0);

				ge.lfingerIndex = urdf->rigidNameMap["gripper_r_finger_l"];
				ge.rfingerIndex = urdf->rigidNameMap["gripper_r_finger_r"];
				ge.cImpulse = 0.0f;
				ge.lImpulse = 0.0f;
				ge.fingerJointIndex = urdf->activeJointNameMap["gripper_r_joint"];
				ge.fingermJointIndex = urdf->activeJointNameMap["gripper_r_joint_m"];
				ge.fingerLJointIndex = urdf->jointNameMap["gripper_r_joint"];
				ge.fingermLJointIndex = urdf->jointNameMap["gripper_r_joint_m"];

				NvFlexRigidJoint& rj = g_buffers->rigidJoints[ge.fingerJointIndex];
				NvFlexRigidJoint& rjm = g_buffers->rigidJoints[ge.fingermJointIndex];
				rj.targets[eNvFlexRigidJointAxisX] = -gripperSqueezeSpeed;//ge.gripperWidth;
				rjm.targets[eNvFlexRigidJointAxisX] = -gripperSqueezeSpeed;//ge.gripperWidth;
				rj.targets[eNvFlexRigidJointAxisX] = 0.0f;//ge.gripperWidth;
				rjm.targets[eNvFlexRigidJointAxisX] = 0.0f;//ge.gripperWidth;
				rj.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeVelocity;
				rjm.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeVelocity;
				rj.compliance[eNvFlexRigidJointAxisX] = 0.0f;
				rjm.compliance[eNvFlexRigidJointAxisX] = 0.0f;

				rj.motorLimit[eNvFlexRigidJointAxisX] = forceLimitInNewton;
				rjm.motorLimit[eNvFlexRigidJointAxisX] = forceLimitInNewton;

				map<string, Transform> fingersTransMap;
				fingersTransMap["gripper_r_finger_l"] = Transform(Vec3(ge.gripperWidth, 0.0f, 0.0f));
				fingersTransMap["gripper_r_finger_r"] = Transform(Vec3(ge.gripperWidth, 0.0f, 0.0f));


				urdf->setReducedPose(fingersTransMap);
				experiments.push_back(ge);
				num++;

				int total = dexnets[objName].size();
				int rr = rand() % total;
				int tt = 0;
				
				for (auto e : dexnets[objName]) {
					if (tt == rr) {
						stpId = e.first;
					}
					tt++;
				}
				graspIndex = rand() % dexnets[objName][stpId].grasps.size();
		
			}

			//cout << "Created " << experiments.size() << " experiments" << endl;
			lfingerMap.clear();
			rfingerMap.clear();
			objMap.clear();
			lfingerMap.resize(g_buffers->rigidBodies.size(), -1);
			rfingerMap.resize(g_buffers->rigidBodies.size(), -1);
			objMap.resize(g_buffers->rigidBodies.size(), -1);

			for (size_t e = 0; e < experiments.size(); e++)
			{
				lfingerMap[experiments[e].lfingerIndex] = e;
				rfingerMap[experiments[e].rfingerIndex] = e;
				objMap[experiments[e].objIndex] = e;
			}

			// set friction on all shapes
			bkFilters.resize(g_buffers->rigidShapes.size());
			for (int i = 0; i < g_buffers->rigidShapes.size(); i++)
			{
				NvFlexRigidShape& shape = g_buffers->rigidShapes[i];
				/*
				shape.material.friction = 1.0f;
				shape.material.torsionFriction = 0.1f;
				shape.material.rollingFriction = 0.01f;
				shape.material.restitution = 0.0f;
				shape.material.compliance = 0.0f;
				*/
				bkFilters[i] = shape.filter;
				//shape.filter = 1;
			}

			for (size_t e = 0; e < experiments.size(); e++)
			{
				GraspExperiment& ge = experiments[e];
				//cout << "Mass for " << e << " is " << g_buffers->rigidBodies[ge.objIndex].mass << endl;
				// Object properties
				for (int i = ge.objectShapeBegin; i < ge.objectShapeEnd; i++)
				{
					NvFlexRigidShape& shape = g_buffers->rigidShapes[i];
					shape.material.friction = 1.0f;
					shape.material.torsionFriction = 0.5f;
					shape.material.rollingFriction = 0.5f;
					shape.material.restitution = 0.0f;
					shape.material.compliance = 0.0f;
				}

				// Gripper properties
				for (int i = ge.gripperShapeBegin; i < ge.gripperShapeEnd; i++)
				{
					NvFlexRigidShape& shape = g_buffers->rigidShapes[i];
					shape.material.friction = 7.0f;
					shape.material.torsionFriction = 0.5f;
					shape.material.rollingFriction = 0.5f;
					shape.material.restitution = 0.0f;
					shape.material.compliance = 0.0f;
				}
			}

			for (int i = 0; i < g_buffers->rigidBodies.size(); i++)
			{
				NvFlexRigidBody& body = g_buffers->rigidBodies[i];
				body.angularDamping = 0.0f;
				body.linearDamping = 0.0f;
				(Vec3&)body.linearVel = Vec3(0.0f, 0.0f, 0.0f);
				(Vec3&)body.angularVel = Vec3(0.0f, 0.0f, 0.0f);
			}

			g_pause = true;
		}
	}

	map<string, pair<Mesh*, NvFlexTriangleMeshId> > objFileMap;
	map<string, pair<Mesh*, RenderMesh*> > visualMeshMap;
	int renderMat;

	void LoadSimpleOBJ(const char* path, const Transform& transform,Vec3* mins = NULL, Vec3* maxs = NULL)
	{
		map<string, pair<Mesh*, NvFlexTriangleMeshId> >::iterator oit = objFileMap.find(path);
		if (oit == objFileMap.end())
		{
			Mesh* m = ImportMesh(path);
			m->CalculateFaceNormals();
			
			NvFlexTriangleMeshId mesh = CreateTriangleMesh(m, dilation);
			objFileMap[path] = make_pair(m, mesh);
		}
		Mesh* m = objFileMap[path].first;
		NvFlexTriangleMeshId mesh = objFileMap[path].second;
		if (mins && maxs)
		{
			mins->x = 1e20f;
			mins->y = 1e20f;
			mins->z = 1e20f;
			maxs->x = -1e20f;
			maxs->y = -1e20f;
			maxs->z = -1e20f;
			for (size_t i = 0; i < m->m_positions.size(); i++)
			{
				Vec3 p = TransformPoint(transform, ((Vec3&)m->m_positions[i]));
				if (p.x < mins->x)
				{
					mins->x = p.x;
				}
				if (p.y < mins->y)
				{
					mins->y = p.y;
				}
				if (p.z < mins->z)
				{
					mins->z = p.z;
				}
				if (p.x > maxs->x)
				{
					maxs->x = p.x;
				}
				if (p.y > maxs->y)
				{
					maxs->y = p.y;
				}
				if (p.z > maxs->z)
				{
					maxs->z = p.z;
				}
			}
		}
		
		float obj_scale_f = grasps_json["object_scale"];
		Vec3 obj_scale(obj_scale_f, obj_scale_f, obj_scale_f);
		NvFlexRigidShape shape;
		NvFlexMakeRigidTriangleMeshShape(&shape, g_buffers->rigidBodies.size(), mesh, NvFlexMakeRigidPose(0, 0), obj_scale[0], obj_scale[1], obj_scale[2]);
		shape.filter = 0;
		shape.material.friction = 1.0f;
		shape.thickness = thickness;
		shape.user = UnionCast<void*>(renderMat);
		const float density = 200.0f / 1.34906f;
		NvFlexRigidBody body;
		NvFlexMakeRigidBody(g_flexLib, &body, transform.p, transform.q, &shape, &density, 1);
		//cout << "Mass = " << body.mass << endl;
		g_buffers->rigidShapes.push_back(shape);
		g_buffers->rigidBodies.push_back(body);

		if (useVisualMesh) 
		{
			map<string, pair<Mesh*, RenderMesh*> >::iterator it = visualMeshMap.find(path);
			Mesh* hostMesh = NULL;
			RenderMesh* renderMesh = NULL;

			if (it == visualMeshMap.end()) 
			{
				hostMesh = ImportMesh(path);
				hostMesh->Transform(ScaleMatrix(obj_scale));
				hostMesh->CalculateFaceNormals();
				renderMesh = CreateRenderMesh(hostMesh);
				visualMeshMap[path] = make_pair(hostMesh, renderMesh);
			}
			else {
				hostMesh = visualMeshMap[path].first;
				renderMesh = visualMeshMap[path].second;				
			}
			
			RenderMaterial renderMaterial;
			renderMaterial.frontColor = Vec3(1.0f, 0.0f, 0.0f);
			renderMaterial.backColor = Vec3(1.0f, 0.0f, 0.0f);
			renderMaterial.specular = 1.0f;
			renderMaterial.roughness = 0.5f;
			renderMaterial.metallic = 0.0f;
			renderMaterial.colorTex = NULL;

			// construct render batch for this mesh/material combination
			RenderAttachment attach;
			attach.parent = g_buffers->rigidBodies.size()-1;
			attach.material = renderMaterial;
			attach.mesh = renderMesh;
			attach.origin = Transform(Vec3(), Quat());
			attach.startTri = 0;
			attach.endTri = 0;

			g_renderAttachments.push_back(attach);
		}

	}

	virtual void CenterCamera(void)
	{
		// Good for 3 or 100
		g_camPos = Vec3(-0.409816f, 0.212568f, 0.0813254f);
		g_camAngle = Vec3(11.0916f, -0.308923f, 0.f);

		// Good for 500
		//g_camPos = Vec3(10.291087f, 1.282653f, 10.791869f);
		//g_camAngle = Vec3(12.4006f, -0.172787f, 0.f);
	}

	~Grasping()
	{
	}

	virtual bool IsSkipSimulation()
	{
		return true;
	}

	virtual void DoStats()
	{
		for (size_t e = 0; e < experiments.size(); e++)
		{
			GraspExperiment& ge = experiments[e];
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.baseIndex], (NvFlexRigidPose*)&pose);
			Vec3 pos(0.0f, 0.0f, -0.02f);
			pos = TransformPoint(pose, pos);
			Vec3 sc = GetScreenCoord(pos);

			if (sc.z < 1.0f)
			{
				//DrawImguiString(int(sc.x - 45.0f), int(sc.y - 5.0f), Vec3(1, 0, 1), 0, "%d %s %s %d -- grasp_%d -- %d -- %f", e, ge.objName.c_str(), ge.stpId.c_str(), ge.grasp->id, ge.graspNum, e, ge.lImpulse+ge.rImpulse);

			}
		}
		GraspExperiment& ge = experiments[0];

		//DrawImguiString(int(10), int(10), Vec3(0.1f, 0.1f, 0.1f), 0, "%s %s %d", ge.objName.c_str(), ge.stpId.c_str(), ge.grasp->id);
	}

	virtual void DoGui()
	{
		imguiSlider("offLen", &offLen, 0.16f, 0.165f, 0.0001f);
		imguiSlider("gripperIW", &gripperInitialWidth, 0.025f, 0.04f, 0.0001f);
		if (g_pause)
		{
			return;
		}
	}

	virtual void Sync()
	{
		if (anyShapeUpdate)
		{
			NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
			anyShapeUpdate = false;
		}
	}

	virtual void Update()
	{	
#if 0
		for (size_t i = 0; i < experiments.size(); i++)
		{
			if (experiments[i].frameCounter == 0)
			{
				for (int s = experiments[i].gripperShapeBegin; s < experiments[i].gripperShapeEnd; s++)
				{
					g_buffers->rigidShapes[s].filter = bkFilters[s];
				}
				for (int s = experiments[i].objectShapeBegin; s < experiments[i].objectShapeEnd; s++)
				{
					g_buffers->rigidShapes[s].filter = bkFilters[s];
				}

				NvFlexRigidJoint& rj = g_buffers->rigidJoints[experiments[i].fingerJointIndex];
				NvFlexRigidJoint& rjm = g_buffers->rigidJoints[experiments[i].fingermJointIndex];

				//rj.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeVelocity;
				//rjm.modes[eNvFlexRigidJointAxisX] = eNvFlexRigidJointModeVelocity;

				rj.compliance[eNvFlexRigidJointAxisX] = 0.0f;
				rjm.compliance[eNvFlexRigidJointAxisX] = 0.0f;

				rj.motorLimit[eNvFlexRigidJointAxisX] = forceLimitInNewton;
				rjm.motorLimit[eNvFlexRigidJointAxisX] = forceLimitInNewton;
				//rj.targets[eNvFlexRigidJointAxisX] = -gripperSqueezeSpeed;
				//rjm.targets[eNvFlexRigidJointAxisX] = -gripperSqueezeSpeed;

				anyShapeUpdate = true;
			}
			experiments[i].frameCounter++;
		}
#endif
		//cout << "g_camPos = Vec3(" << g_camPos.x << ", " << g_camPos.y << ", " << g_camPos.z << ");" << endl;
		//cout << "g_camAngle = Vec3(" << g_camAngle.x << ", " << g_camAngle.y << ", " << g_camAngle.z << ");" << endl;
	}

	virtual void Draw(int pass)
	{

		if (pass == 0)
		{
			SetFillMode(true);

			DrawRigidShapes(false);

			SetFillMode(false);
		}
		/*
		BeginLines(false);

		for (size_t e = 0; e < experiments.size(); e++)
		{
			GraspExperiment& ge = experiments[e];
			Transform pose;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.baseIndex], (NvFlexRigidPose*)&pose);
			Vec3 pos(0.0f, 0.0f, offLen);
			pos = TransformPoint(pose, pos);
			DrawLine(pose.p, pos, Vec4(1.0f, 0.0f, 0.0f,1.0f));

			Transform posel;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.lfingerIndex], (NvFlexRigidPose*)&posel);
			Transform poser;
			NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.rfingerIndex], (NvFlexRigidPose*)&poser);
			
			DrawLine(posel.p, poser.p, Vec4(1.0f, 1.0f, 0.0f, 1.0f));




		}
		EndLines();
		*/
	}

	virtual void PreSimulation()
	{
		if (!g_pause || g_step)
		{
			//NvFlexGetRigidBodies(g_solver, g_buffers->rigidBodies.buffer);
			//g_buffers->rigidBodies.map();
			for (int s = 0; s < numSimulationStepsPerRenderFrame; s++)
			{
				bool reset = false;
				// tick solver
				//NvFlexGetRigidJoints(g_solver, g_buffers->rigidJoints.buffer);
				g_buffers->rigidJoints.map();
				g_buffers->rigidShapes.map();
				g_buffers->rigidBodies.map();

				//Ask Miles about ground contact
				NvFlexGetRigidContacts(g_solver, rigidContacts.buffer, rigidContactCount.buffer);
				rigidContacts.map();
				rigidContactCount.map();

				int numContacts = rigidContactCount[0];
				// check if we overflowed the contact buffers
				if (numContacts > g_solverDesc.maxRigidBodyContacts)
				{
					printf("Overflowing rigid body contact buffers (%d > %d). Contacts will be dropped, increase NvSolverDesc::maxRigidBodyContacts.\n", numContacts, g_solverDesc.maxRigidBodyContacts);
					numContacts = min(numContacts, g_solverDesc.maxRigidBodyContacts);
				}

				bool allExperimentsEnded = true;
				for (auto ge : experiments)
				{
					if (ge.motionPhase != END)
					{
						allExperimentsEnded = false;
						break;
					}
				}
				int badCount = 0;
				for (size_t i = 0; i < experiments.size(); i++)
				{
					auto& ge = experiments[i];
					if (ge.bad) {
						badCount++;
					}
				}
				if (badCount == experiments.size()) {
					// Early termination for bad grasp
					allExperimentsEnded = true;
				}

				if (foundExperiment && (allExperimentsEnded || frameSinceStart > maxExperimentFrames))
				{
					// TODO(jaliang): different success for lift and shake

					// Check how many grasps are successful
					int numSuccess = 0;
					int numBad = 0;

					for (size_t i = 0; i < experiments.size(); i++)
					{
						GraspExperiment& ge = experiments[i];
						Transform ot;
						NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.objIndex], (NvFlexRigidPose*)&ot);
						if (ge.badExperiment)
						{
							numBad++;
						}
						else //if (ot.p.y - ge.objectInitPose.p.y > 0.02f)
						{
							Transform posel;
							NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.lfingerIndex], (NvFlexRigidPose*)&posel);
							Transform poser;
							NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.rfingerIndex], (NvFlexRigidPose*)&poser);
							if (Length(posel.p - poser.p) > gripThreshold) {
								// Object is still in gripper
								numSuccess++;
							}
						}
					}

					GraspExperiment& ge = experiments[0];
					float successProportion = (float)numSuccess / experiments.size();
					cout << "   "<<numSuccess << " grasps are successful out of " << experiments.size() << " Percent = " << 100.0f*successProportion << " bad = "<< 100.0f*(float)numBad / experiments.size() <<endl;
					char str[5000];
					sprintf(str, "%s,%s,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d", ge.objName.c_str(), ge.stpId.c_str(), ge.grasp->id, ge.graspNum, (int)experiments.size(),
							numSuccess, numBad, experiments[0].grasp->robustness, experiments[0].grasp->collides ? 1:0,
							experiments[0].grasp->robustness > binSuccessThrshldDexnet ? 1:0,
							successProportion > 0.25 ? 1:0, successProportion > 0.5 ? 1:0, successProportion > 0.75 ? 1:0);
					csv << str << endl;
					csv.flush();
					reset = true;
					//exit(0);
				}

				for (size_t i = 0; i < experiments.size(); i++)
				{
					experiments[i].lImpulse = 0.0f;
					experiments[i].rImpulse = 0.0f;
					experiments[i].cImpulse = 0.0f;
					experiments[i].robotHitFloor = false;
					experiments[i].robotHitObject = false;
				}

				NvFlexRigidContact* ct = &rigidContacts[0];

				float hitThreshold = 0.01f; // N
				float gripperReleaseSpeed = 0.01f; // m/s
				float gripperMoveSpeed = 0.1f; // m/s 
				float gripperAcceleration = 0.01f; // m/s^2
				float gripperMinWidth = 0.02f; // m  CE: was before 0.02
				for (int i = 0; i < numContacts; ++i)
				{
					if (ct[i].lambda > 0.f)
					{
						if ((ct[i].body0 != -1) && ((rfingerMap[ct[i].body0] != -1) || (lfingerMap[ct[i].body0] != -1)) && (ct[i].body1 == -1))
						{
							int expNum = -1;
							if (rfingerMap[ct[i].body0] != -1)
							{
								expNum = rfingerMap[ct[i].body0];
							}
							if (lfingerMap[ct[i].body0] != -1)
							{
								expNum = lfingerMap[ct[i].body0];
							}
							experiments[expNum].robotHitFloor = true;
							experiments[expNum].cImpulse += fabs(ct[i].lambda);
						}
						else if ((ct[i].body1 != -1) && ((rfingerMap[ct[i].body1] != -1) || (lfingerMap[ct[i].body1] != -1)) && (ct[i].body0 == -1))
						{
							int expNum = -1;
							if (rfingerMap[ct[i].body1] != -1)
							{
								expNum = rfingerMap[ct[i].body1];
							}
							if (lfingerMap[ct[i].body1] != -1)
							{
								expNum = lfingerMap[ct[i].body1];
							}
							experiments[expNum].robotHitFloor = true;
							experiments[expNum].cImpulse += fabs(ct[i].lambda);
						}
						else if ((ct[i].body0 != -1) && (ct[i].body1 != -1))
						{
							int other = 0;
							Vec3 normal;
							if (objMap[ct[i].body0] != -1)
							{
								other = ct[i].body1;
								normal = Vec3(ct[i].normal);
							}
							else if (objMap[ct[i].body1] != -1)
							{
								other = ct[i].body0;
								normal = -Vec3(ct[i].normal);
							}
							else
							{
								continue;
							}
							if (lfingerMap[other] != -1)
							{
								int fid = lfingerMap[other];
								experiments[fid].lImpulse += ct[i].lambda;
								experiments[fid].cImpulse += fabs(ct[i].lambda);
								experiments[fid].robotHitObject = true;
							}
							else if (rfingerMap[other] != -1)
							{
								int fid = rfingerMap[other];
								experiments[fid].rImpulse += ct[i].lambda;
								experiments[fid].cImpulse += fabs(ct[i].lambda);
								experiments[fid].robotHitObject = true;
							}

						}
					}
				}
				rigidContacts.unmap();
				rigidContactCount.unmap();
				
				/*
				if (frameSinceStart == 1)
				{
					for (size_t i = 0; i < experiments.size(); i++)
					{
						if (experiments[i].robotHitFloor )
						{
							cout << "set experiment " << i << " to bad due to robot floor collision" << endl;
							for (int s = experiments[i].objectShapeBegin; s < experiments[i].objectShapeEnd; s++)
							{
								g_buffers->rigidShapes[s].user = floorCollideColorMaterial;
							}
							for (int s = experiments[i].gripperShapeBegin; s < experiments[i].gripperShapeEnd; s++)
							{
								g_buffers->rigidShapes[s].user = floorCollideColorMaterial;
							}
							experiments[i].badExperiment = true;
							experiments[i].bad = 1;
							anyShapeUpdate = true;
						}
						if (experiments[i].robotHitObject)
						{
							cout << "set experiment " << i << " to bad due to robot object collision" << endl;
							for (int s = experiments[i].objectShapeBegin; s < experiments[i].objectShapeEnd; s++)
							{
								g_buffers->rigidShapes[s].user = shapeCollideColorMaterial;
							}
							for (int s = experiments[i].gripperShapeBegin; s < experiments[i].gripperShapeEnd; s++)
							{
								g_buffers->rigidShapes[s].user = shapeCollideColorMaterial;
							}
							experiments[i].badExperiment = true;
							experiments[i].bad = 1;
							anyShapeUpdate = true;
						}

					}
				}
				*/
				
				for (size_t i = 0; i < experiments.size(); i++)
				{
					auto& ge = experiments[i];
					if (experiments[i].badExperiment)
					{
						continue;
					}

					NvFlexRigidJoint& effector0 = g_buffers->rigidJoints[experiments[i].effectorIndex];
					//cout << rImpulse << " " << lImpulse << " " << cImpulse<<" "<<robotHitFloor << endl;
					if (experiments[i].motionPhase == APPROACH)
					{
						if (frameSinceStart > 2) // CE: was 2 before
						{
							if ((experiments[i].robotHitFloor) || (experiments[i].cImpulse > hitThreshold) || (experiments[i].remainingDist <= 0.0f))
							{
								experiments[i].motionPhase = GRIP;
							}
							else
							{
								//cout << "Experiment " << i << " approach with speed" << gripperMoveSpeed << endl;
								/*
								experiments[i].gripperWidth += gripperReleaseSpeed * g_dt;
								if (experiments[i].gripperWidth > gripperInitialWidth)
								{
									experiments[i].gripperWidth = gripperInitialWidth;
								}
								experiments[i].moveSpeed += g_dt*gripperAcceleration;
								if (experiments[i].moveSpeed > gripperMoveSpeed)
								{
									experiments[i].moveSpeed = gripperMoveSpeed;
								}
								*/
								experiments[i].remainingDist -= g_dt*gripperMoveSpeed;
								(Vec3&)effector0.pose0.p -= gripperMoveSpeed * g_dt*experiments[i].liftDir;
								//if (i == 0)
								//cout << "g_dt*ms = "<<g_dt*gripperMoveSpeed <<" len = "<<Length(experiments[i].liftDir)<<" Experiment " << i << " pos = " << effector0.pose0.p[0]<<" "<< effector0.pose0.p[1]<<" "<< effector0.pose0.p[2]<<endl;
							}
						}
					}
					else if (experiments[i].motionPhase == GRIP)
					{
						if (frameSinceStart > 2)
						{
							float totalForce = (experiments[i].rImpulse + experiments[i].lImpulse);
							if ((totalForce > gripThresholdInNewton) || (experiments[i].gripperWidth < gripperMinWidth))
							{
								experiments[i].motionPhase = LIFT;
							}
							else
							{
								experiments[i].gripperWidth -= gripperSqueezeSpeed * g_dt;
								if (experiments[i].gripperWidth < 0.0f)
								{
									experiments[i].gripperWidth = 0.0f;
								}
							}
							/*
							NvFlexRigidJoint& rj = g_buffers->rigidJoints[experiments[i].fingerJointIndex];
							NvFlexRigidJoint& rjm = g_buffers->rigidJoints[experiments[i].fingermJointIndex];
							rj.targets[eNvFlexRigidJointAxisX] = experiments[i].gripperWidth;
							Vec3 lpos, lapos;
							ExtractJointCoordinates(rj, lpos, lapos);
							rjm.targets[eNvFlexRigidJointAxisX] = lpos.x; // Mimic
							*/

							NvFlexRigidJoint& rj = g_buffers->rigidJoints[experiments[i].fingerJointIndex];
							NvFlexRigidJoint& rjm = g_buffers->rigidJoints[experiments[i].fingermJointIndex];

							rj.targets[eNvFlexRigidJointAxisX] = -gripperSqueezeSpeed;//ge.gripperWidth;
							rjm.targets[eNvFlexRigidJointAxisX] = -gripperSqueezeSpeed;//ge.gripperWidth;

							NvFlexRigidJoint& rjL = g_buffers->rigidJoints[experiments[i].fingerLJointIndex];
							NvFlexRigidJoint& rjmL = g_buffers->rigidJoints[experiments[i].fingermLJointIndex];

							Vec3 lpos, lapos;
							ExtractJointCoordinates(rjL, lpos, lapos);
							rjL.upperLimits[eNvFlexRigidJointAxisX] = lpos.x;
							rjmL.upperLimits[eNvFlexRigidJointAxisX] = lpos.x;
						}
					}
					else if (experiments[i].motionPhase == LIFT)
					{
						float totalForce = (experiments[i].rImpulse + experiments[i].lImpulse);
						if ((totalForce > gripThresholdInNewton) || (experiments[i].gripperWidth < gripperMinWidth))
						{
							experiments[i].motionPhase = LIFT;
						}
						else
						{
							experiments[i].gripperWidth -= gripperSqueezeSpeed * g_dt;
							if (experiments[i].gripperWidth < 0.0f)
							{
								experiments[i].gripperWidth = 0.0f;
							}
						}
						/*
						NvFlexRigidJoint& rj = g_buffers->rigidJoints[experiments[i].fingerJointIndex];
						NvFlexRigidJoint& rjm = g_buffers->rigidJoints[experiments[i].fingermJointIndex];
						rj.targets[eNvFlexRigidJointAxisX] = experiments[i].gripperWidth;
						Vec3 lpos, lapos;
						ExtractJointCoordinates(rj, lpos, lapos);
						rjm.targets[eNvFlexRigidJointAxisX] = lpos.x; // Mimic
						*/
						experiments[i].moveSpeed += g_dt*gripperAcceleration;
						if (experiments[i].moveSpeed > gripperMoveSpeed)
						{
							experiments[i].moveSpeed = gripperMoveSpeed;
						}

						(Vec3&)effector0.pose0.p += experiments[i].moveSpeed * g_dt*experiments[i].liftDir;
						if (Dot((Vec3&)effector0.pose0.p - experiments[i].graspInitPose.p, experiments[i].liftDir) >= 0.0f)
						{					
							experiments[i].motionPhase = SHAKE_TRA;
							experiments[i].shakeStartFrame = frameSinceStart;
						}
					}
					else if (experiments[i].motionPhase == SHAKE_ROT)
					{
						int shakeFrame = frameSinceStart - experiments[i].shakeStartFrame;
						if (!doShake || shakeFrame >= (int)shakeDeltaAngles.size())
						{
							experiments[i].motionPhase = WAIT;
							experiments[i].waitStartFrame = frameSinceStart;
						}
						
						Quat deltaQ = QuatFromAxisAngle(Vec3(1.f, 0.f, 0.f), shakeDeltaAngles[shakeFrame]);
						Quat R_grasp_world = Quat(experiments[i].graspInitPose.q);
						Quat R_world_grasp = Inverse(R_grasp_world);
						Quat nextRot = R_grasp_world * deltaQ * R_world_grasp * Quat(effector0.pose0.q);
						
						effector0.pose0.q[0] = nextRot.x;
						effector0.pose0.q[1] = nextRot.y;
						effector0.pose0.q[2] = nextRot.z;
						effector0.pose0.q[3] = nextRot.w;

						g_buffers->rigidJoints[experiments[i].effectorIndex] = effector0;
					}
					else if (experiments[i].motionPhase == SHAKE_TRA)
					{
						int shakeFrame = frameSinceStart - experiments[i].shakeStartFrame;
						if (!doShake || shakeFrame >= shakeDeltaTra.size())
						{
							experiments[i].motionPhase = SHAKE_ROT;
							experiments[i].shakeStartFrame = frameSinceStart;
						}
						effector0.pose0.p[1] += shakeDeltaTra[shakeFrame];

						g_buffers->rigidJoints[experiments[i].effectorIndex] = effector0;
					}
					else if (experiments[i].motionPhase == WAIT)
					{
						if (frameSinceStart - experiments[i].waitStartFrame > waitNumFrames)
						{
							experiments[i].motionPhase = END;
						}
					}

					if ((ge.motionPhase == LIFT) ||
						(ge.motionPhase == SHAKE_TRA) ||
						(ge.motionPhase == SHAKE_ROT) ||
						(ge.motionPhase == WAIT) 
						  ) {

						Transform posel;
						NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.lfingerIndex], (NvFlexRigidPose*)&posel);
						Transform poser;
						NvFlexGetRigidPose(&g_buffers->rigidBodies[ge.rfingerIndex], (NvFlexRigidPose*)&poser);
						if (Length(posel.p - poser.p) < gripThreshold) {
							// Can't grip object
							experiments[i].bad = 1;
						}
					}
				}

				Update();
				if (reset)
				{
					numCurrentExperiments++;
					if (numCurrentExperiments >= numExperiments)
					{
						numCurrentExperiments = 0;
						currentExperimentIndex += numExpIncrement;
					}
					ResetScene();
				}
				g_buffers->rigidJoints.unmap();
				g_buffers->rigidShapes.unmap();
				g_buffers->rigidBodies.unmap();

				if (anyShapeUpdate || reset)
				{
					NvFlexSetRigidShapes(g_solver, g_buffers->rigidShapes.buffer, g_buffers->rigidShapes.size());
					anyShapeUpdate = false;
				}

				NvFlexSetRigidJoints(g_solver, g_buffers->rigidJoints.buffer, g_buffers->rigidJoints.size());
				NvFlexSetParams(g_solver, &g_params);
				if (reset)
				{
					NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!
					reset = false;
				}
				NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

				frameSinceStart++;

				g_step = false;
			}
			//g_buffers->rigidBodies.unmap();
			//NvFlexSetRigidBodies(g_solver, g_buffers->rigidBodies.buffer, g_buffers->rigidBodies.size()); // Need to set bodies here too!
		}
	}
};
extern void ExportToObj(const char* path, const Mesh& m);
