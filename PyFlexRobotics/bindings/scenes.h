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

#pragma once

// disable some warnings
#if _WIN32
#pragma warning(disable: 4267)  // conversion from 'size_t' to 'int', possible loss of data
#endif

//py::array_t<float> tmp_scene_params;

class Scene
{
public:

	Scene() {}
	virtual ~Scene() {}
    virtual void Initialize();
//	virtual void Initialize(py::array_t<float> scene_params = py::array_t<float>() , int thread_idx = 0);
	virtual void PostInitialize() {}

	// Called immediately after scene constructor (used by RL scenes to load parameters and launch Python scripts)
	virtual void PrepareScene() {}

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

	virtual void PreSimulation() {}

};


class SceneFactory
{
public:

	SceneFactory(const char* name, std::function<Scene*()> factory, bool isVR) : mName(name), mFactory(factory), mIsVR(isVR) {}

	const char* mName;
	const bool mIsVR;
	std::function<Scene*()> mFactory;
};


extern std::vector<SceneFactory> g_sceneFactories;

inline void RegisterScene(const char* name, std::function<Scene*()> factory, bool isVR = false)
{
	g_sceneFactories.push_back(SceneFactory(name, factory, isVR));
}

//#include "scenes/softgym_flatten.h"
//#include "scenes/softgym_cloth.h"
//#include "scenes/softgym_pourwater.h"
//#include "scenes/softgym_softbody.h"

#include "scenes/adhesion.h"
#include "scenes/armadilloshower.h"
#include "scenes/bananas.h"
#include "scenes/buoyancy.h"
#include "scenes/bunnybath.h"
#include "scenes/ccdfluid.h"
#include "scenes/clothlayers.h"
#include "scenes/clothgripper.h"
#include "scenes/controller.h"
#include "scenes/dambreak.h"
#include "scenes/darts.h"
#include "scenes/debris.h"
#include "scenes/deformables.h"
#include "scenes/envcloth.h"
#include "scenes/fem.h"
#include "scenes/SoftSnake.h"
#include "scenes/flag.h"
#include "scenes/fluidblock.h"
#include "scenes/fluidclothcoupling.h"
#include "scenes/forcefield.h"
#include "scenes/frictionmoving.h"
#include "scenes/frictionramp.h"
#include "scenes/gamemesh.h"
#include "scenes/googun.h"
#include "scenes/granularpile.h"
#include "scenes/granularshape.h"
#include "scenes/inflatable.h"
#include "scenes/initialoverlap.h"
#include "scenes/lighthouse.h"
#include "scenes/localspacecloth.h"
#include "scenes/localspacefluid.h"
#include "scenes/lowdimensionalshapes.h"
#include "scenes/melting.h"
#include "scenes/mixedpile.h"
#include "scenes/nonconvex.h"
#include "scenes/parachutingbunnies.h"
#include "scenes/pasta.h"
#include "scenes/player.h"
#include "scenes/pneunet.h"
#include "scenes/potpourri.h"
#include "scenes/rayleightaylor.h"
#include "scenes/restitution.h"
#include "scenes/rigidfetch.h"
#include "scenes/rigidbody.h"
#include "scenes/rigidcable.h"
#include "scenes/rigidfluidcoupling.h"
#include "scenes/rigidpile.h"
#include "scenes/rigidrotation.h"
#include "scenes/rigidatlas.h"
#include "scenes/rigidallegro.h"
#include "scenes/rigidbaxter.h"
#include "scenes/rigidgrasp.h"
#include "scenes/rigidurdf.h"
#include "scenes/rigidsawyer.h"
#include "scenes/rigidyumi.h"
#include "scenes/rigidyumicabinet.h"
#include "scenes/rigidyumitasks.h"
#include "scenes/rigidyumigripper.h"
#include "scenes/rigidyumimultiplegrippers.h"
#include "scenes/rigidbvhretargettest.h"
#include "scenes/rigidhumanoidpool.h"
#include "scenes/rigidParticleAttachment.h"
#include "scenes/rlallegro.h"
#include "scenes/rllocomotion.h"
#include "scenes/rllocomotionterrain.h"
#include "scenes/rllocomotionmocap.h"
#include "scenes/rlatlas.h"
#include "scenes/rlanymal.h"
#include "scenes/rlcarter.h"
#include "scenes/rlfetch.h"
#include "scenes/rlyumi.h"
#include "scenes/rlfranka.h"
#include "scenes/rlfrankaallegro.h"
#include "scenes/rlminitaur.h"
#include "scenes/rlsawyer.h"
#include "scenes/rlyumi.h"
#include "scenes/rigidfrankaallegro.h"
#include "scenes/rockpool.h"
#include "scenes/rope.h"
#include "scenes/sdfcollision.h"
#include "scenes/shapecollision.h"
#include "scenes/shapechannels.h"
#include "scenes/softbody.h"
#include "scenes/spherecloth.h"
#include "scenes/surfacetension.h"
#include "scenes/tearing.h"
#include "scenes/thinbox.h"
#include "scenes/trianglecollision.h"
#include "scenes/triggervolume.h"
#include "scenes/viscosity.h"
#include "scenes/waterballoon.h"
#include "scenes/dexnet_test.h"
#include "scenes/vrfetch.h"
#include "scenes/vrrigidallegro.h"
#include "scenes/grasping.h"


inline void RegisterPhysicsScenes()
{
//    RegisterScene("SoftGym Cloth", []() { return new softgym_FlagCloth(); });
//    RegisterScene("SoftGym Cloth Flatten", []() { return new softgym_FlattenCloth(); });
//    RegisterScene("SoftGym Water", []() { return new softgym_PourWater; });

	RegisterScene("RL Sawyer", []() { return new RLSawyerCup(); });
	RegisterScene("Rigid to Particles Attachments", []() { return new RigidParticleAttachment(); });
	RegisterScene("Soft Snake", []() { return new SoftSnake(""); });

	RegisterScene("RL Franka Allegro Base", []() { return new RLFrankaAllegroBase(); });
	RegisterScene("Rigid Franka Allegro", []() { return new RigidFrankaAllegro(); });
	RegisterScene("Grasping", []() { return new Grasping(); });

	RegisterScene("MPC Franka Cabinet", []() { return new MPC_FrankaAllegro(); });
	RegisterScene("RL Franka Cabinet", []() { return new RLFrankaCabinet(); });
	RegisterScene("RL Carter", []() { return new RLCarter(); });

	RegisterScene("Rigid DexNet", []() { return new DEXNetTest(); });
//	RegisterScene("Grasping", []() { return new Grasping(); });

	RegisterScene("Rigid Cable Hand", []() { return new RigidURDFCable();});

//	RegisterScene("RL Humanoid Parkour", []() { return new RLHumanoidParkour(); });
	RegisterScene("Rigid Fetch - Rigid", []() { return new RigidFetch(RigidFetch::eRigid); });
	RegisterScene("Rigid Allegro - Rigid Bunny", []() { return new RigidAllegro(RigidAllegro::eRigidBunny); });
	RegisterScene("Rigid Allegro - Rigid DexNet 1", []() { return new RigidAllegro(RigidAllegro::eDexnet1); });
	RegisterScene("Rigid Allegro - Rigid DexNet 2", []() { return new RigidAllegro(RigidAllegro::eDexnet2); });
	RegisterScene("Rigid Allegro - Rigid DexNet 3", []() { return new RigidAllegro(RigidAllegro::eDexnet3); });
	RegisterScene("Rigid Allegro - Rigid DexNet 4", []() { return new RigidAllegro(RigidAllegro::eDexnet4); });
	RegisterScene("Rigid Allegro - FEM Tomato", []() { return new RigidAllegro(RigidAllegro::eSoftTomato); });
	RegisterScene("Rigid Allegro - FEM Bread", []() { return new RigidAllegro(RigidAllegro::eSoftBread); });
	RegisterScene("Rigid Allegro - FEM Sandwich", []() { return new RigidAllegro(RigidAllegro::eSoftSandwich); });
	RegisterScene("Rigid Allegro - FEM Cube", []() { return new RigidAllegro(RigidAllegro::eSoftCube); });
	RegisterScene("Rigid Allegro - FEM Sphere", []() { return new RigidAllegro(RigidAllegro::eSoftSphere); });
	RegisterScene("Rigid Allegro - Bunny", []() { return new RigidAllegro(RigidAllegro::eRigidBunny); });
//	RegisterScene("Rigid Allegro - Cube", []() { return new RigidAllegro(RigidAllegro::eRigidCube); });
	RegisterScene("Rigid Allegro - Rope", []() { return new RigidAllegro(RigidAllegro::eRope); });

	RegisterScene("PneuNet Sphere", []() { return new PneuNetFinger(PneuNetFinger::eSphere); });

	RegisterScene("RL Simple Humanoid", []() { return new RLSimpleHumanoid(); });
	RegisterScene("RL Humanoid", []() { return new RigidHumanoid(); });
	RegisterScene("RL Humanoid Target Speed", []() { return new RigidHumanoidSpeed(); });
	RegisterScene("RL Hard Flagrun", []() { return new RigidHumanoidHard(); });
	RegisterScene("RL Humanoid Hard Parkour", []() { return new RLHumanoidHardParkour(); });
	RegisterScene("RL CMU Humanoid", []() { return new CMUHumanoid(); });

	RegisterScene("Rigid Baxter", []() { return new RigidBaxter(RigidBaxter::eConnect4); });
	RegisterScene("Rigid Cable Hand", []() { return new RigidURDFCable();});

	RegisterScene("Rigid Terrain", []() { return new RigidTerrain(); });

	RegisterScene("RL Allegro Base", []() { return new RLAllegroBase(); });
	RegisterScene("RL Allegro Object Relocation", []() { return new RLAllegroObjectRelocation(); });

	RegisterScene("RL Fetch Reach", []() { return new RLFetchReach(); });
	RegisterScene("RL Fetch Cube Grasping", []() { return new RLFetchCube(); });
	RegisterScene("RL Fetch Rope Peg", []() { return new RLFetchRopePeg(); });
	RegisterScene("RL Fetch Reach Sensor", []() { return new RLFetchReachSensor(); });
    RegisterScene("RL Fetch Reach Active", []() { return new RLFetchReachActive(); });
    RegisterScene("RL Fetch Reach Active Moving", []() { return new RLFetchReachActiveMoving(); });
	RegisterScene("RL Fetch Push", []() { return new RLFetchPush(); });
	RegisterScene("RL Fetch Rope Simple", []() { return new RLFetchRopeSimple(); });
	RegisterScene("RL Fetch Rope Push", []() { return new RLFetchRopePush(); });
	RegisterScene("RL Fetch Reach MultiGoal", []() { return new RLFetchReachMultiGoal(); });
	RegisterScene("RL Fetch Push MultiGoal", []() { return new RLFetchPushMultiGoal(); });
	RegisterScene("RL Fetch Cube Grasping MultiGoal", []() { return new RLFetchCubeMultiGoal(); });
	RegisterScene("RL Fetch Rope Simple MultiGoal", []() { return new RLFetchRopeSimpleMultiGoal(); });
	RegisterScene("RL Fetch Rope Push MultiGoal", []() { return new RLFetchRopePushMultiGoal(); });
	RegisterScene("RL Fetch Reach Sensor HER", []() { return new RLFetchReachSensorHER(); });
	RegisterScene("RL Fetch LR Sensor HER", []() { return new RLFetchLRSensorHER(); });
	RegisterScene("RL Fetch LR Sensor", []() { return new RLFetchLRSensor(); });
	RegisterScene("RL Fetch LR HER", []() { return new RLFetchLRHER(); });
	RegisterScene("RL Fetch LR", []() { return new RLFetchLR(); });
	RegisterScene("RL Sawyer", []() { return new RLSawyerCup(); });

	// Yumi
	RegisterScene("RL Yumi Reach", []() { return new RLYumiReach(); });
	RegisterScene("RL Yumi Cabinet", []() { return new RLYumiCabinet(); });
	RegisterScene("RL Yumi Rope Peg", []() { return new RLYumiRopePeg(); });
	RegisterScene("RL Yumi Cloth", []() { return new RLYumiCloth(); });

	// Franka
	RegisterScene("RL Franka Reach", []() { return new RLFrankaReach(); });
	RegisterScene("RL Franka Cabinet", []() { return new RLFrankaCabinet(); });

	RegisterScene("RL Humanoid Pool", []() { return new RLHumanoidPool(); });
	RegisterScene("RL ANYmal", []() { return new RLANYmal(false); });

	RegisterScene("RL Ant", []() { return new RLAnt(); });
	RegisterScene("RL AntHF", []() { return new RLAntHF(); });
	RegisterScene("RL Ant Parkour", []() { return new RLAntParkour(); });
	RegisterScene("RL Atlas Flagrun", []() { return new RLAtlas(); });

	RegisterScene("RL Minitaur", []() { return new RLMinitaur(); });
//	RegisterScene("Rigid Minitaur", []() { return new RigidURDF2(); });

#if FLEX_VR
	RegisterScene("VR Fetch - Rigid", []() { return new VRFetch(VRFetch::eRigid); }, true);
	RegisterScene("VR Fetch - Cloth", []() { return new VRFetch(VRFetch::eCloth); }, true);
	RegisterScene("VR Rigide Allegro - Rigid Cube", []() { return new VRRigidAllegro(VRRigidAllegro::eRigidCube); }, true);
#endif // #if FLEX_VR


	//RegisterScene([]() { return new DEXNetTest("Dexnet test"); });
	//RegisterScene([]() { return new RigidFullHumanoidDeepLocoRepro("Deep loco repro"); });
	//RegisterScene([]() { return new RigidBVHRetargetTest("bvh retarget test"); });
	//RegisterScene([]() { return new RigidBVHRetargetTest("bvh"); });
	//RegisterScene([]() { return new RigidFullHumanoidMocapInitTrackMJCF("gan rigid test"); });

	RegisterScene("GAN DeepMind", []() { return new RigidFullHumanoidMocapInitGANFrameFeatures(); });
	RegisterScene("GAN Nearest Neighbor Blend", []() { return new RigidFullHumanoidMocapInitNearestAndGANBlendMJCF(); });
	RegisterScene("GAN Test", []() { return new RigidFullHumanoidMocapInitGANMJCF(); });

	// WIP scenes
	RegisterScene("Rigid Tippe Top", []() { return new RigidTippeTop(); });
	RegisterScene("Rigid Complementarity", []() { return new RigidComplementarity(); });

	RegisterScene("Rigid Atlas", []() { return new RigidAtlas(); });
	RegisterScene("Rigid Yumi", []() { return new RigidYumi(RigidYumi::eRigid); });
	RegisterScene("Rigid Yumi Cabinet", []() { return new RigidYumiCabinet(); });
	RegisterScene("Rigid Yumi Tasks", []() { return new RigidYumiTasks(); });
	RegisterScene("Rigid Rolling Friction", []() { return new RigidRollingFriction(); });
	RegisterScene("Rigid Torsion Friction", []() { return new RigidTorsionFriction(); });
	RegisterScene("Rigid Grasp", []() { return new RigidGrasp(); });
	RegisterScene("Rigid Grasp Simple", []() { return new RigidGraspSimple(); });
	RegisterScene("Rigid Cosserat", []() { return new RigidCosserat(); });
	RegisterScene("Rigid Terrain", []() { return new RigidTerrain(); });
	RegisterScene("Rigid Heavy Stack", []() { return new RigidHeavyStack(); });
	RegisterScene("Rigid Heavy Button", []() { return new RigidHeavyButton(); });
	RegisterScene("Rigid Granular Compression", []() { return new RigidGranularCompression(); });
	RegisterScene("Rigid Broccoli", []() { return new RigidBroccoli(); });
	RegisterScene("Rigid Cable Hand", []() { return new RigidURDFCable();});

	RegisterScene("Rigid Tower", []() { return new RigidTower(); });
	RegisterScene("Rigid Arch", []() { return new RigidArch(); });
	RegisterScene("Rigid Table Pile", []() { return new RigidTablePile(); });

	RegisterScene("Rigid Sawyer", []() { return new RigidSawyer(RigidSawyer::eRigid); });

	RegisterScene("Rigid Fetch - Cloth", []() { return new RigidFetch(RigidFetch::eCloth); });
	RegisterScene("Rigid Fetch - Rigid", []() { return new RigidFetch(RigidFetch::eRigid); });
	RegisterScene("Rigid Fetch - Soft", []() { return new RigidFetch(RigidFetch::eSoft); });
	RegisterScene("Rigid Fetch - Rope Capsules", []() { return new RigidFetch(RigidFetch::eRopeCapsules); });
	RegisterScene("Rigid Fetch - Rope Particles", []() { return new RigidFetch(RigidFetch::eRopeParticles); });
	RegisterScene("Rigid Fetch - Rope Peg", []() { return new RigidFetch(RigidFetch::eRopePeg); });
	RegisterScene("Rigid Fetch - Sand Box", []() { return new RigidFetch(RigidFetch::eSandBucket); });
	RegisterScene("Rigid Fetch - Flexible Beam", []() { return new RigidFetch(RigidFetch::eFlexibleBeam); });

	RegisterScene("Rigid Humanoid Pool", []() { return new RigidHumanoidPool(); });

	RegisterScene("Rigid DexNet", []() { return new DEXNetTest(); });
	RegisterScene("Rigid URDF", []() { return new RigidURDF(); });

	RegisterScene("Rigid FEM", []() { return new RigidFEM(); });
	RegisterScene("Rigid Cloth", []() { return new RigidCloth(); });

	RegisterScene("Rigid Overlap", []() { return new RigidOverlap(); });
	RegisterScene("Rigid Joint Limits", []() { return new RigidJointLimits(); });
	RegisterScene("Rigid Capsule Stack", []() { return new RigidCapsuleStack(); });

	RegisterScene("Rigid Cylinder Stack", []() { return new RigidStack("../../data/cylinder.obj"); });
	RegisterScene("Rigid Rock Stack", []() { return new RigidStack("../../data/rockb.ply"); });
	RegisterScene("Rigid Box Stack", []() { return new RigidStack("../../data/box.ply"); });
	RegisterScene("Rigid Spring", []() { return new RigidSpring(); });
	RegisterScene("Rigid Spring Hard", []() { return new RigidSpringHard(); });
	RegisterScene("Rigid Friction", []() { return new RigidFriction(); });
	RegisterScene("Rigid Friction Aniso", []() { return new RigidFrictionAniso(); });
	RegisterScene("Rigid Collision", []() { return new RigidCollision(); });

	RegisterScene("Rigid Mobile", []() { return new RigidMobile(); });
	RegisterScene("Rigid Angular Motor", []() { return new RigidAngularMotor(); });
	RegisterScene("Rigid Fixed Joint", []() { return new RigidFixedJoint(); });
	RegisterScene("Rigid Hinge Joint", []() { return new RigidHingeJoint(); });
	RegisterScene("Rigid Spherical Joint", []() { return new RigidSphericalJoint(); });
	RegisterScene("Rigid Prismatic Joint", []() { return new RigidPrismaticJoint(); });
	RegisterScene("Rigid Gyroscopic", []() { return new RigidGyroscopic(); });
	RegisterScene("Rigid Pendulum", []() { return new RigidPendulum(); });

	RegisterScene("FEM Stick Slip", []() { return new FEMStickSlip(); });
	RegisterScene("FEM Beam", []() { return new FEM(4, 4, 10, false); });
	RegisterScene("FEM Beam Fixed", []() { return new FEM(4, 20, 4, true); });
	RegisterScene("FEM Poisson", []() { return new FEMPoisson(); });
	RegisterScene("FEM Twist", []() { return new FEMTwist(15, 15, 15, true, true); });

	//RegisterScene([]() { return new FEM("FEM Deer", "../../data/deer_bound.tet"); });
	//RegisterScene([]() { return new FEM("FEM Dragon 1", "../../data/dragon.tet"); });
	//RegisterScene([]() { return new FEM("FEM Dragon 2", "../../data/dragon_iso.tet"); });
	//RegisterScene([]() { return new FEM("FEM Duck", "../../data/duck.tet"); });
	RegisterScene("FEM Feline", []() { return new FEM("../../data/feline.tet"); });
	RegisterScene("FEM Pitbull", []() { return new FEM("../../data/pitbull.tet"); });

	RegisterScene("FEM Palm", []() { return new FEM("../../data/palmNormalized.tet"); });
	RegisterScene("FEM Pig", []() { return new FEM("../../data/pig.tet"); });
	RegisterScene("FEM Sphere", []() { return new FEM("../../data/sphereNormalized.tet"); });
	RegisterScene("FEM Frog", []() { return new FEM("../../data/froggy.tet"); });
	RegisterScene("FEM Net", []() { return new FEMNet(); });

	RegisterScene("Flag Cloth", []() { return new FlagCloth(); });
	RegisterScene("Simple Rope", []() { return new RopeSimple(); });

	// opening scene
	RegisterScene("Pot Pourri", []() { return new PotPourri(); });

	// soft body scenes
	RegisterScene("Soft Octopus", []() { return new SoftOctopus(); });
	RegisterScene("Soft Rope", []() { return new SoftRope(); });
	RegisterScene("Soft Cloth", []() { return new SoftCloth(); });
	RegisterScene("Soft Teapot", []() { return new SoftTeapot(); });
	RegisterScene("Soft Armadillo", []() { return new SoftArmadillo(); });
	RegisterScene("Soft Bunny", []() { return new SoftBunny(); });

	// plastic scenes
	RegisterScene("Plastic Bunnies", []() { return new PlasticBunnies(); });
	RegisterScene("Plastic Stack", []() { return new PlasticStack(); });

	// cloth scenes
	RegisterScene("Env Cloth Small", []() { return new EnvironmentalCloth(6, 6, 40, 16); });
	RegisterScene("Env Cloth Large", []() { return new EnvironmentalCloth(16, 32, 10, 3); });
	RegisterScene("Flag Cloth", []() { return new FlagCloth(); });

	// collision scenes
	RegisterScene("Friction Ramp", []() { return new FrictionRamp(); });
	RegisterScene("Friction Moving Box", []() { return new FrictionMovingShape(0); });
	RegisterScene("Friction Moving Sphere", []() { return new FrictionMovingShape(1); });
	RegisterScene("Friction Moving Capsule", []() { return new FrictionMovingShape(2); });
	RegisterScene("Friction Moving Mesh", []() { return new FrictionMovingShape(3); });
	RegisterScene("Shape Collision", []() { return new ShapeCollision(); });
	RegisterScene("Shape Channels", []() { return new ShapeChannels(); });
	RegisterScene("Triangle Collision", []() { return new TriangleCollision(); });
	RegisterScene("Local Space Fluid", []() { return new LocalSpaceFluid(); });
	RegisterScene("Local Space Cloth", []() { return new LocalSpaceCloth(); });
	RegisterScene("World Space Fluid", []() { return new CCDFluid(); });

	// cloth scenes
	RegisterScene("Env Cloth Small", []() { return new EnvironmentalCloth(6, 6, 40, 16); });
	RegisterScene("Env Cloth Large", []() { return new EnvironmentalCloth(16, 32, 10, 3); });
	RegisterScene("Flag Cloth", []() { return new FlagCloth(); });
	RegisterScene("Inflatables", []() { return new Inflatable(); });
	RegisterScene("Cloth Layers", []() { return new ClothLayers(); });
	RegisterScene("Sphere Cloth", []() { return new SphereCloth(); });
	RegisterScene("Tearing", []() { return new Tearing(); });
	RegisterScene("Pasta", []() { return new Pasta(); });

	// game mesh scenes
	RegisterScene("Game Mesh Rigid", []() { return new GameMesh(0); });
	RegisterScene("Game Mesh Particles", []() { return new GameMesh(1); });
	RegisterScene("Game Mesh Fluid", []() { return new GameMesh(2); });
	RegisterScene("Game Mesh Cloth", []() { return new GameMesh(3); });
	RegisterScene("Rigid Debris", []() { return new RigidDebris(); });

	// viscous fluids
	RegisterScene("Viscosity Low", []() { return new Viscosity(0.5f); });
	RegisterScene("Viscosity Med", []() { return new Viscosity(3.0f); });
	RegisterScene("Viscosity High", []() { return new Viscosity(5.0f, 0.12f); });
	RegisterScene("Adhesion", []() { return new Adhesion(); });
	RegisterScene("Goo Gun", []() { return new GooGun(true); });

	// regular fluids
	RegisterScene("Buoyancy", []() { return new Buoyancy(); });
	RegisterScene("Melting", []() { return new Melting(); });
	RegisterScene("Surface Tension Low", []() { return new SurfaceTension(0.0f); });
	RegisterScene("Surface Tension Med", []() { return new SurfaceTension(10.0f); });
	RegisterScene("Surface Tension High", []() { return new SurfaceTension(20.0f); });
	RegisterScene("DamBreak  5cm", []() { return new DamBreak(0.05f); });
	RegisterScene("DamBreak 10cm", []() { return new DamBreak(0.1f); });
	RegisterScene("DamBreak 15cm", []() { return new DamBreak(0.15f); });
	RegisterScene("Rock Pool", []() { return new RockPool(); });
	RegisterScene("Rayleigh Taylor 2D", []() { return new RayleighTaylor2D(); });


	// misc feature scenes
	RegisterScene("Trigger Volume", []() { return new TriggerVolume(); });
	RegisterScene("Force Field", []() { return new ForceField(); });
	RegisterScene("Initial Overlap", []() { return new InitialOverlap(); });

	// rigid body scenes
	RegisterScene("Rigid2", []() { return new RigidPile(2); });
	RegisterScene("Rigid4", []() { return new RigidPile(4); });
	RegisterScene("Rigid8", []() { return new RigidPile(12); });
	RegisterScene("Bananas", []() { return new BananaPile(); });
	RegisterScene("Low Dimensional Shapes", []() { return new LowDimensionalShapes(); });

	// granular scenes
	RegisterScene("Granular Pile", []() { return new GranularPile(); });

	// coupling scenes
	RegisterScene("Parachuting Bunnies", []() { return new ParachutingBunnies(); });
	RegisterScene("Water Balloons", []() { return new WaterBalloon(); });
	RegisterScene("Rigid Fluid Coupling", []() { return new RigidFluidCoupling(); });

	RegisterScene("Fluid Block", []() { return new FluidBlock(); });
	RegisterScene("Fluid Cloth Coupling Water", []() { return new FluidClothCoupling(false); });
	RegisterScene("Fluid Cloth Coupling Goo", []() { return new FluidClothCoupling(true); });
	RegisterScene("Bunny Bath Dam", []() { return new BunnyBath(true); });
}
