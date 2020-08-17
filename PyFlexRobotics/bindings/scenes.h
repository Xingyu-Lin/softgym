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

#include "softgym_scenes/softgym_robot.h"
class Scene
{
protected:
    SoftgymRobotBase* ptrRobot=NULL;
public:
	Scene() {}
	virtual ~Scene() {}

    virtual SoftgymRobotBase* getPtrRobot() {return ptrRobot;}

	virtual void Initialize(py::array_t<float> scene_params = py::array_t<float>(),
	                        py::array_t<float> robot_params = py::array_t<float>(), int thread_idx = 0) {}

	virtual void PostInitialize() {}

	// Called immediately after scene constructor (used by RL scenes to load parameters and launch Python scripts)
	virtual void PrepareScene() {}

	// update any buffers (all guaranteed to be mapped here)
	virtual void Update() {if (getPtrRobot()!=NULL) getPtrRobot()->Update();}

	// called after update, can be used to read back any additional information that is not read back by the main application
	virtual void PostUpdate() {if (getPtrRobot()!=NULL) getPtrRobot()->PostUpdate();}

	// send any changes to flex (all buffers guaranteed to be unmapped here)
	virtual void Sync() {}

	virtual void Draw(int pass) {}
	virtual void KeyDown(int key) {}
	virtual void DoGui() {if (getPtrRobot()!=NULL) getPtrRobot()->DoGui();}
	virtual void DoStats() {if (getPtrRobot()!=NULL) getPtrRobot()->DoStats();} // draw on-screen graphs, text, etc

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
    void Step(py::array_t<float> control_params = py::array_t<float>())
    {
        ptrRobot = getPtrRobot();
        if (ptrRobot != NULL) ptrRobot->Step(control_params);
    }

    py::array_t<float>  GetRobotState()
    {
        ptrRobot = getPtrRobot();
        if (ptrRobot != NULL) return ptrRobot->GetState();
        else
        {
            std::cout<<"WARNING: Calling GetRobotState function in scenes without a robot."<<std::endl;
        }
    }

    void SetRobotState(py::array_t<float> robot_state)
    {
        ptrRobot = getPtrRobot();
        if (ptrRobot != NULL) ptrRobot->SetState(robot_state);
        else
        {
            std::cout<<"WARNING: Calling GetRobotState function in scenes without a robot."<<std::endl;
        }
    }
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

#include "softgym_scenes/softgym_cloth.h"
#include "softgym_scenes/softgym_fluid.h"
#include "softgym_scenes/softgym_rope.h"
#include "softgym_scenes/softgym_softbody.h"

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
    // PlaceHolder scenes such that the index is compatible with those in the 1.0 version.
    RegisterScene("SoftGym Cloth", []() { return new SoftgymCloth(); });
    RegisterScene("SoftGym Water", []() { return new SoftgymFluid(); });
    RegisterScene("SoftGym Rope", []() { return new SoftgymRope(); });
    RegisterScene("RL Franka Reach", []() { return new RLFrankaReach(); });
    RegisterScene("SoftGym Cloth New", []() { return new SoftgymCloth(); });
    RegisterScene("Rigid Sawyer", []() { return new RigidSawyer(RigidSawyer::eCloth); });
}
