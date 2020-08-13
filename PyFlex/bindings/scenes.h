#pragma once

class Scene
{
public:

	Scene(const char* name) : mName(name) {}
	
	virtual void Initialize(py::array_t<float> scene_params, int thread_idx = 0) = 0;
	virtual void PostInitialize() {}
	
	// update any buffers (all guaranteed to be mapped here)
	virtual void Update(py::array_t<float> update_params) {}

	// send any changes to flex (all buffers guaranteed to be unmapped here)
	virtual void Sync() {}
	
	virtual void Draw(int pass) {}
	virtual void KeyDown(int key) {}
	virtual void DoGui() {}
	virtual void CenterCamera() {}

	virtual Matrix44 GetBasis() { return Matrix44::kIdentity; }	

	virtual const char* GetName() { return mName; }

	const char* mName;
};

#include "softgym_scenes/softgym_cloth.h"
#include "softgym_scenes/softgym_fluid.h"
#include "softgym_scenes/softgym_softbody.h"
#include "softgym_scenes/softgym_rigid_cloth.h"
#include "softgym_scenes/softgym_torus.h"
#include "softgym_scenes/softgym_rope.h"

#include "scenes/adhesion.h"
#include "scenes/armadilloshower.h"
#include "scenes/bananas.h"
#include "scenes/bouyancy.h"
#include "scenes/bunnybath.h"
#include "scenes/ccdfluid.h"
#include "scenes/clothlayers.h"
#include "scenes/dambreak.h"
#include "scenes/darts.h"
#include "scenes/debris.h"
#include "scenes/deformables.h"
#include "scenes/envcloth.h"
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
#include "scenes/potpourri.h"
#include "scenes/rayleightaylor.h"
#include "scenes/restitution.h"
#include "scenes/rigidfluidcoupling.h"
#include "scenes/rigidpile.h"
#include "scenes/rigidrotation.h"
#include "scenes/rockpool.h"
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
