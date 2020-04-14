
class RopeSimple : public Scene
{
public:

	RopeSimple()
	{
		float radius = 0.1f;

		float length = 2.0f;
		int segments = 40;
		float stiffness = 0.9f;

		Rope r;
		CreateRope(r, Vec3(0.0f, 3.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), stiffness, 0.0f, segments, length, NvFlexMakePhase(0, 0), 0, 1.0f, 0.0f);
		g_ropes.push_back(r);

		// fix first particle
		g_buffers->positions[0].w = 0.0f;


		/*
		float stretchStiffness = 0.9f;
		float bendStiffness = 0.8f;
		float shearStiffness = 0.5f;

		int dimx = 4;
		int dimz = 4;

		CreateSpringGrid(Vec3(0.0f, 1.0f, 0.0f), dimx, dimz, 1, radius, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f);
		g_buffers->positions[0].w = 0.0f;
		*/

		// crash ogl with a shape.. ugh
		AddBox(Vec3(0.001f));

		
		g_numSubsteps = 1;

		g_params.numIterations = 4;
		g_params.radius = radius;
		g_params.dynamicFriction = 0.4f;
		g_params.restitution = 0.0f;
		g_params.shapeCollisionMargin = 0.01f;

		g_lightDistance *= 0.5f;
		g_drawPoints = false;		

		g_pause = true;
	}
};