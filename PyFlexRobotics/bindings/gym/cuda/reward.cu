#include "reward.h"

__global__ void AntRewardKernel(
	const NvFlexRigidBody* bodies,
	int numAgents,
	float* reward)
{
	int agent = blockIdx.x*blockDim.x + threadIdx.x;

	if (agent < numAgents)
	{
		const int kNumBodiesPerAnt = 16;

		// reward is x-position of center of mass
		reward[agent] = bodies[agent*kNumBodiesPerAnt].com[0];
	}
}

// reward array should be allocated with cudaMallocHost() so that device can write directly to host memory
void AntCalculateReward(NvFlexSolver* solver, int numAgents, float* reward)
{
	// map device buffers for kernel access
	NvFlexDeviceBuffers buffers;
	NvFlexMapDeviceBuffers(solver, &buffers);

	const int kNumThreadsPerBlock = 256;
	const int kNumBlocks = (numAgents + kNumThreadsPerBlock - 1)/kNumThreadsPerBlock;

	AntRewardKernel<<<kNumBlocks, kNumThreadsPerBlock>>>(buffers.rigidBodies, numAgents, reward);

	// unmap
	NvFlexUnmapDeviceBuffers(solver, &buffers);
}