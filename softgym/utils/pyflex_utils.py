import numpy as np
import pyflex


def random_pick_and_place(pick_num=10, pick_scale=0.01):
    """ Random pick a particle up and the drop it for pick_num times"""
    curr_pos = pyflex.get_positions().reshape(-1, 4)
    num_particles = curr_pos.shape[0]
    for i in range(pick_num):
        pick_id = np.random.randint(num_particles)
        pick_dir = np.random.random(3) * 2 - 1
        pick_dir[1] = (pick_dir[1] + 1)
        pick_dir *= pick_scale
        original_inv_mass = curr_pos[pick_id, 3]
        for _ in range(60):
            curr_pos = pyflex.get_positions().reshape(-1, 4)
            curr_pos[pick_id, :3] += pick_dir
            curr_pos[pick_id, 3] = 0
            pyflex.set_positions(curr_pos.flatten())
            pyflex.step()

        # Revert mass
        curr_pos = pyflex.get_positions().reshape(-1, 4)
        curr_pos[pick_id, 3] = original_inv_mass
        pyflex.set_positions(curr_pos.flatten())
        pyflex.step()

        # Wait to stabalize
        for _ in range(100):
            pyflex.step()
            curr_vel = pyflex.get_velocities()
            if np.alltrue(curr_vel < 0.01):
                break
    for _ in range(500):
        pyflex.step()
        curr_vel = pyflex.get_velocities()
        if np.alltrue(curr_vel < 0.01):
            break


def center_object():
    """
    Center the object to be at the origin
    NOTE: call a pyflex.set_positions and then pyflex.step
    """
    pos = pyflex.get_positions().reshape(-1, 4)
    pos[:, [0, 2]] -= np.mean(pos[:, [0, 2]], axis=0, keepdims=True)
    pyflex.set_positions(pos.flatten())
    pyflex.step()
