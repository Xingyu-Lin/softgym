set (files 
  atlas_cage_and_camera_diffuse_flat.jpg
  atlas_DRC_1.png
  atlas_DRC_carbon_fiber.png
  atlas_DRC_dark_1.png
  drc_extremities_diffuse.jpg
  drc_head_diffuse.png
  drc_labels_1.jpg
  drc_torso_head_diffuse.jpg
  extremities_diffuse.png
  extremities_diffuse_unplugged.jpg
  right_leg_diffuse_unplugged.jpg
  torso_diffuse.png
  torso_diffuse_unplugged.jpg
)

install(FILES ${files}
  DESTINATION ${GAZEBO_INSTALL_PATH}/atlas/materials/textures/)
