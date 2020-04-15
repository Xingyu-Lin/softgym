import os
names = ["base_link_collision",
"bellows_link",
"bellows_link_collision",
"elbow_flex_link_collision",
"estop_link",
"forearm_roll_link_collision",
"gripper_link",
"head_pan_link_collision",
"head_tilt_link_collision",
"laser_link",
"l_gripper_finger_link",
"l_wheel_link",
"l_wheel_link_collision",
"r_gripper_finger_link",
"r_wheel_link",
"r_wheel_link_collision",
"shoulder_lift_link_collision",
"shoulder_pan_link_collision",
"torso_fixed_link",
"torso_lift_link_collision",
"upperarm_roll_link_collision",
"wrist_flex_link_collision",
"wrist_roll_link_collision"]
for x in names:
	os.system("meshlabserver -i "+x+".stl -o "+x+".obj")
	os.system("testVHACD.exe --input "+x+".obj --output "+x+".wrl --resolution=1000000 --maxNumVerticesPerCH=32")
