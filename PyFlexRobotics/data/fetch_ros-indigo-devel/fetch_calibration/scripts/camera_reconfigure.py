#!/usr/bin/env python

# Copyright (C) 2015 Fetch Robotics Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Michael Ferguson

import sys
import rospy
import dynamic_reconfigure.client

class CameraReconfigure(object):

    def __init__(self):
        self.client = dynamic_reconfigure.client.Client("head_camera/driver",
                                                        timeout=30,
                                                        config_callback=self.callback)

    def disable_auto(self):
        self.client.update_configuration({"auto_exposure": False,
                                          "auto_white_balance": False})

    def enable_auto(self):
        self.client.update_configuration({"auto_exposure": True,
                                          "auto_white_balance": True})

    def callback(self, config):
        rospy.loginfo("camera configured")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: camera_reconfigure --enable/disable")
        exit(-1)

    rospy.init_node("camera_reconfigure")
    reconfigure = CameraReconfigure()

    if sys.argv[1] == "--enable":
        reconfigure.enable_auto()
    else:
        reconfigure.disable_auto()

    rospy.sleep(1)
