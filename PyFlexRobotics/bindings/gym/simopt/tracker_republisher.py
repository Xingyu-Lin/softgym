import rospy
import tf2_ros
import tf.transformations as transf
import numpy as np
import zmq
from sensor_msgs.msg import JointState
import geometry_msgs
import tf_conversions

class TrackerRepublisher:

    def __init__(self, pub_port=5555, task='cabinet', robot='franka'):
        rospy.init_node('tracker_republisher')

        self.robot = robot
        if robot == 'yumi':
            flex_origin_offset = np.asarray([-0.02, 0.1, 0.0])
        else:
            flex_origin_offset = np.asarray([0.0, 0.0, 0.0])

        xaxis, yaxis = (1, 0, 0), (0, 1, 0)

        self.flex_rot = np.dot(transf.rotation_matrix(-np.pi / 2.0, yaxis),
                               transf.rotation_matrix(-np.pi / 2.0, xaxis))
        self.flex_translation = transf.translation_matrix(flex_origin_offset)
        self.flex_transform = np.dot(self.flex_translation, self.flex_rot)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        zmq_context = zmq.Context()
        self.pub_socket = zmq_context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:%s" % pub_port)

        if task == 'ropepeg':
            self.run_tf_listener_ropepeg()
        elif task == 'cabinet':
            self.sektion_joint_subscriber = rospy.Subscriber("/tracker/sektion_cabinet/joint_states",
                                                             JointState, self.sektion_joint_callback)
            self.drawer_joint = 0.0
            self.run_tf_listener_cabinet()

    def run_tf_listener_ropepeg(self):
        rate = rospy.Rate(30.0)
        i = 0
        while not rospy.is_shutdown():
            try:

                peg_translation_flex, peg_quat_flex = self.get_obj_tf_flex('yumi', 'peg')
                pegholder_translation_flex, pegholder_quat_flex = self.get_obj_tf_flex('yumi', 'pegHolder')
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print ("ERROR looking up object transforms.")
                continue

            # Move to the middle of the cylinder
            peg_z_corr = self.qv_mult(peg_quat_flex, np.asarray([0.0, 0.0, 1.0]))
            peg_translation_flex += peg_z_corr * 0.03

            # Angle
            # pegholder_translation_flex += (
            #         np.asarray([-0.153353, 0.121009, 0.678978]) -
            #         np.asarray([-0.127683, 0.089186, 0.736121]) +
            #         np.asarray([0.0, -0.01, 0.01]))

            msg_str = "%f,%f,%f,%f,%f,%f" % (
                pegholder_translation_flex[0], pegholder_translation_flex[1], pegholder_translation_flex[2],
                peg_translation_flex[0], peg_translation_flex[1], peg_translation_flex[2])

            if i % 100 == 0:
                print ("Publishing: " + msg_str)
            self.pub_socket.send_string("track " + msg_str)
            i += 1
            rate.sleep()

    def sektion_joint_callback(self, data):
        for i in range(len(data.name)):
            if data.name[i] == 'drawer_top_joint':
                self.drawer_joint = data.position[i]
                self.broadcast_drawer_handle()
                break

    def run_tf_listener_cabinet(self):
        rate = rospy.Rate(30.0)
        i = 0
        while not rospy.is_shutdown():
            try:
                sektion_translation_flex, sektion_quat_flex = self.get_obj_tf_flex(self.robot, 'drawer_handle_top')
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print ("ERROR looking up object transforms.")
                continue

            # drawer_translation = (sektion_translation_flex -
            #                       np.asarray([0.0, 0.39, 1.04170506]) +
            #                       np.asarray([0.0, 0.73, 0.676649]) -
            #                       np.asarray([0.0, 0.0, self.drawer_joint]))

            # drawer_translation = (sektion_translation_flex -
            #                       np.asarray([0.0, 0.4, 0.02]) +
            #                       np.asarray([0.0, 0.71, 0.0]) -
            #                       np.asarray([0.0, 0.0, self.drawer_joint]))

            # drawer_translation += (np.asarray([0.000000, 0.706200, 1.028500]) -
            #                        np.asarray([0.003821, 0.688682, 1.023178]))

            drawer_translation = sektion_translation_flex

            msg_str = "%f,%f,%f" % (
                drawer_translation[0], drawer_translation[1], drawer_translation[2])

            if i % 100 == 0:
                print ("Publishing: " + msg_str)
            self.pub_socket.send_string("track " + msg_str)
            i += 1
            rate.sleep()

    def broadcast_drawer_handle(self):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "sektion"
        t.child_frame_id = "drawer_handle_top"
        t.transform.translation.x = 0.375 + self.drawer_joint
        t.transform.translation.y = 0.03
        t.transform.translation.z = 0.34
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)

    def get_obj_tf_flex(self, from_frame, to_frame):
        obj_trans = self.tf_buffer.lookup_transform(from_frame, to_frame, rospy.Time(0))
        obj_trans = obj_trans.transform

        obj_translation = np.asarray([obj_trans.translation.x, obj_trans.translation.y, obj_trans.translation.z])
        obj_rotation = np.asarray(
            [obj_trans.rotation.x, obj_trans.rotation.y, obj_trans.rotation.z, obj_trans.rotation.w])
        obj_transform = transf.concatenate_matrices(transf.translation_matrix(obj_translation),
                                                    transf.quaternion_matrix(obj_rotation))
        obj_transform_flex = np.dot(self.flex_transform, obj_transform)

        obj_translation_flex = transf.translation_from_matrix(obj_transform_flex)
        obj_quat_flex = transf.quaternion_from_matrix(obj_transform_flex)
        return obj_translation_flex, obj_quat_flex

    def qv_mult(self, q1, v1):
        v1 = transf.unit_vector(v1)
        q2 = list(v1)
        q2.append(0.0)
        return transf.quaternion_multiply(
            transf.quaternion_multiply(q1, q2),
            transf.quaternion_conjugate(q1)
        )[:3]

if __name__ == '__main__':

    TrackerRepublisher(5555, task='cabinet', robot='franka')
    rospy.spin()


