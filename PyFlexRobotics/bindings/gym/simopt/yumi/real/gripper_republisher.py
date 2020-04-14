import rospy
import zmq
from lula_yumi.abb_client import ABBClient


class GripperRepublisher:

    def __init__(self, sub_port=5559):
        rospy.init_node('gripper_republisher')

        zmq_context = zmq.Context()
        topicfilter = u'gripper'
        self.sub_socket = zmq_context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:%s" % sub_port)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
        # self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.abb_client = ABBClient()

    def run(self):

        while not rospy.is_shutdown():
            print ("Waiting for gripper commands...")
            msg = self.sub_socket.recv()
            topic, msg_data = msg.split()
            print ("Gripper command received: " + msg_data)
            msg_elems = msg_data.split(",")
            side = msg_elems[0]
            command = msg_elems[1]

            if ((side == 'left' or side == 'right') and
               (command == 'open' or command == 'close')):
                self.abb_client.gripper(side, command)


if __name__ == '__main__':
    gripper_republisher = GripperRepublisher()
    gripper_republisher.run()


