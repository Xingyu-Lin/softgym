import os
import numpy as np
import threading
from copy import deepcopy

from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv

from autolab_core import YamlConfig


class Simtaur:
    """ Abstraction for one agent in the SimtaurScene """

    def __init__(self, positionControl):
        """ Initializes the simulated minitaur agent """

        self.positionControl = positionControl

        self.targetMotorPositions = None
        self.motorPositions = None
        self.motorVelocities = None
        self.positionError = None

    def setPdParams(self, kp, kd):
        """ Sets kp and kd """

        self.kp = kp
        self.kd = kd

    def setTargetMotorPositions(self, targets):
        """ Sets the target Motor Positions """

        self.targetMotorPositions = targets

    def updateMotorStates(self, position, velocity):
        """ Used to update the motor position and velocity """

        self.motorPositions = position
        self.motorVelocities = velocity

    def calculateTorque(self):
        """ Calcuates torque according to kp, kd """

        self.positionError = self.targetMotorPositions - self.motorPositions
        for i in range(len(self.positionError)):
            if np.abs(self.positionError[i] + 2 * np.pi) < np.abs(self.positionError[i]):
                self.positionError[i] += 2 * np.pi
            if np.abs(self.positionError[i] - 2 * np.pi) < np.abs(self.positionError[i]):
                self.positionError[i] -= 2 * np.pi

        torque = self.kp * self.positionError + self.kd * self.motorVelocities

        return torque

    def updateObservations(self, observations):
        """ Updates variables based on observations """

        self.motorVelocities = np.array(observations)[10:26:2]
        self.motorPositions = np.array(observations)[9:25:2]

    def legSpaceToMotorSpace(self, legSpaceActions):
        """ Converts 4 (e, s) pairs to the actions for 8 motors """

        motorSpaceActions = []
        extensionClip = 0.5
        swingClip = 0.5 * np.pi
        for i in range(0, len(legSpaceActions), 2):
            e = legSpaceActions[i]
            e = np.maximum(1.5 - extensionClip,
                           np.minimum(1.5 + extensionClip, e + 1.5))
            s = legSpaceActions[i + 1]
            s = np.maximum(-swingClip, np.minimum(swingClip, s))

            # Make the legs face the same way
            if i >= 4:
                e *= -1

            motorSpaceActions.append(e + s)
            motorSpaceActions.append(e - s)

        return motorSpaceActions

    def motorSpaceToLegSpace(self, motorSpaceActions):
        """ Converts 8 motor actions to 4 (e, s) pairs """

        pass

    def setLegPositions(self, actions):
        """ Sets the positions of the legs in leg space """

        self.setTargetMotorPositions(self.legSpaceToMotorSpace(actions))


class SimtaurScene:
    """ Abstraction for the rlminitaur.h scene.
    Allows position control
    """

    def __init__(self, positionControl=True):
        """ Initializes the Scene """
        # self.createFlexGym()

        self.local = os.uname()[1] == "josroy-desktop"

        self.numAgents = 100
        self.simtaurs = [Simtaur(positionControl)
                         for _ in range(self.numAgents)]
        self.frames = 0

        self.multithreading = False
        if self.multithreading:
            self.env = None
            self.observations = None
        else:
            self.createFlexGym()
            self.observations = self.env.reset()
        self.rewards = None
        self.dead = None

        for i in range(len(self.simtaurs)):
            self.simtaurs[i].updateObservations(self.observations[i])

        if (positionControl):
            self.setPdParams(-0.05, 0.01)

    def __del__(self):
        """ Destructor """

        # self.closeFlexGym()
        pass

    def createFlexGym(self):
        """ Creates the flex gym. Should be called in __init__ """

        if self.local:
            os.chdir('/home/josroy/Documents/flex/sw/devrel/libdev/flex/dev/rbd/bindings/gym')
        else:
            os.chdir('/workspace/rbd/bindings/gym')

        self.cfg = YamlConfig(
            './cfg/minitaur.yaml')
        set_flex_bin_path(
            '../../bin')
        self.env = FlexVecEnv(self.cfg)

    def resetFlexGym(self):
        """ Resets the flex gym """

        self.env.reset()

    def closeFlexGym(self):
        """ Closes the flex gym """

        self.env.close()

    def loopFlexGym(self, numFrames=None):
        """ Runs the flex gym in a new thread """

        self.gymThread = threading.Thread(
            target=self.flexGymThread, kwargs={"numFrames": numFrames})
        self.gymThread.start()
        self.setAllLegPositions(np.array(
            [[1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi] for _ in self.simtaurs]))
        while self.frames == 0:
            pass
        # self.threadingTest(numFrames)

    def flexGymThread(self, numFrames=None):
        self.createFlexGym()
        self.flexSimLoop(numFrames)

    def flexSimLoop(self, numFrames=None):
        """ Runs the loop of the gym """
        self.frames = 0

        self.observations = self.env.reset()
        for i in range(len(self.simtaurs)):
            self.simtaurs[i].updateObservations(self.observations[i])

        # Runs until numFrames is hit, or forever if numFrames is None
        while numFrames is None or self.frames < numFrames:
            actions = []
            for simtaur in self.simtaurs:
                actions.append(simtaur.calculateTorque().tolist())

            actions = np.array(actions)
            self.observations, self.rewards, self.dead, _ = self.env.step(
                actions)
            self.frames += 1
            # self.observations = self.env.reset()
            for i in range(len(self.simtaurs)):
                self.simtaurs[i].updateObservations(self.observations[i])
        self.closeFlexGym()

    def step(self, posActions):
        """ Only used if not async """

        torqueActions = []
        for simtaur in self.simtaurs:
            torqueActions.append(simtaur.calculateTorque().tolist())
        torqueActions = np.array(torqueActions)
        self.observations, self.rewards, self.dead, _ = self.env.step(
            torqueActions)
        for i in range(len(self.simtaurs)):
            self.simtaurs[i].updateObservations(self.observations[i])

        # print(self.observations[0][3:6])

        return self.observations, self.rewards, self.dead

    def setPdParams(self, kp, kd):
        """ Sets the gain and stiffness of the PD controller for the position
        of the legs
        """
        for simtaur in self.simtaurs:
            simtaur.setPdParams(kp, kd)

    def setAllLegPositions(self, actions):
        """ Lets the leg positions of all simtaurs """
        for i in range(len(self.simtaurs)):
            self.simtaurs[i].setLegPositions(actions[i])

    def getObsRewDead(self):
        """ Returns the observations, rewards, and dead """
        return self.observations, self.rewards, self.dead


def autotune(s, numFrames):
    s.loopFlexGym(numFrames=None)
    prevError = 0.0
    kp = 0.05
    kd = 0.01
    i = s.frames
    stepsize = 1.0
    while numFrames is None or i < numFrames:
        print(i)
        s.setPdParams(-kp, kd)
        error = runTest(s, 1000)
        if prevError != error:
            print(kp, kd, error)
            # if np.abs(error) < np.abs(prevError):
            #     kd += stepsize * np.abs(error)
            # else:
            #     kd -= stepsize * np.abs(error)
        prevError = error
        i = s.frames
    s.gymThread.join()


def runTest(s, numFrames):
    local = False
    if local:
        os.chdir('/home/josroy/Documents/flex/sw/devrel/libdev/flex/dev/rbd/bindings/gym')
    else:
        os.chdir('/workspace/rbd/bindings/gym')
    sumError = 0.0
    i = s.frames
    numStart = i
    while i < numStart + numFrames:
        # print("i", i, "end", numStart + numFrames)
        # print(i, i % numFrames)
        actions = np.array(
            [np.sin(i / 10) / 2, 0.0 * np.pi, np.sin(i / 10) / 2, 0.0 * np.pi, np.sin(i / 10) / 2, 0.0 * np.pi, np.sin(i / 10) / 2, 0.0 * np.pi])
        # print("actions", actions)
        for simtaur in s.simtaurs:
            simtaur.setLegPositions(actions)
            if simtaur.positionError is not None:
                i = s.frames
                sumError += np.mean(simtaur.positionError) + \
                    np.mean(simtaur.motorVelocities)
    if i != 0:
        return sumError / float(i)
    else:
        return 0


def main():
    np.set_printoptions(suppress=True)
    s = SimtaurScene()

    for simtaur in s.simtaurs:
        simtaur.setLegPositions(np.array(
            [1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi]))
    autotune(s, numFrames=None)


if __name__ == "__main__":
    main()
