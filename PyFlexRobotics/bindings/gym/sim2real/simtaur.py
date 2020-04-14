import os
import numpy as np
import threading
import sys

from flex_gym.flex_vec_env import set_flex_bin_path, FlexVecEnv

from autolab_core import YamlConfig


class Simtaur:
    """ Abstraction for one agent in the SimtaurScene """

    def __init__(self, positionControl):
        """ Initializes the simulated minitaur agent """

        self.numMotors = 8

        self.positionControl = positionControl

        self.extensionOffsetTerm = 1.5

        self.targetMotorPositions = None
        self.motorPositions = []
        self.motorVelocities = []
        self.positionError = None
        # in frames, currently broken do not change other than 0. TODO: Josh will fix
        self.pdLatency = 0
        self.maxHistoryLen = 20  # Not super relevant, just has to be more than pdlatency

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

        if len(self.motorPositions) < self.pdLatency + 1:
            motorPositions = self.motorPositions[0]
            motorVelocities = self.motorVelocities[0]
        else:
            motorPositions = self.motorPositions[-self.pdLatency - 1]
            motorVelocities = self.motorVelocities[-self.pdLatency - 1]

        self.positionError = self.targetMotorPositions - motorPositions

        for i in range(len(self.positionError)):
            if np.abs(self.positionError[i] + 2 * np.pi) < np.abs(self.positionError[i]):
                self.positionError[i] += 2 * np.pi
            if np.abs(self.positionError[i] - 2 * np.pi) < np.abs(self.positionError[i]):
                self.positionError[i] -= 2 * np.pi

        torque = self.kp * self.positionError + self.kd * motorVelocities

        return torque

    def updateObservations(self, observations):
        """ Updates variables based on observations """

        self.motorVelocities.append(np.array(observations)[10:26:2])
        self.motorPositions.append(np.array(observations)[9:25:2])

        while len(self.motorVelocities) > self.maxHistoryLen:
            del self.motorVelocities[0]
        while len(self.motorPositions) > self.maxHistoryLen:
            del self.motorPositions[0]

    def legSpaceToMotorSpace(self, legSpaceActions):
        """ Converts 4 (e, s) pairs to the actions for 8 motors """

        motorSpaceActions = []
        extensionClip = 0.5
        swingClip = 0.5 * np.pi
        for i in range(0, self.numMotors, 2):
            e = legSpaceActions[i]
            e = np.maximum(self.extensionOffsetTerm - extensionClip,
                           np.minimum(self.extensionOffsetTerm + extensionClip, e + self.extensionOffsetTerm))
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

    def __init__(self, positionControl=True, randomizing=True):
        """ Initializes the Scene """
        # self.createFlexGym()

        self.frames = 0
        self.latency = 1  # in frames
        self.latencyLow = 1
        self.latencyHigh = 2
        self.maxHistoryLen = 20
        self.minSpeed = -2.  # in m/s
        self.maxSpeed = 2.
        self.targetSpeed = 0.5
        self.random_freq = 10000
        self.cfg = YamlConfig(
            './cfg/minitaur.yaml')

        self.randomizing = True

        self.observations = []

        self.on_saturn = False

        self.multithreading = False
        if self.multithreading:
            self.env = None
            self.observations = None
        else:
            self.createFlexGym()
            self.observations.append(self.env.reset())
        self.rewards = None
        self.dead = None

        self.numAgents = self.env.num_envs
        self.simtaurs = [Simtaur(positionControl)
                         for _ in range(self.numAgents)]

        for i in range(len(self.simtaurs)):
            self.simtaurs[i].updateObservations(self.observations[-1][i])

        if (positionControl):
            self.setPdParams(-5, 0.1)

    def __del__(self):
        """ Destructor """

        # self.closeFlexGym()
        pass

    def createFlexGym(self):
        """ Creates the flex gym. Should be called in __init__ """

        if self.on_saturn:
            os.chdir('/workspace/rbd/bindings/gym')

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

    def step(self, posActions):
        """ Only used if not async """

        self.frames += 1
        self.setAllLegPositions(posActions)

        if self.randomizing and (self.frames % self.random_freq) == 0:
            self.latency = np.random.randint(self.latencyLow, self.latencyHigh
                                             + 1)
            # print("RANDOM LATENCY", self.latency)
            # self.targetSpeed = np.random.rand() * (self.maxSpeed - self.minSpeed) + self.minSpeed
            # print("RANDOM SPEED", self.targetSpeed)

        torqueActions = []
        for simtaur in self.simtaurs:
            simtaurActions = simtaur.calculateTorque().tolist()
            simtaurActions.append(self.targetSpeed)
            torqueActions.append(simtaurActions)

        torqueActions = np.array(torqueActions)

        curObservations, self.rewards, self.dead, _ = self.env.step(
            torqueActions)

        self.observations.append(curObservations)
        while len(self.observations) > self.maxHistoryLen:
            del self.observations[0]
        for i in range(len(self.simtaurs)):
            self.simtaurs[i].updateObservations(self.observations[-1][i])

        return self.observations[-self.latency - 1], self.rewards, self.dead

    def setPdParams(self, kp, kd):
        """ Sets the gain and stiffness of the PD controller for the position
        of the legs
        """
        for simtaur in self.simtaurs:
            simtaur.setPdParams(kp, kd)

    def setAllLegPositions(self, actions):
        """ Lets the leg positions of all simtaurs """

        for i in range(len(self.simtaurs)):
            # print(i)
            # print(len(actions))
            self.simtaurs[i].setLegPositions(actions[i])
            # print("done")

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
    while True:
        e = np.sin(s.frames / 10) / 2
        s.step([np.array(
            [e, 0.0 * np.pi, e, 0.0 * np.pi, e, 0.0 * np.pi, e, 0.0 * np.pi])
            for _ in range(len(s.simtaurs))])

        s.env.reset()


if __name__ == "__main__":
    main()
