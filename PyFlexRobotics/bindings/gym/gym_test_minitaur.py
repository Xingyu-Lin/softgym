import os
import random
import numpy as np
from ctypes import *

# Load Flex Gym library
debug = 0
if (os.name == "nt"):
    flexGymPath = os.path.dirname(os.path.realpath(__file__)) + "/../../bin/win64/"
    os.chdir(flexGymPath)
    if (debug): flexGym = cdll.LoadLibrary("NvFlexGymDebugCUDA_x64")
    else: flexGym = cdll.LoadLibrary("NvFlexGymReleaseCUDA_x64")
else:
    flexGymPath = os.path.dirname(os.path.realpath(__file__)) + "/../../bin/linux64/"
    os.chdir(flexGymPath)
    if (debug): flexGym = cdll.LoadLibrary(flexGymPath + "NvFlexGymDebugCUDA_x64.so")
    else: flexGym = cdll.LoadLibrary(flexGymPath + "NvFlexGymReleaseCUDA_x64.so")
    
# Initialize Flex Gym
flexGym.NvFlexGymInit()

# Parameters
loadPath = '"../../data/ant.xml"'
numAgents = 40
numPerRow = 5
numObservations = 39
numActions = 8
numSubsteps = 4
numIterations = 50
pause = 'true'
doLearning = 'true'

# Load a scene
flexGym.NvFlexGymLoadScene('RL Minitaur', f'''
                                     {{
                                         "LoadPath": {loadPath},
                                         "NumAgents": {numAgents},
                                         "NumPerRow": {numPerRow},
                                         "NumObservations": {numObservations},
                                         "NumActions": {numActions},
                                         "NumSubsteps": {numSubsteps},
                                         "NumIterations": {numIterations},
                                         "Pause": {pause},
                                         "DoLearning": {doLearning}
                                     }}
                                     ''')

# Buffers                                     
totalActions = numAgents * numActions
ActionBuffType = c_float * totalActions
actionBuff = ActionBuffType()
totalObservations = numAgents * numObservations
ObservationBuffType = c_float * totalObservations
observationBuff = ObservationBuffType()
RewardBuffType = c_float * numAgents
rewardBuff = RewardBuffType()
DeathBuffType = c_byte * numAgents
deathBuff = DeathBuffType()

def VelocityControl(frame, agent, invfrequency = 20, velocity = 4.0):
    actionBuff[agent * numActions + 0] = 2.25 * velocity * (-1) ** (frame // invfrequency)
    actionBuff[agent * numActions + 1] = velocity  * (-1) ** (frame // invfrequency)
    actionBuff[agent * numActions + 4] = -velocity * (-1) ** (frame // invfrequency)
    actionBuff[agent * numActions + 5] = -2.25 * velocity * (-1) ** (frame // invfrequency)

    actionBuff[agent * numActions + 2] = 1.3 * velocity * (-1) ** (frame // invfrequency) 
    actionBuff[agent * numActions + 3] = 1.35 * velocity * (-1) ** (frame // invfrequency) 
    actionBuff[agent * numActions + 6] = -1.35 * velocity * (-1) ** (frame // invfrequency)
    actionBuff[agent * numActions + 7] = -1.3 * velocity * (-1) ** (frame // invfrequency)

# Simulation loop
quit = 0
frames = -20
while (quit == 0):
    for agent in range(numAgents):
        
        if True:
            fr = frames
            if frames < 0:
                fr = 0
            VelocityControl(fr, agent, invfrequency = 40, velocity = 2.0)
        else:
            phaseShift = -0.33
            backPhaseShift = 0.05
            frequency = 0.1
            actionsScale = 1.0

            actionBuff[agent * numActions + 0] = actionsScale * np.sin(frequency * frames - phaseShift)
            actionBuff[agent * numActions + 1] = actionsScale * np.sin(frequency * frames)
            actionBuff[agent * numActions + 4] = -actionsScale * np.sin(frequency * frames)
            actionBuff[agent * numActions + 5] = -actionsScale * np.sin(frequency * frames - phaseShift)
            
            actionBuff[agent * numActions + 2] = actionsScale * np.sin(frequency * frames - phaseShift - backPhaseShift)
            actionBuff[agent * numActions + 3] = actionsScale * np.sin(frequency * frames - backPhaseShift)
            actionBuff[agent * numActions + 6] = -actionsScale * np.sin(frequency * frames - backPhaseShift)
            actionBuff[agent * numActions + 7] = -actionsScale * np.sin(frequency * frames - phaseShift - backPhaseShift)

      #  for action in range(0, 1):
      #      actionBuff[agent * numActions + action] = -0.3 * np.cos(0.033 * frames) * (1.0)**action
      #  for action in range(4, 5):
      #      actionBuff[agent * numActions + action] = -0.3 * np.cos(0.033 * frames) * (1.0)**action

     #   for action in range(2, 3):
     #       actionBuff[agent * numActions + action] = 0.3 * np.cos(0.033 * frames) * (-1.0)**action
     #   for action in range(6, 7):
     #       actionBuff[agent * numActions + action] = 0.3 * np.cos(0.033 * frames) * (-1.0)**action

    flexGym.NvFlexGymSetActions(actionBuff, 0, totalActions)  
    quit = flexGym.NvFlexGymUpdate()

    flexGym.NvFlexGymGetRewards(rewardBuff, deathBuff, 0, numAgents)
    for agent in range(numAgents):
        if (deathBuff[agent]):
            flexGym.NvFlexGymResetAgent(agent)
    flexGym.NvFlexGymGetObservations(observationBuff, 0, totalObservations)

    frames += 1
    if (frames > 500):
        flexGym.NvFlexGymResetAllAgents()
        frames -= 515

    #flexGym.NvFlexGymGetExtras(extraBuff, 0, totalExtras) # Number ???
    #flexGym.NvFlexGymResetAllAgents()
    #flexGym.NvFlexGymGetObservations(agentObservationBuff, agent * numObservations, numObservations) # Get agent observations

# Shutdown Flex Gym
flexGym.NvFlexGymShutdown()
