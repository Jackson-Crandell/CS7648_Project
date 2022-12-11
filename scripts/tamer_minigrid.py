import matplotlib.pyplot as plt
import numpy as np
import pickle
from createNeuralNet import create_neural_net
from backprop import backprop
from forwardPass import forward_pass
import os
import sys
import random
#import utils

import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

class ManualControl:
    def __init__(
        self,
        env: MiniGridEnv,
        nn=None,
        agent_view: bool = False,
        window: Window = None,
        seed=None,
    ) -> None:
        self.env = env
        self.agent_view = agent_view
        self.seed = seed
        self.nn = nn

        self.key_value = 0

        if window is None:
            window = Window("minigrid - " + str(env.__class__))
        self.window = window
        self.window.reg_key_handler(self.key_handler)

    def keydown(self,key, trained_agent):

        if key == '0':
            self.key_value = 4
        elif key == '9':
            self.key_value = 3
        elif key == '8':
            self.key_value = 2
        elif key == '7':
            self.key_value = 1
        elif key == '6':
            self.key_value = 0
        elif key == '5':
            self.key_value = -1
        elif key == '4':
            self.key_value = -2
        elif key == '3':
            self.key_value = -3
        elif key == '2':
            self.key_value = -4
        elif key == '1':
            self.key_value = -5
        elif key == K_DOWN:
            pickle.dump(trained_agent, open('trained_agent.pkl', 'wb'))
            sys.exit()


    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        self.window.show(block=True)

    def step(self, action: MiniGridEnv.Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.redraw()

    def redraw(self):
        frame = self.env.get_frame(agent_pov=self.agent_view)
        self.window.show_img(frame)

    def reset(self, seed=None):
        self.env.reset(seed=seed)

        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        self.redraw()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": MiniGridEnv.Actions.left,
            "right": MiniGridEnv.Actions.right,
            "up": MiniGridEnv.Actions.forward,
            " ": MiniGridEnv.Actions.toggle,
            "pageup": MiniGridEnv.Actions.pickup,
            "pagedown": MiniGridEnv.Actions.drop,
            "enter": MiniGridEnv.Actions.done,
        }

        #action = key_to_action[key]

        #key_press_event = env.get_human_feedback()
        #keydown(key_press_event, nn)
        #print(key)
        self.keydown(key,nn)
        #human_feedback = int(human_feedback)
        print("Human Feedback: ",self.key_value)
        #human_feedback = 0
        print(self.env.agent_pos,self.env.agent_dir)

        feature = np.array([*self.env.agent_pos,self.env.agent_dir])
        #print(feature.shape)

        if self.key_value != 0:
            feature = np.reshape(feature, (3, 1))
            gradient = backprop(self.nn, feature, self.key_value, "MSE")
            for j in range(len(self.nn)):
                nn[j][0] -= alpha * gradient[j][0]  # Update the neural network weights
                nn[j][1] -= alpha * gradient[j][1]  # Update the neural network biases
            self.key_value = 0

        best_value = -1e10
        best_action = 0
        for action in ["left","right","up"]:

            next_pos = [*self.env.agent_pos]
            next_dir = self.env.agent_dir

            if action == "up":
                if self.env.agent_dir == 0:
                    next_pos = [self.env.agent_pos[0]+1,self.env.agent_pos[1]]

                if self.env.agent_dir == 1:
                    next_pos = [self.env.agent_pos[0],self.env.agent_pos[1]+1]

                if self.env.agent_dir == 2:
                    next_pos = [self.env.agent_pos[0]-1,self.env.agent_pos[1]]

                if self.env.agent_dir == 3:
                    next_pos = [self.env.agent_pos[0],self.env.agent_pos[1]-1]
                
                if next_pos[0] > 3:
                    next_pos[0] = 3

                if next_pos[1] > 3:
                    next_pos[1] = 3

                if next_pos[0] < 1:
                    next_pos[0] = 1

                if next_pos[1] > 1:
                    next_pos[1] = 1

            elif action == "left":
                next_dir = self.env.agent_dir - 1

                if next_dir < 0:
                    next_dir = 3

            elif action == "right":
                next_dir = self.env.agent_dir + 1

                if next_dir > 3:
                    next_dir = 1
            
            feature = np.array([next_pos[0],next_pos[1],next_dir])
            print("\nAction: ",action)
            print("Next state: ",next_pos[0],next_pos[1])
            print("Direction: ",next_dir)
            value = forward_pass(self.nn, feature.reshape(-1, 1))[0, 0]
            print("Value: ",value)
            print()
            if value > best_value:
                best_value = value
                best_action = action

        print("best action: ",best_action)
        action = key_to_action[best_action]
        self.step(action)  

if __name__ == "__main__":
    save = True
    #env = utils.make_env('MiniGrid-Empty-5x5-v0', 1)

    MiniGridEnv = gym.make('MiniGrid-Empty-5x5-v0')

    if True:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(MiniGridEnv)
        env = ImgObsWrapper(env)

    # network hyperparameters
    hidden_dims = [3]
    input_dim = 3
    output_dim = 1
    alpha = 1e-2
    nn = create_neural_net(numNodesPerLayer=hidden_dims, numInputDims=input_dim, numOutputDims=output_dim)

    manual_control = ManualControl(env, nn,agent_view=False, seed=1)
    manual_control.start()