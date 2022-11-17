#!/usr/bin/env python3

import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

average_reward = 0
count = 0

class MinigridGame:

    def __init__(self,env_name,num_trials=10):

        self.rewards = np.zeros(num_trials)
        self.count = 0
        self.num_trials = num_trials
        self.env_name = env_name

    def redraw(self,window, img):
        window.show_img(img)


    def reset(self,env, window, seed=None):
        env.reset(seed=seed)

        if hasattr(env, "mission"):
            print("Mission: %s" % env.mission)
            window.set_caption(env.mission)

        img = env.get_frame()

        self.redraw(window, img)


    def step(self,env, window, action):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={env.step_count}, reward={reward:.2f}")
        self.rewards[self.count] += reward

        if terminated:
            self.count += 1

            if self.count == self.num_trials:
                window.close()
                #print(self.rewards)
                print("Total Average Reward: ",round(np.mean(self.rewards),3))
                x = np.arange(self.num_trials)

                fig, ax = plt.subplots()
                plt.title("Human Rewards for " + str(self.env_name))
                plt.xlabel("Trials")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.ylabel("Rewards")
                plt.plot(x,self.rewards)
                plt.show()

                #exit()

            print("terminated!")
            self.reset(env, window)
        elif truncated:
            print("truncated!")
            self.reset(env, window)
        else:
            img = env.get_frame()
            self.redraw(window, img)


    def key_handler(self,env, window, event):
        print("pressed", event.key)

        if event.key == "escape":
            window.close()
            return

        if event.key == "backspace":
            self.reset(env, window)
            return

        if event.key == "left":
            self.step(env, window, env.actions.left)
            return
        if event.key == "right":
            self.step(env, window, env.actions.right)
            return
        if event.key == "up":
            self.step(env, window, env.actions.forward)
            return

        # Spacebar
        if event.key == " ":
            self.step(env, window, env.actions.toggle)
            return
        if event.key == "pageup":
            self.step(env, window, env.actions.pickup)
            return
        if event.key == "pagedown":
            self.step(env, window, env.actions.drop)
            return

        if event.key == "enter":
            self.step(env, window, env.actions.done)
            return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-Empty-5x5-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=True,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    env = gym.make(
        args.env,
        tile_size=args.tile_size,
    )

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    game = MinigridGame(args.env)

    window = Window("minigrid - " + args.env)
    window.reg_key_handler(lambda event: game.key_handler(env, window, event))

    seed = None if args.seed == -1 else args.seed
    game.reset(env, window, seed)

    # Blocking event loop
    window.show(block=True)
