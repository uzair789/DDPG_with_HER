import gym
import envs
from algo.ddpg import DDPG


def main():
    OUTPUT_PATH = './Results'
    env = gym.make('Pushing2D-v0')
    algo = DDPG(env,OUTPUT_PATH)
    algo.train(5000, hindsight=True)


if __name__ == '__main__':
    print("Hindsight started")
    main()
