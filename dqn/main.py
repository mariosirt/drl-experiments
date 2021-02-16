"""Main module for running dqn with set hyperparameters as constants below."""

import argparse

from utils.agent import Agent
from utils.models import Models
from utils.atari_wrappers import wrap_deepmind, make_atari

SEED = 42

GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_MAX = 1.0
BATCH_SIZE = 32
MAX_MEMORY_LENGTH = 10000
MAX_STEPS_P_EPISODE = 10000
MAX_EPISODES = 10000000
MAX_TEST_EPISODES = 10
MAX_FRAMES = 10000000
EPS_RANDOM_FRAMES = 50000
EPS_GREEDY_FRAMES = 1000000.
UPDATE_AFTER_ACTIONS = 4
UPDATE_TARGET_NETWORK = 10000
SAVE_MODEL_STEPS = 1000000
SAVE_MODEL_PATH = "models_save/"
LEARNING_RATE = 0.00025
#INPUT_DIMS = (210, 160, 3,)
INPUT_DIMS = (84,84,4,)

parser = argparse.ArgumentParser(description="dqn and dueling dqn")
parser.add_argument("--dueling", type=int, default=0,
                    help="Dueling DQN => 0 = No, 1 = Yes")
parser.add_argument("--double", type=int, default=0,
                    help="DDQN => 0 = No, 1 = Yes")
parser.add_argument("--env", type=str, default="Breakout-v0",
                    help="Type in the env name")
parser.add_argument("--test", type=str,
                    help="provide path to model for testing")
parser.add_argument("--render", type=bool, default=False,
                    help="argument whether to render or not while testing")
args = parser.parse_args()

if __name__ == "__main__":
    env = make_atari(args.env)
    env = wrap_deepmind(env, episode_life=True,
                        clip_rewards=True, frame_stack=True, scale=True)

    print(env.observation_space.shape)
    num_actions = env.action_space.n

    model = Models.dqn(INPUT_DIMS, num_actions)
    model_target = Models.dqn(INPUT_DIMS, num_actions)

    if args.test is not None:
        print("TESTING")

        env = make_atari(args.env)
        env = wrap_deepmind(env, episode_life=False,
                            clip_rewards=False, frame_stack=True, scale=True)

        path = args.test
        model_test = Models.load_model(path)
        agent = Agent(env, model_test, model_test, EPSILON_MAX)
        agent.test(MAX_TEST_EPISODES, MAX_STEPS_P_EPISODE, args.render)

    elif args.dueling == 1:
        print("TRAIN DUELING")

        model_dueling = Models.dueling_dqn(INPUT_DIMS, num_actions)
        model_target_dueling = Models.dueling_dqn(INPUT_DIMS, num_actions)

        agent = Agent(env, model_dueling, model_target_dueling, EPSILON_MAX)
        agent.train(MAX_MEMORY_LENGTH, BATCH_SIZE, GAMMA, LEARNING_RATE,
                    MAX_STEPS_P_EPISODE, MAX_EPISODES, MAX_FRAMES,  EPS_RANDOM_FRAMES,
                    EPS_GREEDY_FRAMES, UPDATE_AFTER_ACTIONS, UPDATE_TARGET_NETWORK,
                    SAVE_MODEL_STEPS, SAVE_MODEL_PATH, False)

    else:
        if args.double == 1:
            print("TRAIN DOUBLE")
        else:
            print("TRAIN VANILLA")

        agent = Agent(env, model, model_target, EPSILON_MAX)
        agent.train(MAX_MEMORY_LENGTH, BATCH_SIZE, GAMMA, LEARNING_RATE,
                    MAX_STEPS_P_EPISODE, MAX_EPISODES, MAX_FRAMES,  EPS_RANDOM_FRAMES,
                    EPS_GREEDY_FRAMES, UPDATE_AFTER_ACTIONS, UPDATE_TARGET_NETWORK,
                    SAVE_MODEL_STEPS, SAVE_MODEL_PATH, args.double == 1)
