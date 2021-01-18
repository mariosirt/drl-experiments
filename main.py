"""Main module for running dqn with set hyperparameters as constants below."""

import argparse

from agent import Agent
from models import Models
from atari_wrappers import wrap_deepmind, make_atari

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
INPUT_DIMS = (84, 84, 4,)
ENV_ID = "Breakout-v0"

parser = argparse.ArgumentParser(description="dqn and dueling dqn")
parser.add_argument("--dueling", type=int)
parser.add_argument("--double", type=int)
parser.add_argument("--test", type=int)
args = parser.parse_args()

if __name__ == "__main__":
    env = make_atari(ENV_ID)
    env = wrap_deepmind(env, episode_life= True, clip_rewards= True, frame_stack= True, scale= True)




    num_actions = env.action_space.n

    model = Models.dqn(INPUT_DIMS, num_actions)
    model_target = Models.dqn(INPUT_DIMS, num_actions)

    model_dueling = Models.dueling_dqn(INPUT_DIMS, num_actions)
    model_target_dueling = Models.dueling_dqn(INPUT_DIMS, num_actions)

    model_test = Models.load_model("models_save/double/model-08-01-202117-57-57.h5")


    if args.test == 1:
        agent = Agent(env, model_test, model_test, EPSILON_MAX)
        agent.test(MAX_TEST_EPISODES, MAX_STEPS_P_EPISODE)
    else:

        if args.dueling ==1:
            print("DUELING")
            agent = Agent(env, model_dueling, model_target_dueling, EPSILON_MAX)
            agent.train(MAX_MEMORY_LENGTH, BATCH_SIZE, GAMMA, LEARNING_RATE,
                        MAX_STEPS_P_EPISODE, MAX_EPISODES, MAX_FRAMES,  EPS_RANDOM_FRAMES,
                        EPS_GREEDY_FRAMES, UPDATE_AFTER_ACTIONS, UPDATE_TARGET_NETWORK,
                        SAVE_MODEL_STEPS, SAVE_MODEL_PATH, True)

        elif args.double ==1:
            print("DOUBLE")
            agent = Agent(env, model,model_target, EPSILON_MAX)
            agent.train(MAX_MEMORY_LENGTH, BATCH_SIZE, GAMMA, LEARNING_RATE,
                        MAX_STEPS_P_EPISODE, MAX_EPISODES, MAX_FRAMES,  EPS_RANDOM_FRAMES,
                        EPS_GREEDY_FRAMES, UPDATE_AFTER_ACTIONS, UPDATE_TARGET_NETWORK,
                        SAVE_MODEL_STEPS, SAVE_MODEL_PATH, True)
        else:
            agent = Agent(env, model,model_target, EPSILON_MAX)
            agent.train(MAX_MEMORY_LENGTH, BATCH_SIZE, GAMMA, LEARNING_RATE,
                        MAX_STEPS_P_EPISODE, MAX_EPISODES, MAX_FRAMES,  EPS_RANDOM_FRAMES,
                        EPS_GREEDY_FRAMES, UPDATE_AFTER_ACTIONS, UPDATE_TARGET_NETWORK,
                        SAVE_MODEL_STEPS, SAVE_MODEL_PATH, False)
