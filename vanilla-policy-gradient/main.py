
import argparse
from agent import Agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5000)
    args = parser.parse_args()

    agent = Agent(args.env, [args.hidden_features])

    agent.train(args.lr, args.epochs, args.batch_size)