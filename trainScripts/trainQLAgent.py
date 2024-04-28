# TODO Write the trainQLAgent script
from absl import app

from env import Env
from runners.main_runner import Runner
from agents.QLearningAgent import QLearningAgent

_CONFIG = dict(
    screen_size=64,
    minimap_size=64,
    visualize=False,
    train=True,
    agent=QLearningAgent
)

def main(unused_argv):

    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size']
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = Runner(
        agent=agent,
        env=env,
        train=_CONFIG['train']
    )

    runner.run()


if __name__ == "__main__":
    app.run(main)
