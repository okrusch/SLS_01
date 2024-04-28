import collections
import os.path
import datetime

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


# TODO Add epsilon and moving average graphs
class Runner:
    def __init__(self, agent, env, train):
        self.agent = agent
        self.env = env
        self.train = train

        self.moving_avg = collections.deque(maxlen=50)
        self.score = 0
        self.scores = []
        self.episode = 1
        self.epsilon = 1.0

        self.graph_path = Path("..", "graphs", type(agent).__name__, datetime.datetime.now().strftime("%y%m%d_%H%M"),
                               'train' if self.train else 'run')

        self.weights_path_save = Path("..", "weights", type(agent).__name__,
                                      datetime.datetime.now().strftime("%y%m%d_%H%M"))

        self.weights_path_load = Path("..", "weights", type(agent).__name__)
        self.writer = SummaryWriter(str(self.graph_path))

        if not self.train and os.path.isdir(self.weights_path_load):
            self.agent.load_model(str(self.weights_path_load))
            pass
        else:
            self.weights_path_load.mkdir(parents=True, exist_ok=True)

    def summarize(self):
        # Graphs in tensorboard
        self.writer.add_scalar('Score per Episode', self.score, global_step=self.episode)
        self.writer.add_scalar('Epsilon', self.epsilon, global_step=self.episode)
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(str(self.weights_path_save))
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...
        self.scores.append(self.score)
        if len(self.scores) > 50:
            self.scores.pop(0)
            self.writer.add_scalar('Moving Average', (sum(self.scores) / 50), global_step=self.episode)

        self.episode += 1
        self.score = 0
        self.writer.flush()

    def run(self):
        while self.score < 20:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs, self.epsilon)
                if obs.last():
                    # epsilon decreases linear until 0.01
                    if self.epsilon > 0.1:
                        self.epsilon -= 0.0001
                    break

                obs = self.env.step(action)
                self.score += obs.reward

            self.summarize()
        self.writer.close()
