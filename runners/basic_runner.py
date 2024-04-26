import os
import datetime

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, agent, env, train):
        self.agent = agent
        self.env = env
        self.train = train

        self.score = 0
        self.episode = 1

        self.graph_path = Path(Path.cwd()).joinpath("graphs", type(agent).__name__,
                                                    datetime.datetime.now().strftime("%y%m%d_%H%M"),
                                                    '_train' if self.train else 'run')

        self.weights_path_save = Path(Path.cwd()).joinpath("weights", type(agent).__name__,
                                                           datetime.datetime.now().strftime("%y%m%d_%H%M"))

        self.weights_path_load = Path(Path.cwd()).joinpath("weights", type(agent).__name__)
        self.writer = SummaryWriter(str(self.graph_path))

        if not self.train and os.path.isdir(self.weights_path_load):
            self.agent.load_model(str(self.weights_path_load))

    def summarize(self):
        # Graphs in tensorboard
        self.writer.add_scalar('Score per Episode', self.score, global_step=self.episode)

        if self.train and self.episode % 10 == 0:
            self.agent.save_model(str(self.weights_path_save))
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...

        self.episode += 1
        self.score = 0
        self.writer.flush()

    def run(self, episodes):
        while self.episode <= episodes:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs)
                if obs.last():
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            self.summarize()
        self.writer.close()
