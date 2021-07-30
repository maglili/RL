from gym import Env, spaces
from gym.spaces import space


class RNAFoldingEnv(Env):
    """
    RNA folding enviroment for OpenAI gym.

    Attribute:
        None

    Methods:
        None
    """

    def __init__(self, rna_seq):
        super().__init__()

        self.rna_len = len(rna_seq)
        self.action_space = spaces.Box(low=0, high=len(self.rna_len), shape=(2,))
