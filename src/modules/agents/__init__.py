REGISTRY = {}

from .rnn_agent import RNNAgent
from .TM_agent import Teammate_model_agent

REGISTRY["rnn"] = RNNAgent
REGISTRY["TM_agent"] = Teammate_model_agent