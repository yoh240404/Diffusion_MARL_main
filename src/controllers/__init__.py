REGISTRY = {}

from .basic_controller import BasicMAC
from .tm_controller import TeammateMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["tm_mac"] = TeammateMAC