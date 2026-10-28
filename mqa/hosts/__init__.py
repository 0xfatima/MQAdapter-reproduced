"""Fine-tuning methods ("hosts") for the end-to-end engine; each can carry the MQAdapter plug-in."""
from mqa.hosts.base import HostBase
from mqa.hosts.factory import METHODS, build_host, is_trainable, method_config, split_name
from mqa.hosts.maple import MaPLeHost
from mqa.hosts.mmrl import MMRLHost, MMRLPlusPlusHost
from mqa.hosts.promptsrc import PromptSRCHost
from mqa.hosts.simple import CLIPAdapterHost, CLIPHost, CoCoOpHost, CoOpHost

__all__ = [
    "HostBase", "METHODS", "build_host", "is_trainable", "method_config", "split_name", "MaPLeHost", "MMRLHost",
    "MMRLPlusPlusHost", "PromptSRCHost", "CLIPAdapterHost", "CLIPHost", "CoCoOpHost", "CoOpHost",
]
