"""Entry point of episodic memory: capsule bank + retriever + capsule builder.

    memory = EpisodicMemory.from_config(config, project_root)   # None if retrieve and evolve are both off
    memory.retriever   # None unless retrieve is on
    memory.builder     # None unless evolve is on (otherwise the memory is frozen)
"""

from XMind.episodic_memory.bank import CapsuleBank
from XMind.episodic_memory.capsule import CapsuleBuilder
from XMind.episodic_memory.retrieval import CapsuleRetriever
from XMind.episodic_memory.settings import load_settings


class EpisodicMemory:
    def __init__(self, settings, log=print):
        self.settings = settings
        self.bank = CapsuleBank(settings.bank_dir)
        self.retriever = CapsuleRetriever(self.bank, settings.retrieval) if settings.retrieve else None
        self.builder = CapsuleBuilder(self.bank, settings, log) if settings.evolve else None

    @classmethod
    def from_config(cls, config, project_root, log=print):
        settings = load_settings(config, project_root)
        if not (settings.retrieve or settings.evolve):
            return None
        return cls(settings, log=log)

    def describe(self):
        return "{} | retrieve: {} | evolve: {} | {}".format(
            self.bank.describe(),
            "on" if self.retriever is not None else "off",
            "on" if self.builder is not None else "off",
            self.settings.describe(),
        )
