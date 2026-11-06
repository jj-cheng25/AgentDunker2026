"""AgentDunker M6: event timeline -> Chinese commentary -> Mossland voice -> aligned, ducked mix."""
from .pipeline import run
from .script import build_slots, write_script

__all__ = ["build_slots", "run", "write_script"]
