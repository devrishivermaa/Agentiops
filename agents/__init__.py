# agents/__init__.py
"""
AgentOps Agents Package.
Contains MasterAgent, SubMaster, WorkerAgent, and ResidualAgent.
"""

from .master_agent import MasterAgent
from .sub_master import SubMaster

__all__ = ['MasterAgent', 'SubMaster']
