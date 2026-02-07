"""
Core module - Protocols, containers, and abstractions.
"""
from .protocols import DataRepository, TaskQueueProtocol
from .container import ServiceContainer

__all__ = [
    "DataRepository",
    "TaskQueueProtocol", 
    "ServiceContainer",
]
