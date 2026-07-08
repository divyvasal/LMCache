# SPDX-License-Identifier: Apache-2.0
"""
Controller protocol definitions for cache management and configuration.

This module defines the protocol for:
- CLEAR: Clear all caches in the server
- GET_CHUNK_SIZE: Get the chunk size configuration from the server
"""

# First Party
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "CLEAR",
    "GET_CHUNK_SIZE",
    "PING",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for controller operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Clear all caches
        # Payload: None
        # Returns: None
        "CLEAR": ProtocolDefinition(
            payload_classes=[],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Get chunk size configuration
        # Payload: None
        # Returns: int - The chunk size value
        "GET_CHUNK_SIZE": ProtocolDefinition(
            payload_classes=[],
            response_class=int,
            handler_type=HandlerType.SYNC,
        ),
        # Ping
        # Payload: [instance_id] -- the sender's worker instance ID, or None
        #   for an untracked prober (the scheduler adapter).
        # Returns: int - PING_HEALTHY (1) tracked/prober, PING_UNTRACKED (2)
        #   healthy transport but the named instance is no longer registered
        #   (reaped) and must re-register. Both truthy: a client that bool()s
        #   the reply still reads "healthy".
        # BLOCKING on the NORMAL pool: keeps PING off the MQ main loop (where a
        # slow SYNC REGISTER_KV_CACHE would stall it) and lets pool saturation
        # surface as worker degraded mode.
        "PING": ProtocolDefinition(
            payload_classes=[int | None],
            response_class=int,
            handler_type=HandlerType.BLOCKING,
        ),
    }
