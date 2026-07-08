# SPDX-License-Identifier: Apache-2.0
"""Every module handler must conform to the protocol table.

The mq dispatcher validates each HandlerSpec's payload and return
annotations against the ProtocolDefinition registry when handlers are
registered — a mismatch is only logged at SERVER STARTUP and the handler
is rejected, so a drifted signature ships silently past unit tests that
call handlers directly (the PING bool->int change crash-looped a
production server before this suite existed).

This test walks every module's get_handlers() through the same
signature inspection the live dispatcher runs, so any handler/protocol
drift fails CI instead of the fleet.
"""

# Third Party
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.mq import MessageQueueServer
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as gpu_mod


def _inspect(module) -> list[str]:
    """Run the dispatcher's signature inspection over a module's handlers.

    Returns:
        The names of request types whose handler failed inspection.
    """
    failures = []
    for spec in module.get_handlers():
        ok = MessageQueueServer._inspect_handler_signature(
            MessageQueueServer.__new__(MessageQueueServer),
            spec.request_type,
            spec.handler,
        )
        if not ok:
            failures.append(str(spec.request_type))
    return failures


def test_management_handlers_conform() -> None:
    mgmt = ManagementModule(MagicMock(), liveness_targets=[])
    assert _inspect(mgmt) == []


def test_lmcache_driven_transfer_handlers_conform(monkeypatch) -> None:
    monkeypatch.setattr(gpu_mod, "DeviceHostFuncDispatcher", MagicMock())
    module = gpu_mod.LMCacheDrivenTransferModule(MagicMock(name="ctx"))
    assert _inspect(module) == []
