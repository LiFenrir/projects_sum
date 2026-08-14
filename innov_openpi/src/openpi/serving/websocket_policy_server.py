import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw_msg = msgpack_numpy.unpackb(await websocket.recv())

                # Unwrap protocol envelope: {"method": "infer", "obs": {...}, "rtc": {...}}
                if isinstance(raw_msg, dict) and "obs" in raw_msg:
                    obs = raw_msg["obs"]
                    rtc_kwargs = raw_msg.get("rtc", {})
                else:
                    # Backward compatibility: raw obs without envelope
                    obs = raw_msg
                    rtc_kwargs = {}

                if rtc_kwargs:
                    logger.debug(
                        "RTC kwargs received: prev_chunk_left_over=%s delay=%s horizon=%s",
                        type(rtc_kwargs.get("prev_chunk_left_over")).__name__
                        if rtc_kwargs.get("prev_chunk_left_over") is not None
                        else "None",
                        rtc_kwargs.get("inference_delay", "?"),
                        rtc_kwargs.get("execution_horizon", "?"),
                    )

                infer_time = time.monotonic()
                action = self._policy.infer(obs, **rtc_kwargs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # Log action chunk being sent (before serialization)
                send_actions = action.get("actions")
                if send_actions is not None and len(send_actions) > 0:
                    rtc_info = ""
                    if rtc_kwargs.get("prev_chunk_left_over") is not None:
                        plo = rtc_kwargs["prev_chunk_left_over"]
                        delay = rtc_kwargs.get("inference_delay", "?")
                        horizon = rtc_kwargs.get("execution_horizon", "?")
                        rtc_info = (
                            f" | RTC prev_shape=({plo.shape[-2]},{plo.shape[-1]})"
                            f" delay={delay} horizon={horizon}"
                        )
                    logger.info(
                        "send | time=%5.0fms | chunk=%d | action_dim=%d | first_action=[%s]%s",
                        infer_time * 1000,
                        send_actions.shape[-2],
                        send_actions.shape[-1],
                        np.array2string(send_actions[0], precision=4, suppress_small=True, max_line_width=200),
                        rtc_info,
                    )

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
