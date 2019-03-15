import ucp_py as ucp
import asyncio

ucp.init()


async def hello(ep, listener):
    await asyncio.sleep(0)
    ep.close()
    ucp.stop_listener(listener)


def test_multiple_listeners():
    lf1 = ucp.start_listener(hello, is_coroutine=True)
    lf2 = ucp.start_listener(hello, is_coroutine=True)
    assert lf1.port > 0
    assert lf2.port > 0
    assert lf1.port != lf2.port

    lf1.coroutine.close()
    lf2.coroutine.close()
    ucp.stop_listener(lf1)
    ucp.stop_listener(lf2)
