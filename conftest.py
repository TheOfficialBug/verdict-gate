"""Suite-wide guard: no test may open a network connection (MASTER_RUNBOOK hard rule 7, IMPLEMENTATION_SPEC test 6).

For the whole session, connecting to or looking up any host other than the local machine raises NetworkBlocked.
Loopback and AF_UNIX stay allowed because the standard library uses them internally (asyncio's socketpair on
Windows). tests/test_no_network.py checks that the guard is active.
"""

import socket

import pytest


class NetworkBlocked(RuntimeError):
    pass


def _is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    return host in (None, "", "localhost", "::1") or (isinstance(host, str) and host.startswith("127."))


@pytest.fixture(autouse=True, scope="session")
def _block_network():
    real_connect, real_connect_ex, real_getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def guard(address):
        if isinstance(address, tuple) and not _is_local(address[0]):  # inet address; AF_UNIX paths are local
            raise NetworkBlocked(f"network access blocked in tests: {address!r}")

    def connect(sock, address):
        guard(address)
        return real_connect(sock, address)

    def connect_ex(sock, address):
        guard(address)
        return real_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            raise NetworkBlocked(f"network access blocked in tests: lookup of {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", connect)
        mp.setattr(socket.socket, "connect_ex", connect_ex)
        mp.setattr(socket, "getaddrinfo", getaddrinfo)
        yield
