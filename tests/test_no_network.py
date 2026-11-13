"""The root conftest.py blocks network access for the whole suite (tests must never touch the network)."""

import socket
import urllib.request

import pytest


def test_tcp_connect_is_blocked():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(RuntimeError, match="network access blocked"):
            s.connect(("93.184.216.34", 80))


def test_connect_ex_is_blocked():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(RuntimeError, match="network access blocked"):
            s.connect_ex(("1.1.1.1", 443))


def test_dns_lookup_is_blocked():
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.getaddrinfo("openrouter.ai", 443)


def test_http_request_is_blocked():
    with pytest.raises(Exception, match="network access blocked"):
        urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=1)


def test_loopback_still_works():
    a, b = socket.socketpair()  # stdlib internals (asyncio on Windows) rely on loopback
    with a, b:
        a.sendall(b"ok")
        assert b.recv(2) == b"ok"
