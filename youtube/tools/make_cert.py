"""Self-signed certificate for the hub, so the microphone works over the network.

Browsers only expose getUserMedia on a secure origin. `http://gpu:9090` is not one, so the
recording pages are dead over the LAN. HTTPS fixes that — a self-signed cert is fine here because
the browser treats the origin as secure once you accept the exception, and this only ever serves
one person on a private network.

    python tools/make_cert.py            # write common/certs/hub.{crt,key}

Covers every name/address the box answers to (hostname, localhost, LAN IP, Tailscale IP), so the
same cert works however you reach it.
"""
import datetime
import ipaddress
import socket
import subprocess
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

OUT = Path(__file__).resolve().parent.parent / "common" / "certs"


def local_addresses():
    names = {"localhost", socket.gethostname()}
    ips = {"127.0.0.1"}
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout
        for tok in out.split():
            try:
                ip = ipaddress.ip_address(tok)
            except ValueError:
                continue
            if not ip.is_link_local:
                ips.add(str(ip))
    except Exception:                                              # noqa: BLE001
        pass
    return sorted(names), sorted(ips)


def make(days=825):
    OUT.mkdir(parents=True, exist_ok=True)
    names, ips = local_addresses()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, names[-1] if names else "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "shorts hub (self-signed)"),
    ])
    san = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))

    (OUT / "hub.key").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    (OUT / "hub.key").chmod(0o600)
    (OUT / "hub.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"wrote {OUT}/hub.crt and hub.key")
    print(f"  names: {', '.join(names)}")
    print(f"  ips  : {', '.join(ips)}")
    print(f"  valid: {days} days")
    return OUT / "hub.crt", OUT / "hub.key"


if __name__ == "__main__":
    make()
