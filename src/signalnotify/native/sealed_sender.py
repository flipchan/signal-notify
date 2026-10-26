"""Sealed-sender (Envelope type 6) unwrap — Sealed Sender **v1**.

``signalnotify`` already implements X3DH/PQXDH, the Double Ratchet and
Note-to-Self, but the receive path raises ``NotImplementedError`` for
sealed-sender envelopes, so messages from *other* accounts are quarantined
instead of delivered.

This implements libsignal's [Sealed Sender v1] single-recipient KEM
(``rust/protocol/src/sealed_sender.rs``) closely enough to unwrap a real
production envelope:

    version_byte (0x11) || UnidentifiedSenderMessage protobuf

1. ``EphemeralKeys`` = HKDF-SHA256(salt = b"UnidentifiedDelivery" ||
   ourIdentityPub || ephPub, ikm = ECDH(ourIdentityPriv, ephPub)) → 96 bytes
   (chain_key, cipher_key, mac_key).
2. ``encryptedStatic`` decrypts (AES-256-CTR + 10-byte HMAC-SHA256) to the
   sender's 33-byte ``0x05``-prefixed identity public key.
3. ``StaticKeys`` = HKDF-SHA256(salt = chain_key || encryptedStatic,
   ikm = ECDH(ourIdentityPriv, senderIdentityPub)) → discard 32, keep
   (cipher_key, mac_key).
4. ``encryptedMessage`` decrypts to an ``UnidentifiedSenderMessageContent``
   protobuf: ``{type, senderCertificate, content, contentHint, groupId}``.
5. The embedded ``SenderCertificate`` is validated: its ``signer`` is a
   *reference id* into Signal's hardcoded server certificates (2 = staging,
   3 = production); the certificate signature must verify under that server
   certificate's key, and its identity key must equal the key recovered in (2).

The returned ``content`` is then a normal ``version_byte || PreKeySignalMessage``
or ``version_byte || SignalMessage`` that the existing
:func:`~signalnotify.native.ratchet.accept_prekey` /
:func:`~signalnotify.native.ratchet.ratchet_decrypt` paths consume.

Only **v1** (version byte ``0x11``) is implemented. Sealed Sender v2
(``0x22`` / ``0x23``, the flat ``C || AT || E.pub || ciphertext`` format) is
**not** handled yet and raises :class:`SealedSenderError` (the envelope is
quarantined as before).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .crypto import encode_point, inv, p, recover_x
from .ratchet import _dh, _hkdf
from .registration import decode_proto

log = logging.getLogger(__name__)

SEALED_SENDER_V1 = 0x11
SALT_PREFIX = b"UnidentifiedDelivery"

# Signal's hardcoded server certificates (libsignal sealed_sender.rs
# KNOWN_SERVER_CERTIFICATES). A SenderCertificate that *references* one of these
# ids is anchored to the embedded server certificate — no server-cert signature
# check is needed at runtime (libsignal treats these as trust roots).
# The value is the serialized ServerCertificate protobuf.
KNOWN_SERVER_CERTS = {
    2: bytes.fromhex(
        "0a25080212210539450d63ebd0752c0fd4038b9d07a916f5e174b756d409b5ca79f4c97400631e"
        "124064c5a38b1e927497d3d4786b101a623ab34a7da3954fae126b04dba9d7a3604ed88cdc8550"
        "950f0d4a9134ceb7e19b94139151d2c3d6e1c81e9d1128aafca806"
    ),
    3: bytes.fromhex(
        "0a250803122105bc9d1d290be964810dfa7e94856480a3f7060d004c9762c24c575a1522353a5a"
        "1240c11ec3c401eb0107ab38f8600e8720a63169e0e2eb8a3fae24f63099f85ea319c3c1c46d34"
        "54706ae2a679d1fee690a488adda98a2290b66c906bb60295ed781"
    ),
}


class SealedSenderError(Exception):
    """A sealed-sender envelope could not be unwrapped/validated."""


def _p33(pub: bytes) -> bytes:
    return pub if len(pub) == 33 else b"\x05" + pub


def _aes_ctr_decrypt(key: bytes, data: bytes) -> bytes:
    """AES-256-CTR with a zero nonce (libsignal ``Ctr32BE``, all-zero IV)."""
    c = Cipher(algorithms.AES(key), modes.CTR(b"\x00" * 16)).encryptor()
    return c.update(data) + c.finalize()


def _ctr_hmac_decrypt(ctext: bytes, cipher_key: bytes, mac_key: bytes) -> bytes:
    """libsignal ``aes256_ctr_hmacsha256_decrypt``: 10-byte truncated HMAC."""
    body, their_mac = ctext[:-10], ctext[-10:]
    if len(body) == 0:
        raise SealedSenderError("truncated ciphertext")
    our_mac = hmac.new(mac_key, body, hashlib.sha256).digest()[:10]
    if not hmac.compare_digest(our_mac, their_mac):
        raise SealedSenderError("MAC verification failed")
    return _aes_ctr_decrypt(cipher_key, body)


def _xed25519_verify(pub33: bytes, signature: bytes, message: bytes) -> bool:
    """Verify an XEdDSA (Signal) signature with an X25519 public key.

    libsignal tries both sign bits of the Montgomery→Edwards map; ``crypto.py``
    only exposes the sign-0 conversion, so try both here.
    """
    raw = pub33[1:] if len(pub33) == 33 else pub33
    try:
        u = int.from_bytes(raw, "little")
        y = (u - 1) * inv(u + 1) % p
    except (ValueError, ZeroDivisionError):
        return False
    for sign in (0, 1):
        try:
            ed_pub = encode_point((recover_x(y, sign), y))
            Ed25519PublicKey.from_public_bytes(ed_pub).verify(signature, message)
            return True
        except Exception:
            continue
    return False


def _server_cert_key(server_cert_bytes: bytes) -> bytes | None:
    """Extract the 33-byte signing key from a ServerCertificate."""
    try:
        sc = decode_proto(server_cert_bytes)
        return decode_proto(sc[1][0])[2][0]
    except Exception:
        return None


def parse_sender_certificate(cert_bytes: bytes) -> dict:
    """Parse a ``SenderCertificate`` → ``{uuid, device, identity_key, signer_id, expires}``.

    ``signer_id`` is set only for the *reference* form (``signer.id``); an
    embedded ``signer.certificate`` is not currently used by produced certs.
    """
    top = decode_proto(cert_bytes)
    body = decode_proto(top[1][0])
    out = {
        "uuid": None,
        "device": body.get(2, [1])[0],
        "identity_key": body.get(4, [None])[0],
        "signer_id": body.get(8, [None])[0],
        "expires_ms": int.from_bytes(body[3][0], "little") if 3 in body else None,
        "signature": top.get(2, [None])[0],
        "certificate": top[1][0],
    }
    raw_uuid = body.get(7, [None])[0]          # uuidBytes
    if raw_uuid and len(raw_uuid) == 16:
        out["uuid"] = str(uuid.UUID(bytes=raw_uuid))
    elif 6 in body:                            # uuidString
        out["uuid"] = body[6][0].decode("utf-8", "replace")
    return out


def validate_sender_certificate(cert_bytes: bytes, sender_identity: bytes) -> tuple[bool, dict]:
    """Verify the certificate chain and that it binds ``sender_identity``."""
    cert = parse_sender_certificate(cert_bytes)
    ok = cert["identity_key"] == sender_identity and cert["signer_id"] in KNOWN_SERVER_CERTS
    if ok:
        server_key = _server_cert_key(KNOWN_SERVER_CERTS[cert["signer_id"]])
        ok = bool(server_key) and _xed25519_verify(
            server_key, cert["signature"], cert["certificate"]
        )
    return ok, cert


def unwrap_v1(account_keys: dict, content: bytes) -> dict:
    """Unwrap a Sealed Sender v1 envelope ``content`` (Envelope field 8).

    Returns ``{inner, msg_type, sender_uuid, sender_device, sender_identity,
    cert_valid}`` where ``inner`` is the version-byte-prefixed inner ciphertext
    (PreKeySignalMessage for ``msg_type == 1``, SignalMessage for ``2``).
    Raises :class:`SealedSenderError` on any failure.
    """
    if not content:
        raise SealedSenderError("empty content")
    version = content[0]
    if version != SEALED_SENDER_V1:
        raise SealedSenderError(
            f"sealed-sender version {version:#x} not implemented (v1 only)"
        )
    usm = decode_proto(content[1:])
    try:
        e_pub = usm[1][0]
        enc_static = usm[2][0]
        enc_msg = usm[3][0]
    except (KeyError, IndexError):
        raise SealedSenderError("malformed UnidentifiedSenderMessage")

    our_priv = account_keys["identity_priv"]
    our_pub = _p33(account_keys["identity_pub"])

    eph = _hkdf(_dh(our_priv, e_pub), SALT_PREFIX + our_pub + _p33(e_pub), b"", 96)
    chain_key, cipher_key, mac_key = eph[:32], eph[32:64], eph[64:96]
    sender_identity = _ctr_hmac_decrypt(enc_static, cipher_key, mac_key)

    stat = _hkdf(_dh(our_priv, sender_identity), chain_key + enc_static, b"", 96)
    usmc_raw = _ctr_hmac_decrypt(enc_msg, stat[32:64], stat[64:96])
    usmc = decode_proto(usmc_raw)
    try:
        msg_type = usmc[1][0]
        cert_bytes = usmc[2][0]
        inner = usmc[3][0]
    except (KeyError, IndexError):
        raise SealedSenderError("malformed UnidentifiedSenderMessageContent")

    cert_valid, cert = validate_sender_certificate(cert_bytes, sender_identity)
    if not cert_valid:
        log.warning(
            "sealed-sender certificate did not validate (signer=%s uuid=%s)",
            cert.get("signer_id"), cert.get("uuid"),
        )
    return {
        "inner": inner,
        "msg_type": msg_type,
        "sender_uuid": cert.get("uuid"),
        "sender_device": cert.get("device") or 1,
        "sender_identity": sender_identity,
        "cert_valid": cert_valid,
    }
