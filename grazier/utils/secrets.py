import logging
import os
import platform
from typing import Optional

import keyring
from keyrings.cryptfile.cryptfile import CryptFileKeyring
from rich.prompt import Confirm

_SECRET_CACHE = {}


def get_secret(secret_id: str, default: Optional[str] = None) -> Optional[str]:
    secret = os.getenv(secret_id, _SECRET_CACHE.get(secret_id, None))
    if secret is not None:
        return secret

    logging.info("Secret %s not found in environment, checking keyring.", secret_id)

    # Check to see if it's in the keyring. If we're on linux, use a cryptfile keyring.
    if platform.system() == "Linux":
        logging.info("System keyring is not supported on linux, using an encrypted file instead.")
        if os.getenv("KEYRING_CRYPTFILE_PASSWORD") is None:
            logging.info(
                "Use KEYRING_CRYPTFILE_PASSWORD to set the password for the keyring, or enter it at the prompt."
            )
        kr = CryptFileKeyring()
        if os.getenv("KEYRING_CRYPTFILE_PASSWORD") is not None:
            kr.keyring_key = os.getenv("KEYRING_CRYPTFILE_PASSWORD")
        keyring.set_keyring(kr)

    logging.info("Keyring backend: %s", keyring.get_keyring())

    try:
        secret = keyring.get_password("grazier", secret_id)
        if secret is not None:
            _SECRET_CACHE[secret_id] = secret
            return secret
    except keyring.errors.KeyringLocked:
        logging.warn("Failed to get secret from keyring. Keyring is locked.")
    except keyring.errors.InitError:
        logging.warn("Failed to get secret from keyring. Keyring is not initialized.")
    except ValueError:
        logging.warn(
            "Invalid keyring password/value -- To reset the password, use `grazier reset-credential-store` -- returning default: %s",
            default,
        )

    logging.info("Secret %s not found in keyring, returning default: %s", secret_id, default)

    return default


def set_secret(secret_id: str, secret: str) -> None:
    if platform.system() == "Linux":
        logging.info("System keyring is not supported on linux, using an encrypted file instead.")
        if os.getenv("KEYRING_CRYPTFILE_PASSWORD") is None:
            logging.info(
                "Use KEYRING_CRYPTFILE_PASSWORD to set the password for the keyring, or enter it at the prompt."
            )
        kr = CryptFileKeyring()
        if os.getenv("KEYRING_CRYPTFILE_PASSWORD") is not None:
            kr.keyring_key = os.getenv("KEYRING_CRYPTFILE_PASSWORD")
        keyring.set_keyring(kr)

    try:
        keyring.set_password("grazier", secret_id, secret)
    except keyring.errors.KeyringLocked:
        logging.warn("Failed to set secret in keyring. Keyring is locked.")
    except keyring.errors.InitError:
        logging.warn("Failed to set secret in keyring. Keyring is not initialized.")


def reset_credential_store():
    if platform.system() == "Linux":
        logging.info("System keyring is not supported on linux, using an encrypted file instead.")
        if os.getenv("KEYRING_CRYPTFILE_PASSWORD") is None:
            logging.info(
                "Use KEYRING_CRYPTFILE_PASSWORD to set the password for the keyring, or enter it at the prompt."
            )
        kr = CryptFileKeyring()
        if os.getenv("KEYRING_CRYPTFILE_PASSWORD") is not None:
            kr.keyring_key = os.getenv("KEYRING_CRYPTFILE_PASSWORD")
        keyring.set_keyring(kr)

    if isinstance(keyring.get_keyring(), CryptFileKeyring):
        if Confirm.ask("Are you sure you want to reset the keyring?", default=False):
            os.remove(keyring.get_keyring().file_path)
            logging.info("Keyring reset.")
        else:
            logging.info("Keyring not reset.")
    else:
        logging.info("Keyring is not a cryptfile keyring, so we can't reset it.")
