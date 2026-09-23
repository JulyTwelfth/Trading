class InvalidLicenseKeyError(Exception):
    reason = "Invalid license key"


class ExpiredLicenseError(Exception):
    reason = "License expired"


class SessionAlreadyActiveError(Exception):
    reason = "Key already in use"


class LicenseServiceUnavailableError(Exception):
    reason = "License server unreachable"


class WalletPersistenceError(Exception):
    reason = "Failed to save wallet"


class WalletSlotsExhaustedError(Exception):
    reason = "No free wallet slot available"


class BlacklistPersistenceError(Exception):
    reason = "Failed to save blacklist"


class SecureOrderError(Exception):
    reason = "SecureClient order rejected"

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class WalletNotDeployedError(Exception):
    reason = "Deposit wallet not deployed on-chain"
