"""Compatibility patch for SafeChain 2.0.3 Azure authentication."""

from openai.lib import azure as _azure_lib

_original_resolve_azure_auth = _azure_lib._resolve_azure_auth
_patched = False


def apply() -> None:
    global _patched
    if _patched:
        return

    def _resolve_azure_auth(api_key, azure_ad_token, azure_ad_token_provider):
        if api_key is not None and azure_ad_token is not None and api_key == azure_ad_token:
            api_key = None
        return _original_resolve_azure_auth(api_key, azure_ad_token, azure_ad_token_provider)

    _azure_lib._resolve_azure_auth = _resolve_azure_auth
    _patched = True


apply()