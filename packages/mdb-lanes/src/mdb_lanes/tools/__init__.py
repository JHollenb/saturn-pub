"""Installed mrun transport with no sibling checkout or credential discovery."""
from ..paths import data_root


def configure_workspace(*, load_credentials=True):
    # Credentials and model/cache locations belong to mrun's explicit config
    # and environment. MDB does not discover common/.env or modify sys.path.
    return data_root()


def scheduler_api():
    from ..local_data_plane import reject_external_operation
    reject_external_operation("mrun scheduler API access")
    try:
        from mrun.client.api import Api
    except ModuleNotFoundError as exc:
        raise RuntimeError("mrun is required for remote execution; install mdb-runtime[mrun]") from exc
    return Api()
