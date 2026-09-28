"""colab drivemount のラッパー: /dev/tty で Enter を待つ代わりに、承認完了（dryrun success）をポーリングする。"""
import json, os, sys, time, io
from colab_cli.commands import automation
from colab_cli.auth import get_credentials
from colab_cli.utils import get_status_code

SESSION = sys.argv[sys.argv.index('-s') + 1]

class _Poll(io.StringIO):
    def readline(self, *a):
        from colab_cli.common import state
        s = state.get_session(SESSION)
        url = f"{state.client.colab_domain}/tun/m/credentials-propagation/{s.endpoint}"
        params = {"authuser": "0", "authtype": "dfs_ephemeral", "version": "2", "dryrun": "true", "propagate": "true", "record": "false"}
        deadline = time.time() + float(os.environ.get("DM_WAIT_SEC", "1500"))
        while time.time() < deadline:
            time.sleep(30)
            creds = get_credentials(state.client_oauth_config, provider=state.auth_provider)
            r = creds.request("GET", url, params=params)
            tok = json.loads(r.text.split("\n", 1)[-1]).get("token") if get_status_code(r) == 200 else None
            p2 = dict(params, dryrun="false")
            r = creds.request("POST", url, params=p2, headers={"x-goog-colab-token": tok}, files={"file_id": (None, "empty.ipynb")})
            print("[wrapper] poll", get_status_code(r), r.text[-160:].replace("\n", " "), flush=True)
            if get_status_code(r) == 200 and json.loads(r.text.split("\n", 1)[-1]).get("success"):
                print("\n[wrapper] approval detected", flush=True)
                return "\n"
        print("\n[wrapper] approval timeout", flush=True)
        return "\n"

_real_open = open
automation.open = lambda f, *a, **k: _Poll() if f == "/dev/tty" else _real_open(f, *a, **k)
automation.INTERACTIVE_AUTOMATION_TIMEOUT_SEC = float(os.environ.get("DM_WAIT_SEC", "1500")) + 300

from colab_cli.cli import main
sys.argv = ['colab'] + sys.argv[1:]
sys.exit(main())
