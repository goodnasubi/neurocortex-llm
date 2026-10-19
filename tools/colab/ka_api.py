import sys
from colab_cli.auth import get_credentials
from colab_cli.utils import get_status_code
from colab_cli.common import state
sys.argv=['colab','--auth','oauth2','sessions']
s = state.get_session('step44')
creds = get_credentials(state.client_oauth_config, provider=state.auth_provider)
url = f"{state.client.colab_domain}/tun/m/{s.endpoint}/keep-alive/"
r = creds.request("GET", url, params={"authuser":"0"}, headers={"X-Colab-Tunnel":"Google"})
print(get_status_code(r), r.text[:200])
