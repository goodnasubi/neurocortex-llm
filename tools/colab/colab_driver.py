"""Colab CLI で ステップ44.4（③: 4エポック）を回し続けるドライバ。この環境（コンテナ）側で動かす。"""
import json, os, subprocess, sys, time
from pathlib import Path

os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':' + os.environ['PATH']
S = 'step44'
VM = '/content/step44/results'
LOCAL = Path('/home/user/neurocortex-llm/results/step44_full_scale_rerun')
TMP = Path(__file__).parent / 'colab_tmp'
RUN = ['python', 'colab_step44_full_scale_experiment.py', '--phases', 'A', 'B', 'C', '--epochs', '4',
       '--batch-size', '8', '--grad-accum', '8', '--max-hours', '200',
       '--results-dir', VM, '--checkpoint-dir', VM + '/checkpoints', '--token-cache-dir', '/content/step44/token_cache',
       '--stop-file', '/content/step44/STOP']
MIN_BALANCE = 5.0
# Colab は長時間つなぎっぱなしにできないので、区切りで VM を捨てて張り直す。
# - 毎回: (phase, seed) の完了直後に止める（チェックポイント不要なので Drive 承認なしで再開できる）
# - 次のエポック末まで続けると接続の上限（12時間、余裕 1 時間）を超える見込みになったら、その前のエポック末で止める
#   （Drive 経由で再開。新 VM の Drive 承認が必要）
MAX_SESSION_H = float(os.environ.get('MAX_SESSION_H', 12))
MARGIN_H = 1.0
EPOCH_H_DEFAULT = 6.0  # 実測できないときのエポック所要時間（実測: B 約4.9時間、C 約5.7時間の見込み）
SESSION_T0 = Path(__file__).parent / 'session_started'
RECONNECT_NOW = Path(__file__).parent / 'RECONNECT_NOW'  # 置くと、次のチェックポイント保存直後に張り直す
RESUME_NOW = Path(__file__).parent / 'RESUME_NOW'  # 承認待ちで VM を止めたあと、置くと VM を作り直す
DRIVE = '/content/drive/MyDrive/neurocortex/step44'
# VM 上で 30 分ごとに結果とチェックポイントを Drive へ複製する（.part に書いてから rename。完了済み実行の削除も反映）
SYNC = r'''
import os, shutil, time
SRC, DST = %r, %r
def mirror(src, dst):
    os.makedirs(dst, exist_ok=True)
    names = {n for n in os.listdir(src) if n.endswith(('.json', '.pt'))}
    for n in names:
        a, b = os.path.join(src, n), os.path.join(dst, n)
        if not os.path.exists(b) or os.path.getmtime(a) > os.path.getmtime(b):
            shutil.copyfile(a, b + '.part'); os.replace(b + '.part', b)
    for n in os.listdir(dst):
        if n.endswith('.pt') and n not in names:
            os.remove(os.path.join(dst, n))
def progress():
    # 進捗（Drive の progress.txt）と学習ログの末尾（run_tail.log）を書き出す。スマホの Drive アプリで見られるように
    import glob, re
    raw = open('/content/step44/run.log', errors='replace').read() if os.path.exists('/content/step44/run.log') else ''
    lines = [l for l in raw.replace('\r', '\n').splitlines() if l.strip()]
    info = [l for l in lines if ' - INFO - ' in l and 'httpx' not in l]
    bars = [l for l in lines if l.startswith('Train Phase')]
    pid = open('/content/step44/pid').read().strip() if os.path.exists('/content/step44/pid') else ''
    alive = bool(pid) and os.path.exists('/proc/' + pid) and open('/proc/' + pid + '/stat').read().split()[2] != 'Z'
    run = [l for l in info if re.search(r'Phase [ABC] \| Seed \d', l)]
    ep = [l for l in info if re.search(r'Epoch \d+/\d+$', l)]
    ppl = [l for l in info if 'Val PPL' in l]
    out = ['更新: ' + time.strftime('%%Y-%%m-%%d %%H:%%M:%%S UTC'),
           '学習プロセス: ' + ('動作中' if alive else '停止'),
           '完了: ' + ', '.join(sorted(os.path.basename(f) for f in glob.glob(SRC + '/phase*_seed*.json'))),
           '実行中: ' + (run[-1].split(' - INFO - ')[-1] if run else '-'),
           'エポック: ' + (ep[-1].split(' - INFO - ')[-1] if ep else '-'),
           '進捗: ' + (bars[-1][:160] if bars else '-'),
           '', 'この実行の Val PPL:'] + [l.split(' - INFO - ')[-1] for l in ppl[-4:]]
    with open(DST + '/progress.txt.part', 'w') as f:
        f.write('\n'.join(out) + '\n')
    os.replace(DST + '/progress.txt.part', DST + '/progress.txt')
    with open(DST + '/run_tail.log.part', 'w') as f:
        f.write('\n'.join([l for l in lines if 'httpx' not in l and not l.startswith('Train Phase')][-150:] + bars[-1:]) + '\n')
    os.replace(DST + '/run_tail.log.part', DST + '/run_tail.log')
n = 0
while True:
    if os.path.ismount('/content/drive'):
        try:
            os.makedirs(DST, exist_ok=True); progress()
        except Exception as e:
            print('progress error', e, flush=True)
    if n %% 6 == 0 and os.path.ismount('/content/drive'):
        try:
            mirror(SRC, DST); mirror(SRC + '/checkpoints', DST + '/checkpoints')
            if os.path.isdir('/content/step44/token_cache'):
                mirror('/content/step44/token_cache', DST + '/token_cache')  # 新しい VM でのトークン化（約20分）を省く
            print(time.strftime('%%H:%%M:%%S'), 'synced', flush=True)
        except Exception as e:
            print('sync error', e, flush=True)
    n += 1
    time.sleep(300)  # 進捗は5分ごと、結果とチェックポイントは30分ごと
''' % (VM, DRIVE)
LOCAL.mkdir(parents=True, exist_ok=True); TMP.mkdir(exist_ok=True)


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def colab(*args, timeout=600):
    try:
        r = subprocess.run(['colab', '--auth', 'oauth2', *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, 'TIMEOUT'  # VM の応答待ちで固まっても、ドライバは落とさず次の周期で再試行する
    return r.returncode, r.stdout + r.stderr


def vm_exec(code, timeout=600):
    f = TMP / 'cell.py'; f.write_text(code)
    return colab('exec', '-s', S, '-f', str(f), timeout=timeout)


def balance():
    _, out = colab('usage', timeout=120)
    for line in out.splitlines():
        if 'balance' in line:
            return float(line.split(':')[1].split()[0])
    return None


def done_count():
    return len(list(LOCAL.glob('phase*_seed*.json')))


def session_alive():
    for _ in range(3):  # 一時的な失敗で VM を作り直さないよう、失敗時は再試行する
        rc, out = colab('sessions', timeout=120)
        if rc == 0:
            return S in out
        time.sleep(60)
    return None


def drive_mounted():
    rc, out = vm_exec("import os; print('MOUNTED' if os.path.ismount('/content/drive') else 'UNMOUNTED')", timeout=300)
    return 'MOUNTED' in out and 'UNMOUNTED' not in out


def mount_drive():
    # 新しい VM では Drive の承認がやり直しになる。承認 URL は dm_auto.log に出る（ユーザーに渡す）
    cmd = [os.path.expanduser('~/.local/share/uv/tools/google-colab-cli/bin/python'), str(Path(__file__).parent / 'colab_dm.py'),
           '--auth', 'oauth2', 'drivemount', '-s', S]
    with open(Path(__file__).parent / 'dm_auto.log', 'w') as f:
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:  # 承認待ちが上限を超えて戻らないことがある（09-30 22:20〜22:50）。ドライバごと落とさない
            rc = -1
    if not drive_mounted():
        # 承認後の1回目が 'mount failed' で終わり、すぐやり直すと再承認なしで通ることがある（09-29 に2回）。短い待ちで1回だけ再試行
        with open(Path(__file__).parent / 'dm_retry.log', 'w') as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=900,
                                    env=dict(os.environ, DM_WAIT_SEC='120')).returncode
            except subprocess.TimeoutExpired:
                rc = -1
    log('drivemount', rc, 'mounted' if drive_mounted() else 'NOT MOUNTED（dm_auto.log の URL でユーザー承認が必要）')


def session_age_h():
    try:
        return (time.time() - float(SESSION_T0.read_text())) / 3600
    except (OSError, ValueError):
        return 0.0


def new_session():
    for i in range(6):  # T4 の割り当てが Colab 側で一時的に断られる（Service Unavailable）ことがあるので、間を置いて再試行
        rc, out = colab('new', '-s', S, '--gpu', 'T4', timeout=900)
        SESSION_T0.write_text(str(time.time()))
        log('new', rc, out.strip()[-200:])
        if rc == 0 or session_alive():
            break
        time.sleep(60 * (i + 1))
    # 初期設定（clone・pip）は VM 上の別プロセスで走らせ、結果ファイルを確認する。
    # exec の応答待ちのまま 900 秒で切れ、repo がないまま学習開始に失敗したため（10-04 11:01）
    code = '''
import subprocess, os
R = '/content/neurocortex-llm'
os.makedirs('%s/checkpoints', exist_ok=True)
if not os.path.exists('/content/setup.started'):
    open('/content/setup.started', 'w').close()
    subprocess.Popen("(test -d %%s || git clone -q -b master-m7kte3 https://github.com/goodnasubi/neurocortex-llm.git %%s) && pip install -q datasets transformers && echo SETUP_OK > /content/setup.out || echo SETUP_NG > /content/setup.out" %% (R, R), shell=True, start_new_session=True)
print('setup ' + 'started')
''' % VM
    for _ in range(3):
        rc, out = vm_exec(code, timeout=180)
        log('setup start', rc, out.strip()[-120:])
        if 'setup started' in out:
            break
    for _ in range(40):
        time.sleep(30)
        rc, out = vm_exec("import os; f='/content/setup.out'; print(open(f).read().strip() if os.path.exists(f) else 'PEND' + 'ING')", timeout=120)
        if 'SETUP_OK' in out or 'SETUP_NG' in out:
            break
    log('setup', rc, out.strip()[-200:])


def start_session():
    new_session()
    mount_drive()
    start_session_restore_only()


def set_stop(mode):
    rc, out = vm_exec("open('/content/step44/STOP','w').write(%r); print('stop', %r)" % (mode, mode))
    log('stop-file', mode, rc)


def final_sync():
    # VM を捨てる前に、未完了の実行のチェックポイントを Drive へ確実に写す（サイズ一致で確認）
    code = SYNC.split('while True:')[0] + '''
import json
if not os.path.ismount('/content/drive'):
    print('SYNC_NG unmounted')
else:
    mirror(SRC, DST)
    os.makedirs(DST + '/checkpoints', exist_ok=True)
    for n in os.listdir(SRC + '/checkpoints'):
        if n.endswith('.pt'):
            shutil.copyfile(f'{SRC}/checkpoints/{n}', f'{DST}/checkpoints/{n}.part')
            os.replace(f'{DST}/checkpoints/{n}.part', f'{DST}/checkpoints/{n}')
    bad = [n for n in os.listdir(SRC + '/checkpoints') if n.endswith('.pt') and
           os.path.getsize(f'{SRC}/checkpoints/{n}') != os.path.getsize(f'{DST}/checkpoints/{n}')]
    print('SYNC_NG ' + json.dumps(bad) if bad else 'SYNC_OK')
'''
    # exec の応答待ちは約40秒で切れる（数GBの写しは終わらない）ので、VM 上で別プロセスとして走らせ、結果をファイルで確認する。
    # 応答待ち切れのエラー表示には実行コードの文字列が出るので、目印（RESULT:・SYNC_OK）はコード中に同じ文字列が現れない形で出す。
    R = '/content/step44/final_sync.out'
    rc, out = vm_exec('''
import subprocess, os
for f in [%r, %r + '.done']:
    if os.path.exists(f):
        os.remove(f)
open('/content/step44/final_sync.py', 'w').write(%r)
subprocess.Popen('python /content/step44/final_sync.py > %s 2>&1; touch %s.done', shell=True, start_new_session=True); print('final-sync started')
''' % (R, R, code, R, R))
    log('final-sync start', rc, out.strip()[-100:])
    for _ in range(120):
        time.sleep(30)
        rc, out = vm_exec("import os; print('RES' + 'ULT:' + open(%r).read() if os.path.exists(%r + '.done') else 'WAIT')" % (R, R))
        if 'RESULT:' in out:
            res = out.split('RESULT:', 1)[1]
            log('final-sync', res.strip()[-200:])
            return 'SYNC_OK' in [l.strip() for l in res.splitlines()]
    log('final-sync: 60分待っても終わらない'); return False


def kill_training():
    rc, out = vm_exec('''
import os, signal, time
pid = int(open('/content/step44/pid').read())
try:
    os.killpg(pid, signal.SIGTERM)
except ProcessLookupError:
    pass
for _ in range(60):
    if not os.path.exists(f'/proc/{pid}') or open(f'/proc/{pid}/stat').read().split()[2] == 'Z':
        break
    time.sleep(1)
print('killed', pid)
''', timeout=300)
    log('kill', rc, out.strip()[-100:])


def kill_after_ckpt():
    # 旧コード VM 用: 次のチェックポイント保存（2000 ステップごと、約10分）を待ってから止める
    killer = '''
import os, glob, signal, time
pid = int(open('/content/step44/pid').read())
cks = glob.glob('%s/checkpoints/*.pt')
m0 = max(os.path.getmtime(f) for f in cks) if cks else 0
for _ in range(1500):
    cks = glob.glob('%s/checkpoints/*.pt')
    if cks and max(os.path.getmtime(f) for f in cks) > m0 and not glob.glob('%s/checkpoints/*.tmp'):
        break
    time.sleep(1)
os.killpg(pid, signal.SIGTERM)
for _ in range(60):
    if not os.path.exists(f'/proc/{pid}') or open(f'/proc/{pid}/stat').read().split()[2] == 'Z':
        break
    time.sleep(1)
open('/content/step44/killed', 'w').write('killed after ckpt %%d %%s' %% (pid, time.strftime('%%H:%%M:%%S')))
''' % (VM, VM, VM)
    rc, out = vm_exec('''
import subprocess, os
if os.path.exists('/content/step44/killed'):
    os.remove('/content/step44/killed')
open('/content/step44/killer.py', 'w').write(%r)
subprocess.Popen(['python', '/content/step44/killer.py'], start_new_session=True); print('killer started')
''' % killer)
    log('kill', rc, out.strip()[-100:])
    for _ in range(60):
        time.sleep(30)
        rc, out = vm_exec("import os; print(open('/content/step44/killed').read() if os.path.exists('/content/step44/killed') else 'WAIT')")
        if 'killed after ckpt' in out:
            log('kill', out.strip()[-80:]); return True
    log('kill: 30分待っても止まらない'); return False


def wait_user(reason):
    """VM を作る前にユーザーの承認を得る（Drive 承認待ちで CU を浪費しないため）。
    RESUME_NOW があれば消して続行。なければ理由を記録して終了する（終了がセッションへの通知になる）。"""
    if RESUME_NOW.exists():
        RESUME_NOW.unlink()
        log('RESUME: ユーザーの承認を得たので VM を作成する')
        return
    log('WAIT_USER:', reason)
    sys.exit(3)


def wait_drive(tries=1):
    """Drive がマウントされるまで承認を待つ（1回約25分）。承認がなければ VM を止めて False を返す。"""
    while not drive_mounted():
        if session_alive() is False:
            log('VM が失われた（承認待ちの間の切断など）')
            return False
        if tries <= 0:
            colab('stop', '-s', S, timeout=300)
            log('APPROVAL_TIMEOUT: 承認がないため VM を止めた（CU 節約）')
            return False
        log('NEED_DRIVE_APPROVAL: 再開には Drive が必要。承認待ち（dm_auto.log の URL）')
        mount_drive(); tries -= 1
    return True


def flush_drive():
    # Drive の FUSE は書き込みを手元のキャッシュに受けてから非同期に送る。サイズ一致の確認はキャッシュを読むだけなので、
    # すぐ VM を止めると Drive 側は前の定期同期の版のまま残る（09-29 C0・10-02 B1 で、エポック末ではなく batch 56000 から再開）。
    # 停止前に flush_and_unmount で送り切るのを待つ（カーネル内の別スレッドで走らせ、印のファイルで確認）。
    M = '/content/step44/flush.done'
    code = """import threading, os
from google.colab import drive
if os.path.exists(%r): os.remove(%r)
def _f():
    try:
        drive.flush_and_unmount(); r = 'ok'
    except Exception as e:
        r = 'ng ' + repr(e)
    open(%r, 'w').write(r)
threading.Thread(target=_f, daemon=True).start(); print('flush ' + 'started')""" % (M, M, M)
    for _ in range(3):  # 起動の exec が届かずタイムアウトすることがある（10-03 13:21、スレッドは起動していなかった）。届くまで再送
        rc, out = vm_exec(code, timeout=180)
        log('flush start', rc, out.strip()[-120:])
        if 'flush started' in out:
            break
    for _ in range(60):
        time.sleep(30)
        rc, out = vm_exec("import os; print('FL' + 'AG:' + open(%r).read() if os.path.exists(%r) else 'WAIT')" % (M, M))
        if 'FLAG:' in out:
            log('flush', out.split('FLAG:', 1)[1].strip()[:200]); return
    log('flush: 30分待っても終わらない（そのまま止める）')


def reconnect(st):
    """区切りで止まった VM を捨て、新しい VM で続きを始める。"""
    backup(st)
    need_drive = bool(st['ckpt'])  # 未完了の実行のチェックポイントがあれば Drive 経由で引き継ぐ
    if need_drive and not final_sync():
        log('Drive へ退避できないため、張り直さず同じ VM で続ける')
        start_training(); return
    if need_drive:
        flush_drive()
    rc, out = colab('stop', '-s', S, timeout=300)
    log('reconnect: 旧 VM を停止', rc, f'age={session_age_h():.1f}h', 'need_drive', need_drive)
    if need_drive:
        return  # Drive 承認が必要な VM は、ユーザーの承認を得てから作る（メインループの VM なしの処理へ）
    new_session()
    start_session_restore_only()  # 完了済みの結果 JSON を戻す
    if True:
        start_training()  # 先に学習を始め、Drive（途中経過の退避用）はそのあとでマウントする
        mount_drive()
        if not drive_mounted():
            log('NEED_DRIVE_APPROVAL: 学習は進行中。Drive への退避のため承認が必要（dm_auto.log の URL）')


def start_session_restore_only():
    # Drive なしの VM では Drive からの復元は空振りなので飛ばす（10-02 11:51、この exec が接続切れで上限3600秒まで戻らなかった）
    if not drive_mounted():
        rc, out = 0, 'skip（Drive 未マウント）'
    else:
        rc, out = vm_exec('''
import os, shutil
D, V = %r, %r
if os.path.ismount('/content/drive'):
    for sub in ['', '/checkpoints']:
        if os.path.isdir(D + sub):
            os.makedirs(V + sub, exist_ok=True)
            for n in os.listdir(D + sub):
                if n.endswith(('.json', '.pt')):
                    shutil.copy2(f'{D}{sub}/{n}', f'{V}{sub}/{n}')
    if os.path.isdir(D + '/token_cache'):
        os.makedirs('/content/step44/token_cache', exist_ok=True)
        for n in os.listdir(D + '/token_cache'):
            shutil.copy2(f'{D}/token_cache/{n}', f'/content/step44/token_cache/{n}')
    print('restored from drive', os.listdir(V), os.listdir(V + '/checkpoints'))
else:
    print('drive not mounted')
''' % (DRIVE, VM), timeout=3600)
    log('restore', rc, out.strip()[-300:])
    # 完了済みの結果 JSON を exec 経由で戻す（contents API は 500 のため使えない）
    for f in LOCAL.glob('phase*_seed*.json'):
        rc, out = vm_exec('open(%r, "w").write(%r); print("restored")' % (VM + '/' + f.name, f.read_text()))
        log('restore', f.name, rc)


def start_training():
    rc, out = vm_exec('''
import subprocess, os
open('/content/step44/STOP','w').write('run')
open('/content/step44/sync.py','w').write(%r)
if not (os.path.exists('/content/step44/sync.pid') and os.path.exists('/proc/' + open('/content/step44/sync.pid').read()) and open('/proc/' + open('/content/step44/sync.pid').read() + '/stat').read().split()[2] != 'Z'):
    q = subprocess.Popen(['python', '/content/step44/sync.py'], stdout=open('/content/step44/sync.log','a'), stderr=subprocess.STDOUT, start_new_session=True)
    open('/content/step44/sync.pid','w').write(str(q.pid))
p = subprocess.Popen(%r, cwd='/content/neurocortex-llm', stdout=open('/content/step44/run.log','a'), stderr=subprocess.STDOUT, start_new_session=True)
open('/content/step44/pid','w').write(str(p.pid)); print('started', p.pid)
''' % (SYNC, RUN))
    log('start', rc, out.strip()[-200:])


def poll():
    rc, out = vm_exec('''
import os, glob, json
pid = int(open('/content/step44/pid').read()) if os.path.exists('/content/step44/pid') else None
alive = pid is not None and os.path.exists(f'/proc/{pid}') and open(f'/proc/{pid}/stat').read().split()[2] != 'Z'
raw = open('/content/step44/run.log').read().splitlines() if os.path.exists('/content/step44/run.log') else []
lines = [l for l in raw if ' - INFO - ' in l and 'httpx' not in l][-3:]
stop_support = alive and '--stop-file' in open(f'/proc/{pid}/cmdline').read()
import re, time
starts = [time.mktime(time.strptime(l[:19], '%%Y-%%m-%%d %%H:%%M:%%S')) for l in raw if re.search(r' - INFO - Epoch \d+/\d+$', l)]
epoch_h = (starts[-1] - starts[-2]) / 3600 if len(starts) >= 2 else None
stop = open('/content/step44/STOP').read().strip() if os.path.exists('/content/step44/STOP') else None
print(json.dumps({'alive': alive, 'stop_support': stop_support, 'epoch_h': epoch_h, 'stop': stop, 'stopped': any('STOPPED_AT_BREAKPOINT' in l for l in raw[-20:]), 'error': any(l.startswith('Traceback') for l in raw[-80:]), 'done': sorted(os.path.basename(f) for f in glob.glob('%s/phase*_seed*.json')),
                  'ckpt': sorted(os.path.basename(f) for f in glob.glob('%s/checkpoints/*.pt')), 'log': lines}))
''' % (VM, VM), timeout=300)
    for line in out.splitlines():
        if line.startswith('{'):
            return json.loads(line)
    log('poll failed', rc, out.strip()[-300:])
    return None


def backup(st):
    # 結果 JSON を exec の標準出力（base64）で取得する。チェックポイント（約3GB）は VM ローカルのみ
    for name in st['done'] + ['summary.json']:
        if name != 'summary.json' and (LOCAL / name).exists():
            continue
        rc, out = vm_exec('import base64,os; p=%r; print("B64:" + base64.b64encode(open(p,"rb").read()).decode() if os.path.exists(p) else "NONE")' % f'{VM}/{name}')
        for line in out.splitlines():
            if line.startswith('B64:'):
                import base64
                (LOCAL / name).write_bytes(base64.b64decode(line[4:]))
                log('fetched', name)


def ensure_sync():
    rc, out = vm_exec('''
import subprocess, os
open('/content/step44/sync.py','w').write(%r)
if not (os.path.exists('/content/step44/sync.pid') and os.path.exists('/proc/' + open('/content/step44/sync.pid').read()) and open('/proc/' + open('/content/step44/sync.pid').read() + '/stat').read().split()[2] != 'Z'):
    q = subprocess.Popen(['python', '/content/step44/sync.py'], stdout=open('/content/step44/sync.log','a'), stderr=subprocess.STDOUT, start_new_session=True)
    open('/content/step44/sync.pid','w').write(str(q.pid)); print('sync started', q.pid)
else:
    print('sync running')
''' % SYNC)
    log('sync', rc, out.strip()[-100:])


last_backup = 0
drive_retry = 0
poll_fail = 0
if session_alive():
    ensure_sync()
    if not SESSION_T0.exists():  # 現在の VM の作成時刻（driver.log の new）
        SESSION_T0.write_text(str(time.mktime(time.strptime('2026-09-27 15:33:27', '%Y-%m-%d %H:%M:%S'))))
while done_count() < 9:
    b = balance()
    log('balance', b, 'done', done_count())
    if b is not None and b < 20:
        log('LOW_BALANCE: 残高が 20 CU を下回った。ユーザーに 30 CU の追加を依頼する')
    if b is not None and b < MIN_BALANCE:
        log('残高が下限を下回ったため停止'); colab('stop', '-s', S, timeout=300); break
    if RECONNECT_NOW.exists() and session_alive():
        RECONNECT_NOW.unlink()
        log('RECONNECT_NOW: チェックポイント保存直後に止めて張り直す')
        st = poll() if kill_after_ckpt() else None
        if st and not st['alive']:
            reconnect(st); time.sleep(600); continue
    alive_now = session_alive()
    if alive_now is None:
        log('sessions の取得に失敗。次の周期で再試行'); time.sleep(300); continue
    if not alive_now:
        wait_user('VM がない（エポック末で停止・承認待ちで停止・喪失のいずれか）。Drive 承認の準備ができたら VM を作成する')
        start_session()
        if not drive_mounted():
            # Drive から復元できないまま学習を始めると、実行中の (phase, seed) を最初からやり直してしまう
            log('NEED_DRIVE_APPROVAL: Drive 未マウントのため学習を開始しない。承認後に再試行する')
            if not wait_drive(tries=0):  # start_session で1回（約25分）待ち済み
                continue
            start_session_restore_only()
        start_training(); last_backup = time.time(); time.sleep(600); continue
    st = poll()
    if st is None:
        poll_fail += 1
        if poll_fail >= 2:
            # カーネルが固まると exec が全部タイムアウトする。学習は別セッションのプロセスなので、再起動しても止まらない
            rc, out = colab('restart-kernel', '-s', S, timeout=300)
            log('restart-kernel', rc, out.strip()[-100:]); poll_fail = 0
        time.sleep(300); continue
    poll_fail = 0
    log('status', json.dumps(st, ensure_ascii=False)[-400:])
    if not st['alive']:
        backup(st)
        if st['error']:
            log('学習プロセスがエラーで終了したため停止（run.log を確認）')
            vm_exec("print(open('/content/step44/run.log').read()[-3000:])")
            colab('stop', '-s', S, timeout=300); break
        if len(st['done']) >= 9:
            break
        if st['stopped']:
            reconnect(st); time.sleep(600); continue
        # 想定外の正常終了: 同じセッションで再開する（チェックポイントは VM ローカルにある）
        start_training(); time.sleep(600); continue
    if len(st['done']) > done_count():
        backup(st)
        if not st['stop_support']:
            # 旧コードの VM は止まらないので、次の実行が始まった直後に止めて張り直す（失うのは最大15分）
            kill_training(); st = poll() or st; st['ckpt'] = []
            reconnect(st); time.sleep(600); continue
    epoch_h = max(st.get('epoch_h') or 0, 4.0) if st.get('epoch_h') else EPOCH_H_DEFAULT
    if st['stop_support'] and st['stop'] != 'epoch' and session_age_h() + epoch_h + MARGIN_H >= MAX_SESSION_H:
        log(f'次のエポック末で止める（稼働 {session_age_h():.1f}h + エポック {epoch_h:.1f}h）')
        set_stop('epoch')
    if st['stop_support'] and time.time() - drive_retry > 3600 and not drive_mounted():
        drive_retry = time.time(); mount_drive()
    time.sleep(900)

log('finished', done_count())
colab('stop', '-s', S, timeout=300)
