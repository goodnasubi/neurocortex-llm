"""ステップ44.4 フル規模実験を、Colab のブラウザ画面から手動で再開するスクリプト。

Claude のクラウドセッションが使えないときの手動運用用。Colab の新しいノートブック（GPU: T4）で、
次の2セルを実行する（1セル目で Drive の承認画面が出る）。

    from google.colab import drive; drive.mount('/content/drive')

    !git clone -q -b master-m7kte3 https://github.com/goodnasubi/neurocortex-llm.git /content/neurocortex-llm || git -C /content/neurocortex-llm pull -q
    !pip -q install datasets transformers
    !python /content/neurocortex-llm/tools/colab/step44_resume.py

やること:
- Drive（MyDrive/neurocortex/step44）から、完了済みの結果・チェックポイント・トークン化済みデータを VM に戻す。
- 学習を続きから再開する（完了済みの (phase, seed) は飛ばす）。
- 30分ごとと終了時に、結果とチェックポイントを Drive へ写す。
- Colab の接続は最長12時間。エポックは約5〜6時間かかるので、開始から --hours（既定5時間）を過ぎたら、
  次のエポック末で止める（12時間までに終わる見込み。実質1接続あたり約2エポック）。
"""
import argparse
import os
import shutil
import subprocess
import threading
import time

DRIVE = '/content/drive/MyDrive/neurocortex/step44'
WORK = '/content/step44'
RESULTS = WORK + '/results'
CKPT = RESULTS + '/checkpoints'
TOKENS = WORK + '/token_cache'
STOP = WORK + '/STOP'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = ['python', 'colab_step44_full_scale_experiment.py', '--phases', 'A', 'B', 'C', '--epochs', '4',
       '--batch-size', '8', '--grad-accum', '8', '--max-hours', '200',
       '--results-dir', RESULTS, '--checkpoint-dir', CKPT, '--token-cache-dir', TOKENS, '--stop-file', STOP]


def copy_dir(src, dst, force=False):
    """src の .json / .pt を dst へ写す（force でなければ新しいものだけ）。.part に書いてから置き換える。"""
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for n in os.listdir(src):
        if not n.endswith(('.json', '.pt')):
            continue
        a, b = os.path.join(src, n), os.path.join(dst, n)
        if force or not os.path.exists(b) or os.path.getmtime(a) > os.path.getmtime(b):
            shutil.copyfile(a, b + '.part')
            os.replace(b + '.part', b)


def save_to_drive(force=False):
    copy_dir(RESULTS, DRIVE, force)
    copy_dir(CKPT, DRIVE + '/checkpoints', force)
    copy_dir(TOKENS, DRIVE + '/token_cache')
    # 完了した実行のチェックポイントは学習スクリプトが消すので、Drive 側も消す
    if os.path.isdir(DRIVE + '/checkpoints'):
        for n in os.listdir(DRIVE + '/checkpoints'):
            if n.endswith('.pt') and not os.path.exists(os.path.join(CKPT, n)):
                os.remove(os.path.join(DRIVE + '/checkpoints', n))
    print(time.strftime('%H:%M:%S'), 'saved to Drive', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=5.0,
                    help='この時間を過ぎたら、次のエポック末で止める（12時間 − 余裕1時間 − エポック約6時間）')
    args = ap.parse_args()
    if not os.path.ismount('/content/drive'):
        raise SystemExit('先に drive.mount("/content/drive") を実行してください')

    copy_dir(DRIVE, RESULTS, force=True)
    copy_dir(DRIVE + '/checkpoints', CKPT, force=True)
    copy_dir(DRIVE + '/token_cache', TOKENS, force=True)
    print('restored:', sorted(os.listdir(RESULTS)), sorted(os.listdir(CKPT)), flush=True)
    if os.path.exists(STOP):
        os.remove(STOP)

    t0 = time.time()
    p = subprocess.Popen(RUN, cwd=REPO)
    stop = threading.Event()

    def background():
        while not stop.wait(1800):
            if time.time() - t0 > args.hours * 3600 and not os.path.exists(STOP):
                open(STOP, 'w').write('epoch')
                print('次のエポック末で止めます', flush=True)
            try:
                save_to_drive()
            except Exception as e:  # 同期の失敗で学習は止めない
                print('save error', e, flush=True)

    threading.Thread(target=background, daemon=True).start()
    rc = p.wait()
    stop.set()
    save_to_drive(force=True)
    print('終了コード', rc, '— 続きは、新しい接続でもう一度このスクリプトを実行してください', flush=True)


if __name__ == '__main__':
    main()
