#!/usr/bin/env python3
"""
Colab Executor: CLI で Colab ノートブック実行を管理

使用法:
  python3 colab_executor.py submit --notebook <name>
  python3 colab_executor.py list
  python3 colab_executor.py status --run-id <id>
  python3 colab_executor.py setup
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any


class ColabExecutor:
    """Colab 実行管理"""

    def __init__(self, output_dir: str = "results/colab_runs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.output_dir / "executor_config.json"
        self.load_config()

    def load_config(self):
        """設定を読み込む"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {"runs": [], "last_run_id": 0}

    def save_config(self):
        """設定を保存"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def submit_run(self, notebook: str, timeout: int = 259200, params: Optional[Dict] = None) -> str:
        """Colab 実行をサブミット"""
        run_id = f"run_{self.config['last_run_id']:06d}"
        self.config['last_run_id'] += 1

        run_record = {
            "run_id": run_id,
            "notebook": notebook,
            "status": "submitted",
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "params": params or {},
            "timeout": timeout
        }

        self.config["runs"].append(run_record)
        self.save_config()

        print(f"\n✓ Run submitted: {run_id}")
        print(f"  Notebook: {notebook}")
        print(f"  Timeout: {timeout}s ({timeout//3600}h)")
        print(f"  Status: Check with 'python3 colab_executor.py status --run-id {run_id}'")

        return run_id

    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """実行ステータスを取得"""
        for run in self.config["runs"]:
            if run["run_id"] == run_id:
                return run
        return None

    def list_runs(self):
        """すべての実行一覧を表示"""
        print("\n" + "=" * 80)
        print("Colab Executor - Run History")
        print("=" * 80 + "\n")

        if not self.config["runs"]:
            print("No runs yet\n")
            return

        for run in reversed(self.config["runs"]):
            print(f"Run ID: {run['run_id']}")
            print(f"  Notebook: {run['notebook']}")
            print(f"  Status: {run['status']}")
            print(f"  Submitted: {run['submitted_at']}")
            print(f"  Timeout: {run['timeout']}s ({run['timeout']//3600}h)")
            if run.get('completed_at'):
                print(f"  Completed: {run['completed_at']}")
            if run.get('result_file'):
                print(f"  Result: {run['result_file']}")
            print()

    def mark_completed(self, run_id: str, result_file: str = None):
        """実行を完了済みにマーク"""
        for run in self.config["runs"]:
            if run["run_id"] == run_id:
                run["status"] = "completed"
                run["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC")
                if result_file:
                    run["result_file"] = result_file
                self.save_config()
                print(f"✓ Run marked as completed: {run_id}")
                return

    def show_setup(self):
        """手動セットアップ手順を表示"""
        print("\n" + "=" * 80)
        print("Colab Setup Instructions")
        print("=" * 80 + "\n")
        print("1️⃣  Open Google Colab:")
        print("   https://colab.research.google.com\n")
        print("2️⃣  Create new notebook or upload")
        print("   Name: 'neurocortex_step44_experiment'\n")
        print("3️⃣  Copy code from:")
        print("   colab_step44_full_scale_experiment.py\n")
        print("4️⃣  Run the notebook\n")
        print("5️⃣  When complete, download results\n")
        print("6️⃣  Track progress with:")
        print("   python3 colab_executor.py list\n")
        print("7️⃣  Mark as complete:")
        print("   python3 colab_executor.py mark-complete --run-id <run_id>\n")


def main():
    parser = argparse.ArgumentParser(
        description="Colab Executor - リモート Colab 実行管理ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 colab_executor.py submit --notebook neurocortex_step44
  python3 colab_executor.py list
  python3 colab_executor.py status --run-id run_000000
  python3 colab_executor.py setup
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # submit コマンド
    submit = subparsers.add_parser('submit', help='Submit Colab run')
    submit.add_argument('--notebook', required=True, help='Notebook name')
    submit.add_argument('--timeout', type=int, default=259200, help='Timeout (sec)')

    # list コマンド
    subparsers.add_parser('list', help='List all runs')

    # status コマンド
    status = subparsers.add_parser('status', help='Show run status')
    status.add_argument('--run-id', help='Run ID')

    # mark-complete コマンド
    mark = subparsers.add_parser('mark-complete', help='Mark run as complete')
    mark.add_argument('--run-id', required=True, help='Run ID')
    mark.add_argument('--result-file', help='Result file path')

    # setup コマンド
    subparsers.add_parser('setup', help='Show setup instructions')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    executor = ColabExecutor()

    if args.command == 'submit':
        executor.submit_run(notebook=args.notebook, timeout=args.timeout)

    elif args.command == 'list':
        executor.list_runs()

    elif args.command == 'status':
        if args.run_id:
            status = executor.get_run_status(args.run_id)
            if status:
                print("\n" + "=" * 80)
                print(f"Run Status: {args.run_id}")
                print("=" * 80 + "\n")
                print(json.dumps(status, indent=2) + "\n")
            else:
                print(f"❌ Run not found: {args.run_id}\n")
        else:
            executor.list_runs()

    elif args.command == 'mark-complete':
        executor.mark_completed(args.run_id, args.result_file)

    elif args.command == 'setup':
        executor.show_setup()


if __name__ == '__main__':
    main()
