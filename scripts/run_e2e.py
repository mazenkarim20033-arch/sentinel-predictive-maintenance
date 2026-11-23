"""Full-stack check: starts the real API, then runs the frontend's live tests against it (dataset -> model -> API -> UI)."""
import os
import subprocess
import sys

from serve_helper import ROOT, running_server


def main() -> int:
    with running_server() as (base, _, log, seconds):
        print(f"API healthy at {base} after {seconds:.1f}s", flush=True)
        env = {**os.environ, "VITE_API_BASE": base}
        code = subprocess.run(["npm.cmd" if os.name == "nt" else "npm", "test", "--", "src/test/live.test.tsx", "src/test/snapshot.test.ts"], cwd=ROOT / "web", env=env).returncode
        bad = "Traceback" in log.read_text()
        print("server log contains a traceback" if bad else "server log clean", flush=True)
        return code or int(bad)


if __name__ == "__main__":
    sys.exit(main())
