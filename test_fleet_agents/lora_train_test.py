"""Data-wise self-test for the lora-train agent.

1. The agent module imports and is registered in fleet_agents.HANDLERS.
2. Its command builder emits the right CLI.
3. DATA-WISE: the engine's --dry-run is GREEN on the REAL pilkwang weights + REAL external data
   (warm-resume clean, LoRA injected with small trainable %, one train step finite, one competition eval).
"""
import subprocess
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent


def main() -> None:
    sys.path.insert(0, str(COMP))
    import fleet_agents
    assert "lora-train" in fleet_agents.HANDLERS, "lora-train not registered in HANDLERS"

    from fleet_agents import lora_train
    cmd = lora_train._AGENT._cmd({"div_weight": 3.0, "r": 8, "use_dora": True}, dry=True)
    assert "--dry-run" in cmd and "--div-weight" in cmd and "3.0" in cmd and "--use-dora" in cmd, cmd

    py = COMP / "research" / "cellmot_venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    r = subprocess.run([py, str(COMP / "research/lora_finetune/train_lora.py"), "--dry-run", "--bf16", "0"],
                       cwd=str(COMP), capture_output=True, text=True, timeout=580)
    out = (r.stdout or "") + (r.stderr or "")
    assert "DRY-RUN GREEN" in out, f"engine dry-run not GREEN:\n{out[-800:]}"
    # sanity: warm-resume was clean and trainable % is small
    assert "missing=0 unexpected=0" in out, "warm-resume not clean"
    assert "trainable-outside-(lora/detect_head): 0" in out, "freeze not confirmed"
    print("lora_train_test PASS")


if __name__ == "__main__":
    main()
