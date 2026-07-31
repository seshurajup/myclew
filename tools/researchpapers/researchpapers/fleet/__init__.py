"""research_fleet — a small pool of DETERMINISTIC Python worker-agents.

They pull research questions off a shared board, do the work in pure Python (metric
decomposition, augmentation ablations, architecture probes), and post progress + findings
into the SAME runtime thread the Claude agents use — so they appear in the :7788 chat.

Claude is NOT used by the workers. The only time a Claude agent is involved is escalation:
a worker sends a 'reason' question to the leader/researcher inbox when a step genuinely needs
LLM reasoning. That is the whole point — max Python, Claude only when needed.
"""
