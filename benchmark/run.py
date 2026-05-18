#!/usr/bin/env python3
import os
import sys
import time
import glob
import subprocess
from typing import List, Dict, Tuple
import threading
from typing import List


# Hardcoded config execution order (relative paths, e.g., configs/...) 
# Edit this list to control run order.
CONFIG_FILES: List[str] = [
	# "configs/config_quest_256.json",
	# "configs/config_quest_1024.json",
	# "configs/config_quest_4096.json",
	# "configs/config_quest_8192.json",
	# "configs/config_ds_256.json",
	# "configs/config_ds_1024.json",
	# "configs/config_ds_4096.json",
	# "configs/config_ds_8192.json",
	# "configs/config_sparq_1024.json",
	# "configs/config_full.json",
	# "configs/rabitq_example.json"
	# "configs/config_oracle_topk_1024.json",
	# "configs/config_quest_256.json"
	"configs/configs_rabitq/rabitq_0.65.json",
	"configs/configs_rabitq/rabitq_0.70.json",
	"configs/configs_rabitq/rabitq_0.75.json",
	"configs/configs_rabitq/rabitq_0.80.json",
	"configs/configs_rabitq/rabitq_0.85.json"

]


def list_config_files(config_dir: str) -> List[str]:
	pattern = os.path.join(config_dir, "*.json")
	files = sorted(glob.glob(pattern))
	return files


def query_gpus() -> List[Dict[str, int]]:
	"""Return list of GPUs with fields: index, mem_used, mem_total, util.
	Uses nvidia-smi for reliable parsing.
	"""
	cmd = [
		"nvidia-smi",
		"--query-gpu=index,memory.used,memory.total,utilization.gpu",
		"--format=csv,noheader,nounits",
	]
	try:
		out = subprocess.check_output(cmd, encoding="utf-8")
	except subprocess.CalledProcessError as e:
		print(f"Failed to run nvidia-smi: {e}")
		return []
	except FileNotFoundError:
		print("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
		return []

	gpus = []
	for line in out.strip().splitlines():
		parts = [p.strip() for p in line.split(",")]
		if len(parts) != 4:
			continue
		try:
			idx = int(parts[0])
			mem_used = int(parts[1])
			mem_total = int(parts[2])
			util = int(parts[3])
		except ValueError:
			continue
		gpus.append({"index": idx, "mem_used": mem_used, "mem_total": mem_total, "util": util})
	return gpus


def idle_gpu_indices(gpus: List[Dict[str, int]], mem_threshold_mb: int = 10000, util_threshold_pct: int = 30) -> List[int]:
	"""Consider a GPU idle if memory used and utilization are below thresholds."""
	idle = []
	for g in gpus: 
		if g["mem_used"] <= mem_threshold_mb and g["util"] <= util_threshold_pct:
			idle.append(g["index"])
	return idle


def launch_job(gpu_idx: int, config_path: str) -> subprocess.Popen:
	env = os.environ.copy()
	env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
	# Assume running from benchmark/; scripts is one level up
	cmd = ["bash", "scripts/run_longbench.sh", config_path]
	print(f"[LAUNCH] GPU {gpu_idx} -> {' '.join(cmd)}")
	time.sleep(3)
	# Start the process; merge stdout/stderr for simpler logging
	return subprocess.Popen(
		cmd,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
	)


def stream_output(tag: str, proc: subprocess.Popen):
	assert proc.stdout is not None
	for line in proc.stdout:
		if not line:
			break
		print(f"[{tag}] {line.rstrip()}")


def schedule_runs(
	configs: List[str],
	reserve_free: bool = True,
	poll_interval: float = 5.0,
	max_concurrent: int = 3,
):
	"""Schedule runs across available GPUs, keeping optionally one GPU free.

	- configs: list of config file paths to run
	- reserve_free: if True, keep at least one idle GPU unassigned
	- poll_interval: seconds between scheduling checks
	- max_concurrent: upper bound on number of GPUs used concurrently
	"""
	pending = configs[:]
	running: Dict[int, Tuple[str, subprocess.Popen]] = {}  # gpu_idx -> (config_path, proc)
	log_threads: Dict[int, threading.Thread] = {}

	while pending or running:
		gpus = query_gpus()
		if not gpus:
			print("No GPUs detected, waiting...")
			time.sleep(poll_interval)
			continue

		idle = idle_gpu_indices(gpus)
		# Exclude GPUs already running jobs
		idle = [i for i in idle if i not in running]

		# Conditionally reserve one GPU free only if more than one idle is available
		effective_reserve_free = reserve_free and len(idle) > 1
		max_assignable = max(len(idle) - (1 if effective_reserve_free else 0), 0)
		# Respect upper bound on concurrent GPUs in use
		available_slots = max(min(max_assignable, max_concurrent - len(running)), 0)

		# Assign jobs to idle GPUs
		while available_slots > 0 and pending:
			gpu_idx = idle.pop(0)
			config_path = pending.pop(0)
			proc = launch_job(gpu_idx, config_path)
			running[gpu_idx] = (config_path, proc)
			# Start a dedicated thread to stream logs so scheduler loop isn't blocked
			t = threading.Thread(target=stream_output, args=(f"GPU{gpu_idx}", proc), daemon=True)
			t.start()
			log_threads[gpu_idx] = t
			available_slots -= 1

		# Check for finished processes
		finished_gpus = []
		for gpu_idx, (cfg, proc) in running.items():
			ret = proc.poll()
			if ret is not None:
				print(f"[DONE] GPU {gpu_idx} finished {cfg} with exit code {ret}")
				finished_gpus.append(gpu_idx)

		for gpu_idx in finished_gpus:
			running.pop(gpu_idx, None)
			# Let log thread finish naturally; if it's still alive, give it a moment
			t = log_threads.pop(gpu_idx, None)
			if t and t.is_alive():
				# No join with timeout to avoid blocking; thread is daemon
				pass

		if pending:
			print(f"[STATUS] Pending: {len(pending)}, Running: {len(running)}, Idle GPUs: {idle}")
		else:
			print(f"[STATUS] All jobs dispatched. Running: {len(running)}")

		time.sleep(poll_interval)

	print("All runs completed.")


def main():
	# If hardcoded CONFIG_FILES is non-empty, use it as the run order
	if CONFIG_FILES:
		configs = [c for c in CONFIG_FILES if c.endswith(".json") and os.path.exists(c)]
		missing = [c for c in CONFIG_FILES if not os.path.exists(c)]
		if missing:
			print(f"[WARN] Missing config files (skipped): {missing}")
	else:
		# Fallback to auto-scan when list is empty
		config_dir = os.path.join("configs")
		configs = list_config_files(config_dir)
		if not configs:
			print(f"No config files found in {config_dir}")
			sys.exit(1)

	# Allow user to filter by prefix or specific files via args
	args = sys.argv[1:]
	if args:
		# If args are provided, treat them as paths or glob patterns
		selected = []
		for a in args:
			selected.extend(sorted(glob.glob(a)))
		# Keep only those that exist and are json
		selected = [p for p in selected if p.endswith(".json") and os.path.exists(p)]
		if selected:
			configs = selected
			print(f"Using {len(configs)} configs from arguments.")
		else:
			print("Args provided but no matching .json files found; using default scan.")

	print(f"Discovered {len(configs)} config(s). Starting scheduling...")
	schedule_runs(configs, reserve_free=True, poll_interval=5.0, max_concurrent=2)


if __name__ == "__main__":
	main()

