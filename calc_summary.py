import os
import re
import glob

log_dir = "/mnt/data/cxw/Hiagent/HiAgent/logs"

def parse_pddl_txt(path):
    with open(path, 'r') as f:
        content = f.read()
    
    success_rates = []
    progress_rates = []
    grounding_accs = []
    
    for line in content.splitlines():
        if line.startswith("[EXP]"):
            sr = re.search(r"\[success_rate\]:\s*([\w\.]+)", line)
            pr = re.search(r"\[progress_rate\]:\s*([\w\.]+)", line)
            ga = re.search(r"\[grounding_acc\]:\s*([\w\.]+)", line)
            if sr:
                val = sr.group(1)
                success_rates.append(float(val) if val.replace('.', '', 1).isdigit() else (1.0 if val == "True" else 0.0))
            if pr:
                progress_rates.append(float(pr.group(1)))
            if ga:
                grounding_accs.append(float(ga.group(1)))
    
    if success_rates:
        return {
            "episodes": len(success_rates),
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "avg_progress_rate": sum(progress_rates) / len(progress_rates),
            "avg_grounding_acc": sum(grounding_accs) / len(grounding_accs),
        }
    return None

print("=" * 80)
print(f"{'Agent/Task':<60} {'Episodes':>8} {'SR':>8} {'PR':>8} {'GA':>8}")
print("=" * 80)

for pddl_file in sorted(glob.glob(os.path.join(log_dir, "*/pddl.txt"))):
    result = parse_pddl_txt(pddl_file)
    if result:
        name = os.path.basename(os.path.dirname(pddl_file))
        print(f"{name:<60} {result['episodes']:>8} {result['avg_success_rate']:>8.4f} {result['avg_progress_rate']:>8.4f} {result['avg_grounding_acc']:>8.4f}")

print("=" * 80)
