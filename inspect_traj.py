"""
用法：python inspect_traj.py <path_to_pddl.jsonl> [exp_id]
示例：
    python inspect_traj.py logs/test_run/logs/pddl.jsonl
    python inspect_traj.py logs/test_run/logs/pddl.jsonl 0
"""
import json
import sys


def load_records(path):
    with open(path) as f:
        txt = f.read()
    dec = json.JSONDecoder()
    i = 0
    objs = []
    while i < len(txt):
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            break
        obj, end = dec.raw_decode(txt, i)
        objs.append(obj)
        i = end
    return objs


def print_summary(rec):
    print(f"=== EXP {rec.get('id')} | {rec.get('task_name')} | {rec.get('difficulty')} ===")
    print(f"goal: {rec.get('goal')}")
    print(f"is_done={rec.get('is_done')}  progress_rate={rec.get('progress_rate')}  "
          f"grounding_acc={rec.get('grounding_acc')}")
    print(f"score_change_record: {rec.get('score_change_record')}")
    traj = rec.get("trajectory", {})
    print(f"# turns: {len(traj)}")


def print_full(rec):
    print_summary(rec)
    traj = rec.get("trajectory", {})
    for turn_key, turn in traj.items():
        print(f"\n--- {turn_key} ---")
        for k in ("Goal", "Observation", "Action", "Progress Rate"):
            if k in turn:
                v = str(turn[k])
                print(f"  {k}: {v}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    recs = load_records(path)
    if len(sys.argv) >= 3:
        exp_id = int(sys.argv[2])
        for r in recs:
            if r.get("id") == exp_id:
                print_full(r)
                return
        print(f"exp_id {exp_id} not found")
    else:
        for r in recs:
            print_summary(r)
            print()


if __name__ == "__main__":
    main()
