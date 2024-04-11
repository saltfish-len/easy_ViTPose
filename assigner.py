"""
This is a demo script to auto assign easy_ViTPose to a tmux session.
"""

from tmux_web.utils.tmux_helper import TmuxHelper

views = ['1', '3', '5', '8', '10', '12', '14', '17', '30']
CUDA_VISIBLE_DEVICES = ['0', '1', '2', '3', '4', '5', '6', '7']


# CUDA_VISIBLE_DEVICES=1 python easymocap.py -s [sequences] -o output_new/myvit -v [views]


def prepare_cmds(seqs: [str], views: [str], output: str):
    pairs = []
    for seq in seqs:
        for view in views:
            pairs.append((seq, view))
    # assign to len(devices)
    devices_cmd = [[] for _ in CUDA_VISIBLE_DEVICES]
    for i, (s, v) in enumerate(pairs):
        devices_cmd[i % len(CUDA_VISIBLE_DEVICES)].append(f"python easymocap.py -s {s} -o {output} -v {v}")
    for i, cmds in enumerate(devices_cmd):
        devices_cmd[i] = " && ".join(cmds)
    return devices_cmd


def assign_cmds(cmds: [str]):
    tmux = TmuxHelper()
    session_created = []
    for i, cmd in enumerate(cmds):
        session_name = f"myvit_{i}"
        tmux.create_session(session_name)
        tmux.set_environment(session_name, "CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES[i])
        tmux.run_command(session_name, cmd)
        session_created.append(session_name)
    return session_created

if __name__ == "__main__":
    cmds = prepare_cmds(["seq1", "seq2", "seq3"], views, "output_new/myvit")
    session_created = assign_cmds(cmds)
    print(session_created)

