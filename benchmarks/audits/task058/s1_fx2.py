import json,numpy as np,os
os.chdir(__import__("os").environ.get("JEPA_ROOT", ".") + "/outputs/task051-scratch/forensics-a")
for s in (49112,49114,49115,49103):
    d=json.load(open(f'{s}-ceiling.json')); rows=d['rows']
    close=[r for r in rows if r['phase']=='close']
    a0=np.array(close[0]['apple'][:3])
    out=[]
    for r in rows:
        if r['phase'] not in ('close','lift'): continue
        disp=100*np.linalg.norm(np.array(r['apple'][:2])-a0[:2])
        out.append(f"{r['phase'][0]}{r['phase_commands']}:{disp:.1f}{'C' if r['contact'] else '-'}")
    print(s, " ".join(out[:60]))
