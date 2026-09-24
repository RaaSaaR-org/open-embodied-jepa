import json,glob,numpy as np,os
os.chdir(__import__("os").environ.get("JEPA_ROOT", ".") + "/outputs/task051-scratch/forensics-a")
d=json.load(open('49112-ceiling.json'))
print(d['reason'], d['close_start'], d['final'] if not isinstance(d['final'],dict) else list(d['final'].keys()))
print(list(d['rows'][0].keys()))
for f in sorted(glob.glob("491*-ceiling.json")):
    d=json.load(open(f)); rows=d['rows']
    cs=[r for r in rows]
    key='apple' if 'apple' in rows[0] else [k for k in rows[0] if 'apple' in k][0]
    a=np.array([r[key][:3] if isinstance(r[key],list) else r[key]['pos'] for r in rows])
    disp=100*np.linalg.norm(a[:,:2]-a[0,:2],axis=1)
    ck=[k for k in rows[0] if 'contact' in k]
    print(f[:5], d['reason'], "n",len(rows), "disp", " ".join("%.1f"%x for x in disp[::3]), ck[:3])
