import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'explore5.py')).read().split("res=[]")[0])
PAIRS=[(1,3),(2,5),(4,7)]
for cond in ["drift","shared_change"]:
    eps = q[q.condition==cond].episode_id.unique()[:3]
    for ep in eps:
        print("==",cond,ep)
        for m in q[q.episode_id==ep].index:
            s=H[m]; out=[]
            for lo in range(0,256,48):
                cells=[]
                for p in PAIRS:
                    c=tab(s,*p,lo,lo+48); r=c[:,1]/np.maximum(c.sum(1),1)
                    cells.append(f"{r[0]:.1f}/{r[1]:.1f}")
                out.append(" ".join(cells))
            print(m[-4:], q.loc[m].machine_type[:4], " | ".join(out))
