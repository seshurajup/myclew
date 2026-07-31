"""afp_engine — a tiny complete proof-search testbed for *Advancing Mathematics Research with
AI-Driven Formal Proof Search* (Tsoukalas et al., arXiv:2605.22763, https://arxiv.org/pdf/2605.22763).

Support module for learning/paper_packs/ai_formal_proof.py. A synthetic formal system with the properties
that make the paper's domain special: expressions over {+, *, constants, x}, a sound rewrite rule set
(every rule preserves the polynomial's value — checkable), a proof CHECKER that cannot be fooled, planted
theorems of controllable depth, and search (BFS / best-first) whose proofs are machine-verified before any
number is reported. Lean and Mathlib are NOT reproduced; this testbed reproduces their load-bearing
property — a perfect, cheap verifier — at a scale where search behaviour is measurable in seconds.
"""
import random, heapq
from collections import deque

# expression: ('x',) | ('c', k) | ('+', a, b) | ('*', a, b)
def size(e): return 1 if e[0] in ('x','c') else 1 + size(e[1]) + size(e[2])

def val(e, x):
    if e[0]=='x': return x
    if e[0]=='c': return e[1]
    a,b = val(e[1],x), val(e[2],x)
    return a+b if e[0]=='+' else a*b

def rw_at(e):
    """all one-step rewrites of e at the ROOT only"""
    out=[]
    if e[0]=='+':
        a,b=e[1],e[2]
        if b==('c',0): out.append(('add0',a))
        if a==('c',0): out.append(('add0',b))
        out.append(('commA',('+',b,a)))
        if a[0]=='c' and b[0]=='c': out.append(('fold',('c',a[1]+b[1])))
        # undistribute: (a*b)+(a*c) -> a*(b+c)
        if a[0]=='*' and b[0]=='*':
            if a[1]==b[1]: out.append(('undist',('*',a[1],('+',a[2],b[2]))))
            if a[2]==b[2]: out.append(('undist',('*',a[2],('+',a[1],b[1]))))
    if e[0]=='*':
        a,b=e[1],e[2]
        if b==('c',1): out.append(('mul1',a))
        if a==('c',1): out.append(('mul1',b))
        if b==('c',0) or a==('c',0): out.append(('mul0',('c',0)))
        out.append(('commM',('*',b,a)))
        if a[0]=='c' and b[0]=='c': out.append(('fold',('c',a[1]*b[1])))
        if b[0]=='+': out.append(('dist',('+',('*',a,b[1]),('*',a,b[2]))))
        if a[0]=='+': out.append(('dist',('+',('*',a[1],b),('*',a[2],b))))
    # expanding (inverse) rules
    out.append(('addI',('+',e,('c',0))))
    out.append(('mulI',('*',e,('c',1))))
    return out

def positions(e, pre=()):
    yield pre
    if e[0] in ('+','*'):
        yield from positions(e[1], pre+(1,))
        yield from positions(e[2], pre+(2,))

def at(e,p):
    for i in p: e=e[i]
    return e

def replace(e,p,new):
    if not p: return new
    i=p[0]; sub=replace(e[i],p[1:],new)
    return (e[0],) + tuple(sub if j==i else e[j] for j in (1,2))

def neighbours(e, cap):
    out=[]
    for p in positions(e):
        for name,new in rw_at(at(e,p)):
            ne=replace(e,p,new)
            if size(ne)<=cap: out.append((ne,(name,p)))
    return out

def check(e0, proof, target, cap=10**6):
    cur=e0
    for name,p in proof:
        opts={n:new for n,new in rw_at(at(cur,p))}
        if name not in opts: return False
        cur=replace(cur,p,opts[name])
    return cur==target

def plant(t,L,rng,cap):
    cur=t
    for _ in range(L):
        nb=neighbours(cur,cap)
        cur,_=rng.choice(nb)
    return cur

def bfs(s,t,cap,maxexp=200000):
    if s==t: return 0,[]
    seen={s:None}; q=deque([s]); exp=0
    while q:
        cur=q.popleft(); exp+=1
        if exp>maxexp: return None,None
        for nx,mv in neighbours(cur,cap):
            if nx not in seen:
                seen[nx]=(cur,mv)
                if nx==t:
                    path=[];c=nx
                    while seen[c] is not None:
                        p_,m=seen[c]; path.append(m); c=p_
                    return exp,path[::-1]
                q.append(nx)
    return None,None

def guided(s,t,h,cap,maxexp=200000):
    if s==t: return 0,[]
    seen={s:None}; pq=[(h(s),0,s)]; exp=0; tie=0
    while pq:
        _,_,cur=heapq.heappop(pq); exp+=1
        if exp>maxexp: return None,None
        for nx,mv in neighbours(cur,cap):
            if nx not in seen:
                seen[nx]=(cur,mv)
                if nx==t:
                    path=[];c=nx
                    while seen[c] is not None:
                        p_,m=seen[c]; path.append(m); c=p_
                    return exp,path[::-1]
                tie+=1; heapq.heappush(pq,(h(nx),tie,nx))
    return None,None



# ---------------- features + instance generation (for value-guided search) ----------------
def depth(e):
    return 1 if e[0] in ('x', 'c') else 1 + max(depth(e[1]), depth(e[2]))

def counts(e):
    d = {'+': 0, '*': 0, 'x': 0, 'c': 0}
    st = [e]
    while st:
        n = st.pop(); d[n[0]] += 1
        if n[0] in ('+', '*'): st += [n[1], n[2]]
    return d

def subterms(e):
    st = [e]; out = []
    while st:
        n = st.pop(); out.append(n)
        if n[0] in ('+', '*'): st += [n[1], n[2]]
    return out

def feats(e, t):
    ce, ct = counts(e), counts(t)
    se, st_ = set(subterms(e)), set(subterms(t))
    return [size(e), size(t), size(e) - size(t), depth(e), depth(t),
            ce['+'], ce['*'], ce['x'], ce['c'], ct['+'], ct['*'], ct['x'], ct['c'],
            float(('c', 0) in se), float(('c', 1) in se), len(se & st_), len(se - st_)]

NF = 17

def instances(targets, L, n, seed, capd=5):
    rng = random.Random(seed); out = []
    for i in range(n):
        t = targets[i % len(targets)]
        cap = size(t) + capd
        s = plant(t, L, rng, cap)
        if s != t: out.append((s, t, cap))
    return out

def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())
