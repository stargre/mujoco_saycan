import os,json
class Belief:
 def __init__(self):self.objects={};self.held=None;self.completed=[]
 def summary(self):return {"objects":self.objects,"held_object":self.held,"completed":self.completed}
class Scorer:
 def __init__(self,kind="qwen"):
  self.kind=kind
  if kind=="qwen":
   from openai import OpenAI
   key=os.getenv("DASHSCOPE_API_KEY")
   if not key:raise RuntimeError("Set DASHSCOPE_API_KEY")
   self.client=OpenAI(api_key=key,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1");self.model=os.getenv("QWEN_MODEL","qwen-plus")
 def score(self,task,calls,b):
  if self.kind=="keyword":
   return {c["id"]:{"relevance":.9 if any(w in task.lower() for w in {"Find":["find"],"PickUp":["pick","grab"],"PlaceNear":["place","near"],"Done":["done","finish"],"Explore":["search","explore"]}.get(c["name"],[])) else .3} for c in calls}
  prompt={"task":task,"belief":b.summary(),"candidates":calls,"instruction":"Return JSON keyed by candidate id with relevance from 0 to 1 and rationale. Score semantic next-step relevance only, not physical feasibility."}
  r=self.client.chat.completions.create(model=self.model,temperature=0,response_format={"type":"json_object"},messages=[{"role":"user","content":json.dumps(prompt)}]);d=json.loads(r.choices[0].message.content)
  if set(d)!={c["id"] for c in calls}:raise ValueError("Qwen candidate IDs mismatch")
  return {k:{"relevance":float(v["relevance"])} for k,v in d.items()}
class SayCan:
 def __init__(self,ctx,scorer):self.c=ctx;self.scorer=scorer;self.history=[]
 def run(self,task,max_rounds=12):
  for i in range(max_rounds):
   self.c.observe();b=self.c.belief;t=task.lower();o=b.objects;h=b.held
   if "place" in t or "near" in t:
    if "red_cube" not in o:calls=[("Find",{"target":"red_cube"})]
    elif "blue_cube" not in o:calls=[("Find",{"target":"blue_cube"})]
    elif h!="red_cube":calls=[("PickUp",{"target":"red_cube"})]
    else:calls=[("PlaceNear",{"object":"red_cube","target":"blue_cube","relation":"near"})]
   elif "pick" in t or "grab" in t:
    x="red_cube" if "red" in t else "blue_cube";calls=[("Find",{"target":x})] if x not in o else ([("PickUp",{"target":x})] if h!=x else [("Done",{})])
   else:calls=[("Explore",{})]
   cs=[{"id":f"{n}:{j}","name":n,"args":a} for j,(n,a) in enumerate(calls)];say=self.scorer.score(task,cs,b);rank=[]
   for c in cs:
    n=c["name"];a=c["args"];can=1.
    if n=="PickUp" and (h or a["target"] not in o):can=0.
    if n=="PlaceNear" and (h!=a["object"] or a["target"] not in o):can=0.
    rel=say[c["id"]]["relevance"];rank.append((rel*can,rel,can,c))
   rank.sort(reverse=True,key=lambda z:z[0])
   if not rank or rank[0][2]==0:return {"success":False,"status":"no_feasible_candidate","history":self.history}
   score,rel,can,c=rank[0];res=self.c.execute(c);self.history.append({"call":c,"say":rel,"can":can,"score":score,"result":res})
   if c["name"]=="Done" and res["success"]:return {"success":True,"status":"task_complete","history":self.history}
   if not res["success"]:return {"success":False,"status":res["status"],"history":self.history}
  return {"success":False,"status":"max_rounds","history":self.history}
