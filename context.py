from robot import Panda
from perception import Vision
from saycan import Belief
class Context:
 def __init__(self,rt,task):self.rt=rt;self.robot=Panda(rt);self.vision=Vision(rt);self.belief=Belief();self.task_place=("place" in task.lower() or "near" in task.lower())
 def observe(self):self.belief.objects.update(self.vision.observe());return self.belief
 def execute(self,c):
  from skills import Skills
  return Skills(self).execute(c)
