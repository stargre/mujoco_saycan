from robot import Panda
from perception import Vision
from saycan import Belief, task_entities


class Context:
    def __init__(self, rt, task):
        self.rt = rt
        self.robot = Panda(rt)
        self.vision = Vision(rt)
        self.belief = Belief()
        self.held_tcp_to_object_offset_world = None
        text = task.lower()
        self.task_place = any(
            word in text
            for word in ("place", "near", "put", "放", "旁边", "附近")
        )
        self.task_target, _ = task_entities(task)

    def observe(self):
        self.belief.objects.update(self.vision.observe())
        return self.belief

    def execute(self, call):
        from skills import Skills
        return Skills(self).execute(call)
