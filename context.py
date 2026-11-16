from robot import Panda
from perception import Vision
from saycan import Belief


class Context:
    def __init__(self, rt, task):
        self.rt = rt
        self.robot = Panda(rt)
        self.vision = Vision(rt)
        self.belief = Belief()
        self.held_tcp_to_object_offset_world = None
        text = task.lower()
        self.task_place = ("place" in text) or ("near" in text)
        self.task_target = (
            "red_cube" if "red" in text else
            "blue_cube" if "blue" in text else None
        )

    def observe(self):
        self.belief.objects.update(self.vision.observe())
        return self.belief

    def execute(self, call):
        from skills import Skills
        return Skills(self).execute(call)
