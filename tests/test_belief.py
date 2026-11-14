import unittest
from saycan import Belief
class T(unittest.TestCase):
 def test_empty(self):self.assertEqual(Belief().objects,{})
if __name__=="__main__":unittest.main()
