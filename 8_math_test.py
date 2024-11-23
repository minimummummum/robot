import math
import random
import numpy as np
arm_length = 1
x = 0
y = -arm_length
angle = 45
x = arm_length * math.sin(np.radians(angle))
y = -(arm_length * math.cos(np.radians(angle)))
print(angle, x, y)
