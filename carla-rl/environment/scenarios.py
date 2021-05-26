""" Scenarios file for different road paths """

import os
import glob
import sys

# CARLA_9_4_PATH = "/home/swapnil/carla910"#os.environ.get("CARLA_9_4_PATH")
# if CARLA_9_4_PATH == None:
#     raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

# user_paths = os.environ['PYTHONPATH'].split(os.pathsep)

# # try:
# # sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
# #     sys.version_info.major,
# #     sys.version_info.minor,
# #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# # except IndexError:
# #     pass

# sys.path.append("/home/swapnil/carla910/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg")

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")

if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

import carla

from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation
import random

WAYPOINT_DICT_Town01 = {
    0: Transform(Location(x=271.0400085449219, y=129.489990234375, z=1.32), Rotation(yaw=179.999755859375)),
    1: Transform(Location(x=270.79998779296875, y=133.43003845214844, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    2: Transform(Location(x=237.6999969482422, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    3: Transform(Location(x=237.6999969482422, y=133.239990234375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    4: Transform(Location(x=216.26998901367188, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    5: Transform(Location(x=216.26998901367188, y=133.239990234375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    6: Transform(Location(x=191.3199920654297, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    7: Transform(Location(x=191.3199920654297, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    8: Transform(Location(x=157.1899871826172, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    9: Transform(Location(x=157.1899871826172, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    10: Transform(Location(x=338.97998046875, y=301.2599792480469, z=1.32), Rotation(yaw=-90.00029754638672)),
    11: Transform(Location(x=128.94998168945312, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    12: Transform(Location(x=128.94998168945312, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    13: Transform(Location(x=119.46998596191406, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    14: Transform(Location(x=105.43998718261719, y=133.24002075195312, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    15: Transform(Location(x=92.11000061035156, y=39.709999084472656, z=1.32), Rotation(yaw=-90.00029754638672)),
    16: Transform(Location(x=88.6199951171875, y=26.559999465942383, z=1.32), Rotation(yaw=90.00004577636719)),
    17: Transform(Location(x=92.11000061035156, y=30.820009231567383, z=1.32), Rotation(yaw=-90.00029754638672)),
    18: Transform(Location(x=88.6199951171875, y=15.279999732971191, z=1.32), Rotation(yaw=90.00004577636719)),
    19: Transform(Location(x=92.11000061035156, y=86.95999908447266, z=1.32), Rotation(yaw=-90.00029754638672)),
    20: Transform(Location(x=88.6199951171875, y=72.6199951171875, z=1.32), Rotation(yaw=90.00004577636719)),
    21: Transform(Location(x=335.489990234375, y=298.80999755859375, z=1.32), Rotation(yaw=90.00004577636719)),
    22: Transform(Location(x=92.1099853515625, y=95.44999694824219, z=1.32), Rotation(yaw=-90.00029754638672)),
    23: Transform(Location(x=88.61998748779297, y=95.44999694824219, z=1.32), Rotation(yaw=90.00004577636719)),
    24: Transform(Location(x=92.1099853515625, y=113.05999755859375, z=1.32), Rotation(yaw=-90.00029754638672)),
    25: Transform(Location(x=88.61998748779297, y=103.37999725341797, z=1.32), Rotation(yaw=90.00004577636719)),
    26: Transform(Location(x=92.1099853515625, y=159.9499969482422, z=1.32), Rotation(yaw=-90.00029754638672)),
    27: Transform(Location(x=88.61998748779297, y=145.83999633789062, z=1.32), Rotation(yaw=90.00004577636719)),
    28: Transform(Location(x=92.1099853515625, y=176.88999938964844, z=1.32), Rotation(yaw=-90.00029754638672)),
    29: Transform(Location(x=88.61998748779297, y=169.84999084472656, z=1.32), Rotation(yaw=90.00004577636719)),
    30: Transform(Location(x=-2.4200193881988525, y=187.97000122070312, z=1.32), Rotation(yaw=89.9996109008789)),
    31: Transform(Location(x=1.5599803924560547, y=187.9700164794922, z=1.32), Rotation(yaw=-90.00040435791016)),
    32: Transform(Location(x=338.97998046875, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    33: Transform(Location(x=-2.4200096130371094, y=149.8300018310547, z=1.32), Rotation(yaw=89.9996109008789)),
    34: Transform(Location(x=1.5599901676177979, y=149.83001708984375, z=1.32), Rotation(yaw=-90.00040435791016)),
    35: Transform(Location(x=-2.4200096130371094, y=120.0199966430664, z=1.32), Rotation(yaw=89.9996109008789)),
    36: Transform(Location(x=1.5599901676177979, y=120.02001953125, z=1.32), Rotation(yaw=-90.00040435791016)),
    37: Transform(Location(x=-2.4200048446655273, y=79.31999969482422, z=1.32), Rotation(yaw=89.9996109008789)),
    38: Transform(Location(x=1.5599950551986694, y=79.32001495361328, z=1.32), Rotation(yaw=-90.00040435791016)),
    39: Transform(Location(x=-2.4200048446655273, y=48.70000076293945, z=1.32), Rotation(yaw=89.9996109008789)),
    40: Transform(Location(x=1.5599950551986694, y=48.70001983642578, z=1.32), Rotation(yaw=-90.00040435791016)),
    41: Transform(Location(x=-2.420001268386841, y=17.779998779296875, z=1.32), Rotation(yaw=89.9996109008789)),
    42: Transform(Location(x=1.55999755859375, y=22.440019607543945, z=1.32), Rotation(yaw=-90.00040435791016)),
    43: Transform(Location(x=335.489990234375, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    44: Transform(Location(x=21.770000457763672, y=-1.9599987268447876, z=1.32), Rotation(yaw=179.9996337890625)),
    45: Transform(Location(x=14.139999389648438, y=2.0200109481811523, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    46: Transform(Location(x=47.939998626708984, y=-1.9599950313568115, z=1.32), Rotation(yaw=179.9996337890625)),
    47: Transform(Location(x=47.939998626708984, y=2.020014524459839, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    48: Transform(Location(x=72.5999984741211, y=-1.9599950313568115, z=1.32), Rotation(yaw=179.9996337890625)),
    49: Transform(Location(x=62.12999725341797, y=2.020014524459839, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    50: Transform(Location(x=116.63999938964844, y=-1.95999014377594, z=1.32), Rotation(yaw=179.9996337890625)),
    51: Transform(Location(x=110.02999877929688, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    52: Transform(Location(x=137.7899932861328, y=-1.95999014377594, z=1.32), Rotation(yaw=179.9996337890625)),
    53: Transform(Location(x=126.38999938964844, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    54: Transform(Location(x=338.97998046875, y=226.75, z=1.32), Rotation(yaw=-90.00029754638672)),
    55: Transform(Location(x=185.55999755859375, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    56: Transform(Location(x=173.14999389648438, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    57: Transform(Location(x=209.5800018310547, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    58: Transform(Location(x=209.5800018310547, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    59: Transform(Location(x=244.09999084472656, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    60: Transform(Location(x=244.09999084472656, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    61: Transform(Location(x=278.80999755859375, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    62: Transform(Location(x=278.80999755859375, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    63: Transform(Location(x=316.8500061035156, y=-1.9599803686141968, z=1.32), Rotation(yaw=179.9996337890625)),
    64: Transform(Location(x=306.28997802734375, y=2.02001953125, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    65: Transform(Location(x=334.8299865722656, y=217.0800018310547, z=1.32), Rotation(yaw=90.00004577636719)),
    66: Transform(Location(x=363.0, y=-1.9599609375, z=1.32), Rotation(yaw=179.9996337890625)),
    67: Transform(Location(x=356.79998779296875, y=2.0200390815734863, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    68: Transform(Location(x=378.17999267578125, y=-1.9599609375, z=1.32), Rotation(yaw=179.9996337890625)),
    69: Transform(Location(x=378.17999267578125, y=2.0200390815734863, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    70: Transform(Location(x=396.4499816894531, y=19.9200382232666, z=1.32), Rotation(yaw=-90.00029754638672)),
    71: Transform(Location(x=392.4700012207031, y=19.9200382232666, z=1.32), Rotation(yaw=90.00004577636719)),
    72: Transform(Location(x=395.9599914550781, y=164.1699981689453, z=1.32), Rotation(yaw=-90.00029754638672)),
    73: Transform(Location(x=392.4700012207031, y=164.1699981689453, z=1.32), Rotation(yaw=90.00004577636719)),
    74: Transform(Location(x=395.9599914550781, y=105.38999938964844, z=1.32), Rotation(yaw=-90.00029754638672)),
    75: Transform(Location(x=392.4700012207031, y=105.38999938964844, z=1.32), Rotation(yaw=90.00004577636719)),
    76: Transform(Location(x=395.9599914550781, y=68.86003875732422, z=1.32), Rotation(yaw=-90.00029754638672)),
    77: Transform(Location(x=392.4700012207031, y=68.86003875732422, z=1.32), Rotation(yaw=90.00004577636719)),
    78: Transform(Location(x=395.9599914550781, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
    79: Transform(Location(x=392.4700012207031, y=308.2099914550781, z=1.32), Rotation(yaw=90.00004577636719)),
    80: Transform(Location(x=395.9599914550781, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    81: Transform(Location(x=392.4700012207031, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    82: Transform(Location(x=395.9599914550781, y=212.89999389648438, z=1.32), Rotation(yaw=-90.00029754638672)),
    83: Transform(Location(x=392.4700012207031, y=212.89999389648438, z=1.32), Rotation(yaw=90.00004577636719)),
    84: Transform(Location(x=1.5099804401397705, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
    85: Transform(Location(x=-1.2800195217132568, y=309.4599914550781, z=1.32), Rotation(yaw=90.00004577636719)),
    86: Transform(Location(x=1.5099804401397705, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    87: Transform(Location(x=-1.980019450187683, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    88: Transform(Location(x=121.22996520996094, y=195.00999450683594, z=1.32), Rotation(yaw=179.999755859375)),
    89: Transform(Location(x=105.22998809814453, y=198.5, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    90: Transform(Location(x=118.94999694824219, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    91: Transform(Location(x=111.56999969482422, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    92: Transform(Location(x=141.12998962402344, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    93: Transform(Location(x=125.9699935913086, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    94: Transform(Location(x=22.17997932434082, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    95: Transform(Location(x=22.17997932434082, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    96: Transform(Location(x=92.10997772216797, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
    97: Transform(Location(x=46.14997863769531, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    98: Transform(Location(x=46.14997863769531, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    99: Transform(Location(x=65.3499755859375, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    100: Transform(Location(x=60.10997772216797, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    101: Transform(Location(x=381.3399963378906, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    102: Transform(Location(x=381.3399658203125, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    103: Transform(Location(x=366.53997802734375, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    104: Transform(Location(x=358.39996337890625, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    105: Transform(Location(x=320.8699645996094, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    106: Transform(Location(x=306.76995849609375, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    107: Transform(Location(x=88.61997985839844, y=295.32000732421875, z=1.32), Rotation(yaw=90.00004577636719)),
    108: Transform(Location(x=301.3399658203125, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    109: Transform(Location(x=301.3399658203125, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    110: Transform(Location(x=262.5999755859375, y=327.04998779296875, z=1.32), Rotation(yaw=179.999755859375)),
    111: Transform(Location(x=262.5999755859375, y=330.53997802734375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    112: Transform(Location(x=232.19998168945312, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    113: Transform(Location(x=232.19998168945312, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    114: Transform(Location(x=199.94998168945312, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    115: Transform(Location(x=199.94998168945312, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    116: Transform(Location(x=173.11997985839844, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    117: Transform(Location(x=173.11997985839844, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    118: Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    119: Transform(Location(x=124.73997497558594, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    120: Transform(Location(x=114.3499755859375, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    121: Transform(Location(x=142.91998291015625, y=326.9700012207031, z=1.32), Rotation(yaw=179.999755859375)),
    122: Transform(Location(x=142.91998291015625, y=330.4599914550781, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    123: Transform(Location(x=142.91998291015625, y=195.26998901367188, z=1.32), Rotation(yaw=179.999755859375)),
    124: Transform(Location(x=142.91998291015625, y=198.75999450683594, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    125: Transform(Location(x=178.7699737548828, y=195.26998901367188, z=1.32), Rotation(yaw=179.999755859375)),
    126: Transform(Location(x=178.7699737548828, y=198.75999450683594, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    127: Transform(Location(x=217.50997924804688, y=195.26998901367188, z=1.32), Rotation(yaw=179.999755859375)),
    128: Transform(Location(x=217.50997924804688, y=198.75999450683594, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    129: Transform(Location(x=88.61997985839844, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
    130: Transform(Location(x=256.3499755859375, y=195.5699920654297, z=1.32), Rotation(yaw=179.999755859375)),
    131: Transform(Location(x=256.3499755859375, y=199.05999755859375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    132: Transform(Location(x=299.39996337890625, y=195.5699920654297, z=1.32), Rotation(yaw=179.999755859375)),
    133: Transform(Location(x=299.39996337890625, y=199.05999755859375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    134: Transform(Location(x=158.0800018310547, y=27.18000030517578, z=1.32), Rotation(yaw=-90.00029754638672)),
    135: Transform(Location(x=153.75999450683594, y=18.889999389648438, z=1.32), Rotation(yaw=90.00004577636719)),
    136: Transform(Location(x=157.25, y=39.709999084472656, z=1.32), Rotation(yaw=-90.00029754638672)),
    137: Transform(Location(x=153.75999450683594, y=28.899999618530273, z=1.32), Rotation(yaw=90.00004577636719)),
    138: Transform(Location(x=191.0800018310547, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    139: Transform(Location(x=172.2899932861328, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    140: Transform(Location(x=92.1099853515625, y=227.22000122070312, z=1.32), Rotation(yaw=-90.00029754638672)),
    141: Transform(Location(x=202.5500030517578, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    142: Transform(Location(x=202.5500030517578, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    143: Transform(Location(x=234.26998901367188, y=55.84001922607422, z=1.32), Rotation(yaw=179.999755859375)),
    144: Transform(Location(x=234.26998901367188, y=59.33001708984375, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    145: Transform(Location(x=272.2900085449219, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    146: Transform(Location(x=272.2900085449219, y=59.33003616333008, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    147: Transform(Location(x=299.3999938964844, y=55.84000015258789, z=1.32), Rotation(yaw=179.999755859375)),
    148: Transform(Location(x=299.3999938964844, y=59.33003616333008, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    149: Transform(Location(x=299.3999938964844, y=129.75, z=1.32), Rotation(yaw=179.999755859375)),
    150: Transform(Location(x=299.3999938964844, y=133.2400360107422, z=1.32), Rotation(yaw=-9.1552734375e-05)),
    151: Transform(Location(x=88.61998748779297, y=212.89999389648438, z=1.32), Rotation(yaw=90.00004577636719))
}

WAYPOINT_DICT_Town02 = {
    0: Transform(Location(x=-3.679999828338623, y=251.36000061035156, z=1.32), Rotation(yaw=-89.99981689453125)),
    1: Transform(Location(x=-3.679999828338623, y=142.19000244140625, z=1.32), Rotation(yaw=-89.99981689453125)),
    2: Transform(Location(x=132.02999877929688, y=211.0, z=1.32), Rotation(yaw=89.99995422363281)),
    3: Transform(Location(x=135.87998962402344, y=226.04998779296875, z=1.32), Rotation(yaw=-89.99981689453125)),
    4: Transform(Location(x=132.02999877929688, y=201.1699981689453, z=1.32), Rotation(yaw=89.99995422363281)),
    5: Transform(Location(x=135.87998962402344, y=220.8300018310547, z=1.32), Rotation(yaw=-89.99981689453125)),
    6: Transform(Location(x=41.38999938964844, y=212.97999572753906, z=1.32), Rotation(yaw=89.99995422363281)),
    7: Transform(Location(x=-7.529999732971191, y=158.97999572753906, z=1.32), Rotation(yaw=89.99995422363281)),
    8: Transform(Location(x=45.23999786376953, y=225.58999633789062, z=1.32), Rotation(yaw=-89.99981689453125)),
    9: Transform(Location(x=41.38999938964844, y=203.89999389648438, z=1.32), Rotation(yaw=89.99995422363281)),
    10: Transform(Location(x=45.88999938964844, y=216.25999450683594, z=1.32), Rotation(yaw=-89.99981689453125)),
    11: Transform(Location(x=41.38999938964844, y=275.0299987792969, z=1.32), Rotation(yaw=89.99995422363281)),
    12: Transform(Location(x=45.23999786376953, y=291.2900085449219, z=1.32), Rotation(yaw=-89.99981689453125)),
    13: Transform(Location(x=41.38999938964844, y=257.4599914550781, z=1.32), Rotation(yaw=89.99995422363281)),
    14: Transform(Location(x=45.59000015258789, y=271.5099792480469, z=1.32), Rotation(yaw=-89.99981689453125)),
    15: Transform(Location(x=150.24002075195312, y=191.77003479003906, z=1.32), Rotation(yaw=-0.00018310546875)),
    16: Transform(Location(x=165.0900421142578, y=187.1199493408203, z=1.32), Rotation(yaw=-179.9996337890625)),
    17: Transform(Location(x=9.469999313354492, y=191.76998901367188, z=1.32), Rotation(yaw=-0.00018310546875)),
    18: Transform(Location(x=-3.679999828338623, y=172.42999267578125, z=1.32), Rotation(yaw=-89.99981689453125)),
    19: Transform(Location(x=21.9000244140625, y=187.9199981689453, z=1.32), Rotation(yaw=-179.9996337890625)),
    20: Transform(Location(x=14.200018882751465, y=191.76998901367188, z=1.32), Rotation(yaw=-0.00018310546875)),
    21: Transform(Location(x=27.72001838684082, y=187.9199981689453, z=1.32), Rotation(yaw=-179.9996337890625)),
    22: Transform(Location(x=63.34002685546875, y=191.76998901367188, z=1.32), Rotation(yaw=-0.00018310546875)),
    23: Transform(Location(x=75.4100341796875, y=187.9199981689453, z=1.32), Rotation(yaw=-179.9996337890625)),
    24: Transform(Location(x=92.34001922607422, y=191.76998901367188, z=1.32), Rotation(yaw=-0.00018310546875)),
    25: Transform(Location(x=92.34004974365234, y=187.9199981689453, z=1.32), Rotation(yaw=-179.9996337890625)),
    26: Transform(Location(x=162.02005004882812, y=191.77003479003906, z=1.32), Rotation(yaw=-0.00018310546875)),
    27: Transform(Location(x=181.800048828125, y=187.9199981689453, z=1.32), Rotation(yaw=-179.9996337890625)),
    28: Transform(Location(x=104.59001922607422, y=191.77003479003906, z=1.32), Rotation(yaw=-0.00018310546875)),
    29: Transform(Location(x=-7.529999732971191, y=208.9199981689453, z=1.32), Rotation(yaw=89.99995422363281)),
    30: Transform(Location(x=117.93003845214844, y=187.91995239257812, z=1.32), Rotation(yaw=-179.9996337890625)),
    31: Transform(Location(x=151.30001831054688, y=241.280029296875, z=1.32), Rotation(yaw=-0.00018310546875)),
    32: Transform(Location(x=162.92002868652344, y=237.42996215820312, z=1.32), Rotation(yaw=-179.9996337890625)),
    33: Transform(Location(x=59.71002960205078, y=241.27999877929688, z=1.32), Rotation(yaw=-0.00018310546875)),
    34: Transform(Location(x=71.0400390625, y=237.42999267578125, z=1.32), Rotation(yaw=-179.9996337890625)),
    35: Transform(Location(x=88.71001434326172, y=241.27999877929688, z=1.32), Rotation(yaw=-0.00018310546875)),
    36: Transform(Location(x=88.71004486083984, y=237.42999267578125, z=1.32), Rotation(yaw=-179.9996337890625)),
    37: Transform(Location(x=162.7600555419922, y=241.280029296875, z=1.32), Rotation(yaw=-0.00018310546875)),
    38: Transform(Location(x=174.33004760742188, y=237.42999267578125, z=1.32), Rotation(yaw=-179.9996337890625)),
    39: Transform(Location(x=104.30001831054688, y=241.280029296875, z=1.32), Rotation(yaw=-0.00018310546875)),
    40: Transform(Location(x=-3.679999828338623, y=219.4099884033203, z=1.32), Rotation(yaw=-89.99981689453125)),
    41: Transform(Location(x=118.77003479003906, y=237.42996215820312, z=1.32), Rotation(yaw=-179.9996337890625)),
    42: Transform(Location(x=9.530024528503418, y=302.57000732421875, z=1.32), Rotation(yaw=-179.9996337890625)),
    43: Transform(Location(x=14.010019302368164, y=306.41998291015625, z=1.32), Rotation(yaw=-0.00018310546875)),
    44: Transform(Location(x=26.940019607543945, y=302.57000732421875, z=1.32), Rotation(yaw=-179.9996337890625)),
    45: Transform(Location(x=59.60002899169922, y=306.41998291015625, z=1.32), Rotation(yaw=-0.00018310546875)),
    46: Transform(Location(x=71.53003692626953, y=302.57000732421875, z=1.32), Rotation(yaw=-179.9996337890625)),
    47: Transform(Location(x=88.83001708984375, y=306.41998291015625, z=1.32), Rotation(yaw=-0.00018310546875)),
    48: Transform(Location(x=88.83004760742188, y=302.57000732421875, z=1.32), Rotation(yaw=-179.9996337890625)),
    49: Transform(Location(x=178.29005432128906, y=306.4200439453125, z=1.32), Rotation(yaw=-0.00018310546875)),
    50: Transform(Location(x=-7.529999732971191, y=288.2200012207031, z=1.32), Rotation(yaw=89.99995422363281)),
    51: Transform(Location(x=178.29005432128906, y=302.57000732421875, z=1.32), Rotation(yaw=-179.9996337890625)),
    52: Transform(Location(x=136.11001586914062, y=306.4200439453125, z=1.32), Rotation(yaw=-0.00018310546875)),
    53: Transform(Location(x=136.1100311279297, y=302.5699462890625, z=1.32), Rotation(yaw=-179.9996337890625)),
    54: Transform(Location(x=1.5399999618530273, y=109.39999389648438, z=1.32), Rotation(yaw=-0.00018310546875)),
    55: Transform(Location(x=1.5400243997573853, y=105.54998779296875, z=1.32), Rotation(yaw=-179.9996337890625)),
    56: Transform(Location(x=21.420019149780273, y=109.39999389648438, z=1.32), Rotation(yaw=-0.00018310546875)),
    57: Transform(Location(x=25.530019760131836, y=105.54998779296875, z=1.32), Rotation(yaw=-179.9996337890625)),
    58: Transform(Location(x=55.41002655029297, y=109.39998626708984, z=1.32), Rotation(yaw=-0.00018310546875)),
    59: Transform(Location(x=55.410037994384766, y=105.54998779296875, z=1.32), Rotation(yaw=-179.9996337890625)),
    60: Transform(Location(x=84.41001892089844, y=109.29999542236328, z=1.32), Rotation(yaw=-0.00018310546875)),
    61: Transform(Location(x=-3.679999828338623, y=288.2200012207031, z=1.32), Rotation(yaw=-89.99981689453125)),
    62: Transform(Location(x=84.41004943847656, y=105.55001068115234, z=1.32), Rotation(yaw=-179.9996337890625)),
    63: Transform(Location(x=173.87005615234375, y=109.40003967285156, z=1.32), Rotation(yaw=-0.00018310546875)),
    64: Transform(Location(x=173.87005615234375, y=105.55001068115234, z=1.32), Rotation(yaw=-179.9996337890625)),
    65: Transform(Location(x=131.6900177001953, y=109.40003967285156, z=1.32), Rotation(yaw=-0.00018310546875)),
    66: Transform(Location(x=131.69003295898438, y=105.54996490478516, z=1.32), Rotation(yaw=-179.9996337890625)),
    67: Transform(Location(x=189.92999267578125, y=121.20999908447266, z=1.32), Rotation(yaw=89.99995422363281)),
    68: Transform(Location(x=193.77999877929688, y=121.20999908447266, z=1.32), Rotation(yaw=-89.99981689453125)),
    69: Transform(Location(x=189.92999267578125, y=142.19000244140625, z=1.32), Rotation(yaw=89.99995422363281)),
    70: Transform(Location(x=193.77999877929688, y=142.19000244140625, z=1.32), Rotation(yaw=-89.99981689453125)),
    71: Transform(Location(x=189.92999267578125, y=160.5800018310547, z=1.32), Rotation(yaw=89.99995422363281)),
    72: Transform(Location(x=-7.529999732971191, y=251.36000061035156, z=1.32), Rotation(yaw=89.99995422363281)),
    73: Transform(Location(x=193.77999877929688, y=171.2899932861328, z=1.32), Rotation(yaw=-89.99981689453125)),
    74: Transform(Location(x=189.92999267578125, y=208.11000061035156, z=1.32), Rotation(yaw=89.99995422363281)),
    75: Transform(Location(x=193.77999877929688, y=218.7899932861328, z=1.32), Rotation(yaw=-89.99981689453125)),
    76: Transform(Location(x=189.92999267578125, y=293.5400085449219, z=1.32), Rotation(yaw=89.99995422363281)),
    77: Transform(Location(x=193.77999877929688, y=293.5400085449219, z=1.32), Rotation(yaw=-89.99981689453125)),
    78: Transform(Location(x=189.92999267578125, y=252.3300018310547, z=1.32), Rotation(yaw=89.99995422363281)),
    79: Transform(Location(x=193.77999877929688, y=266.42999267578125, z=1.32), Rotation(yaw=-89.99981689453125)),
    80: Transform(Location(x=-7.529999732971191, y=121.20999908447266, z=1.32), Rotation(yaw=89.99995422363281)),
    81: Transform(Location(x=-3.679999828338623, y=121.20999908447266, z=1.32), Rotation(yaw=-89.99981689453125)),
    82: Transform(Location(x=-7.529999732971191, y=142.19000244140625, z=1.32), Rotation(yaw=89.99995422363281))
}

WAYPOINT_DICT_Town03 = {
    0 : Transform(Location(x=-6.446170, y=-79.055023, z=1.842997), Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000)) ,
    1 : Transform(Location(x=65.516594, y=7.808423, z=1.843102), Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000)) ,
    2 : Transform(Location(x=65.568863, y=4.308813, z=1.843102), Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000)) ,
    3 : Transform(Location(x=-2.419357, y=204.005676, z=1.843104), Rotation(pitch=0.000000, yaw=-0.142975, roll=0.000000)) ,
    4 : Transform(Location(x=-2.410624, y=207.505676, z=1.843104), Rotation(pitch=0.000000, yaw=-0.142975, roll=0.000000)) ,
    5 : Transform(Location(x=-20.033100, y=204.005676, z=1.843238), Rotation(pitch=0.000000, yaw=-0.142975, roll=0.000000)) ,
    6 : Transform(Location(x=-20.024366, y=207.505676, z=1.843238), Rotation(pitch=0.000000, yaw=-0.142975, roll=0.000000)) ,
    7 : Transform(Location(x=-2.948311, y=-78.932617, z=1.842997), Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000)) ,
    8 : Transform(Location(x=-55.049702, y=0.545833, z=1.843102), Rotation(pitch=0.000000, yaw=1.666941, roll=0.000000)) ,
    9 : Transform(Location(x=-117.493454, y=-3.221493, z=1.843102), Rotation(pitch=0.000000, yaw=-179.705399, roll=0.000000)) ,
    10 : Transform(Location(x=-77.887169, y=21.444204, z=1.805250), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    11 : Transform(Location(x=-74.387177, y=21.434658, z=1.805250), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    12 : Transform(Location(x=-77.887169, y=40.306927, z=1.805265), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    13 : Transform(Location(x=-74.387177, y=40.297382, z=1.805265), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    14 : Transform(Location(x=-77.887169, y=57.541164, z=1.805267), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    15 : Transform(Location(x=-74.387177, y=57.531620, z=1.805267), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    16 : Transform(Location(x=-77.887169, y=75.658577, z=1.805257), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    17 : Transform(Location(x=140.592041, y=5.371150, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    18 : Transform(Location(x=140.539780, y=8.870760, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    19 : Transform(Location(x=120.127945, y=5.371150, z=1.843097), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    20 : Transform(Location(x=120.075668, y=8.870760, z=1.843097), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    21 : Transform(Location(x=192.449112, y=6.308762, z=1.843052), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    22 : Transform(Location(x=192.396835, y=9.808371, z=1.843052), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    23 : Transform(Location(x=174.485291, y=6.308762, z=1.843052), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    24 : Transform(Location(x=174.432999, y=9.808371, z=1.843052), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    25 : Transform(Location(x=-74.387177, y=75.649033, z=1.805257), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    26 : Transform(Location(x=-77.887169, y=99.725639, z=1.805257), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    27 : Transform(Location(x=-74.387177, y=99.716110, z=1.805257), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    28 : Transform(Location(x=-88.154869, y=108.906883, z=1.843102), Rotation(pitch=0.000000, yaw=89.787704, roll=0.000000)) ,
    29 : Transform(Location(x=-84.654892, y=108.895164, z=1.843102), Rotation(pitch=0.000000, yaw=89.787674, roll=0.000000)) ,
    30 : Transform(Location(x=-88.167549, y=36.942276, z=1.842973), Rotation(pitch=0.000000, yaw=89.787704, roll=0.000000)) ,
    31 : Transform(Location(x=-84.767197, y=37.030930, z=1.842973), Rotation(pitch=0.000000, yaw=89.787674, roll=0.000000)) ,
    32 : Transform(Location(x=-95.444824, y=136.115875, z=1.843221), Rotation(pitch=0.000000, yaw=-0.597992, roll=0.000000)) ,
    33 : Transform(Location(x=-107.510582, y=136.115875, z=1.843285), Rotation(pitch=0.000000, yaw=-0.597992, roll=0.000000)) ,
    34 : Transform(Location(x=-117.501511, y=136.115875, z=1.843284), Rotation(pitch=0.000000, yaw=-0.597992, roll=0.000000)) ,
    35 : Transform(Location(x=-66.303299, y=131.964951, z=1.843102), Rotation(pitch=0.000000, yaw=176.631271, roll=0.000000)) ,
    36 : Transform(Location(x=-53.170971, y=131.964951, z=1.843102), Rotation(pitch=0.000000, yaw=176.631271, roll=0.000000)) ,
    37 : Transform(Location(x=-5.924474, y=111.378662, z=1.843097), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    38 : Transform(Location(x=-9.424404, y=111.396446, z=1.843099), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    39 : Transform(Location(x=-5.941558, y=87.476509, z=1.843094), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    40 : Transform(Location(x=-9.441488, y=87.494293, z=1.843097), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    41 : Transform(Location(x=-6.448509, y=63.542019, z=1.843089), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    42 : Transform(Location(x=-9.948440, y=63.559803, z=1.843092), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    43 : Transform(Location(x=240.112717, y=119.093483, z=1.843102), Rotation(pitch=0.000000, yaw=-88.605339, roll=0.000000)) ,
    44 : Transform(Location(x=-149.063583, y=91.171509, z=1.843102), Rotation(pitch=0.000000, yaw=89.622032, roll=0.000000)) ,
    45 : Transform(Location(x=243.611694, y=119.178131, z=1.843102), Rotation(pitch=0.000000, yaw=-88.606743, roll=0.000000)) ,
    46 : Transform(Location(x=9.284539, y=-105.343163, z=1.843105), Rotation(pitch=0.000000, yaw=-88.586418, roll=0.000000)) ,
    47 : Transform(Location(x=6.078289, y=-105.429489, z=1.843106), Rotation(pitch=0.000000, yaw=-88.876099, roll=0.000000)) ,
    48 : Transform(Location(x=-24.304346, y=-135.292160, z=0.995580), Rotation(pitch=1.217140, yaw=1.227265, roll=0.000000)) ,
    49 : Transform(Location(x=-36.630997, y=-194.923615, z=1.843102), Rotation(pitch=0.000000, yaw=1.439547, roll=0.000000)) ,
    50 : Transform(Location(x=-36.542839, y=-198.422501, z=1.843102), Rotation(pitch=0.000000, yaw=1.439544, roll=0.000000)) ,
    51 : Transform(Location(x=120.364235, y=-190.399994, z=1.843102), Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000)) ,
    52 : Transform(Location(x=120.364235, y=-193.899994, z=1.843102), Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000)) ,
    53 : Transform(Location(x=151.030304, y=-163.982391, z=4.947135), Rotation(pitch=0.000000, yaw=90.996483, roll=0.000000)) ,
    54 : Transform(Location(x=149.958023, y=-102.336494, z=9.843105), Rotation(pitch=0.000000, yaw=90.996483, roll=0.000000)) ,
    55 : Transform(Location(x=-148.989471, y=-36.040218, z=1.843102), Rotation(pitch=0.000000, yaw=90.029823, roll=0.000000)) ,
    56 : Transform(Location(x=-78.124168, y=-95.038681, z=1.277069), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    57 : Transform(Location(x=-74.624176, y=-95.048233, z=1.277069), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    58 : Transform(Location(x=-77.985069, y=-44.039780, z=1.587068), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    59 : Transform(Location(x=-74.485085, y=-44.049332, z=1.587068), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    60 : Transform(Location(x=-84.956627, y=-33.578300, z=1.030320), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
    61 : Transform(Location(x=-88.456612, y=-33.568760, z=1.030320), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
    62 : Transform(Location(x=-85.040810, y=-64.377121, z=1.283939), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
    63 : Transform(Location(x=-88.540787, y=-64.367584, z=1.283939), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
    64 : Transform(Location(x=-85.146881, y=-103.275650, z=1.604257), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
    65 : Transform(Location(x=-88.646866, y=-103.266113, z=1.604257), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
    66 : Transform(Location(x=-148.970520, y=-72.440216, z=1.843102), Rotation(pitch=0.000000, yaw=90.029823, roll=0.000000)) ,
    67 : Transform(Location(x=-34.225540, y=131.243164, z=1.843102), Rotation(pitch=0.000000, yaw=178.703156, roll=0.000000)) ,
    68 : Transform(Location(x=-49.565033, y=135.091202, z=1.843102), Rotation(pitch=0.000000, yaw=-1.296814, roll=0.000000)) ,
    69 : Transform(Location(x=-30.669865, y=134.663498, z=1.843102), Rotation(pitch=0.000000, yaw=-1.296814, roll=0.000000)) ,
    70 : Transform(Location(x=31.282980, y=-207.221069, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    71 : Transform(Location(x=31.195051, y=-203.722168, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    72 : Transform(Location(x=54.040394, y=-192.644958, z=1.843102), Rotation(pitch=0.000000, yaw=1.439547, roll=0.000000)) ,
    73 : Transform(Location(x=54.128548, y=-196.143845, z=1.843102), Rotation(pitch=0.000000, yaw=1.439544, roll=0.000000)) ,
    74 : Transform(Location(x=117.401649, y=-205.057190, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    75 : Transform(Location(x=117.313583, y=-201.558304, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    76 : Transform(Location(x=154.557892, y=-165.535995, z=3.895221), Rotation(pitch=-2.322660, yaw=-89.003517, roll=0.000000)) ,
    77 : Transform(Location(x=-145.519318, y=21.290268, z=1.843102), Rotation(pitch=0.000000, yaw=-89.970146, roll=0.000000)) ,
    78 : Transform(Location(x=153.459579, y=-102.397400, z=9.843105), Rotation(pitch=0.000000, yaw=-89.003517, roll=0.000000)) ,
    79 : Transform(Location(x=124.949852, y=-132.094757, z=9.843105), Rotation(pitch=0.000000, yaw=1.227265, roll=0.000000)) ,
    80 : Transform(Location(x=110.063339, y=-135.914536, z=9.843105), Rotation(pitch=0.000000, yaw=-178.772690, roll=0.000000)) ,
    81 : Transform(Location(x=84.183815, y=-105.191704, z=9.843105), Rotation(pitch=0.000000, yaw=-87.975883, roll=0.000000)) ,
    82 : Transform(Location(x=115.532143, y=-76.355240, z=9.843105), Rotation(pitch=0.000000, yaw=-178.490875, roll=0.000000)) ,
    83 : Transform(Location(x=230.824677, y=32.609890, z=1.751008), Rotation(pitch=0.167927, yaw=91.393204, roll=0.000000)) ,
    84 : Transform(Location(x=234.323654, y=32.695038, z=1.751008), Rotation(pitch=0.167927, yaw=91.393204, roll=0.000000)) ,
    85 : Transform(Location(x=-54.750820, y=-2.898878, z=1.843102), Rotation(pitch=0.000000, yaw=-179.705383, roll=0.000000)) ,
    86 : Transform(Location(x=-149.024933, y=31.959774, z=1.843102), Rotation(pitch=0.000000, yaw=90.029823, roll=0.000000)) ,
    87 : Transform(Location(x=131.038910, y=62.489872, z=1.843102), Rotation(pitch=0.000000, yaw=-0.147400, roll=0.000000)) ,
    88 : Transform(Location(x=124.700043, y=59.006218, z=1.843102), Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000)) ,
    89 : Transform(Location(x=204.702850, y=58.800388, z=1.843102), Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000)) ,
    90 : Transform(Location(x=-117.975456, y=33.972942, z=1.843102), Rotation(pitch=0.000000, yaw=134.676804, roll=0.000000)) ,
    91 : Transform(Location(x=-114.782623, y=35.864052, z=1.843102), Rotation(pitch=0.000000, yaw=134.676804, roll=0.000000)) ,
    92 : Transform(Location(x=125.359840, y=-135.586823, z=9.843105), Rotation(pitch=0.000000, yaw=-178.772690, roll=0.000000)) ,
    93 : Transform(Location(x=109.653336, y=-132.422455, z=9.843105), Rotation(pitch=0.000000, yaw=1.227265, roll=0.000000)) ,
    94 : Transform(Location(x=34.646507, y=-193.132416, z=1.843102), Rotation(pitch=0.000000, yaw=1.439547, roll=0.000000)) ,
    95 : Transform(Location(x=34.734661, y=-196.631302, z=1.843102), Rotation(pitch=0.000000, yaw=1.439544, roll=0.000000)) ,
    96 : Transform(Location(x=55.675301, y=-206.608185, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    97 : Transform(Location(x=55.587372, y=-203.109268, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    98 : Transform(Location(x=-118.079285, y=0.275539, z=1.843102), Rotation(pitch=0.000000, yaw=0.294608, roll=0.000000)) ,
    99 : Transform(Location(x=199.419632, y=-5.502129, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    100 : Transform(Location(x=199.367340, y=-2.002518, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    101 : Transform(Location(x=172.722595, y=-5.900934, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    102 : Transform(Location(x=172.670303, y=-2.401323, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    103 : Transform(Location(x=127.130280, y=-6.581954, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    104 : Transform(Location(x=127.077606, y=-3.082350, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    105 : Transform(Location(x=106.032616, y=-6.897115, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    106 : Transform(Location(x=105.979942, y=-3.397510, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    107 : Transform(Location(x=-87.967049, y=91.041870, z=1.842973), Rotation(pitch=0.000000, yaw=89.787704, roll=0.000000)) ,
    108 : Transform(Location(x=-84.566696, y=91.130524, z=1.842973), Rotation(pitch=0.000000, yaw=89.787674, roll=0.000000)) ,
    109 : Transform(Location(x=-88.056679, y=66.842079, z=1.842973), Rotation(pitch=0.000000, yaw=89.787704, roll=0.000000)) ,
    110 : Transform(Location(x=-84.656326, y=66.930733, z=1.842973), Rotation(pitch=0.000000, yaw=89.787674, roll=0.000000)) ,
    111 : Transform(Location(x=80.517311, y=-104.026123, z=9.891966), Rotation(pitch=-0.190398, yaw=92.024086, roll=0.000000)) ,
    112 : Transform(Location(x=4.997698, y=55.387558, z=1.843089), Rotation(pitch=0.000000, yaw=-88.891235, roll=0.000000)) ,
    113 : Transform(Location(x=1.498355, y=55.319832, z=1.843089), Rotation(pitch=0.000000, yaw=-88.891235, roll=0.000000)) ,
    114 : Transform(Location(x=4.672596, y=69.624977, z=1.843080), Rotation(pitch=0.000000, yaw=-88.891235, roll=0.000000)) ,
    115 : Transform(Location(x=239.623291, y=98.075317, z=1.843102), Rotation(pitch=0.000000, yaw=-88.605339, roll=0.000000)) ,
    116 : Transform(Location(x=243.122284, y=98.159966, z=1.843102), Rotation(pitch=0.000000, yaw=-88.606743, roll=0.000000)) ,
    117 : Transform(Location(x=245.194214, y=17.595139, z=1.843102), Rotation(pitch=0.000000, yaw=-88.606743, roll=0.000000)) ,
    118 : Transform(Location(x=1.173256, y=69.557251, z=1.843080), Rotation(pitch=0.000000, yaw=-88.891235, roll=0.000000)) ,
    119 : Transform(Location(x=241.695251, y=17.510042, z=1.843102), Rotation(pitch=0.000000, yaw=-88.606743, roll=0.000000)) ,
    120 : Transform(Location(x=57.014038, y=-7.778184, z=1.843103), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    121 : Transform(Location(x=56.961758, y=-4.278575, z=1.843103), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    122 : Transform(Location(x=-6.446170, y=-61.717133, z=1.843102), Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000)) ,
    123 : Transform(Location(x=-2.948311, y=-61.594730, z=1.843102), Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000)) ,
    124 : Transform(Location(x=227.309662, y=-5.085508, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144211, roll=0.000000)) ,
    125 : Transform(Location(x=227.257401, y=-1.585898, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144211, roll=0.000000)) ,
    126 : Transform(Location(x=-88.710991, y=-126.865234, z=1.798586), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
    127 : Transform(Location(x=-74.546471, y=-148.432693, z=1.843102), Rotation(pitch=0.000000, yaw=-88.434715, roll=0.000000)) ,
    128 : Transform(Location(x=-78.045143, y=-148.528915, z=1.843102), Rotation(pitch=0.000000, yaw=-88.434692, roll=0.000000)) ,
    129 : Transform(Location(x=-2.798443, y=-189.813995, z=1.843102), Rotation(pitch=0.000000, yaw=91.413536, roll=0.000000)) ,
    130 : Transform(Location(x=-6.682256, y=-204.674316, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    131 : Transform(Location(x=0.700499, y=-189.727951, z=1.843102), Rotation(pitch=0.000000, yaw=91.413536, roll=0.000000)) ,
    132 : Transform(Location(x=15.552521, y=-193.612244, z=1.843102), Rotation(pitch=0.000000, yaw=1.439547, roll=0.000000)) ,
    133 : Transform(Location(x=-6.594831, y=-208.173233, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    134 : Transform(Location(x=15.640681, y=-197.111130, z=1.843102), Rotation(pitch=0.000000, yaw=1.439544, roll=0.000000)) ,
    135 : Transform(Location(x=82.695610, y=-184.773468, z=1.728263), Rotation(pitch=-0.396629, yaw=90.153564, roll=0.000000)) ,
    136 : Transform(Location(x=-26.262718, y=-7.955658, z=1.843102), Rotation(pitch=0.000000, yaw=143.077484, roll=0.000000)) ,
    137 : Transform(Location(x=94.797409, y=-191.620804, z=1.843102), Rotation(pitch=0.000000, yaw=1.439547, roll=0.000000)) ,
    138 : Transform(Location(x=74.169441, y=-206.143616, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    139 : Transform(Location(x=74.081512, y=-202.644699, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    140 : Transform(Location(x=94.885727, y=-195.119690, z=1.843102), Rotation(pitch=0.000000, yaw=1.439544, roll=0.000000)) ,
    141 : Transform(Location(x=163.864227, y=-193.899994, z=1.843102), Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000)) ,
    142 : Transform(Location(x=143.193497, y=-204.409012, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    143 : Transform(Location(x=143.105438, y=-200.910126, z=1.843102), Rotation(pitch=0.000000, yaw=-178.560471, roll=0.000000)) ,
    144 : Transform(Location(x=163.864227, y=-197.399994, z=1.843102), Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000)) ,
    145 : Transform(Location(x=80.874001, y=-11.542412, z=1.993637), Rotation(pitch=2.348458, yaw=-87.975861, roll=0.000000)) ,
    146 : Transform(Location(x=71.362877, y=-7.414977, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    147 : Transform(Location(x=86.206383, y=7.808423, z=1.843102), Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000)) ,
    148 : Transform(Location(x=86.258659, y=4.308813, z=1.843102), Rotation(pitch=0.000000, yaw=0.855823, roll=0.000000)) ,
    149 : Transform(Location(x=71.310326, y=-3.915372, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    150 : Transform(Location(x=81.363167, y=-124.476494, z=9.843105), Rotation(pitch=0.000000, yaw=92.024109, roll=0.000000)) ,
    151 : Transform(Location(x=73.193413, y=-136.704315, z=9.837398), Rotation(pitch=-0.647311, yaw=-178.772690, roll=0.000000)) ,
    152 : Transform(Location(x=85.590614, y=-144.997437, z=9.731206), Rotation(pitch=-4.122893, yaw=-87.975883, roll=0.000000)) ,
    153 : Transform(Location(x=93.756905, y=-132.762970, z=9.843105), Rotation(pitch=0.000000, yaw=1.227265, roll=0.000000)) ,
    154 : Transform(Location(x=-88.644112, y=144.550552, z=1.843102), Rotation(pitch=0.000000, yaw=89.787704, roll=0.000000)) ,
    155 : Transform(Location(x=-95.432480, y=132.628662, z=1.843102), Rotation(pitch=0.000000, yaw=178.703156, roll=0.000000)) ,
    156 : Transform(Location(x=-85.144127, y=144.538834, z=1.843102), Rotation(pitch=0.000000, yaw=89.787674, roll=0.000000)) ,
    157 : Transform(Location(x=-66.160751, y=135.466934, z=1.843102), Rotation(pitch=0.000000, yaw=-1.296814, roll=0.000000)) ,
    158 : Transform(Location(x=-77.526985, y=123.926384, z=1.883463), Rotation(pitch=0.395058, yaw=-90.156242, roll=0.000000)) ,
    159 : Transform(Location(x=-74.027000, y=123.916832, z=1.883463), Rotation(pitch=0.395058, yaw=-90.156242, roll=0.000000)) ,
    160 : Transform(Location(x=-9.437694, y=143.109421, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    161 : Transform(Location(x=-5.937763, y=143.087265, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    162 : Transform(Location(x=1.930856, y=122.298492, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    163 : Transform(Location(x=5.430785, y=122.276344, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    164 : Transform(Location(x=13.994347, y=134.392639, z=1.843102), Rotation(pitch=0.000000, yaw=-0.816833, roll=0.000000)) ,
    165 : Transform(Location(x=-16.630026, y=130.844788, z=1.843102), Rotation(pitch=0.000000, yaw=178.703156, roll=0.000000)) ,
    166 : Transform(Location(x=-149.063583, y=107.705582, z=1.843102), Rotation(pitch=0.000000, yaw=89.622032, roll=0.000000)) ,
    167 : Transform(Location(x=-13.384322, y=193.564453, z=1.843102), Rotation(pitch=0.000000, yaw=179.856995, roll=0.000000)) ,
    168 : Transform(Location(x=-13.375589, y=197.064423, z=1.843102), Rotation(pitch=0.000000, yaw=179.856995, roll=0.000000)) ,
    169 : Transform(Location(x=10.181385, y=204.005676, z=1.843102), Rotation(pitch=0.000000, yaw=-0.142975, roll=0.000000)) ,
    170 : Transform(Location(x=10.190118, y=207.505676, z=1.843102), Rotation(pitch=0.000000, yaw=-0.142975, roll=0.000000)) ,
    171 : Transform(Location(x=2.354272, y=189.215149, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    172 : Transform(Location(x=5.854201, y=189.192886, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    173 : Transform(Location(x=-145.560196, y=99.700966, z=1.843102), Rotation(pitch=0.000000, yaw=-89.970146, roll=0.000000)) ,
    174 : Transform(Location(x=151.357269, y=-182.784271, z=1.978469), Rotation(pitch=3.294118, yaw=90.996483, roll=0.000000)) ,
    175 : Transform(Location(x=97.278854, y=63.117493, z=1.843102), Rotation(pitch=0.000000, yaw=-10.416641, roll=0.000000)) ,
    176 : Transform(Location(x=-149.047928, y=76.143700, z=1.843102), Rotation(pitch=0.000000, yaw=90.029823, roll=0.000000)) ,
    177 : Transform(Location(x=2.294559, y=179.778244, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    178 : Transform(Location(x=5.794489, y=179.756088, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    179 : Transform(Location(x=-9.205211, y=179.851013, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    180 : Transform(Location(x=-5.705280, y=179.828857, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    181 : Transform(Location(x=142.528610, y=-6.351933, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    182 : Transform(Location(x=151.938690, y=-14.957595, z=1.968941), Rotation(pitch=2.312470, yaw=-89.003517, roll=0.000000)) ,
    183 : Transform(Location(x=-145.535782, y=52.828339, z=1.843102), Rotation(pitch=0.000000, yaw=-89.970146, roll=0.000000)) ,
    184 : Transform(Location(x=142.475937, y=-2.852329, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    185 : Transform(Location(x=157.377090, y=5.371150, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    186 : Transform(Location(x=157.324814, y=8.870760, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    187 : Transform(Location(x=6.316784, y=-25.297159, z=1.843102), Rotation(pitch=0.000000, yaw=-108.033249, roll=0.000000)) ,
    188 : Transform(Location(x=9.644855, y=-26.380650, z=1.843102), Rotation(pitch=0.000000, yaw=-108.033249, roll=0.000000)) ,
    189 : Transform(Location(x=220.145645, y=6.308762, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    190 : Transform(Location(x=-136.979019, y=0.178358, z=1.843102), Rotation(pitch=0.000000, yaw=0.294608, roll=0.000000)) ,
    191 : Transform(Location(x=220.093353, y=9.808371, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    192 : Transform(Location(x=220.317398, y=-5.189955, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    193 : Transform(Location(x=220.265106, y=-1.690346, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    194 : Transform(Location(x=-59.603539, y=187.931732, z=1.843102), Rotation(pitch=0.000000, yaw=-144.446411, roll=0.000000)) ,
    195 : Transform(Location(x=-61.638359, y=190.779465, z=1.843102), Rotation(pitch=0.000000, yaw=-144.447281, roll=0.000000)) ,
    196 : Transform(Location(x=-50.591991, y=203.186874, z=1.843102), Rotation(pitch=0.000000, yaw=12.333570, roll=0.000000)) ,
    197 : Transform(Location(x=-51.339600, y=206.606094, z=1.843102), Rotation(pitch=0.000000, yaw=12.333570, roll=0.000000)) ,
    198 : Transform(Location(x=-149.010956, y=5.159778, z=1.843102), Rotation(pitch=0.000000, yaw=90.029823, roll=0.000000)) ,
    199 : Transform(Location(x=245.865112, y=-9.996704, z=1.843102), Rotation(pitch=0.000000, yaw=-88.606743, roll=0.000000)) ,
    200 : Transform(Location(x=-145.503876, y=-8.409729, z=1.843102), Rotation(pitch=0.000000, yaw=-89.970146, roll=0.000000)) ,
    201 : Transform(Location(x=104.794487, y=62.557415, z=1.843102), Rotation(pitch=0.000000, yaw=-0.147430, roll=0.000000)) ,
    202 : Transform(Location(x=-9.261773, y=170.911835, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    203 : Transform(Location(x=-5.761843, y=170.889679, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    204 : Transform(Location(x=2.184859, y=162.441132, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    205 : Transform(Location(x=242.366150, y=-10.081800, z=1.843102), Rotation(pitch=0.000000, yaw=-88.606743, roll=0.000000)) ,
    206 : Transform(Location(x=5.684789, y=162.419006, z=1.843102), Rotation(pitch=0.000000, yaw=-90.362541, roll=0.000000)) ,
    207 : Transform(Location(x=-13.931132, y=168.728027, z=1.843102), Rotation(pitch=0.000000, yaw=168.900085, roll=0.000000)) ,
    208 : Transform(Location(x=79.494911, y=-71.614845, z=9.842793), Rotation(pitch=-0.190398, yaw=92.024086, roll=0.000000)) ,
    209 : Transform(Location(x=83.276054, y=-79.507767, z=9.843105), Rotation(pitch=0.000000, yaw=-87.975883, roll=0.000000)) ,
    210 : Transform(Location(x=4.926103, y=40.578606, z=1.843102), Rotation(pitch=0.000000, yaw=-88.891235, roll=0.000000)) ,
    211 : Transform(Location(x=1.426758, y=40.510876, z=1.843102), Rotation(pitch=0.000000, yaw=-88.891235, roll=0.000000)) ,
    212 : Transform(Location(x=-10.073487, y=42.628510, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    213 : Transform(Location(x=-6.573557, y=42.606361, z=1.843102), Rotation(pitch=0.000000, yaw=89.637466, roll=0.000000)) ,
    214 : Transform(Location(x=144.755371, y=-135.171249, z=9.843105), Rotation(pitch=0.000000, yaw=-178.772690, roll=0.000000)) ,
    215 : Transform(Location(x=154.126831, y=-140.758759, z=9.804370), Rotation(pitch=-2.322660, yaw=-89.003517, roll=0.000000)) ,
    216 : Transform(Location(x=234.769882, y=14.350552, z=1.843102), Rotation(pitch=0.000000, yaw=91.393204, roll=0.000000)) ,
    217 : Transform(Location(x=150.365875, y=-125.786667, z=9.843105), Rotation(pitch=0.000000, yaw=90.996483, roll=0.000000)) ,
    218 : Transform(Location(x=-6.446170, y=-42.193752, z=1.843102), Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000)) ,
    219 : Transform(Location(x=-2.948311, y=-42.071350, z=1.843102), Rotation(pitch=0.000000, yaw=92.004189, roll=0.000000)) ,
    220 : Transform(Location(x=7.603518, y=-43.829613, z=1.843102), Rotation(pitch=0.000000, yaw=-88.586418, roll=0.000000)) ,
    221 : Transform(Location(x=4.104583, y=-43.915955, z=1.843102), Rotation(pitch=0.000000, yaw=-88.586418, roll=0.000000)) ,
    222 : Transform(Location(x=240.817612, y=53.589058, z=1.843102), Rotation(pitch=0.000000, yaw=-88.605339, roll=0.000000)) ,
    223 : Transform(Location(x=231.270920, y=14.265452, z=1.843102), Rotation(pitch=0.000000, yaw=91.393204, roll=0.000000)) ,
    224 : Transform(Location(x=225.702789, y=58.746357, z=1.843102), Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000)) ,
    225 : Transform(Location(x=229.973785, y=67.599396, z=1.853589), Rotation(pitch=0.167927, yaw=91.393204, roll=0.000000)) ,
    226 : Transform(Location(x=233.472748, y=67.684540, z=1.853589), Rotation(pitch=0.167927, yaw=91.393204, roll=0.000000)) ,
    227 : Transform(Location(x=244.316589, y=53.673721, z=1.843102), Rotation(pitch=0.000000, yaw=-88.606743, roll=0.000000)) ,
    228 : Transform(Location(x=-42.350990, y=-2.835118, z=1.843102), Rotation(pitch=0.000000, yaw=-179.705383, roll=0.000000)) ,
    229 : Transform(Location(x=-40.411064, y=0.685463, z=1.843102), Rotation(pitch=0.000000, yaw=1.666941, roll=0.000000)) ,
    230 : Transform(Location(x=-11.748788, y=26.601166, z=1.843102), Rotation(pitch=0.000000, yaw=78.624901, roll=0.000000)) ,
    231 : Transform(Location(x=-8.317538, y=25.910858, z=1.843102), Rotation(pitch=0.000000, yaw=78.624901, roll=0.000000)) ,
    232 : Transform(Location(x=-21.252514, y=11.105013, z=1.843102), Rotation(pitch=0.000000, yaw=61.806828, roll=0.000000)) ,
    233 : Transform(Location(x=-18.167728, y=9.451503, z=1.843102), Rotation(pitch=0.000000, yaw=61.807007, roll=0.000000)) ,
    234 : Transform(Location(x=143.722351, y=-75.612572, z=9.843105), Rotation(pitch=0.000000, yaw=-178.490875, roll=0.000000)) ,
    235 : Transform(Location(x=149.391037, y=-69.741432, z=9.843105), Rotation(pitch=0.000000, yaw=90.996483, roll=0.000000)) ,
    236 : Transform(Location(x=153.030029, y=-77.701157, z=9.843105), Rotation(pitch=0.000000, yaw=-89.003517, roll=0.000000)) ,
    237 : Transform(Location(x=10.139042, y=-146.582535, z=1.843102), Rotation(pitch=0.000000, yaw=-88.586418, roll=0.000000)) ,
    238 : Transform(Location(x=6.640107, y=-146.668869, z=1.843102), Rotation(pitch=0.000000, yaw=-88.586418, roll=0.000000)) ,
    239 : Transform(Location(x=-0.865857, y=-126.250847, z=1.843102), Rotation(pitch=0.000000, yaw=91.413536, roll=0.000000)) ,
    240 : Transform(Location(x=-4.364792, y=-126.337181, z=1.843102), Rotation(pitch=0.000000, yaw=91.413536, roll=0.000000)) ,
    241 : Transform(Location(x=-11.102114, y=-138.510193, z=1.843102), Rotation(pitch=0.000000, yaw=-178.772690, roll=0.000000)) ,
    242 : Transform(Location(x=16.876915, y=-134.409973, z=1.870730), Rotation(pitch=1.217140, yaw=1.227265, roll=0.000000)) ,
    243 : Transform(Location(x=26.509409, y=7.425340, z=1.843102), Rotation(pitch=0.000000, yaw=-13.668415, roll=0.000000)) ,
    244 : Transform(Location(x=25.682356, y=4.024460, z=1.843102), Rotation(pitch=0.000000, yaw=-13.668415, roll=0.000000)) ,
    245 : Transform(Location(x=11.530827, y=16.089361, z=1.843102), Rotation(pitch=0.000000, yaw=-38.797035, roll=0.000000)) ,
    246 : Transform(Location(x=13.723803, y=18.817156, z=1.843102), Rotation(pitch=0.000000, yaw=-38.797035, roll=0.000000)) ,
    247 : Transform(Location(x=42.648182, y=-7.843905, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    248 : Transform(Location(x=42.595901, y=-4.344296, z=1.843102), Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000)) ,
    249 : Transform(Location(x=44.476276, y=3.684685, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    250 : Transform(Location(x=44.424007, y=7.184294, z=1.843102), Rotation(pitch=0.000000, yaw=0.855804, roll=0.000000)) ,
    251 : Transform(Location(x=161.899918, y=58.910496, z=1.843102), Rotation(pitch=0.000000, yaw=179.852554, roll=0.000000)) ,
    252 : Transform(Location(x=167.173080, y=71.142326, z=1.946012), Rotation(pitch=0.362574, yaw=89.989159, roll=0.000000)) ,
    253 : Transform(Location(x=175.938858, y=62.374382, z=1.843102), Rotation(pitch=0.000000, yaw=-0.147400, roll=0.000000)) ,
    254 : Transform(Location(x=-100.468056, y=16.266956, z=1.843102), Rotation(pitch=0.000000, yaw=134.676804, roll=0.000000)) ,
    255 : Transform(Location(x=-97.623611, y=19.079279, z=1.843102), Rotation(pitch=0.000000, yaw=134.676804, roll=0.000000)) ,
    256 : Transform(Location(x=-95.793716, y=-3.109917, z=1.843102), Rotation(pitch=0.000000, yaw=-179.705399, roll=0.000000)) ,
    257 : Transform(Location(x=-67.323837, y=0.536519, z=1.843102), Rotation(pitch=0.000000, yaw=0.294601, roll=0.000000)) ,
    258 : Transform(Location(x=-88.306274, y=21.530605, z=1.862779), Rotation(pitch=0.356591, yaw=89.843613, roll=-0.000000)) ,
    259 : Transform(Location(x=-84.806297, y=21.521088, z=1.862779), Rotation(pitch=0.356591, yaw=89.843742, roll=-0.000000)) ,
    260 : Transform(Location(x=-77.887169, y=-8.140573, z=1.805281), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    261 : Transform(Location(x=-74.387177, y=-8.150118, z=1.805281), Rotation(pitch=-0.348271, yaw=-90.156235, roll=-0.000000)) ,
    262 : Transform(Location(x=-67.154907, y=-136.210205, z=1.843102), Rotation(pitch=0.000000, yaw=1.227276, roll=0.000000)) ,
    263 : Transform(Location(x=-96.520119, y=-140.340103, z=1.983913), Rotation(pitch=0.848767, yaw=-178.772690, roll=0.000000)) ,
    264 : Transform(Location(x=-85.211014, y=-126.874771, z=1.798586), Rotation(pitch=-0.471802, yaw=89.843742, roll=0.000000)) ,
}


def paths_straight_Town01_train():

    # paths contain list of list
    # paths = [path_1, .. , path_n]
    # path_i = [start_transform, target_transform]

    paths = [
       [
           WAYPOINT_DICT_Town01[36],
           WAYPOINT_DICT_Town01[40]
       ],
       [
           WAYPOINT_DICT_Town01[7],
           WAYPOINT_DICT_Town01[3]
       ],
       [
           WAYPOINT_DICT_Town01[110],
           WAYPOINT_DICT_Town01[114]
       ],
       [
           WAYPOINT_DICT_Town01[68],
           WAYPOINT_DICT_Town01[50]
       ],
       [
           WAYPOINT_DICT_Town01[147],
           WAYPOINT_DICT_Town01[90]
       ],
       [
           WAYPOINT_DICT_Town01[33],
           WAYPOINT_DICT_Town01[87]
       ],
       [
           WAYPOINT_DICT_Town01[80],
           WAYPOINT_DICT_Town01[76]
       ],
       [
           WAYPOINT_DICT_Town01[45],
           WAYPOINT_DICT_Town01[49]
       ],
       [
           WAYPOINT_DICT_Town01[95],
           WAYPOINT_DICT_Town01[104]
       ],
       [
           WAYPOINT_DICT_Town01[20],
           WAYPOINT_DICT_Town01[107]
       ],
       [
           WAYPOINT_DICT_Town01[78],
           WAYPOINT_DICT_Town01[70]
       ],
       [
           WAYPOINT_DICT_Town01[68],
           WAYPOINT_DICT_Town01[44]
       ],
       [
           WAYPOINT_DICT_Town01[45],
           WAYPOINT_DICT_Town01[69]
       ]
    ]
    # paths = [
    #    [
    #        WAYPOINT_DICT_Town01[0],
    #        WAYPOINT_DICT_Town01[8]
    #    ],
    #    [
    #        WAYPOINT_DICT_Town01[130],
    #        WAYPOINT_DICT_Town01[123]
    #    ]
    # ]
    # paths = [
    #     [
    #         Transform(Location(x=12.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
    #         Transform(Location(x=76.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=92.11, y=316, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0)),
    #         Transform(Location(x=92.11, y=213, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=324.0, y=129.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0)),
    #         Transform(Location(x=108.5, y=199.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=102.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0)),
    #         Transform(Location(x=320.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0))
    #     ],
    #     [
    #         Transform(Location(x=140.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
    #         Transform(Location(x=320.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
    #     ]
    # ]

    return paths[0:1]

def paths_straight_Town01_test():
    paths = [
       [
           WAYPOINT_DICT_Town01[39],
           WAYPOINT_DICT_Town01[35]
       ],
       [
           WAYPOINT_DICT_Town01[0],
           WAYPOINT_DICT_Town01[4]
       ],
       [
           WAYPOINT_DICT_Town01[61],
           WAYPOINT_DICT_Town01[59]
       ],
       [
           WAYPOINT_DICT_Town01[55],
           WAYPOINT_DICT_Town01[44]
       ],
       [
           WAYPOINT_DICT_Town01[47],
           WAYPOINT_DICT_Town01[64]
       ],
       [
           WAYPOINT_DICT_Town01[26],
           WAYPOINT_DICT_Town01[19]
       ],
       [
           WAYPOINT_DICT_Town01[29],
           WAYPOINT_DICT_Town01[107]
       ],
       [
           WAYPOINT_DICT_Town01[84],
           WAYPOINT_DICT_Town01[34]
       ],
       [
           WAYPOINT_DICT_Town01[53],
           WAYPOINT_DICT_Town01[67]
       ],
       [
           WAYPOINT_DICT_Town01[22],
           WAYPOINT_DICT_Town01[17]
       ],
       [
           WAYPOINT_DICT_Town01[91],
           WAYPOINT_DICT_Town01[148]
       ],
       [
           WAYPOINT_DICT_Town01[95],
           WAYPOINT_DICT_Town01[102]
       ]
    ]

    return paths[0:1]
def paths_straight_Town01_dynamic():

    paths = [
        [
            WAYPOINT_DICT_Town01[84],
            WAYPOINT_DICT_Town01[40]
        ]
        # [
        #     WAYPOINT_DICT_Town01[96],
        #     WAYPOINT_DICT_Town01[140]
        # ]
    ]

    return paths

def paths_straight_crowded():
    paths = [
        [
            WAYPOINT_DICT_Town01[113],
            WAYPOINT_DICT_Town01[119]
        ]
    ]

    return paths

def paths_long_straight():
    paths = [
        [
            WAYPOINT_DICT_Town01[96],
            WAYPOINT_DICT_Town01[17]
            # Transform(Location(x=92.10997772216797, y=320.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
            # Transform(Location(x=92.11000061035156, y=10.820009231567383, z=1.32), Rotation(yaw=-90.00029754638672))
        ]
    ]

    return paths

def paths_long_straight_junction():
    paths = [
        [
            # WAYPOINT_DICT_Town01[96],
            # WAYPOINT_DICT_Town01[17]
            Transform(Location(x=92.10997772216797, y=320.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
            Transform(Location(x=92.11000061035156, y=10.820009231567383, z=1.32), Rotation(yaw=-90.00029754638672))
        ],
        [
            WAYPOINT_DICT_Town01[17],
            # WAYPOINT_DICT_Town01[53]
            WAYPOINT_DICT_Town01[56]
        ],
        [
            WAYPOINT_DICT_Town01[17],
            # WAYPOINT_DICT_Town01[46]
            WAYPOINT_DICT_Town01[44]
        ]
    ]

    return paths

def paths_crowded():

    paths = [
        [
            WAYPOINT_DICT_Town01[15],
            WAYPOINT_DICT_Town01[53]
        ]
    ]

    return paths

def paths_curved_town03():
    # paths = [
    #     [
    #         WAYPOINT_DICT_Town03[247],
    #         WAYPOINT_DICT_Town03[136]
    #     ]
    # ]
    paths = [
        [
            WAYPOINT_DICT_Town03[247],
            WAYPOINT_DICT_Town03[232]
        ]
    ]
    # paths = [
    #     [
    #         WAYPOINT_DICT_Town03[247],
    #         WAYPOINT_DICT_Town03[243]
    #     ]
    # ]
    return paths

def benchmark_paths_straight_Town01():

    paths = [
       [
           WAYPOINT_DICT_Town01[36],
           WAYPOINT_DICT_Town01[40]
       ],
       [
           WAYPOINT_DICT_Town01[39],
           WAYPOINT_DICT_Town01[35]
       ],
       [
           WAYPOINT_DICT_Town01[110],
           WAYPOINT_DICT_Town01[114]
       ],
       [
           WAYPOINT_DICT_Town01[7],
           WAYPOINT_DICT_Town01[3]
       ],
       [
           WAYPOINT_DICT_Town01[0],
           WAYPOINT_DICT_Town01[4]
       ],
       [
           WAYPOINT_DICT_Town01[68],
           WAYPOINT_DICT_Town01[50]
       ],
       [
           WAYPOINT_DICT_Town01[61],
           WAYPOINT_DICT_Town01[59]
       ],
       [
           WAYPOINT_DICT_Town01[47],
           WAYPOINT_DICT_Town01[64]
       ],
       [
           WAYPOINT_DICT_Town01[147],
           WAYPOINT_DICT_Town01[90]
       ],
       [
           WAYPOINT_DICT_Town01[33],
           WAYPOINT_DICT_Town01[87]
       ],
       [
           WAYPOINT_DICT_Town01[26],
           WAYPOINT_DICT_Town01[19]
       ],
       [
           WAYPOINT_DICT_Town01[80],
           WAYPOINT_DICT_Town01[76]
       ],
       [
           WAYPOINT_DICT_Town01[45],
           WAYPOINT_DICT_Town01[49]
       ],
       [
           WAYPOINT_DICT_Town01[55],
           WAYPOINT_DICT_Town01[44]
       ],
       [
           WAYPOINT_DICT_Town01[29],
           WAYPOINT_DICT_Town01[107]
       ],
       [
           WAYPOINT_DICT_Town01[95],
           WAYPOINT_DICT_Town01[104]
       ],
       [
           WAYPOINT_DICT_Town01[84],
           WAYPOINT_DICT_Town01[34]
       ],
       [
           WAYPOINT_DICT_Town01[53],
           WAYPOINT_DICT_Town01[67]
       ],
       [
           WAYPOINT_DICT_Town01[22],
           WAYPOINT_DICT_Town01[17]
       ],
       [
           WAYPOINT_DICT_Town01[91],
           WAYPOINT_DICT_Town01[148]
       ],
       [
           WAYPOINT_DICT_Town01[20],
           WAYPOINT_DICT_Town01[107]
       ],
       [
           WAYPOINT_DICT_Town01[78],
           WAYPOINT_DICT_Town01[70]
       ],
       [
           WAYPOINT_DICT_Town01[95],
           WAYPOINT_DICT_Town01[102]
       ],
       [
           WAYPOINT_DICT_Town01[68],
           WAYPOINT_DICT_Town01[44]
       ],
       [
           WAYPOINT_DICT_Town01[45],
           WAYPOINT_DICT_Town01[69]
       ]
    ]

    return paths

def benchmark_paths_straight_Town02():

    paths = [
       [    # Swapped to make it align in the correct direction
           WAYPOINT_DICT_Town02[34],
           WAYPOINT_DICT_Town02[38]
       ],
       [
           WAYPOINT_DICT_Town02[4],
           WAYPOINT_DICT_Town02[2]
       ],
       [
           WAYPOINT_DICT_Town02[12],
           WAYPOINT_DICT_Town02[10]
       ],
       [
           WAYPOINT_DICT_Town02[62],
           WAYPOINT_DICT_Town02[55]
       ],
       [
           WAYPOINT_DICT_Town02[43],
           WAYPOINT_DICT_Town02[47]
       ],
       [
           WAYPOINT_DICT_Town02[64],
           WAYPOINT_DICT_Town02[66]
       ],
       [
           WAYPOINT_DICT_Town02[78],
           WAYPOINT_DICT_Town02[76]
       ],
       [
           WAYPOINT_DICT_Town02[59],
           WAYPOINT_DICT_Town02[57]
       ],
       [
           WAYPOINT_DICT_Town02[61],
           WAYPOINT_DICT_Town02[18]
       ],
       [
           WAYPOINT_DICT_Town02[35],
           WAYPOINT_DICT_Town02[39]
       ],
       [
           WAYPOINT_DICT_Town02[12],
           WAYPOINT_DICT_Town02[8]
       ],
       [
           WAYPOINT_DICT_Town02[0],
           WAYPOINT_DICT_Town02[18]
       ],
       [
           WAYPOINT_DICT_Town02[75],
           WAYPOINT_DICT_Town02[68]
       ],
       [
           WAYPOINT_DICT_Town02[54],
           WAYPOINT_DICT_Town02[60]
       ],
       [
           WAYPOINT_DICT_Town02[45],
           WAYPOINT_DICT_Town02[49]
       ],
       [
           WAYPOINT_DICT_Town02[46],
           WAYPOINT_DICT_Town02[42]
       ],
       [
           WAYPOINT_DICT_Town02[53],
           WAYPOINT_DICT_Town02[46]
       ],
       [
           WAYPOINT_DICT_Town02[80],
           WAYPOINT_DICT_Town02[29]
       ],
       [
           WAYPOINT_DICT_Town02[65],
           WAYPOINT_DICT_Town02[63]
       ],
       [
           WAYPOINT_DICT_Town02[0],
           WAYPOINT_DICT_Town02[81]
       ],
       [
           WAYPOINT_DICT_Town02[54],
           WAYPOINT_DICT_Town02[63]
       ],
       [
           WAYPOINT_DICT_Town02[51],
           WAYPOINT_DICT_Town02[42]
       ],
       [
           WAYPOINT_DICT_Town02[16],
           WAYPOINT_DICT_Town02[19]
       ],
       [
           WAYPOINT_DICT_Town02[17],
           WAYPOINT_DICT_Town02[26]
       ],
       [
           WAYPOINT_DICT_Town02[77],
           WAYPOINT_DICT_Town02[68]
       ]
    ]

    return paths

def benchmark_paths_turn_Town01():

    paths = [
       [
           WAYPOINT_DICT_Town01[138],
           WAYPOINT_DICT_Town01[17]
       ],
       [
           WAYPOINT_DICT_Town01[47],
           WAYPOINT_DICT_Town01[16]
       ],
       [
           WAYPOINT_DICT_Town01[26],
           WAYPOINT_DICT_Town01[9]
       ],
       [
           WAYPOINT_DICT_Town01[42],
           WAYPOINT_DICT_Town01[49]
       ],
       [
           WAYPOINT_DICT_Town01[140],
           WAYPOINT_DICT_Town01[124]
       ],
       [
           WAYPOINT_DICT_Town01[85],
           WAYPOINT_DICT_Town01[98]
       ],
       [
           WAYPOINT_DICT_Town01[65],
           WAYPOINT_DICT_Town01[133]
       ],
       [
           WAYPOINT_DICT_Town01[137],
           WAYPOINT_DICT_Town01[51]
       ],
       [
           WAYPOINT_DICT_Town01[76],
           WAYPOINT_DICT_Town01[66]
       ],
       [
           WAYPOINT_DICT_Town01[46],
           WAYPOINT_DICT_Town01[39]
       ],
       [
           WAYPOINT_DICT_Town01[40],
           WAYPOINT_DICT_Town01[60]
       ],
       [
           WAYPOINT_DICT_Town01[0],
           WAYPOINT_DICT_Town01[29]
       ],
       [
           WAYPOINT_DICT_Town01[4],
           WAYPOINT_DICT_Town01[129]
       ],
       [
           WAYPOINT_DICT_Town01[121],
           WAYPOINT_DICT_Town01[140]
       ],
       [
           WAYPOINT_DICT_Town01[2],
           WAYPOINT_DICT_Town01[129]
       ],
       [
           WAYPOINT_DICT_Town01[78],
           WAYPOINT_DICT_Town01[44]
       ],
       [
           WAYPOINT_DICT_Town01[68],
           WAYPOINT_DICT_Town01[85]
       ],
       [
           WAYPOINT_DICT_Town01[41],
           WAYPOINT_DICT_Town01[102]
       ],
       [
           WAYPOINT_DICT_Town01[95],
           WAYPOINT_DICT_Town01[70]
       ],
       [
           WAYPOINT_DICT_Town01[68],
           WAYPOINT_DICT_Town01[129]
       ],
       [
           WAYPOINT_DICT_Town01[84],
           WAYPOINT_DICT_Town01[69]
       ],
       [
           WAYPOINT_DICT_Town01[47],
           WAYPOINT_DICT_Town01[79]
       ],
       [
           WAYPOINT_DICT_Town01[110],
           WAYPOINT_DICT_Town01[15]
       ],
       [
           WAYPOINT_DICT_Town01[130],
           WAYPOINT_DICT_Town01[17]
       ],
       [
           WAYPOINT_DICT_Town01[0],
           WAYPOINT_DICT_Town01[17]
       ]
    ]

    return paths

def benchmark_paths_turn_Town02():

    paths = [
       [
           WAYPOINT_DICT_Town02[37],
           WAYPOINT_DICT_Town02[76]
       ],
       [
           WAYPOINT_DICT_Town02[8],
           WAYPOINT_DICT_Town02[24]
       ],
       [
           WAYPOINT_DICT_Town02[60],
           WAYPOINT_DICT_Town02[69]
       ],
       [
           WAYPOINT_DICT_Town02[38],
           WAYPOINT_DICT_Town02[10]
       ],
       [
           WAYPOINT_DICT_Town02[21],
           WAYPOINT_DICT_Town02[1]
       ],
       [
           WAYPOINT_DICT_Town02[58],
           WAYPOINT_DICT_Town02[71]
       ],
       [
           WAYPOINT_DICT_Town02[74],
           WAYPOINT_DICT_Town02[32]
       ],
       [
           WAYPOINT_DICT_Town02[44],
           WAYPOINT_DICT_Town02[0]
       ],
       [
           WAYPOINT_DICT_Town02[71],
           WAYPOINT_DICT_Town02[16]
       ],
       [
           WAYPOINT_DICT_Town02[14],
           WAYPOINT_DICT_Town02[24]
       ],
       [
           WAYPOINT_DICT_Town02[34],
           WAYPOINT_DICT_Town02[11]
       ],
       [
           WAYPOINT_DICT_Town02[43],
           WAYPOINT_DICT_Town02[14]
       ],
       [
           WAYPOINT_DICT_Town02[75],
           WAYPOINT_DICT_Town02[16]
       ],
       [
           WAYPOINT_DICT_Town02[80],
           WAYPOINT_DICT_Town02[21]
       ],
       [
           WAYPOINT_DICT_Town02[3],
           WAYPOINT_DICT_Town02[23]
       ],
       [
           WAYPOINT_DICT_Town02[75],
           WAYPOINT_DICT_Town02[59]
       ],
       [
           WAYPOINT_DICT_Town02[50],
           WAYPOINT_DICT_Town02[47]
       ],
       [
           WAYPOINT_DICT_Town02[11],
           WAYPOINT_DICT_Town02[19]
       ],
       [
           WAYPOINT_DICT_Town02[77],
           WAYPOINT_DICT_Town02[34]
       ],
       [
           WAYPOINT_DICT_Town02[79],
           WAYPOINT_DICT_Town02[25]
       ],
       [
           WAYPOINT_DICT_Town02[40],
           WAYPOINT_DICT_Town02[63]
       ],
       [
           WAYPOINT_DICT_Town02[58],
           WAYPOINT_DICT_Town02[76]
       ],
       [
           WAYPOINT_DICT_Town02[79],
           WAYPOINT_DICT_Town02[55]
       ],
       [
           WAYPOINT_DICT_Town02[16],
           WAYPOINT_DICT_Town02[61]
       ],
       [
           WAYPOINT_DICT_Town02[27],
           WAYPOINT_DICT_Town02[11]
       ]
    ]

    return paths

def benchmark_paths_navigation_Town01():
    # paths = [[WAYPOINT_DICT_Town01[8], WAYPOINT_DICT_Town01[39]]]
    paths = [
    #     [
    #        WAYPOINT_DICT_Town01[6],
    #        WAYPOINT_DICT_Town01[134]
    #    ],
    #     [
    #        WAYPOINT_DICT_Town01[38],
    #        WAYPOINT_DICT_Town01[93]
    #    ],
       [
           WAYPOINT_DICT_Town01[105],
           WAYPOINT_DICT_Town01[29]
       ],
       [
           WAYPOINT_DICT_Town01[27],
           WAYPOINT_DICT_Town01[130]
       ],
       [
           WAYPOINT_DICT_Town01[102],
           WAYPOINT_DICT_Town01[87]
       ],
       [
           WAYPOINT_DICT_Town01[132],
           WAYPOINT_DICT_Town01[27]
       ],
       [
           WAYPOINT_DICT_Town01[24],
           WAYPOINT_DICT_Town01[44]
       ],
       [
           WAYPOINT_DICT_Town01[96],
           WAYPOINT_DICT_Town01[26]
       ],
       [
           WAYPOINT_DICT_Town01[34],
           WAYPOINT_DICT_Town01[67]
       ],
       [
           WAYPOINT_DICT_Town01[28],
           WAYPOINT_DICT_Town01[1]
       ],
       [
           WAYPOINT_DICT_Town01[140],
           WAYPOINT_DICT_Town01[134]
       ],
       [
           WAYPOINT_DICT_Town01[105],
           WAYPOINT_DICT_Town01[9]
       ],
       [
           WAYPOINT_DICT_Town01[148],
           WAYPOINT_DICT_Town01[129]
       ],
       [
           WAYPOINT_DICT_Town01[65],
           WAYPOINT_DICT_Town01[18]
       ],
       [
           WAYPOINT_DICT_Town01[21],
           WAYPOINT_DICT_Town01[16]
       ],
       [
           WAYPOINT_DICT_Town01[147],
           WAYPOINT_DICT_Town01[97]
       ],
       [
           WAYPOINT_DICT_Town01[42],
           WAYPOINT_DICT_Town01[51]
       ],
       [ # Swapped to make it align in the correct direction
           WAYPOINT_DICT_Town01[41],
           WAYPOINT_DICT_Town01[30]
       ],
       [
           WAYPOINT_DICT_Town01[18],
           WAYPOINT_DICT_Town01[107]
       ],
       [
           WAYPOINT_DICT_Town01[69],
           WAYPOINT_DICT_Town01[45]
       ],
       [
           WAYPOINT_DICT_Town01[102],
           WAYPOINT_DICT_Town01[95]
       ],
       [
           WAYPOINT_DICT_Town01[18],
           WAYPOINT_DICT_Town01[145]
       ],
       [
           WAYPOINT_DICT_Town01[111],
           WAYPOINT_DICT_Town01[64]
       ],
       [
           WAYPOINT_DICT_Town01[79],
           WAYPOINT_DICT_Town01[45]
       ],
       [
           WAYPOINT_DICT_Town01[84],
           WAYPOINT_DICT_Town01[69]
       ],
       [
           WAYPOINT_DICT_Town01[73],
           WAYPOINT_DICT_Town01[31]
       ],
       [
           WAYPOINT_DICT_Town01[37],
           WAYPOINT_DICT_Town01[81]
       ]
    ]

    # return paths[0:5]
    return paths

def benchmark_paths_navigation_Town02():

    paths = [
       [
           WAYPOINT_DICT_Town02[19],
           WAYPOINT_DICT_Town02[66]
       ],
       [
           WAYPOINT_DICT_Town02[79],
           WAYPOINT_DICT_Town02[14]
       ],
       [
           WAYPOINT_DICT_Town02[19],
           WAYPOINT_DICT_Town02[57]
       ],
       [
           WAYPOINT_DICT_Town02[23],
           WAYPOINT_DICT_Town02[1]
       ],
       [
           WAYPOINT_DICT_Town02[53],
           WAYPOINT_DICT_Town02[76]
       ],
       [
           WAYPOINT_DICT_Town02[42],
           WAYPOINT_DICT_Town02[13]
       ],
       [
           WAYPOINT_DICT_Town02[31],
           WAYPOINT_DICT_Town02[71]
       ],
       [
           WAYPOINT_DICT_Town02[33],
           WAYPOINT_DICT_Town02[5]
       ],
       [
           WAYPOINT_DICT_Town02[54],
           WAYPOINT_DICT_Town02[30]
       ],
       [
           WAYPOINT_DICT_Town02[10],
           WAYPOINT_DICT_Town02[61]
       ],
       [
           WAYPOINT_DICT_Town02[66],
           WAYPOINT_DICT_Town02[3]
       ],
       [
           WAYPOINT_DICT_Town02[27],
           WAYPOINT_DICT_Town02[12]
       ],
       [
           WAYPOINT_DICT_Town02[79],
           WAYPOINT_DICT_Town02[19]
       ],
       [
           WAYPOINT_DICT_Town02[2],
           WAYPOINT_DICT_Town02[29]
       ],
       [
           WAYPOINT_DICT_Town02[16],
           WAYPOINT_DICT_Town02[14]
       ],
       [
           WAYPOINT_DICT_Town02[5],
           WAYPOINT_DICT_Town02[57]
       ],
       [ # Swapped to make it align in the correct direction
           WAYPOINT_DICT_Town02[73],
           WAYPOINT_DICT_Town02[70]
       ],
       [
           WAYPOINT_DICT_Town02[46],
           WAYPOINT_DICT_Town02[67]
       ],
       [
           WAYPOINT_DICT_Town02[57],
           WAYPOINT_DICT_Town02[50]
       ],
       [
           WAYPOINT_DICT_Town02[61],
           WAYPOINT_DICT_Town02[49]
       ],
       [
           WAYPOINT_DICT_Town02[21],
           WAYPOINT_DICT_Town02[12]
       ],
       [
           WAYPOINT_DICT_Town02[51],
           WAYPOINT_DICT_Town02[81]
       ],
       [
           WAYPOINT_DICT_Town02[77],
           WAYPOINT_DICT_Town02[68]
       ],
       [
           WAYPOINT_DICT_Town02[56],
           WAYPOINT_DICT_Town02[65]
       ],
       [
           WAYPOINT_DICT_Town02[43],
           WAYPOINT_DICT_Town02[54]
       ]
    ]

    return paths

def paths_t_junction_Town01():
    paths = [
        [
            WAYPOINT_DICT_Town01[17],
            # WAYPOINT_DICT_Town01[53]
            WAYPOINT_DICT_Town01[56]
        ],
        [
            WAYPOINT_DICT_Town01[17],
            # WAYPOINT_DICT_Town01[46]
            WAYPOINT_DICT_Town01[44]
        ]
    ]

    return paths

def paths_left_Town01_train():

    paths = [
        [
            WAYPOINT_DICT_Town01[85],
            WAYPOINT_DICT_Town01[98]
        ],
        [
            WAYPOINT_DICT_Town01[87],
            WAYPOINT_DICT_Town01[100]
        ],
        [
            WAYPOINT_DICT_Town01[76],
            WAYPOINT_DICT_Town01[63]
        ],
        [
            WAYPOINT_DICT_Town01[70],
            WAYPOINT_DICT_Town01[66]
        ],
        [
            WAYPOINT_DICT_Town01[104],
            WAYPOINT_DICT_Town01[78]
        ],
        [
            WAYPOINT_DICT_Town01[106],
            WAYPOINT_DICT_Town01[80]
        ],
        [
            WAYPOINT_DICT_Town01[46],
            WAYPOINT_DICT_Town01[37]
        ],
        [
            WAYPOINT_DICT_Town01[44],
            WAYPOINT_DICT_Town01[39]
        ],
        [
            WAYPOINT_DICT_Town01[48],
            WAYPOINT_DICT_Town01[37]
        ]
    ]
    return paths

def paths_left_Town01_test():

    paths = [
        [
            WAYPOINT_DICT_Town01[85],
            WAYPOINT_DICT_Town01[100]
        ],
        [
            WAYPOINT_DICT_Town01[87],
            WAYPOINT_DICT_Town01[98]
        ],
        [
            WAYPOINT_DICT_Town01[76],
            WAYPOINT_DICT_Town01[77]
        ],
        [
            WAYPOINT_DICT_Town01[70],
            WAYPOINT_DICT_Town01[63]
        ],
        [
            WAYPOINT_DICT_Town01[104],
            WAYPOINT_DICT_Town01[80]
        ],
        [
            WAYPOINT_DICT_Town01[106],
            WAYPOINT_DICT_Town01[78]
        ],
        [
            WAYPOINT_DICT_Town01[46],
            WAYPOINT_DICT_Town01[39]
        ],
        [
            WAYPOINT_DICT_Town01[44],
            WAYPOINT_DICT_Town01[37]
        ],
        [
            WAYPOINT_DICT_Town01[48],
            WAYPOINT_DICT_Town01[39]
        ]
    ]
    return paths

def paths_right_Town01_train_():

    paths = [
        [
            WAYPOINT_DICT_Town01[42],
            WAYPOINT_DICT_Town01[49]
        ],
        [
            WAYPOINT_DICT_Town01[40],
            WAYPOINT_DICT_Town01[47]
        ],
        [
            WAYPOINT_DICT_Town01[38],
            WAYPOINT_DICT_Town01[49]
        ],
        [
            WAYPOINT_DICT_Town01[79],
            WAYPOINT_DICT_Town01[103]
        ],
        [
            WAYPOINT_DICT_Town01[81],
            WAYPOINT_DICT_Town01[105]
        ],
        [
            WAYPOINT_DICT_Town01[67],
            WAYPOINT_DICT_Town01[77]
        ],
        [
            WAYPOINT_DICT_Town01[64],
            WAYPOINT_DICT_Town01[71]
        ],
        [
            WAYPOINT_DICT_Town01[99],
            WAYPOINT_DICT_Town01[86]
        ],
        [
            WAYPOINT_DICT_Town01[97],
            WAYPOINT_DICT_Town01[84]
        ]
    ]
    return paths


def paths_right_Town01_train():
    paths = [
        [
            WAYPOINT_DICT_Town01[42],
            WAYPOINT_DICT_Town01[49]
        ],
        [
            WAYPOINT_DICT_Town01[40],
            WAYPOINT_DICT_Town01[47]
        ],
        [
            WAYPOINT_DICT_Town01[38],
            WAYPOINT_DICT_Town01[49]
        ]

    ]
    return paths[0:1]


def paths_right_Town01_test():

    paths = [
        # [
        #     WAYPOINT_DICT_Town01[42],
        #     WAYPOINT_DICT_Town01[47]
        # ],
        # [
        #     WAYPOINT_DICT_Town01[40],
        #     WAYPOINT_DICT_Town01[49]
        # ],
        # [
        #     WAYPOINT_DICT_Town01[38],
        #     WAYPOINT_DICT_Town01[47]
        # ],
        [
            WAYPOINT_DICT_Town01[79],
            WAYPOINT_DICT_Town01[105]
        ],
        [
            WAYPOINT_DICT_Town01[81],
            WAYPOINT_DICT_Town01[103]
        ],
        [
            WAYPOINT_DICT_Town01[67],
            WAYPOINT_DICT_Town01[71]
        ],
        [
            WAYPOINT_DICT_Town01[64],
            WAYPOINT_DICT_Town01[77]
        ],
        [
            WAYPOINT_DICT_Town01[99],
            WAYPOINT_DICT_Town01[84]
        ],
        [
            WAYPOINT_DICT_Town01[97],
            WAYPOINT_DICT_Town01[86]
        ]
    ]
    return paths[0:1]

'''
Deprecated
def paths_left_and_right_train():

    # # Old longer paths
    # paths = [
    #     # left
    #     [
    #         WAYPOINT_DICT_Town01[48],
    #         WAYPOINT_DICT_Town01[37]
    #     ],
    #     # right
    #     [
    #         WAYPOINT_DICT_Town01[38],
    #         WAYPOINT_DICT_Town01[49]
    #     ]
    # ]
    paths = [
        # left
        [
            WAYPOINT_DICT_Town01[44],
            WAYPOINT_DICT_Town01[39]
        ],
        # right
        [
            WAYPOINT_DICT_Town01[42],
            WAYPOINT_DICT_Town01[47]
        ]
    ]
    return paths

def paths_left_and_right_test():

    # Old longer paths
    # paths = [
    #     # left
    #     [
    #         WAYPOINT_DICT_Town01[87],
    #         WAYPOINT_DICT_Town01[100]
    #     ],
    #     # right
    #     [
    #         WAYPOINT_DICT_Town01[99],
    #         WAYPOINT_DICT_Town01[86]
    #     ]
    # ]
    paths = [
        # left
        [
            WAYPOINT_DICT_Town01[85],
            WAYPOINT_DICT_Town01[98]
        ],
        # right
        [
            WAYPOINT_DICT_Town01[94],
            Transform(Location(x=1.5099804401397705, y=278.81, z=1.32), Rotation(yaw=-90.00029754638672))
            # Destination is mid of 84 and 86
            # 84: Transform(Location(x=1.5099804401397705, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
            # 86: Transform(Location(x=1.5099804401397705, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),

        ]
    ]
    return paths
'''

def get_straight_dynamic_path(unseen=False, town="Town01", index=0):
    " Returns a list of [start_transform, target_transform]"
    if not unseen:
        if town == "Town01":
            return random.choice(paths_straight_Town01_dynamic())
        elif town == "Town02":
            return random.choice(benchmark_paths_straight_Town02())
    else:
        if town == "Town01":
            return paths_straight_Town01_dynamic()[index]
        elif town == "Town02":
            return benchmark_paths_straight_Town02()[index]

def get_long_straight_path(unseen=False, town='Town01', index=0):
    if town == "Town01":
        return paths_long_straight()[index]
    else:
        raise NotImplementedError("Long-straight scenarios only implemented for Town01!")

def get_long_straight_junction_path(unseen=False, town='Town01', index=0):
    if town == "Town01":
        return paths_long_straight_junction()[index % 3]
    else:
        raise NotImplementedError("Long-straight-junction scenarios only implemented for Town01!")

def get_crowded_path(unseen=False, town="Town01", index=0):
    return random.choice(paths_crowded())

def get_straight_crowded_path(unseen=False, town="Town01", index=0):
    return random.choice(paths_straight_crowded())

def get_curved_town03_path(unseen=False, town="Town03", index=0):
    if town != 'Town03':
        print('Error: must be Town03')
        return -1
    return random.choice(paths_curved_town03())

def get_crowded_npcs(num_npcs):
    npc_list = [
        WAYPOINT_DICT_Town01[50],
        WAYPOINT_DICT_Town01[51],
        WAYPOINT_DICT_Town01[52],
        WAYPOINT_DICT_Town01[16],
        WAYPOINT_DICT_Town01[17],
        WAYPOINT_DICT_Town01[18],
        WAYPOINT_DICT_Town01[46],
        WAYPOINT_DICT_Town01[47],
        WAYPOINT_DICT_Town01[48],
        WAYPOINT_DICT_Town01[49],
    ]
    random.shuffle(npc_list)
    return npc_list[:num_npcs]

def get_long_straight_npcs():
    # npc_list = [
    #     WAYPOINT_DICT_Town01[15],
    #     WAYPOINT_DICT_Town01[16],
    #     WAYPOINT_DICT_Town01[17],
    #     WAYPOINT_DICT_Town01[18],
    #     WAYPOINT_DICT_Town01[19],
    #     WAYPOINT_DICT_Town01[20],
    #     WAYPOINT_DICT_Town01[22],
    #     WAYPOINT_DICT_Town01[23],
    #     WAYPOINT_DICT_Town01[24],
    #     WAYPOINT_DICT_Town01[25],
    #     WAYPOINT_DICT_Town01[26],
    #     WAYPOINT_DICT_Town01[27],
    #     WAYPOINT_DICT_Town01[28],
    #     WAYPOINT_DICT_Town01[29],
    #     WAYPOINT_DICT_Town01[140],
    #     WAYPOINT_DICT_Town01[118],
    # ]

    npc_list = [
        WAYPOINT_DICT_Town01[17],
        WAYPOINT_DICT_Town01[15],
        Transform(Location(x=92.11000061035156, y=50.95999908447266, z=1.32), Rotation(yaw=-90.00029754638672)),
        Transform(Location(x=92.11000061035156, y=70.95999908447266, z=1.32), Rotation(yaw=-90.00029754638672)),
        WAYPOINT_DICT_Town01[19],
        WAYPOINT_DICT_Town01[22],
        WAYPOINT_DICT_Town01[24],
        Transform(Location(x=92.11000061035156, y=125.95999908447266, z=1.32), Rotation(yaw=-90.00029754638672)),
        WAYPOINT_DICT_Town01[26],
        WAYPOINT_DICT_Town01[28],
        Transform(Location(x=92.1099853515625, y=200.88999938964844, z=1.32), Rotation(yaw=-90.00029754638672)),
        WAYPOINT_DICT_Town01[140],
        WAYPOINT_DICT_Town01[118],
        Transform(Location(x=92.10997772216797, y=270.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
        # Transform(Location(x=92.10997772216797, y=290.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),

        # Adding some vehicles on left lane as well for intersection crossing scenario
        # WAYPOINT_DICT_Town01[16],
        WAYPOINT_DICT_Town01[18],
        WAYPOINT_DICT_Town01[20],
        WAYPOINT_DICT_Town01[23],
        # WAYPOINT_DICT_Town01[25],
        WAYPOINT_DICT_Town01[27],
        WAYPOINT_DICT_Town01[29],
    ]

    # Copied here to refer exact location
    # 15: Transform(Location(x=92.11000061035156, y=39.709999084472656, z=1.32), Rotation(yaw=-90.00029754638672)),
    # 17: Transform(Location(x=92.11000061035156, y=30.820009231567383, z=1.32), Rotation(yaw=-90.00029754638672)),
    # 19: Transform(Location(x=92.11000061035156, y=86.95999908447266, z=1.32), Rotation(yaw=-90.00029754638672)),
    # 96: Transform(Location(x=92.10997772216797, y=308.2099914550781, z=1.32), Rotation(yaw=-90.00029754638672)),
    # 24: Transform(Location(x=92.1099853515625, y=113.05999755859375, z=1.32), Rotation(yaw=-90.00029754638672)),
    # 118: Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672)),
    # 28: Transform(Location(x=92.1099853515625, y=176.88999938964844, z=1.32), Rotation(yaw=-90.00029754638672)),

    random.shuffle(npc_list)
    return npc_list

def get_straight_crowded_npcs(num_npcs):
    npc_list = [
        WAYPOINT_DICT_Town01[101],
        WAYPOINT_DICT_Town01[102],
        WAYPOINT_DICT_Town01[103],
        WAYPOINT_DICT_Town01[104],
        WAYPOINT_DICT_Town01[105],
        WAYPOINT_DICT_Town01[106],
        WAYPOINT_DICT_Town01[107],
        WAYPOINT_DICT_Town01[108],
        WAYPOINT_DICT_Town01[109],
        WAYPOINT_DICT_Town01[110],
        WAYPOINT_DICT_Town01[111],
        WAYPOINT_DICT_Town01[112],
        #WAYPOINT_DICT_Town01[113],
        WAYPOINT_DICT_Town01[114],
        WAYPOINT_DICT_Town01[115],
        WAYPOINT_DICT_Town01[116],
        WAYPOINT_DICT_Town01[117],
        WAYPOINT_DICT_Town01[118],
        #WAYPOINT_DICT_Town01[119],
        WAYPOINT_DICT_Town01[120],
        WAYPOINT_DICT_Town01[121],
        WAYPOINT_DICT_Town01[122],
    ]
    random.shuffle(npc_list)
    return npc_list[:num_npcs]

'''
todo: this was only for points inside the circle
do it for points outside as well
'''
def get_curved_town03_npcs(num_npcs):
    npc_list = [
        WAYPOINT_DICT_Town03[257],
        WAYPOINT_DICT_Town03[85],
        WAYPOINT_DICT_Town03[86],
        WAYPOINT_DICT_Town03[228],
        WAYPOINT_DICT_Town03[229],
        # WAYPOINT_DICT_Town03[232],
        WAYPOINT_DICT_Town03[136],
        WAYPOINT_DICT_Town03[233],
        WAYPOINT_DICT_Town03[230],
        WAYPOINT_DICT_Town03[231],
        WAYPOINT_DICT_Town03[245],
        WAYPOINT_DICT_Town03[246],
        WAYPOINT_DICT_Town03[243],
        WAYPOINT_DICT_Town03[136],
        WAYPOINT_DICT_Town03[244],
        WAYPOINT_DICT_Town03[187],
        WAYPOINT_DICT_Town03[188],
        WAYPOINT_DICT_Town03[218],
        WAYPOINT_DICT_Town03[219],
        WAYPOINT_DICT_Town03[122],
        WAYPOINT_DICT_Town03[123],
        WAYPOINT_DICT_Town03[0],
        WAYPOINT_DICT_Town03[7],
        WAYPOINT_DICT_Town03[221],
        WAYPOINT_DICT_Town03[220],
        WAYPOINT_DICT_Town03[248],
        WAYPOINT_DICT_Town03[121],
        WAYPOINT_DICT_Town03[120],
        WAYPOINT_DICT_Town03[149],
        WAYPOINT_DICT_Town03[146],
        WAYPOINT_DICT_Town03[250],
        WAYPOINT_DICT_Town03[249],
        WAYPOINT_DICT_Town03[1],
        WAYPOINT_DICT_Town03[2],
        WAYPOINT_DICT_Town03[243],
        WAYPOINT_DICT_Town03[244],
        WAYPOINT_DICT_Town03[211],
        WAYPOINT_DICT_Town03[210],
        WAYPOINT_DICT_Town03[113],
        WAYPOINT_DICT_Town03[112],
        WAYPOINT_DICT_Town03[118],
        WAYPOINT_DICT_Town03[114],
        WAYPOINT_DICT_Town03[212],
        WAYPOINT_DICT_Town03[213],
        WAYPOINT_DICT_Town03[42],
        WAYPOINT_DICT_Town03[41],
        WAYPOINT_DICT_Town03[40],
        WAYPOINT_DICT_Town03[39],
        WAYPOINT_DICT_Town03[8],
        WAYPOINT_DICT_Town03[85],
        WAYPOINT_DICT_Town03[257],
        WAYPOINT_DICT_Town03[162],
        WAYPOINT_DICT_Town03[163],
        WAYPOINT_DICT_Town03[47],
        WAYPOINT_DICT_Town03[46],
        WAYPOINT_DICT_Town03[145],
        WAYPOINT_DICT_Town03[84],
    ]
    random.shuffle(npc_list)
    # return npc_list[:num_npcs]
    return npc_list

def get_straight_path(unseen=False, town="Town01", index=0):
    " Returns a list of [start_transform, target_transform]"
    if not unseen:
        if town == "Town01":
            return random.choice(benchmark_paths_straight_Town01())
        elif town == "Town02":
            return random.choice(benchmark_paths_straight_Town02())
    else:
        if town == "Town01":
            return benchmark_paths_straight_Town01()[index]
        elif town == "Town02":
            return benchmark_paths_straight_Town02()[index]

def get_straight_path_updated(unseen=False, town="Town01", index=0):
    " Returns a list of [start_idx, target_idx]"

    paths_Town01 = [[220, 216], [217, 72], [41, 37], [249, 253], [256, 252],
                    [188, 206], [195, 197], [209, 192], [4, 61], [223, 88],
                    [230, 237], [165, 180], [123, 207], [201, 212], [227, 44],
                    [56, 47], [121, 222], [203, 120], [234, 239], [60, 3],
                    [103, 44], [178, 186], [56, 129], [188, 212], [123, 187]]

    paths_Town02 = [[47, 51], [81, 83], [73, 75], [23, 30], [42, 38],
                    [21, 19], [7, 9], [26, 28], [24, 67], [50, 46],
                    [73, 77], [100, 67], [10, 17], [31, 25], [40, 36],
                    [39, 43], [32, 39], [5, 56], [20, 22], [100, 4],
                    [31, 22], [34, 43], [69, 66], [68, 59], [8, 17]]

    if not unseen:
        if town == "Town01":
            return random.choice(paths_Town01)
        elif town == "Town02":
            return random.choice(paths_Town02)
    else:
        if town == "Town01":
            return paths_Town01[index]
        elif town == "Town02":
            return paths_Town02[index]

def get_curved_path(unseen=False, town="Town01", index=0):
    " Returns a list of [start_transform, target_transform]"
    if not unseen:
        if town == "Town01":
            return random.choice(benchmark_paths_turn_Town01())
        elif town == "Town02":
            return random.choice(benchmark_paths_turn_Town02())
    else:
        if town == "Town01":
            return benchmark_paths_turn_Town01()[index]
        elif town == "Town02":
            return benchmark_paths_turn_Town02()[index]

def get_curved_path_updated(unseen=False, town="Town01", index=0):
    " Returns a list of [start_idx, target_idx]"

    paths_Town01 = [[13, 239], [209, 240], [230, 247], [214, 207], [11, 27],
                    [110, 53], [191, 18], [14, 205], [180, 190], [210, 217],
                    [216, 196], [256, 227], [252, 177], [30, 11], [254, 177],
                    [178, 212], [188, 110], [215, 129], [56, 186], [188, 177],
                    [121, 187], [209, 176], [41, 234], [21, 239], [256, 239]]

    paths_Town02 = [[48, 9], [77, 61], [25, 16], [47, 75], [64, 89],
                    [27, 14], [11, 53], [41, 100], [14, 69], [71, 61],
                    [51, 74], [42, 71], [10, 69], [5, 64], [82, 62],
                    [10, 26], [35, 38], [74, 66], [8, 51], [6, 60],
                    [45, 22], [27, 9], [6, 30], [69, 24], [58, 74]]

    if not unseen:
        if town == "Town01":
            return random.choice(paths_Town01)
        elif town == "Town02":
            return random.choice(paths_Town02)
    else:
        if town == "Town01":
            return paths_Town01[index]
        elif town == "Town02":
            return paths_Town02[index]

def get_navigation_path(unseen=False, town="Town01", index=0):
    " Returns a list of [start_transform, target_transform]"
    if not unseen:
        if town == "Town01":
            return random.choice(benchmark_paths_navigation_Town01())
        elif town == "Town02":
            return random.choice(benchmark_paths_navigation_Town02())
    else:
        if town == "Town01":
            return benchmark_paths_navigation_Town01()[index]
        elif town == "Town02":
            return benchmark_paths_navigation_Town02()[index]

def get_navigation_path_updated(unseen=False, town="Town01", index=0):
    " Returns a list of [start_idx, target_idx]"

    paths_Town01 = [[79, 227], [105, 21], [129, 88], [19, 105], [104, 212],
                    [84, 230], [222, 120], [228, 255], [11, 17], [79, 247],
                    [3, 177], [191, 240], [235, 240], [4, 54], [214, 205],
                    [56, 215], [114, 44], [187, 123], [129, 56], [114, 6],
                    [40, 192], [176, 123], [121, 187], [238, 225], [219, 154]]

    paths_Town02 = [[66, 19], [6, 71], [66, 28], [62, 89], [32, 9],
                    [43, 72], [54, 14], [52, 82], [31, 55], [75, 24],
                    [19, 82], [58, 73], [6, 66], [83, 56], [69, 71],
                    [82, 28], [19, 12], [39, 18], [28, 35], [24, 36],
                    [64, 73], [34, 4], [8, 17], [29, 20], [42, 31]]

    if not unseen:
        if town == "Town01":
            return random.choice(paths_Town01)
        elif town == "Town02":
            return random.choice(paths_Town02)
    else:
        if town == "Town01":
            return paths_Town01[index]
        elif town == "Town02":
            return paths_Town02[index]

def get_no_crash_path(unseen=False, town="Town01", index=0):
    " Returns a list of [start_idx, target_idx]"

    paths_Town01 = [[79, 227], [105, 21], [129, 88], [19, 105], [231, 212],
                    [252, 192], [222, 120], [202, 226], [11, 17], [79, 247],
                    [3, 177], [191, 114], [235, 240], [4, 54], [17, 207],
                    [223, 212], [154, 66], [187, 123], [129, 56], [114, 6],
                    [40, 192], [176, 123], [121, 187], [238, 225], [219, 154]]

    paths_Town02 = [[66, 19], [6, 71], [66, 28], [46, 32], [25, 59],
                    [32, 9], [43, 72], [54, 14], [26, 50], [38, 69],
                    [75, 24], [19, 82], [65, 6], [71, 29], [59, 16],
                    [6, 66], [83, 56], [69, 71], [82, 28], [8, 17],
                    [19, 12], [39, 18], [51, 8], [24, 36], [64, 73]]

    if not unseen:
        if town == "Town01":
            return random.choice(paths_Town01)
        elif town == "Town02":
            return random.choice(paths_Town02)
    else:
        if town == "Town01":
            return paths_Town01[index]
        elif town == "Town02":
            return paths_Town02[index]

def get_t_junction_path(unseen=False, town="Town01", index=0):
    " Returns a list of [start_transform, target_transform]"
    if not unseen:
        if town == "Town01":
            return random.choice(paths_t_junction_Town01())
        else:
            raise NotImplementedError("T-Junction scenarios only implemented for Town01!")
    else:
        if town == "Town01":
            return paths_t_junction_Town01()[index%2]
        else:
            raise NotImplementedError("T-Junction scenarios only implemented for Town01!")


def get_fixed_long_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return paths_straight_Town01_train()[0]

def get_fixed_long_curved_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return benchmark_paths_turn_Town01()[0]

def get_random_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return random.choice(paths_straight_Town01_train())

def get_left_right_randomly(unseen = False):
    return random.choice([get_right_turn(unseen=unseen), get_left_turn(unseen=unseen)])

def get_right_turn(unseen = False):
    if unseen:
        return random.choice(paths_right_Town01_test())
    else:
        return random.choice(paths_right_Town01_train())

def get_left_turn(unseen = False):
    if unseen:
        return random.choice(paths_left_Town01_test())
    else:
        return random.choice(paths_left_Town01_train())

from environment.carla_9_4.agents.navigation.local_planner import RoadOption

def get_test_route():

    # Route: 17, 56, 77
    route =[({'z': 1.3200000524520874, 'lat': -0.00027686085348932465, 'lon': 0.0008274382136853726}, RoadOption.LEFT),
    ({'z': 1.3200000524520874, 'lat': -1.8146144199704395e-05, 'lon': 0.0015554328596241375}, RoadOption.RIGHT),
    ({'z': 1.3200000524520874, 'lat': -0.0006185802527767237, 'lon': 0.0035256180065496483  }, RoadOption.STRAIGHT)
    ]

    source_transform = Transform(Location(x=271.0400085449219, y=129.489990234375, z=1.32), Rotation(yaw=179.999755859375)),
    return route

import xml.etree.ElementTree as ET

def get_leaderboard_route(unseen=False, curr_town=None, index=0, max_idx=None, avail_map_list=None, mode='train'):
    xml_file = {
            'train': '../../leaderboard/data/routes_training.xml',
            'test': '../../leaderboard/data/routes_testing.xml'
        }[mode]

    _route_tree = ET.parse(xml_file)
    _routes = _route_tree.findall('route')
    # print(2119, index, max_idx)
    if not max_idx or index > max_idx: # have had enough with previous town
        _town = random.choice(_routes).attrib['town']
        if avail_map_list and _town not in avail_map_list:
            _town = curr_town
    else:
        _town = curr_town

    _route_in_this_town = []
    for _r in _routes: 
        if _r.attrib['town'] == _town:
            _route_in_this_town.append(_r)

    if not unseen:
        _route = random.choice(_route_in_this_town)
    else:
        _route = _route_in_this_town[index % len(_route_in_this_town)]

    _wps = _route.findall('waypoint')
    # _src = Transform(Location(x=float(_wps[0].attrib['x']), y=float(_wps[0].attrib['y']), z=float(_wps[0].attrib['z']) + .5))
    _src = Transform(Location(x=float(_wps[0].attrib['x']), y=float(_wps[0].attrib['y']), z=float(_wps[0].attrib['z']) + .5), 
        Rotation(pitch=float(_wps[0].attrib['pitch']), yaw=float(_wps[0].attrib['yaw']), roll=float(_wps[0].attrib['roll'])))
    _wp_list = [_src.location]
    for _wp in _wps[1:-1]: 
        _wp_loc = Transform(Location(x=float(_wp.attrib['x']), y=float(_wp.attrib['y']), z=float(_wp.attrib['z']) + .5))
        _wp_list.append(_wp_loc.location)
    _dst = Transform(Location(x=float(_wps[-1].attrib['x']), y=float(_wps[-1].attrib['y']), z=float(_wps[0].attrib['z']) + .5))
    _wp_list.append(_dst.location)
    # _src = Transform(Location(x=float(_wps[0].attrib['x']), y=float(_wps[0].attrib['y']), z=float(_wps[0].attrib['z']) + .5), 
    #     Rotation(pitch=float(_wps[0].attrib['pitch']), yaw=float(_wps[0].attrib['yaw']), roll=float(_wps[0].attrib['roll'])))
    # _dst = Transform(Location(x=float(_wps[-1].attrib['x']), y=float(_wps[-1].attrib['y']), z=float(_wps[0].attrib['z']) + .5), 
    #     Rotation(pitch=float(_wps[0].attrib['pitch']), yaw=float(_wps[0].attrib['yaw']), roll=float(_wps[-1].attrib['roll'])))
    # print(_src, _dst)
    # print(2149, curr_town, _town)
    return _src, _dst, _wp_list, _town

# def get_leaderboard_route(unseen=False, town="Town01", index=0, mode='train'):
#     xml_file = {
#             'train': '../../leaderboard/data/routes_training.xml',
#             'test': '../../leaderboard/data/routes_testing.xml'
#         }[mode]

#     _route_tree = ET.parse(xml_file)
#     _routes = _route_tree.findall('route')
#     _route_in_this_town = []
#     for _r in _routes: 
#         if _r.attrib['town'] == town:
#             _route_in_this_town.append(_r)

#     if not unseen:
#         _route = random.choice(_route_in_this_town)
#     else:
#         _route = _route_in_this_town[index % len(_route_in_this_town)]
#     _wps = _route.findall('waypoint')
#     # _src = Transform(Location(x=float(_wps[0].attrib['x']), y=float(_wps[0].attrib['y']), z=float(_wps[0].attrib['z']) + .5))
#     _src = Transform(Location(x=float(_wps[0].attrib['x']), y=float(_wps[0].attrib['y']), z=float(_wps[0].attrib['z']) + .5), 
#         Rotation(pitch=float(_wps[0].attrib['pitch']), yaw=float(_wps[0].attrib['yaw']), roll=float(_wps[0].attrib['roll'])))
#     _wp_list = [_src.location]
#     for _wp in _wps[1:-1]: 
#         _wp_loc = Transform(Location(x=float(_wp.attrib['x']), y=float(_wp.attrib['y']), z=float(_wp.attrib['z']) + .5))
#         _wp_list.append(_wp_loc.location)
#     _dst = Transform(Location(x=float(_wps[-1].attrib['x']), y=float(_wps[-1].attrib['y']), z=float(_wps[0].attrib['z']) + .5))
#     _wp_list.append(_dst.location)
#     # _src = Transform(Location(x=float(_wps[0].attrib['x']), y=float(_wps[0].attrib['y']), z=float(_wps[0].attrib['z']) + .5), 
#     #     Rotation(pitch=float(_wps[0].attrib['pitch']), yaw=float(_wps[0].attrib['yaw']), roll=float(_wps[0].attrib['roll'])))
#     # _dst = Transform(Location(x=float(_wps[-1].attrib['x']), y=float(_wps[-1].attrib['y']), z=float(_wps[0].attrib['z']) + .5), 
#     #     Rotation(pitch=float(_wps[0].attrib['pitch']), yaw=float(_wps[0].attrib['yaw']), roll=float(_wps[-1].attrib['roll'])))
#     # print(_src, _dst)
#     return _src, _dst, _wp_list

'''
# Deprecated Helper functions
def get_train_right_turn():
    return paths_left_and_right_train()[1]

def get_test_right_turn():
    return paths_left_and_right_test()[1]

def get_train_left_turn():
    return paths_left_and_right_train()[0]

def get_test_left_turn():
    return paths_left_and_right_test()[0]

def get_train_left_right_randomly():
    return random.choice(paths_left_and_right_train())

def get_test_left_right_randomly():
    return random.choice(paths_left_and_right_test())
'''
