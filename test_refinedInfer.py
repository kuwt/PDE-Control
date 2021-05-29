import sys; sys.path.append('PhiFlow'); sys.path.append('src')
from control.pde.burgers import GaussianClash, GaussianForce
import burgers_plots as bplt
import matplotlib.pyplot as plt
from phi.flow import *

domain = Domain([128], box=box[0:1])  # 1D Grid resolution and physical size
viscosity = 0.003  # Viscosity constant for Burgers equation
step_count = 32  # how many solver steps to perform
dt = 0.03  # Time increment per solver step

data_path = 'notebooks/forced-burgers-clash'
scene_count = 1000  # how many examples to generate (training + validation + test)
batch_size = 100  # How many examples to generate in parallel

from control.pde.burgers import BurgersPDE
from control.control_training import ControlTraining
from control.sequences import StaggeredSequence, RefinedSequence

test_range = range(100)
val_range = range(100, 200)
train_range = range(200, 1000)

refined_app = ControlTraining(step_count,
                      BurgersPDE(domain, viscosity, dt),
                      datapath=data_path,
                      val_range=val_range,
                      train_range=train_range,
                      trace_to_channel=lambda trace: 'burgers_velocity',
                      obs_loss_frames=[],
                      trainable_networks=['OP%d' % n for n in [2,4,8,16,32]],
                      sequence_class=RefinedSequence,
                      batch_size=1,
                      view_size=20,
                      learning_rate=1e-3,
                      learning_rate_half_life=1000,
                      dt=dt).prepare()
refined_app.load_checkpoints({'OP%d'%n: 'networks/burgers/staggered/OPn_18000' for n in [2,4,8,16,32]})

print('Total Force (supervised): %f' % refined_app.infer_scalars(range(100, 101))['Total Force'])