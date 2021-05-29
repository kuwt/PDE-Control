import sys; sys.path.append('PhiFlow'); sys.path.append('src')
from control.pde.burgers import GaussianClash, GaussianForce
import burgers_plots as bplt
import matplotlib.pyplot as plt
from phi.flow import *

domain = Domain([128], box=box[0:1])  # 1D Grid resolution and physical size
viscosity = 0.003  # Viscosity constant for Burgers equation
step_count = 2  # how many solver steps to perform
dt = 0.03  # Time increment per solver step

# --- Set up physics ---
world = World()
u0 = BurgersVelocity(domain, velocity=GaussianClash(1), viscosity=viscosity)
print(u0)
u = world.add(u0, physics=Burgers(diffusion_substeps=4))
force = world.add(FieldEffect(GaussianForce(1), ['velocity']))
# --- Plot ---
print('Force: %f at %f' % (force.field.amp[0], force.field.loc[0]))
bplt.burgers_figure('Training data')
plt.plot(u.velocity.data[0,:,0], color=bplt.gradient_color(0, step_count+1), linewidth=0.8)  # data[example, values, component]
plt.legend(['Initial state in dark red, final state in dark blue.'])
for frame in range(1, step_count + 1):
    world.step(dt=dt)  # runs one simulation step
    plt.plot(u.velocity.data[0,:,0], color=bplt.gradient_color(frame, step_count+1), linewidth=0.8)
plt.show()


