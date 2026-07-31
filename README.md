Lab of Machine Learning and Complex Systems (2026).

A software package for the manuscript "	Variational autoregressive networks for discrete diffusion modeling" 

--------------------------------------------------------------------------------------------------------------------------------------------

System requirements: 
All simulations were done using Python.
We have used the package Pytorch. The code requires Python >= 3.6 and PyTorch >= 1.0.

--------------------------------------------------------------------------------------------------------------------------------------------

# Inputs

The main files of 2D, 3D are Main2D.py, Main3D.py.

The system parameters and neural-network hyperparameters are in these scripts. 

## Note: 

(a) See args.py for a the help information on the parameters if needed.

--------------------------------------------------------------------------------------------------------------------------------------------

# Platforms

PC users (Windows): you may use Spyder to run Main2D.py. You can properly adjust the hyperparameters in Main2D.py, including those listed below.

```
    # #Initialize parameters
    # #System parameters:
    args.L=2# Lattice size: 1D  
    args.Tstep=100#Time step of iterating the dynamical equation P_tnew=Q*P_t, where Q=(I+W*delta t)
    args.delta_t=0.1#Time step length
    
    # #Neural-network hyperparameters:
    args.net ='made'#'pixelcnn'#'lstm'#'rnn'Type of neural network in the VAN
    args.max_stepAll=500#0 #The epoch for the 1st  time steps 
    args.max_stepLater=50#00 #The epoch at time steps except for the 1st step
    args.print_step=1#0   # The time step size of print and save results
    args.net_depth=3#4  # Depth of the neural network
    args.net_width=32#16 # Width of the neural network
    args.batch_size=100#1000 #batch size  
```

--------------------------------------------------------------------------------------------------------------------------------------------

# Results


(1) 2D Ising model,

(2) 3D Ising model. 

--------------------------------------------------------------------------------------------------------------------------------------------

## Acknowledgments
This code is built upon [DPT](https://github.com/Machine-learning-and-complex-systems/DPT)
We thank the original authors for their open-source contributions.
