from utils import (
    clear_checkpoint,
    clear_log,
    default_dtype_torch,
    ensure_dir,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
)
from Transition2D import (
    SCGF,
    AllTransitionState1D,
    OptimizeFunction,
    Optimizer,
    TransitionState,
    gen_all_binary_vectors,
)
from stacked_pixelcnnFA import StackedPixelCNN
from pixelcnn import PixelCNN
from mdtensorizedrnn import MDTensorizedRNN
from mdrnn import RNN2D
from made1D import MADE1D
from made import MADE
from lstm2D import LSTM2D
from gru2D import GRU2D
from bernoulli import BernoulliMixture
from args import args
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import copy
import os
import time
import itertools
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


plt.rc('font', size=16)


def Test():
    # #Initialize parameters: see args.py for a the help information on the parameters if needed
    # #System parameters:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.z2 = 1
    args.dtype = 'float32'
    args.lr = 6e-3 
    #args.lr_schedule = 1
    #args.lr_schedule_type == 2
    args.L = 16
    args.L1 = 16
    args.size = args.L * args.L1
    args.Tstep = 101  # Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
    args.load_step = 1
    args.delta_t = 1/(args.size*2) # Time step length
    args.flip_p = (1 - math.exp(-2 * args.delta_t)) / 2
    args.dlambda = (
        0.1  # The steplength from the left to the rigth boundary value of the count field  (lambda=s)
    )
    args.dlambdaL = -1.6  # left boundary value of the count field  (lambda=s)
    args.dlambdaR = -1.55  # Rigth boundary value of the count field  (lambda=s)
    
    args.beta = torch.tensor(0.5, dtype=torch.float32)
    args.c = 0.35#0.5 #Flip-up probability
    args.negativeS = True  # For the negative counting field (s<0) or not (s>0)
    # #Neural-network hyperparameters:
    args.net = 'pixelcnn'  # 'rnn'#'lstm'
    #'rnn'Type of neural network in the VAN
    args.max_stepAll = 1500 # 0 #The epoch for the 1st  time steps
    args.max_stepLater = 150  # 00 #The epoch at time steps except for the 1st step
    args.print_step = 10  # 0   # The time step size of print and save results
    args.net_depth = 3 # 3#3  # Depth of the neural network
    args.net_width = 32  # 128 # Width of the neural network
    args.batch_size = 512  # 00#1000 #batch size

    ###################################################
    # #Optional initial parameters
    # # args.half_kernel_size=2 #kernel size for pixelcnn
    # # args.loadVAN=False# Load the saved VAN at some time steps
    # args.loadTime=0# The loaded time steps
    # args.lrLoad=0.001# The loaded lr
    # args.max_stepLoad=100 # The epoch after the load
    # args.compareVAN=False#True # Compare the loaded VAN and new VAN, or not
    # args.Hermitian=False # Do Hermitian transform or not
    # args.Doob=False #Use Doob operator
    # args.BC=0 #Other boundary conditions
    # args.reverse=False #Use reverse order in RNN

    ###################################################
    # #Default parameters
    args.max_step = args.max_stepAll  # args.max_step=max_step
    args.lossType = 'kl'  # 'ss'# #type of loss
    args.clip_grad = 1  # clip gradient
    # args.lr_schedule=1#1, 2
    args.bias = True  # With bias or not in the neural network
    args.size = args.L * args.L1  # the number of spin: doesnt' count the boundary spins
    args.epsilon = torch.tensor(1e-300, dtype=torch.float32)  # 1e-300#0#1 # avoid 0 value in log below
    args.free_energy_batch_size = args.batch_size  # 1000
    # args.Hermitian=False # Do Hermitian transform or not
    lambda_tilt_Range = 10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda))  # plot heatmap
    start_time2 = time.time()

    # start_time = time.time()
    init_out_dir()
    SummaryListDynPartiFuncLog2 = []
    SummaryListDynPartiFuncLog3 = []
    net_new = []
    SummaryLoss1 = []
    SummaryLoss1_2 = []
    SummaryLoss2 = []
    SummarySampleSum = []
    
    
    count = -1
    for lambda_tilt in lambda_tilt_Range:
        count += 1
        args.lambda_tilt = lambda_tilt
        args.max_step = args.max_stepAll  # args.max_step=max_step

        ###############################
        # Initialize net and optimizer
        if args.net == 'made':
            net = MADE(**vars(args))  # net2 = MADE1D(**vars(args))
        elif args.net == 'rnn':
            net = GRU2D(**vars(args))
        elif args.net == 'lstm':
            net = LSTM2D(**vars(args))
        elif args.net == 'rnn2':
            net = RNN2D(**vars(args))
        elif args.net == 'rnn3':
            net = MDTensorizedRNN(**vars(args))
        elif args.net == 'pixelcnn':
            net = PixelCNN(**vars(args))
        elif args.net == 'stackedpixelcnn':
            net = StackedPixelCNN(**vars(args))
        elif args.net == 'bernoulli':
            net = BernoulliMixture(**vars(args))
        else:
            raise ValueError('Unknown net: {}'.format(args.net))
        net.to(args.device)

        TP_Exact = []
        ListDynPartiFuncLog3 = []
        DynPartiFuncLog3 = 0
        ListDynPartiFuncLog2_3 = []
        DynPartiFuncLog2_3 = 0
        Loss1 = []
        Loss1_2 = []
        Loss2 = []
        TP_ExactRecord = []
        SampleSum = []

        # Load NN starting from certain time point:
        startT = 0
        if args.loadVAN:
            PATH = args.out_filename + str(round(args.lambda_tilt, 3)) + 'Tstep' + str(args.loadTime)
            startT = args.loadTime
            print(PATH)
            if args.cuda == -1:
                state = torch.load(PATH, map_location=torch.device('cpu'))  # CPU
            else:
                state = torch.load(PATH)  # GPU
            if not args.UseNewVAN:
                net1 = GRU2D(**vars(args))
                net1.load_state_dict(state['net'])  # new save format
                net1.to(args.device)
                if args.compareVAN:
                    optimizer1, params1, nparams1 = Optimizer(
                        net1, args
                    )  # From steady-state VAN: not evolve itself
                    optimizer1.load_state_dict(state['optimizer'])  # load saved optimizer1

                print('Use saved VAN')
                if args.lossType == 'ss':  # or args.lossType=='kl':
                    net = copy.deepcopy(net1)
                    print('Use saved VAN for ss')
            net_new = copy.deepcopy(net)  # net
            net_new.to(args.device)
            net_new.requires_grad = False
            if args.compareVAN:
                args.out_filename = (
                    args.out_filename
                    + 'CompareVAN'
                    + str(args.loadTime)
                    + 'lr'
                    + str(args.lrLoad)
                    + 'epoch'
                    + str(args.max_stepLoad)
                )
            else:
                args.out_filename = (
                    args.out_filename
                    + 'loadVAN'
                    + str(args.loadTime)
                    + 'lr'
                    + str(args.lrLoad)
                    + 'epoch'
                    + str(args.max_stepLoad)
                )
            ensure_dir(args.out_filename + '_img/')

        optimizer, params, nparams = Optimizer(net, args)

        if args.Doob == 1:
            data = np.load(
                '{}_img/DataSSL{}c{}s{}'.format(args.out_filename, args.L, args.c, args.dlambdaL) + '.npz'
            )
            print(list(data))
            print(data['arr_3'].shape)
            args.thetaLoad = -data['arr_3'][count, -1]  # loss=-theta

        for Tstep in range(startT, args.Tstep):  # Time step of the dynamical equation
            # This If is to increase epoch is finding that loss increases
            if Tstep >= 1 and not args.loadVAN:
                args.max_step = args.max_stepLater
            if args.loadVAN:
                if Tstep == startT:
                    args.max_step = args.max_stepAll
                else:
                    args.max_step = args.max_stepLoad

            scheduler = []
            if args.lr_schedule:
                if args.lr_schedule_type == 1:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=0.5,
                        patience=int(args.max_step * args.Percent),
                        verbose=True,
                        threshold=1e-4,
                        min_lr=1e-5,
                    )
                if args.lr_schedule_type == 2:
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=lambda epoch: 1 / (epoch * 10 * args.lr + 1)
                    )
                if args.lr_schedule_type == 3:
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, 10 ** (-2 / args.max_step)
                    )  # lr_final = 0.01 * lr_init (1e-2 -> 1e-4)
                if args.lr_schedule_type == 4:
                    scheduler = torch.optim.lr_scheduler.CyclicLR(
                        optimizer,
                        base_lr=1e-4,
                        max_lr=1e-2,
                        step_size_up=20,
                        step_size_down=20,
                        mode='exp_range',
                        gamma=0.999,
                        scale_fn=None,
                        scale_mode='cycle',
                        cycle_momentum=False,
                        last_epoch=-1,
                    )

            # Train VAN: if args.loadVAN, use loaded VAN
            if not args.compareVAN:
                (
                    net,
                    optimizer,
                    SampleT,
                    free_energy_mean3Temp,
                    Loss1Temp,
                    Loss1_2Temp,
                    ListDistanceCheck_Eucli,
                    DistanceCheck_Eucli,
                    ListDistanceCheck,
                    Listloss_mean,
                    Listloss_std,
                    magnetization,
                ) = OptimizeFunction(net, params, optimizer, scheduler, net_new, args, lambda_tilt, Tstep)

            # Compare VAN: if args.compareVAN, compare Null VAN and saved VAN at steady state for phase transitoin
            if args.compareVAN:
                print('Use two VANs to compare')
                (
                    net,
                    optimizer,
                    SampleT,
                    free_energy_mean3Temp,
                    Loss1Temp,
                    Loss1_2Temp,
                    ListDistanceCheck_Eucli,
                    DistanceCheck_Eucli,
                    ListDistanceCheck,
                    Listloss_mean,
                    Listloss_std,
                ) = OptimizeFunction(net, params, optimizer, scheduler, net_new, args, lambda_tilt, Tstep)

                net2 = copy.deepcopy(net1)
                [optimizer2, params2, nparams2] = [optimizer1, params1, nparams1]
                (
                    net2,
                    optimizer2,
                    SampleT2,
                    free_energy_mean3Temp2,
                    Loss1Temp2,
                    Loss1_2Temp2,
                    ListDistanceCheck_Eucli2,
                    DistanceCheck_Eucli2,
                    ListDistanceCheck2,
                    Listloss_mean2,
                    Listloss_std2,
                ) = OptimizeFunction(net2, params2, optimizer2, scheduler, net_new, args, lambda_tilt, Tstep)
                LossCompare1 = np.array(
                    torch.mean(torch.tensor(Loss1Temp, dtype=torch.float32).to(args.device)).detach().cpu()
                )
                LossCompare2 = np.array(
                    torch.mean(torch.tensor(Loss1Temp2, dtype=torch.float32).to(args.device)).detach().cpu()
                )
                if LossCompare1 > LossCompare2:
                    print('Steady state VAN is used')
                    net = copy.deepcopy(net2)
                    [
                        optimizer,
                        params,
                        SampleT,
                        free_energy_mean3Temp,
                        Loss1Temp,
                        Loss1_2Temp,
                        ListDistanceCheck_Eucli,
                        DistanceCheck_Eucli,
                        ListDistanceCheck,
                        Listloss_mean,
                        Listloss_std,
                    ] = [
                        optimizer2,
                        params2,
                        SampleT2,
                        free_energy_mean3Temp2,
                        Loss1Temp2,
                        Loss1_2Temp2,
                        ListDistanceCheck_Eucli2,
                        DistanceCheck_Eucli2,
                        ListDistanceCheck2,
                        Listloss_mean2,
                        Listloss_std2,
                    ]
                
            with torch.no_grad():
                if Tstep >= 0 and (Tstep+1) % args.load_step == 0:
                    # Custom save path.
                    model_save_path = 'models/VAN_step' + str(Tstep+1) + '.pth'

                    # Ensure the directory exists.
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

                    torch.save(net, model_save_path)

                net_new = copy.deepcopy(net)  # net
                net_new.requires_grad = False
                
            
            
    #####################################
    # Save data
    end_time2 = time.time()
    print('Time ', (end_time2 - start_time2) / 60)
    print('Time ', (end_time2 - start_time2) / 3600)
    
    # plt.figure(num=None,  dpi=400, edgecolor='k')
    # fig, axes = plt.subplots(1,1)
    # fig.tight_layout()
    # print(SummaryLoss1/args.size)
    # plt.plot(lambda_tilt_Range, SummaryLoss1[:,-1]/args.size, 'bo',markersize=8,label='L '+str(args.L))
    # plt.xlim((1e-4,0.8))
    # plt.ylim((1e-5,1e1))
    # axes.set_yscale('log')
    # axes.set_xscale('log')
    # plt.xlabel('lambda')
    # plt.ylabel('-theta/L^2')#plt.xlabel('Time (Lyapunov time)')
    # plt.legend()
    # fig.set_size_inches(5, 5)
    # plt.savefig('{}_img/theta.jpg'.format(args.out_filename), dpi=400)#, 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)


if __name__ == '__main__':
    Test()
