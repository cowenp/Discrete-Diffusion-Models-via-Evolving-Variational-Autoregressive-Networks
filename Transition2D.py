# Transition rule for 2D KCM
import itertools
import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
from numpy import sqrt
from torch import nn
import math

from args import args
from bernoulli import BernoulliMixture
from gru2D import GRU2D
from lstm2D import LSTM2D
from made import MADE
from made1D import MADE1D
from mdrnn import RNN2D
from mdtensorizedrnn import MDTensorizedRNN
from pixelcnn import PixelCNN
from stacked_pixelcnnFA import StackedPixelCNN
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
    cholesky_solve_fast
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


plt.rc('font', size=16)

def calculate_magnetization(config):
    """
    Compute the total magnetization of an Ising configuration.
    - Input: tensor of shape [batch_size, 1, height, width] with spin values ±1.
    - Output: scalar representing the average magnetization across all configurations.
    """
    if not isinstance(config, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    if config.dim() != 4 or config.size(1) != 1:
        raise ValueError("Input shape must be [batch_size, 1, height, width]")

    # Compute the total magnetization for each configuration by summing over the last two dimensions.
    magnetization_per_config = torch.abs(config.sum(dim=(2, 3))/args.size)  # shape: [batch_size, 1]

    # Compute the average magnetization across all configurations.
    avg_magnetization = magnetization_per_config.mean()

    return avg_magnetization


def calculate_energy_2d_torch(config, J=1.0):
    """Compute the 2D Ising energy under fully periodic boundary conditions in both horizontal and vertical directions."""
    if not isinstance(config, torch.Tensor) or len(config.shape) != 3:
        raise ValueError("Input must be a 3D PyTorch tensor [batch, height, width]")

    # Horizontal periodic boundary: right neighbors are obtained by rolling the tensor.
    right = (config * torch.roll(config, shifts=-1, dims=2)).sum(dim=(1, 2))

    # Vertical periodic boundary: down neighbors are obtained by rolling the tensor so the bottom wraps to the top.
    down = (config * torch.roll(config, shifts=-1, dims=1)).sum(dim=(1, 2))

    return -J * (right + down)

'''
def calculate_energy_2d_torch(config, J=1.0):
    """Compute the 2D Ising energy under cylindrical boundary conditions."""
    if not isinstance(config, torch.Tensor) or len(config.shape) != 3:
        raise ValueError("Input must be a 3D PyTorch tensor [batch, height, width]")

    # Horizontal periodic boundary and vertical open boundary.
    right = (config * torch.roll(config, shifts=-1, dims=2)).sum(dim=(1, 2))
    down = (config[:, :-1, :] * config[:, 1:, :]).sum(dim=(1, 2))
    return -J * (right + down)
'''

def TransitionState(sample, args, Tstep, step, net_new):
    Sample1D = (sample.view(-1, args.size) + 1) / 2  # sample has size batchsize X systemSize
    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size).expand(Sample1D.shape[0], args.size, args.size).to(args.device)
    SampleNeighbor1DExtend = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)
    #RNN reshape_Neighbor = SampleNeighbor1DExtend.view(args.size,args.batch_size,args.L,args.L)
    #pixelCNN
    reshape_Neighbor = SampleNeighbor1DExtend.view(args.size,args.batch_size,1,args.L,args.L1)
    reshape_Neighbor = reshape_Neighbor * 2 - 1
    #print(sample.shape)
    
    #print(SampleNeighbor1DExtend)
    #print(SampleNeighbor1DExtend.shape)
    # New code 2: use log-sum-exp. In PyTorch, tensor operations normally build a computation graph for backpropagation.
    # During inference, however, no gradient computation is needed, so torch.no_grad() avoids the overhead of graph construction.
    # Typical use cases include model evaluation and inference after training.
    with torch.no_grad():
        if Tstep == 0:
            with torch.no_grad():
                LogP_t = -args.beta * calculate_energy_2d_torch(sample.view(args.batch_size, args.L,args.L1))
                
            return LogP_t
        else:
            with torch.no_grad():
                LogP_t = torch.exp(net_new.log_prob(sample)).detach()
                LogTP_t1 = LogP_t * (1 - args.size * args.delta_t)
                
                # Compute all batch probabilities in one shot.
                # reshape_Neighbor is assumed to have shape [args.size, args.batch_size, 1, args.L, args.L].
                all_probs = torch.exp(net_new.log_prob(
                    reshape_Neighbor.view(-1, 1, args.L, args.L1)  # Merge the first two dimensions.
                )).view(args.size, args.batch_size)  # Restore the shape.

                # Sum the neighborhood probabilities for each batch sample.
                transition_rates = all_probs.sum(dim=0) * args.delta_t  # shape [args.batch_size]
        
        with torch.no_grad():
            LogTP_t1 += transition_rates 
            LogTP_t1 = torch.log(LogTP_t1)
    
        return LogTP_t1
    

def SCGF(sample, args, Tstep, step, net):
    Sample1D = (sample.view(-1, args.size) + 1) / 2  # sample has size batchsize X systemSize

    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size).expand(Sample1D.shape[0], args.size, args.size).to(args.device)
    SampleNeighbor1DExtend = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)
    Sample1D = Sample1D[:, 1:]  # L^3 to L^3-1 neighbor by fixing the first spin up

    Sample2D = (sample.view(-1, args.L, args.L) + 1) / 2  # sample has size batchsize X L X L
    Win = (Sample1D - 1).abs() * (
        1 - args.c
    ) + Sample1D * args.c  # The previous state flip into these sampled states
    Col1 = torch.cat(
        (torch.zeros(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
    )  # down sites
    Col2 = torch.cat((Sample2D[:, :, 1:], torch.zeros(Sample2D.shape[0], args.L, 1).to(args.device)), 2)
    Row1 = torch.cat((torch.zeros(Sample2D.shape[0], 1, args.L).to(args.device), Sample2D[:, :-1, :]), 1)
    Row2 = torch.cat((Sample2D[:, 1:, :], torch.zeros(Sample2D.shape[0], 1, args.L).to(args.device)), 1)
    if args.BC == 1:
        if step == 0:
            print('Left-up BC SCGF.')
        Col1 = torch.cat(
            (torch.ones(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
        )  # down sites
    if args.BC == 2:
        if step == 0:
            print('All-up BC SCGF.')
        Col1 = torch.cat(
            (torch.ones(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
        )  # down sites
        Col2 = torch.cat((Sample2D[:, :, 1:], torch.ones(Sample2D.shape[0], args.L, 1).to(args.device)), 2)
        Row1 = torch.cat((torch.ones(Sample2D.shape[0], 1, args.L).to(args.device), Sample2D[:, :-1, :]), 1)
        Row2 = torch.cat((Sample2D[:, 1:, :], torch.ones(Sample2D.shape[0], 1, args.L).to(args.device)), 1)
    if args.BC == 3:
        if step == 0:
            print('Periodic BC.')
        Col1 = torch.cat(
            (Sample2D[:, :, -1].view(Sample2D.shape[0], args.L, 1), Sample2D[:, :, :-1]), 2
        )  # down sites
        Col2 = torch.cat((Sample2D[:, :, 1:], Sample2D[:, :, 0].view(Sample2D.shape[0], args.L, 1)), 2)
        Row1 = torch.cat((Sample2D[:, -1, :].view(Sample2D.shape[0], 1, args.L), Sample2D[:, :-1, :]), 1)
        Row2 = torch.cat((Sample2D[:, 1:, :], Sample2D[:, 0, :].view(Sample2D.shape[0], 1, args.L)), 1)

    if args.Model == '2DFA':
        # torch.cat((torch.ones(Sample2D.shape[0],args.L,1),Sample2D[:,:-1,:]) ,2)+torch.cat((Sample1D[:,1:],torch.ones(Sample1D.shape[0],1)) ,1)
        fNeighbor = (Col1 + Col2 + Row1 + Row2).view(-1, args.size)
    if args.Model == '2DNoEast':
        # torch.cat((torch.ones(Sample2D.shape[0],args.L,1),Sample2D[:,:-1,:]) ,2)+torch.cat((Sample1D[:,1:],torch.ones(Sample1D.shape[0],1)) ,1)
        fNeighbor = (Col1 + Row1 + Row2).view(-1, args.size)
    if args.Model == '2DEast':
        fNeighbor = Col1.view(-1, args.size)
    if args.Model == '2DSouthEast':
        fNeighbor = (Col1 + Row1).view(-1, args.size)
    if args.Model == '2DNorthWest':
        fNeighbor = (Col2 + Row2).view(-1, args.size)
    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size - 1, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size - 1).expand(Sample1D.shape[0], args.size - 1, args.size - 1).to(args.device)
    SampleNeighbor1D = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)
    # BatchSize: The escape-probability for each sampled state to all connected states#torch.sum((Sample1D-1).abs()*args.c+Sample1D*(1-args.c),1)
    R = torch.as_tensor(torch.sum((1 - Win) * fNeighbor[:, 1:], 1), dtype=torch.float64).to(args.device)
    # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
    Win = Win * fNeighbor[:, 1:]

    Win_lambda = torch.as_tensor(Win * np.float64(np.exp(-args.lambda_tilt)), dtype=torch.float64).to(
        args.device
    )
    if args.Hermitian:
        WinHermitian = (Sample1D - 1).abs() * np.sqrt(args.c * (1 - args.c)) + Sample1D * np.sqrt(
            args.c * (1 - args.c)
        )  # The previous state flip into these sampled states
        # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
        WinHermitian = WinHermitian * fNeighbor
        Win_lambda = torch.as_tensor(
            WinHermitian * np.float64(np.exp(-args.lambda_tilt)), dtype=torch.float64
        ).to(args.device)

    LogP_t = net.log_prob(sample).detach()
    # .view(sample.shape[0], args.size, args.size) #BatchSize X NeighborSize X SystemSize
    Temp = torch.transpose(SampleNeighbor1DExtend, 0, 1)
    # Set the first spin up to avoid numerical problem when generating prob
    Temp[:, 0, 0] = torch.as_tensor(1).to(args.device, dtype=default_dtype_torch)

    Temp = Temp + (Temp - 1)  # Change 0 to -1 back
    if args.net == 'rnn' or args.net == 'rnn2' or args.net == 'lstm' or args.net == 'rnn3':
        Temp3 = torch.reshape(Temp, (args.batch_size * args.size, args.L, args.L))  # For RNN
    else:
        Temp3 = torch.reshape(Temp, (args.batch_size * args.size, 1, args.L, args.L))  # For VAN
    # BatchSize X NeighborSize: checked, it is consistent with for loop
    LogP_t_other = torch.reshape(net.log_prob(Temp3), (args.batch_size, args.size)).detach()
    LogP_t_other = LogP_t_other[:, 1:]  # fixing the first spin up

    thetaLoc = (
        torch.sum(torch.sqrt(torch.exp(LogP_t_other - LogP_t.repeat(args.size - 1, 1).t())) * Win_lambda, 1)
        - R
    )  # Conversion from probability P to state \psi

    return thetaLoc


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1).float()


def AllTransitionState1D(args, dict2, Tstep, P_tnew):
    Win_lambda = torch.as_tensor(
        dict2['Win'] * np.float64(np.exp(-args.lambda_tilt)), dtype=torch.float64
    ).to(args.device)
    P_t = P_tnew.to(args.device)  # torch.exp(net_new.log_prob(sample)).detach()
    P_t_other = (
        torch.cat([P_tnew[dict2['IdConnec'][:, i].numpy()] for i in range(args.size - 1)])
        .to(args.device)
        .view(args.size - 1, -1)
        .t()
    )  # Correct: BatchSize X NeighborSize: checked, it is consistent with for loop
    TP_Exact = (
        P_t + (torch.sum(P_t_other * Win_lambda[:, :], 1) - dict2['R'].to(args.device) * P_t) * args.delta_t
    )

    return TP_Exact


def OptimizeFunction(net, params, optimizer, scheduler, net_new, args, lambda_tilt, Tstep):
    SampleT = []
    free_energy_mean3Temp = []
    Loss1Temp = []
    Loss1_2Temp = []
    ListDistanceCheck_Eucli = []
    ListDistanceCheck = []
    Listloss_mean = []
    Listloss_std = []
    List_M = []
    
    
    for step in range(0, args.max_step + 1):
        optimizer.zero_grad()
        with torch.no_grad():
            sample3, x_hat1 = net.sample(10000)
            cal_M =  calculate_magnetization(sample3)
            print(cal_M)

        if step > 2000:
            args.lr = 1e-3
        with torch.no_grad():
            sample, _ = net.sample(args.batch_size)
        log_prob = net.log_prob(sample).to(torch.float32)
        
        with torch.no_grad():
            LogTP_t = TransitionState(sample, args, Tstep, step, net_new)
            
            loss = (log_prob - LogTP_t.detach()).to(torch.float32)
            TP_t_normalize = (
                torch.exp(LogTP_t) / (torch.exp(LogTP_t)).sum() * (torch.exp(log_prob)).sum()
            ).detach()

            
            #NG
            # Reshape the input sample.
            s = sample.reshape(args.batch_size, args.L*args.L1)

            # Compute per-sample gradients (dictionary form: parameter name -> gradient tensor).
            grads = net.per_sample_grad(s)  # returns dict{parameter_name: gradient_tensor}

            # Flatten all parameter gradients and concatenate them into a matrix with shape [batch_size × total_parameter_count].
            grads_flatten = torch.cat([torch.flatten(v, start_dim=1) for v in grads.values()], dim=1).to(torch.float32)

            # Scale the gradient matrix by 1 / sqrt(batch_size).
            O_mat = (grads_flatten) / math.sqrt(args.batch_size)

            # baseline = (weights * loss).mean()
            # Compute the target vector: the mean of the gradient matrix times the centered loss.
            F_vec = torch.einsum("nm,n->m", grads_flatten, (loss - loss.mean())) / args.batch_size

            # Solve the linear system using Cholesky decomposition.
            updates_flatten = cholesky_solve_fast(O_mat, F_vec)

            # Update the network parameters.
            net.update_params(updates_flatten, args.lr)
            
        '''
        assert not LogTP_t.requires_grad
        if args.lossType == 'kl':
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        elif args.lossType == 'klreweight':
            loss3 = torch.exp(log_prob) * (loss) / torch.exp(log_prob).mean()
            loss_reinforce = torch.mean((loss3 - loss3.mean()) * log_prob)
        elif args.lossType == 'l2':
            loss_reinforce = torch.mean((lossL2 - lossL2.mean()) * log_prob)
        elif args.lossType == 'he':
            loss_reinforce = torch.mean((losshe) * log_prob)
        elif args.lossType == 'ss':
            # steady state# Conversion from probability P to state \psi
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob / 2)
        #print("loss_reinforce requires_grad:", loss_reinforce.requires_grad)
        #print("loss requires_grad:", loss.requires_grad)
        #print("log_prob requires_grad:", log_prob.requires_grad)

        loss_reinforce.backward()
        
        
        if args.clip_grad:
            nn.utils.clip_grad_norm_(params, args.clip_grad)
        optimizer.step()
        if args.lr_schedule:
            scheduler.step(loss.mean())
        '''
        loss_std = loss.std()  # /args.size
        loss_mean = loss.mean()  # / args.size#(P_tnew * (P_tnew / (TP_t/torch.sum(TP_t))).log()).sum()
        List_M.append(cal_M.detach().cpu().numpy())
        Listloss_mean.append(loss_mean.detach().cpu().numpy())
        Listloss_std.append(loss_std.detach().cpu().numpy())
        DistanceCheck_Eucli = torch.sqrt(torch.sum((torch.exp(net.log_prob(sample)) - TP_t_normalize) ** 2))
        # function kl_div is not the same as wiki's explanation.
        DistanceCheck = torch.nn.functional.kl_div(net.log_prob(sample), TP_t_normalize, None, None, 'sum')
        ListDistanceCheck_Eucli.append(DistanceCheck_Eucli.detach().cpu().numpy())
        ListDistanceCheck.append(DistanceCheck.detach().cpu().numpy())
        if step > int(args.max_step * (1 - args.Percent)):
            with torch.no_grad():
                # loss.mean() #/ args.beta / args.size
                free_energy_mean3Temp.append(
                    torch.mean(torch.exp(log_prob) * (loss) / torch.exp(log_prob).mean())
                )
                Loss1Temp.append(loss_mean)
                Loss1_2Temp.append(loss_std)
        if step > int(args.max_step - 2):  # max(int(args.max_step*(1-0.02)),1):
            with torch.no_grad():
                SampleT.append(np.array(sample.detach().cpu()))
        if args.print_step and step % args.print_step == 0 and Tstep % int(args.print_step) == 0:
            my_log('Training...')
            my_log(
                'lambda={}, Time step of equation={}, Training step = {}, loss_std={:.20f},loss_mean={}'.format(  # ',DynPartiFuncFactorLog={:.8f}'#' F = {:.8g}, F_std = {:.8g}, S = {:.8g}, E = {:.8g}, M = {:.8g}, Q = {:.8g}, lr = {:.3g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                    lambda_tilt,
                    Tstep,
                    step,
                    #torch.abs(loss_reinforce),
                    torch.abs(loss_std),
                    torch.abs(loss_mean),  # DynPartiFuncFactorLog,
                )
            )

    return (
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
        List_M
    )  # return TP_Exact


def Optimizer(net, args):
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    return optimizer, params, nparams
