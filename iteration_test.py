# -*- coding: utf-8 -*-
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

from tqdm import tqdm 
from args import args
# Minor ticks (finer grid lines)
from matplotlib.ticker import AutoMinorLocator
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
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

replicas_num = 64
args.Tstep = 30
args.L = 16
args.L1 = 16
args.size = args.L * args.L1
args.delta_t = 1/(args.size*2)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
args.z2 = 1
beta = torch.tensor(1 / 1, dtype=torch.float64)
jump_step = 10

total_sum1 = 0.0    
total_sum2 = 0.0
total_sum3 = 0.0
total_count = 0    # Total number of samples
n_iterations = 50000              # Number of iterations (controls the total sample count)
true_statistic2 = 1
line_style = '-'                 # Line style (solid; can be changed to --/-. /:)


# Initialize storage variables
statistic_list1 = []
statistic_list2 = []
statistic_list3 = []
sample_size_list = []
Acc_dm = []
Acc_nmcmc = []
Acc_all = []

plt.rcParams['text.usetex'] = True

plt.rcParams.update({
    'font.family': 'Times New Roman + SimSun + Nowar Sans GB18030',
    'axes.linewidth': 1.0,
    'axes.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'lines.linewidth': 1.8,  # Thicker lines improve visual clarity
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

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

def normalized_autocorrelation(samples_k, samples_k_plus_mu):
    # Flatten the samples.
    flattened_k = samples_k.reshape(samples_k.shape[0], -1)          # (N, D)
    flattened_mu = samples_k_plus_mu.reshape(samples_k_plus_mu.shape[0], -1)  # (N, D)

    # Explicitly remove the mean.
    mean_k = flattened_k.mean(axis=0)
    mean_mu = flattened_mu.mean(axis=0)
    flattened_k_centered = flattened_k - mean_k
    flattened_mu_centered = flattened_mu - mean_mu

    # Compute the covariance term (numerator).
    cross_term = (flattened_k_centered * flattened_mu_centered).mean(axis=0)
    numerator = np.sum(cross_term)

    # Compute the variance term (denominator).
    var_k = np.var(flattened_k, axis=0)
    var_mu = np.var(flattened_mu, axis=0)
    denominator = np.sqrt(np.sum(var_k) * np.sum(var_mu))  # or simply use np.sum(var_k)

    if np.isclose(denominator, 0):
        return 0.0

    C_mu = numerator / denominator
    return C_mu

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

def calculate_energy_2d_torch(config, J=1.0):
    """Compute the 2D Ising energy under fully periodic boundary conditions in both horizontal and vertical directions."""
    if not isinstance(config, torch.Tensor) or len(config.shape) != 3:
        raise ValueError("Input must be a 3D PyTorch tensor [batch, height, width]")

    # Horizontal periodic boundary: right neighbors are obtained by rolling the tensor.
    right = (config * torch.roll(config, shifts=-1, dims=2)).sum(dim=(1, 2))

    # Vertical periodic boundary: down neighbors are obtained by rolling the tensor so the bottom wraps to the top.
    down = (config * torch.roll(config, shifts=-1, dims=1)).sum(dim=(1, 2))

    return -J * (right + down)


def sample_event(prob_tensor):
    """
    Sample an event index for each row of a probability tensor, handling cases where the row sum is less than 1,
    exactly 1, or greater than 1.

    Args:
        prob_tensor (torch.Tensor): probability tensor of shape [a, b].

    Returns:
        indices (torch.Tensor): tensor of shape [a] with sampled event indices.
            - Normal event: 0 to b - 1.
            - No-event case (row sum < 1 and random draw falls in the remaining interval): -1.
            - Overflow event (row sum > 1 and random draw falls in the overflow interval): -2.
    """
    device = prob_tensor.device
    a, b = prob_tensor.shape

    # Compute cumulative probabilities and total probability.
    cum_prob = torch.cumsum(prob_tensor, dim=1)  # [a, b]
    total_prob = cum_prob[:, -1]  # [a]

    # Generate random numbers.
    random_values = torch.rand(a, device=device)  # [a]

    # Initialize the result tensor.
    results = torch.full((a,), -3, dtype=torch.long, device=device)  # Initialize to -3 for easier debugging.

    # Case 1: total probability is exactly 1.
    exact_mask = (total_prob == 1.0)
    if exact_mask.any():
        print('luck')
        mask = random_values[exact_mask].unsqueeze(1) < cum_prob[exact_mask]
        results[exact_mask] = torch.argmax(mask.int(), dim=1)

    # Case 2: total probability is less than 1.
    underflow_mask = (total_prob < 1.0)
    if underflow_mask.any():
        mask = random_values[underflow_mask].unsqueeze(1) < cum_prob[underflow_mask]
        event_indices = torch.argmax(mask.int(), dim=1)

        # Handle the no-event case.
        no_event_mask = random_values[underflow_mask] > total_prob[underflow_mask]
        event_indices[no_event_mask] = -1
        results[underflow_mask] = event_indices

    # Case 3: total probability is greater than 1 (updated implementation).
    overflow_mask = (total_prob > 1.0)
    if overflow_mask.any():
        print('exceed')
        # 1. First collect all positions that should return -2.
        minus_two_mask = torch.zeros(a, dtype=torch.bool, device=device)

        # Case 3a: overflow interval returns -2.
        extended_range = 2 * total_prob[overflow_mask] - 1
        scaled_random = random_values[overflow_mask] * extended_range
        excess_mask = scaled_random >= total_prob[overflow_mask]

        # Get the row indices that need to return -2.
        overflow_indices = overflow_mask.nonzero().squeeze(1)
        minus_two_mask[overflow_indices[excess_mask]] = True

        global drop_mask  # Declare use of a global variable.

        drop_mask = minus_two_mask | drop_mask

        results[drop_mask] = -2

        # 3. Handle the normal region.
        normal_mask = overflow_mask & (~minus_two_mask)
        if normal_mask.any():
            mask = random_values[normal_mask].unsqueeze(1) < cum_prob[normal_mask]
            results[normal_mask] = torch.argmax(mask.int(), dim=1)

    return results

def zero_out_elements(tensor, event_indices):
    """
    Vectorized implementation.
    """
    valid_mask = (event_indices >= 0) & (event_indices < args.size)
    rows = torch.arange(tensor.size(0), device=tensor.device)[valid_mask]
    cols = event_indices[valid_mask]

    # Use vectorized operations.
    tensor[rows, cols] = 1 - tensor[rows, cols]


def flip_tensor(tensor, p):
    """
    Improved tensor flip function.

    Args:
        tensor (torch.Tensor): tensor to operate on, shape [n, m].
        p (float): probability of not flipping (0 <= p <= 1).
    """
    # Ensure that random numbers are generated on the same device as the input tensor.
    device = tensor.device
    n = tensor.size(0)
    
    # 1. Generate random numbers to decide which rows need flipping.
    flip_mask = torch.rand(n, device=device) > p  # [n]

    # 2. Randomly choose column indices for the rows that need flipping.
    flip_rows = flip_mask.nonzero().squeeze(-1)  # indices of rows that need flipping
    flip_cols = torch.randint(0, tensor.size(1), (flip_rows.size(0),), device=device)  # random column indices

    # 3. Perform the vectorized flip operation.
    tensor[flip_rows, flip_cols] = 1 - tensor[flip_rows, flip_cols]
    
def ising_2d_rectangle(
    Lx=30,
    Ly=8,
    J=1.0,
    beta=2.0,
    n_steps=200000,
    burn_in=20000,
    n_chains=10,
    init_n_flips=10,
    init_sanmples=None
):
    np.random.seed(42)
    torch.manual_seed(42)

    # Fix 1: initialize n_chains spin chains.
    chains = init_sanmples.view(n_chains, Lx, Ly).cpu().detach().numpy().astype(np.int8)

    # Parameters.
    n_flips = init_n_flips
    accept_history = []
    n_flips_history = []
    histories = [[] for _ in range(n_chains)]
    target_accept = 0.5  # Correct adaptive target: 0.2 to 0.4.

    # Main loop.
    for step in tqdm(range(n_steps), desc="MC Steps"):
        total_accept = 0

        # Fix 2: update each chain independently (correct MCMC).
        for c in range(n_chains):
            S = chains[c]
            N = Lx * Ly

            # Randomly select non-repeating sites.
            indices = np.random.choice(N, size=n_flips, replace=False)
            positions = [(idx // Ly, idx % Ly) for idx in indices]

            # Save the original spins.
            original_spins = {(i, j): S[i, j] for (i, j) in positions}

            # Flip all selected sites in batch.
            for (i, j) in positions:
                S[i, j] *= -1

            # Fix 3: use uniform periodic boundary conditions (full PBC).
            delta_E = 0.0
            for (i, j) in positions:
                nb = (
                    S[(i+1) % Lx, j] +
                    S[(i-1) % Lx, j] +
                    S[i, (j+1) % Ly] +
                    S[i, (j-1) % Ly]
                )
                delta_E += 2 * J * original_spins[(i, j)] * nb

            # Metropolis criterion.
            accept = False
            if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
                accept = True
                total_accept += 1
            else:
                # Reject -> restore the original spins.
                for (i, j) in positions:
                    S[i, j] = original_spins[(i, j)]

        # Fix 4: adapt n_flips correctly.
        accept_rate = total_accept / n_chains
        if step >= burn_in:
            # Record once every 20 steps.
            if step % 10 == 0:
                accept_history.append(accept_rate)
                n_flips_history.append(n_flips)

            # Correct adaptive rule.
            if accept_rate > target_accept + 0.05:
                n_flips += 1
            elif accept_rate < target_accept - 0.05 and n_flips > 1:
                n_flips -= 1

        # Fix 5: record magnetization after burn-in.
        if step >= burn_in:
            for c in range(n_chains):
                M = np.abs(np.mean(chains[c]))
                histories[c].append(M)

    # Final statistics.
    histories = np.array(histories)
    mean_curve = np.mean(histories, axis=0)
    M_total_avg = np.mean(mean_curve)
    return M_total_avg, mean_curve, accept_history, n_flips_history

#forward process
model1 = torch.load('models/VAN_step1.pth', weights_only=False)
model1.eval()  # Set to evaluation mode.

with torch.no_grad():
    sample, x_hat = model1.sample(replicas_num)
    
    flip = sample.clone()
    
    print(flip.shape)
    cal_energy = calculate_energy_2d_torch(flip.view(replicas_num, args.L,args.L1)).sum() / (replicas_num * args.size)
    print("VAN_sample_energy:")
    print(cal_energy)
    
    Sample1D = ((sample.view(-1, args.size) + 1) / 2).clone()
    
   
    for k in range(n_iterations+1): 
        print("iteration:"+str(k))
        
        if k==0:
            cal_M = calculate_magnetization((Sample1D*2 - 1).view(replicas_num, 1, args.L,args.L1))
            
            current_statistic1 = (total_sum1 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
            #current_statistic2 = (total_sum2 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
            
            total_sum1 = current_statistic1
            #total_sum2 = current_statistic2
            total_count += replicas_num
            
            # Store results (only statistics and sample count, not raw samples).
            statistic_list1.append(current_statistic1.cpu().numpy())
            #statistic_list2.append(current_statistic2.cpu().numpy())
            sample_size_list.append(total_count)
            continue

        start_replicas = Sample1D.clone()

        Notransition_prob = 1 - args.size * args.delta_t
        for i in range(args.Tstep-1):
            flip_tensor(Sample1D, Notransition_prob)

        num_segments = (args.Tstep - 2) // jump_step + 1  # Round up

        for seg_idx in range(num_segments):
            # Load the model (make sure the model path is correct).
            if seg_idx == 0:
                model_now = torch.load(f'models/VAN_step{args.Tstep}.pth', weights_only=False)
                model_now.eval()
                if args.Tstep == (num_segments * jump_step + 1):
                    tau = jump_step * args.delta_t
                else:
                    tau = (args.Tstep - 1) % jump_step * args.delta_t
            else:
                temp_step = jump_step * (num_segments - seg_idx) + 1
                model_now = torch.load(f'models/VAN_step{temp_step}.pth', weights_only=False)
                model_now.eval()
                tau = jump_step * args.delta_t

            # Preprocess Sample1D.
            sample_scaled = (Sample1D * 2 - 1).view(replicas_num, 1, args.L, args.L1)

            # Compute neighbor samples and masks.
            SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
            SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
            Mask = torch.eye(args.size, device=args.device).expand(Sample1D.shape[0], args.size, args.size)
            SampleNeighbor1DExtend = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)

            # Reshape neighbor samples and rescale them.
            reshape_Neighbor = SampleNeighbor1DExtend.view(args.size, replicas_num, 1, args.L, args.L1)
            reshape_Neighbor = reshape_Neighbor * 2 - 1

            # Compute probabilities.
            p_next = model_now.log_prob(sample_scaled)
            flat_neighbor = reshape_Neighbor.view(-1, 1, args.L, args.L1)
            all_p1 = model_now.log_prob(flat_neighbor)
            all_p2 = torch.exp(-(torch.exp(all_p1.view(args.size, replicas_num).T - p_next.unsqueeze(1)) * tau))

            # Generate random numbers and flip states.
            random_matrix = torch.rand_like(all_p2)
            flip_mask = random_matrix > all_p2
            Sample1D[flip_mask] = 1 - Sample1D[flip_mask]

        if k % 2 == 0:
            flip, x_hat = model1.sample(replicas_num)
            flip = flip.view(-1, args.size)
            Sample1D = (flip + 1) / 2
        else:
            flip = Sample1D.clone()
            flip = flip*2-1

        #MCMC
        result_conf = start_replicas.clone()
        start_replicas = start_replicas*2 - 1

        P_acc = torch.exp(model1.log_prob(start_replicas.view(replicas_num,1,args.L,args.L1).detach()) - beta * calculate_energy_2d_torch(flip.view(replicas_num,args.L,args.L1)) - model1.log_prob(flip.view(replicas_num,1,args.L,args.L1)) + beta * calculate_energy_2d_torch(start_replicas.view(replicas_num,args.L,args.L1).detach()) )
        # Clamp values larger than 1 to 1.
        P_acc[P_acc > 1] = 1

        # Draw random numbers in [0, 1) and compare them to the probability.
        random_values = torch.rand(replicas_num).to(args.device)
        accepted_mask = random_values <= P_acc  # Boolean mask indicating whether each row is accepted.

        # Select rows according to the mask.
        result_conf[accepted_mask] = Sample1D[accepted_mask]
        
        print("correlation:")
        print(normalized_autocorrelation((start_replicas*2-1).cpu().numpy(),(result_conf*2-1).cpu().numpy()))
        
        acceptance = flip[accepted_mask].shape[0]/replicas_num
        print("acceptance:" + str(acceptance))
        
        if k % 2 == 1:
            if acceptance < 0.5 and args.Tstep > 2:
                args.Tstep -= 1
            
            if acceptance > 0.5 and args.Tstep < 100:
                args.Tstep += 1
            
        
        Sample1D = result_conf.clone()
        #cal_energy = calculate_energy_2d_torch((Sample1D*2-1).view(flip.shape[0],args.L,args.L1)).sum()/Sample1D.shape[0]
        
        #print("after_MCMC_energy:"+f"{k:.0f}"+str(cal_energy))
    
    
        #cal_M =  calculate_magnetization((Sample1D*2-1).view(replicas_num, 1, args.L, args.L1))
        cal_energy = calculate_energy_2d_torch((Sample1D*2 - 1).view(replicas_num, args.L,args.L1)).sum()/(replicas_num*args.size)
        
        cal_M = calculate_magnetization((Sample1D*2 - 1).view(replicas_num, 1, args.L,args.L1))
        print(cal_energy)
        print(cal_M)
        
        current_statistic1 = (total_sum1 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
        #current_statistic2 = (total_sum2 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
        
        total_sum1 = current_statistic1
        #total_sum2 = current_statistic2
        total_count += replicas_num
        
        # Store results (only statistics and sample count, not raw samples).
        statistic_list1.append(current_statistic1.cpu().numpy())
        #statistic_list2.append(current_statistic2.cpu().numpy())
        sample_size_list.append(total_count)
        if k % 40 == 0:
            Acc_all.append(acceptance)
        if k % 40 == 21:
            Acc_all.append(acceptance)
        print(current_statistic1)
        #print(current_statistic2)
        print(args.Tstep)
        

    Sample1D = ((sample.view(-1, args.size) + 1) / 2).clone()
    total_count = 0
    args.Tstep = 30
    
    for k in range(n_iterations+1): 
        print("iteration:"+str(k))
        
        if k==0:
            cal_M = calculate_magnetization((Sample1D*2 - 1).view(replicas_num, 1, args.L,args.L1))
            
            current_statistic2 = (total_sum2 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
            #current_statistic2 = (total_sum2 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
            
            total_sum2 = current_statistic2
            #total_sum2 = current_statistic2
            total_count += replicas_num
            
            # Store results (only statistics and sample count, not raw samples).
            statistic_list2.append(current_statistic2.cpu().numpy())
            continue

        start_replicas = Sample1D.clone()
        
        Notransition_prob = 1 - args.size * args.delta_t
        for i in range(args.Tstep-1):
            flip_tensor(Sample1D, Notransition_prob)
        
        num_segments = (args.Tstep - 2) // jump_step + 1  # Round up

        for seg_idx in range(num_segments):
            # Load the model (make sure the model path is correct).
            if seg_idx == 0:
                model_now = torch.load(f'models/VAN_step{args.Tstep}.pth', weights_only=False)
                model_now.eval()
                if args.Tstep == (num_segments * jump_step + 1):
                    tau = jump_step * args.delta_t
                else:
                    tau = (args.Tstep - 1) % jump_step * args.delta_t
            else:
                temp_step = jump_step * (num_segments - seg_idx) + 1
                model_now = torch.load(f'models/VAN_step{temp_step}.pth', weights_only=False)
                model_now.eval()
                tau = jump_step * args.delta_t

            # Preprocess Sample1D.
            sample_scaled = (Sample1D * 2 - 1).view(replicas_num, 1, args.L, args.L1)

            # Compute neighbor samples and masks.
            SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
            SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
            Mask = torch.eye(args.size, device=args.device).expand(Sample1D.shape[0], args.size, args.size)
            SampleNeighbor1DExtend = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)

            # Reshape neighbor samples and rescale them.
            reshape_Neighbor = SampleNeighbor1DExtend.view(args.size, replicas_num, 1, args.L, args.L1)
            reshape_Neighbor = reshape_Neighbor * 2 - 1

            # Compute probabilities.
            p_next = model_now.log_prob(sample_scaled)
            flat_neighbor = reshape_Neighbor.view(-1, 1, args.L, args.L1)
            all_p1 = model_now.log_prob(flat_neighbor)
            all_p2 = torch.exp(-(torch.exp(all_p1.view(args.size, replicas_num).T - p_next.unsqueeze(1)) * tau))

            # Generate random numbers and flip states.
            random_matrix = torch.rand_like(all_p2)
            flip_mask = random_matrix > all_p2
            Sample1D[flip_mask] = 1 - Sample1D[flip_mask]


        flip = Sample1D.clone()
        flip = flip*2-1

        #MCMC
        result_conf = start_replicas.clone()
        start_replicas = start_replicas*2 - 1

        P_acc = torch.exp(model1.log_prob(start_replicas.view(replicas_num,1,args.L,args.L1).detach()) - beta * calculate_energy_2d_torch(flip.view(replicas_num,args.L,args.L1)) - model1.log_prob(flip.view(replicas_num,1,args.L,args.L1)) + beta * calculate_energy_2d_torch(start_replicas.view(replicas_num,args.L,args.L1).detach()) )
        # Clamp values larger than 1 to 1.
        P_acc[P_acc > 1] = 1

        # Draw random numbers in [0, 1) and compare them to the probability.
        random_values = torch.rand(replicas_num).to(args.device)
        accepted_mask = random_values <= P_acc  # Boolean mask indicating whether each row is accepted.

        # Select rows according to the mask.
        result_conf[accepted_mask] = Sample1D[accepted_mask]
        
        print("correlation:")
        print(normalized_autocorrelation((start_replicas*2-1).cpu().numpy(),(result_conf*2-1).cpu().numpy()))
        
        acceptance = flip[accepted_mask].shape[0]/replicas_num
        print("acceptance:" + str(acceptance))
        

        if acceptance < 0.5 and args.Tstep > 2:
            args.Tstep -= 1
        
        if acceptance > 0.5 and args.Tstep < 100:
            args.Tstep += 1
            
        
        Sample1D = result_conf.clone()
        #cal_energy = calculate_energy_2d_torch((Sample1D*2-1).view(flip.shape[0],args.L,args.L1)).sum()/Sample1D.shape[0]
        
        #print("after_MCMC_energy:"+f"{k:.0f}"+str(cal_energy))
    
    
        #cal_M =  calculate_magnetization((Sample1D*2-1).view(replicas_num, 1, args.L, args.L1))
        cal_energy = calculate_energy_2d_torch((Sample1D*2 - 1).view(replicas_num, args.L,args.L1)).sum()/(replicas_num*args.size)
        print(cal_energy)
        cal_M = calculate_magnetization((Sample1D*2 - 1).view(replicas_num, 1, args.L,args.L1))
        
        current_statistic2 = (total_sum2 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
        
        total_sum2 = current_statistic2
        total_count += replicas_num
        
        # Store results (only statistics and sample count, not raw samples).
        statistic_list2.append(current_statistic2.cpu().numpy())
        if k % 10 == 0:
            Acc_dm.append(acceptance)
    
    start_replicas = ((sample.view(-1, args.size) + 1) / 2).clone()
    total_count = 0
    result_conf = start_replicas.clone()
    Sample1D = (sample.view(-1, args.size) + 1) / 2
    
    for k in range(n_iterations+1): 
        if k==0:
            cal_M = calculate_magnetization((Sample1D*2 - 1).view(replicas_num, 1, args.L,args.L1))
            
            current_statistic3 = (total_sum3 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
            #current_statistic2 = (total_sum2 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
            
            total_sum3 = current_statistic3
            #total_sum2 = current_statistic2
            total_count += replicas_num
            
            # Store results (only statistics and sample count, not raw samples).
            statistic_list3.append(current_statistic3.cpu().numpy())
            continue

        flip, x_hat = model1.sample(replicas_num)
        flip = flip.view(-1, args.size)
        Sample1D = (flip + 1) / 2

        P_acc = torch.exp(model1.log_prob((result_conf*2 - 1).view(replicas_num,1,args.L,args.L1).detach()) - beta * calculate_energy_2d_torch(flip.view(replicas_num,args.L,args.L1)) - model1.log_prob(flip.view(replicas_num,1,args.L,args.L1)) + beta * calculate_energy_2d_torch((result_conf*2 - 1).view(replicas_num,args.L,args.L1).detach()) )
        # Clamp values larger than 1 to 1.
        P_acc[P_acc > 1] = 1

        # Draw random numbers in [0, 1) and compare them to the probability.
        random_values = torch.rand(replicas_num).to(args.device)
        accepted_mask = random_values <= P_acc  # Boolean mask indicating whether each row is accepted.

        # Select rows according to the mask.
        result_conf[accepted_mask] = Sample1D[accepted_mask]
        
        '''
        print("correlation:")
        print(normalized_autocorrelation((start_replicas*2-1).cpu().numpy(),(result_conf*2-1).cpu().numpy()))
        '''
        acceptance = flip[accepted_mask].shape[0]/replicas_num
        print("acceptance:" + str(acceptance))
        
        
        Sample1D = result_conf.clone()
        #cal_energy = calculate_energy_2d_torch((Sample1D*2-1).view(flip.shape[0],args.L,args.L1)).sum()/Sample1D.shape[0]
        
        #print("after_MCMC_energy:"+f"{k:.0f}"+str(cal_energy))
    
    
        #cal_M =  calculate_magnetization((Sample1D*2-1).view(replicas_num, 1, args.L, args.L1))
        cal_energy = calculate_energy_2d_torch((Sample1D*2 - 1).view(replicas_num, args.L,args.L1)).sum()/(replicas_num*args.size)
        
        cal_M = calculate_magnetization((Sample1D*2 - 1).view(replicas_num, 1, args.L,args.L1))
        print(cal_energy)
        print(cal_M)
        
        current_statistic3 = (total_sum3 * total_count + cal_M * replicas_num) / (total_count + replicas_num)
         
        total_sum3 = current_statistic3
 
        total_count += replicas_num
        
        # Store results (only statistics and sample count, not raw samples).
        statistic_list3.append(current_statistic3.cpu().numpy())
        if k % 10 == 0:
            Acc_nmcmc.append(acceptance)
        print(current_statistic3)
    print("LMMC")
    M_avg, mean_curve, acc_rates, flips_hist = ising_2d_rectangle(
        Lx=args.L, Ly=args.L1, beta=beta,
        n_steps=n_iterations, burn_in=0,
        n_chains=replicas_num, init_n_flips=30, init_sanmples=sample.clone()
    )
    Sample1D = (sample.view(-1, args.size) + 1) / 2 
    cal_M = calculate_magnetization((Sample1D*2 - 1).view(replicas_num, 1, args.L,args.L1))
    mean_curve = np.insert(mean_curve, 0, cal_M.detach().cpu().numpy())
    cum_mean = np.cumsum(mean_curve) / (np.arange(len(mean_curve)) + 1)
