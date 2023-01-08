import numpy as np
import torch
import geomloss # also needs: pip install pykeops
import time
import os 

def test_sinkhorn_time(M):

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    
    device = torch.device(dev) 
    start = time.time()

    coords0, coords1 = np.random.rand(2, M, 2)
    weights0, weights1 = np.random.rand(2, M) 

    loss = geomloss.SamplesLoss(loss='sinkhorn', p = 1, blur= 0.05, verbose=False)

    labels0t = torch.tensor(np.arange(0, coords0.shape[0], 1), dtype=torch.int).to(device)
    labels1t = torch.tensor(np.arange(0, coords1.shape[0], 1), dtype=torch.int).to(device)

    coords0t = torch.tensor(coords0, dtype=torch.float).to(device)
    coords1t = torch.tensor(coords0, dtype=torch.float).to(device)

    weights0t = torch.tensor(weights0, dtype=torch.float).to(device)
    weights1t = torch.tensor(weights1, dtype=torch.float).to(device)

    sinkhornLoss = loss(labels0t, weights0t, coords0t, labels1t, weights1t, coords1t)

    print(labels0t.shape)
    print(labels1t.shape)
    print(coords0t.shape)
    print(coords1t.shape)
    print(weights0t.shape)
    print(weights1t.shape)


    end = time.time()
    run_time = (end - start)

    print(f'Runtime: {run_time:.1f} sec')
    print(f'Distance: {sinkhornLoss.item():.3f}')

def main():

    #os.environ['CXX'] = 'g++-8' # does not appear to make a difference

    M = input('Input number of cells (e.g. 259200 for full prio grid):')
    M = int(M)
    
    test_sinkhorn_time(M)

#full prio grid is 360×720 = 259200 cells.
#M = 4096 # =64x64 #10000 # 100000 = 57m 0.2 sec on laptop cpu

if __name__ == '__main__':
    main()
