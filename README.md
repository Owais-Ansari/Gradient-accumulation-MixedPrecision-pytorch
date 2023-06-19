## Gradient-accumulation-MixedPrecision-pytorch



Gradient accumulation is a technique used in machine learning and deep learning to mitigate the limitations of limited GPU memory when training large models or processing large batches of data. It involves accumulating gradients over multiple mini-batches before performing a weight update step.

During the training process, the model parameters are updated using gradients calculated from a mini-batch of training examples. Typically, these gradients are computed and applied in each training iteration, which requires loading the entire mini-batch into GPU memory. However, if the mini-batch size is large or the model is memory-intensive, it may exceed the available GPU memory, leading to out-of-memory errors.

Gradient accumulation addresses this issue by dividing a mini-batch into smaller sub-batches and computing the gradients separately for each sub-batch. Instead of updating the model parameters immediately, the gradients from each sub-batch are accumulated over multiple iterations. After accumulating gradients for a predetermined number of iterations or sub-batches, the accumulated gradients are used to update the model parameters in a single weight update step.


**Without gradient accumulation**
```
import torch

scaler = torch.cuda.amp.GradScaler()

for batch_idx, data in enumerate(trainloader):
    inputs =  data['image']
    targets = data['label']
   
    inputs = inputs.to(device)
    targets = targets.to(device)
   
    with torch.set_grad_enabled(True):
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    #Compute gradient and do backpropagation step
    scaler.scale(loss).backward()
   
    #Weights update

    #Updating the optimizer state after back propagation
    scaler.step(optimizer)
    #Updates the scale (grad-scale) for next iteration
    scaler.update()
```

The process can be summarized as follows:

- Initialize the model parameters.
- Divide a mini-batch of training examples into smaller sub-batches.
- Iterate through the sub-batches and perform the following steps:
    a. Load a sub-batch into GPU memory.
    b. Forward pass: Compute the loss and the gradients for the current sub-batch.
    c. Accumulate the gradients by adding them to a running total.
    d. Repeat steps a-c for the desired number of iterations or sub-batches.
- After accumulating gradients over the desired number of iterations or sub-batches, perform a weight update step using the accumulated gradients.
- Repeat steps 2-4 for the remaining mini-batches or until the desired number of training iterations is reached.

**With gradient accumulation**
```
import torch

scaler = torch.cuda.amp.GradScaler()
for batch_idx, data in enumerate(trainloader):
    inputs =  data['image']
    targets = data['label']
   
    inputs = inputs.to(device)
    targets = targets.to(device)
   
    with torch.set_grad_enabled(True):
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    #Gradient Accumulation
    loss =  loss / accu_factor
    #Compute gradient and do backpropagation step
    scaler.scale(loss).backward()
   
    #Weights update
    if ((batch_idx + 1) % accum_iter ==0) or (batch_idx + 1 == len(trainloader)):
       

        #Updating the optimizer state after back propagation
        scaler.step(optimizer)
        #Updates the scale (grad-scale) for next iteration
        scaler.update()
```
By accumulating gradients over multiple iterations, gradient accumulation reduces the memory requirements during each iteration, enabling training with larger batch sizes or models that would otherwise exceed GPU memory limitations. It can help improve training efficiency and achieve better convergence, although it may introduce a slight delay in the weight updates due to the accumulation step.
