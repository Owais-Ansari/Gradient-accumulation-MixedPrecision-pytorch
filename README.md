# Gradient-accumulation-MixedPrecision-pytorch

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

```
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
