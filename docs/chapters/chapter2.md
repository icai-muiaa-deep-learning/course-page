---
icon: lucide/library
---
  
# Chapter 2: Foundations of Neural Networks

This chapter explains the foundations of neural networks. It will combine theory with practical examples using PyTorch, so you can learn to implement the concepts that we will study during the course.

## Adding Hidden Layers

We can overcome the limitations of linear models by incorporating one or more hidden layers. The easiest way to do this is to stack many fully connected layers on top of one another. Each layer feeds into the layer above it, until we generate outputs. We can think of the first layers as our representation and the final layer as our linear predictor. This architecture is commonly called a multilayer perceptron, often abbreviated as MLP, as it can be visualized in this [Figure](#fig-hidden_layers).

![Hidden Layers](assets/images/hidden_layers.png){#fig-hidden_layers}

This MLP has four inputs, three outputs, and its hidden layer contains five hidden units. Since the input layer does not involve any calculations, producing outputs with this network requires implementing the computations for both the hidden and output layers; thus, the number of layers in this MLP is two. Note that both layers are fully connected. Every input influences every neuron in the hidden layer, and each of these in turn influences every neuron in the output layer.

## Activation Functions

Activations functions are non-linear functions that transform the output of the layers in a deep learning model. These non-linear behavior is precisely what allows model complex behavior. 

!!! note

    Note that without non-linear layers, stacking hidden layers would be useless, since it would be exactly the same as having one single linear layer. This is because several linear transformation are equal to a single transformation.

### ReLU

This is the most widely used non-linear layer. The main benefit of teh ReLU is that it does not saturate with the input in the positive part. The only negative part is that is not differentiable in the zero. However, this usually does not have a significative impact since 

The formula is the following:

$$
ReLU(x) = max(x, 0)
$$

And you can also visualize this function in this image:

<figure markdown="span">
  ![ReLU Layer](assets/images/relu.png)
  <figcaption>ReLU layer</figcaption>
</figure>

About the implementation, you can use the two different implementations that you have in PyTorch for layers without parameters:

```python
import torch
import torch.nn.functional as F

# Use ReLU as a class
layer = torch.nn.ReLU()
outputs: torch.Tensor = layer(inputs)

# Use it as a function
outputs = F.relu(inputs)
```

There is no difference in these two implementations, they yield exactly the same result. Also take into account that you can reuse the same layer throughout the model. The main reason for this is that under the hood PyTorch saves in the computational graph the operations that it has executed. That is why there is no difference between using functional or a class instance of `torch.nn` or reusing the same instance. In the following example you can visualize this precise situation.

??? example

    ```python
    import torch

    # Using different instances
    class MyModule(torch.nn.Module):
      def __init__(
        self, 
        input_dim: int, 
        hidden_dims: tuple[int, int], 
        output_dim: int
      ) -> None:
        # Call super class constructor
        super().__init__()

        # Define layers
        self.layers = torch.nn.Sequential([
          torch.nn.Linear(input_dim, hidden_dims[0]),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_dims[1], output_dim)
        ])

        return None

    # Reusing the same instance
    class MyModule(torch.nn.Module):
      def __init__(
        self, 
        input_dim: int, 
        hidden_dims: tuple[int, int], 
        output_dim: int
      ) -> None:
        # Call super class constructor
        super().__init__()

        # Define ReLU layer
        relu = torch.nn.ReLU()

        # Define layers
        self.layers = torch.nn.Sequential([
          torch.nn.Linear(input_dim, hidden_dims[0]),
          relu,
          torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
          relu,
          torch.nn.Linear(hidden_dims[1], output_dim)
        ])

        return None
    ```

### Dead ReLUs

However, this layer also has a main problem that can prevent your model from learning. 
