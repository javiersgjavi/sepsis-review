import gpytorch
import torch
import torch.nn as nn


# Exact Hadamard Multi-task Gaussian Process Model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, output_device, num_tasks=2, n_devices=1, kernel='rbf', mode='normal', keops=False):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.output_device = output_device
        self.mean_module = gpytorch.means.ConstantMean()
        valid_kernels = ['rbf', 'ou']
        if kernel not in valid_kernels:
            raise ValueError(f'parsed kernel: {kernel} not among implemented kernels: {valid_kernels}')
        elif kernel == 'rbf':
            if keops:
                base_covar_module = gpytorch.kernels.keops.RBFKernel()
            else:
                base_covar_module = gpytorch.kernels.RBFKernel()
        elif kernel == 'ou':
            if keops:
                base_covar_module = gpytorch.kernels.keops.MaternKernel(nu=0.5)
            else:
                base_covar_module = gpytorch.kernels.MaternKernel(nu=0.5)

        if n_devices > 1: #in multi-gpu setting
            if mode != 'normal':
                raise NotImplementedError('scalable-GPs and multi-device have not been implemented!')
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_covar_module, device_ids=range(n_devices),
                output_device=self.output_device)
            #self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=3)
            base_task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=3)
            self.task_covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_task_covar_module, device_ids=range(n_devices),
                output_device=self.output_device)
        else:
            if mode == 'kiss':
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                            gpytorch.kernels.RBFKernel(), grid_size=30, num_dims=1
                ) 
            elif mode == 'normal':
                self.covar_module = base_covar_module #gpytorch.kernels.RBFKernel()
            else:
                raise NotImplementedError(f'Current mode {mode} not among implemented ones: [normal, kiss] ')
            self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=3)
   
 
    def forward(self,x,i):

        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        
        res = gpytorch.distributions.MultivariateNormal(mean_x, covar)

        return res


# MGP Layer for Neural Network using MultitaskGPModel
class MGP_Layer(MultitaskGPModel):
    def __init__(self, train_x, train_y, likelihood, num_tasks, n_devices, output_device, kernel, mode, keops=False):
        super().__init__(train_x, train_y, likelihood, output_device, num_tasks, n_devices, kernel, mode, keops)
        #we don't intialize with train data for more flexibility
        likelihood.train()

    def forward(self, inputs, indices):
        return super(MGP_Layer, self).forward(inputs, indices)

    def condition_on_train_data(self, inputs, indices, values):
        self.set_train_data(inputs=(inputs, indices), targets=values, strict=False)

# Custom GP Adapter:
# MGP Adapter
class GPAdapter(nn.Module):
    def __init__(self, train_x, train_y, n_mc_smps, sampling_type, likelihood, num_tasks, n_devices, output_device, kernel, mode, keops=False):
        super(GPAdapter, self).__init__()
        self.n_mc_smps = 1 
        # num_tasks includes dummy task for padedd zeros
        self.n_tasks = num_tasks
        self.mgp = MGP_Layer(train_x, train_y, likelihood, num_tasks, n_devices, output_device, kernel, mode, keops)
        #more generic would be something like: self.clf = clf(n_input_dims) #e.g. SimpleDeepModel(n_input_dims)
        self.sampling_type = sampling_type # 'monte_carlo', 'moments'
        self.return_gp = False

    def forward(self, *data):
        with torch.no_grad():
            """
            The GP Adapter takes input data as a list of 5 torch tensors (3 for train points, 2 for prediction points)
                - inputs: input points of time grid (batch, timesteps, 1)
                - indices: indices of task or channels (batch, timesteps, 1)
                - values: values (or targets) of actual observations (batch, timesteps)
                - test_inputs: query points in time (batch, timesteps, 1)
                - test_indices: query tasks for given point in time (batch, timesteps, 1)
            """
            posterior = self.gp_forward(*data)

            inputs, indices, values, test_inputs, test_indices, valid_lengths = data

            # Get regularly-spaced "latent" timee series Z:
            if self.sampling_type == 'monte_carlo':
                # draw sample in MGP format (all tasks in same dimension)
                Z = self.draw_samples(posterior, self.n_mc_smps)
                valid_lengths = valid_lengths.repeat(self.n_mc_smps)
            elif self.sampling_type == 'moments':
                # feed moments of GP posterior to classifier (mean, variance)
                Z = self.feed_moments(posterior)

            Z = self._channel_reshape(Z)

        return Z

    def gp_forward(self, *data):
        # Unpack data:
        inputs, indices, values, test_inputs, test_indices, _ = data

        # Condition MGP on training data:
        #self.mgp.condition_on_train_data(inputs, indices, values)

        # Return posterior distribution:
        return self.mgp.forward(test_inputs, test_indices)

    def draw_samples(self, posterior, n_mc_smps):
        # Draw monte carlo samples (with gradient) from posterior:
        return posterior.rsample(torch.Size([n_mc_smps]))  # mc_samples form a new (outermost) dimension

    def feed_moments(self, posterior):
        """
        Get mean and variance of posterior and concatenate them along the channel dimension for feeding them to feed the clf
        """
        mean = posterior.mean
        var = posterior.variance
        # return torch.stack([mean, var], axis=0) # stacked in a innermost dim to replace mc_smp dim
        return torch.cat([mean, var], axis=-1)  # concat in channel dim
        
    def parameters(self):
        return list(self.mgp.parameters()) + list(self.clf.parameters())

    def train(self, mode=True):
        """
        only set classifier to train mode, MGP always in eval mode for posterior draws
        """
        if mode:
            super().train()
            self.mgp.eval()
        else:
            super().train(False)

    def eval(self):
        """
        eval simply calls eval of super class (which in turn activates train with False)
        """
        super().eval()
   
    def _channel_reshape(self, X, return_gp=False):
        """
        reshaping function required as hadamard MGP's output format is not directly compatible with subsequent network
        """
        # first check if we have to doubele the number of channels
        if self.sampling_type == 'moments':
            channel_dim = 2*self.n_tasks
        else:
            channel_dim = self.n_tasks

        X_reshaped = X.view(X.shape[:-1]                    # batch-dim (or mc_smp and batch_dim)
            + torch.Size([channel_dim])                     # channel-dim
            + torch.Size([int(X.shape[-1] / channel_dim)])  # time steps dim
        )
        # finally, swap last two dims: timestep and channel dim for Signature Augmentations
        X_reshaped = X_reshaped.transpose(-2,-1)
        
        if self.sampling_type == 'monte_carlo':
            X_reshaped = X_reshaped.flatten(0,1) #SigNet requires 3 dim setup, so we flatten out the mc dimension with batch
        if return_gp:
            return X_reshaped, X
        else: 
            return X_reshaped

