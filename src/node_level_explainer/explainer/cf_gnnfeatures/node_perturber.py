from omegaconf import DictConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Tuple

from ...utils.utils import discretize_tensor, discretize_to_nearest_integer
from ...perturber.pertuber import Perturber   
from src.datasets.dataset import DataInfo



class RangeSigmoid(nn.Module):
    def __init__(self, a: float, b: float):
        super(RangeSigmoid, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x: Tensor) -> Tensor:
        return self.a + (self.b - self.a) * torch.sigmoid(x)
    
    
class NodePerturber(Perturber):

    def __init__(self, 
                    cfg: DictConfig, 
                    model: nn.Module, 
                    graph: Data,
                    datainfo: DataInfo,
                    device: str = "cuda") -> None:
        
        super().__init__(cfg=cfg, model=model)
        
        
        self.device = device
        
        # Dataset characteristics
        self.num_classes = datainfo.num_classes
        self.num_nodes = graph.x.shape[0]
        self.num_features = datainfo.num_features
        self.min_range = datainfo.min_range.to(device)
        self.max_range = datainfo.max_range.to(device)
        
        # Explainer characteristics
        self.discrete_features_addition: bool = True
        self.discrete_features_mask: Tensor = datainfo.discrete_mask.to(device)
        self.continous_features_mask: Tensor = 1 - datainfo.discrete_mask.to(device)
        # Model's parameters
        self.P_x = Parameter(torch.zeros(self.num_nodes, self.num_features, device=self.device))
        
        # Graph's components
        self.edge_index = graph.edge_index
        self.x = graph.x
    
    def forward(self, V_x: Tensor) -> Tensor:
        """
        Forward pass for the NodePerturber.

        Args:
        V_x (Tensor): The input feature tensor.
        adj (Tensor): The adjacency matrix.

        Returns:
        Tensor: The output of the model after applying perturbations.
        """
        
        
        discrete_perturbation = self.discrete_features_mask * torch.clamp(self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x) + V_x, min=self.min_range, max=self.max_range)
        
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        
        V_pert = discrete_perturbation + continuous_perturbation

        return self.model(V_pert, self.edge_index)

    
    def forward_prediction(self, V_x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward prediction for the NodePerturber.

        Args:
        V_x (Tensor): The input feature tensor.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: The output of the model, the perturbed features, and the perturbation tensor.
        """
        discrete_perturbation = self.discrete_features_mask * discretize_to_nearest_integer(self.min_range + (self.max_range - self.min_range) * F.tanh(self.P_x) + V_x)
        
        discrete_perturbation = torch.clamp(discrete_perturbation, min=self.min_range, max=self.max_range)
        
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        
        V_pert = discrete_perturbation + continuous_perturbation

        out = self.model(V_pert, self.edge_index)
        return out, V_pert, self.P_x
    
    
    def loss(self, graph: Data, output: Tensor, y_node_non_differentiable) -> Tuple[Tensor, dict, Tensor]:
        node_to_explain = graph.new_idx
        # Get the prediction vector for the node to explain.
        y_node_predicted = output[node_to_explain].unsqueeze(0)  # shape: (1, num_classes)
        y_target = graph.targets[node_to_explain].unsqueeze(0).float()  # shape: (1, num_classes)

        # Get binary prediction by thresholding the sigmoid of logits.
        predicted_binary = (torch.sigmoid(y_node_predicted) > 0.5).float()
        # Check if the prediction matches the target exactly (all elements equal)
        correct = torch.all(predicted_binary == y_target).float()
        # If correct, we set a scaling factor (constant) to 0; if not, to 1
        constant = 1.0 - correct

        # Use BCEWithLogitsLoss for multilabel classification
        loss_pred = F.binary_cross_entropy_with_logits(y_node_predicted, y_target)
        loss_discrete = F.l1_loss(
            graph.x * self.discrete_features_mask,
            torch.clamp(self.discrete_features_mask * F.tanh(self.P_x) + graph.x, 0, 1)
        )
        loss_continue = F.mse_loss(
            graph.x * self.continous_features_mask,
            self.continous_features_mask * (self.P_x + graph.x)
        )

        loss_total = constant * loss_pred + loss_discrete + loss_continue

        return loss_total, self.edge_index

