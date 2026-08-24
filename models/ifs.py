"""Affine Iterated Function System (IFS) model in PyTorch.

The SVD-format affine parameterization (per-map rotation angles theta1, theta2 and
singular values s1, s2) and the chaos-game sampling of points follow the
synthetic-fractal pre-training line (Anderson et al., "Improving Fractal
Pre-training", WACV 2022) and the public implementation of Tu et al., "Learning
Fractals by Gradient Descent" (AAAI 2023).

Note: in the current pipeline this class is used only to sample the affine-map
parameters (theta) that define each dataset instance. The point-cloud and density
rendering used for training and evaluation is performed by the separate
differentiable renderer, not by the trajectory-generation methods of this class
(forward / generate_ifs_trajectories).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AffineIFS(nn.Module):
    """
    Affine Iterated Function System (IFS) model for generating fractals.
    
    This class implements an IFS using affine transformations parameterized in SVD format.
    Each transformation is defined by rotation angles (theta1, theta2), scaling factors (s_1, s_2),
    and translation parameters (tx, ty).
    """
    
    def __init__(self, num_transforms, use_htan=True, contractive_init=True, name="Unnamed IFS"):
        """
        Initialize the AffineIFS model.
        
        Args:
            num_transforms (int): Number of affine transformations in the IFS
            use_htan (bool): Whether to use hyperbolic tangent activation for scaling factors s_1 and s_2
            contractive_init (bool): Whether to initialize with contractive parameters
            init_seed (int): Random seed for initialization
            name (str): Name identifier for this IFS instance
        """
        super(AffineIFS, self).__init__()
        
        self.num_transforms = num_transforms
        self.use_htan = use_htan
        self.name = name

        # Set random seeds for reproducible initialization
        # np.random.seed(init_seed)
        # torch.manual_seed(init_seed)
        
        # Embedding layers for transformation parameters
        # ifs_w stores: [theta1, theta2, sigma1, sigma2] for each transformation
        self.ifs_w = nn.Embedding(num_transforms, 4)
        # ifs_b stores translation parameters [tx, ty] for each transformation
        self.ifs_b = nn.Embedding(num_transforms, 2)
        
        # Internal logits for transformation probabilities (learnable parameters)
        # These will be converted to probabilities via softmax
        self.prob_logits = nn.Parameter(torch.zeros(num_transforms))
        
        # Initialize parameters
        self._initialize_parameters(contractive_init)
    
    @property
    def probs(self):
        """Get the current transformation probabilities via softmax."""
        return torch.softmax(self.prob_logits, dim=0)
    
    def set_probs(self, probabilities):
        """
        Set transformation probabilities by computing corresponding logits.
        
        Args:
            probabilities (torch.Tensor or list): Probability values that sum to 1
        """
        if isinstance(probabilities, list):
            probabilities = torch.tensor(probabilities, dtype=torch.float32)
        
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("probabilities must be a torch.Tensor or list")
        
        # Ensure probabilities are valid
        probabilities = probabilities.float()
        if torch.any(probabilities <= 0):
            # Add small epsilon to avoid log(0)
            probabilities = torch.clamp(probabilities, min=1e-8)
        
        # Normalize to ensure they sum to 1
        probabilities = probabilities / probabilities.sum()
        
        # Compute logits using the inverse softmax (log probabilities)
        # We use log(p) - log(mean(p)) to avoid the indeterminacy of softmax inverse
        log_probs = torch.log(probabilities)
        logits = log_probs - log_probs.mean()
        
        # Set the logits
        with torch.no_grad():
            self.prob_logits.data = logits.to(self.prob_logits.device)
    
    def _initialize_parameters(self, contractive_init):
        """Initialize the transformation parameters."""
        if contractive_init:
            for index in range(self.num_transforms):
                # Random rotation angles
                th_range = 0.5 * np.pi
                theta1 = np.random.rand(1).item()  *2*th_range - th_range
                theta2 = np.random.rand(1).item() *2*th_range - th_range
                # Random scaling factors #  0.3< a< 0.6
                s1 = 0.3 + 0.3 * np.random.rand(1).item()
                s2 = 0.3 + 0.3 * np.random.rand(1).item()

                # Set transformation matrix parameters
                params = torch.from_numpy(
                    np.array([theta1, theta2, s1, s2]).astype(np.float32)
                )
                self.ifs_w.weight.data[index].copy_(params.view(4))
                
                # set b so that a fixed point x* s.t. x* = W*x* + b lies within [0,1]x[0,1]
                # i.e. b = (I-W)x*
                all_w, _ = self.make_matrices_from_svdformat()
                # r1 = self.make_rotation_matrix(torch.tensor(theta1).unsqueeze(0))
                # r2 = self.make_rotation_matrix(torch.tensor(theta2).unsqueeze(0))
                # sig = self.make_diagonal_matrix(torch.tensor(s1).unsqueeze(0),
                #                                 torch.tensor(s2).unsqueeze(0))
                w = all_w[index]
                x_star = torch.from_numpy(np.random.rand(2).astype(np.float32)) # random point in [0,1]x[0,1]
                b = x_star - torch.matmul(x_star,w)

                # b = torch.from_numpy(0.4 * np.random.rand(2).astype(np.float32) + 0.3)
                self.ifs_b.weight.data[index].copy_(b.view(2))
    
    def make_rotation_matrix(self, theta):
        """Create a 2D rotation matrix from angle theta."""
        r_mat = torch.stack([
            torch.cos(theta), -torch.sin(theta),
            torch.sin(theta), torch.cos(theta)
        ]).view(2, 2)
        return r_mat
    
    def make_diagonal_matrix(self, s1, s2):
        """Create a diagonal matrix from two values."""
        # Ensure s1 and s2 are 0-dimensional tensors (scalars)
        if s1.dim() > 0:
            s1 = s1.squeeze()
        if s2.dim() > 0:
            s2 = s2.squeeze()
        
        zero = torch.zeros_like(s1)
        d_mat = torch.stack([
            s1, zero,
            zero, s2
        ]).view(2, 2)
        return d_mat
    
    def make_matrices_from_svdformat(self):
        """
        Convert SVD format parameters to transformation matrices.
        
        Returns:
            all_w (torch.Tensor): Transformation matrices for each IFS function
            all_sgv (torch.Tensor): Singular values for each transformation
        """
        all_w = []
        all_sgv = []
        
        for i in range(self.ifs_w.weight.shape[0]):
            theta_1, theta_2, s_1, s_2 = self.ifs_w.weight[i]
            
            # Create rotation matrices
            r_mat1 = self.make_rotation_matrix(theta_1.unsqueeze(0))
            r_mat2 = self.make_rotation_matrix(theta_2.unsqueeze(0))
            
            # Create scaling matrix with sigmoid/softplus activation
            if self.use_htan:
                sig_mat = self.make_diagonal_matrix(
                    torch.tanh(s_1.unsqueeze(0)),
                    torch.tanh(s_2.unsqueeze(0))
                )
            else:
                sig_mat = self.make_diagonal_matrix(
                    s_1.unsqueeze(0),
                    s_2.unsqueeze(0)
                )
            
            all_sgv.append(torch.diag(sig_mat).unsqueeze(0))
                        
            # Combine matrices: W = R1 * S * R2 
            w = torch.matmul(torch.matmul(r_mat1, sig_mat), r_mat2).T # transpose for later bmm
            all_w.append(w.unsqueeze(0))
        
        return torch.cat(all_w, dim=0), torch.cat(all_sgv, dim=0)
    
    def sample_transformations(self, num_samples, num_steps):
        """
        Sample transformation sequences based on probabilities.
        
        Args:
            num_samples (int): Number of sample sequences to generate
            num_steps (int): Number of steps in each sequence
            
        Returns:
            seqs (torch.Tensor): Sampled transformation indices [num_samples, num_steps]
        """
              
        # Sample transformation indices
        random_ints = torch.multinomial(
            self.probs , num_samples * num_steps, replacement=True
        )
        seqs = random_ints.view(num_samples, num_steps)        
        
        return seqs
    
    def transform_points(self, seqs, initial_cs):
        """
        Apply transformation sequences to starting coordinates.
        
        Args:
            seqs (torch.Tensor): Transformation sequences [batch_size, num_steps]
            initial_cs (torch.Tensor): Initial coordinates [batch_size, 2] or [batch_size, 1, 2]

        Returns:
            all_cs (torch.Tensor): All transformed coordinates [batch_size, num_steps, 2]
            all_sgv (torch.Tensor): Singular values for transformations
        """
        all_w, all_sgv = self.make_matrices_from_svdformat()
        n_iters = seqs.shape[1]

        curr_cs = initial_cs.view(-1, 1, 2)
        all_cs = []
        
        for i in range(n_iters):
            ids = seqs[:, i].long()
            w = all_w[ids]
            b = self.ifs_b(ids).view(-1, 1, 2)
            curr_cs = torch.bmm(curr_cs, w) + b
            all_cs.append(curr_cs)
        
        return torch.cat(all_cs, dim=1), all_sgv


    def compute_jacobian_determinants(self):
        """
        Compute the absolute determinants of the Jacobian matrices for all transformations.
        
        For affine transformation T(x) = Wx + b, the Jacobian is simply W.
        Since W = U*S*V^T where U and V^T are rotation matrices (det = ±1),
        we have |det(W)| = |s1 * s2|
        
        Returns:
            torch.Tensor: Absolute determinants for each transformation [num_transforms]
        """
        det_values = []
        
        for i in range(self.num_transforms):
            # Get scaling parameters
            _, _, s_1, s_2 = self.ifs_w.weight[i]
            
            # Apply activation to get actual scaling values
            if self.use_htan:
                s1 = torch.tanh(s_1)
                s2 = torch.tanh(s_2)
            else:
                s1 = s_1
                s2 = s_2
            
            # Compute absolute determinant: |det(W)| = |s1 * s2|
            det = torch.abs(s1 * s2)
            det_values.append(det)
        
        return torch.stack(det_values)


    def forward(self, initial_cs, num_steps):
        """
        Forward pass: sample transformations and apply them to generate fractal points.
        
        Args:
            initial_cs (torch.Tensor): Initial coordinates [batch_size, 2] or [batch_size, 1, 2]
            num_samples (int): Number of sample trajectories
            num_steps (int): Number of iteration steps
            
        Returns:
            coords (torch.Tensor): Generated fractal coordinates [num_samples, num_steps, 2]
            all_sgv (torch.Tensor): Singular values for the transformations
            seqs (torch.Tensor): Sampled transformation sequences [num_samples, num_steps]
        """

        num_samples = initial_cs.shape[0]
        # Sample transformation sequences
        seqs = self.sample_transformations(num_samples, num_steps)
        
        # Apply transformations
        coords, all_sgv = self.transform_points(seqs, initial_cs)

        return coords, all_sgv, seqs
    
    @staticmethod
    def matrix_to_svdformat(matrix):
        """
        Convert a 2x2 transformation matrix to SVD format parameters.
        
        Args:
            matrix (torch.Tensor): 2x2 transformation matrix
            
        Returns:
            tuple: (theta1, theta2, s1, s2) parameters for SVD format
        """
        # Perform SVD: M = U * S * V^T
        U, S, Vt = torch.linalg.svd(matrix)
        
        # Handle case where U has determinant -1 (includes reflection)
        det_U = torch.det(U)
        if det_U < 0:
            # Flip the sign of the second column of U
            U[:, 1] = -U[:, 1]
            # change the sign of  s2 to compensate
            S[1] = -S[1]
        
        # Handle case where Vt has determinant -1 (includes reflection)
        det_Vt = torch.det(Vt)
        if det_Vt < 0:
            # Flip the sign of the second row of Vt (second column of V)
            Vt[1, :] = -Vt[1, :]
            # change the sign of  s2 to compensate
            S[1] = -S[1]
        
        # Extract rotation angles from corrected U and V^T
        # theta1 from U matrix (left rotation)
        theta1 = torch.atan2(U[1, 0], U[0, 0])
        
        # theta2 from V^T matrix (right rotation) 
        theta2 = torch.atan2(Vt[1, 0], Vt[0, 0])
        
        # Singular values (scaling factors) - now can be negative
        s1, s2 = S[0], S[1]  # hyperbolic tangent not applied here
        
        return theta1, theta2, s1, s2
    
    def apply_single_transform(self, cs, index):
        """
        Apply a single transformation to a set of coordinates.
        Args:
            cs (torch.Tensor): Coordinates to transform [batch_size, 2]
            index (int): Index of the transformation to apply
        Returns:
            torch.Tensor: Transformed coordinates [batch_size, 2]
        """
        batch_size = cs.shape[0]
        
        # Get transformation matrix and bias for the specified index
        all_w, _ = self.make_matrices_from_svdformat()
        w = all_w[index]  # Shape [2, 2]
        b = self.ifs_b.weight[index]  # Shape [2]
        
        # Expand w to match batch size: [batch_size, 2, 2]
        w_expanded = w.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformation: cs should be [batch_size, 1, 2] for bmm
        cs_reshaped = cs.unsqueeze(1)  # [batch_size, 1, 2]
        cs_transformed = torch.bmm(cs_reshaped, w_expanded).squeeze(1)  # [batch_size, 2]
        
        # Add bias (broadcasted automatically)
        cs_transformed = cs_transformed + b
        
        return cs_transformed
  
    def compute_inverse_matrices_and_biases(self, eps=1e-8):
        """
        Compute inverse transformation matrices and biases while preserving gradients.
        
        For each transformation T(x) = Wx + b where W = U*S*V^T, the inverse is:
        T_inv(x) = W^(-1)(x - b) = V*S^(-1)*U^T*x - W^(-1)*b
        
        This method computes W_inv and b_inv for all transformations while maintaining
        the computational graph for backpropagation.
        
        Args:
            eps (float): Small constant for numerical stability when inverting singular values
        
        Returns:
            tuple: (all_w_inv, all_b_inv)
                - all_w_inv: List of inverse matrices [2, 2] for each transformation
                - all_b_inv: List of inverse biases [2] for each transformation
        """
        all_w_inv = []
        all_b_inv = []
        
        for i in range(self.num_transforms):
            # Get SVD parameters from the embedding (gradients preserved)
            theta_1, theta_2, s_1, s_2 = self.ifs_w.weight[i]
            b = self.ifs_b.weight[i]  # (2,)
            
            # Apply activation to get actual scaling values
            if self.use_htan:
                s1 = torch.tanh(s_1)
                s2 = torch.tanh(s_2)
            else:
                s1 = s_1
                s2 = s_2
            
            # Compute inverse scaling factors with numerical stability
            s1_inv = 1.0 / (s1 + eps * torch.sign(s1))
            s2_inv = 1.0 / (s2 + eps * torch.sign(s2))
            
            # For inverse transformation W^(-1) = V * S^(-1) * U^T:
            # Original: W = U * S * V^T where U = R(theta1), V^T = R(theta2)
            # Inverse: W^(-1) = V * S^(-1) * U^T = R(-theta2) * S^(-1) * R(-theta1)
            r_inv1 = self.make_rotation_matrix(-theta_2)  # V = R(-theta2)
            r_inv2 = self.make_rotation_matrix(-theta_1)  # U^T = R(-theta1)
            S_inv = self.make_diagonal_matrix(s1_inv, s2_inv)
            
            # Compute W_inv = V * S^(-1) * U^T
            W_inv = torch.matmul(torch.matmul(r_inv1, S_inv), r_inv2)  # [2, 2]
            
            # Compute inverse bias: b_inv = -W^(-1) * b
            b_inv = -torch.matmul(W_inv, b)  # [2]
            
            all_w_inv.append(W_inv)
            all_b_inv.append(b_inv)
        
        return all_w_inv, all_b_inv
    
    def apply_inverse_transform(self, cs, indices, eps=1e-8):
        """
        Apply the inverse of transformations to a set of coordinates.
        Each point can use a different transformation.
        Gradients are preserved for backpropagation to the original IFS parameters.
        
        Args:
            cs (torch.Tensor): Coordinates to transform [batch_size, 2]
            indices (torch.Tensor): Indices of transformations to apply [batch_size]
            eps (float): Small constant for numerical stability
    
        Returns:
            torch.Tensor: Inverse-transformed coordinates [batch_size, 2]
        """
        batch_size = cs.shape[0]
        
        # Get all inverse transformation matrices and biases
        all_w_inv = []
        all_b_inv = []
        
        for i in range(self.num_transforms):
            # Get SVD parameters from the embedding (gradients preserved)
            theta_1, theta_2, s_1, s_2 = self.ifs_w.weight[i]
            b = self.ifs_b.weight[i]  # (2,)
            
            # Apply activation to get actual scaling values
            if self.use_htan:
                s1 = torch.tanh(s_1)
                s2 = torch.tanh(s_2)
            else:
                s1 = s_1
                s2 = s_2
            
            # Compute inverse scaling factors
            s1_inv = 1.0 / (s1 + eps * torch.sign(s1))
            s2_inv = 1.0 / (s2 + eps * torch.sign(s2))
            
            # Compute inverse matrix W^(-1) = V * S^(-1) * U^T
            r_inv1 = self.make_rotation_matrix(-theta_2)  # V = R(-theta2)
            r_inv2 = self.make_rotation_matrix(-theta_1)  # U^T = R(-theta1)
            S_inv = self.make_diagonal_matrix(s1_inv, s2_inv)
            W_inv = torch.matmul(torch.matmul(r_inv1, S_inv), r_inv2)  # [2, 2]
            
            # Compute inverse bias
            b_inv = -torch.matmul(W_inv, b)  # [2]
            
            all_w_inv.append(W_inv.unsqueeze(0))  # [1, 2, 2]
            all_b_inv.append(b_inv.unsqueeze(0))  # [1, 2]
        
        # Stack to get [num_transforms, 2, 2] and [num_transforms, 2]
        all_w_inv = torch.cat(all_w_inv, dim=0)  # [num_transforms, 2, 2]
        all_b_inv = torch.cat(all_b_inv, dim=0)  # [num_transforms, 2]
    
        # Select transformations based on indices
        w_inv = all_w_inv[indices.long()]  # [batch_size, 2, 2]
        b_inv = all_b_inv[indices.long()]  # [batch_size, 2]
    
        # Apply inverse transformation
        # T_inv(x) = W_inv * x + b_inv
        cs_reshaped = cs.unsqueeze(1)  # [batch_size, 1, 2]
        cs_transformed = torch.bmm(cs_reshaped, w_inv).squeeze(1)  # [batch_size, 2]
        cs_transformed = cs_transformed + b_inv  # [batch_size, 2]
        
        return cs_transformed


    def create_inverse_ifs(self, eps=1e-8):
        """
        Create a new AffineIFS object with inverse transformations.
        
        For each transformation T(x) = Wx + b where W = U*S*V^T, the inverse is:
        T_inv(x) = W^(-1)(x - b) = V*S^(-1)*U^T*(x - b)
        
        Args:
            eps (float): Small constant for numerical stability when inverting singular values
        
        Returns:
            AffineIFS: New IFS object with inverse transformations
        """
        # Create new IFS with same number of transforms
        inverse_ifs = AffineIFS(
            num_transforms=self.num_transforms, 
            use_htan=False, # generally inverse may not be contractive
            contractive_init=False,  # We'll set parameters manually
            name=f"Inverse of {self.name}"
        )
        
        with torch.no_grad():
            for i in range(self.num_transforms):
                # Get SVD parameters directly from the embedding
                theta_1, theta_2, s_1, s_2 = self.ifs_w.weight[i]
                b = self.ifs_b.weight[i]  # (2,)
                
                # Apply activation to get actual scaling values

                if self.use_htan:
                    s1 = torch.tanh(s_1)
                    s2 = torch.tanh(s_2)
                # inverse may not be contractive, so no activation used
                s1_inv_param = 1.0 / (s1 + eps * torch.sign(s1))
                s2_inv_param = 1.0 / (s2 + eps * torch.sign(s2))

                # For inverse transformation W^(-1) = V * S^(-1) * U^T:
                # - U^T corresponds to rotation by -theta_1
                # - V corresponds to rotation by -theta_2  
                # So the inverse parameters are simply the negated angles
                theta1_inv = -theta_2  # Note: swapped and negated
                theta2_inv = -theta_1  # Note: swapped and negated
                
                # Set parameters directly in inverse IFS (using parameter space values)
                inverse_ifs.ifs_w.weight.data[i] = torch.stack([theta1_inv, theta2_inv, s1_inv_param, s2_inv_param])
                
                
                # Compute inverse bias using the mathematical inverse matrix
                # W^(-1) = V * S^(-1) * U^T (no transpose needed for matrix-vector multiplication)
                r_inv1 = self.make_rotation_matrix(theta1_inv.unsqueeze(0))  # R(-theta2) = V
                r_inv2 = self.make_rotation_matrix(theta2_inv.unsqueeze(0))  # R(-theta1) = U^T
                S_inv = self.make_diagonal_matrix(s1_inv_param, s2_inv_param)
                W_inv = torch.matmul(torch.matmul(r_inv1, S_inv), r_inv2)  # V * S^(-1) * U^T
                b_inv = -torch.matmul(W_inv, b)  # (2,2) @ (2,) = (2,)
                
                inverse_ifs.ifs_b.weight.data[i] = b_inv
            
            # Copy probabilities (inverse transformations use same probabilities)
            inverse_ifs.set_probs(self.probs.detach())
        
        return inverse_ifs

    def create_perturbed_ifs(self, perturbation_factor=0.1, name=None):
        """
        Create a new IFS with perturbed parameters for initialization testing.
        
        Each parameter (rotation angles, scaling factors, translations) is perturbed
        by adding random noise proportional to the parameter value.
        
        Args:
            perturbation_factor (float): Fractional perturbation to apply (e.g., 0.1 for ±10%)
            name (str): Optional name for the perturbed IFS
        
        Returns:
            AffineIFS: New IFS object with perturbed parameters
        """
        # Create new IFS with same configuration
        if name is None:
            name = f"{self.name} (perturbed {perturbation_factor:.0%})"
        
        perturbed_ifs = AffineIFS(
            num_transforms=self.num_transforms,
            use_htan=self.use_htan,
            contractive_init=False,  # We'll set parameters manually
            name=name
        )
        
        # Convert percentage to fraction

        
        with torch.no_grad():
            for i in range(self.num_transforms):
                # Get original parameters
                theta_1, theta_2, s_1, s_2 = self.ifs_w.weight[i]
                tx, ty = self.ifs_b.weight[i]
                
                # Perturb rotation angles: add noise as a fraction of 2π
                # For example, 10% means ± 0.1 * 2π uniform noise
                angle_noise = perturbation_factor * 2 * np.pi
                theta_1_perturbed = theta_1 + angle_noise * (2 * torch.rand(1).item() - 1)
                theta_2_perturbed = theta_2 + angle_noise * (2 * torch.rand(1).item() - 1)
                
                # Perturb scaling factors
                # Note: these are in parameter space (before activation if use_htan=True)
                s_1_perturbed = s_1 * (1 + perturbation_factor * (2 * torch.rand(1).item() - 1))
                s_2_perturbed = s_2 * (1 + perturbation_factor * (2 * torch.rand(1).item() - 1))
                
                # If using tanh activation, ensure perturbed values stay in reasonable range
                if self.use_htan:
                    # Clamp to avoid extreme values that would saturate tanh
                    s_1_perturbed = torch.clamp(s_1_perturbed, -3.0, 3.0)
                    s_2_perturbed = torch.clamp(s_2_perturbed, -3.0, 3.0)
                
                # Perturb translation parameters
                tx_perturbed = tx * (1 + perturbation_factor * (2 * torch.rand(1).item() - 1))
                ty_perturbed = ty * (1 + perturbation_factor * (2 * torch.rand(1).item() - 1))
                
                # Set perturbed parameters
                perturbed_ifs.ifs_w.weight.data[i] = torch.tensor([
                    theta_1_perturbed, theta_2_perturbed, s_1_perturbed, s_2_perturbed
                ])
                perturbed_ifs.ifs_b.weight.data[i] = torch.tensor([tx_perturbed, ty_perturbed])
            
            # TODO:
            # Perturb probabilities slightly
            # Add small noise to logits, then normalize
            # perturbed_logits = self.prob_logits + perturbation_factor * torch.randn_like(self.prob_logits)
            # perturbed_ifs.prob_logits.data = perturbed_logits
        
        return perturbed_ifs
    
    def clone_ifs(self, name=None):
        """
        Create an exact copy of this IFS with all parameters.
        
        Args:
            name (str): Optional name for the cloned IFS
        
        Returns:
            AffineIFS: New IFS object with identical parameters
        """
        if name is None:
            name = f"{self.name} (clone)"
        
        cloned_ifs = AffineIFS(
            num_transforms=self.num_transforms,
            use_htan=self.use_htan,
            contractive_init=False,
            name=name
        )
        
        with torch.no_grad():
            # Copy all parameters exactly
            cloned_ifs.ifs_w.weight.data = self.ifs_w.weight.data.clone()
            cloned_ifs.ifs_b.weight.data = self.ifs_b.weight.data.clone()
            cloned_ifs.prob_logits.data = self.prob_logits.data.clone()
        
        return cloned_ifs    
    
    def plot_transformations(self, points=None, ax=None):
        """
        Visualize the effect of each transformation on a set of points.

        Args:
            points (torch.Tensor): Points to transform [num_points, 2]
            ax (matplotlib.axes.Axes): Optional matplotlib Axes to plot on
            
        Returns:
            matplotlib.axes.Axes: Axes with the plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if points is None:
            # use unit square points
            points = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)

        # Draw lines between original points to form shape
        # Connect points in order to form closed shape (like a square)
        for i in range(points.size(0)):
            next_i = (i + 1) % points.size(0)            
            # ax.plot([points[i, 0], points[next_i, 0]], [points[i, 1], points[next_i, 1]], 
            #        c='black',linewidth=2, alpha=0.8, label='Original' if i == 0 else '')
            ax.arrow(points[i, 0], points[i, 1], #type: ignore
                     points[next_i, 0] - points[i, 0], #type: ignore
                     points[next_i, 1] - points[i, 1], #type: ignore
                     head_width=0.01, head_length=0.02, fc='black', ec='black', 
                     head_starts_at_zero=False, length_includes_head=True,
                     label='Original' if i == 0 else '')

        # Plot original points
        ax.scatter(points[0, 0], points[0, 1], marker="s", s=50, c='black', zorder=5)        
        #ax.scatter(points[1:, 0], points[1:, 1], s=50, c='black', zorder=5)

        # Define colors for each transformation
        colors = cm.Set1(np.linspace(0, 1, self.num_transforms)) #type: ignore

        # Apply each transformation and plot results
        for transform_idx in range(self.num_transforms):
            # Apply transformation to all points
            points = points.to(next(self.parameters()).device) # make sure points are on the same device as the model
            transformed_points = self.apply_single_transform(points, transform_idx)

            # Convert to numpy for plotting
            transformed_np = transformed_points.detach().cpu().numpy()
            
            # Draw lines between transformed points to form shape
            for i in range(transformed_np.shape[0]):
                next_i = (i + 1) % transformed_np.shape[0]
                # ax.plot([transformed_np[i, 0], transformed_np[next_i, 0]], 
                #        [transformed_np[i, 1], transformed_np[next_i, 1]], 
                #        c=colors[transform_idx], linewidth=1.5, alpha=0.7,
                #        label=f'Transform {transform_idx + 1}' if i == 0 else '')
                ax.arrow(transformed_np[i, 0], transformed_np[i, 1],
                            transformed_np[next_i, 0] - transformed_np[i, 0],
                            transformed_np[next_i, 1] - transformed_np[i, 1],
                            head_width=0.01, head_length=0.01, fc=colors[transform_idx], ec=colors[transform_idx], 
                            head_starts_at_zero=False, length_includes_head=True,
                            label=f'Transform {transform_idx + 1}' if i == 0 else '')   
            # Plot transformed points
            ax.scatter(transformed_np[0, 0], transformed_np[0, 1], marker='s',
                      s=30, c=[colors[transform_idx]], alpha=0.8, zorder=5)

            # ax.scatter(transformed_np[1:, 0], transformed_np[1:, 1], 
            #           s=30, c=[colors[transform_idx]], alpha=0.8, zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'IFS Transformations: {self.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return ax

    def __str__(self):
        """String representation of the IFS."""
        return f"AffineIFS(name='{self.name}', num_transforms={self.num_transforms}, use_htan={self.use_htan})"
    
    def __repr__(self):
        """Detailed representation of the IFS."""
        return self.__str__()
    
    def info(self):
        """Print detailed information about this IFS."""
        print(f"IFS Name: {self.name}")
        print(f"Number of transformations: {self.num_transforms}")
        print(f"Use hyperbolic tangent activation: {self.use_htan}")
        print(f"Probabilities: {self.probs.detach().cpu().numpy()}")
        print(f"Internal prob_logits: {self.prob_logits.detach().cpu().numpy()}")
        print("\nTransformation parameters:")
        
        for i in range(self.num_transforms):
            theta1, theta2, s1, s2 = self.ifs_w.weight[i]
            tx, ty = self.ifs_b.weight[i]
            
            if self.use_htan:
                actual_s1 = torch.tanh(s1).item()
                actual_s2 = torch.tanh(s2).item()
            else:
                actual_s1 = s1.item()
                actual_s2 = s2.item()
            
            print(f"  Transform {i+1}: rotation=({theta1:.3f}, {theta2:.3f}), "
                  f"scale=({actual_s1:.3f}, {actual_s2:.3f}), "
                  f"translate=({tx:.3f}, {ty:.3f})")
    
    def set_name(self, name):
        """Set or update the name of this IFS."""
        self.name = name


#####
## utility function to generate points from an IFS
#####
def generate_ifs_trajectories(ifs, num_samples=1000, num_steps=5000, initial_cs=None):
    """
    Generate points by iterating the IFS.
    
    Args:
        ifs (AffineIFS): Configured IFS
        num_samples (int): Number of sample trajectories
        num_steps (int): Number of iteration steps per trajectory
        initial_cs (torch.Tensor): Optional initial points [num_samples, 2]
        
    Returns:
        torch.Tensor: Generated points [num_samples, num_steps, 2]
    """
    # Get the device of the IFS model
    device = next(ifs.parameters()).device
    
    # Generate random initial points within unit square for each sample
    if initial_cs is None:
        initial_cs = torch.rand(num_samples, 2, device=device)  # Shape: (num_samples, 2), values in [0, 1]
    else:
        # Ensure initial_cs is on the same device as the model
        initial_cs = initial_cs.to(device)
    
    # Generate fractal points
    with torch.no_grad():
        coords, all_sgv, seqs = ifs(initial_cs, num_steps)

    return coords, seqs

###
