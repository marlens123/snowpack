import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from PIL import Image
import cv2

from tqdm import tqdm
import argparse

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from train_epochs import MulticlassSAMWrapper

import torchvision.transforms.functional as TF

from scipy.ndimage import label


parser = argparse.ArgumentParser()


parser.add_argument('--image_path', type=str, default='snowpack/data/multiclass_10_2/test/images/9.tiff')
parser.add_argument('--saved_model_location', type=str, default='snowpack/model/model_checkpoints/snowpack_sam2_revert_boundary_resize_simple_100.torch')
parser.add_argument('--save_image_location', type=str, default='final_mask.pt')

parser.add_argument('--n_classes', default=21, type=int)

parser.add_argument('--patch_size', default=400, type=int)
parser.add_argument('--min_overlap', default=200, type=int)
parser.add_argument('--edge_buffer', default=2, type=int) # if too high, may create lines. but same if too low

parser.add_argument('--gaussian_smoothing', default=False, action='store_true')
parser.add_argument('--total_variation_smoothing', default=True, action='store_true') # denoises, preservers boundaries more than gaussian


def main():
    args = parser.parse_args()

    image_path = args.image_path
    saved_model_location = args.saved_model_location
    save_image_location = args.save_image_location
    n_classes = args.n_classes
    patch_size = args.patch_size
    min_overlap = args.min_overlap
    edge_buffer = args.edge_buffer
    gaussian_smoothing = args.gaussian_smoothing
    total_variation_smoothing = args.total_variation_smoothing  ####### denoising
    temperature = 0.001 ## 1 is softer, lower means more sharpening

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(3),
            v2.Resize((1024, 1024)),
            v2.Normalize(mean=mean, std=std),
        ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam2_checkpoint = "snowpack/model/model_checkpoints/sam2.1_hiera_small.pt"
    #with resources.open_text('snowpack', ) as file:
    model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
    sam2_model = build_sam2(config_file=model_cfg, ckpt_path=sam2_checkpoint, device=device)
    sam2_model = MulticlassSAMWrapper(sam2_model, n_classes).to(device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load(saved_model_location))
    sparse_embeddings = torch.zeros((1, 1, 256), device=predictor.model.device)
    dense_prompt_embeddings = torch.zeros((1, 256, 256), device=predictor.model.device)
    dense_embeddings = F.interpolate(
        dense_prompt_embeddings.unsqueeze(0),  # Add batch dimension
        size=(64, 64),  # Match expected spatial resolution of src
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension if necessary

    image = Image.open(image_path)
    patches, patch_coords = chunk_image_no_padding(image, patch_size, min_overlap)
    patches = [test_transform(i).permute(1, 2, 0) for i in patches]


    width, height = image.size
    patch_height, patch_width = args.patch_size, args.patch_size

    combined_probabilities = torch.zeros((args.n_classes, height, width), dtype=patches[0].dtype)
    overlap_count = torch.zeros((height, width), dtype=torch.float32)
    
    print('Generating mask')
    with torch.no_grad():
        for i, (patch, (x, y)) in tqdm(enumerate(zip(patches, patch_coords)), total=len(patches)):
            predictor.set_image(np.array(patch))
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks = predictor.model(sparse_embeddings, dense_embeddings, high_res_features,
                                            predictor._features["image_embed"][-1].unsqueeze(0))
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]).float()
            prd_masks = F.interpolate(prd_masks, size=args.patch_size, mode='bilinear', align_corners=False)
            prd_masks = prd_masks.squeeze(0).to(torch.float16)    

            if total_variation_smoothing:
                prd_masks = total_variation_smooth(prd_masks, tv_weight=0.3)

            prd_masks = torch.softmax(prd_masks / temperature, dim=0)
  
            prd_masks = apply_weight_mask(prd_masks, args.patch_size, edge_buffer=edge_buffer)

            # prd_masks = amplify_boundaries(prd_masks)
            # prd_masks = clean_broken_lines(prd_masks)

            # Determine valid region
            right = min(x + patch_width, width)
            bottom = min(y + patch_height, height)

            # Update the combined map incrementally
            combined_probabilities[:, y:bottom, x:right] += prd_masks[:, :bottom-y, :right-x]
            overlap_count[y:bottom, x:right] += 1

            # Discard the patch tensor to free memory
            del prd_masks

    overlap_count = overlap_count.clamp(min=1)
    combined_probabilities /= overlap_count.unsqueeze(0)

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Different smoothings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if gaussian_smoothing:
       print('Adding gaussian smoothing')
       combined_probabilities = smooth_probabilities(combined_probabilities, kernel_size=3)

    # if total_variation_smoothing:
    #     print('Adding total variation smoothing (denoises)')
    #     combined_probabilities = total_variation_smooth(combined_probabilities, tv_weight=0.7)

    # combined_probabilities = smooth_line_interiors(combined_probabilities, kernel_size=11)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Different smoothings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    final_mask = torch.argmax(combined_probabilities, dim=0)

    if total_variation_smoothing:
        print('Adding total variation smoothing (denoises) again')
        final_mask = total_variation_smooth_final(final_mask, kernel_size=5)


    torch.save(final_mask, save_image_location)
    print(f'Saved mask at {save_image_location}')




def chunk_image_no_padding(image, patch_size, min_overlap):
    """
    Splits an image into consistent-sized overlapping patches without padding.

    Args:
        image (PIL.Image or np.ndarray): The input image.
        patch_size (int): Size of each patch.
        min_overlap (int): Minimum desired overlap in pixels.

    Returns:
        list of np.ndarray: List of image patches as NumPy arrays.
        list of tuple: List of (x, y) coordinates for the top-left corner of each patch.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)  # Convert PIL image to NumPy array

    height, width = image.shape[:2]

    # Calculate the stride dynamically to ensure coverage without padding
    stride_x = max(patch_size - min_overlap, (width - patch_size) // (width // patch_size))
    stride_y = max(patch_size - min_overlap, (height - patch_size) // (height // patch_size))

    patches = []
    patch_coords = []

    for y in range(0, height - patch_size + 1, stride_y):
        for x in range(0, width - patch_size + 1, stride_x):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            patch_coords.append((x, y))

    # Handle edge cases to ensure the bottom-right corner is included
    if height % patch_size != 0:
        for x in range(0, width - patch_size + 1, stride_x):
            patch = image[height - patch_size:height, x:x + patch_size]
            patches.append(patch)
            patch_coords.append((x, height - patch_size))
    if width % patch_size != 0:
        for y in range(0, height - patch_size + 1, stride_y):
            patch = image[y:y + patch_size, width - patch_size:width]
            patches.append(patch)
            patch_coords.append((width - patch_size, y))

    # Add the bottom-right corner patch if needed
    if width % patch_size != 0 and height % patch_size != 0:
        patch = image[height - patch_size:height, width - patch_size:width]
        patches.append(patch)
        patch_coords.append((width - patch_size, height - patch_size))

    return patches, patch_coords





# def apply_weight_mask(probability_mask, patch_size, edge_buffer):
#     """
#     Applies a weight mask to suppress edge effects in predictions.

#     Args:
#         probability_mask (torch.Tensor): Probability mask of shape (C, H, W).
#         patch_size (int): Size of the patch.
#         edge_buffer (int): Number of pixels to ignore along the edges.

#     Returns:
#         torch.Tensor: Weighted probability mask.
#     """
#     C, H, W = probability_mask.shape
#     weight_mask = torch.ones((H, W), dtype=probability_mask.dtype, device=probability_mask.device)

#     # Create a buffer region near the edges
#     weight_mask[:edge_buffer, :] = 0  # Top edge
#     weight_mask[-edge_buffer:, :] = 0  # Bottom edge
#     weight_mask[:, :edge_buffer] = 0  # Left edge
#     weight_mask[:, -edge_buffer:] = 0  # Right edge

#     # Apply weight mask to the probability mask
#     return probability_mask * weight_mask.unsqueeze(0)  

# def apply_weight_mask(probability_mask, patch_size, edge_buffer):
#     """
#     Applies a linear weight map to reduce sharp transitions in overlapping patches.

#     Args:
#         probability_mask (torch.Tensor): Probability mask of shape (C, H, W).
#         patch_size (int): Size of the patch.
#         edge_buffer (int): Number of pixels to reduce weight along the edges.

#     Returns:
#         torch.Tensor: Weighted probability mask.
#     """
#     C, H, W = probability_mask.shape

#     # Create 1D linear weights
#     linear_weights = torch.linspace(0, 1, patch_size - 2 * edge_buffer, device=probability_mask.device)
#     ramp = torch.cat([torch.zeros(edge_buffer, device=probability_mask.device), linear_weights, 
#                       torch.ones(patch_size - len(linear_weights) - edge_buffer, device=probability_mask.device)])
#     weight_map = torch.outer(ramp, ramp)  # Create a 2D weight map
#     weight_map = weight_map / weight_map.max()  # Normalize weights

#     # Create a buffer region near the edges
#     weight_map[:edge_buffer, :] = 0  # Top edge
#     weight_map[-edge_buffer:, :] = 0  # Bottom edge
#     weight_map[:, :edge_buffer] = 0  # Left edge
#     weight_map[:, -edge_buffer:] = 0  # Right edge

#     # Apply weight map to probability mask
#     return probability_mask * weight_map.unsqueeze(0)  # Add channel dimension
# import torch

def apply_weight_mask(probability_mask, patch_size, boost_factor=5, radius_factor=0.7, edge_buffer=3):
    """
    Applies a center boost to the weight map while keeping edge weights stable.

    Args:
        probability_mask (torch.Tensor): Probability mask of shape (C, H, W).
        patch_size (int): Size of the patch.
        boost_factor (float): Multiplier for the center weight boost.
        radius_factor (float): Determines the size of the boosted region (0.5 = half the patch).

    Returns:
        torch.Tensor: Weighted probability mask.
    """
    C, H, W = probability_mask.shape

    # Base weight map with uniform weights
    weight_map = torch.ones((H, W), dtype=probability_mask.dtype, device=probability_mask.device)

    # Create a 2D grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=probability_mask.device),
        torch.linspace(-1, 1, W, device=probability_mask.device),
        indexing='ij'
    )
    # Calculate radial distance from the center
    radius = torch.sqrt(x**2 + y**2)

    # Create a center boost (Gaussian-like bump)
    center_boost = torch.exp(-radius**2 / (2 * radius_factor**2)) * boost_factor

    # Combine the base weight map with the center boost
    final_weight_map = weight_map + center_boost
    final_weight_map = final_weight_map / final_weight_map.max()  # Normalize weights

  # Create a buffer region near the edges
    final_weight_map[:edge_buffer, :] = 0  # Top edge
    final_weight_map[-edge_buffer:, :] = 0  # Bottom edge
    final_weight_map[:, :edge_buffer] = 0  # Left edge
    final_weight_map[:, -edge_buffer:] = 0  # Right edge

    # Apply the weight map to the probability mask
    return probability_mask * final_weight_map.unsqueeze(0)  # Add channel dimension


def total_variation_smooth(probability_map, tv_weight=0.1):
    """
    Apply Total Variation (TV) smoothing to the probability map.

    Args:
        probability_map (torch.Tensor): Probability map of shape (C, H, W).
        tv_weight (float): Weight for TV regularization.

    Returns:
        torch.Tensor: Smoothed probability map.
    """
    smoothed_map = probability_map.clone()

    # Compute TV loss gradients
    for c in range(probability_map.size(0)):  # Loop through classes
        dx = torch.roll(probability_map[c], shifts=-1, dims=1) - probability_map[c]
        dy = torch.roll(probability_map[c], shifts=-1, dims=0) - probability_map[c]
        tv_loss = dx**2 + dy**2
        smoothed_map[c] -= tv_weight * tv_loss

    return smoothed_map


def total_variation_smooth_final(segmentation_mask, kernel_size=3):
    """
    Applies Total Variation-like smoothing to a segmented mask after argmax.

    Args:
        segmentation_mask (torch.Tensor): Segmented mask of shape (H, W), with integer class labels.
        kernel_size (int): Size of the smoothing kernel.

    Returns:
        torch.Tensor: Smoothed segmentation mask.
    """
    height, width = segmentation_mask.shape

    # Convert to one-hot encoding
    num_classes = int(segmentation_mask.max()) + 1
    one_hot = F.one_hot(segmentation_mask.long(), num_classes=num_classes).permute(2, 0, 1).float()  # (C, H, W)

    # Apply a smoothing kernel (Gaussian or Mean Filter)
    kernel = torch.ones((num_classes, 1, kernel_size, kernel_size), device=segmentation_mask.device) / (kernel_size ** 2)

    smoothed = F.conv2d(one_hot.unsqueeze(0), kernel, padding=kernel_size // 2, groups=num_classes).squeeze(0)  # (C, H, W)

    # Assign each pixel to the class with the highest smoothed probability
    smoothed_mask = torch.argmax(smoothed, dim=0)

    return smoothed_mask



def smooth_across_layers(probability_map, weight=0.8):
    """
    Smooth probabilities across class layers by blending each layer with its neighbors.

    Args:
        probability_map (torch.Tensor): Probability map of shape (C, H, W), where C is the number of classes.
        weight (float): Weight for the original layer (default: 0.8). Remaining weight is distributed to neighbors.

    Returns:
        torch.Tensor: Smoothed probability map.
    """
    num_classes, height, width = probability_map.shape
    smoothed_map = torch.zeros_like(probability_map)

    for c in range(num_classes):
        # Original map retains `weight`
        smoothed_layer = probability_map[c] * weight
        
        # Add contribution from neighbors (previous and next classes)
        if c > 0:
            smoothed_layer += probability_map[c - 1] * (1 - weight) / 2
        if c < num_classes - 1:
            smoothed_layer += probability_map[c + 1] * (1 - weight) / 2

        smoothed_map[c] = smoothed_layer

    return smoothed_map





def smooth_probabilities(probability_map, kernel_size=5):
    """
    Applies a Gaussian blur to smooth the probability map.

    Args:
        probability_map (torch.Tensor): Probability map of shape (C, H, W).
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        torch.Tensor: Smoothed probability map.
    """
    smoothed_map = TF.gaussian_blur(probability_map.unsqueeze(0), kernel_size=kernel_size).squeeze(0)
    return smoothed_map





def amplify_boundaries(probability_map, amplification_factor=2):
    """
    Amplifies line boundaries in a probability map using edge detection.

    Args:
        probability_map (torch.Tensor): Probability map of shape (C, H, W).
        amplification_factor (float): Factor by which to amplify the boundaries.

    Returns:
        torch.Tensor: Adjusted probability map.
    """
    num_classes, height, width = probability_map.shape
    adjusted_map = probability_map.clone()

    for c in range(num_classes):
        # Convert to NumPy for edge detection
        mask_np = probability_map[c].cpu().numpy()
        
        # Detect edges using Sobel or Canny
        sobel_x = cv2.Sobel(mask_np, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask_np, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = (edges > edges.mean()).astype(np.float32)  # Threshold for strong edges

        # Amplify detected boundaries
        adjusted_map[c] += torch.tensor(edges, device=probability_map.device) * amplification_factor

    return adjusted_map



def clean_broken_lines(probability_map, kernel_size=3):
    """
    Cleans broken lines using morphological closing.

    Args:
        probability_map (torch.Tensor): Probability map of shape (C, H, W).
        kernel_size (int): Size of the closing kernel.

    Returns:
        torch.Tensor: Cleaned probability map.
    """
    num_classes, height, width = probability_map.shape
    cleaned_map = probability_map.clone()

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for c in range(num_classes):
        mask_np = probability_map[c].cpu().numpy()
        
        # Apply morphological closing
        closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        
        cleaned_map[c] = torch.tensor(closed, device=probability_map.device)

    return cleaned_map
def smooth_line_interiors(probability_map, kernel_size=5):
    """
    Smooths the interiors of lines in a probability map.

    Args:
        probability_map (torch.Tensor): Probability map of shape (C, H, W).
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        torch.Tensor: Smoothed probability map.
    """
    num_classes, height, width = probability_map.shape
    smoothed_map = probability_map.clone()

    for c in range(num_classes):
        mask_np = probability_map[c].cpu().numpy()
        
        # Detect edges and create a mask
        sobel_x = cv2.Sobel(mask_np, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask_np, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = (edges > edges.mean()).astype(np.uint8)

        # Dilate the edge mask
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # Smooth the interior of the dilated edges
        smoothed_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
        smoothed_map[c] = torch.tensor(mask_np * dilated_edges + smoothed_np * (1 - dilated_edges), 
                                       device=probability_map.device)

    return smoothed_map



if __name__ == "__main__":
    main()


