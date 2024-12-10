import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from PIL import Image
from tqdm import tqdm
import argparse

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from train_epochs import MulticlassSAMWrapper


parser = argparse.ArgumentParser()


parser.add_argument('--image_path', type=str, default='snowpack/data/multiclass_10_2/train/images/1.tiff')
parser.add_argument('--saved_model_location', type=str, default='snowpack/model/model_checkpoints/snowpack_sam2_revert_boundary_resize_simple_100.torch')
parser.add_argument('--save_image_location', type=str, default='final_mask.pt')

parser.add_argument('--n_classes', default=21, type=int)
parser.add_argument('--multiclass', default=True, action='store_true')

parser.add_argument('--patch_size', default=400, type=int)
parser.add_argument('--min_overlap', default=300, type=int)
parser.add_argument('--edge_buffer', default=2, type=int) # if too high, may create lines. but same if too low


def main():
    args = parser.parse_args()

    image_path = args.image_path
    saved_model_location = args.saved_model_location
    save_image_location = args.save_image_location
    multiclass = args.multiclass
    n_classes = args.n_classes
    patch_size = args.patch_size
    min_overlap = args.min_overlap
    edge_buffer = args.edge_buffer
    more_overlap = True # if issue with borders that edge buffer doesn't solve


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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if multiclass:
        sam2_model = MulticlassSAMWrapper(sam2_model, n_classes).to(device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ multiclass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
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

    # if more_overlap:
    #     patches2, patch_coords2 = chunk_image_no_padding(image, patch_size, min_overlap+50)
    #     patches2 = [test_transform(i).permute(1, 2, 0) for i in patches2]

    #     patches.extend(patches2)
    #     patch_coords.extend(patch_coords2)


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
            prd_masks = apply_weight_mask(prd_masks, args.patch_size, edge_buffer)

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
    final_mask = torch.argmax(combined_probabilities, dim=0)

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





def apply_weight_mask(probability_mask, patch_size, edge_buffer):
    """
    Applies a weight mask to suppress edge effects in predictions.

    Args:
        probability_mask (torch.Tensor): Probability mask of shape (C, H, W).
        patch_size (int): Size of the patch.
        edge_buffer (int): Number of pixels to ignore along the edges.

    Returns:
        torch.Tensor: Weighted probability mask.
    """
    C, H, W = probability_mask.shape
    weight_mask = torch.ones((H, W), dtype=probability_mask.dtype, device=probability_mask.device)

    # Create a buffer region near the edges
    weight_mask[:edge_buffer, :] = 0  # Top edge
    weight_mask[-edge_buffer:, :] = 0  # Bottom edge
    weight_mask[:, :edge_buffer] = 0  # Left edge
    weight_mask[:, -edge_buffer:] = 0  # Right edge

    # Apply weight mask to the probability mask
    return probability_mask * weight_mask.unsqueeze(0)  



if __name__ == "__main__":
    main()
