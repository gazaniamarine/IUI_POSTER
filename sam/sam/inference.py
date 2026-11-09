"""
Video Inference Script for SAM2 Model
Processes video frame-by-frame, segments bigrock and bedrock, extracts bounding boxes,
and saves segmented frames and output video.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import model loading and preprocessing functions
from evaluate_sam2_simple_fast import (
    load_best_model,
    preprocess_image_for_model,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    DEVICE,
)

# Class indices
BEDROCK_CLASS = 1  # bedrock
BIGROCK_CLASS = 3  # big_rock

# Colors for visualization (BGR for OpenCV)
BEDROCK_COLOR = (128, 128, 128)  # gray
BIGROCK_COLOR = (20, 60, 220)    # red (BGR)


def extract_bounding_boxes(mask: np.ndarray, class_id: int) -> List[Dict]:
    """
    Extract bounding boxes for a specific class from a segmentation mask.
    
    Args:
        mask: (H, W) segmentation mask with class indices
        class_id: Class ID to extract bounding boxes for
        
    Returns:
        List of dicts with keys: 'x', 'y', 'width', 'height', 'area', 'class_id'
    """
    # Create binary mask for the specific class
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': int(cv2.contourArea(contour)),
                'class_id': int(class_id),
                'class_name': CLASS_NAMES[class_id]
            })
    
    return boxes


def create_filtered_mask(pred_mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a mask containing only bigrock and bedrock classes.
    
    Args:
        pred_mask: (H, W) predicted mask
        original_shape: (height, width) of original frame
        
    Returns:
        (H, W) filtered mask with only bigrock (3) and bedrock (1), others set to 0
    """
    filtered = np.zeros_like(pred_mask)
    filtered[pred_mask == BEDROCK_CLASS] = BEDROCK_CLASS
    filtered[pred_mask == BIGROCK_CLASS] = BIGROCK_CLASS
    
    # Resize to original frame size if needed
    if filtered.shape != original_shape[:2]:
        filtered = cv2.resize(filtered, (original_shape[1], original_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    
    return filtered


def draw_bounding_boxes(image: np.ndarray, boxes: List[Dict], class_id: int) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: (H, W, 3) BGR image
        boxes: List of bounding box dicts
        class_id: Class ID to filter boxes
        
    Returns:
        Image with bounding boxes drawn
    """
    color = BEDROCK_COLOR if class_id == BEDROCK_CLASS else BIGROCK_COLOR
    class_name = CLASS_NAMES[class_id]
    
    for box in boxes:
        if box['class_id'] == class_id:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # Add label
            label = f"{class_name} ({box['area']})"
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
    
    return image


def run_inference_on_frame(
    image,
    model: nn.Module = None,
) -> Dict[str, List[Dict]]:
    """
    Run SAM2 segmentation on a single in-memory frame and extract bedrock/
    bigrock bounding boxes, without requiring a video file.

    Mirrors process_video()'s per-frame body (resize -> preprocess -> model
    -> resize prediction back to original resolution -> filter classes ->
    extract boxes) but returns the boxes directly instead of accumulating
    them alongside video I/O.

    Args:
        image: path to an image file (str/Path), or an already-loaded
            (H, W, 3) uint8 BGR np.ndarray (OpenCV convention, matching
            every other entry point in this module - e.g. a raw RGB array
            from elsewhere must be converted to BGR before being passed
            in here).
        model: preloaded model, as returned by load_best_model(). Loaded
            fresh via load_best_model() if not given.

    Returns:
        {'bedrock': [...], 'bigrock': [...]}, each a list of box dicts
        with the same keys process_video() emits per frame ('x', 'y',
        'width', 'height', 'area', 'class_id', 'class_name').
    """
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Failed to read image: {image_path}")
    else:
        frame = image

    if model is None:
        model = load_best_model()
    model.eval()

    height, width = frame.shape[:2]
    original_shape = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    frame_bgr_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

    img_tensor = preprocess_image_for_model(frame_bgr_resized)

    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(DEVICE)
        logits = model(inp)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    pred_resized = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
    filtered_mask = create_filtered_mask(pred_resized, original_shape)

    bedrock_boxes = extract_bounding_boxes(filtered_mask, BEDROCK_CLASS)
    bigrock_boxes = extract_bounding_boxes(filtered_mask, BIGROCK_CLASS)

    return {"bedrock": bedrock_boxes, "bigrock": bigrock_boxes}


def process_video(
    video_path: str,
    output_dir: str,
    save_frames: bool = True,
    save_video: bool = True,
    draw_boxes: bool = True
) -> None:
    """
    Process video frame-by-frame with SAM2 model.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save outputs
        save_frames: Whether to save individual segmented frames
        save_video: Whether to save output video
        draw_boxes: Whether to draw bounding boxes on output
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    if save_frames:
        frames_dir.mkdir(exist_ok=True)
    
    # Output paths
    bbox_file = output_dir / "bounding_boxes.json"
    video_output_path = output_dir / f"{video_path.stem}_segmented.mp4"
    
    print("=" * 80)
    print("Video Inference with SAM2 Model")
    print("=" * 80)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Classes to segment: bedrock (1), bigrock (3)")
    print()
    
    # Load model
    print("Loading SAM2 model.")
    model = load_best_model()
    model.eval()
    print("Model loaded successfully.\n")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()
    
    # Setup video writer
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_output_path), fourcc, fps, (width, height)
        )
    
    # Storage for bounding boxes
    all_bboxes = []
    
    # Process frames
    frame_idx = 0
    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            original_frame = frame.copy()
            original_shape = frame.shape
            
            # Resize frame for model input (keep aspect ratio or resize to IMAGE_SIZE)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            frame_bgr_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            
            # Preprocess for model
            img_tensor = preprocess_image_for_model(frame_bgr_resized)
            
            # Run inference
            inp = img_tensor.unsqueeze(0).to(DEVICE)
            logits = model(inp)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
            
            # Resize prediction back to original frame size
            pred_resized = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Filter mask: keep only bedrock and bigrock
            filtered_mask = create_filtered_mask(pred_resized, original_shape)
            
            # Extract bounding boxes
            bedrock_boxes = extract_bounding_boxes(filtered_mask, BEDROCK_CLASS)
            bigrock_boxes = extract_bounding_boxes(filtered_mask, BIGROCK_CLASS)
            
            # Store bounding boxes
            frame_bboxes = {
                'frame_number': frame_idx,
                'timestamp_sec': frame_idx / fps if fps > 0 else 0.0,
                'bedrock': bedrock_boxes,
                'bigrock': bigrock_boxes
            }
            all_bboxes.append(frame_bboxes)
            
            # Create visualization
            # Create colored mask overlay
            overlay_frame = original_frame.copy()
            
            # Create color mask for filtered classes
            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            color_mask[filtered_mask == BEDROCK_CLASS] = BEDROCK_COLOR
            color_mask[filtered_mask == BIGROCK_CLASS] = BIGROCK_COLOR
            
            # Overlay mask on frame
            mask_binary = (filtered_mask > 0).astype(np.float32)
            mask_binary = np.stack([mask_binary] * 3, axis=-1)
            overlay_frame = (overlay_frame * (1 - 0.5 * mask_binary) + 
                           color_mask * (0.5 * mask_binary)).astype(np.uint8)
            
            # Draw bounding boxes if requested
            if draw_boxes:
                overlay_frame = draw_bounding_boxes(overlay_frame, bedrock_boxes, BEDROCK_CLASS)
                overlay_frame = draw_bounding_boxes(overlay_frame, bigrock_boxes, BIGROCK_CLASS)
            
            # Save frame if requested
            if save_frames:
                frame_filename = frames_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_filename), overlay_frame)
            
            # Write to output video
            if save_video and video_writer is not None:
                video_writer.write(overlay_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
    
    # Save bounding boxes to JSON
    with open(bbox_file, 'w') as f:
        json.dump({
            'video_path': str(video_path),
            'video_properties': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            },
            'frames': all_bboxes
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Processed {frame_idx} frames")
    print(f"Frames saved to: {frames_dir}" if save_frames else "")
    print(f"Output video saved to: {video_output_path}" if save_video else "")
    print(f"Bounding boxes saved to: {bbox_file}")
    
    # Print summary statistics
    total_bedrock = sum(len(f['bedrock']) for f in all_bboxes)
    total_bigrock = sum(len(f['bigrock']) for f in all_bboxes)
    print(f"\nSummary:")
    print(f"  Total bedrock detections: {total_bedrock}")
    print(f"  Total bigrock detections: {total_bigrock}")
    print(f"  Frames with bedrock: {sum(1 for f in all_bboxes if len(f['bedrock']) > 0)}")
    print(f"  Frames with bigrock: {sum(1 for f in all_bboxes if len(f['bigrock']) > 0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process video with SAM2 model for bigrock and bedrock segmentation"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <video_name>_output)"
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Don't save individual frames"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save output video"
    )
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="Don't draw bounding boxes on output"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    video_path = Path(args.video_path)
    if args.output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_output"
    else:
        output_dir = Path(args.output_dir)
    
    # Process video
    process_video(
        video_path=str(video_path),
        output_dir=str(output_dir),
        save_frames=not args.no_frames,
        save_video=not args.no_video,
        draw_boxes=not args.no_boxes
    )


if __name__ == "__main__":
    main()