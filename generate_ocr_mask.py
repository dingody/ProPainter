# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    print("RapidOCR not installed. Please install with: pip install rapidocr-onnxruntime")
    RAPIDOCR_AVAILABLE = False


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image with automatic directory creation"""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def read_frame_from_videos(frame_root):
    """Read frames from video file or image folder"""
    import torchvision
    
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI', 'mkv', 'MKV', 'webm', 'WEBM')):
        # Input video path
        video_name = os.path.basename(frame_root)[:-4]
        try:
            vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec')
            frames = list(vframes.numpy())
            frames = [Image.fromarray(f) for f in frames]
            fps = info['video_fps']
        except Exception as e:
            print(f"Error reading video with torchvision: {e}")
            print("Trying with OpenCV...")
            frames, fps = read_video_with_opencv(frame_root)
            video_name = os.path.basename(frame_root)[:-4]
    else:
        # Input image folder
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            if fr.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                frame = cv2.imread(os.path.join(frame_root, fr))
                if frame is not None:
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(frame)
        fps = None
    
    if not frames:
        raise ValueError(f"No frames found in {frame_root}")
    
    size = frames[0].size
    return frames, fps, size, video_name


def read_video_with_opencv(video_path):
    """Fallback method to read video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames, fps


def create_text_mask(image, text_boxes, dilation_kernel_size=5, margin=10):
    """Create mask from OCR text detection results"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    height, width = img_array.shape[:2]
    
    # Create blank mask
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw rectangles for each detected text region
    for box in text_boxes:
        if box is None:
            continue
            
        # Extract coordinates from box format
        if len(box) >= 4:
            # box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = box
            if isinstance(points[0], (list, tuple, np.ndarray)):
                # Flatten coordinates and add margin
                coords = []
                for point in points:
                    coords.extend([max(0, point[0] - margin), max(0, point[1] - margin)])
                
                # Also add margin to opposite corners
                coords_with_margin = []
                for i in range(0, len(coords), 2):
                    x, y = coords[i], coords[i+1]
                    coords_with_margin.extend([
                        max(0, min(width-1, x)), 
                        max(0, min(height-1, y))
                    ])
                
                # Draw filled polygon
                if len(coords_with_margin) >= 6:  # At least 3 points
                    draw.polygon(coords_with_margin, fill=255)
    
    # Convert to numpy array for dilation
    mask_array = np.array(mask)
    
    # Apply dilation to expand text regions
    if dilation_kernel_size > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)
    
    return mask_array


def process_video_ocr_mask(input_path, output_path, confidence_threshold=0.5, 
                          dilation_kernel=5, margin=10, sample_rate=1, 
                          min_text_size=10, languages=None, batch_size=1,
                          use_gpu=False, optimize_memory=True):
    """Process video to generate OCR-based masks with performance optimizations"""
    
    if not RAPIDOCR_AVAILABLE:
        raise ImportError("RapidOCR is not available. Please install with: pip install rapidocr-onnxruntime")
    
    # Initialize RapidOCR with optimizations
    try:
        # Check GPU availability for RapidOCR
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("🚀 Using GPU acceleration for OCR")
        else:
            print("💻 Using CPU for OCR processing")
        
        # Initialize with optimized settings
        ocr_engine = RapidOCR(
            providers=providers,
            
        )
    except Exception as e:
        print(f"Warning: GPU setup failed, falling back to CPU: {e}")
        ocr_engine = RapidOCR()
    
    print("Loading video frames...")
    frames, fps, size, video_name = read_frame_from_videos(input_path)
    
    print(f"Processing {len(frames)} frames from {video_name}")
    print(f"Frame size: {size}")
    print(f"Sample rate: every {sample_rate} frame(s)")
    
    # Create output directory
    mask_output_dir = os.path.join(output_path, f"{video_name}_mask")
    os.makedirs(mask_output_dir, exist_ok=True)
    
    # Calculate frames to process
    frame_indices = list(range(0, len(frames), sample_rate))
    print(f"Will process {len(frame_indices)} frames")
    
    # Process frames with memory optimization
    processed_count = 0
    total_text_regions = 0
    frame_batch = []
    index_batch = []
    
    def process_batch(batch_frames, batch_indices):
        """Process a batch of frames"""
        nonlocal total_text_regions, processed_count
        
        for frame, frame_idx in zip(batch_frames, batch_indices):
            # Convert PIL image to cv2 format for OCR
            frame_cv2 = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
            # Perform OCR
            try:
                result = ocr_engine(frame_cv2)
                
                if result is None or len(result) == 0:
                    # No text detected, create empty mask
                    mask_array = np.zeros((frame.size[1], frame.size[0]), dtype=np.uint8)
                else:
                    # Extract text boxes from OCR results
                    text_boxes = []
                    frame_text_count = 0
                    
                    for item in result:
                        if item is None or len(item) < 3:
                            continue
                        
                        # RapidOCR format: [bbox, text, confidence]
                        bbox, text, conf = item
                        
                        # Filter by confidence threshold
                        if conf < confidence_threshold:
                            continue
                        
                        # Filter by text size (number of characters)
                        if len(str(text).strip()) < min_text_size:
                            continue
                        
                        text_boxes.append(bbox)
                        frame_text_count += 1
                    
                    total_text_regions += frame_text_count
                    
                    # Create mask from detected text boxes
                    mask_array = create_text_mask(frame, text_boxes, 
                                               dilation_kernel_size=dilation_kernel, 
                                               margin=margin)
            
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                # Create empty mask on error
                mask_array = np.zeros((frame.size[1], frame.size[0]), dtype=np.uint8)
            
            # Save mask with frame index
            mask_filename = f"{frame_idx:05d}.png"
            mask_path = os.path.join(mask_output_dir, mask_filename)
            imwrite(mask_array, mask_path)
            
            processed_count += 1
            
            # Memory cleanup for large frames
            if optimize_memory:
                del frame_cv2, mask_array
                if processed_count % 50 == 0:  # Periodic cleanup
                    import gc
                    gc.collect()
    
    # Process frames in batches
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Generating OCR masks")):
        frame = frames[frame_idx]
        frame_batch.append(frame)
        index_batch.append(frame_idx)
        
        # Process batch when full or at end
        if len(frame_batch) >= batch_size or i == len(frame_indices) - 1:
            process_batch(frame_batch, index_batch)
            frame_batch = []
            index_batch = []
    
    # Generate masks for skipped frames (duplicate previous mask)
    if sample_rate > 1:
        print("Filling in skipped frames with nearest masks...")
        for i in range(len(frames)):
            mask_filename = f"{i:05d}.png"
            mask_path = os.path.join(mask_output_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                # Find nearest existing mask
                nearest_idx = (i // sample_rate) * sample_rate
                if nearest_idx >= len(frames):
                    nearest_idx = len(frames) - sample_rate
                
                nearest_mask_path = os.path.join(mask_output_dir, f"{nearest_idx:05d}.png")
                if os.path.exists(nearest_mask_path):
                    # Copy nearest mask
                    import shutil
                    shutil.copy2(nearest_mask_path, mask_path)
                else:
                    # Create empty mask
                    empty_mask = np.zeros(frames[0].size[::-1], dtype=np.uint8)
                    imwrite(empty_mask, mask_path)
    
    avg_text_per_frame = total_text_regions / processed_count if processed_count > 0 else 0
    
    print(f"\n✅ Mask generation completed!")
    print(f"📊 Statistics:")
    print(f"   - Processed frames: {processed_count}")
    print(f"   - Total frames generated: {len(frames)}")
    print(f"   - Total text regions detected: {total_text_regions}")
    print(f"   - Average text regions per frame: {avg_text_per_frame:.2f}")
    print(f"📁 Masks saved to: {mask_output_dir}")
    
    return mask_output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate OCR-based masks for video text removal")
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to input video file or image folder'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='ocr_masks',
        help='Output directory for generated masks. Default: ocr_masks'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='OCR confidence threshold (0.0-1.0). Default: 0.5'
    )
    parser.add_argument(
        '--dilation', type=int, default=5,
        help='Dilation kernel size for expanding text regions. Default: 5'
    )
    parser.add_argument(
        '--margin', type=int, default=10,
        help='Margin around detected text boxes. Default: 10'
    )
    parser.add_argument(
        '--sample_rate', type=int, default=1,
        help='Process every N-th frame (1=all frames). Default: 1'
    )
    parser.add_argument(
        '--min_text_size', type=int, default=3,
        help='Minimum text length to consider. Default: 3'
    )
    parser.add_argument(
        '--languages', type=str, nargs='+', default=None,
        help='Languages for OCR (e.g., ch en). Default: auto-detect'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for processing frames. Default: 1'
    )
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Try to use GPU acceleration for OCR. Default: False'
    )
    parser.add_argument(
        '--no_memory_optimization', action='store_true',
        help='Disable memory optimization. Default: False (optimization enabled)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: Input path '{args.input}' does not exist")
        return
    
    if not RAPIDOCR_AVAILABLE:
        print("❌ Error: RapidOCR is not installed.")
        print("Please install with: pip install rapidocr-onnxruntime")
        return
    
    # Display configuration
    print("🔧 Configuration:")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Confidence threshold: {args.confidence}")
    print(f"   Dilation kernel: {args.dilation}")
    print(f"   Margin: {args.margin}")
    print(f"   Sample rate: {args.sample_rate}")
    print(f"   Min text size: {args.min_text_size}")
    print(f"   Languages: {args.languages or 'auto-detect'}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   GPU acceleration: {args.use_gpu}")
    print(f"   Memory optimization: {not args.no_memory_optimization}")
    print()
    
    try:
        # Process video to generate OCR masks
        mask_dir = process_video_ocr_mask(
            input_path=args.input,
            output_path=args.output,
            confidence_threshold=args.confidence,
            dilation_kernel=args.dilation,
            margin=args.margin,
            sample_rate=args.sample_rate,
            min_text_size=args.min_text_size,
            languages=args.languages,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            optimize_memory=not args.no_memory_optimization
        )
        
        print(f"\n🎉 OCR mask generation completed successfully!")
        print(f"📁 Masks saved to: {mask_dir}")
        print(f"\n💡 Next steps:")
        print(f"   1. Use these masks with ProPainter:")
        print(f"      python inference_propainter.py -i {args.input} -m {mask_dir}")
        print(f"   2. For better results, consider adjusting:")
        print(f"      - Confidence threshold (--confidence)")
        print(f"      - Dilation size (--dilation)")
        print(f"      - Margin size (--margin)")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()