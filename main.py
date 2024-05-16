import sys
import argparse
import os
import json
from PIL import Image
import torchvision.transforms.functional as F
from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
from face_v2.utils import extract_data, to_serializable

def save_cropped_faces_from_tensor(response_img, save_dir, frame_id, bboxes):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    pil_image = F.to_pil_image(response_img)
    saved_faces = []
    
    for i, bbox in enumerate(bboxes):
        cropped_face = pil_image.crop(bbox)
        width, height = cropped_face.size
        
        # Check if the face is larger than 40x40 pixels
        if width >= 40 and height >= 40:
            save_path = os.path.join(save_dir, f"{frame_id}_face_{i}.jpg")
            cropped_face.save(save_path)
            print(f"Saved face {i} from frame {frame_id} to {save_path}")
            saved_faces.append(bbox)
        else:
            print(f"Skipped face {i} from frame {frame_id} due to small size: {width}x{height}px")
    
    return saved_faces

def analyze_and_save_faces(parent_dir, analyzer, cfg):
    print(f"Analyzing frames in directory: {parent_dir}")

    # Create the analyzed directory in the parent directory
    analyzed_base_dir = os.path.join(parent_dir, 'analyzed')
    if not os.path.exists(analyzed_base_dir):
        os.makedirs(analyzed_base_dir)
    
    for root, dirs, files in os.walk(parent_dir):
        # Skip the 'analyzed' directory to avoid processing results
        if 'analyzed' in root:
            continue

        if not files:
            continue

        # Create the analyzed directories in the analyzed_base_dir
        sub_dir = os.path.relpath(root, parent_dir)
        analyzed_dir = os.path.join(analyzed_base_dir, sub_dir)
        faces_save_dir = os.path.join(analyzed_dir, 'faces')
        results_path = os.path.join(analyzed_dir, 'analysis_results.json')

        if not os.path.exists(analyzed_dir):
            os.makedirs(analyzed_dir)
        if not os.path.exists(faces_save_dir):
            os.makedirs(faces_save_dir)
        
        all_data = {}

        for frame_file in files:
            frame_id = os.path.splitext(frame_file)[0]
            full_image_save_path = os.path.join(analyzed_dir, f"{frame_id}_analyzed.jpg")

            # Skip already analyzed frames
            if os.path.exists(full_image_save_path):
                print(f"Skipping already analyzed file: {frame_file}")
                continue

            frame_path = os.path.join(root, frame_file)
            if os.path.isfile(frame_path):
                print(f"Processing file: {frame_path}")
                response = analyzer.run(
                    path_image=frame_path,
                    batch_size=cfg.batch_size,
                    fix_img_size=cfg.fix_img_size,
                    return_img_data=True,
                    include_tensors=True
                )
                data = extract_data(response, include_faces=True)
                serializable_data = to_serializable(data)

                if 'bboxes' in serializable_data and serializable_data['bboxes']:
                    # Save cropped faces and filter out small faces
                    filtered_bboxes = save_cropped_faces_from_tensor(response.img, faces_save_dir, frame_id, serializable_data['bboxes'])
                    serializable_data['bboxes'] = filtered_bboxes

                # Only add to JSON if there are valid faces
                if serializable_data['bboxes']:
                    all_data[frame_id] = serializable_data

                    pil_image = F.to_pil_image(response.img)
                    pil_image.save(full_image_save_path)
                    print(f"Saved full image {frame_id} to {full_image_save_path}")

        with open(results_path, 'w') as f:
            json.dump(all_data, f, indent=4)

        print(f"Analysis completed for directory: {root}. Results and faces saved in: {analyzed_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze video frames for faces using FaceAnalyzer.")
    parser.add_argument("input_dir", type=str, help="Directory containing video frames.")
    args = parser.parse_args()

    path_config = "gpu.config.yaml"
    cfg = OmegaConf.load(path_config)
    analyzer = FaceAnalyzer(cfg.analyzer)

    analyze_and_save_faces(args.input_dir, analyzer, cfg)

if __name__ == "__main__":
    main()
