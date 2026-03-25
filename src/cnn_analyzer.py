"""
CNN-based Image Analysis Module
Direct CNN implementation for object detection, classification, and feature extraction
"""

import numpy as np
from PIL import Image
import cv2


class CNNAnalyzer:
    """Direct CNN-based image analysis"""
    
    def __init__(self):
        self.resnet_model = None
        self.yolo_net = None
        self.mobilenet_model = None
        
    @staticmethod
    def load_resnet_model():
        """Load pre-trained ResNet50 for image classification"""
        try:
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input
            
            model = ResNet50(weights='imagenet')
            return model, preprocess_input
        except Exception as e:
            print(f"Error loading ResNet: {e}")
            return None, None
    
    @staticmethod
    def load_mobilenet_model():
        """Load pre-trained MobileNetV2 (lighter, faster)"""
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            
            model = MobileNetV2(weights='imagenet')
            return model, preprocess_input
        except Exception as e:
            print(f"Error loading MobileNet: {e}")
            return None, None
    
    @staticmethod
    def classify_image_resnet(frame, top_k=5):
        """
        Classify image using ResNet50 CNN
        
        Args:
            frame: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        try:
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
            
            # Load model
            model = ResNet50(weights='imagenet')
            
            # Preprocess image - ensure RGB, use numpy directly
            img = frame.convert("RGB").resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            decoded = decode_predictions(predictions, top=top_k)[0]
            
            # Format results as 2-tuples
            results = [(str(label), float(conf)) for (_, label, conf) in decoded]
            return results
            
        except Exception as e:
            return [("Error", 0.0, str(e))]
    
    @staticmethod
    def classify_image_mobilenet(frame, top_k=5):
        """
        Classify image using MobileNetV2 CNN (faster, lighter)
        
        Args:
            frame: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
            
            # Load model
            model = MobileNetV2(weights='imagenet')
            
            # Preprocess image - ensure RGB, use numpy directly
            img = frame.convert("RGB").resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            decoded = decode_predictions(predictions, top=top_k)[0]
            
            # Format results as 2-tuples
            results = [(str(label), float(conf)) for (_, label, conf) in decoded]
            return results
            
        except Exception as e:
            return [("Error", 0.0, str(e))]
    
    @staticmethod
    def extract_features_cnn(frame):
        """
        Extract CNN features from image using ResNet50
        
        Args:
            frame: PIL Image
            
        Returns:
            Feature vector (numpy array)
        """
        try:
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input
            from tensorflow.keras.preprocessing import image as keras_image
            from tensorflow.keras.models import Model
            
            # Load model without top layers (feature extractor)
            base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            
            # Preprocess image
            img = frame.resize((224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = base_model.predict(img_array, verbose=0)
            return features.flatten()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    @staticmethod
    def detect_objects_yolo(frame):
        """
        Detect objects using YOLO CNN (requires model files)
        
        Args:
            frame: PIL Image
            
        Returns:
            List of detected objects with bounding boxes
        """
        try:
            # Convert PIL to OpenCV format
            img = np.array(frame)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            height, width = img.shape[:2]
            
            # Note: This requires YOLO weights and config files
            # For demo, we'll use a simpler approach
            
            # Load YOLO (if files exist)
            try:
                net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
                layer_names = net.getLayerNames()
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                
                # Detect objects
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)
                
                # Process detections
                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                return {"boxes": boxes, "confidences": confidences, "class_ids": class_ids}
                
            except:
                return {"error": "YOLO model files not found. Using alternative detection."}
                
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def analyze_frames_with_cnn(frames, method="mobilenet"):
        """
        Analyze multiple frames using CNN
        
        Args:
            frames: List of PIL Images
            method: "resnet", "mobilenet", or "features"
            
        Returns:
            Analysis results for all frames
        """
        results = []
        
        for i, frame in enumerate(frames):
            frame_result = {
                "frame_number": i + 1,
                "classifications": [],
                "features": None
            }
            
            # Classify image
            if method == "resnet":
                classifications = CNNAnalyzer.classify_image_resnet(frame, top_k=3)
                frame_result["classifications"] = classifications
            elif method == "mobilenet":
                classifications = CNNAnalyzer.classify_image_mobilenet(frame, top_k=3)
                frame_result["classifications"] = classifications
            elif method == "features":
                features = CNNAnalyzer.extract_features_cnn(frame)
                frame_result["features"] = features
            
            results.append(frame_result)
        
        return results
    
    @staticmethod
    def generate_cnn_description(frames, method="mobilenet"):
        """
        Generate text description of frames using CNN analysis
        
        Args:
            frames: List of PIL Images
            method: CNN method to use
            
        Returns:
            Text description of visual content
        """
        try:
            analysis = CNNAnalyzer.analyze_frames_with_cnn(frames, method=method)
            
            # Aggregate results
            all_objects = {}
            for frame_result in analysis:
                for label, conf in frame_result["classifications"]:
                    if label == "Error":
                        continue
                    if label not in all_objects:
                        all_objects[label] = []
                    all_objects[label].append(conf)
            
            # Sort by average confidence
            sorted_objects = sorted(
                all_objects.items(), 
                key=lambda x: np.mean(x[1]), 
                reverse=True
            )
            
            # Generate description
            description = "CNN Visual Analysis:\n\n"
            description += "Detected objects and scenes:\n"
            
            for label, confidences in sorted_objects[:10]:
                avg_conf = np.mean(confidences)
                freq = len(confidences)
                description += f"- {label}: {avg_conf:.1%} confidence (appears in {freq}/{len(frames)} frames)\n"
            
            return description
            
        except Exception as e:
            return f"Error generating CNN description: {e}"
    
    @staticmethod
    def compare_frames_similarity(frame1, frame2):
        """
        Compare two frames using CNN feature similarity
        
        Args:
            frame1, frame2: PIL Images
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Extract features
            features1 = CNNAnalyzer.extract_features_cnn(frame1)
            features2 = CNNAnalyzer.extract_features_cnn(frame2)
            
            if features1 is None or features2 is None:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(features1, features2) / (
                np.linalg.norm(features1) * np.linalg.norm(features2)
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error comparing frames: {e}")
            return 0.0
    
    @staticmethod
    def detect_scene_changes(frames, threshold=0.7):
        """
        Detect scene changes using CNN feature similarity
        
        Args:
            frames: List of PIL Images
            threshold: Similarity threshold (lower = more sensitive)
            
        Returns:
            List of frame indices where scenes change
        """
        try:
            scene_changes = [0]  # First frame is always a scene
            
            for i in range(1, len(frames)):
                similarity = CNNAnalyzer.compare_frames_similarity(
                    frames[i-1], 
                    frames[i]
                )
                
                if similarity < threshold:
                    scene_changes.append(i)
            
            return scene_changes
            
        except Exception as e:
            print(f"Error detecting scene changes: {e}")
            return [0]
    
    @staticmethod
    def analyze_frame_composition(frame):
        """
        Analyze frame composition using CNN and traditional CV
        
        Args:
            frame: PIL Image
            
        Returns:
            Composition analysis
        """
        try:
            # Convert to numpy
            img = np.array(frame)
            
            # Color analysis
            avg_color = np.mean(img, axis=(0, 1))
            brightness = np.mean(img)
            
            # Edge detection (Canny)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            # CNN classification
            classifications = CNNAnalyzer.classify_image_mobilenet(frame, top_k=3)
            
            return {
                "avg_color_rgb": avg_color.tolist(),
                "brightness": float(brightness),
                "edge_density": float(edge_density),
                "top_objects": classifications
            }
            
        except Exception as e:
            return {"error": str(e)}
