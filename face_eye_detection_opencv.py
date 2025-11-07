"""
Advanced Face and Eye Detection System
Supporting multiple detection methods: Haar Cascade, DNN, MTCNN, and MediaPipe
Complete all-in-one implementation
"""

import cv2
import numpy as np
import time
import logging
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class DetectionMethod(Enum):
    HAAR_CASCADE = "haar"
    DNN = "dnn"
    MTCNN = "mtcnn"
    MEDIAPIPE = "mediapipe"

@dataclass
class DetectionConfig:
    """Configuration for detection parameters"""
    method: DetectionMethod = DetectionMethod.DNN
    confidence_threshold: float = 0.7
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (30, 30)
    enable_landmarks: bool = True
    enable_emotion: bool = False
    enable_age_gender: bool = False
    blur_faces: bool = False
    draw_landmarks: bool = True

class AdvancedFaceDetector:
    """Advanced face detection with multiple methods and enhanced features"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.detection_method = config.method
        self.face_detector = None
        self.eye_detector = None
        self.dnn_net = None
        self.mtcnn_detector = None
        self.mediapipe_face_detection = None
        self.landmark_detector = None
        self.logger = None
        
        self._setup_logging()
        self._initialize_detectors()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_detectors(self):
        """Initialize selected detection method"""
        try:
            if self.detection_method == DetectionMethod.HAAR_CASCADE:
                self._initialize_haar_cascade()
            elif self.detection_method == DetectionMethod.DNN:
                self._initialize_dnn()
            elif self.detection_method == DetectionMethod.MTCNN:
                self._initialize_mtcnn()
            elif self.detection_method == DetectionMethod.MEDIAPIPE:
                self._initialize_mediapipe()
                
            self.logger.info(f"Initialized {self.detection_method.value} detector")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.detection_method.value}: {e}")
            # Fallback to Haar Cascade
            self.detection_method = DetectionMethod.HAAR_CASCADE
            self._initialize_haar_cascade()
    
    def _initialize_haar_cascade(self):
        """Initialize Haar Cascade detectors"""
        cascade_path = Path('haarcascades_models')
        cascade_path.mkdir(exist_ok=True)
        
        # Download Haar Cascade models if they don't exist
        self._download_haar_models()
        
        # Face detectors
        self.face_detector = cv2.CascadeClassifier()
        if not self.face_detector.load(str(cascade_path / 'haarcascade_frontalface_default.xml')):
            raise Exception("Error loading face cascade")
        
        # Eye detectors
        self.eye_detector = cv2.CascadeClassifier()
        if not self.eye_detector.load(str(cascade_path / 'haarcascade_eye.xml')):
            self.logger.warning("Eye cascade not available")
        
        # Additional cascade files for better detection
        self.profile_face_detector = cv2.CascadeClassifier()
        self.profile_face_detector.load(str(cascade_path / 'haarcascade_profileface.xml'))
        
        self.smile_detector = cv2.CascadeClassifier()
        self.smile_detector.load(str(cascade_path / 'haarcascade_smile.xml'))
    
    def _download_haar_models(self):
        """Download Haar Cascade models if they don't exist"""
        import urllib.request
        
        models = {
            'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
            'haarcascade_profileface.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml',
            'haarcascade_smile.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml',
        }
        
        for filename, url in models.items():
            output_path = Path('haarcascades_models') / filename
            if not output_path.exists():
                self.logger.info(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, output_path)
                    self.logger.info(f"Downloaded {filename}")
                except Exception as e:
                    self.logger.warning(f"Failed to download {filename}: {e}")
    
    def _initialize_dnn(self):
        """Initialize DNN-based face detector"""
        model_path = Path('dnn_models')
        model_path.mkdir(exist_ok=True)
        
        # Try to initialize DNN model
        prototxt = model_path / "deploy.prototxt"
        model_file = model_path / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        
        # Download DNN models if they don't exist
        if not prototxt.exists() or not model_file.exists():
            self.logger.warning("DNN models not found, using built-in Haar Cascade")
            self._initialize_haar_cascade()
            self.detection_method = DetectionMethod.HAAR_CASCADE
            return
        
        try:
            self.dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model_file))
            self.logger.info("DNN model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load DNN model: {e}")
            self._initialize_haar_cascade()
            self.detection_method = DetectionMethod.HAAR_CASCADE
    
    def _initialize_mtcnn(self):
        """Initialize MTCNN detector"""
        try:
            # Try to import MTCNN
            from mtcnn import MTCNN
            self.mtcnn_detector = MTCNN()
            self.logger.info("MTCNN detector initialized")
        except ImportError:
            self.logger.warning("MTCNN not installed, falling back to DNN")
            self._initialize_dnn()
            self.detection_method = DetectionMethod.DNN
        except Exception as e:
            self.logger.error(f"Failed to initialize MTCNN: {e}")
            self._initialize_dnn()
            self.detection_method = DetectionMethod.DNN
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe face detection"""
        try:
            # Try to import MediaPipe
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.mediapipe_face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            self.logger.info("MediaPipe face detection initialized")
        except ImportError:
            self.logger.warning("MediaPipe not installed, falling back to DNN")
            self._initialize_dnn()
            self.detection_method = DetectionMethod.DNN
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe: {e}")
            self._initialize_dnn()
            self.detection_method = DetectionMethod.DNN
    
    def detect_faces_haar(self, gray: np.ndarray) -> List[Tuple]:
        """Detect faces using Haar Cascade"""
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple]:
        """Detect faces using DNN"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
    
    def detect_faces_mtcnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN"""
        try:
            results = self.mtcnn_detector.detect_faces(frame)
            formatted_results = []
            for result in results:
                if result['confidence'] > self.config.confidence_threshold:
                    formatted_results.append(result)
            return formatted_results
        except Exception as e:
            self.logger.error(f"MTCNN detection failed: {e}")
            return []
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mediapipe_face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    if detection.score[0] > self.config.confidence_threshold:
                        bbox = detection.location_data.relative_bounding_box
                        (h, w) = frame.shape[:2]
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        faces.append({
                            'box': (x, y, width, height),
                            'confidence': detection.score[0],
                            'keypoints': detection.location_data.relative_keypoints
                        })
            return faces
        except Exception as e:
            self.logger.error(f"MediaPipe detection failed: {e}")
            return []
    
    def detect_eyes(self, gray_roi: np.ndarray) -> List[Tuple]:
        """Detect eyes in face ROI"""
        if self.eye_detector is None:
            return []
        
        try:
            eyes = self.eye_detector.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            return eyes
        except Exception as e:
            self.logger.warning(f"Eye detection failed: {e}")
            return []
    
    def detect_smile(self, gray_roi: np.ndarray) -> List[Tuple]:
        """Detect smile in face ROI"""
        if not hasattr(self, 'smile_detector') or self.smile_detector is None:
            return []
        
        try:
            smiles = self.smile_detector.detectMultiScale(
                gray_roi,
                scaleFactor=1.8,
                minNeighbors=20,
                minSize=(25, 25)
            )
            return smiles
        except Exception as e:
            self.logger.warning(f"Smile detection failed: {e}")
            return []
    
    def enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better detection"""
        try:
            # Contrast Limited Adaptive Histogram Equalization
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Denoising
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            return enhanced
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return frame
    
    def draw_detection_info(self, frame: np.ndarray, faces: List, 
                          eyes: List = None, smiles: List = None) -> np.ndarray:
        """Draw detection information on frame"""
        output_frame = frame.copy()
        
        for i, face in enumerate(faces):
            if self.detection_method == DetectionMethod.MTCNN:
                x, y, w, h = face['box']
                confidence = face['confidence']
            elif self.detection_method == DetectionMethod.MEDIAPIPE:
                x, y, w, h = face['box']
                confidence = face['confidence']
            else:
                x, y, w, h = face
                confidence = 1.0
            
            # Ensure coordinates are valid
            x, y, w, h = int(x), int(y), int(w), int(h)
            if w <= 0 or h <= 0:
                continue
            
            # Draw face bounding box
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            thickness = 3 if confidence > 0.9 else 2
            
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, thickness)
            
            # Draw confidence score
            if confidence < 1.0:
                cv2.putText(output_frame, f'Face: {confidence:.2f}', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw face ID
            cv2.putText(output_frame, f'ID: {i}', (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Process ROI for additional detections if within bounds
            if (y < output_frame.shape[0] and x < output_frame.shape[1] and 
                y+h <= output_frame.shape[0] and x+w <= output_frame.shape[1]):
                
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                
                # Detect eyes
                if eyes and i < len(eyes):
                    for (ex, ey, ew, eh) in eyes[i]:
                        ex, ey, ew, eh = int(ex), int(ey), int(ew), int(eh)
                        if (ex >= 0 and ey >= 0 and ex+ew <= w and ey+eh <= h):
                            cv2.rectangle(output_frame, (x+ex, y+ey), 
                                        (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
                
                # Detect smile
                if smiles and i < len(smiles):
                    for (sx, sy, sw, sh) in smiles[i]:
                        sx, sy, sw, sh = int(sx), int(sy), int(sw), int(sh)
                        if (sx >= 0 and sy >= 0 and sx+sw <= w and sy+sh <= h):
                            cv2.rectangle(output_frame, (x+sx, y+sy), 
                                        (x+sx+sw, y+sy+sh), (0, 0, 255), 2)
                            cv2.putText(output_frame, 'Smile', (x+sx, y+sy-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return output_frame
    
    def blur_faces(self, frame: np.ndarray, faces: List) -> np.ndarray:
        """Apply blur to detected faces for privacy"""
        blurred_frame = frame.copy()
        
        for face in faces:
            if self.detection_method in [DetectionMethod.MTCNN, DetectionMethod.MEDIAPIPE]:
                x, y, w, h = face['box']
            else:
                x, y, w, h = face
            
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure coordinates are within frame boundaries
            if (x >= 0 and y >= 0 and x+w <= blurred_frame.shape[1] and 
                y+h <= blurred_frame.shape[0] and w > 0 and h > 0):
                
                # Extract face ROI
                face_roi = blurred_frame[y:y+h, x:x+w]
                
                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                
                # Put blurred face back
                blurred_frame[y:y+h, x:x+w] = blurred_face
        
        return blurred_frame
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame and return detection results"""
        start_time = time.time()
        
        # Enhance image for better detection
        enhanced_frame = self.enhance_image(frame)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces based on selected method
        faces = []
        eyes = []
        smiles = []
        
        try:
            if self.detection_method == DetectionMethod.HAAR_CASCADE:
                faces = self.detect_faces_haar(gray)
            elif self.detection_method == DetectionMethod.DNN:
                faces = self.detect_faces_dnn(enhanced_frame)
            elif self.detection_method == DetectionMethod.MTCNN:
                faces = self.detect_faces_mtcnn(enhanced_frame)
            elif self.detection_method == DetectionMethod.MEDIAPIPE:
                faces = self.detect_faces_mediapipe(enhanced_frame)
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            faces = []
        
        # Additional detections for each face
        for face in faces:
            if self.detection_method in [DetectionMethod.MTCNN, DetectionMethod.MEDIAPIPE]:
                x, y, w, h = face['box']
            else:
                x, y, w, h = face
            
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure ROI is valid
            if (x >= 0 and y >= 0 and x+w <= gray.shape[1] and 
                y+h <= gray.shape[0] and w > 0 and h > 0):
                
                roi_gray = gray[y:y+h, x:x+w]
                
                # Detect eyes
                detected_eyes = self.detect_eyes(roi_gray)
                eyes.append(detected_eyes)
                
                # Detect smile
                detected_smiles = self.detect_smile(roi_gray)
                smiles.append(detected_smiles)
            else:
                eyes.append([])
                smiles.append([])
        
        processing_time = time.time() - start_time
        
        return {
            'faces': faces,
            'eyes': eyes,
            'smiles': smiles,
            'processing_time': processing_time,
            'face_count': len(faces)
        }

class DetectionAnalytics:
    """Analytics and reporting for face detection"""
    
    def __init__(self):
        self.detection_history = []
        self.start_time = time.time()
    
    def update(self, detection_result: Dict):
        """Update analytics with new detection result"""
        detection_result['timestamp'] = time.time()
        self.detection_history.append(detection_result)
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        if not self.detection_history:
            return {}
        
        face_counts = [result['face_count'] for result in self.detection_history]
        processing_times = [result['processing_time'] for result in self.detection_history]
        
        return {
            'total_frames': len(self.detection_history),
            'average_faces': np.mean(face_counts) if face_counts else 0,
            'max_faces': np.max(face_counts) if face_counts else 0,
            'min_faces': np.min(face_counts) if face_counts else 0,
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'total_detection_time': time.time() - self.start_time,
            'fps': len(self.detection_history) / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
        }
    
    def generate_report(self) -> str:
        """Generate analytics report"""
        stats = self.get_statistics()
        report = {
            'detection_statistics': stats,
            'detection_history': self.detection_history[-100:]  # Last 100 frames
        }
        
        try:
            with open('detection_report.json', 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            return f"Error generating report: {e}"
        
        return f"Report generated with {stats['total_frames']} frames analyzed"

def test_camera_sources():
    """Test available camera sources"""
    print("Testing camera sources...")
    cams_test = 10
    available_cams = []
    for i in range(cams_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cams.append(i)
                print(f"Camera {i}: Available")
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    return available_cams

def demo_static_image_detection():
    """Demo face detection on static image"""
    print("=== Static Image Detection Demo ===")
    
    # Test image path
    test_image = "obamas.jpg"
    
    if not Path(test_image).exists():
        print(f"Test image {test_image} not found. Please provide an image path.")
        return
    
    # Test all detection methods
    methods = [DetectionMethod.HAAR_CASCADE, DetectionMethod.DNN]
    
    # Try to initialize advanced methods if available
    try:
        from mtcnn import MTCNN
        methods.append(DetectionMethod.MTCNN)
    except ImportError:
        print("MTCNN not available, skipping...")
    
    try:
        import mediapipe as mp
        methods.append(DetectionMethod.MEDIAPIPE)
    except ImportError:
        print("MediaPipe not available, skipping...")
    
    for method in methods:
        print(f"\nTesting {method.value} method...")
        
        config = DetectionConfig(method=method)
        detector = AdvancedFaceDetector(config)
        
        image = cv2.imread(test_image)
        if image is None:
            print(f"Error: Could not load image {test_image}")
            continue
        
        result = detector.process_frame(image)
        
        # Draw detections
        output_image = detector.draw_detection_info(
            image, result['faces'], result['eyes'], result['smiles']
        )
        
        # Add info text
        cv2.putText(output_image, f"Method: {method.value}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_image, f"Faces: {result['face_count']}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_image, f"Time: {result['processing_time']*1000:.1f}ms", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow(f'Face Detection - {method.value}', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Advanced Face Detection System')
    parser.add_argument('--method', type=str, default='dnn', 
                       choices=['haar', 'dnn', 'mtcnn', 'mediapipe'],
                       help='Detection method to use')
    parser.add_argument('--source', type=int, default=0,
                       help='Camera source (0 for default camera)')
    parser.add_argument('--blur-faces', action='store_true',
                       help='Blur detected faces for privacy')
    parser.add_argument('--image', type=str,
                       help='Process static image instead of video')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold for DNN detection')
    parser.add_argument('--test-cameras', action='store_true',
                       help='Test available camera sources')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration on static image')
    
    args = parser.parse_args()
    
    if args.test_cameras:
        test_camera_sources()
        return
    
    if args.demo:
        demo_static_image_detection()
        return
    
    # Configuration
    config = DetectionConfig(
        method=DetectionMethod(args.method),
        confidence_threshold=args.confidence,
        blur_faces=args.blur_faces
    )
    
    # Initialize detector
    detector = AdvancedFaceDetector(config)
    analytics = DetectionAnalytics()
    
    if args.image:
        # Process static image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        result = detector.process_frame(image)
        analytics.update(result)
        
        # Draw detections
        if config.blur_faces:
            output_image = detector.blur_faces(image, result['faces'])
        else:
            output_image = detector.draw_detection_info(
                image, result['faces'], result['eyes'], result['smiles']
            )
        
        # Add analytics info
        cv2.putText(output_image, f"Faces: {result['face_count']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_image, f"Time: {result['processing_time']*1000:.1f}ms", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_image, f"Method: {args.method}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        # Process video stream
        print("Initializing camera...")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open camera source {args.source}")
            # Test available cameras
            available_cams = test_camera_sources()
            if available_cams:
                print(f"Available cameras: {available_cams}")
                cap = cv2.VideoCapture(available_cams[0])
            else:
                print("No cameras available!")
                return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting video capture. Press 'q' to quit, 's' to save snapshot")
        print(f"Using detection method: {args.method}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Process frame
            result = detector.process_frame(frame)
            analytics.update(result)
            
            # Apply blur or draw detections
            if config.blur_faces:
                output_frame = detector.blur_faces(frame, result['faces'])
            else:
                output_frame = detector.draw_detection_info(
                    frame, result['faces'], result['eyes'], result['smiles']
                )
            
            # Calculate real-time FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            else:
                fps = 0
            
            # Display analytics
            stats = analytics.get_statistics()
            cv2.putText(output_frame, f"Faces: {result['face_count']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Time: {result['processing_time']*1000:.1f}ms", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Method: {args.method}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Advanced Face Detection', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                timestamp = int(time.time())
                cv2.imwrite(f'snapshot_{timestamp}.jpg', output_frame)
                print(f"Snapshot saved: snapshot_{timestamp}.jpg")
            elif key == ord('d'):
                # Toggle detection method (demo)
                methods = ['haar', 'dnn', 'mtcnn', 'mediapipe']
                current_index = methods.index(args.method)
                next_index = (current_index + 1) % len(methods)
                args.method = methods[next_index]
                config.method = DetectionMethod(args.method)
                detector = AdvancedFaceDetector(config)
                print(f"Switched to detection method: {args.method}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate final report
        report = analytics.generate_report()
        print(f"\n=== Detection Report ===")
        print(report)
        
        stats = analytics.get_statistics()
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Average faces per frame: {stats['average_faces']:.2f}")
        print(f"Average processing time: {stats['average_processing_time']*1000:.1f}ms")
        print(f"Average FPS: {stats['fps']:.1f}")

if __name__ == "__main__":
    main()
