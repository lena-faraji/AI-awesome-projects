import cv2
import numpy as np
import tensorflow as tf
from time import time, perf_counter
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    EMOTION = "emotion"
    AGE = "age"
    GENDER = "gender"

@dataclass
class DetectionResult:
    """Data class to store detection results"""
    emotion: str
    gender: str
    age: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h

@dataclass
class PerformanceMetrics:
    """Data class to track performance metrics"""
    fps: float = 0.0
    inference_time: float = 0.0
    face_detection_time: float = 0.0
    frame_processing_time: float = 0.0
    frame_count: int = 0

class Config:
    """Configuration class for model and processing parameters"""
    # Model paths
    MODEL_PATHS = {
        'haar': 'haarcascades_models/haarcascade_frontalface_default.xml',
        'emotion': 'emotion_detection_model_100epochs_no_opt.tflite',
        'age': 'age_detection_model_50epochs_no_opt.tflite',
        'gender': 'gender_detection_model_50epochs_no_opt.tflite'
    }
    
    # Camera settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Processing parameters
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_DETECTION_MIN_NEIGHBORS = 5
    FACE_DETECTION_MIN_SIZE = (30, 30)
    
    # Model input sizes
    INPUT_SIZES = {
        ModelType.EMOTION: (48, 48),
        ModelType.AGE: (200, 200),
        ModelType.GENDER: (200, 200)
    }
    
    # Visualization settings
    COLORS = {
        'bbox': (0, 255, 255),
        'text_bg': (0, 0, 0),
        'text': (0, 255, 0),
        'performance': (255, 0, 0)
    }
    
    # Performance smoothing
    FPS_SMOOTHING = 0.9

class ModelManager:
    """Manages TensorFlow Lite models with caching and optimization"""
    
    def __init__(self):
        self.models: Dict[ModelType, Any] = {}
        self._verify_model_files()
        self._load_models()
    
    def _verify_model_files(self):
        """Verify all model files exist"""
        for key, path in Config.MODEL_PATHS.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            logger.info(f"Found model: {path}")
    
    def _load_models(self):
        """Load all TFLite models with error handling"""
        try:
            # Load Haar Cascade
            self.face_classifier = cv2.CascadeClassifier(Config.MODEL_PATHS['haar'])
            if self.face_classifier.empty():
                raise RuntimeError("Failed to load Haar Cascade classifier")
            
            # Load TFLite models
            self.models[ModelType.EMOTION] = self._load_tflite_model(
                Config.MODEL_PATHS['emotion'], 
                (1, 48, 48, 1)
            )
            self.models[ModelType.AGE] = self._load_tflite_model(
                Config.MODEL_PATHS['age'], 
                (1, 200, 200, 3)
            )
            self.models[ModelType.GENDER] = self._load_tflite_model(
                Config.MODEL_PATHS['gender'], 
                (1, 200, 200, 3)
            )
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_tflite_model(self, model_path: str, expected_shape: Tuple) -> Dict:
        """Load a TFLite model with validation"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        actual_shape = tuple(input_details[0]['shape'])
        
        if actual_shape != expected_shape:
            raise ValueError(f"Model input shape mismatch. Expected {expected_shape}, got {actual_shape}")
        
        return {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': interpreter.get_output_details()
        }
    
    def predict(self, model_type: ModelType, input_data: np.ndarray) -> np.ndarray:
        """Run inference on specified model"""
        model = self.models[model_type]
        
        # Verify input shape
        expected_shape = tuple(model['input_details'][0]['shape'])
        if input_data.shape != expected_shape:
            raise ValueError(f"Input shape {input_data.shape} doesn't match model expectation {expected_shape}")
        
        model['interpreter'].set_tensor(model['input_details'][0]['index'], input_data)
        model['interpreter'].invoke()
        return model['interpreter'].get_tensor(model['output_details'][0]['index'])

class FaceProcessor:
    """Handles face detection and processing"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.labels = {
            'emotion': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
            'gender': ['Male', 'Female']
        }
    
    def detect_faces(self, gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in grayscale frame with optimized parameters"""
        return self.model_manager.face_classifier.detectMultiScale(
            gray_frame,
            scaleFactor=Config.FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=Config.FACE_DETECTION_MIN_NEIGHBORS,
            minSize=Config.FACE_DETECTION_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    def preprocess_face(self, face_img: np.ndarray, target_size: Tuple[int, int], 
                       grayscale: bool = False) -> np.ndarray:
        """Preprocess face image for model input with optimization"""
        if face_img.size == 0:
            raise ValueError("Empty face image")
        
        # Resize with optimal interpolation
        interpolation = cv2.INTER_AREA if face_img.shape[0] > target_size[0] else cv2.INTER_LINEAR
        face_img = cv2.resize(face_img, target_size, interpolation=interpolation)
        
        if grayscale:
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=-1)
        else:
            face_img = face_img.astype('float32') / 255.0  # Normalize color images too
        
        return np.expand_dims(face_img, axis=0)  # Add batch dimension
    
    def process_face(self, face_gray: np.ndarray, face_color: np.ndarray, 
                    bbox: Tuple[int, int, int, int]) -> Optional[DetectionResult]:
        """Process a single face and return detection results"""
        try:
            # Preprocess for each model
            emotion_input = self.preprocess_face(face_gray, Config.INPUT_SIZES[ModelType.EMOTION], grayscale=True)
            demo_input = self.preprocess_face(face_color, Config.INPUT_SIZES[ModelType.AGE])
            
            # Run predictions
            emotion_pred = self.model_manager.predict(ModelType.EMOTION, emotion_input)
            gender_pred = self.model_manager.predict(ModelType.GENDER, demo_input)
            age_pred = self.model_manager.predict(ModelType.AGE, demo_input)
            
            # Calculate confidence (use emotion prediction confidence)
            confidence = float(np.max(emotion_pred))
            
            # Process results
            return DetectionResult(
                emotion=self.labels['emotion'][np.argmax(emotion_pred)],
                gender=self.labels['gender'][int(gender_pred[0][0] >= 0.5)],
                age=int(age_pred[0][0]),
                confidence=confidence,
                bbox=bbox
            )
            
        except Exception as e:
            logger.warning(f"Error processing face: {e}")
            return None

class VisualizationEngine:
    """Handles all visualization and display operations"""
    
    @staticmethod
    def draw_detection_results(frame: np.ndarray, results: List[DetectionResult], 
                             metrics: PerformanceMetrics) -> None:
        """Draw bounding boxes, labels, and performance metrics on frame"""
        for result in results:
            x, y, w, h = result.bbox
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), 
                         Config.COLORS['bbox'], 2)
            
            # Create label text with confidence
            label_text = f"{result.emotion} ({result.confidence:.2f}) | {result.gender} | {result.age}y"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), Config.COLORS['text_bg'], -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLORS['text'], 2)
        
        # Draw performance metrics
        VisualizationEngine._draw_performance_metrics(frame, metrics)
    
    @staticmethod
    def _draw_performance_metrics(frame: np.ndarray, metrics: PerformanceMetrics) -> None:
        """Draw performance metrics on frame"""
        metrics_text = [
            f"FPS: {metrics.fps:.1f}",
            f"Inference: {metrics.inference_time*1000:.1f}ms",
            f"Faces: {metrics.frame_count}",
            f"Frame: {metrics.frame_processing_time*1000:.1f}ms"
        ]
        
        for i, text in enumerate(metrics_text):
            y_position = 30 + i * 25
            cv2.putText(frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLORS['performance'], 2)

class RealTimeFaceAnalyzer:
    """Main class for real-time face analysis with performance optimization"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.face_processor = FaceProcessor(self.model_manager)
        self.visualizer = VisualizationEngine()
        self.metrics = PerformanceMetrics()
        
        self._setup_video_capture()
        self._initialize_performance_tracking()
        
        logger.info("RealTimeFaceAnalyzer initialized successfully")
    
    def _setup_video_capture(self):
        """Initialize video capture with optimized settings"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Set camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        
        # Verify camera settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera resolution: {actual_width}x{actual_height}")
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking variables"""
        self.last_fps_update = perf_counter()
        self.smoothed_fps = 30.0  # Initial guess
        self.frame_times = []
    
    def update_performance_metrics(self, processing_time: float, inference_time: float, 
                                 face_detection_time: float, face_count: int):
        """Update performance metrics with exponential smoothing"""
        current_time = perf_counter()
        
        # Calculate instantaneous FPS
        if hasattr(self, 'last_frame_time'):
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 10:
                self.frame_times.pop(0)
            
            instantaneous_fps = 1.0 / frame_time
            # Apply exponential smoothing
            self.smoothed_fps = (Config.FPS_SMOOTHING * self.smoothed_fps + 
                               (1 - Config.FPS_SMOOTHING) * instantaneous_fps)
        
        self.last_frame_time = current_time
        
        # Update metrics
        self.metrics.fps = self.smoothed_fps
        self.metrics.inference_time = inference_time
        self.metrics.face_detection_time = face_detection_time
        self.metrics.frame_processing_time = processing_time
        self.metrics.frame_count = face_count
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Tuple], List[DetectionResult]]:
        """Process a single frame and return faces and results"""
        start_time = perf_counter()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_detection_start = perf_counter()
        faces = self.face_processor.detect_faces(gray)
        face_detection_time = perf_counter() - face_detection_start
        
        results = []
        inference_time = 0.0
        
        for (x, y, w, h) in faces:
            try:
                # Extract face regions with boundary checks
                y1, y2 = max(0, y), min(y + h, frame.shape[0])
                x1, x2 = max(0, x), min(x + w, frame.shape[1])
                
                face_gray = gray[y1:y2, x1:x2]
                face_color = frame[y1:y2, x1:x2]
                
                # Skip if face region is too small
                if face_gray.size == 0 or face_color.size == 0:
                    continue
                
                # Process face
                inference_start = perf_counter()
                result = self.face_processor.process_face(face_gray, face_color, (x, y, w, h))
                inference_time += perf_counter() - inference_start
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Error processing face at {x},{y}: {e}")
                continue
        
        total_processing_time = perf_counter() - start_time
        self.update_performance_metrics(total_processing_time, inference_time, 
                                      face_detection_time, len(results))
        
        return faces, results
    
    def run(self):
        """Main execution loop with optimized performance"""
        logger.info("Starting face analysis. Press 'q' to quit, 's' to save frame.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Process frame
                faces, results = self.process_frame(frame)
                
                # Draw results
                if results:
                    self.visualizer.draw_detection_results(frame, results, self.metrics)
                
                # Display frame
                cv2.imshow('Face Analysis Dashboard - Improved', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit signal received")
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"face_analysis_{int(time())}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved as {filename}")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources cleaned up")

def main():
    """Main function with proper error handling"""
    try:
        analyzer = RealTimeFaceAnalyzer()
        analyzer.run()
    except Exception as e:
        logger.error(f"Failed to initialize Face Analyzer: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
