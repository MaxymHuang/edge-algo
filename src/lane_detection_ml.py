"""ML-based lane detection module using Fast SCNN and YOLOv8 segmentation models."""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from abc import ABC, abstractmethod
import psutil
import onnxruntime as ort

from .config import (
    FAST_SCNN_MODEL_PATH,
    FAST_SCNN_INPUT_SIZE,
    FAST_SCNN_FRAME_SKIP,
    YOLO_MODEL_PATH,
    YOLO_INPUT_SIZE,
    YOLO_FRAME_SKIP,
    YOLO_CONFIDENCE_THRESHOLD,
    STEERING_ANGLE_CLASSES,
)

logger = logging.getLogger(__name__)


class LaneDetectionModel(ABC):
    """Abstract base class for ML-based lane detection models."""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int], frame_skip: int = 1):
        """
        Initialize the lane detection model.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (width, height)
            frame_skip: Process every Nth frame
        """
        self.model_path = model_path
        self.input_size = input_size
        self.frame_skip = frame_skip
        self.frame_counter = 0
        self.session: Optional[ort.InferenceSession] = None
        self.is_loaded = False
        self.last_prediction = None
        self.last_confidence = 0.0
        
    def load_model(self):
        """Lazy load the ONNX model."""
        if self.is_loaded:
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Create ONNX Runtime session with optimizations for CPU
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4  # Use 4 threads for Pi 5
            
            # Use CPU execution provider (Pi 5 doesn't have optimized GPU support)
            providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.is_loaded = True
            
            # Log memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Model loaded: {self.model_path}, Memory: {memory_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.session is not None:
            self.session = None
        self.is_loaded = False
        self.last_prediction = None
        logger.info(f"Model unloaded: {self.model_path}")
    
    def should_process_frame(self) -> bool:
        """Check if current frame should be processed (frame skipping)."""
        self.frame_counter += 1
        if self.frame_counter >= self.frame_skip:
            self.frame_counter = 0
            return True
        return False
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame (BGR or RGB)
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        if len(normalized.shape) == 3:
            normalized = np.expand_dims(normalized, axis=0)
        
        # Convert to NCHW format: (1, H, W, C) -> (1, C, H, W)
        if normalized.shape[-1] == 3:
            normalized = np.transpose(normalized, (0, 3, 1, 2))
        
        return normalized
    
    @abstractmethod
    def postprocess(self, model_output, original_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """
        Postprocess model output to lane segmentation mask.
        
        Args:
            model_output: Raw model output
            original_shape: Original frame shape (height, width)
            
        Returns:
            Tuple of (segmentation_mask, confidence)
        """
        pass
    
    @abstractmethod
    def detect_lanes(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, Dict]:
        """
        Detect lanes in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (segmentation_mask, confidence, metadata)
        """
        pass
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.is_loaded:
            return 0.0
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024


class FastSCNNLaneDetector(LaneDetectionModel):
    """Fast SCNN lane detection model."""
    
    def __init__(self, model_path: str = None, input_size: Tuple[int, int] = None, frame_skip: int = None):
        """Initialize Fast SCNN detector."""
        model_path = model_path or FAST_SCNN_MODEL_PATH
        input_size = input_size or FAST_SCNN_INPUT_SIZE
        frame_skip = frame_skip or FAST_SCNN_FRAME_SKIP
        
        super().__init__(model_path, input_size, frame_skip)
        self.input_name = None
        self.output_name = None
    
    def load_model(self):
        """Load Fast SCNN model."""
        super().load_model()
        
        # Get input/output names
        if self.session is not None:
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
    
    def postprocess(self, model_output, original_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """
        Postprocess Fast SCNN output.
        
        Fast SCNN outputs semantic segmentation with class probabilities.
        We extract the lane class (typically class 6 or 7 for road/lane).
        """
        # Model output shape: (1, num_classes, H, W)
        if isinstance(model_output, list):
            output = model_output[0]
        else:
            output = model_output
        
        # Get lane class (assuming lane/road is class 6 or 7)
        # For Fast SCNN on Cityscapes: road=6, lane markings might be in road class
        # We'll use the road class as lane detection
        lane_class_idx = 6  # Road class in Cityscapes
        
        if len(output.shape) == 4:
            # Shape: (1, C, H, W)
            lane_probs = output[0, lane_class_idx, :, :]
        else:
            # Shape: (C, H, W) or (H, W, C)
            if output.shape[0] < output.shape[-1]:
                # (C, H, W)
                lane_probs = output[lane_class_idx, :, :]
            else:
                # (H, W, C) - take argmax first
                lane_class = np.argmax(output, axis=-1)
                lane_probs = (lane_class == lane_class_idx).astype(np.float32)
        
        # Threshold to get binary mask
        threshold = 0.5
        lane_mask = (lane_probs > threshold).astype(np.uint8) * 255
        
        # Resize to original frame size
        h, w = original_shape[:2]
        lane_mask = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Calculate confidence as mean probability
        confidence = float(np.mean(lane_probs))
        
        return lane_mask, confidence
    
    def detect_lanes(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, Dict]:
        """
        Detect lanes using Fast SCNN.
        
        Returns:
            (segmentation_mask, confidence, metadata)
        """
        if not self.should_process_frame() and self.last_prediction is not None:
            # Return cached prediction
            return self.last_prediction, self.last_confidence, {"cached": True}
        
        if not self.is_loaded:
            self.load_model()
        
        if self.session is None:
            return None, 0.0, {"error": "Model not loaded"}
        
        try:
            original_shape = frame.shape
            preprocessed = self.preprocess(frame)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed})
            
            # Postprocess
            lane_mask, confidence = self.postprocess(outputs[0], original_shape)
            
            # Cache result
            self.last_prediction = lane_mask
            self.last_confidence = confidence
            
            return lane_mask, confidence, {
                "cached": False,
                "model": "fast_scnn",
                "input_size": self.input_size,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Fast SCNN inference error: {e}")
            return None, 0.0, {"error": str(e)}


class YOLOLaneDetector(LaneDetectionModel):
    """YOLOv8 segmentation lane detection model."""
    
    def __init__(self, model_path: str = None, input_size: Tuple[int, int] = None, 
                 frame_skip: int = None, confidence_threshold: float = None):
        """Initialize YOLOv8 detector."""
        model_path = model_path or YOLO_MODEL_PATH
        input_size = input_size or YOLO_INPUT_SIZE
        frame_skip = frame_skip or YOLO_FRAME_SKIP
        confidence_threshold = confidence_threshold or YOLO_CONFIDENCE_THRESHOLD
        
        super().__init__(model_path, input_size, frame_skip)
        self.confidence_threshold = confidence_threshold
        self.input_name = None
        self.output_names = None
    
    def load_model(self):
        """Load YOLOv8 model."""
        super().load_model()
        
        if self.session is not None:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
    
    def postprocess(self, model_output, original_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """
        Postprocess YOLOv8 segmentation output.
        
        YOLOv8 segmentation outputs:
        - Boxes: (num_detections, 4) - bounding boxes
        - Masks: (num_detections, H, W) - segmentation masks
        - Scores: (num_detections,) - confidence scores
        """
        # YOLOv8 ONNX output format varies, but typically:
        # output[0]: boxes (1, num_detections, 4)
        # output[1]: scores (1, num_detections)
        # output[2]: masks (1, num_detections, mask_h, mask_w) or segmentation output
        
        if isinstance(model_output, list):
            outputs = model_output
        else:
            outputs = [model_output]
        
        h, w = original_shape[:2]
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        max_confidence = 0.0
        
        try:
            # YOLOv8 segmentation typically has:
            # - Detection output with boxes and scores
            # - Segmentation masks
            
            # Find output with masks (usually the largest tensor)
            mask_output = None
            for output in outputs:
                if len(output.shape) >= 3:
                    if mask_output is None or output.size > mask_output.size:
                        mask_output = output
            
            if mask_output is not None:
                # Process masks
                # Shape might be (1, num_detections, mask_h, mask_w) or (num_detections, mask_h, mask_w)
                if len(mask_output.shape) == 4:
                    masks = mask_output[0]  # Remove batch dimension
                else:
                    masks = mask_output
                
                # Get scores if available
                scores = None
                for output in outputs:
                    if len(output.shape) == 2 and output.shape[0] == masks.shape[0]:
                        scores = output[0] if len(output.shape) == 2 else output
                        break
                
                # Combine masks above confidence threshold
                for i in range(masks.shape[0]):
                    mask = masks[i]
                    conf = float(scores[i]) if scores is not None else 1.0
                    
                    if conf >= self.confidence_threshold:
                        # Resize mask to original frame size
                        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        # Threshold mask
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        # Combine with lane mask
                        lane_mask = np.maximum(lane_mask, mask_binary)
                        max_confidence = max(max_confidence, conf)
            
            # If no masks found, try to extract from first output (might be direct segmentation)
            if max_confidence == 0.0 and len(outputs) > 0:
                output = outputs[0]
                if len(output.shape) == 4:
                    # (1, C, H, W) - take first channel or argmax
                    seg = output[0]
                    if seg.shape[0] > 1:
                        # Multi-class: take argmax
                        seg_class = np.argmax(seg, axis=0)
                        # Assume lane class is 0 or 1
                        lane_mask = (seg_class == 0).astype(np.uint8) * 255
                    else:
                        # Single channel: threshold
                        lane_mask = (seg[0] > 0.5).astype(np.uint8) * 255
                    lane_mask = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    max_confidence = 0.7  # Default confidence
            
        except Exception as e:
            logger.warning(f"YOLO postprocessing error: {e}, using fallback")
            # Fallback: try to use first output as segmentation
            if len(outputs) > 0:
                output = outputs[0]
                if len(output.shape) >= 2:
                    h_out, w_out = output.shape[-2], output.shape[-1]
                    if len(output.shape) == 4:
                        seg = output[0, 0] if output.shape[1] > 0 else output[0]
                    else:
                        seg = output
                    lane_mask = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
                    lane_mask = ((lane_mask > 0.5) * 255).astype(np.uint8)
                    max_confidence = 0.5
        
        return lane_mask, max_confidence
    
    def detect_lanes(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, Dict]:
        """
        Detect lanes using YOLOv8 segmentation.
        
        Returns:
            (segmentation_mask, confidence, metadata)
        """
        if not self.should_process_frame() and self.last_prediction is not None:
            # Return cached prediction
            return self.last_prediction, self.last_confidence, {"cached": True}
        
        if not self.is_loaded:
            self.load_model()
        
        if self.session is None:
            return None, 0.0, {"error": "Model not loaded"}
        
        try:
            original_shape = frame.shape
            preprocessed = self.preprocess(frame)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
            
            # Postprocess
            lane_mask, confidence = self.postprocess(outputs, original_shape)
            
            # Cache result
            self.last_prediction = lane_mask
            self.last_confidence = confidence
            
            return lane_mask, confidence, {
                "cached": False,
                "model": "yolo",
                "input_size": self.input_size,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return None, 0.0, {"error": str(e)}
