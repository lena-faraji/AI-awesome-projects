"""
GeoPixGAN: Enhanced Geological Image Generation with Pix2Pix GAN

An improved implementation for generating realistic geological thin section images
from segmented masks using Pix2Pix GAN. Features enhanced data handling, training
monitoring, model checkpointing, and comprehensive evaluation.

Author: Enhanced from original implementation
License: MIT
"""

import os
import glob
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json
import argparse

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

from numpy.random import randint
from numpy import vstack
from pix2pix_model import define_discriminator, define_generator, define_gan, train

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('geopixgan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for model parameters and paths."""
    
    def __init__(self):
        # Image dimensions
        self.IMG_HEIGHT = 256
        self.IMG_WIDTH = 256
        self.IMG_CHANNELS = 3
        
        # Training parameters
        self.EPOCHS = 100
        self.BATCH_SIZE = 1
        self.BUFFER_SIZE = 400
        self.LEARNING_RATE = 0.0002
        self.BETA_1 = 0.5
        
        # Paths
        self.DATA_DIR = Path("sandstone")
        self.OUTPUT_DIR = Path("outputs")
        self.MODEL_DIR = Path("models")
        self.LOG_DIR = Path("logs")
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [self.OUTPUT_DIR, self.MODEL_DIR, self.LOG_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS)
    
    @property
    def model_save_path(self) -> Path:
        return self.MODEL_DIR / "geopixgan_generator.h5"
    
    @property
    def checkpoint_path(self) -> Path:
        return self.MODEL_DIR / "checkpoint_epoch_{epoch:03d}.h5"
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class DataLoader:
    """Enhanced data loader for geological images."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_path: Path, is_mask: bool = False) -> Optional[np.ndarray]:
        """Load and preprocess a single image."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Resize image
            interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
            img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT), 
                           interpolation=interpolation)
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def load_dataset(self, images_dir: Path, masks_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and validate the entire dataset."""
        self.logger.info(f"Loading images from {images_dir} and masks from {masks_dir}")
        
        # Find image files
        image_paths = sorted(glob.glob(str(images_dir / "*.tif"))) + \
                     sorted(glob.glob(str(images_dir / "*.png"))) + \
                     sorted(glob.glob(str(images_dir / "*.jpg")))
        
        mask_paths = sorted(glob.glob(str(masks_dir / "*.tif"))) + \
                    sorted(glob.glob(str(masks_dir / "*.png"))) + \
                    sorted(glob.glob(str(masks_dir / "*.jpg")))
        
        if not image_paths or not mask_paths:
            raise ValueError(f"No images found in {images_dir} or {masks_dir}")
        
        self.logger.info(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
        
        # Load images and masks
        images = []
        masks = []
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            img = self.load_image(Path(img_path), is_mask=False)
            mask = self.load_image(Path(mask_path), is_mask=True)
            
            if img is not None and mask is not None:
                images.append(img)
                masks.append(mask)
        
        if not images:
            raise ValueError("No valid images were loaded")
        
        images_array = np.array(images, dtype=np.float32)
        masks_array = np.array(masks, dtype=np.float32)
        
        self.logger.info(f"Successfully loaded {len(images_array)} image-mask pairs")
        self.logger.info(f"Image shape: {images_array.shape}, Mask shape: {masks_array.shape}")
        
        return masks_array, images_array  # Source, Target

class Preprocessor:
    """Data preprocessing utilities."""
    
    @staticmethod
    def normalize_images(images: np.ndarray) -> np.ndarray:
        """Normalize images from [0, 255] to [-1, 1]."""
        return (images - 127.5) / 127.5
    
    @staticmethod
    def denormalize_images(images: np.ndarray) -> np.ndarray:
        """Denormalize images from [-1, 1] to [0, 1]."""
        return (images + 1) / 2.0
    
    @staticmethod
    def augment_data(images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to increase dataset diversity."""
        augmented_images = []
        augmented_masks = []
        
        for img, mask in zip(images, masks):
            # Original images
            augmented_images.append(img)
            augmented_masks.append(mask)
            
            # Horizontal flip
            augmented_images.append(cv2.flip(img, 1))
            augmented_masks.append(cv2.flip(mask, 1))
            
            # Vertical flip
            augmented_images.append(cv2.flip(img, 0))
            augmented_masks.append(cv2.flip(mask, 0))
        
        return np.array(augmented_images), np.array(augmented_masks)

class TrainingMonitor(Callback):
    """Custom callback to monitor training progress."""
    
    def __init__(self, config: Config, test_dataset: Tuple[np.ndarray, np.ndarray], 
                 monitor_interval: int = 5):
        super().__init__()
        self.config = config
        self.test_dataset = test_dataset
        self.monitor_interval = monitor_interval
        self.history = {
            'd_loss': [], 'd_acc': [],
            'g_loss': [], 'g_gan_loss': [], 'g_l1_loss': [],
            'ssim': [], 'psnr': []
        }
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Generate sample images and compute metrics at epoch end."""
        if (epoch + 1) % self.monitor_interval == 0:
            # Generate sample predictions
            X_test, y_test = self.test_dataset
            sample_idx = np.random.randint(0, len(X_test))
            src_img, tar_img = X_test[sample_idx:sample_idx+1], y_test[sample_idx:sample_idx+1]
            gen_img = self.model.generator.predict(src_img)
            
            # Compute metrics
            ssim_val, psnr_val = GeoPixGANEvaluator.compute_metrics(
                Preprocessor.denormalize_images(tar_img[0]), 
                Preprocessor.denormalize_images(gen_img[0])
            )
            
            self.history['ssim'].append(ssim_val)
            self.history['psnr'].append(psnr_val)
            
            # Save sample image
            self._save_sample_image(src_img, gen_img, tar_img, epoch, ssim_val, psnr_val)
            
            logger.info(f"Epoch {epoch+1}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.4f}")
    
    def _save_sample_image(self, src_img: np.ndarray, gen_img: np.ndarray, 
                          tar_img: np.ndarray, epoch: int, ssim_val: float, psnr_val: float):
        """Save sample comparison image."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        images = [src_img[0], gen_img[0], tar_img[0]]
        titles = ['Input Mask', 'Generated', 'Ground Truth']
        
        for ax, img, title in zip(axes, images, titles):
            denorm_img = Preprocessor.denormalize_images(img)
            ax.imshow(denorm_img[:, :, 0] if denorm_img.shape[-1] == 3 else denorm_img[:, :, 0], 
                     cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        axes[1].set_title(f'Generated\nSSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}')
        
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / f"sample_epoch_{epoch+1:03d}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()

class GeoPixGANEvaluator:
    """Comprehensive model evaluation utilities."""
    
    @staticmethod
    def compute_metrics(real_img: np.ndarray, gen_img: np.ndarray) -> Tuple[float, float]:
        """Compute SSIM and PSNR between real and generated images."""
        # Ensure images are 2D for metric computation
        if len(real_img.shape) == 3:
            real_img = real_img[:, :, 0] if real_img.shape[2] in [1, 3] else real_img[:, :, 0]
        if len(gen_img.shape) == 3:
            gen_img = gen_img[:, :, 0] if gen_img.shape[2] in [1, 3] else gen_img[:, :, 0]
        
        try:
            ssim_val = ssim(real_img, gen_img, data_range=1.0)
            psnr_val = psnr(real_img, gen_img, data_range=1.0)
            return ssim_val, psnr_val
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            return 0.0, 0.0
    
    @staticmethod
    def evaluate_model(model: Model, dataset: Tuple[np.ndarray, np.ndarray], 
                      n_samples: int = 10) -> Dict[str, float]:
        """Comprehensive model evaluation on multiple samples."""
        X, y = dataset
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        
        ssim_scores = []
        psnr_scores = []
        
        for idx in indices:
            src_img, tar_img = X[idx:idx+1], y[idx:idx+1]
            gen_img = model.predict(src_img)
            
            ssim_val, psnr_val = GeoPixGANEvaluator.compute_metrics(
                Preprocessor.denormalize_images(tar_img[0]),
                Preprocessor.denormalize_images(gen_img[0])
            )
            
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
        
        return {
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'n_samples': len(ssim_scores)
        }

class GeoPixGANTrainer:
    """Main training class for GeoPixGAN."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.generator = None
        self.discriminator = None
        self.gan_model = None
    
    def setup_models(self) -> None:
        """Initialize and compile models."""
        self.logger.info("Setting up models...")
        
        self.discriminator = define_discriminator(self.config.image_shape)
        self.generator = define_generator(self.config.image_shape)
        self.gan_model = define_gan(self.generator, self.discriminator, self.config.image_shape)
        
        self.logger.info("Models initialized successfully")
    
    def train(self, dataset: Tuple[np.ndarray, np.ndarray], 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the GeoPixGAN model."""
        self.logger.info("Starting training...")
        
        # Split dataset
        X, y = dataset
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        train_dataset = (X_train, y_train)
        val_dataset = (X_val, y_val)
        
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            filepath=str(self.config.checkpoint_path),
            save_weights_only=False,
            save_freq='epoch',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Training monitor
        monitor_callback = TrainingMonitor(self.config, val_dataset)
        callbacks.append(monitor_callback)
        
        # Train the model
        start_time = datetime.now()
        
        train(
            self.discriminator, 
            self.generator, 
            self.gan_model, 
            train_dataset,
            n_epochs=self.config.EPOCHS,
            n_batch=self.config.BATCH_SIZE,
            callbacks=callbacks
        )
        
        training_time = datetime.now() - start_time
        
        # Save final model
        self.generator.save(str(self.config.model_save_path))
        self.logger.info(f"Model saved to {self.config.model_save_path}")
        
        return {
            'training_time': str(training_time),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'history': monitor_callback.history
        }

class GeoPixGANPipeline:
    """Main pipeline class for GeoPixGAN."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.trainer = GeoPixGANTrainer(self.config)
        self.evaluator = GeoPixGANEvaluator()
        self.logger = logging.getLogger(__name__)
    
    def run_pipeline(self, use_augmentation: bool = False) -> Dict[str, Any]:
        """Run the complete GeoPixGAN pipeline."""
        self.logger.info("Starting GeoPixGAN Pipeline")
        
        results = {}
        
        try:
            # Step 1: Load data
            src_images, tar_images = self.data_loader.load_dataset(
                self.config.DATA_DIR / "images",
                self.config.DATA_DIR / "masks"
            )
            
            # Step 2: Data augmentation (optional)
            if use_augmentation:
                self.logger.info("Applying data augmentation...")
                src_images, tar_images = Preprocessor.augment_data(src_images, tar_images)
                self.logger.info(f"After augmentation: {len(src_images)} samples")
            
            # Step 3: Preprocess data
            src_images = Preprocessor.normalize_images(src_images)
            tar_images = Preprocessor.normalize_images(tar_images)
            dataset = (src_images, tar_images)
            
            # Step 4: Visualize samples
            self._visualize_samples(src_images, tar_images)
            
            # Step 5: Setup and train models
            self.trainer.setup_models()
            training_results = self.trainer.train(dataset)
            results.update(training_results)
            
            # Step 6: Evaluate model
            evaluation_results = self.evaluator.evaluate_model(
                self.trainer.generator, dataset
            )
            results['evaluation'] = evaluation_results
            
            # Step 7: Generate test predictions
            self._generate_test_predictions()
            
            # Step 8: Save results
            self._save_results(results)
            
            self.logger.info("GeoPixGAN Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return results
    
    def _visualize_samples(self, src_images: np.ndarray, tar_images: np.ndarray, 
                          n_samples: int = 3) -> None:
        """Visualize sample images."""
        plt.figure(figsize=(15, 5))
        
        for i in range(n_samples):
            # Source (mask)
            plt.subplot(2, n_samples, i + 1)
            src_display = Preprocessor.denormalize_images(src_images[i])
            plt.imshow(src_display[:, :, 0] if src_display.shape[-1] == 3 else src_display[:, :, 0], 
                      cmap='gray')
            plt.title(f"Input Mask {i+1}")
            plt.axis('off')
            
            # Target (image)
            plt.subplot(2, n_samples, i + 1 + n_samples)
            tar_display = Preprocessor.denormalize_images(tar_images[i])
            plt.imshow(tar_display[:, :, 0] if tar_display.shape[-1] == 3 else tar_display[:, :, 0], 
                      cmap='gray')
            plt.title(f"Ground Truth {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.config.OUTPUT_DIR / "data_samples.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def _generate_test_predictions(self, n_test_samples: int = 5) -> None:
        """Generate and save test predictions."""
        # Load test data (using training data for demonstration)
        src_images, tar_images = self.data_loader.load_dataset(
            self.config.DATA_DIR / "images",
            self.config.DATA_DIR / "masks"
        )
        src_images = Preprocessor.normalize_images(src_images)
        
        # Select random test samples
        indices = np.random.choice(len(src_images), min(n_test_samples, len(src_images)), 
                                replace=False)
        
        for i, idx in enumerate(indices):
            src_img = src_images[idx:idx+1]
            gen_img = self.trainer.generator.predict(src_img)
            
            # Plot comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Input mask
            src_display = Preprocessor.denormalize_images(src_img[0])
            axes[0].imshow(src_display[:, :, 0] if src_display.shape[-1] == 3 else src_display[:, :, 0], 
                          cmap='gray')
            axes[0].set_title('Input Mask')
            axes[0].axis('off')
            
            # Generated image
            gen_display = Preprocessor.denormalize_images(gen_img[0])
            axes[1].imshow(gen_display[:, :, 0] if gen_display.shape[-1] == 3 else gen_display[:, :, 0], 
                          cmap='gray')
            axes[1].set_title('Generated Image')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.config.OUTPUT_DIR / f"test_prediction_{i+1}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save training results and configuration."""
        # Save results as JSON
        results_file = self.config.OUTPUT_DIR / "training_results.json"
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=2)
        
        # Save configuration
        config_file = self.config.OUTPUT_DIR / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Configuration saved to {config_file}")

def main():
    """Main execution function with command line argument support."""
    parser = argparse.ArgumentParser(description='GeoPixGAN: Geological Image Generation')
    parser.add_argument('--data-dir', type=str, default='sandstone',
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Image size (width and height)')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Update configuration
    config = Config()
    config.DATA_DIR = Path(args.data_dir)
    config.OUTPUT_DIR = Path(args.output_dir)
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.IMG_WIDTH = args.img_size
    config.IMG_HEIGHT = args.img_size
    
    # Run pipeline
    pipeline = GeoPixGANPipeline(config)
    results = pipeline.run_pipeline(use_augmentation=args.augment)
    
    logger.info("GeoPixGAN execution completed successfully")

if __name__ == "__main__":
    main()