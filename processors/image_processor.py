#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Processor Module

This module implements image processing capabilities for the AI Problem Solver,
allowing it to handle and analyze image-based inputs including diagrams,
charts, handwritten text, and more.
"""

import logging
import os
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple

# Third-party imports (these would need to be installed via requirements.txt)
try:
    import numpy as np
    from PIL import Image
    import cv2
    from langchain.schema import HumanMessage
except ImportError:
    logging.warning("Some image processing dependencies are not installed. "
                   "Install them with: pip install pillow opencv-python numpy")

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image processor for handling image-based inputs.
    
    This processor analyzes and extracts information from images,
    including diagrams, charts, handwritten text, and more.
    
    Attributes:
        llm: Optional language model for image understanding
        settings: Application settings
    """
    
    def __init__(self, llm=None, settings=None):
        """
        Initialize the Image Processor.
        
        Args:
            llm: Optional language model for image understanding
            settings: Optional application settings
        """
        self.llm = llm
        self.settings = settings
        logger.info("Image processor initialized")
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image and extract relevant information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict[str, Any]: Processed image with metadata and extracted information
        """
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {
                "error": f"Image file not found: {image_path}",
                "content_type": "error"
            }
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Initialize result
            result = {
                "content_type": "image",
                "file_path": image_path,
                "metadata": {},
                "extracted_text": "",
                "description": ""
            }
            
            # Extract basic metadata
            result["metadata"] = self._extract_metadata(image)
            
            # Detect image type
            result["metadata"]["image_type"] = self._detect_image_type(image)
            
            # Extract text if possible
            if self._has_text(image):
                result["extracted_text"] = self._extract_text(image_path)
            
            # Generate description if LLM is available
            if self.llm:
                result["description"] = self._generate_description(image_path)
            
            logger.debug(f"Processed image: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "error": f"Error processing image: {str(e)}",
                "content_type": "error"
            }
    
    def _extract_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract basic metadata from an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict[str, Any]: Image metadata
        """
        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "aspect_ratio": image.width / image.height if image.height > 0 else 0
        }
        
        # Extract EXIF data if available
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            if exif:
                metadata["exif"] = {}
                for tag_id, value in exif.items():
                    # Convert tag ID to string to ensure JSON serialization
                    metadata["exif"][str(tag_id)] = str(value)
        
        return metadata
    
    def _detect_image_type(self, image: Image.Image) -> str:
        """
        Detect the type of image (photo, diagram, chart, etc.).
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Detected image type
        """
        # Convert to numpy array for OpenCV processing
        try:
            img_array = np.array(image.convert('RGB'))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
            
            # Color analysis
            color_std = np.std(img_array, axis=(0, 1)).mean()
            
            # Heuristic classification
            if edge_ratio > 0.1 and color_std < 50:
                return "diagram"
            elif edge_ratio > 0.05 and color_std < 70:
                return "chart"
            elif edge_ratio < 0.05 and color_std < 30:
                return "document"
            elif self._detect_handwriting(gray):
                return "handwritten"
            else:
                return "photo"
                
        except Exception as e:
            logger.warning(f"Error detecting image type: {str(e)}")
            return "unknown"
    
    def _detect_handwriting(self, gray_image: np.ndarray) -> bool:
        """
        Detect if an image contains handwriting.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            bool: Whether handwriting is detected
        """
        try:
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours
            significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
            
            # Analyze contour properties
            if len(significant_contours) > 20:
                # Calculate average contour properties
                avg_area = np.mean([cv2.contourArea(c) for c in significant_contours])
                avg_perimeter = np.mean([cv2.arcLength(c, True) for c in significant_contours])
                
                # Handwriting typically has many small, irregular contours
                if avg_area < 500 and avg_perimeter < 100:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error detecting handwriting: {str(e)}")
            return False
    
    def _has_text(self, image: Image.Image) -> bool:
        """
        Detect if an image likely contains text.
        
        Args:
            image: PIL Image object
            
        Returns:
            bool: Whether text is detected
        """
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image.convert('RGB'))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            text_like_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text-like contours typically have certain aspect ratios and sizes
                if 0.1 < aspect_ratio < 15 and 10 < w < 300 and 5 < h < 100:
                    text_like_contours += 1
            
            # If we have enough text-like contours, assume there's text
            return text_like_contours > 10
            
        except Exception as e:
            logger.warning(f"Error detecting text: {str(e)}")
            return False
    
    def _extract_text(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        This method requires pytesseract to be installed.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Extracted text
        """
        try:
            # Try to import pytesseract
            import pytesseract
            
            # Load image
            image = Image.open(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(image)
            return text.strip()
            
        except ImportError:
            logger.warning("pytesseract is not installed. Install it with: pip install pytesseract")
            return "[OCR not available: pytesseract not installed]"
            
        except Exception as e:
            logger.warning(f"Error extracting text: {str(e)}")
            return f"[Error extracting text: {str(e)}]"
    
    def _generate_description(self, image_path: str) -> str:
        """
        Generate a description of the image using a language model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Image description
        """
        if not self.llm:
            return ""
        
        try:
            # Read image and convert to base64
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create message with image
            message = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image in detail, including any text, diagrams, or relevant information visible."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                )
            ]
            
            # Generate description
            response = self.llm.generate([message])
            description = response.generations[0][0].text.strip()
            
            return description
            
        except Exception as e:
            logger.warning(f"Error generating image description: {str(e)}")
            return f"[Error generating description: {str(e)}]"
    
    def analyze_diagram(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a diagram or chart image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Process the image first
        result = self.process(image_path)
        
        # If there was an error or it's not a diagram/chart, return the result as is
        if "error" in result or (result["metadata"]["image_type"] != "diagram" and 
                                result["metadata"]["image_type"] != "chart"):
            return result
        
        # If we have a language model, use it to analyze the diagram
        if self.llm:
            try:
                # Read image and convert to base64
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Create message with image
                message = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Analyze this diagram/chart in detail. Identify the type of diagram, key components, relationships, data points, trends, and any conclusions that can be drawn. If there's text, include it in your analysis."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    )
                ]
                
                # Generate analysis
                response = self.llm.generate([message])
                analysis = response.generations[0][0].text.strip()
                
                # Add analysis to result
                result["analysis"] = analysis
                
            except Exception as e:
                logger.warning(f"Error analyzing diagram: {str(e)}")
                result["analysis"] = f"[Error analyzing diagram: {str(e)}]"
        
        return result
    
    def extract_math_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract mathematical expressions from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict[str, Any]: Extracted math expressions and analysis
        """
        # Process the image first
        result = self.process(image_path)
        
        # If there was an error, return the result as is
        if "error" in result:
            return result
        
        # If we have a language model, use it to extract math
        if self.llm:
            try:
                # Read image and convert to base64
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Create message with image
                message = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Extract and transcribe all mathematical expressions, equations, formulas, or calculations visible in this image. Format them using LaTeX notation where appropriate. If there are multiple expressions, number them and provide them in the order they appear."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    )
                ]
                
                # Generate extraction
                response = self.llm.generate([message])
                math_extraction = response.generations[0][0].text.strip()
                
                # Add extraction to result
                result["math_expressions"] = math_extraction
                
                # If it looks like there are math expressions, also get an explanation
                if len(math_extraction) > 10 and ('=' in math_extraction or '\\' in math_extraction):
                    # Create message for explanation
                    explanation_message = [
                        HumanMessage(
                            content=[
                                {"type": "text", "text": f"Explain the following mathematical expressions in detail, including what they represent and how they might be used:\n\n{math_extraction}"},
                            ]
                        )
                    ]
                    
                    # Generate explanation
                    explanation_response = self.llm.generate([explanation_message])
                    explanation = explanation_response.generations[0][0].text.strip()
                    
                    # Add explanation to result
                    result["math_explanation"] = explanation
                
            except Exception as e:
                logger.warning(f"Error extracting math from image: {str(e)}")
                result["math_expressions"] = f"[Error extracting math: {str(e)}]"
        
        return result
    
    def extract_code_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract code snippets from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict[str, Any]: Extracted code and analysis
        """
        # Process the image first
        result = self.process(image_path)
        
        # If there was an error, return the result as is
        if "error" in result:
            return result
        
        # If we have a language model, use it to extract code
        if self.llm:
            try:
                # Read image and convert to base64
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Create message with image
                message = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Extract and transcribe all code visible in this image. Preserve indentation and formatting as much as possible. Identify the programming language if possible."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    )
                ]
                
                # Generate extraction
                response = self.llm.generate([message])
                code_extraction = response.generations[0][0].text.strip()
                
                # Add extraction to result
                result["code"] = code_extraction
                
                # If it looks like there is code, also get an explanation
                if len(code_extraction) > 20 and ('(' in code_extraction or '{' in code_extraction or '=' in code_extraction):
                    # Create message for explanation
                    explanation_message = [
                        HumanMessage(
                            content=[
                                {"type": "text", "text": f"Explain the following code in detail, including what it does, how it works, and any potential issues or improvements:\n\n{code_extraction}"},
                            ]
                        )
                    ]
                    
                    # Generate explanation
                    explanation_response = self.llm.generate([explanation_message])
                    explanation = explanation_response.generations[0][0].text.strip()
                    
                    # Add explanation to result
                    result["code_explanation"] = explanation
                
            except Exception as e:
                logger.warning(f"Error extracting code from image: {str(e)}")
                result["code"] = f"[Error extracting code: {str(e)}]"
        
        return result
    
    def resize_image(self, image_path: str, max_width: int = 800, max_height: int = 600) -> str:
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image_path: Path to the image file
            max_width: Maximum width of the resized image
            max_height: Maximum height of the resized image
            
        Returns:
            str: Path to the resized image
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Calculate new dimensions
            width, height = image.size
            aspect_ratio = width / height
            
            if width > max_width or height > max_height:
                if width / max_width > height / max_height:
                    # Width is the limiting factor
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    # Height is the limiting factor
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
                
                # Resize image
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Create output path
                filename, ext = os.path.splitext(image_path)
                output_path = f"{filename}_resized{ext}"
                
                # Save resized image
                resized_image.save(output_path)
                
                return output_path
            else:
                # Image is already within size limits
                return image_path
                
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image_path