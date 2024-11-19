import torch
from PIL import Image, ImageDraw, ImageFont
from OpenGroundingDino.tools.inference_on_a_image import load_model, get_grounding_output, load_image

class GroundingDINOInference:
    # Initialize the Grounding DINO inference class.
    def __init__(self, model_config_path, model_checkpoint_path, device = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_config_path, model_checkpoint_path)

    # Load the Grounding DINO model.
    def _load_model(self, config_path, checkpoint_path):

        return load_model(config_path, checkpoint_path, cpu_only=(self.device == "cpu"))

    # Run inference on an image with a text prompt.
    def predict(self, image_path, text_prompt, box_threshold = 0.3, text_threshold = 0.25, iou_threshold = 0.5):

        # Preprocess and load the image
        image_pil, image_tensor = load_image(image_path)

        # Run the GroundingDINO model
        boxes_filt, pred_phrases = get_grounding_output(
            self.model,
            image_tensor,
            text_prompt,
            box_threshold = box_threshold,
            text_threshold = text_threshold,
            cpu_only = (self.device == "cpu")
        )

        # Extract confidence scores from the predicted phrases.
        scores = torch.tensor([float(phrase.split('(')[-1][:-1]) for phrase in pred_phrases])

        # Apply NMS to remove overlapping boxes
        keep_indices = self._apply_nms(boxes_filt, scores, iou_threshold)

        # Filter boxes and phrases based on NMS
        boxes_filt = boxes_filt[keep_indices]
        pred_phrases = [pred_phrases[i] for i in keep_indices]

        return image_pil, boxes_filt, pred_phrases

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes.
    def _apply_nms(self, boxes, scores, iou_threshold):

        # Convert cxcywh to xyxy for IOU calculation
        boxes_xyxy = self._convert_cxcywh_to_xyxy(boxes)

        # Apply PyTorch's built-in NMS
        keep = torch.ops.torchvision.nms(boxes_xyxy, scores, iou_threshold)
        return keep

    # Convert bounding boxes from center-x, center-y, width, height (cxcywh) to top-left-x, top-left-y, bottom-right-x, bottom-right-y (xyxy) format.
    def _convert_cxcywh_to_xyxy(self, boxes):

        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)  # x_min
        boxes_xyxy[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)  # y_min
        boxes_xyxy[:, 2] = boxes[:, 0] + (boxes[:, 2] / 2)  # x_max
        boxes_xyxy[:, 3] = boxes[:, 1] + (boxes[:, 3] / 2)  # y_max
        return boxes_xyxy

    # Visualize predictions on the image by drawing bounding boxes and labels.
    def visualize_predictions(self, image_pil, boxes, labels, save_path = None):
        
        W, H = image_pil.size
        draw = ImageDraw.Draw(image_pil)

        for box, label in zip(boxes, labels):
            # Scale and convert boxes from cxcywh to xyxy
            box = box * torch.tensor([W, H, W, H])
            box[:2] -= box[2:] / 2  # Convert center to top-left
            box[2:] += box[:2]  # Convert width/height to bottom-right
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            # Draw the bounding box
            draw.rectangle([x0, y0, x1, y1], outline = "red", width = 3)

            # Draw the label
            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), label, font)
            else:
                w, h = draw.textsize(label, font)
                bbox = (x0, y0, w + x0, y0 + h)
            draw.rectangle(bbox, fill = "red")
            draw.text((x0, y0), label, fill = "white", font = font)

        if save_path:
            image_pil.save(save_path)
        return image_pil