# Feature Roadmap

## High Priority
- [x] Take multiple X-ray images as input and display the prediction for each image (maximum 10 images).
- [x] For each image, display the predicted class and an understandable confidence score.
- [x] Improve error handling for missing or corrupted model files.
- [x] Add clear messages for unsupported and invalid image uploads.
- [x] Add image validation for file size, dimensions, and format.
- [x] Support grayscale images safely.
- [x] Add confidence visualization with prediction probability charts.
- [x] Show a warning for uncertain predictions.
- [x] Display model information and supported classes.
- [x] Display the device used for inference.
- [x] Improve the user interface with sidebar instructions and sample input guidance.

## Medium Priority

- [x] Add an ensemble classification pipeline using ResNet18, SVM, and KNN.
- [x] Combine the three model scores to produce the final classification decision.
- [x] Define and validate the score-normalization and model-weighting strategy.
- [x] Display the individual model scores and the combined ensemble result.
- [ ] Add batch prediction for multiple X-ray uploads.
- [ ] Display batch results in a table.
- [ ] Add CSV download for batch predictions.
- [ ] Add Grad-CAM visualization to highlight influential image regions.
- [ ] Add session-only prediction history.
- [ ] Avoid storing patient-identifiable information.
- [ ] Add configurable upload limits and prediction thresholds.
- [ ] Add environment-based model path support.
- [ ] Add automated tests for model loading and preprocessing.
- [ ] Add tests for valid and invalid image uploads.

## Production Readiness

- [ ] Pin tested versions of Streamlit, PyTorch, TorchVision, and Pillow.
- [ ] Do not save uploaded images.
- [ ] Remove image metadata where appropriate.
- [ ] Add a clear privacy notice.
- [ ] Add non-sensitive error and inference-time monitoring.
- [ ] Add authentication for private deployments.
- [ ] Document installation and deployment procedures.
- [ ] Document model limitations and responsible use.

## Medical Safety

- [ ] Keep the application clearly labeled as a research/demo tool.
- [ ] State that predictions are not medical diagnoses.
- [ ] State that results must not replace review by a qualified clinician.
- [ ] Validate the model on an independent dataset before real-world use.
- [ ] Obtain applicable clinical, privacy, security, and regulatory approvals.
