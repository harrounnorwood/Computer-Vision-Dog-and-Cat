"use strict";

(() => {
    const MAX_FILE_SIZE = 10 * 1024 * 1024;
    const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"];
    const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"];

    const uploadForm = document.getElementById("upload-form");
    const dropZone = document.getElementById("drop-zone");
    const imageInput = document.getElementById("image-input");
    const previewPanel = document.getElementById("preview-panel");
    const previewImage = document.getElementById("preview-image");
    const fileName = document.getElementById("file-name");
    const fileDetails = document.getElementById("file-details");
    const removeImageButton = document.getElementById("remove-image-button");
    const analyzeButton = document.getElementById("analyze-button");
    const analyzeButtonText = analyzeButton.querySelector("span:last-child");

    const errorMessage = document.getElementById("error-message");
    const errorText = document.getElementById("error-text");
    const loadingPanel = document.getElementById("loading-panel");
    const loadingMessage = document.getElementById("loading-message");

    const resultsSection = document.getElementById("results-section");
    const analyzeAnotherButton = document.getElementById(
        "analyze-another-button"
    );

    const predictionCard = document.getElementById("prediction-card");
    const resultBadge = document.getElementById("result-badge");
    const predictionValue = document.getElementById("prediction-value");
    const predictionSummary = document.getElementById("prediction-summary");
    const confidenceDisplay = document.getElementById("confidence-display");
    const confidenceValue = document.getElementById("confidence-value");

    const validationStatus = document.getElementById("validation-status");
    const similarityValue = document.getElementById("similarity-value");
    const similarityThreshold = document.getElementById(
        "similarity-threshold"
    );
    const similarityProgress = document.getElementById("similarity-progress");
    const similarityBar = document.getElementById("similarity-bar");
    const thresholdMarker = document.getElementById("threshold-marker");
    const nearestMajority = document.getElementById("nearest-majority");
    const neighborAgreement = document.getElementById("neighbor-agreement");
    const averageDistance = document.getElementById("average-distance");
    const distanceVariation = document.getElementById("distance-variation");

    const classifierStatus = document.getElementById("classifier-status");
    const classifierDescription = document.getElementById(
        "classifier-description"
    );
    const catProbability = document.getElementById("cat-probability");
    const dogProbability = document.getElementById("dog-probability");
    const catProbabilityBar = document.getElementById("cat-probability-bar");
    const dogProbabilityBar = document.getElementById("dog-probability-bar");

    const explanationSource = document.getElementById("explanation-source");
    const explanationText = document.getElementById("explanation-text");

    const loadingMessages = [
        "Preparing the image for MobileNetV2.",
        "Extracting visual features from the image.",
        "Checking similarity against Cat and Dog references.",
        "Determining whether the classifier should run.",
        "Preparing the result explanation.",
    ];

    let selectedFile = null;
    let previewUrl = null;
    let loadingTimer = null;
    let dragDepth = 0;
    let isAnalyzing = false;

    function getFileExtension(filename) {
        const parts = filename.toLowerCase().split(".");
        return parts.length > 1 ? parts.pop() : "";
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return `${bytes} B`;
        }

        if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(1)} KB`;
        }

        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }

    function toFiniteNumber(value, fallback = 0) {
        const numericValue = Number(value);
        return Number.isFinite(numericValue) ? numericValue : fallback;
    }

    function clampPercentage(value) {
        return Math.min(100, Math.max(0, toFiniteNumber(value)));
    }

    function formatPercentage(value) {
        return `${toFiniteNumber(value).toFixed(2)}%`;
    }

    function formatDecimal(value) {
        return toFiniteNumber(value).toFixed(4);
    }

    function showError(message) {
        errorText.textContent = message;
        errorMessage.hidden = false;
    }

    function hideError() {
        errorMessage.hidden = true;
        errorText.textContent = "";
    }

    function revokePreviewUrl() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    }

    function validateFile(file) {
        if (!file) {
            return "Please select an image first.";
        }

        const extension = getFileExtension(file.name);
        const hasAllowedExtension = ALLOWED_EXTENSIONS.includes(extension);
        const hasAllowedMimeType =
            !file.type || ALLOWED_MIME_TYPES.includes(file.type);

        if (!hasAllowedExtension || !hasAllowedMimeType) {
            return "Only valid JPG, JPEG, and PNG files are allowed.";
        }

        if (file.size === 0) {
            return "The selected image is empty.";
        }

        if (file.size > MAX_FILE_SIZE) {
            return "The image is too large. The maximum file size is 10 MB.";
        }

        return null;
    }

    function clearSelectedImage({ keepError = false } = {}) {
        selectedFile = null;
        imageInput.value = "";
        revokePreviewUrl();

        previewImage.removeAttribute("src");
        previewPanel.hidden = true;
        dropZone.hidden = false;
        analyzeButton.disabled = true;

        fileName.textContent = "Selected image";
        fileDetails.textContent = "Image details";

        if (!keepError) {
            hideError();
        }
    }

    function displaySelectedFile(file) {
        hideError();
        resultsSection.hidden = true;
        selectedFile = file;

        revokePreviewUrl();
        previewUrl = URL.createObjectURL(file);

        previewImage.onload = () => {
            const dimensions =
                `${previewImage.naturalWidth} × ` +
                `${previewImage.naturalHeight}px`;

            fileDetails.textContent =
                `${formatFileSize(file.size)} · ${dimensions}`;
        };

        previewImage.onerror = () => {
            clearSelectedImage({ keepError: true });
            showError(
                "The selected file could not be displayed as an image."
            );
        };

        previewImage.src = previewUrl;
        fileName.textContent = file.name;
        fileDetails.textContent =
            `${formatFileSize(file.size)} · Reading dimensions…`;

        dropZone.hidden = true;
        previewPanel.hidden = false;
        analyzeButton.disabled = false;
    }

    function handleSelectedFile(file) {
        if (isAnalyzing) {
            return;
        }

        const validationError = validateFile(file);

        if (validationError) {
            clearSelectedImage({ keepError: true });
            showError(validationError);
            return;
        }

        displaySelectedFile(file);
    }

    function startLoadingMessages() {
        let messageIndex = 0;
        loadingMessage.textContent = loadingMessages[messageIndex];

        loadingTimer = window.setInterval(() => {
            messageIndex =
                (messageIndex + 1) % loadingMessages.length;

            loadingMessage.textContent =
                loadingMessages[messageIndex];
        }, 1500);
    }

    function stopLoadingMessages() {
        if (loadingTimer !== null) {
            window.clearInterval(loadingTimer);
            loadingTimer = null;
        }
    }

    function setAnalyzingState(active) {
        isAnalyzing = active;

        uploadForm.setAttribute("aria-busy", String(active));
        imageInput.disabled = active;
        removeImageButton.disabled = active;
        analyzeButton.disabled = active || !selectedFile;
        analyzeButtonText.textContent =
            active ? "Analyzing…" : "Analyze image";
        loadingPanel.hidden = !active;

        if (active) {
            startLoadingMessages();
        } else {
            stopLoadingMessages();
        }
    }

    function setStatusBadge(element, text, stateClass) {
        element.textContent = text;

        element.classList.remove(
            "is-passed",
            "is-failed",
            "is-skipped"
        );

        element.classList.add(stateClass);
    }

    function setProgressWidth(element, value) {
        element.style.width =
            `${clampPercentage(value)}%`;
    }

    function updatePredictionSummary(prediction) {
        if (prediction === "Cat") {
            resultBadge.textContent = "Final prediction";

            predictionSummary.textContent =
                "The image passed validation and the classifier " +
                "identified it as a cat.";

            return;
        }

        if (prediction === "Dog") {
            resultBadge.textContent = "Final prediction";

            predictionSummary.textContent =
                "The image passed validation and the classifier " +
                "identified it as a dog.";

            return;
        }

        resultBadge.textContent =
            "Outside the accepted similarity range";

        predictionSummary.textContent =
            "The image was not similar enough to the stored Cat " +
            "and Dog references, so the classifier was not used.";
    }

    function displayClassifierResult(data, prediction) {
        const classifierRan =
            prediction === "Cat" || prediction === "Dog";

        confidenceDisplay.classList.toggle(
            "is-hidden",
            !classifierRan
        );

        if (!classifierRan) {
            confidenceValue.textContent = "—";

            setStatusBadge(
                classifierStatus,
                "Skipped",
                "is-skipped"
            );

            classifierDescription.textContent =
                "The classifier was skipped because the image did " +
                "not pass similarity validation.";

            catProbability.textContent = "Not run";
            dogProbability.textContent = "Not run";

            setProgressWidth(catProbabilityBar, 0);
            setProgressWidth(dogProbabilityBar, 0);

            return;
        }

        confidenceValue.textContent =
            formatPercentage(data.confidence);

        setStatusBadge(
            classifierStatus,
            "Completed",
            "is-passed"
        );

        classifierDescription.textContent =
            "The image passed the similarity gate, so the " +
            "Cat/Dog classifier was executed.";

        catProbability.textContent =
            formatPercentage(data.cat_probability);

        dogProbability.textContent =
            formatPercentage(data.dog_probability);

        setProgressWidth(
            catProbabilityBar,
            data.cat_probability
        );

        setProgressWidth(
            dogProbabilityBar,
            data.dog_probability
        );
    }

    function displaySimilarityResult(data) {
        const passed =
            String(data.validation).toLowerCase() === "passed";

        const similarity =
            toFiniteNumber(data.similarity);

        const threshold =
            toFiniteNumber(data.similarity_threshold);

        setStatusBadge(
            validationStatus,
            passed ? "Passed" : "Failed",
            passed ? "is-passed" : "is-failed"
        );

        similarityValue.textContent =
            formatPercentage(similarity);

        similarityThreshold.textContent =
            formatPercentage(threshold);

        nearestMajority.textContent =
            data.nearest_majority || "—";

        neighborAgreement.textContent =
            formatPercentage(data.neighbor_agreement);

        averageDistance.textContent =
            formatDecimal(data.average_distance);

        distanceVariation.textContent =
            formatDecimal(data.distance_std);

        similarityProgress.setAttribute(
            "aria-valuenow",
            String(clampPercentage(similarity))
        );

        similarityProgress.setAttribute(
            "aria-valuetext",
            `${formatPercentage(similarity)} similarity; ` +
            `${formatPercentage(threshold)} required`
        );

        similarityBar.classList.remove(
            "is-passed",
            "is-failed"
        );

        similarityBar.classList.add(
            passed ? "is-passed" : "is-failed"
        );

        setProgressWidth(similarityBar, similarity);

        thresholdMarker.style.left =
            `${clampPercentage(threshold)}%`;
    }

    function displayResults(data) {
        const rawPrediction =
            String(data.prediction || "Unknown").trim();

        const prediction =
            ["Cat", "Dog"].includes(rawPrediction)
                ? rawPrediction
                : "Unknown";

        const predictionClass =
            `is-${prediction.toLowerCase()}`;

        predictionCard.classList.remove(
            "is-cat",
            "is-dog",
            "is-unknown"
        );

        predictionCard.classList.add(predictionClass);
        predictionValue.textContent = prediction;

        updatePredictionSummary(prediction);
        displaySimilarityResult(data);
        displayClassifierResult(data, prediction);

        explanationSource.textContent =
            data.explanation_source === "Gemini"
                ? "Gemini explanation"
                : "Built-in explanation";

        explanationText.textContent =
            data.explanation ||
            "No explanation was returned.";

        resultsSection.hidden = false;

        window.requestAnimationFrame(() => {
            resultsSection.scrollIntoView({
                behavior: window.matchMedia(
                    "(prefers-reduced-motion: reduce)"
                ).matches
                    ? "auto"
                    : "smooth",
                block: "start",
            });
        });
    }

    async function readJsonResponse(response) {
        const contentType =
            response.headers.get("content-type") || "";

        if (!contentType.includes("application/json")) {
            if (response.status === 502 || response.status === 503) {
                throw new Error(
                    "The online model is waking up or has run out of memory. " +
                    "Please wait 30 seconds and try the image again."
                );
            }

            throw new Error(
                "The server returned an unexpected response. " +
                "Please check the Flask terminal."
            );
        }

        return response.json();
    }

    async function analyzeImage(event) {
        event.preventDefault();

        if (isAnalyzing) {
            return;
        }

        const validationError =
            validateFile(selectedFile);

        if (validationError) {
            showError(validationError);
            return;
        }

        hideError();
        resultsSection.hidden = true;
        setAnalyzingState(true);

        const formData = new FormData();

        formData.append(
            "image",
            selectedFile,
            selectedFile.name
        );

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            const data =
                await readJsonResponse(response);

            if (!response.ok || !data.success) {
                throw new Error(
                    data.error ||
                    "The image could not be analyzed."
                );
            }

            displayResults(data);
        } catch (error) {
            const message =
                error instanceof TypeError
                    ? "The Flask server could not be reached. " +
                      "Make sure it is running, then try again."
                    : error.message;

            showError(
                message ||
                "An unexpected error occurred."
            );
        } finally {
            setAnalyzingState(false);
        }
    }

    function resetInterface({
        scrollToUpload = false,
    } = {}) {
        setAnalyzingState(false);
        clearSelectedImage();

        resultsSection.hidden = true;

        predictionCard.classList.remove(
            "is-cat",
            "is-dog",
            "is-unknown"
        );

        similarityBar.style.width = "0%";
        catProbabilityBar.style.width = "0%";
        dogProbabilityBar.style.width = "0%";

        if (scrollToUpload) {
            uploadForm.scrollIntoView({
                behavior: window.matchMedia(
                    "(prefers-reduced-motion: reduce)"
                ).matches
                    ? "auto"
                    : "smooth",
                block: "center",
            });
        }
    }

    imageInput.addEventListener("change", () => {
        handleSelectedFile(imageInput.files[0]);
    });

    uploadForm.addEventListener(
        "submit",
        analyzeImage
    );

    removeImageButton.addEventListener(
        "click",
        () => {
            clearSelectedImage();
        }
    );

    analyzeAnotherButton.addEventListener(
        "click",
        () => {
            resetInterface({
                scrollToUpload: true,
            });
        }
    );

    dropZone.addEventListener(
        "dragenter",
        (event) => {
            event.preventDefault();
            dragDepth += 1;

            dropZone.classList.add(
                "is-dragging"
            );
        }
    );

    dropZone.addEventListener(
        "dragover",
        (event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect =
                "copy";
        }
    );

    dropZone.addEventListener(
        "dragleave",
        (event) => {
            event.preventDefault();

            dragDepth = Math.max(
                0,
                dragDepth - 1
            );

            if (dragDepth === 0) {
                dropZone.classList.remove(
                    "is-dragging"
                );
            }
        }
    );

    dropZone.addEventListener(
        "drop",
        (event) => {
            event.preventDefault();
            dragDepth = 0;

            dropZone.classList.remove(
                "is-dragging"
            );

            const droppedFiles =
                event.dataTransfer.files;

            if (droppedFiles.length > 1) {
                showError(
                    "Please upload only one image at a time."
                );
                return;
            }

            handleSelectedFile(
                droppedFiles[0]
            );
        }
    );

    dropZone.tabIndex = 0;
    dropZone.setAttribute("role", "button");

    dropZone.addEventListener(
        "keydown",
        (event) => {
            if (
                event.key === "Enter" ||
                event.key === " "
            ) {
                event.preventDefault();
                imageInput.click();
            }
        }
    );

    window.addEventListener(
        "beforeunload",
        revokePreviewUrl
    );
})();