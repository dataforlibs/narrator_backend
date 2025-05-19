

// script.js - Use ES modules to import Transformers.js correctly
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.12.1/+esm';

// Disable Transformers.js warning about progress tracking
window.process = {
    env: {
        TRANSFORMERS_JS_DISABLE_PROGRESS: "true"
    }
};

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('input-text');
    const analyzeButton = document.getElementById('analyze-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');
    const resultsContainer = document.getElementById('results-container');
    
    // Initialize the pipeline lazily (when button is first clicked)
    let pipelinePromise = null;
    
    // Define emotion colors for visualization
    const emotionColors = {
        'sadness': '#6495ED',
        'joy': '#FFD700',
        'love': '#FF69B4',
        'anger': '#FF6347',
        'fear': '#9370DB',
        'surprise': '#FF8C00'
    };
    
    // Initialize the pipeline when needed
    function getPipeline() {
        if (!pipelinePromise) {
            // Display loading indicator when we first load the model
            loadingIndicator.textContent = "Loading model... This may take a minute on first load";
            loadingIndicator.classList.remove('hidden');
            
            // Import the pipeline from Transformers.js
            pipelinePromise = pipeline('text-classification', 'j-hartmann/emotion-english-distilroberta-base')
                .then(pipeline => {
                    loadingIndicator.classList.add('hidden');
                    return pipeline;
                })
                .catch(error => {
                    console.error("Error loading model:", error);
                    loadingIndicator.classList.add('hidden');
                    errorMessage.textContent = "Error loading model. Please check console for details.";
                    errorMessage.classList.remove('hidden');
                });
        }
        return pipelinePromise;
    }
    
    // Function to analyze text and display results
    async function analyzeEmotion() {
        const text = inputText.value.trim();
        
        if (!text) {
            errorMessage.textContent = "Please enter some text to analyze.";
            errorMessage.classList.remove('hidden');
            return;
        }
        
        // Clear previous results and show loading
        resultsContainer.innerHTML = '';
        errorMessage.classList.add('hidden');
        loadingIndicator.textContent = "Analyzing...";
        loadingIndicator.classList.remove('hidden');
        
        try {
            const pipeline = await getPipeline();
            const results = await pipeline(text, { topk: 6 });
            
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
            
            // Display results
            results.forEach(result => {
                const { label, score } = result;
                const percentage = (score * 100).toFixed(1);
                
                const emotionBar = document.createElement('div');
                emotionBar.className = 'emotion-bar';
                
                const emotionLabel = document.createElement('div');
                emotionLabel.className = 'emotion-label';
                emotionLabel.textContent = label;
                
                const progressContainer = document.createElement('div');
                progressContainer.className = 'progress-container';
                
                const progressBar = document.createElement('div');
                progressBar.className = `progress-bar ${label}`;
                progressBar.style.width = `${percentage}%`;
                progressBar.style.backgroundColor = emotionColors[label] || '#ccc';
                
                const percentageElement = document.createElement('div');
                percentageElement.className = 'percentage';
                percentageElement.textContent = `${percentage}%`;
                
                progressContainer.appendChild(progressBar);
                emotionBar.appendChild(emotionLabel);
                emotionBar.appendChild(progressContainer);
                emotionBar.appendChild(percentageElement);
                
                resultsContainer.appendChild(emotionBar);
            });
        } catch (error) {
            console.error("Error analyzing text:", error);
            loadingIndicator.classList.add('hidden');
            errorMessage.textContent = "Error analyzing text. Please try again.";
            errorMessage.classList.remove('hidden');
        }
    }
    
    // Add click event to the analyze button
    analyzeButton.addEventListener('click', analyzeEmotion);
    
    // Allow pressing Enter in the textarea to trigger analysis
    inputText.addEventListener('keypress', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            analyzeEmotion();
        }
    });
});
