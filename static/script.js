// ================================
// DOM Elements
// ================================
const newsInput = document.getElementById('newsInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const charCount = document.getElementById('charCount');
const resultsCard = document.getElementById('resultsCard');
const verdictContainer = document.getElementById('verdictContainer');
const verdictLabel = document.getElementById('verdictLabel');
const verdictDesc = document.getElementById('verdictDesc');
const probValue = document.getElementById('probValue');
const probFill = document.getElementById('probFill');
const loadingOverlay = document.getElementById('loadingOverlay');

// ================================
// Event Listeners
// ================================
newsInput.addEventListener('input', updateCharCount);
analyzeBtn.addEventListener('click', handleAnalyze);
clearBtn.addEventListener('click', clearInput);

// Ctrl+Enter to submit
newsInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        handleAnalyze();
    }
});

// ================================
// Character Count
// ================================
function updateCharCount() {
    const count = newsInput.value.length;
    charCount.textContent = `${count} character${count !== 1 ? 's' : ''}`;
}

// ================================
// Clear Input
// ================================
function clearInput() {
    newsInput.value = '';
    updateCharCount();
    newsInput.focus();
}

// ================================
// Main Analysis Handler
// ================================
async function handleAnalyze() {
    const text = newsInput.value.trim();

    // Validation
    if (!text) {
        showToast('Please enter some text to analyze', 'error');
        newsInput.focus();
        return;
    }

    if (text.length < 10) {
        showToast('Please enter at least 10 characters for accurate analysis', 'error');
        return;
    }

    // Show loading
    showLoading(true);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Analysis error:', error);
        showToast('Failed to analyze. Please ensure the server is running.', 'error');
    } finally {
        showLoading(false);
    }
}

// ================================
// Display Results
// ================================
function displayResults(result) {
    const { fake_probability, label } = result;
    const isFake = label === 'Fake';
    const percentage = (fake_probability * 100).toFixed(1);

    // Show results card
    resultsCard.classList.add('visible');

    // Update verdict
    verdictContainer.classList.remove('real', 'fake');
    verdictContainer.classList.add(isFake ? 'fake' : 'real');

    verdictLabel.textContent = isFake ? 'Likely Fake News' : 'Likely Authentic';
    verdictDesc.textContent = isFake 
        ? 'Our AI models indicate this content may contain misinformation'
        : 'Our AI models suggest this content appears credible';

    // Update probability
    probValue.textContent = `${percentage}%`;

    // Animate probability bar
    setTimeout(() => {
        probFill.style.width = `${percentage}%`;
    }, 100);

    // Scroll to results
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 200);
}

// ================================
// Loading State
// ================================
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('visible');
        analyzeBtn.disabled = true;
    } else {
        loadingOverlay.classList.remove('visible');
        analyzeBtn.disabled = false;
    }
}

// ================================
// Toast Notifications
// ================================
function showToast(message, type = 'info') {
    // Remove existing toasts
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    // Icon based on type
    const icon = type === 'error' 
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';

    toast.innerHTML = `${icon}<span>${message}</span>`;

    // Styles
    Object.assign(toast.style, {
        position: 'fixed',
        top: '24px',
        left: '50%',
        transform: 'translateX(-50%) translateY(-20px)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '16px 24px',
        background: type === 'error' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(59, 130, 246, 0.15)',
        border: `1px solid ${type === 'error' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`,
        borderRadius: '12px',
        backdropFilter: 'blur(20px)',
        color: type === 'error' ? '#f87171' : '#60a5fa',
        fontSize: '0.9375rem',
        fontWeight: '500',
        fontFamily: 'inherit',
        zIndex: '2000',
        opacity: '0',
        transition: 'all 0.3s ease',
    });

    // Icon styles
    const svgStyle = toast.querySelector('svg');
    if (svgStyle) {
        svgStyle.style.width = '20px';
        svgStyle.style.height = '20px';
        svgStyle.style.flexShrink = '0';
    }

    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
    });

    // Remove after delay
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-50%) translateY(-20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ================================
// Initialize
// ================================
document.addEventListener('DOMContentLoaded', () => {
    updateCharCount();
    newsInput.focus();
});
