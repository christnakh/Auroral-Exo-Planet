/**
 * 🔭 Exoplanet ML Analysis System - Main JavaScript
 * Ultra-modern interactive features and animations
 */

// ===== GLOBAL VARIABLES =====
let isAnimating = false;
let currentTheme = 'dark';

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Exoplanet ML Analysis System - Initialized');
    
    // Initialize all components
    initializeAnimations();
    initializeTooltips();
    initializeModals();
    initializeForms();
    initializeCharts();
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Add loading states
    addLoadingStates();
    
    // Remove theme switching (disabled)
    // initializeTheme();
    
    // Initialize mobile menu
    initializeMobileMenu();
});

// ===== MOBILE MENU =====
function initializeMobileMenu() {
    const mobileMenuIcon = document.querySelector('.mobile-menu-icon');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuIcon && navLinks) {
        mobileMenuIcon.addEventListener('click', function() {
            navLinks.classList.toggle('active');
            const icon = this.querySelector('i');
            if (icon.classList.contains('fa-bars')) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-times');
            } else {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        });
    }
}

// Global function for mobile menu toggle
function toggleMobileMenu() {
    const navLinks = document.getElementById('navLinks');
    const mobileMenuIcon = document.querySelector('.mobile-menu-icon i');
    
    if (navLinks) {
        navLinks.classList.toggle('active');
    }
    
    if (mobileMenuIcon) {
        if (mobileMenuIcon.classList.contains('fa-bars')) {
            mobileMenuIcon.classList.remove('fa-bars');
            mobileMenuIcon.classList.add('fa-times');
        } else {
            mobileMenuIcon.classList.remove('fa-times');
            mobileMenuIcon.classList.add('fa-bars');
        }
    }
}

// ===== ANIMATIONS =====
function initializeAnimations() {
    // Keep UI simple: no JS-driven entrance or hover animations
    const cards = document.querySelectorAll('.card');
    cards.forEach((card) => {
        card.style.opacity = '';
        card.style.transform = '';
        card.style.transition = '';
    });
}

// ===== TOOLTIPS =====
function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// ===== MODALS =====
function initializeModals() {
    // Add custom modal animations
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.addEventListener('show.bs.modal', function() {
            this.style.opacity = '0';
            this.style.transform = 'scale(0.8)';
            
            setTimeout(() => {
                this.style.transition = 'all 0.3s ease-out';
                this.style.opacity = '1';
                this.style.transform = 'scale(1)';
            }, 10);
        });
    });
}

// ===== FORMS =====
function initializeForms() {
    // Add form validation feedback
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            // Add loading state
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                submitBtn.disabled = true;
                
                // Re-enable after 3 seconds (for demo)
                setTimeout(() => {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }, 3000);
            }
        });
    });
    
    // Add input animations
    const inputs = document.querySelectorAll('.form-control');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.style.borderColor = '#00d4ff';
            this.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.1)';
        });
        
        input.addEventListener('blur', function() {
            this.style.borderColor = '';
            this.style.boxShadow = '';
        });
    });
}

// ===== CHARTS =====
function initializeCharts() {
    // Check if Chart.js is loaded and not disabled
    if (typeof Chart === 'undefined' || Chart === null) {
        console.warn('Chart.js not available, skipping chart initialization');
        return;
    }
    
    try {
        // Chart.js configuration
        Chart.defaults.color = '#ffffff';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
        Chart.defaults.backgroundColor = 'rgba(0, 212, 255, 0.1)';
        
        // Add chart animations
        Chart.defaults.animation = {
            duration: 2000,
            easing: 'easeInOutQuart'
        };
    } catch (error) {
        console.warn('Error initializing charts:', error);
    }
}

// ===== SMOOTH SCROLLING =====
function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ===== LOADING STATES =====
function addLoadingStates() {
    // Add loading animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('click', function() {
            if (!this.classList.contains('loading')) {
                this.classList.add('loading');
                setTimeout(() => {
                    this.classList.remove('loading');
                }, 1500);
            }
        });
    });
}

// ===== THEME SWITCHING =====
function initializeTheme() {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        currentTheme = savedTheme;
        applyTheme(currentTheme);
    }
    
    // Add theme toggle button if it doesn't exist
    if (!document.querySelector('.theme-toggle')) {
        const themeToggle = document.createElement('button');
        themeToggle.className = 'btn btn-outline-primary theme-toggle';
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        themeToggle.addEventListener('click', toggleTheme);
        
        // Add to navbar (fix selector to match markup)
        const navbar = document.querySelector('.navbar .navbar-container');
        if (navbar) {
            navbar.appendChild(themeToggle);
        }
    }
}

function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(currentTheme);
    localStorage.setItem('theme', currentTheme);
}

function applyTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.innerHTML = theme === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }
}

// ===== UTILITY FUNCTIONS =====
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'} me-2"></i>
        ${message}
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ===== API HELPERS =====
async function makeAPICall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showNotification('API call failed: ' + error.message, 'error');
        throw error;
    }
}

// ===== FORM HELPERS =====
function serializeForm(form) {
    const formData = new FormData(form);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    return data;
}

function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// ===== CHART HELPERS =====
function createChart(canvas, config) {
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        responsive: true,
        maintainAspectRatio: false,
        ...config
    });
}

// ===== EXPORT FUNCTIONS =====
window.ExoplanetML = {
    showNotification,
    makeAPICall,
    serializeForm,
    validateForm,
    createChart,
    debounce,
    throttle
};

// ===== ERROR HANDLING =====
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    showNotification('An error occurred. Please refresh the page.', 'error');
});

// ===== PERFORMANCE MONITORING =====
window.addEventListener('load', function() {
    const loadTime = performance.now();
    console.log(`🚀 Page loaded in ${loadTime.toFixed(2)}ms`);
    
    // Show performance notification for slow loads
    if (loadTime > 3000) {
        showNotification('Page loaded slowly. Consider checking your connection.', 'warning');
    }
});

// ===== ACCESSIBILITY =====
document.addEventListener('keydown', function(e) {
    // Add keyboard navigation
    if (e.key === 'Tab') {
        document.body.classList.add('keyboard-navigation');
    }
});

document.addEventListener('mousedown', function() {
    document.body.classList.remove('keyboard-navigation');
});

// ===== END OF MAIN.JS =====