/**
 * Mobile Navbar JavaScript
 * Handles mobile menu functionality
 */

// Mobile menu toggle function
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

// Close mobile menu when clicking outside
document.addEventListener('click', function(event) {
    const navLinks = document.getElementById('navLinks');
    const mobileMenuIcon = document.querySelector('.mobile-menu-icon');
    
    if (navLinks && mobileMenuIcon) {
        if (!navLinks.contains(event.target) && !mobileMenuIcon.contains(event.target)) {
            navLinks.classList.remove('active');
            const icon = mobileMenuIcon.querySelector('i');
            if (icon) {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        }
    }
});

// Close mobile menu when clicking on a link
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            const mobileMenu = document.getElementById('navLinks');
            const mobileMenuIcon = document.querySelector('.mobile-menu-icon i');
            
            if (mobileMenu && mobileMenu.classList.contains('active')) {
                mobileMenu.classList.remove('active');
                if (mobileMenuIcon) {
                    mobileMenuIcon.classList.remove('fa-times');
                    mobileMenuIcon.classList.add('fa-bars');
                }
            }
        });
    });
});

// Handle dropdown functionality on mobile
document.addEventListener('DOMContentLoaded', function() {
    const dropdowns = document.querySelectorAll('.dropdown');
    
    dropdowns.forEach(dropdown => {
        const dropbtn = dropdown.querySelector('.dropbtn');
        const dropdownContent = dropdown.querySelector('.dropdown-content');
        
        if (dropbtn && dropdownContent) {
            dropbtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                // Close other dropdowns
                dropdowns.forEach(otherDropdown => {
                    if (otherDropdown !== dropdown) {
                        const otherContent = otherDropdown.querySelector('.dropdown-content');
                        if (otherContent) {
                            otherContent.style.display = 'none';
                        }
                    }
                });
                
                // Toggle current dropdown
                if (dropdownContent.style.display === 'block') {
                    dropdownContent.style.display = 'none';
                } else {
                    dropdownContent.style.display = 'block';
                }
            });
        }
    });
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', function(event) {
        dropdowns.forEach(dropdown => {
            const dropdownContent = dropdown.querySelector('.dropdown-content');
            if (dropdownContent && !dropdown.contains(event.target)) {
                dropdownContent.style.display = 'none';
            }
        });
    });
});