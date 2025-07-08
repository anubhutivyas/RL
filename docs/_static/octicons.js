// Octicons JavaScript for NeMo RL Documentation

document.addEventListener('DOMContentLoaded', function() {
    // Enhance octicon rendering
    const octicons = document.querySelectorAll('.octicon');
    
    octicons.forEach(function(octicon) {
        // Ensure proper display
        octicon.style.display = 'inline-block';
        octicon.style.verticalAlign = 'text-bottom';
        
        // Add hover effects for interactive elements
        if (octicon.closest('.grid-item-card')) {
            octicon.style.transition = 'transform 0.2s ease';
            
            octicon.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.1)';
            });
            
            octicon.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        }
    });
    
    // Ensure octicons are loaded from CDN
    if (typeof octicons !== 'undefined') {
        console.log('Octicons loaded successfully');
    }
}); 