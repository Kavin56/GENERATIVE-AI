const imageUpload = document.getElementById('imageUpload');
const previewArea = document.getElementById('preview-area');
const uploadForm = document.getElementById('uploadForm');
const successContainer = document.getElementById('success-container');
const closeSuccessButton = document.getElementById('close-success');
const parallaxBg = document.querySelector('.parallax-bg');

// Array of aurora image URLs
const auroraImages = [
  'https://images.unsplash.com/photo-1527481785475-c84b751b680d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8YXVyb3JhfGVufDB8fDB8fHww&auto=format&fit=crop&w=500&q=60',
  'https://images.unsplash.com/photo-1530275311954-01bb043f4747?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fGF1cm9yYXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60',
  'https://images.unsplash.com/photo-1507974422426-a229695207f5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTl8fGF1cm9yYXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60',
   'https://images.unsplash.com/photo-1470254038294-9ca75230929b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjB8fGF1cm9yYXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60',
    'https://images.unsplash.com/photo-1485222279018-65310a1489e3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjR8fGF1cm9yYXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60'
];
let currentImageIndex = 0;
// Function to change the background image with animation
function changeBackgroundImage() {
  parallaxBg.classList.remove('active'); // Remove active class to start fading out
    setTimeout(() => {
         parallaxBg.style.backgroundImage = `url('${auroraImages[currentImageIndex]}')`; // Set next image
         parallaxBg.classList.add('active'); // Add active class to start fading in
        currentImageIndex = (currentImageIndex + 1) % auroraImages.length; // Increment index
    }, 1000);
}
// Initial background image
changeBackgroundImage();
// Set interval for background image change
setInterval(changeBackgroundImage, 8000);


// Track cursor movement and apply dynamic gradient
const body = document.body;
document.addEventListener("mousemove", (event) => {
    const { clientX: x, clientY: y } = event;
    const width = window.innerWidth;
    const height = window.innerHeight;

    const red = Math.round((x / width) * 255);
    const blue = Math.round((y / height) * 255);
    const green = 150;
    body.style.background = `radial-gradient(circle at ${x}px ${y}px,
        rgba(${red}, ${green}, ${blue}, 0.8),
        #000000 90%)`;
});

// Stop the gradient change after the cursor stops moving
let timeout;
document.addEventListener("mousemove", () => {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
        body.style.transition = "background 1s ease-out";
    }, 100);
});

// Image upload preview
imageUpload.addEventListener('change', function(event) {
    previewArea.innerHTML = '';
    const files = event.target.files;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                previewArea.appendChild(img);
            }
            reader.readAsDataURL(file);
        }
    }
});
uploadForm.addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    const fileInput = document.getElementById('imageUpload');
    if(fileInput.files.length > 0) { // Show success only if a file is selected
        successContainer.classList.remove('hidden');
       fireworks()// show fire works
    }
});

closeSuccessButton.addEventListener('click', function() {
    successContainer.classList.add('hidden');
});
function fireworks() {
    confetti({
        particleCount: 150,
        spread: 80,
        origin: { y: 0.6 },
        colors: ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'],
    });
}