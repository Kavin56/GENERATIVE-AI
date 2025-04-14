// Track cursor movement and apply dynamic gradient
const body = document.body;

// Track cursor movement and apply dynamic gradient
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

// Handle Google Sign-In Response
function handleCredentialResponse(response) {
    try {
        const data = jwt_decode(response.credential);  // Decode JWT for user info
        const email = data.email;

        // If login is successful, alert the user and redirect to index.html
        alert(`Welcome, ${data.name}!`);

        // Redirect to index.html after successful login
        window.location.href = "index.html";
    } catch (error) {
        console.error('Error decoding JWT or handling login', error);
        alert("❌ Login failed. Please try again.");
    }
}

// Initialize Google Sign-In Button
function initializeGoogleSignIn() {
    window.google.accounts.id.initialize({
        client_id: 'YOUR_GOOGLE_CLIENT_ID',  // Replace with your actual Google Client ID
        callback: handleCredentialResponse,
        auto_select: false, // Allow users to choose their account
        prompt: 'select_account', // Prompt for selecting an account
        scope: 'openid profile email',  // Ensure the right scopes are included
    });

    // Automatically render the Google Sign-In button
    window.google.accounts.id.renderButton(
        document.getElementById('g_id_signin'),
        {
            theme: 'outline',
            size: 'large',
            shape: 'pill',
            text: 'signin_with'
        }
    );
}

// Load Google Identity Service API
function loadGoogleAPI() {
    const script = document.createElement('script');
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    document.body.appendChild(script);
    
    script.onload = function() {
        // Once the script is loaded, initialize Google Sign-In
        initializeGoogleSignIn();
    };
}

// Load the Google API and initialize Google Sign-In
loadGoogleAPI();
