document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadBtn = document.getElementById('upload-btn');
    const loading = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const personsList = document.getElementById('persons-list');
    
    // Update file name when file is selected
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
        } else {
            fileName.textContent = 'Choose a file';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files || !fileInput.files[0]) {
            alert('Please select an image file first.');
            return;
        }
        
        // Show loading spinner
        uploadBtn.disabled = true;
        loading.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Send request to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading spinner
            loading.classList.add('hidden');
            uploadBtn.disabled = false;
            
            // Display results
            if (data.image_path) {
                resultImage.src = data.image_path;
                resultsSection.classList.remove('hidden');
                
                // Clear previous results
                personsList.innerHTML = '';
                
                if (data.persons && data.persons.length > 0) {
                    // Add each person to the list
                    data.persons.forEach(person => {
                        const li = document.createElement('li');
                        li.innerHTML = `
                            <strong>${person.name}</strong>
                            <span class="confidence">${(person.confidence * 100).toFixed(2)}%</span>
                        `;
                        personsList.appendChild(li);
                    });
                } else {
                    // No faces detected
                    const li = document.createElement('li');
                    li.textContent = 'No faces detected in the image.';
                    personsList.appendChild(li);
                }
            } else {
                alert(data.error || 'An error occurred while processing the image.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loading.classList.add('hidden');
            uploadBtn.disabled = false;
            alert('An error occurred while processing the image. Please try again.');
        });
    });
}); 