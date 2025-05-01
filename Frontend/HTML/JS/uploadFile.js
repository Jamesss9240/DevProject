document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
      
      if (!file.type.startsWith('text/')) {
        alert('Invalid file type.');
        return;
      }
      
      const reader = new FileReader();
      reader.onload = function(e) {
        document.getElementById('codeInput').value = e.target.result;
      };
      reader.onerror = function() {
        alert('Error reading file.');
      };
      reader.readAsText(file);
    }
  });