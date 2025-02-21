
document.addEventListener('DOMContentLoaded', function() {
    const lightThemeButton = document.getElementById('light-theme');
    const darkThemeButton = document.getElementById('dark-theme');
    const body = document.body;
    const navbar = document.querySelector('.navbar');
    function setTheme(theme) {
        if (theme === 'dark') {
            $('body, .navbar').fadeOut(300, function() {
                document.documentElement.classList.add('dark-theme');
                body.classList.add('dark-theme');
                navbar.classList.remove('navbar-light', 'bg-light');
                navbar.classList.add('navbar-dark', 'bg-dark');
                localStorage.setItem('theme', 'dark');
                $(this).fadeIn(1000);
            });
        } else {
            $('body, .navbar').fadeOut(300, function() {
                document.documentElement.classList.remove('dark-theme');
                body.classList.remove('dark-theme');
                navbar.classList.remove('navbar-dark', 'bg-dark');
                navbar.classList.add('navbar-light', 'bg-light');
                localStorage.setItem('theme', 'light');
                $(this).fadeIn(1000); 
            });
        }
    }

    lightThemeButton.addEventListener('click', () => setTheme('light'));
    darkThemeButton.addEventListener('click', () => setTheme('dark'));

    // Load saved theme on page load/preferred theme based off users OS setting
    const savedTheme = localStorage.getItem('theme') || (window.matchMedia("(prefers-color-scheme: dark)").matches ? 'dark' : 'light');
    setTheme(savedTheme);
});