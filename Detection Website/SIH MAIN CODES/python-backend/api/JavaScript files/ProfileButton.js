   // JavaScript to toggle the sliding profile bar
   const profileBar = document.getElementById("profileBar");
   const toggleProfile = document.getElementById("toggleProfile");
   const closeProfile = document.getElementById("closeProfile");

   let isProfileOpen = false;

   toggleProfile.addEventListener("click", function() {
       isProfileOpen = !isProfileOpen;
       const rightValue = isProfileOpen ? 0 : -250; // Show/hide the profile bar
       profileBar.style.right = rightValue + "px";
   });

   closeProfile.addEventListener("click", function() {
       isProfileOpen = false;
       profileBar.style.right = -250 + "px"; // Close the profile bar
   });