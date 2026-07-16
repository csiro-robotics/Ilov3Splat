window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


function initVideoCompare(rootId, baseVideoId, overlayVideoId, overlayId, sliderId) {
  var root = document.getElementById(rootId);
  var baseVideo = document.getElementById(baseVideoId);
  var overlayVideo = document.getElementById(overlayVideoId);
  var overlay = document.getElementById(overlayId);
  var slider = document.getElementById(sliderId);

  if (!root || !baseVideo || !overlayVideo || !overlay || !slider) {
    return;
  }

  var dragging = false;

  function syncOverlaySize() {
    overlayVideo.style.width = root.clientWidth + 'px';
    overlayVideo.style.height = root.clientHeight + 'px';
  }

  function setPosition(clientX) {
    var rect = root.getBoundingClientRect();
    var ratio = (clientX - rect.left) / rect.width;
    ratio = Math.max(0.02, Math.min(0.98, ratio));
    var percent = (ratio * 100).toFixed(2) + '%';
    overlay.style.width = percent;
    slider.style.left = percent;
    slider.setAttribute('aria-valuenow', Math.round(ratio * 100));
  }

  function syncPlayback() {
    if (Math.abs(baseVideo.currentTime - overlayVideo.currentTime) > 0.08) {
      overlayVideo.currentTime = baseVideo.currentTime;
    }
  }

  function playBoth() {
    var playBase = baseVideo.play();
    var playOverlay = overlayVideo.play();
    if (playBase && playBase.catch) {
      playBase.catch(function() {});
    }
    if (playOverlay && playOverlay.catch) {
      playOverlay.catch(function() {});
    }
  }

  function onPointerDown(event) {
    dragging = true;
    root.setPointerCapture(event.pointerId);
    setPosition(event.clientX);
    event.preventDefault();
  }

  function onPointerMove(event) {
    if (!dragging) {
      return;
    }
    setPosition(event.clientX);
  }

  function onPointerUp(event) {
    dragging = false;
    if (root.hasPointerCapture(event.pointerId)) {
      root.releasePointerCapture(event.pointerId);
    }
  }

  root.addEventListener('pointerdown', onPointerDown);
  root.addEventListener('pointermove', onPointerMove);
  root.addEventListener('pointerup', onPointerUp);
  root.addEventListener('pointercancel', onPointerUp);

  slider.addEventListener('keydown', function(event) {
    var current = parseInt(slider.getAttribute('aria-valuenow') || '50', 10);
    if (event.key === 'ArrowLeft') {
      setPosition(root.getBoundingClientRect().left + ((current - 3) / 100) * root.clientWidth);
      event.preventDefault();
    } else if (event.key === 'ArrowRight') {
      setPosition(root.getBoundingClientRect().left + ((current + 3) / 100) * root.clientWidth);
      event.preventDefault();
    }
  });

  baseVideo.addEventListener('loadedmetadata', syncOverlaySize);
  overlayVideo.addEventListener('loadedmetadata', syncOverlaySize);
  window.addEventListener('resize', syncOverlaySize);

  baseVideo.addEventListener('play', function() {
    if (overlayVideo.paused) {
      overlayVideo.play().catch(function() {});
    }
  });
  baseVideo.addEventListener('pause', function() {
    if (!overlayVideo.paused) {
      overlayVideo.pause();
    }
  });
  baseVideo.addEventListener('timeupdate', syncPlayback);
  baseVideo.addEventListener('seeked', function() {
    overlayVideo.currentTime = baseVideo.currentTime;
  });
  baseVideo.addEventListener('ended', function() {
    baseVideo.currentTime = 0;
    overlayVideo.currentTime = 0;
    playBoth();
  });

  syncOverlaySize();
  setPosition(root.getBoundingClientRect().left + root.clientWidth * 0.5);
  playBoth();
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    // Feature-field comparison: left = RGB color (overlay), right = PCA features (base)
    initVideoCompare(
      'feature-compare',
      'feature-video-instance',
      'feature-video-rgb',
      'feature-compare-overlay',
      'feature-compare-slider'
    );

})
