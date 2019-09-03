'use strict';

var load = (function (window) {

    var init = function () {

        var pattern = Trianglify({
            width: window.innerWidth,
            height: window.innerHeight,
            cell_size: 90,
            variance: 1,
            stroke_width: 1,
            x_colors: 'random',
            y_colors: 'random'
        }).svg(); // Render as SVG.

        _setImagePath();

    };

    var _setImagePath = function() {
        var image_path = $('body').attr('name');
        $('.content img').each(function(i, image) {
            var src = $(image).attr('src');
            $(image).attr('src', image_path+src)
        });
    }

    // Expose methods.
    return {
        init: init
    };

})(window);

// Kickstart

window.onload = load.init;
