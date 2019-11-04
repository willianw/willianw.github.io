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

        _fullSizeImages();

        _repositoryCover();

    };

    var _setImagePath = function() {
        var image_path = $('body').attr('name');
        $('.content p img').each(function(i, image) {
            var src = $(image).attr('src');
            $(image).attr('src', image_path+src)
        });
    }

    var _fullSizeImages = function() {
        $('img').click(function() {
            var src = $(this).attr('src');
            var body_img = $('body').css('background-image');
            $('div#full').css('background-image', 'url(' + src + ')');
            $('div#full').css('display', 'block');
        });
        $('div#full').click(function() {
            $(this).css('display', 'none');
        });
    };

    var _repositoryCover = function() {
        $('.gh-card').attr('data-image', $('body').css('background-image').slice(5, -2))
    };

    return {
        init: init
    };

})(window);

window.onload = load.init;
