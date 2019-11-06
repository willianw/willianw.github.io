'use strict';

/**
 * Demo.
 */
var demo = (function (window) {

    /**
     * Enum of CSS selectors.
     */
    var SELECTORS = {
        pattern: '.pattern',
        card: '.card',
        cardImage: '.card__image',
        cardClose: '.card__btn-close'
    };

    /**
     * Enum of CSS classes.
     */
    var CLASSES = {
        patternHidden: 'pattern--hidden',
        polygon: 'polygon',
        polygonHidden: 'polygon--hidden'
    };

    var ATTRIBUTES = {
        index: 'data-index',
        id: 'data-id'
    };

    /**
     * Map of svg paths and points.
     */
    var polygonMap = {
        paths: null,
        points: null
    };

    /**
     * Container of Card instances.
     */
    var layout = {};

    /**
     * Initialise demo.
     */
    var init = function () {

        // For options see: https://github.com/qrohlf/Trianglify
        var pattern = Trianglify({
            width: window.innerWidth,
            height: window.innerHeight,
            cell_size: 90,
            variance: 1,
            stroke_width: 1,
            x_colors: 'random',
            y_colors: 'random'
        }).svg(); // Render as SVG.

        _mapPolygons(pattern);

        _toogleCV();

        _setImagePath();

        _selectCategory();

        _paintThemeColor();

        _fullSizeImages();
    };

    /**
     * Store path elements, map coordinates and sizes.
     * @param {Element} pattern The SVG Element generated with Trianglify.
     * @private
     */
    var _mapPolygons = function (pattern) {

        // Append SVG to pattern container.
        $(SELECTORS.pattern).append(pattern);

        // Convert nodelist to array,
        // Used `.childNodes` because IE doesn't support `.children` on SVG.
        polygonMap.paths = [].slice.call(pattern.childNodes);

        polygonMap.points = [];

        polygonMap.paths.forEach(function (polygon) {

            // Hide polygons by adding CSS classes to each svg path (used attrs because of IE).
            $(polygon).attr('class', CLASSES.polygon);

            var rect = polygon.getBoundingClientRect();

            var point = {
                x: rect.left + rect.width / 2,
                y: rect.top + rect.height / 2
            };

            polygonMap.points.push(point);
        });

        // All polygons are hidden now, display the pattern container.
        $(SELECTORS.pattern).removeClass(CLASSES.patternHidden);
    };

    var _toogleCV = function(){
        $('a#button-cv').on('click', function() {
            var CV = $('embed#cv');
            var hiddenCV = CV.css('display') == 'none';
            var blogContent = $('div.content');
            var body = $('body');
            if (hiddenCV) {
                CV.css('display', 'block');
                body.css('overflow-y', 'hidden');
                blogContent.css('display', 'none');
                $('a#button-cv i, a#button-cv span').addClass('cv-active');
            } else {
                CV.css('display', 'none');
                body.css('overflow-y', 'auto');
                blogContent.css('display', 'block');
                $('a#button-cv i, a#button-cv span').removeClass('cv-active');
            };
        });
    };

    var _setImagePath = function() {
        var image_path = $('body').name;
        $('.content p img').each(function(image, i) {
            var src = image.attr('src');
            image.attr('src', image_path+src);
        });
    };

    var _selectCategory = function() {
        $('.categories a, .middle a').on('click', function(){
            var category = this.name;
            $('.content .card').each(function(post, i) {
                if ($(post).attr('category').includes(category)) {
                    $(post).css('display', 'block');
                } else {
                    $(post).css('display', 'none');
                }
            });
        })

    };

    var _paintThemeColor = function() {
        var getColor = function(color) {
            return color
                .slice(4, -1)
                .split(', ')
                .map(function(i){return i / 255;});
        };
        var colors = Array.from($('div.pattern path'))
            .map(function(i) {return $(i).css('stroke')});
        var maxmins = colors.map(function(color, i) {
            const rgb = getColor(color);
            const max = Math.max(...rgb);
            const min = Math.min(...rgb);
            return [min, max]
        });
        var ls = maxmins.map(function(mm, i) {
            const L = (mm[0] + mm[1])/2;
            return (L > 0.80)? 0: L;
        });
        var sats = maxmins.map(function(mm, i) {
            const L = (mm[1] + mm[0]) / 2;
            return (L === 0 || L === 1)
                ? 0
                : (mm[1] - mm[0])/(1 - Math.abs(2 * L - 1));
        });
        var bst = sats.indexOf(Math.max(...sats));
        var lit = ls.indexOf(Math.max(...ls));
        
        document.documentElement.style.setProperty('--theme-mid', colors[colors.length/2>>0]);
        document.documentElement.style.setProperty('--theme-sat', colors[bst]);
        document.documentElement.style.setProperty('--theme-lit', colors[lit]);
    };

    var _fullSizeImages = function() {
        $('img').on('click', function() {
            var src = this.attr('src');
            $('div#full')
                .css('background-image', 'url(' + src + ') no-repeat center')
                .css('display', 'block');
            $('body').css('display', 'None');
        });
    };

    // Expose methods.
    return {
        init: init
    };

})(window);

// Kickstart Demo.
window.onload = demo.init;
