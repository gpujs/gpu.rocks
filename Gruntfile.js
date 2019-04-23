module.exports = function(grunt) {
  const sass = require('node-sass');
  require('load-grunt-tasks')(grunt);

  grunt.initConfig({
    pkg: grunt.file.readJSON("package.json"),

    watch: {
      options: {
        livereload: 35729
      },
      js: {
        files: ["src/js/*", "Gruntfile.js"],
        tasks: ["browserify:main"]
      },
      sass: {
        files: ["src/sass/*", "src/*", "Gruntfile.js"],
        tasks: ["sass:main"]
      },
      libs: {
        files: ["src/**/libs.*"],
        tasks: ["sass:libs", "browserify:libs", "uglify:libs"]
      }
    },

    sass: {
      options: {
        implementation: sass,
        sourceMap: true
      },
      main: {
        files: {
          'dist/css/index.css': 'src/sass/index.scss'
        }
      },
      libs: {
        files: {
          'dist/css/libs.min.css': 'src/sass/libs.scss'
        },
        options: {
          outputStyle: 'compressed'
        }
      },
      prod: {
        options: {
          outputStyle: 'compressed'
        },
        files: {
          'dist/css/index.min.css': 'src/sass/index.scss'
        }
      }
    },

    browserify: {
      main: {
        src: ["src/index.js"],
        dest: "dist/js/index.js"
      },
      prod: {
        src: ["src/index.js"],
        dest: "dist/js/brow.js"
      },
      libs: {
        src: ["src/js/libs.js"],
        dest: "dist/js/libs.js"
      }
    },

    uglify: {
      main: {
        src: ["dist/js/index.js"],
        dest: "dist/js/index.min.js"
      },
      prod: {
        src: ["dist/js/brow.js"],
        dest: "dist/js/index.js"
      },
      libs: {
        src: ["dist/js/libs.js"],
        dest: "dist/js/libs.min.js"
      }
    },

    browserSync: {
      dev: {
        options: {
          watchTask: true,
          server: "./"
        }
      }
    }
  });

  /* Default (development): Watch files and build on change. */
  grunt.registerTask("default", ["serve"]);
  grunt.registerTask("build", ["browserify:main", "browserify:libs", "uglify:main", "uglify:libs", "sass"]);
  grunt.registerTask("serve", ["browserify:main", "browserify:libs", "uglify:libs", "sass:libs", "sass:main", "browserSync", "watch"]);
  grunt.registerTask("compile", ["browserify:main", "browserify:libs", "uglify:libs", "sass:libs", "sass:main"]);
  grunt.registerTask("production", ["browserify:prod", "browserify:libs", "uglify:prod", "uglify:libs", "sass:prod", "sass:libs"]);
};