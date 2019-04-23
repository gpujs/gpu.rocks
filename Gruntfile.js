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
        files: ["src/sass/*.scss"],
        tasks: ["sass:main"]
      },
      libs: {
        files: ["src/**/libs.*"],
        tasks: ["sass:libs", "browserify:libs"]
      }
    },

    autoprefixer: {
      main: {
        'dist/css/libs.css': 'dist/css/libs.css',
        'dist/css/index.css': 'dist/css/index.css'
      }
    },

    sass: {
      options: {
        implementation: sass,
      },
      main: {
        files: {
          'dist/css/index.css': 'src/sass/index.scss'
        }
      },
      libs: {
        files: {
          'dist/css/libs.css': 'src/sass/libs.scss'
        }
      }
    },

    css_import: {
      main: {
        files: {
          'dist/css/index.css': ['dist/css/index.css']
        }
      },
      libs: {
        files: {
          'dist/css/libs.css': ['dist/css/libs.css']
        }
      }
    },

    cssmin: {
      libs: {
        files: {
          'dist/css/libs.css': ['dist/css/libs.css']
        }
      },
      main: {
        files: {
          'dist/css/index.css': ['dist/css/index.css']
        }
      }
    },

    browserify: {
      main: {
        src: ["src/index.js"],
        dest: "dist/js/index.js"
      },
      libs: {
        src: ["src/js/libs.js"],
        dest: "dist/js/libs.js"
      }
    },

    uglify: {
      main: {
        src: ["dist/js/index.js"],
        dest: "dist/js/index.js"
      },
      libs: {
        src: ["dist/js/libs.js"],
        dest: "dist/js/libs.js"
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
  grunt.registerTask("serve", ["compile", "browserSync", "watch"]);
  grunt.registerTask("compile", ["browserify", "sass", "autoprefixer"]);
  grunt.registerTask("production", ["compile", "uglify", "css_import", "autoprefixer", "cssmin"]);
};