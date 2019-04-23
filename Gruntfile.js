module.exports = function(grunt) {
  grunt.loadNpmTasks("grunt-browserify");
  grunt.loadNpmTasks("grunt-contrib-uglify-es");
  grunt.loadNpmTasks("grunt-browser-sync");
  grunt.loadNpmTasks("grunt-contrib-watch");
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
        tasks: ["sass"]
      }
    },

    sass: {
      options: {
        implementation: sass,
        sourceMap: true
      },
      dist: {
        files: {
          'dist/css/index.css': 'sass/index.scss'
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
  grunt.registerTask("build", ["browserify:main", "uglify:main", "sass"]);
  grunt.registerTask("serve", ["browserify:main", "sass", "browserSync", "watch"]);
  grunt.registerTask("compile", ["browserify:main"]);
  grunt.registerTask("production", ["browserify:prod", "uglify:prod", "sass"]);
};